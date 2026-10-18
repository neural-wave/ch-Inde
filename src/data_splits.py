"""Whole-date monthly and expanding-window split definitions (no model fitting)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ZONES = tuple(f'ZONE_{i}' for i in range(4))
MONTHS = tuple(str(m) for m in pd.period_range('2025-10', '2026-09', freq='M'))
KEYS = ['delivery_date', 'market_id', 'hour_index', 'quarter_index']


@dataclass(frozen=True)
class Fold:
    strategy: str
    fold_id: str
    train: tuple[str, ...]
    validation: tuple[str, ...] = ()
    test: tuple[str, ...] = ()
    reserved: tuple[str, ...] = ()
    unused: tuple[str, ...] = ()

    def validate(self, source_dates: tuple[str, ...]) -> None:
        parts = [self.train, self.validation, self.test, self.reserved, self.unused]
        all_dates = [d for part in parts for d in part]
        if len(all_dates) != len(set(all_dates)) or set(all_dates) != set(source_dates):
            raise ValueError('fold assignments must partition source dates without overlap')
        if not self.train or not (self.validation or self.test):
            raise ValueError('each fold needs nonempty training and evaluation dates')
        if self.strategy == 'chronological' and max(self.train) >= min(self.validation):
            raise ValueError('chronological training dates must precede validation dates')


def read_prices(path: str | Path) -> pd.DataFrame:
    """Validate complete day-zone grids; never infer or fill missing dates."""
    f = pd.read_csv(path)
    required = KEYS + ['price_eur_mwh']
    if not set(required) <= set(f.columns) or f.empty:
        raise ValueError('source must contain nonempty official quarter-price columns')
    if not f.delivery_date.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
        raise ValueError('delivery dates must use YYYY-MM-DD')
    parsed = pd.to_datetime(f.delivery_date, format='%Y-%m-%d', errors='coerce')
    if parsed.isna().any() or not parsed.dt.strftime('%Y-%m-%d').eq(f.delivery_date).all():
        raise ValueError('invalid delivery date')
    for col, upper in [('hour_index', 24), ('quarter_index', 4)]:
        f[col] = pd.to_numeric(f[col], errors='coerce')
        if f[col].isna().any() or not (f[col].between(1, upper) & f[col].mod(1).eq(0)).all():
            raise ValueError(f'invalid {col}')
        f[col] = f[col].astype(int)
    prices = pd.to_numeric(f.price_eur_mwh, errors='coerce')
    if not np.isfinite(prices).all():
        raise ValueError('prices must be finite; negative prices remain valid')
    f.price_eur_mwh = prices
    if not f.market_id.isin(ZONES).all() or f.duplicated(KEYS).any():
        raise ValueError('unknown zone or duplicate quarter key')
    if not f.groupby(KEYS[:2]).size().eq(96).all():
        raise ValueError('each date-zone must contain all 96 quarters')
    if not f.groupby('delivery_date').market_id.nunique().eq(4).all():
        raise ValueError('every date must contain all four synchronized zones')
    if f.delivery_date.isin(['2025-10-26', '2026-03-29']).any():
        raise ValueError('excluded DST transition dates must remain absent')
    months = tuple(sorted(f.delivery_date.str[:7].unique()))
    if months != MONTHS:
        raise ValueError('source must cover all months October 2025–September 2026')
    return f.sort_values(KEYS).reset_index(drop=True)


def build_folds(dates, monthly_scope: str = 'reserve-october', min_train_months: int = 3) -> list[Fold]:
    dates = tuple(sorted(set(dates)))
    if tuple(sorted({d[:7] for d in dates})) != MONTHS:
        raise ValueError('all twelve source months are required')
    if monthly_scope not in ('reserve-october', 'all-12'):
        raise ValueError('monthly_scope must be reserve-october or all-12')
    if type(min_train_months) is not int or not 1 <= min_train_months <= 10:
        raise ValueError('min_train_months must be between 1 and 10')
    october = tuple(d for d in dates if d.startswith('2025-10'))
    development = tuple(d for d in dates if not d.startswith('2025-10'))
    eligible = development if monthly_scope == 'reserve-october' else dates
    folds = []
    for month in sorted({d[:7] for d in eligible}):
        folds.append(Fold('monthly_cv', 'holdout-' + month,
                          tuple(d for d in eligible if d[:7] != month),
                          tuple(d for d in eligible if d[:7] == month),
                          reserved=october if monthly_scope == 'reserve-october' else ()))
    folds.append(Fold('october_holdout', 'october-2025', development, test=october))
    # Chronological robustness always excludes October, including in all-12 CV mode.
    months = MONTHS[1:]
    for position in range(min_train_months, len(months)):
        month = months[position]
        folds.append(Fold('chronological', 'through-' + months[position-1] + '-validate-' + month,
                          tuple(d for d in development if d[:7] < month),
                          tuple(d for d in development if d[:7] == month),
                          reserved=october, unused=tuple(d for d in development if d[:7] > month)))
    for fold in folds:
        fold.validate(dates)
    return folds


EVALUATION = {
    'metric': 'MAE over 96 quarters per date-zone, mean over dates per zone, then equal mean of four zones',
    'hourly_mean_tolerance_eur_mwh': 1e-4,
    'standalone_baseline': {'method': 'makima', 'amplitude': 2.0, 'hour_centers': True,
                            'quarter_centers': True, 'extrapolate': True, 'center_residual_per_hour': True},
    'report': ['model_mae', 'makima_mae', 'absolute_improvement=makima_mae-model_mae',
               'relative_improvement_pct=100*(makima_mae-model_mae)/makima_mae'],
    'fold_aggregation': ['per_fold_scores', 'equal_month_macro_average', 'pooled_date_weighted_equal_zone_MAE'],
    'history': 'historical features must use strictly earlier dates with explicitly allowed label availability',
    'zero_baseline_policy': 'relative improvement is null when MAKIMA MAE equals zero',
    'selection': 'preprocessing fits train only; gate calibration stays inside training dates; test never tunes',
    'windows': 'whole-date validation/test; any sliding training window must stay within contiguous training dates',
}


def write_splits(data_path: str | Path, output: str | Path, *, monthly_scope='reserve-october',
                 min_train_months=3, export_data=False, enrich_features=False, feature_format="parquet") -> dict:
    data_path, output = Path(data_path), Path(output)
    if output.exists():
        raise FileExistsError(f'output already exists; choose a fresh directory: {output}')
    if feature_format not in ("csv", "parquet"):
        raise ValueError("feature_format must be csv or parquet")
    frame = read_prices(data_path)
    dates = tuple(sorted(frame.delivery_date.unique()))
    folds = build_folds(dates, monthly_scope, min_train_months)
    missing = sorted(set(pd.date_range('2025-10-01', '2026-09-30').strftime('%Y-%m-%d')) - set(dates))
    manifest = {'schema_version': 1, 'data_path': str(data_path.resolve()),
                'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
                'monthly_scope': monthly_scope, 'chronological_min_train_months': min_train_months,
                'zones': ZONES, 'source_dates': len(dates), 'source_rows': len(frame),
                'missing_dates_preserved': missing, 'evaluation': EVALUATION,
                'october_status': 'reserved by this protocol; earlier project experiments already used October in training',
                'all_12_caveat': 'all-12 monthly CV uses October in fitting/selection and is incompatible with an untouched October final holdout' if monthly_scope == 'all-12' else None,
                'feature_enrichment': {'enabled': enrich_features, 'extractor': 'src.gradient.features.build_features' if enrich_features else None, 'format': feature_format if enrich_features else None, 'history_policy': 'train-only, strictly earlier dates' if enrich_features else None},
                'folds': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.split-build-', dir=output.parent) as temporary:
        staging = Path(temporary)
        summary = []
        for fold in folds:
            leaf = staging / fold.strategy / fold.fold_id; leaf.mkdir(parents=True)
            payload = asdict(fold)
            counts = {}
            assignments = []
            for partition in ('train', 'validation', 'test', 'reserved', 'unused'):
                assigned = getattr(fold, partition)
                counts[partition] = {'dates': len(assigned), 'day_zone_samples': len(assigned)*4, 'quarter_rows': len(assigned)*384}
                pd.DataFrame({'delivery_date': list(assigned)}).to_csv(leaf/(partition+'_dates.csv'), index=False)
                assignments += [(d, z, partition) for d in assigned for z in ZONES]
                if export_data and partition in ('train', 'validation', 'test'):
                    frame[frame.delivery_date.isin(assigned)].to_csv(leaf/(partition+'.csv'), index=False)
            if enrich_features:
                from .enrichment import enrich_fold
                enriched, metadata = enrich_fold(frame, fold)
                feature_files = {}
                for partition in ('train', 'validation', 'test'):
                    part = enriched[enriched['split'] == partition].copy()
                    filename = f'{partition}_enriched.{feature_format}'
                    if feature_format == 'parquet':
                        part.to_parquet(leaf/filename, index=False)
                    else:
                        part.to_csv(leaf/filename, index=False)
                    feature_files[partition] = filename
                metadata['files'] = feature_files
                (leaf/'feature_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
                payload['features'] = {'files': feature_files, 'metadata': 'feature_metadata.json',
                                       'history_policy': metadata['history_policy']}
            payload['counts'] = counts; payload['evaluation'] = EVALUATION
            (leaf/'split.json').write_text(json.dumps(payload, indent=2)+'\n')
            pd.DataFrame(assignments, columns=['delivery_date','market_id','partition']).sort_values(['delivery_date','market_id']).to_csv(leaf/'assignments.csv', index=False)
            row = {'strategy': fold.strategy, 'fold_id': fold.fold_id,
                   **{k+'_dates': v['dates'] for k,v in counts.items()}}
            summary.append(row)
            manifest['folds'].append(row | {'path': str(Path(fold.strategy)/fold.fold_id/'split.json')})
        pd.DataFrame(summary).to_csv(staging/'summary.csv', index=False)
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        # Publish only after every fold validates and every file has been written.
        staging.rename(output)
    return manifest
