"""Fold-aware integration with the shared gradient feature extractor."""
from __future__ import annotations
import pandas as pd
from src.gradient import features as F
from .splits import Fold


def enrich_fold(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, dict]:
    """Use training-only history, and attach targets only after feature extraction.

    The extractor itself restricts every history lookup to dates strictly before
    the evaluated date. Held-out, reserved and unused dates never enter history.
    """
    normalized = frame.copy()
    normalized[F.DATE] = pd.to_datetime(normalized[F.DATE]).dt.normalize()
    train_dates = pd.to_datetime(list(fold.train))
    active_dates = pd.to_datetime(list(fold.train + fold.validation + fold.test))
    history = normalized[normalized[F.DATE].isin(train_dates)].copy()
    active = normalized[normalized[F.DATE].isin(active_dates)].copy()
    enriched = F.build_features(F.derive_hourly(active), history, target_df=active)
    labels = {pd.Timestamp(d): partition for partition in ('train', 'validation', 'test')
              for d in getattr(fold, partition)}
    enriched['split'] = pd.Categorical(enriched[F.DATE].map(labels), categories=['train','validation','test'])
    if enriched['split'].isna().any() or len(enriched) != len(active):
        raise ValueError('enriched rows do not match fold assignments')
    metadata = F.feature_metadata(enriched, extra={
        'splits': {name: list(getattr(fold, name)) for name in ('train','validation','test','reserved','unused')},
        'strategy': fold.strategy, 'fold_id': fold.fold_id,
        'history_policy': 'training dates only; each feature lookup uses dates strictly before its target date',
        'history_dates': list(fold.train),
        'reserved_and_unused_rows_exported': False,
        'missing_history_policy': 'preserve NaN; no imputation fitted across partitions',
    })
    return enriched, metadata
