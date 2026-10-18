"""Generate monthly/chronological splits and fold-aware gradient features."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.splits import write_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/train_prices_15min.csv'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/splits/monthly_protocol_enriched'))
    parser.add_argument('--monthly-scope', choices=['reserve-october', 'all-12'], default='reserve-october',
                        help='Default: 11 selection folds with October excluded. all-12 is a separate exploratory protocol.')
    parser.add_argument('--min-train-months', type=int, default=3,
                        help='Initial expanding-window training months, starting November 2025 (default 3).')
    parser.add_argument('--export-data', action='store_true', help='Also export train/validation/test price CSVs for every fold.')
    parser.add_argument('--no-enrich', action='store_true', help='Generate split manifests only, without gradient features.')
    parser.add_argument('--feature-format', choices=['parquet', 'csv'], default='parquet',
                        help='Enriched partition output format; default parquet preserves categorical dtypes.')
    args = parser.parse_args()
    try:
        manifest = write_splits(args.data, args.output, monthly_scope=args.monthly_scope,
                                min_train_months=args.min_train_months, export_data=args.export_data,
                                enrich_features=not args.no_enrich, feature_format=args.feature_format)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Created {len(manifest['folds'])} folds under {args.output}")
    print(f"Source: {manifest['source_dates']} dates; SHA256 {manifest['data_sha256']}")
    print('No models were trained. October was used by earlier experiments; it is reserved prospectively here.')


if __name__ == '__main__':
    main()
