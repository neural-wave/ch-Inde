"""Run delivery inference and write validated quarter-hour predictions.

The input must be a complete hourly grid with columns delivery_date, market_id,
hour_index, and hourly_price_eur_mwh.  The output has 96 rows per delivery
date and zone and is verified after CSV serialization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.core import validate_predictions_frame, write_predictions
from src.model import ControlledReconstructionModel
from src.datamodule import DailyPriceDataset, _collate, validate_hourly_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-input", type=Path, required=True, help="Complete supplied hourly price grid CSV.")
    parser.add_argument(
        "--model", choices=("ensemble", "b2"), default="ensemble",
        help="Frozen 50/50 ensemble (default) or B2 component only.",
    )
    parser.add_argument("--bundle", type=Path, default=Path("checkpoints"), help="Ensemble checkpoint bundle.")
    parser.add_argument("--history-input", type=Path, help="Optional strictly earlier history for ensemble retrieval.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/b2-base.ckpt"), help="B2 checkpoint for --model b2.")
    parser.add_argument("--output", type=Path, default=Path("data/predictions/predictions.csv"), help="Output prediction CSV.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    if args.model == "ensemble":
        from src.context.full_data_inference import predict

        output = predict(args.bundle, args.hourly_input, args.history_input, args.output, device, args.batch_size)
        print(f"Wrote validated ensemble predictions: {output}")
        return

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"B2-base checkpoint not found: {args.checkpoint}. Place it in checkpoints/ (or pass --checkpoint)."
        )
    hourly_frame = validate_hourly_frame(pd.read_csv(args.hourly_input))
    dataset = DailyPriceDataset(hourly_frame, labeled=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
    model = ControlledReconstructionModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()
    predictions, hourly, dates, zones = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            pred = model(batch["hourly"].to(device), batch["zone"].to(device), batch["weekday"].to(device))
            predictions.append(pred.cpu())
            hourly.append(batch["hourly_original"])
            dates.extend(batch["delivery_date"])
            zones.extend(batch["zone"].tolist())
    output = write_predictions(torch.cat(predictions), torch.cat(hourly), dates, zones, args.output)
    # Defend the final on-disk representation against precision/format drift.
    validate_predictions_frame(pd.read_csv(output), hourly_frame)
    print(f"Wrote validated predictions: {output}")


if __name__ == "__main__":
    main()
