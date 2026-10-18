"""Frozen 50/50 inference from the full-data A0-R3-CNN + B2 Original bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from src.constraints import validate_predictions_frame, write_predictions
from src.gating import gate_to_flat

from .retrieval_datamodule import RetrievalDataModule
from .rope_model import RopeContextTransformer


def predict(
    bundle: str | Path,
    hourly_input: str | Path,
    history_input: str | Path | None,
    output: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
) -> Path:
    """Write validated predictions from the frozen A0-R3-CNN + B2 recipe."""
    bundle = Path(bundle).resolve()
    recipe = json.loads((bundle / "ensemble.json").read_text())
    expected_labels = ["A0-base/R3-CNN", "B2-base/ORIGINAL"]
    if recipe["weights"] != [0.5, 0.5] or [item["label"] for item in recipe["components"]] != expected_labels:
        raise ValueError("Expected frozen A0-R3-CNN + B2 Original 50/50 recipe")

    models, gates = [], []
    for component in recipe["components"]:
        checkpoint = (bundle / component["checkpoint"]).resolve()
        if bundle not in checkpoint.parents:
            raise ValueError("Checkpoint must be inside the supplied bundle")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != component["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash differs from the bundle manifest")
        model = RopeContextTransformer.load_from_checkpoint(
            checkpoint, map_location=device, weights_only=False
        ).eval().to(device)
        models.append(model)
        gates.append(
            json.loads((bundle / component["gate"]).read_text())["params"]
            if "gate" in component
            else None
        )

    datamodule = RetrievalDataModule(
        train_path="unused-training-data.csv",
        predict_path=str(hourly_input),
        history_path=str(history_input) if history_input is not None else None,
        retrieval_policy="R3",
        batch_size=batch_size,
    )
    datamodule.setup("predict")
    predictions, hourly, dates, zones = [], [], [], []
    with torch.inference_mode():
        for original in datamodule.predict_dataloader():
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in original.items()}
            components = []
            for model, gate in zip(models, gates, strict=True):
                value = model.batch_prediction(batch)
                if gate is not None:
                    value = gate_to_flat(value, batch["hourly"], **gate)
                components.append(value)
            predictions.append((0.5 * components[0] + 0.5 * components[1]).cpu())
            hourly.append(original["hourly_original"])
            dates.extend(original["delivery_date"])
            zones.extend(original["zone"].tolist())

    path = write_predictions(torch.cat(predictions), torch.cat(hourly), dates, zones, output)
    validate_predictions_frame(pd.read_csv(path), pd.read_csv(hourly_input))
    return path
