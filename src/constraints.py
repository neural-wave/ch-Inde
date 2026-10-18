"""Hourly mean projection and validated prediction export."""
from pathlib import Path

import numpy as np
import pandas as pd


def project_hourly(residuals, hourly, scale):
    import torch

    if residuals.ndim != 2 or residuals.shape[1] != 96:
        raise ValueError("residuals must be [B,96]")
    if hourly.shape != (residuals.shape[0], 24):
        raise ValueError("hourly must be [B,24]")
    if scale.shape not in {(residuals.shape[0], 1), (residuals.shape[0],)}:
        raise ValueError("scale must be [B,1] or [B]")
    if not torch.isfinite(residuals).all() or not torch.isfinite(hourly).all() or not torch.isfinite(scale).all():
        raise ValueError("projection inputs must be finite")
    if (scale <= 0).any():
        raise ValueError("scale must be positive")

    dtype = torch.promote_types(residuals.dtype, hourly.dtype)
    dtype = torch.promote_types(dtype, scale.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    residual_values = residuals.to(dtype=dtype).reshape(-1, 24, 4)
    hourly_values = hourly.to(dtype=dtype)
    scale_values = scale.to(dtype=dtype).reshape(-1, 1, 1)
    centered = residual_values - residual_values.mean(dim=2, keepdim=True)
    return (hourly_values.unsqueeze(-1) + centered * scale_values).reshape(-1, 96)


def _as_numpy(values, dtype=np.float64):
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values, dtype=dtype)


def _zone_name(zone):
    if isinstance(zone, (int, np.integer)) or (
        isinstance(zone, float) and zone.is_integer()
    ):
        return f"ZONE_{int(zone)}"
    return str(zone)


def _pred_frame(predictions, hourly, dates, zones):
    predictions = _as_numpy(predictions)
    hourly = _as_numpy(hourly)
    if (
        predictions.ndim != 2
        or predictions.shape[1] != 96
        or hourly.shape != (len(predictions), 24)
        or len(dates) != len(predictions)
        or len(zones) != len(predictions)
    ):
        raise ValueError("predictions/hourly/metadata shapes must be [N,96]/[N,24]/N")
    if not np.isfinite(predictions).all() or not np.isfinite(hourly).all():
        raise ValueError("predictions and hourly inputs must be finite")

    # Correct in float64 against the exact supplied hourly grid before writing.
    quarters = predictions.reshape(-1, 24, 4)
    quarters = quarters + (hourly[:, :, None] - quarters.mean(axis=2, keepdims=True))
    predictions = quarters.reshape(-1, 96)

    rows = []
    for i, (day, zone) in enumerate(zip(dates, zones)):
        for index, value in enumerate(predictions[i]):
            rows.append(
                (str(day), _zone_name(zone), index // 4 + 1, index % 4 + 1, float(value))
            )
    return pd.DataFrame(
        rows,
        columns=[
            "delivery_date",
            "market_id",
            "hour_index",
            "quarter_index",
            "predicted_price_eur_mwh",
        ],
    )


def validate_predictions_frame(frame, hourly_frame):
    required = {
        "delivery_date",
        "market_id",
        "hour_index",
        "quarter_index",
        "predicted_price_eur_mwh",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("prediction file is empty")

    from .datamodule import ZONES, _dates, validate_hourly_frame

    hourly_grid = validate_hourly_frame(hourly_frame)
    predictions = frame[list(required)].copy()
    for column in ("hour_index", "quarter_index", "predicted_price_eur_mwh"):
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    if predictions[list(required)].isna().any().any():
        raise ValueError("predictions contain invalid values")
    if not np.isfinite(predictions.predicted_price_eur_mwh).all():
        raise ValueError("predictions contain non-finite values")

    predictions["delivery_date"] = _dates(predictions.delivery_date)
    predictions["market_id"] = predictions.market_id.map(_zone_name)
    if not (predictions.hour_index % 1 == 0).all() or not (
        predictions.quarter_index % 1 == 0
    ).all():
        raise ValueError("prediction hour and quarter indices must be integers")
    keys = ["delivery_date", "market_id", "hour_index", "quarter_index"]
    if predictions.duplicated(keys).any():
        raise ValueError("prediction keys are not unique")
    if (
        not predictions.hour_index.between(1, 24).all()
        or not predictions.quarter_index.between(1, 4).all()
        or not predictions.market_id.isin(ZONES).all()
    ):
        raise ValueError("invalid prediction grid key")

    expected_hours = set(
        map(tuple, hourly_grid[["delivery_date", "market_id", "hour_index"]].to_numpy())
    )
    found_hours = set(
        map(
            tuple,
            predictions[["delivery_date", "market_id", "hour_index"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    if found_hours != expected_hours:
        raise ValueError("prediction hourly grid does not exactly match supplied hourly grid")
    counts = predictions.groupby(["delivery_date", "market_id"]).size()
    if (counts != 96).any():
        raise ValueError("every date-zone prediction must contain 96 quarters")
    quarters = predictions.groupby(
        ["delivery_date", "market_id", "hour_index"]
    ).quarter_index.agg(list)
    if any(sorted(values) != [1, 2, 3, 4] for values in quarters):
        raise ValueError("every hour needs quarters 1..4")

    hourly_means = (
        predictions.groupby(["delivery_date", "market_id", "hour_index"])
        .predicted_price_eur_mwh.mean()
        .rename("predicted_mean")
        .reset_index()
    )
    checked = hourly_means.merge(
        hourly_grid, on=["delivery_date", "market_id", "hour_index"], validate="one_to_one"
    )
    errors = np.abs(
        checked.predicted_mean.to_numpy() - checked.hourly_price_eur_mwh.to_numpy()
    )
    if (errors > 1e-4).any():
        raise ValueError(f"hourly consistency violation: max error {errors.max():.8g}")
    return predictions.sort_values(keys).reset_index(drop=True)


def write_predictions(predictions, hourly, dates, zones, path):
    frame = _pred_frame(predictions, hourly, dates, zones)
    hourly = _as_numpy(hourly)
    from .datamodule import validate_hourly_frame

    hourly_grid = pd.DataFrame(
        [
            (str(day), _zone_name(zone), hour + 1, float(value))
            for day, zone, curve in zip(dates, zones, hourly)
            for hour, value in enumerate(curve)
        ],
        columns=[
            "delivery_date",
            "market_id",
            "hour_index",
            "hourly_price_eur_mwh",
        ],
    )
    hourly_grid = validate_hourly_frame(hourly_grid)
    validate_predictions_frame(frame, hourly_grid)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format="%.10f")
    # Validate the actual serialized representation.
    validate_predictions_frame(pd.read_csv(output), hourly_grid)
    return output
