"""Input-driven shrinkage toward the supplied hourly curve."""

import numpy as np
import math


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def gate_to_flat(predictions, hourly, price_threshold, variation_threshold, strength, radius=1):
    """Shrink eligible hours toward their hourly mean while preserving that mean.

    ``predictions`` is [B,96], ``hourly`` is [B,24]. Local variation is the
    population standard deviation of the hourly curve over the clipped
    ``[hour-radius, hour+radius]`` neighborhood. Negative thresholds disable
    their corresponding eligibility test. The gate is computed per hour and
    applied equally to its four quarter values.
    """
    if not all(math.isfinite(float(v)) for v in (price_threshold, variation_threshold, strength)) or not 0 <= strength <= 1:
        raise ValueError("gate thresholds must be finite and strength must be in [0,1]")
    if type(radius) is not int or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    torch = None
    is_torch = type(predictions).__module__.startswith("torch") or type(hourly).__module__.startswith("torch")
    if is_torch:
        import torch as _torch
        torch = _torch
        device = predictions.device if torch.is_tensor(predictions) else hourly.device
        p = torch.as_tensor(predictions, device=device)
        h = torch.as_tensor(hourly, device=device)
        if p.ndim != 2 or tuple(p.shape[1:]) != (96,) or tuple(h.shape) != (p.shape[0], 24):
            raise ValueError("predictions/hourly must have shapes [B,96] and [B,24]")
        if not torch.isfinite(p).all() or not torch.isfinite(h).all():
            raise ValueError("predictions and hourly must be finite")
        if radius < 0 or int(radius) != radius:
            raise ValueError("radius must be a nonnegative integer")
        dtype = torch.promote_types(p.dtype, h.dtype)
        if dtype in (torch.float16, torch.bfloat16): dtype = torch.float32
        p, h = p.to(dtype), h.to(dtype)
        windows = [h[:, max(0, i-radius):min(24, i+radius+1)].std(dim=1, unbiased=False) for i in range(24)]
        variation = torch.stack(windows, dim=1)
        eligible = torch.zeros_like(h, dtype=torch.bool)
        if price_threshold >= 0: eligible |= h.abs() <= price_threshold
        if variation_threshold >= 0: eligible |= variation <= variation_threshold
        g = 1.0 - float(strength) * eligible.to(dtype)
        return (h[:, :, None] + g[:, :, None] * (p.reshape(-1, 24, 4) - h[:, :, None])).reshape(-1, 96)

    p, h = np.asarray(predictions), np.asarray(hourly)
    if p.ndim != 2 or p.shape[1] != 96 or h.shape != (p.shape[0], 24):
        raise ValueError("predictions/hourly must have shapes [B,96] and [B,24]")
    if not np.isfinite(p).all() or not np.isfinite(h).all():
        raise ValueError("predictions and hourly must be finite")
    if radius < 0 or int(radius) != radius:
        raise ValueError("radius must be a nonnegative integer")
    dtype = np.result_type(p.dtype, h.dtype, np.float32)
    p, h = p.astype(dtype, copy=False), h.astype(dtype, copy=False)
    variation = np.stack([h[:, max(0, i-radius):min(24, i+radius+1)].std(axis=1) for i in range(24)], axis=1)
    eligible = np.zeros(h.shape, dtype=bool)
    if price_threshold >= 0: eligible |= np.abs(h) <= price_threshold
    if variation_threshold >= 0: eligible |= variation <= variation_threshold
    g = 1.0 - float(strength) * eligible.astype(dtype)
    return (h[:, :, None] + g[:, :, None] * (p.reshape(-1, 24, 4) - h[:, :, None])).reshape(-1, 96)


def select_gate(predictions, targets, hourly):
    """Select a fixed gate from July rows using the balanced daily MAE objective.

    Inputs are complete [N,96] daily rows, expected to contain all four zones
    for every date. Since each date/zone contributes one row, their mean MAE
    is exactly the required equal-zone, equal-day objective.
    """
    p, y, h = map(lambda x: np.asarray(_numpy(x), dtype=np.float64), (predictions, targets, hourly))
    if p.ndim != 2 or p.shape[1] != 96 or y.shape != p.shape or h.shape != (len(p), 24):
        raise ValueError("predictions/targets/hourly must have shapes [N,96], [N,96], [N,24]")
    if not all(np.isfinite(x).all() for x in (p, y, h)):
        raise ValueError("tuner inputs must be finite")
    price_grid = [-1.0, 0.1, 1.0, 5.0]
    variation_grid = [-1.0, 0.1, 1.0, 5.0]
    strength_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    candidates = []
    for pt in price_grid:
        for vt in variation_grid:
            for strength in strength_grid:
                adjusted = gate_to_flat(p, h, pt, vt, strength, radius=1)
                # Equal contribution per delivery date and zone.
                daily_zone_mae = np.abs(adjusted-y).mean(axis=1)
                mae = float(daily_zone_mae.mean())
                candidates.append({"price_threshold": pt, "variation_threshold": vt,
                                   "strength": strength, "radius": 1, "mae": mae,
                                   "eligible_hour_count": int(_eligible_count(h, pt, vt)),
                                   "gate_change_count": int(_eligible_count(h, pt, vt) if strength else 0)})
    # Prefer identity for objective ties; then fewer affected hours and least strength.
    best = min(candidates, key=lambda c: (round(c["mae"], 12), c["strength"] != 0,
                                          c["gate_change_count"], c["strength"],
                                          c["price_threshold"], c["variation_threshold"]))
    ungated = float(np.abs(p-y).mean())
    return {"params": {"price_threshold": best["price_threshold"],
                        "variation_threshold": best["variation_threshold"],
                        "strength": best["strength"], "radius": 1},
            "validation_mae": best["mae"], "ungated_validation_mae": ungated,
            "candidates": candidates,
            "protocol": "July-only; fixed grid; radius=1; complete daily zone rows"}


def _eligible_count(hourly, price_threshold, variation_threshold):
    variation = np.stack([hourly[:, max(0, i-1):min(24, i+2)].std(axis=1) for i in range(24)], axis=1)
    eligible = np.zeros(hourly.shape, dtype=bool)
    if price_threshold >= 0: eligible |= np.abs(hourly) <= price_threshold
    if variation_threshold >= 0: eligible |= variation <= variation_threshold
    return int(eligible.sum())
