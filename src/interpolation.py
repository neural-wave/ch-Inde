"""Deterministic hourly-to-quarter-hour MAKIMA residual baseline."""
from __future__ import annotations

import math

import torch


def makima_residual(hourly: torch.Tensor, amplitude: float = 2.0) -> torch.Tensor:
    """Return a zero-mean-per-hour MAKIMA curve in physical price units.

    ``hourly`` contains the complete known hourly grid with shape ``[B, 24]``.
    Hourly observations and quarter-hour evaluations are located at their
    respective interval centers. SciPy performs the interpolation on CPU;
    the result is returned on the input device, with a floating dtype of at
    least float32. This operation is deterministic and intentionally has no
    dependency on quarter-hour targets.
    """
    if not torch.is_tensor(hourly) or hourly.ndim != 2 or hourly.shape[1] != 24:
        raise ValueError("hourly must be a tensor with shape [B,24]")
    if not hourly.is_floating_point():
        raise ValueError("hourly must have a floating-point dtype")
    if not torch.isfinite(hourly).all():
        raise ValueError("hourly must contain only finite values")
    try:
        amplitude = float(amplitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("amplitude must be a finite number") from exc
    if not math.isfinite(amplitude):
        raise ValueError("amplitude must be a finite number")

    # Import lazily so unrelated neural components remain importable in
    # environments that have not installed the optional SciPy dependency.
    try:
        import numpy as np
        from scipy.interpolate import Akima1DInterpolator
    except ImportError as exc:
        raise ImportError("MAKIMA interpolation requires scipy") from exc

    x = np.arange(24, dtype=np.float64) + 0.5
    qx = np.arange(96, dtype=np.float64) / 4.0 + 0.125
    values = hourly.detach().to(device="cpu", dtype=torch.float64).numpy()
    interpolator = Akima1DInterpolator(x, values, axis=1, method="makima", extrapolate=True)
    sampled = np.asarray(interpolator(qx), dtype=np.float64).reshape(-1, 24, 4)
    residual = amplitude * (sampled - sampled.mean(axis=-1, keepdims=True))
    dtype = hourly.dtype if hourly.dtype in (torch.float32, torch.float64) else torch.float32
    return torch.as_tensor(residual.reshape(-1, 96), device=hourly.device, dtype=dtype)
