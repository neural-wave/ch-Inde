"""Deterministic time-of-day Fourier features for daily price tokens."""
from __future__ import annotations

import math

import torch


def absolute_time_features(length: int, harmonics: tuple[int, ...] | list[int], *,
                           device: torch.device, dtype: torch.dtype,
                           start_hour: torch.Tensor | None = None) -> torch.Tensor:
    """Return Fourier features at token centers, optionally offset per sample.

    Supports the physical 96-quarter and 24-hour grids described by the model.
    Without ``start_hour`` the shape is [length, 2*len(harmonics)]; with it,
    the shape is [B, length, 2*len(harmonics)].
    """
    if length not in (24, 96):
        raise ValueError("absolute time features support sequence lengths 24 or 96")
    if not harmonics or any(isinstance(h, bool) or int(h) != h or h <= 0 for h in harmonics):
        raise ValueError("harmonics must be a non-empty sequence of positive integers")
    if start_hour is not None:
        if (not torch.is_tensor(start_hour) or start_hour.ndim != 1 or start_hour.dtype == torch.bool
                or start_hour.is_floating_point() or torch.any((start_hour < 0) | (start_hour >= 24))):
            raise ValueError("start_hour must contain integer hours in [0, 23] per sample")
        start_hour = start_hour.to(device=device, dtype=torch.float32)
    steps = torch.arange(length, device=device, dtype=torch.float32)
    hour = steps + (0.5 if length == 24 else 0.5)  # both grids use center coordinates
    if length == 96:
        hour = steps / 4.0 + 0.125
    if start_hour is not None:
        hour = hour[None, :] + start_hour[:, None]
    cols = []
    for harmonic in harmonics:
        phase = (2.0 * math.pi * int(harmonic) / 24.0) * hour
        cols.extend((phase.sin(), phase.cos()))
    return torch.stack(cols, dim=-1).to(dtype=dtype)


def fourier_time_features(length: int, harmonics: tuple[int, ...] | list[int] = (1, 2)) -> torch.Tensor:
    """Convenience float32 features for the standard daily grids."""
    return absolute_time_features(length, harmonics, device=torch.device("cpu"), dtype=torch.float32)
