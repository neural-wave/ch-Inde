"""Input-only affine normalization for daily hourly curves."""
import math

import torch
from torch import nn


class DailyNormalizer(nn.Module):
    def __init__(self, mode: str = "global", scale_floor_fraction: float = 0.05):
        super().__init__()
        if mode not in {"global", "per_day"}:
            raise ValueError("mode must be 'global' or 'per_day'")
        if not math.isfinite(scale_floor_fraction) or scale_floor_fraction <= 0:
            raise ValueError("scale_floor_fraction must be positive and finite")
        self.mode = mode
        self.scale_floor_fraction = float(scale_floor_fraction)
        self.register_buffer("global_mean", torch.tensor(float("nan"), dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(float("nan"), dtype=torch.float32))
        self.register_buffer("stats_ready", torch.tensor(False))

    def set_global_stats(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.global_mean.dtype, device=self.global_mean.device)
        std = torch.as_tensor(std, dtype=self.global_std.dtype, device=self.global_std.device)
        if (
            mean.numel() != 1
            or std.numel() != 1
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or std <= 0
        ):
            raise ValueError("global mean/std must be finite scalars and std positive")
        self.global_mean.copy_(mean.reshape(()))
        self.global_std.copy_(std.reshape(()))
        self.stats_ready.fill_(True)

    def _ready(self):
        if not bool(self.stats_ready):
            raise RuntimeError("global statistics have not been set")

    @staticmethod
    def _check_hourly(hourly):
        if hourly.ndim != 2 or hourly.shape[1] != 24 or not torch.isfinite(hourly).all():
            raise ValueError("hourly must be finite [B,24]")

    def statistics(self, hourly):
        self._ready()
        self._check_hourly(hourly)
        mu = torch.full(
            (hourly.shape[0], 1),
            self.global_mean.item(),
            dtype=hourly.dtype,
            device=hourly.device,
        )
        if self.mode == "global":
            scale = torch.full_like(mu, self.global_std.item())
        else:
            floor = self.scale_floor_fraction * self.global_std.to(hourly)
            scale = torch.maximum(hourly.std(dim=1, correction=0, keepdim=True), floor)
            mu = hourly.mean(dim=1, keepdim=True)
        return mu, scale

    def window_features(self, hourly):
        self._ready()
        self._check_hourly(hourly)
        # These features always describe the known day itself, in either mode.
        mu = hourly.mean(dim=1, keepdim=True)
        floor = self.scale_floor_fraction * self.global_std.to(hourly)
        daily_scale = torch.maximum(hourly.std(dim=1, correction=0, keepdim=True), floor)
        centered_mean = (mu - self.global_mean.to(hourly)) / self.global_std.to(hourly)
        log_scale = torch.log(daily_scale / self.global_std.to(hourly))
        return torch.cat((centered_mean, log_scale), dim=1)

    def normalize(self, values, mu, scale):
        return (values - mu) / scale

    def denormalize(self, values, mu, scale):
        return values * scale + mu
