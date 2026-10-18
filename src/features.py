"""Local, target-free features derived from the supplied hourly curve."""
from __future__ import annotations

import torch


# Keep this ordering stable: the complete 19-channel set is used by existing
# checkpoints.  Channels 0:9 describe the curve in day/global-normalized units;
# channels 9:13 are zone indicators, 13:17 are hour-of-day Fourier terms, and
# 17:19 are local-volatility summaries in day-normalized units.  Local windows
# at the start/end of the day use replicated boundary hours.
STRUCTURED_FEATURE_NAMES = (
    "hourly_day_normalized", "left_difference_day_normalized",
    "right_difference_day_normalized", "curvature_day_normalized",
    "left_difference_global_normalized", "right_difference_global_normalized",
    "hourly_global_normalized", "daily_mean_global_normalized",
    "log_daily_scale_global_ratio", "zone_0", "zone_1", "zone_2", "zone_3",
    "hour_sin_1", "hour_cos_1", "hour_sin_2", "hour_cos_2",
    "local_std_radius_1_day_normalized", "local_std_radius_2_day_normalized",
)
STRUCTURED_NUMERICAL_FEATURE_NAMES = tuple(
    STRUCTURED_FEATURE_NAMES[i] for i in (*range(9), 17, 18)
)


def structured_hourly_features(hourly: torch.Tensor, zones: torch.Tensor | None, global_scale: float) -> torch.Tensor:
    """Torch equivalent of the shared 19-channel L1 hourly feature contract."""
    if hourly.ndim != 2 or hourly.shape[1] != 24:
        raise ValueError("hourly must have shape [B,24]")
    b = hourly.shape[0]
    if zones is None:
        zones = torch.zeros(b, dtype=torch.long, device=hourly.device)
    scale = hourly.std(dim=1, correction=0).clamp_min(.05 * max(float(global_scale), 1e-12))
    mean = hourly.mean(dim=1)
    prev = torch.cat((hourly[:, :1], hourly[:, :-1]), dim=1)
    nxt = torch.cat((hourly[:, 1:], hourly[:, -1:]), dim=1)
    left, right = hourly-prev, nxt-hourly
    curvature = right-left
    padded = torch.nn.functional.pad(hourly[:, None], (2, 2), mode="replicate")[:, 0]
    vols = []
    for radius in (1, 2):
        window = torch.stack([padded[:, 2-radius+j:2-radius+j+24] for j in range(2*radius+1)], dim=-1)
        vols.append(window.std(dim=-1, correction=0))
    local_vol = torch.stack(vols, dim=-1)
    hour = torch.arange(24, device=hourly.device, dtype=hourly.dtype)
    periodic = torch.stack((torch.sin(2*torch.pi*hour/24), torch.cos(2*torch.pi*hour/24),
                            torch.sin(4*torch.pi*hour/24), torch.cos(4*torch.pi*hour/24)), dim=-1)
    zone_oh = torch.nn.functional.one_hot(zones.long(), 4).to(hourly.dtype)
    g = max(float(global_scale), 1e-12)
    return torch.cat((hourly.div(scale[:, None])[..., None], left.div(scale[:, None])[..., None],
                      right.div(scale[:, None])[..., None], curvature.div(scale[:, None])[..., None],
                      left.div(g)[..., None], right.div(g)[..., None], hourly.div(g)[..., None],
                      (mean/g)[:, None, None].expand(-1,24,1), (scale/g).log()[:,None,None].expand(-1,24,1),
                      zone_oh[:,None,:].expand(-1,24,-1), periodic[None].expand(b,-1,-1),
                      local_vol.div(scale[:,None,None])), dim=-1)


def structured_numerical_features(hourly: torch.Tensor, global_scale: float) -> torch.Tensor:
    """Return the 11 numerical channels of the structured hourly contract.

    The output is exactly columns ``0:9, 17:19`` of
    :func:`structured_hourly_features`, in the order named by
    ``STRUCTURED_NUMERICAL_FEATURE_NAMES``. It contains no zone or hour-of-day
    channels. Scale floors and replicated day-boundary windows match the full
    19-channel implementation for checkpoint and ablation consistency.
    """
    full = structured_hourly_features(hourly, zones=None, global_scale=global_scale)
    indices = torch.tensor((*range(9), 17, 18), device=hourly.device)
    return full.index_select(-1, indices)


def local_hourly_features(hourly: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Return physical-unit [d-, d+, curvature, clipped-neighborhood std].

    The neighborhood standard deviation is a population statistic over the
    available hours in the clipped window, including the center hour.
    """
    if (not torch.is_tensor(hourly) or hourly.dtype != torch.float32 or hourly.ndim != 2
            or hourly.shape[1] != 24 or not torch.isfinite(hourly).all()):
        raise ValueError("hourly must be finite float32 with shape [B,24]")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer")
    left = torch.cat((hourly[:, :1], hourly[:, :-1]), dim=1)
    right = torch.cat((hourly[:, 1:], hourly[:, -1:]), dim=1)
    dminus = hourly - left
    dplus = right - hourly
    curvature = right - 2 * hourly + left
    curvature = curvature.clone()
    curvature[:, 0] = 0
    curvature[:, -1] = 0
    stds = []
    for h in range(24):
        neighborhood = hourly[:, max(0, h-radius):min(24, h+radius+1)]
        stds.append(neighborhood.std(dim=1, correction=0))
    return torch.stack((dminus, dplus, curvature, torch.stack(stds, dim=1)), dim=-1)
