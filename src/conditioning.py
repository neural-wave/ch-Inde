"""adaLN-Zero conditioning components for the daily transformer."""
from __future__ import annotations

import torch
from torch import nn


class DailyConditioner(nn.Module):
    """Combine optional categorical metadata and input-only window statistics."""

    def __init__(self, d_model: int, zone: bool = True, weekday: bool = False, window_stats: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_zone, self.use_weekday, self.use_window_stats = zone, weekday, window_stats
        self.zone_embedding = nn.Embedding(4, d_model) if zone else None
        self.weekday_embedding = nn.Embedding(7, d_model) if weekday else None
        self.stats_projection = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model)) if window_stats else None

    def forward(self, batch_size: int, device: torch.device, zone: torch.Tensor | None = None,
                weekday: torch.Tensor | None = None, window_stats: torch.Tensor | None = None) -> torch.Tensor:
        result = torch.zeros(batch_size, self._dim, device=device)
        if self.zone_embedding is not None:
            if zone is None:
                raise ValueError("zone is required when zone conditioning is enabled")
            if (zone.shape != (batch_size,) or zone.dtype == torch.bool or zone.is_floating_point()
                    or torch.any((zone < 0) | (zone > 3))):
                raise ValueError("zone must contain one integer ID in [0, 3] per sample")
            result = result + self.zone_embedding(zone.long())
        if self.weekday_embedding is not None:
            if weekday is None:
                raise ValueError("weekday is required when weekday conditioning is enabled")
            if (weekday.shape != (batch_size,) or weekday.dtype == torch.bool or weekday.is_floating_point()
                    or torch.any((weekday < 0) | (weekday > 6))):
                raise ValueError("weekday must contain one integer ID in [0, 6] per sample")
            result = result + self.weekday_embedding(weekday.long())
        if self.stats_projection is not None:
            if window_stats is None or window_stats.shape != (batch_size, 2):
                raise ValueError("window_stats [B,2] are required when statistics conditioning is enabled")
            result = result + self.stats_projection(window_stats.to(result.dtype))
        return result

    @property
    def _dim(self) -> int:
        for embedding in (self.zone_embedding, self.weekday_embedding):
            if embedding is not None:
                return embedding.embedding_dim
        if self.stats_projection is not None:
            return self.stats_projection[-1].out_features
        return getattr(self, "d_model", 0)


class AdaLNZeroBlock(nn.Module):
    """Transformer block with zero-initialized shift/scale/gate modulation."""
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float, rope_base: float) -> None:
        super().__init__()
        from .attention import RotarySelfAttention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = RotarySelfAttention(d_model, n_heads, dropout, rope_base)
        self.mlp = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        s1, a1, g1, s2, a2, g2 = self.modulation(condition).chunk(6, dim=-1)
        x = x + g1[:, None] * self.dropout(self.attn(self.norm1(x) * (1 + a1[:, None]) + s1[:, None]))
        x = x + g2[:, None] * self.dropout(self.mlp(self.norm2(x) * (1 + a2[:, None]) + s2[:, None]))
        return x
