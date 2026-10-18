"""Shared Fourier timing and three resolution-preserving history encoders."""
import math

import torch
from torch import nn


def fourier_time(length, device, dtype):
    if length not in (24, 96):
        raise ValueError('Expected a 24-hour or 96-quarter grid')
    time = (torch.arange(length, device=device, dtype=dtype) + .5) * (24 / length)
    angles = time[:, None] * time.new_tensor([2 * math.pi / 24, 4 * math.pi / 24])
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


class ConvResidual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Conv1d(width, width, 5, padding=2))
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x):
        return x + self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)


class HistoryEncoder(nn.Module):
    def __init__(self, policy, width=64, heads=4, ff=128, dropout=.1):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(4, width), nn.LayerNorm(width))
        if policy == 'linear':
            self.encoder = nn.Identity()
        elif policy == 'cnn':
            self.encoder = nn.Sequential(ConvResidual(width, dropout), ConvResidual(width, dropout))
        elif policy == 'transformer':
            self.encoder = nn.TransformerEncoderLayer(width, heads, ff, dropout, activation='gelu',
                                                      batch_first=True, norm_first=True)
        else:
            raise ValueError('Unknown historical encoder')

    def forward(self, features, time_and_type, available):
        x = self.value(features) + time_and_type
        x = x * available[:, None, None]
        # No time embedding is added again after preprocessing. Missing blocks are
        # suppressed both before and after biased convolutions/attention.
        return self.encoder(x) * available[:, None, None]


class HourShapeEncoder(nn.Module):
    """Keep ordered quarter corrections as a four-vector for each hour."""
    def __init__(self, policy, width=64, dropout=.1):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(4, width), nn.LayerNorm(width))
        if policy not in ('linear', 'cnn'):
            raise ValueError('Hour-shape encoding supports linear or CNN')
        self.encoder = (nn.Identity() if policy == 'linear' else
                        nn.Sequential(ConvResidual(width, dropout), ConvResidual(width, dropout)))

    def forward(self, features, time_and_type, available):
        # Only the time embedding is pooled; corrections retain all four slots.
        shape = features[..., 0].reshape(-1, 24, 4)
        time = time_and_type.reshape(-1, 24, 4, time_and_type.shape[-1]).mean(2)
        x = (self.value(shape) + time) * available[:, None, None]
        return self.encoder(x) * available[:, None, None]


class FusionBlock(nn.Module):
    """adaLN-Zero backbone with no RoPE and an explicit historical key mask."""
    def __init__(self, width, heads, ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, condition, mask=None):
        s1, a1, g1, s2, a2, g2 = self.modulation(condition).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + a1[:, None]) + s1[:, None]
        y = self.attn(y, y, y, key_padding_mask=mask, need_weights=False)[0]
        x = x + g1[:, None] * self.dropout(y)
        return x + g2[:, None] * self.dropout(self.mlp(self.norm2(x) * (1 + a2[:, None]) + s2[:, None]))
