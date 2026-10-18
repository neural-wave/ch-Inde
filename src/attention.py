"""Bidirectional self-attention with rotary position encoding."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    """Apply float32 cached RoPE rotations to even-dimensional Q/K pairs."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2:
            raise ValueError("RoPE requires a positive even head_dim")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("RoPE base must be finite and positive")
        self.head_dim = head_dim
        self.base = float(base)
        inv_freq = self._make_inv_freq(torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = self._make_cache(torch.device("cpu"))
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _make_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))

    def _make_cache(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._make_inv_freq(device)
        angles = torch.outer(torch.arange(96, dtype=torch.float32, device=device), inv_freq)
        return angles.cos(), angles.sin()

    def _apply(self, fn):
        # Preserve float32 phase precision even when the containing model is cast
        # to half/bfloat16. Rebuild the fixed daily table on the target device.
        super()._apply(fn)
        device = self.inv_freq.device
        self.inv_freq = self._make_inv_freq(device)
        self.cos_cached, self.sin_cached = self._make_cache(device)
        return self

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: [batch, heads, sequence, head_dim]
        length = x.shape[-2]
        if positions is None:
            if length <= self.cos_cached.shape[0]:
                cos, sin = self.cos_cached[:length], self.sin_cached[:length]
            else:
                positions = torch.arange(length, device=x.device)
        if positions is not None:
            if positions.ndim != 1 or positions.numel() != length:
                raise ValueError("positions must be a 1D tensor matching the sequence length")
            angles = torch.outer(positions.to(device=x.device, dtype=torch.float32), self.inv_freq.to(device=x.device, dtype=torch.float32))
            cos, sin = angles.cos(), angles.sin()
        cos, sin = cos.to(dtype=x.dtype)[None, None], sin.to(dtype=x.dtype)[None, None]
        paired = x.reshape(*x.shape[:-1], -1, 2)
        even, odd = paired[..., 0], paired[..., 1]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        return rotated


class RotarySelfAttention(nn.Module):
    """Small explicit Q/K/V attention; full bidirectional grid, no semantic mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, rope_base: float = 10_000.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.rope(q), self.rope(k)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, n, d))
