"""Signed, zero-mean quarter-hour redistribution conditioned on hourly context."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SignedRedistributionHead(nn.Module):
    """Predict two quarter distributions and a positive scale per hour.

    The correction is ``amplitude * (softmax(up) - softmax(down))``; its
    quarter mean is therefore exactly zero, while its sign is unrestricted.
    Input and output shapes are ``[..., context_dim]`` and ``[..., 4]``.
    """
    def __init__(self, context_dim: int, hidden_dim: int = 64, initial_amplitude: float = 0.25):
        super().__init__()
        if context_dim <= 0 or hidden_dim <= 0 or initial_amplitude <= 0:
            raise ValueError("dimensions and initial_amplitude must be positive")
        self.net = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 9))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)
        self.initial_amplitude = float(initial_amplitude)

    def diagnostics(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(context)
        up, down, amp_raw = raw[..., :4], raw[..., 4:8], raw[..., 8]
        # Inverse softplus initializes the amplitude to a small, active value.
        amplitude = F.softplus(amp_raw + torch.log(torch.expm1(amp_raw.new_tensor(self.initial_amplitude))))
        p, q = up.softmax(-1), down.softmax(-1)
        return {"up": p, "down": q, "amplitude": amplitude,
                "overlap": torch.minimum(p, q).sum(-1)}

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        d = self.diagnostics(context)
        correction = d["amplitude"].unsqueeze(-1) * (d["up"] - d["down"])
        return correction - correction.mean(dim=-1, keepdim=True)
