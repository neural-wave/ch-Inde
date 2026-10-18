"""Controlled encoder/head ablations built on the daily price transformer."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .redistribution import SignedRedistributionHead
from .transformer import DailyPriceTransformer


class ControlledDailyPriceTransformer(DailyPriceTransformer):
    """Daily transformer with explicit feature, encoder, and output controls.

    ``control`` is intentionally small and persisted in Lightning hyperparameters
    so checkpoints cannot silently be loaded under a different ablation.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ff_dim: int = 128, dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 normalization: dict[str, Any] | None = None,
                 conditioning: dict[str, Any] | None = None,
                 position: dict[str, Any] | None = None,
                 teacher_forcing: dict[str, Any] | None = None,
                 tokenization: dict[str, Any] | None = None,
                 prediction: dict[str, Any] | None = None,
                 interpolation: dict[str, Any] | None = None,
                 local_features: dict[str, Any] | None = None,
                 control: dict[str, Any] | None = None) -> None:
        control = {"feature_set": "local4", "encoder": "transformer", "head": "centered", **(control or {})}
        if set(control) != {"feature_set", "encoder", "head"}:
            raise ValueError("control requires exactly feature_set, encoder, and head")
        feature_set, encoder, head = control["feature_set"], control["encoder"], control["head"]
        if feature_set not in ("local4", "numerical11", "all19"):
            raise ValueError("control.feature_set must be local4, numerical11, or all19")
        if encoder not in ("transformer", "mlp") or head not in ("centered", "signed"):
            raise ValueError("control.encoder/head must be transformer|mlp and centered|signed")
        if encoder == "mlp" and feature_set != "all19":
            raise ValueError("the controlled MLP encoder requires all19 features")
        tok = {"mode": "masked" if feature_set == "local4" else "hourly", **(tokenization or {})}
        hourly_mode = tok["mode"] == "hourly"
        pred = {"mode": ("boundary_flux" if head == "signed" else "hourly_residual") if hourly_mode else "residual",
                "context": encoder, **(prediction or {})}
        pred["mode"] = (("boundary_flux" if head == "signed" else "hourly_residual")
                         if hourly_mode else "residual")
        pred["context"] = encoder
        cond = {"mechanism": "adaln_zero", "zone": feature_set != "all19",
                "weekday": False, "window_stats": True, **(conditioning or {})}
        if feature_set == "all19":
            if cond.get("zone") is True:
                raise ValueError("all19 features already contain zone indicators; conditioning.zone must be false")
            cond["zone"] = False
        pos = {"type": "rope", "base": 10000.0,
               "absolute_time": {"enabled": feature_set != "all19", "harmonics": [1, 2], "fusion": "additive"},
               **(position or {})}
        absolute = dict(pos.get("absolute_time") or {})
        absolute["enabled"] = feature_set != "all19"
        pos["absolute_time"] = absolute
        interp = {"enabled": head == "signed" and bool(interpolation and interpolation.get("enabled", False)),
                  **(interpolation or {})}
        if head == "centered":
            interp["enabled"] = bool((interpolation or {}).get("enabled", False))
        local = {"enabled": feature_set == "local4", "radius": 1, **(local_features or {})}
        local["enabled"] = feature_set == "local4"
        super().__init__(d_model=d_model, n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim,
                         dropout=dropout, lr=lr, weight_decay=weight_decay,
                         normalization=normalization, conditioning=cond, position=pos,
                         teacher_forcing=teacher_forcing, tokenization=tok, prediction=pred,
                         interpolation=interp, local_features=local)
        self.control_config = dict(control)
        self.hparams["control"] = dict(control)
        self.control_feature_projection = None
        if feature_set != "local4" and encoder == "transformer":
            # The parent output head has head-specific width.  Isolate this
            # shared projection's initialization from that unrelated RNG draw.
            seed = torch.initial_seed()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.control_feature_projection = nn.Linear(11 if feature_set == "numerical11" else 19, d_model)
        self.control_signed_head = SignedRedistributionHead(d_model) if head == "signed" else None
        if hourly_mode and feature_set != "local4":
            # Hourly packets already contain H; the parent scalar projection
            # is unused in this path and should not count as trainable.
            for parameter in self.value_projection.parameters():
                parameter.requires_grad_(False)
        if self.control_signed_head is not None:
            for parameter in self.head.parameters():
                parameter.requires_grad_(False)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        saved = checkpoint.get("hyper_parameters", {}).get("control")
        if saved != self.control_config:
            raise RuntimeError(f"checkpoint controlled-model configuration mismatch: checkpoint={saved!r}, model={self.control_config!r}")

    def _encode(self, hourly: torch.Tensor, zone: torch.Tensor | None,
                weekday: torch.Tensor | None, target: torch.Tensor | None,
                allow_reveal: bool, start_hour: torch.Tensor | None):
        feature_set, encoder = self.control_config["feature_set"], self.control_config["encoder"]
        tokens, types = self.build_tokens(hourly, zone, weekday, target, allow_reveal)
        length = tokens.shape[1]
        if self.time_projection is not None:
            from .temporal import absolute_time_features
            time = absolute_time_features(length, self.position_config["absolute_time"]["harmonics"],
                                          device=tokens.device, dtype=tokens.dtype, start_hour=start_hour)
            projected = self.time_projection(time)
            tokens = tokens + (projected if start_hour is not None else projected[None])
        if feature_set == "local4":
            x = tokens
        else:
            from .features import structured_hourly_features, structured_numerical_features
            global_scale = float(self.normalizer.global_std.detach().cpu()) if bool(self.normalizer.stats_ready) else 1.0
            features = (structured_numerical_features(hourly, global_scale) if feature_set == "numerical11"
                        else structured_hourly_features(hourly, zone, global_scale))
            if encoder == "mlp":
                x = self.context_mlp(features)
            else:
                projected = self.control_feature_projection(features)
                x = projected if self.tokenization_config["mode"] == "hourly" else tokens + projected.repeat_interleave(4, dim=1)
                if self.tokenization_config["mode"] == "hourly" and self.time_projection is not None:
                    projected_time = self.time_projection(time)
                    x = x + (projected_time if start_hour is not None else projected_time[None])
        condition = self._condition(hourly, zone, weekday)
        if encoder == "transformer":
            for block in self.blocks:
                x = block(x, condition)
        shift, scale_mod = self.final_modulation(condition).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale_mod[:, None]) + shift[:, None]
        return x, types

    def _predict_with_types(self, hourly: torch.Tensor, zone: torch.Tensor | None = None,
                            weekday: torch.Tensor | None = None, target: torch.Tensor | None = None,
                            allow_reveal: bool = False,
                            start_hour: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_start_hour(start_hour, hourly.shape[0])
        x, types = self._encode(hourly, zone, weekday, target, allow_reveal, start_hour)
        _, price_scale = self.normalizer.statistics(hourly)
        if self.control_config["head"] == "centered":
            from .constraints import project_hourly
            raw = self.head(x)
            if self.tokenization_config["mode"] == "masked":
                raw = raw.squeeze(-1)
            if raw.ndim == 3:
                raw = raw.reshape(-1, 96)
            pred = project_hourly(raw, hourly, price_scale)
            if self.interpolation_config["enabled"]:
                from .interpolation import makima_residual
                pred = pred + makima_residual(hourly, float(self.interpolation_config["amplitude"]))
                q = pred.reshape(-1, 24, 4)
                pred = (q + (hourly.unsqueeze(-1) - q.mean(-1, keepdim=True))).reshape(-1, 96)
        else:
            residual = self.control_signed_head(x)
            pred_q = hourly.unsqueeze(-1) + price_scale.unsqueeze(-1) * residual
            if self.interpolation_config["enabled"]:
                from .interpolation import makima_residual
                pred_q = pred_q + makima_residual(hourly, float(self.interpolation_config["amplitude"])).reshape(-1, 24, 4)
            pred_q = pred_q + hourly.unsqueeze(-1) - pred_q.mean(-1, keepdim=True)
            pred = pred_q.reshape(-1, 96)
        return pred, types

    @property
    def signed_head(self):
        """Compatibility alias without registering the module twice."""
        return self.control_signed_head


# Name used by controlled-study runner configs.
ControlledReconstructionModel = ControlledDailyPriceTransformer
