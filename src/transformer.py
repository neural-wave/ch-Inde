"""Daily masked-reconstruction transformer with hard hourly-mean consistency."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from lightning.pytorch import LightningModule

from .conditioning import AdaLNZeroBlock, DailyConditioner


class DailyPriceTransformer(LightningModule):
    """Bidirectional 96-token model; public forward is always label-free."""
    TOKEN_HOURLY, TOKEN_MASK, TOKEN_REVEALED = 0, 1, 2

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, ff_dim: int = 128,
                 dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
                 normalization: dict[str, Any] | None = None,
                 conditioning: dict[str, Any] | None = None,
                 position: dict[str, Any] | None = None,
                 teacher_forcing: dict[str, Any] | None = None,
                 tokenization: dict[str, Any] | None = None,
                 prediction: dict[str, Any] | None = None,
                 interpolation: dict[str, Any] | None = None,
                 local_features: dict[str, Any] | None = None) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or n_layers <= 0 or ff_dim <= 0 or d_model % n_heads:
            raise ValueError("invalid transformer dimensions")
        if not all(torch.isfinite(torch.tensor(float(x))) for x in (dropout, lr, weight_decay)) or not 0 <= dropout < 1 or lr <= 0 or weight_decay < 0:
            raise ValueError("invalid optimizer/dropout parameters")
        normalization = {"mode": "global", "scale_floor_fraction": 0.05, **(normalization or {})}
        conditioning = {"mechanism": "adaln_zero", "zone": True, "weekday": False, "window_stats": True, **(conditioning or {})}
        position = {"type": "rope", "base": 10000.0,
                    "absolute_time": {"enabled": False, "harmonics": [1, 2], "fusion": "additive"}, **(position or {})}
        position["absolute_time"] = {"enabled": False, "harmonics": [1, 2], "fusion": "additive", **(position.get("absolute_time") or {})}
        teacher_forcing = {"sample_probability": 0.0, "hour_probability": 0.25, **(teacher_forcing or {})}
        tokenization = {"mode": "masked", **(tokenization or {})}
        prediction = {"mode": "residual", **(prediction or {})}
        interpolation = {"enabled": False, "method": "makima", "amplitude": 2.0,
                         "boundary_policy": "extrapolate", "center_per_hour": True,
                         "inject_as_input": False, **(interpolation or {})}
        local_features = {"enabled": False, "radius": 1, **(local_features or {})}
        for name, config, allowed in (
            ("normalization", normalization, {"mode", "scale_floor_fraction"}),
            ("conditioning", conditioning, {"mechanism", "zone", "weekday", "window_stats"}),
            ("position", position, {"type", "base", "absolute_time"}),
            ("teacher_forcing", teacher_forcing, {"sample_probability", "hour_probability"}),
            ("local_features", local_features, {"enabled", "radius"}),
        ):
            unknown = set(config) - allowed
            if unknown:
                raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
        for name, config, allowed in (("tokenization", tokenization, {"mode", "sequence_length"}),
                                      ("prediction", prediction, {"mode", "coefficients_per_hour", "basis", "interpolation_residual", "potential_curvature_weight", "ode_steps", "context"}),
                                      ("interpolation", interpolation, {"enabled", "method", "amplitude", "boundary_policy", "center_per_hour", "inject_as_input"})):
            unknown = set(config) - allowed
            if unknown:
                raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
        if normalization.get("mode") not in ("global", "per_day"):
            raise ValueError("normalization.mode must be global or per_day")
        if not 0 <= float(normalization.get("scale_floor_fraction", 0.05)):
            raise ValueError("scale_floor_fraction must be nonnegative")
        if conditioning.get("mechanism", "adaln_zero") != "adaln_zero":
            raise ValueError("conditioning.mechanism must be adaln_zero")
        for key in ("zone", "weekday", "window_stats"):
            if not isinstance(conditioning[key], bool):
                raise ValueError(f"conditioning.{key} must be boolean")
        if position.get("type", "rope") != "rope" or float(position.get("base", 10000)) <= 0:
            raise ValueError("position must use RoPE with positive base")
        absolute = position["absolute_time"]
        if set(absolute) - {"enabled", "harmonics", "fusion"} or not isinstance(absolute["enabled"], bool) or absolute["fusion"] != "additive":
            raise ValueError("position.absolute_time requires enabled, harmonics, and additive fusion")
        if not absolute["harmonics"] or any(isinstance(h, bool) or int(h) != h or h <= 0 for h in absolute["harmonics"]):
            raise ValueError("position.absolute_time.harmonics must be positive integers")
        if tokenization["mode"] not in ("masked", "hourly"):
            raise ValueError("tokenization.mode must be masked or hourly")
        requested_prediction_mode = prediction["mode"]
        if prediction["mode"] in ("interpolation_residual",):
            prediction["mode"] = "residual"
        if prediction["mode"] in ("four_outputs",):
            prediction["mode"] = "hourly_residual"
        if prediction["mode"] not in ("residual", "hourly_residual", "coefficient_basis", "potential", "boundary_flux", "neural_ode"):
            raise ValueError("unsupported prediction.mode")
        integral_mode = prediction["mode"] in ("potential", "boundary_flux", "neural_ode")
        if (tokenization["mode"] == "masked") != (prediction["mode"] == "residual"):
            raise ValueError("masked tokenization requires residual prediction; hourly tokenization requires an hourly prediction mode")
        if integral_mode and tokenization["mode"] != "hourly":
            raise ValueError("integral prediction modes require hourly tokenization")
        if tokenization["mode"] == "hourly" and float(teacher_forcing["sample_probability"]) != 0:
            raise ValueError("hourly tokenization forbids teacher forcing; set both probabilities to zero")
        if tokenization.get("sequence_length", 24 if tokenization["mode"] == "hourly" else 96) != (24 if tokenization["mode"] == "hourly" else 96):
            raise ValueError("tokenization.sequence_length does not match tokenization.mode")
        if interpolation["method"] != "makima" or interpolation["boundary_policy"] != "extrapolate" or interpolation["center_per_hour"] is not True or not isinstance(interpolation["inject_as_input"], bool):
            raise ValueError("interpolation requires makima, extrapolate, and boolean centering/input options")
        if not torch.isfinite(torch.tensor(float(interpolation["amplitude"]))):
            raise ValueError("interpolation.amplitude must be finite")
        if interpolation["inject_as_input"]:
            raise ValueError("interpolation.inject_as_input=true is not supported")
        if requested_prediction_mode == "interpolation_residual" and not interpolation["enabled"]:
            raise ValueError("prediction.mode=interpolation_residual requires interpolation.enabled=true")
        if not isinstance(prediction.get("interpolation_residual", False), bool):
            raise ValueError("prediction.interpolation_residual must be boolean")
        if prediction.get("interpolation_residual", False) and not interpolation["enabled"]:
            raise ValueError("prediction.interpolation_residual=true requires interpolation.enabled=true")
        if not isinstance(local_features["enabled"], bool) or isinstance(local_features["radius"], bool) or not isinstance(local_features["radius"], int) or local_features["radius"] < 1:
            raise ValueError("local_features requires a boolean enabled and positive integer radius")
        if prediction["mode"] == "coefficient_basis" and (prediction.get("coefficients_per_hour", 3) != 3 or prediction.get("basis", "orthonormal_polynomial") != "orthonormal_polynomial"):
            raise ValueError("coefficient_basis requires three coefficients and the orthonormal_polynomial basis")
        if float(prediction.get("potential_curvature_weight", 0.0)) < 0:
            raise ValueError("potential_curvature_weight must be nonnegative")
        if int(prediction.get("ode_steps", 8)) not in (4, 8, 16):
            raise ValueError("prediction.ode_steps must be 4, 8, or 16")
        if prediction.get("context", "transformer") not in ("transformer", "mlp"):
            raise ValueError("prediction.context must be transformer or mlp")
        if prediction.get("context", "transformer") == "mlp" and prediction["mode"] != "boundary_flux":
            raise ValueError("prediction.context=mlp is currently supported for boundary_flux only")
        for key in ("sample_probability", "hour_probability"):
            value = float(teacher_forcing.get(key, 0.0 if key == "sample_probability" else 0.25))
            if not 0 <= value <= 1:
                raise ValueError(f"teacher_forcing.{key} must be in [0,1]")
        self.save_hyperparameters()
        # Checkpoint hparams must contain the resolved dictionaries so a Trainer
        # loading weights under a different CLI config cannot change inference.
        self.hparams["normalization"] = dict(normalization)
        self.hparams["conditioning"] = dict(conditioning)
        self.hparams["position"] = dict(position)
        self.hparams["teacher_forcing"] = dict(teacher_forcing)
        for key, value in (("tokenization", tokenization), ("prediction", prediction), ("interpolation", interpolation), ("local_features", local_features)):
            self.hparams[key] = dict(value)
        self.normalization_config = dict(normalization)
        self.conditioning_config = dict(conditioning)
        self.teacher_forcing_config = dict(teacher_forcing)
        self.position_config = dict(position)
        self.tokenization_config = dict(tokenization)
        self.prediction_config = dict(prediction)
        self.interpolation_config = dict(interpolation)
        self.local_features_config = dict(local_features)
        self.prediction_context = prediction.get("context", "transformer")
        from .normalization import DailyNormalizer
        self.normalizer = DailyNormalizer(**self.normalization_config)
        self.d_model = d_model
        self.value_projection = nn.Linear(1, d_model)
        self.local_feature_projection = nn.Linear(4, d_model) if local_features["enabled"] else None
        self.time_projection = nn.Linear(2 * len(absolute["harmonics"]), d_model) if absolute["enabled"] else None
        self.interpolation_projection = None
        if tokenization["mode"] == "masked":
            self.token_type = nn.Embedding(3, d_model)
            self.mask_token = nn.Parameter(torch.zeros(d_model))
        else:
            self.token_type = None
            self.register_parameter("mask_token", None)
        self.conditioner = DailyConditioner(d_model, bool(conditioning.get("zone", True)),
                                            bool(conditioning.get("weekday", False)),
                                            bool(conditioning.get("window_stats", True)))
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d_model, n_heads, ff_dim, dropout, float(position.get("base", 10000))) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        out_dim = 3 if prediction["mode"] in ("coefficient_basis", "boundary_flux") else (4 if prediction["mode"] == "hourly_residual" else 1)
        self.head = nn.Linear(d_model, out_dim if tokenization["mode"] == "hourly" else 1)
        self.context_mlp = nn.Sequential(nn.Linear(19, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, d_model)) if self.prediction_context == "mlp" else None
        inv = 1.0 / (20.0 ** 0.5)
        basis = torch.tensor([[-3*inv, .5, -inv], [-inv, -.5, 3*inv], [inv, -.5, -3*inv], [3*inv, .5, inv]], dtype=torch.float32)
        self.register_buffer("coefficient_basis", basis, persistent=False)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.potential_head = self.ode_head = None
        if prediction["mode"] == "potential":
            from .integral import PotentialHead
            self.potential_head = PotentialHead(d_model)
        elif prediction["mode"] == "neural_ode":
            from .integral import NeuralODEHead
            self.ode_head = NeuralODEHead(d_model, steps=int(prediction.get("ode_steps", 8)))
        if prediction["mode"] in ("potential", "neural_ode"):
            for parameter in self.head.parameters():
                parameter.requires_grad_(False)
        if self.prediction_context == "mlp":
            for module in (self.value_projection, self.blocks, self.token_type, self.mask_token):
                if isinstance(module, nn.Parameter):
                    module.requires_grad_(False)
                elif module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        from .metrics import DailyZoneMAE
        self.val_metric = DailyZoneMAE()
        self.test_metric = DailyZoneMAE()

    def set_normalization_stats(self, mean: torch.Tensor | float, std: torch.Tensor | float) -> None:
        self.normalizer.set_global_stats(mean, std)

    @property
    def hourly_head(self):
        """Compatibility alias for the shared hourly-token output projection."""
        return self.head if self.tokenization_config["mode"] == "hourly" else None

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject state-only restores when the instantiated scientific config differs."""
        saved = checkpoint.get("hyper_parameters")
        if saved is None:
            raise RuntimeError("checkpoint has no model hyperparameters; refusing an unverifiable weight load")
        keys = ("d_model", "n_heads", "n_layers", "ff_dim", "dropout", "lr", "weight_decay",
                "normalization", "conditioning", "position", "teacher_forcing", "tokenization", "prediction", "interpolation", "local_features")
        mismatches = []
        for key in keys:
            expected = self.hparams.get(key)
            actual = saved.get(key) if hasattr(saved, "get") else None
            if key == "position" and isinstance(actual, dict):
                actual = {"type": "rope", "base": 10000.0, **actual}
                actual["absolute_time"] = {"enabled": False, "harmonics": [1, 2], "fusion": "additive", **(actual.get("absolute_time") or {})}
            if key in ("tokenization", "prediction", "interpolation", "local_features") and actual is None:
                legacy_defaults = {"tokenization": {"mode": "masked"}, "prediction": {"mode": "residual"},
                    "interpolation": {"enabled": False, "method": "makima", "amplitude": 2.0,
                                      "boundary_policy": "extrapolate", "center_per_hour": True, "inject_as_input": False},
                    "local_features": {"enabled": False, "radius": 1}}
                actual = legacy_defaults[key]
            if actual != expected:
                mismatches.append(f"{key}: checkpoint={actual!r}, model={expected!r}")
        if mismatches:
            raise RuntimeError("checkpoint model configuration mismatch; use the matching config: " + "; ".join(mismatches))

    def on_fit_start(self) -> None:
        dm = self.trainer.datamodule
        if getattr(dm, "global_mean", None) is None or getattr(dm, "global_std", None) is None:
            raise RuntimeError("training data module has no training-only global statistics")
        if bool(self.normalizer.stats_ready):
            saved = torch.stack((self.normalizer.global_mean, self.normalizer.global_std))
            current = torch.stack((torch.as_tensor(dm.global_mean, device=saved.device), torch.as_tensor(dm.global_std, device=saved.device))).to(saved)
            if not torch.allclose(saved, current, rtol=0, atol=1e-6):
                raise RuntimeError("checkpoint normalization statistics differ from the training data")
        else:
            self.set_normalization_stats(dm.global_mean, dm.global_std)

    def _condition(self, hourly: torch.Tensor, zone: torch.Tensor | None, weekday: torch.Tensor | None) -> torch.Tensor:
        stats = self.normalizer.window_features(hourly) if self.conditioning_config.get("window_stats", True) else None
        return self.conditioner(hourly.shape[0], hourly.device, zone, weekday, stats)

    def build_tokens(self, hourly: torch.Tensor, zone: torch.Tensor | None = None,
                     weekday: torch.Tensor | None = None, target: torch.Tensor | None = None,
                     allow_reveal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(hourly) or hourly.dtype != torch.float32 or hourly.ndim != 2 or hourly.shape[1] != 24:
            raise ValueError("hourly must be float32 with shape [B,24]")
        self._validate_optional_id(zone, hourly.shape[0], 4, "zone")
        self._validate_optional_id(weekday, hourly.shape[0], 7, "weekday")
        if allow_reveal and target is None:
            raise ValueError("target is required when allow_reveal=True")
        if target is not None and (not torch.is_tensor(target) or target.dtype != torch.float32 or target.shape != (hourly.shape[0], 96)):
            raise ValueError("target must be float32 with shape [B,96]")
        if target is not None and (target.device != hourly.device or not torch.isfinite(target).all()):
            raise ValueError("target must be finite and on the hourly input device")
        mu, scale = self.normalizer.statistics(hourly)
        hnorm = self.normalizer.normalize(hourly, mu, scale)
        projected_local = None
        if self.local_feature_projection is not None:
            from .features import local_hourly_features
            local = local_hourly_features(hourly, int(self.local_features_config["radius"])) / scale[:, :, None]
            projected_local = self.local_feature_projection(local)
        if self.tokenization_config["mode"] == "hourly":
            tokens = self.value_projection(hnorm.unsqueeze(-1))
            if projected_local is not None:
                tokens = tokens + projected_local
            return tokens, torch.full((hourly.shape[0], 24), self.TOKEN_HOURLY, device=hourly.device, dtype=torch.long)
        tokens = self.mask_token[None, None].expand(hourly.shape[0], 96, -1).clone()
        types = torch.full((hourly.shape[0], 96), self.TOKEN_MASK, device=hourly.device, dtype=torch.long)
        tokens = tokens + self.token_type.weight[self.TOKEN_MASK]
        anchors = torch.arange(0, 96, 4, device=hourly.device)
        tokens[:, anchors] = self.value_projection(hnorm.unsqueeze(-1)) + self.token_type.weight[self.TOKEN_HOURLY]
        types[:, anchors] = self.TOKEN_HOURLY
        if allow_reveal and float(self.teacher_forcing_config.get("sample_probability", 0)) > 0:
            prob = float(self.teacher_forcing_config["sample_probability"])
            hprob = float(self.teacher_forcing_config["hour_probability"])
            eligible = (torch.rand((hourly.shape[0], 24), device=hourly.device) < hprob)
            selected = torch.rand((hourly.shape[0], 1), device=hourly.device) < prob
            eligible &= selected
            # choose one of the three former mask positions per selected hour
            q = torch.randint(1, 4, (hourly.shape[0], 24), device=hourly.device)
            slots = torch.arange(24, device=hourly.device)[None, :] * 4 + q
            bidx, hidx = eligible.nonzero(as_tuple=True)
            if bidx.numel():
                pos = slots[bidx, hidx]
                tnorm = self.normalizer.normalize(target, mu, scale)
                tokens[bidx, pos] = self.value_projection(tnorm[bidx, pos, None]) + self.token_type.weight[self.TOKEN_REVEALED]
                types[bidx, pos] = self.TOKEN_REVEALED
        if projected_local is not None:
            tokens = tokens + projected_local.repeat_interleave(4, dim=1)
        return tokens, types

    @staticmethod
    def _validate_optional_id(value: torch.Tensor | None, batch_size: int, upper: int, name: str) -> None:
        if value is None:
            return
        if (not torch.is_tensor(value) or value.shape != (batch_size,) or value.dtype == torch.bool
                or value.is_floating_point() or torch.any((value < 0) | (value >= upper))):
            raise ValueError(f"{name} must contain integer IDs in [0, {upper - 1}] per sample")

    @staticmethod
    def _validate_start_hour(value: torch.Tensor | None, batch_size: int) -> None:
        if value is None:
            return
        if (not torch.is_tensor(value) or value.shape != (batch_size,) or value.dtype == torch.bool
                or value.is_floating_point() or torch.any((value < 0) | (value >= 24))):
            raise ValueError("start_hour must contain integer hours in [0, 23] per sample")

    def _predict_with_types(self, hourly: torch.Tensor, zone: torch.Tensor | None = None,
                            weekday: torch.Tensor | None = None, target: torch.Tensor | None = None,
                            allow_reveal: bool = False,
                            start_hour: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_start_hour(start_hour, hourly.shape[0])
        tokens, types = self.build_tokens(hourly, zone, weekday, target, allow_reveal)
        length = tokens.shape[1]
        if self.time_projection is not None:
            from .temporal import absolute_time_features
            time = absolute_time_features(length, self.position_config["absolute_time"]["harmonics"], device=tokens.device, dtype=tokens.dtype, start_hour=start_hour)
            projected_time = self.time_projection(time)
            tokens = tokens + (projected_time if start_hour is not None else projected_time[None])
        interpolation = None
        if self.interpolation_config["enabled"]:
            from .interpolation import makima_residual
            interpolation = makima_residual(hourly, float(self.interpolation_config["amplitude"]))
        condition = self._condition(hourly, zone, weekday)
        x = tokens
        if self.context_mlp is not None:
            from .features import structured_hourly_features
            global_scale = float(self.normalizer.global_std.detach().cpu()) if bool(self.normalizer.stats_ready) else 1.0
            features = structured_hourly_features(hourly, zone, global_scale)
            x = self.context_mlp(features)
        else:
            for block in self.blocks:
                x = block(x, condition)
        shift, scale_mod = self.final_modulation(condition).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale_mod[:, None]) + shift[:, None]
        _, price_scale = self.normalizer.statistics(hourly)
        from .constraints import project_hourly
        if self.tokenization_config["mode"] == "hourly":
            raw = self.head(x)
            mode = self.prediction_config["mode"]
            if mode == "potential":
                from .integral import potential_to_quarters
                g = self.potential_head(x)
                pred = potential_to_quarters(g, hourly, price_scale).reshape(hourly.shape[0], 96)
            elif mode == "boundary_flux":
                from .integral import flux_to_quarters
                # Independent flux channels amplify output variance by 32x versus
                # centered four-output heads; this 1/sqrt(32) factor matches it.
                pred = flux_to_quarters(raw / (4.0 * (2.0 ** 0.5)), hourly, price_scale).reshape(hourly.shape[0], 96)
            elif mode == "neural_ode":
                residual = self.ode_head(x)
                pred = (hourly.unsqueeze(-1) + price_scale.unsqueeze(-1) * residual).reshape(hourly.shape[0], 96)
            elif mode == "coefficient_basis":
                residual = torch.einsum("bhc,qc->bhq", raw, self.coefficient_basis.to(raw)).reshape(hourly.shape[0], 96)
                pred = project_hourly(residual, hourly, price_scale)
            else:
                residual = raw.reshape(hourly.shape[0], 96)
                pred = project_hourly(residual, hourly, price_scale)
        else:
            residual = self.head(x).squeeze(-1)
            pred = project_hourly(residual, hourly, price_scale)
        if interpolation is not None:
            pred = pred + interpolation
            # Remove floating point drift while retaining the requested hourly grid.
            q = pred.reshape(-1, 24, 4)
            pred = (q + (hourly.unsqueeze(-1) - q.mean(dim=-1, keepdim=True))).reshape(-1, 96)
        return pred, types

    def _predict(self, hourly: torch.Tensor, zone: torch.Tensor | None = None,
                 weekday: torch.Tensor | None = None, target: torch.Tensor | None = None,
                 allow_reveal: bool = False, start_hour: torch.Tensor | None = None) -> torch.Tensor:
        return self._predict_with_types(hourly, zone, weekday, target, allow_reveal, start_hour)[0]

    def forward(self, hourly: torch.Tensor, zone: torch.Tensor | None = None,
                weekday: torch.Tensor | None = None,
                start_hour: torch.Tensor | None = None) -> torch.Tensor:
        return self._predict(hourly, zone, weekday, start_hour=start_hour)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        pred, token_types = self._predict_with_types(
            batch["hourly"], batch.get("zone"), batch.get("weekday"), batch["target"], True,
            batch.get("start_hour")
        )
        errors = torch.abs(pred - batch["target"])
        loss = errors.mean()
        objective = loss
        if self.prediction_config["mode"] == "potential" and float(self.prediction_config.get("potential_curvature_weight", 0.0)):
            quarters = pred.reshape(-1, 24, 4)
            curvature = quarters[..., 2:] - 2 * quarters[..., 1:-1] + quarters[..., :-2]
            penalty = float(self.prediction_config["potential_curvature_weight"]) * curvature.abs().mean()
            objective = loss + penalty
            self.log("train/potential_curvature_penalty", penalty, on_step=False, on_epoch=True, batch_size=pred.shape[0])
        batch_size = pred.shape[0]
        if token_types.shape[1] == 24:
            token_types = token_types.repeat_interleave(4, dim=1)
        revealed = token_types == self.TOKEN_REVEALED
        revealed_count = int(revealed.sum().item())
        unrevealed = ~revealed
        unrevealed_count = int(unrevealed.sum().item())
        self.log("train/mae", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if objective is not loss:
            self.log("train/objective", objective, on_step=False, on_epoch=True, batch_size=batch_size)
        if revealed_count:
            self.log("train/revealed_mae", errors[revealed].mean(), on_step=False, on_epoch=True,
                     batch_size=revealed_count)
        self.log("train/unrevealed_mae", errors[unrevealed].mean(), on_step=False, on_epoch=True,
                 batch_size=unrevealed_count)
        self.log("train/reveal_count", errors.new_tensor(revealed_count / batch_size),
                 on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/reveal_fraction", errors.new_tensor(revealed_count / errors.numel()),
                 on_step=False, on_epoch=True, batch_size=batch_size)
        return objective

    def _metric_update(self, metric, batch):
        pred = self(batch["hourly"], batch.get("zone"), batch.get("weekday"), batch.get("start_hour"))
        metric.update(pred, batch["target"], batch["delivery_date"], batch["zone"], hourly=batch["hourly"])
        original_hourly = batch.get("hourly_original", batch["hourly"])
        if torch.is_tensor(original_hourly):
            original_hourly = original_hourly.detach()
        return {"predictions": pred, "hourly": original_hourly, "zone": batch["zone"],
                "delivery_date": batch["delivery_date"], "target": batch["target"]}

    def on_validation_epoch_start(self): self.val_metric.reset()
    def validation_step(self, batch, batch_idx): return self._metric_update(self.val_metric, batch)
    def on_validation_epoch_end(self):
        result = self.val_metric.compute()
        for key, value in result.items(): self.log("val/" + key, value, prog_bar=(key == "mae"), sync_dist=False)
        self.val_metric.reset()
    def on_test_epoch_start(self): self.test_metric.reset()
    def test_step(self, batch, batch_idx): return self._metric_update(self.test_metric, batch)
    def on_test_epoch_end(self):
        result = self.test_metric.compute()
        for key, value in result.items(): self.log("test/" + key, value, sync_dist=False)
        self.test_metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        hourly_original = batch.get("hourly_original", batch["hourly"])
        if torch.is_tensor(hourly_original):
            hourly_original = hourly_original.detach()
        return {"predictions": self(batch["hourly"], batch.get("zone"), batch.get("weekday"), batch.get("start_hour")),
                "hourly": hourly_original, "zone": batch["zone"], "delivery_date": batch["delivery_date"]}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay))
