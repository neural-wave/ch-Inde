"""Hourly-only, exactly constrained structured quarter-hour reconstructions.

The estimators in this module fit only on the arrays passed to ``fit``.  In
particular, ``predict`` accepts no quarter-hour labels, so evaluation-day
targets cannot enter feature construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.special import expit
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


_BASIS = np.array([
    [-3 / np.sqrt(20), .5, -1 / np.sqrt(20)],
    [-1 / np.sqrt(20), -.5, 3 / np.sqrt(20)],
    [1 / np.sqrt(20), -.5, -3 / np.sqrt(20)],
    [3 / np.sqrt(20), .5, 1 / np.sqrt(20)],
], dtype=np.float64)


def zero_mean_basis() -> np.ndarray:
    """Return the fixed orthonormal 4-by-3 zero-sum basis."""
    return _BASIS.copy()


def _validate_inputs(hourly: np.ndarray, zones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(hourly, dtype=np.float64)
    z = np.asarray(zones)
    if h.ndim != 2 or h.shape[1] != 24:
        raise ValueError("hourly must have shape [days, 24]")
    if z.shape != (h.shape[0],):
        raise ValueError("zones must have shape [days]")
    if not np.isfinite(h).all():
        raise ValueError("hourly inputs must be finite")
    if not np.isin(z, np.arange(4)).all():
        raise ValueError("zones must be integer IDs 0..3")
    return h, z.astype(np.int64)


def makima_centered(hourly: np.ndarray) -> np.ndarray:
    """Hour-centered MAKIMA quarters, extrapolated at the day boundaries.

    Returns [days, 24, 4], with each hour's four values centered to mean zero.
    """
    h = np.asarray(hourly, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 24 or not np.isfinite(h).all():
        raise ValueError("hourly must be a finite [days, 24] array")
    x = np.arange(24, dtype=np.float64) + .5
    qx = np.arange(96, dtype=np.float64) / 4 + .125
    out = np.empty((len(h), 24, 4), dtype=np.float64)
    for i, row in enumerate(h):
        curve = Akima1DInterpolator(x, row, method="makima", extrapolate=True)(qx)
        curve = curve.reshape(24, 4)
        out[i] = curve - curve.mean(axis=1, keepdims=True)
    return out


_STAGE_LIMITS = {
    "gradients": 3,
    "curvature": 4,
    "level_scale": 9,
    "zone_time": 17,
    "local_volatility": 19,
}


def _stage_limit(stage: str | int) -> int:
    if isinstance(stage, (int, np.integer)):
        if stage not in range(1, 6):
            raise ValueError("feature_stage integer must be from 1 to 5")
        return list(_STAGE_LIMITS.values())[int(stage) - 1]
    try:
        return _STAGE_LIMITS[stage]
    except KeyError as exc:
        raise ValueError(f"feature_stage must be one of {tuple(_STAGE_LIMITS)}") from exc


def _features(hourly: np.ndarray, zones: np.ndarray, global_scale: float) -> np.ndarray:
    """Hourly-only context; one row per day/hour, shared by L1 and L2."""
    n = hourly.shape[0]
    scale = np.maximum(hourly.std(axis=1, ddof=0), .05 * global_scale)
    mean = hourly.mean(axis=1)
    prev = np.concatenate([hourly[:, :1], hourly[:, :-1]], axis=1)
    nxt = np.concatenate([hourly[:, 1:], hourly[:, -1:]], axis=1)
    left = hourly - prev
    right = nxt - hourly
    curvature = right - left
    padded = np.pad(hourly, ((0, 0), (2, 2)), mode="edge")
    local_vol = np.stack([
        np.stack([padded[:, j:j + 24] for j in range(2 - radius, 2 + radius + 1)], axis=-1).std(axis=-1)
        for radius in (1, 2)
    ], axis=-1)
    hour = np.arange(24, dtype=np.float64)
    periodic = np.stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                         np.sin(4*np.pi*hour/24), np.cos(4*np.pi*hour/24)], axis=-1)
    zone_onehot = np.eye(4, dtype=np.float64)[zones]
    g = max(global_scale, np.finfo(np.float64).eps)
    pieces = [(hourly / scale[:, None])[..., None], (left / scale[:, None])[..., None],
              (right / scale[:, None])[..., None], (curvature / scale[:, None])[..., None],
              (left / g)[..., None], (right / g)[..., None], (hourly / g)[..., None],
              np.broadcast_to((mean / g)[:, None, None], (n, 24, 1)),
              np.broadcast_to(np.log(np.maximum(scale / g, 1e-12))[:, None, None], (n, 24, 1)),
              np.broadcast_to(zone_onehot[:, None, :], (n, 24, 4)),
              np.broadcast_to(periodic[None, :, :], (n, 24, 4)),
              local_vol / scale[:, None, None]]
    return np.concatenate(pieces, axis=-1).reshape(n * 24, -1)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std, mean, std


@dataclass
class LinearReconstructor:
    """L1 ridge or robust linear coefficient regressor.

    ``baseline`` is ``flat`` or ``makima``; ``loss`` is ``ridge`` or ``huber``.
    All target transforms and feature statistics are learned from fit data only.
    """
    baseline: str = "flat"
    loss: str = "ridge"
    alpha: float = 1.0
    huber_delta: float = 1.35
    huber_iterations: int = 500
    seed: int = 42
    feature_stage: str | int = "local_volatility"

    def fit(self, hourly: np.ndarray, targets: np.ndarray, zones: np.ndarray) -> "LinearReconstructor":
        h, z = _validate_inputs(hourly, zones)
        y = np.asarray(targets, dtype=np.float64)
        if y.shape != (len(h), 96) or not np.isfinite(y).all():
            raise ValueError("targets must be finite with shape [days, 96]")
        if self.baseline not in {"flat", "makima"} or self.loss not in {"ridge", "huber"}:
            raise ValueError("baseline must be flat/makima and loss ridge/huber")
        limit = _stage_limit(self.feature_stage)
        self.global_scale_ = max(float(h.std()), np.finfo(float).eps)
        raw_x = _features(h, z, self.global_scale_)
        # Feature stages mirror the proposal's cumulative ablations.
        raw_x = raw_x[:, :limit]
        scale = np.maximum(h.std(axis=1), .05 * self.global_scale_)
        base = 2.0 * makima_centered(h) if self.baseline == "makima" else np.zeros((len(h), 24, 4))
        residual = y.reshape(-1, 24, 4) - h[:, :, None] - base
        coeff = np.einsum("qk,nhq->nhk", _BASIS, residual) / scale[:, None, None]
        targets_z = coeff.reshape(-1, 3)
        if self.loss == "ridge":
            self.estimator_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
            self.estimator_.fit(raw_x, targets_z)
        else:
            self.estimators_ = []
            for j in range(3):
                pipeline = make_pipeline(
                    StandardScaler(),
                    HuberRegressor(epsilon=self.huber_delta, alpha=self.alpha,
                                   max_iter=self.huber_iterations, fit_intercept=True),
                )
                pipeline.fit(raw_x, targets_z[:, j])
                self.estimators_.append(pipeline)
        self.scale_floor_ = .05 * self.global_scale_
        self.is_fitted_ = True
        return self

    def predict(self, hourly: np.ndarray, zones: np.ndarray) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("fit must be called before predict")
        h, z = _validate_inputs(hourly, zones)
        raw_x = _features(h, z, self.global_scale_)[:, :_stage_limit(self.feature_stage)]
        if self.loss == "ridge":
            coefficient = self.estimator_.predict(raw_x)
        else:
            coefficient = np.column_stack([est.predict(raw_x) for est in self.estimators_])
        coefficient = coefficient.reshape(len(h), 24, 3)
        scale = np.maximum(h.std(axis=1), self.scale_floor_)
        base = 2.0 * makima_centered(h) if self.baseline == "makima" else np.zeros((len(h), 24, 4))
        result = h[:, :, None] + base + scale[:, None, None] * np.einsum("qk,nhk->nhq", _BASIS, coefficient)
        # Roundoff-only projection; model residuals already lie in the basis.
        result += (h - result.mean(axis=2))[:, :, None]
        return result.reshape(len(h), 96)


@dataclass
class AdaptiveMakima:
    """L2 MAKIMA amplitude with fixed controls and a bounded trained gate.

    ``mode`` is ``fixed`` (alpha 1 or 2), ``global`` (one training-fit alpha),
    or ``adaptive`` (hourwise linear-logit sigmoid, bounded by max_amplitude).
    The global/adaptive fits optimize physical-price MAE using deterministic
    torch LBFGS; no validation data are accepted or inspected.
    """
    mode: str = "adaptive"
    amplitude: float = 2.0
    max_amplitude: float = 4.0
    l2: float = 1e-4
    max_iter: int = 250
    seed: int = 42
    feature_stage: str | int = "local_volatility"

    def fit(self, hourly: np.ndarray, targets: np.ndarray, zones: np.ndarray) -> "AdaptiveMakima":
        import torch

        h, z = _validate_inputs(hourly, zones)
        y = np.asarray(targets, dtype=np.float64)
        if y.shape != (len(h), 96) or not np.isfinite(y).all():
            raise ValueError("targets must be finite with shape [days, 96]")
        if self.mode not in {"fixed", "global", "adaptive"}:
            raise ValueError("mode must be fixed, global, or adaptive")
        limit = _stage_limit(self.feature_stage)
        self.global_scale_ = max(float(h.std()), np.finfo(float).eps)
        self.feature_mean_ = self.feature_scale_ = None
        shape = makima_centered(h)
        target_residual = y.reshape(-1, 24, 4) - h[:, :, None]
        if self.mode == "fixed":
            if self.amplitude not in (1, 2):
                raise ValueError("fixed amplitude control must be 1 or 2")
            self.alpha_ = float(self.amplitude)
            return self
        torch.manual_seed(self.seed)
        xt = torch.as_tensor(_features(h, z, self.global_scale_), dtype=torch.float64)
        yt = torch.as_tensor(target_residual, dtype=torch.float64)
        bt = torch.as_tensor(shape, dtype=torch.float64)
        if self.mode == "global":
            raw = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            optimizer = torch.optim.LBFGS([raw], max_iter=self.max_iter, line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                alpha = self.max_amplitude * torch.sigmoid(raw)
                pred = bt * alpha
                loss = torch.mean(torch.abs(pred - yt)) + self.l2 * (alpha - 2).square()
                loss.backward()
                return loss
            optimizer.step(closure)
            self.alpha_ = float((self.max_amplitude * torch.sigmoid(raw)).detach())
        else:
            raw = xt.numpy()[:, :limit]
            x, self.feature_mean_, self.feature_scale_ = _standardize_fit(raw)
            xt = torch.as_tensor(x, dtype=torch.float64)
            weights = torch.nn.Parameter(torch.zeros(xt.shape[1], dtype=torch.float64))
            bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            optimizer = torch.optim.LBFGS([weights, bias], max_iter=self.max_iter, line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                alpha = self.max_amplitude * torch.sigmoid(xt @ weights + bias)
                pred = bt * alpha.reshape(len(h), 24, 1)
                loss = torch.mean(torch.abs(pred - yt)) + self.l2 * (alpha - 2).square().mean()
                loss.backward()
                return loss
            optimizer.step(closure)
            self.weights_ = weights.detach().numpy().copy()
            self.bias_ = float(bias.detach())
        return self

    def predict(self, hourly: np.ndarray, zones: np.ndarray) -> np.ndarray:
        if not hasattr(self, "alpha_") and not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before predict")
        h, z = _validate_inputs(hourly, zones)
        shape = makima_centered(h)
        if self.mode == "adaptive":
            raw = _features(h, z, self.global_scale_)[:, :_stage_limit(self.feature_stage)]
            raw = (raw - self.feature_mean_) / self.feature_scale_
            alpha = self.max_amplitude * expit(raw @ self.weights_ + self.bias_)
            alpha = alpha.reshape(len(h), 24, 1)
        else:
            alpha = self.alpha_
        result = h[:, :, None] + alpha * shape
        result += (h - result.mean(axis=2))[:, :, None]
        return result.reshape(len(h), 96)


# Concise aliases for callers that prefer model-family names.
L1Model = LinearReconstructor
L2Model = AdaptiveMakima

