"""Official date-then-zone MAE accumulator."""
from collections import OrderedDict

import numpy as np
import torch


class DailyZoneMAE:
    def __init__(self):
        self.reset()

    def reset(self):
        self._rows = []

    def update(self, predictions, targets, dates, zones, hourly=None):
        p = np.asarray(
            predictions.detach().cpu() if torch.is_tensor(predictions) else predictions,
            dtype=np.float64,
        )
        t = np.asarray(
            targets.detach().cpu() if torch.is_tensor(targets) else targets,
            dtype=np.float64,
        )
        z = np.asarray(zones.detach().cpu() if torch.is_tensor(zones) else zones)
        if p.shape != t.shape or p.ndim != 2 or p.shape[1] != 96:
            raise ValueError("predictions and targets must have matching [B,96] shapes")
        if len(dates) != len(p) or len(z) != len(p):
            raise ValueError("date/zone batch size mismatch")
        if not np.isfinite(p).all() or not np.isfinite(t).all():
            raise ValueError("predictions and targets must be finite")
        h = None
        if hourly is not None:
            h = np.asarray(
                hourly.detach().cpu() if torch.is_tensor(hourly) else hourly,
                dtype=np.float64,
            )
            if h.shape != (len(p), 24):
                raise ValueError("hourly must be [B,24]")
            if not np.isfinite(h).all():
                raise ValueError("hourly inputs must be finite")

        for i, (day, zone) in enumerate(zip(dates, z)):
            zi = int(zone)
            if zi not in range(4):
                raise ValueError(f"invalid zone id {zi}")
            hourly_error = 0.0
            if h is not None:
                means = p[i].reshape(24, 4).mean(axis=1)
                hourly_error = float(np.max(np.abs(means - h[i])))
            self._rows.append((str(day), zi, p[i].copy(), t[i].copy(), hourly_error))

    def compute(self):
        rows = self._rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for part in gathered for row in part]

        unique = OrderedDict()
        for day, zone, pred, target, hourly_error in rows:
            key = (day, zone)
            if key in unique:
                old_pred, old_target, old_hourly_error = unique[key]
                if (
                    not np.array_equal(old_pred, pred)
                    or not np.array_equal(old_target, target)
                    or old_hourly_error != hourly_error
                ):
                    raise ValueError(f"conflicting duplicate distributed prediction for {key}")
            else:
                unique[key] = (pred, target, hourly_error)

        if not unique:
            raise ValueError("no daily predictions accumulated")
        missing = set(range(4)) - {zone for _, zone in unique}
        if missing:
            raise ValueError(f"missing zones from metric: {sorted(missing)}")
        for day in {day for day, _ in unique}:
            present = {zone for date_, zone in unique if date_ == day}
            if present != set(range(4)):
                raise ValueError(
                    f"delivery date {day} has incomplete zone coverage: {sorted(present)}"
                )

        zone_errors = {zone: [] for zone in range(4)}
        quarter_errors = [[] for _ in range(4)]
        max_hourly_error = 0.0
        for (day, zone), (pred, target, hourly_error) in unique.items():
            errors = np.abs(pred - target).reshape(24, 4)
            zone_errors[zone].append(float(errors.mean()))
            for quarter in range(4):
                quarter_errors[quarter].append(float(errors[:, quarter].mean()))
            max_hourly_error = max(max_hourly_error, hourly_error)

        zone_mae = {zone: float(np.mean(errors)) for zone, errors in zone_errors.items()}
        result = {"mae": float(np.mean(list(zone_mae.values())))}
        result.update({f"zone_{zone}_mae": zone_mae[zone] for zone in range(4)})
        result.update(
            {
                f"quarter_{quarter + 1}_mae": float(np.mean(quarter_errors[quarter]))
                for quarter in range(4)
            }
        )
        result["max_hourly_mean_error"] = max_hourly_error
        return result
