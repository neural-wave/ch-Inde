"""Daily, zone-keyed data pipeline for 15-minute price reconstruction."""
from __future__ import annotations

from datetime import date
from collections import Counter
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule

ZONES = ("ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3")
TRAIN_COLUMNS = {"delivery_date", "market_id", "hour_index", "quarter_index", "price_eur_mwh"}
HOURLY_COLUMNS = {"delivery_date", "market_id", "hour_index", "hourly_price_eur_mwh"}


def _dates(values: pd.Series) -> pd.Series:
    raw = values.astype("string")
    if raw.isna().any() or not raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("delivery_date must use YYYY-MM-DD")
    parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any() or (parsed.dt.strftime("%Y-%m-%d") != raw).any():
        raise ValueError("delivery_date contains invalid dates")
    return parsed.dt.strftime("%Y-%m-%d")


def _check_keys(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame[keys].isna().any().any():
        raise ValueError(f"{label} contains missing keys")
    dup = frame.duplicated(keys, keep=False)
    if dup.any():
        raise ValueError(f"{label} contains duplicate keys: {frame.loc[dup, keys].head().to_dict('records')}")


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = TRAIN_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"training data missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("training data is empty")
    f = frame[list(TRAIN_COLUMNS)].copy()
    f["delivery_date"] = _dates(f.delivery_date)
    for col in ("hour_index", "quarter_index"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
    f["price_eur_mwh"] = pd.to_numeric(f.price_eur_mwh, errors="coerce")
    if f[["hour_index", "quarter_index", "price_eur_mwh"]].isna().any().any() or not torch.isfinite(torch.tensor(f.price_eur_mwh.to_numpy(), dtype=torch.float64)).all():
        raise ValueError("training data contains non-finite or non-numeric values")
    if not f.hour_index.between(1, 24).all() or not f.quarter_index.between(1, 4).all() or not (f.hour_index % 1 == 0).all() or not (f.quarter_index % 1 == 0).all():
        raise ValueError("hour_index must be 1..24 and quarter_index must be 1..4")
    if not f.market_id.isin(ZONES).all():
        raise ValueError(f"market_id must be one of {ZONES}")
    f[["hour_index", "quarter_index"]] = f[["hour_index", "quarter_index"]].astype(int)
    _check_keys(f, ["delivery_date", "market_id", "hour_index", "quarter_index"], "training data")
    expected = 24 * 4
    counts = f.groupby(["delivery_date", "market_id"]).size()
    if (counts != expected).any():
        raise ValueError("every date-zone must contain exactly 24 hours x 4 quarters")
    zones = f.groupby("delivery_date").market_id.agg(lambda x: set(x))
    if any(z != set(ZONES) for z in zones):
        raise ValueError("every delivery date must contain all four synchronized zones")
    hours = f.groupby(["delivery_date", "market_id", "hour_index"]).quarter_index.agg(list)
    if any(sorted(qs) != [1, 2, 3, 4] for qs in hours):
        raise ValueError("every delivery date-zone-hour must contain quarters 1..4")
    return f.sort_values(["delivery_date", "market_id", "hour_index", "quarter_index"]).reset_index(drop=True)


def validate_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = HOURLY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"hourly prediction data missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("hourly prediction data is empty")
    f = frame[list(HOURLY_COLUMNS)].copy()
    f["delivery_date"] = _dates(f.delivery_date)
    f["hour_index"] = pd.to_numeric(f.hour_index, errors="coerce")
    f["hourly_price_eur_mwh"] = pd.to_numeric(f.hourly_price_eur_mwh, errors="coerce")
    if f[["hour_index", "hourly_price_eur_mwh"]].isna().any().any() or not torch.isfinite(torch.tensor(f.hourly_price_eur_mwh.to_numpy(), dtype=torch.float64)).all():
        raise ValueError("hourly prediction data contains non-finite or non-numeric values")
    if not f.hour_index.between(1, 24).all() or not (f.hour_index % 1 == 0).all() or not f.market_id.isin(ZONES).all():
        raise ValueError("hourly grid has invalid hour or market_id")
    f.hour_index = f.hour_index.astype(int)
    _check_keys(f, ["delivery_date", "market_id", "hour_index"], "hourly prediction data")
    counts = f.groupby(["delivery_date", "market_id"]).size()
    if (counts != 24).any() or any(set(z) != set(ZONES) for z in f.groupby("delivery_date").market_id.agg(set)):
        raise ValueError("each prediction date must have 24 hours in all four zones")
    return f.sort_values(["delivery_date", "market_id", "hour_index"]).reset_index(drop=True)


class DailyPriceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, labeled: bool):
        self.frame, self.labeled = frame, labeled
        self.rows = []
        for (day, zone), group in frame.groupby(["delivery_date", "market_id"], sort=True):
            group = group.sort_values("hour_index")
            hourly_original = group.drop_duplicates("hour_index").hourly_price_eur_mwh.to_numpy(dtype="float64")
            hourly = hourly_original.astype("float32")
            target = None
            if labeled:
                target = group.sort_values(["hour_index", "quarter_index"]).price_eur_mwh.to_numpy(dtype="float32")
            self.rows.append((day, ZONES.index(zone), hourly, hourly_original, target))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        day, zone, hourly, hourly_original, target = self.rows[i]
        result = {"hourly": torch.tensor(hourly), "hourly_original": torch.tensor(hourly_original, dtype=torch.float64), "zone": torch.tensor(zone, dtype=torch.long),
                  "weekday": torch.tensor(date.fromisoformat(day).weekday(), dtype=torch.long),
                  "start_hour": torch.tensor(0, dtype=torch.long), "delivery_date": day}
        if target is not None: result["target"] = torch.tensor(target)
        return result


class SlidingPriceDataset(Dataset):
    """24-consecutive-hour training windows, indexed by absolute delivery hours."""

    def __init__(self, frame: pd.DataFrame, stride_hours: int = 1):
        if type(stride_hours) is not int or stride_hours < 1:
            raise ValueError("stride_hours must be a positive integer")
        self.frame, self.stride_hours = frame, stride_hours
        self.rows = []
        self.window_metadata = []
        # Each zone is built independently, then synchronized by date and start hour.
        # Missing calendar dates split runs, so no window can bridge excluded DST dates.
        date_sets = {zone: set(frame.loc[frame.market_id == zone, "delivery_date"].unique()) for zone in ZONES}
        if any(days != date_sets[ZONES[0]] for days in date_sets.values()):
            raise ValueError("sliding windows require synchronized delivery dates across all zones")
        for zone in ZONES:
            by_day = {}
            for day, group in frame[frame.market_id == zone].groupby("delivery_date", sort=True):
                group = group.sort_values(["hour_index", "quarter_index"])
                hours = group.drop_duplicates("hour_index").sort_values("hour_index")
                if len(hours) != 24 or hours.hour_index.tolist() != list(range(1, 25)):
                    raise ValueError(f"incomplete hourly grid for {day} {zone}")
                target = group.price_eur_mwh.to_numpy(dtype="float32")
                by_day[day] = (hours.hourly_price_eur_mwh.to_numpy(dtype="float64"), target)

            runs, run = [], []
            for day in sorted(by_day):
                if run and (date.fromisoformat(day) - date.fromisoformat(run[-1])).days != 1:
                    runs.append(run)
                    run = []
                run.append(day)
            if run:
                runs.append(run)

            for days in runs:
                hourly_original = torch.tensor([v for day in days for v in by_day[day][0]], dtype=torch.float64)
                target = torch.tensor([v for day in days for v in by_day[day][1]], dtype=torch.float32)
                for absolute_start in range(0, len(hourly_original) - 24 + 1, stride_hours):
                    day_offset, start_hour = divmod(absolute_start, 24)
                    day = days[day_offset]
                    start = absolute_start
                    original_window = hourly_original[start:start + 24]
                    self.rows.append((day, ZONES.index(zone), original_window.float(), original_window,
                                      target[start * 4:(start + 24) * 4], start_hour))
                    self.window_metadata.append({"delivery_date": day, "market_id": zone,
                                                 "start_hour": start_hour,
                                                 "absolute_start_hour": absolute_start})

        # One window for each zone at each start timestamp is required.
        counts = Counter((m["delivery_date"], m["start_hour"], m["absolute_start_hour"], m["market_id"])
                         for m in self.window_metadata)
        timestamps = {(day, hour, absolute) for day, hour, absolute, _ in counts}
        if (any(counts[(day, hour, absolute, zone)] != 1
                for day, hour, absolute in timestamps for zone in ZONES)
                or len(counts) != len(timestamps) * len(ZONES)):
            raise ValueError("sliding windows are not synchronized across zones")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        day, zone, hourly, hourly_original, target, start_hour = self.rows[i]
        window_start = f"{day}T{start_hour:02d}:00:00"
        return {"hourly": hourly, "hourly_original": hourly_original,
                "zone": torch.tensor(zone, dtype=torch.long), "weekday": torch.tensor(date.fromisoformat(day).weekday(), dtype=torch.long),
                "start_hour": torch.tensor(start_hour, dtype=torch.long), "delivery_date": day,
                "window_start": window_start, "target": target}


def _collate(rows):
    string_keys = {k for k, value in rows[0].items() if isinstance(value, str)}
    out = {k: [r[k] for r in rows] if k in string_keys else torch.stack([r[k] for r in rows]) for k in rows[0]}
    return out


class DailyPriceDataModule(LightningDataModule):
    def __init__(self, train_path="data/train_prices_15min.csv", predict_path: Optional[str] = None,
                 batch_size: int = 32, num_workers: int = 0, train_end="2026-06-30",
                 val_start="2026-07-01", val_end="2026-07-31", test_start="2026-08-01", test_end="2026-09-30",
                 split_protocol: str = "static", cv_fold: int = 0,
                 training_window_stride_hours: int | None = None,
                 split_manifest: Optional[str] = None):
        super().__init__()
        self.train_path, self.predict_path = train_path, predict_path
        self.split_manifest = split_manifest
        self.batch_size, self.num_workers = batch_size, num_workers
        self.train_end, self.val_start, self.val_end = train_end, val_start, val_end
        self.test_start, self.test_end = test_start, test_end
        if split_manifest is None and split_protocol not in ("static", "cv"):
            raise ValueError("split_protocol must be static or cv")
        if type(cv_fold) is not int or cv_fold not in range(4):
            raise ValueError("cv_fold must be an integer from 0 to 3")
        self.split_protocol, self.cv_fold = split_protocol, cv_fold
        if training_window_stride_hours is not None and (type(training_window_stride_hours) is not int or training_window_stride_hours < 1):
            raise ValueError("training_window_stride_hours must be None or a positive integer")
        self.training_window_stride_hours = training_window_stride_hours
        self.split_dates = {}
        self.window_metadata = {}
        self.global_mean = None
        self.global_std = None
        self._sets = {}

    def prepare_data(self):
        # Lightning calls this before the stage is known; prediction-only installs may omit training data.
        if self.predict_path and not Path(self.predict_path).is_file():
            raise FileNotFoundError(f"prediction hourly file not found: {self.predict_path}")

    def setup(self, stage=None):
        if stage in (None, "fit", "validate", "test"):
            if not Path(self.train_path).is_file():
                raise FileNotFoundError(self.train_path)
            f = validate_training_frame(pd.read_csv(self.train_path))
            dt = pd.to_datetime(f.delivery_date)
            if self.split_manifest:
                assignments = self._read_split_manifest(f)
                pieces = [f[f.delivery_date.isin(assignments[name])] for name in ("train", "validation", "test")]
                self.split_dates = {name: sorted(assignments[key]) for name, key in zip(("train", "val", "test"), ("train", "validation", "test"))}
            elif self.split_protocol == "cv":
                held_out = dt.dt.day.between(7 * self.cv_fold + 1, 7 * self.cv_fold + 7)
                pieces = [f[~held_out], f.iloc[:0], f[held_out]]
            else:
                train_end, vs, ve, ts, te = map(pd.Timestamp, (self.train_end, self.val_start, self.val_end, self.test_start, self.test_end))
                if vs <= train_end or ts <= ve or vs > ve or ts > te:
                    raise ValueError("chronological split ranges must be valid, disjoint, and ordered")
                ranges = [(pd.Timestamp.min, train_end), (vs, ve), (ts, te)]
                pieces = [f[dt.between(a, b)] for a, b in ranges]
            self.split_dates = {name: sorted(part.delivery_date.unique().tolist())
                                for name, part in zip(("train", "val", "test"), pieces)}
            if not len(pieces[0]):
                raise ValueError("training split is empty")
            # Fit train-only statistics during fit; evaluate/predict uses checkpoint buffers.
            if stage in (None, "fit"):
                h = pieces[0].groupby(["delivery_date", "market_id", "hour_index"], as_index=False).price_eur_mwh.mean()
                vals = torch.tensor(h.price_eur_mwh.to_numpy(), dtype=torch.float64)
                self.global_mean = vals.mean().float()
                self.global_std = vals.std(correction=0).float()
                if not torch.isfinite(self.global_std) or self.global_std <= 0:
                    raise ValueError("training hourly global standard deviation must be finite and positive")
            for name, part in zip(("train", "val", "test"), pieces):
                hourly = part.groupby(["delivery_date", "market_id", "hour_index"], as_index=False).price_eur_mwh.mean().rename(columns={"price_eur_mwh": "hourly_price_eur_mwh"})
                targets = part.rename(columns={"price_eur_mwh": "_target"})
                merged = hourly.merge(targets[["delivery_date", "market_id", "hour_index", "quarter_index", "_target"]], on=["delivery_date", "market_id", "hour_index"], validate="one_to_many")
                labeled = merged.rename(columns={"_target":"price_eur_mwh"})
                if name == "train" and self.training_window_stride_hours is not None:
                    dataset = SlidingPriceDataset(labeled, stride_hours=self.training_window_stride_hours)
                    self.window_metadata[name] = dataset.window_metadata
                    self._sets[name] = dataset
                else:
                    self._sets[name] = DailyPriceDataset(labeled, labeled=True)
        if stage == "predict" or (stage is None and self.predict_path):
            if not self.predict_path:
                raise ValueError("predict_path is required for prediction")
            pf = validate_hourly_frame(pd.read_csv(self.predict_path))
            self._sets["predict"] = DailyPriceDataset(pf, labeled=False)

    def _loader(self, name, shuffle=False):
        if name not in self._sets:
            self.setup("predict" if name == "predict" else "fit")
        return DataLoader(self._sets[name], batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, collate_fn=_collate)
    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        if self.split_manifest and not self.split_dates.get("val"):
            return []
        return [] if self.split_protocol == "cv" else self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    def predict_dataloader(self):
        return self._loader("predict")

    def _read_split_manifest(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        """Read an exact split.json produced by src/evaluation/splits.py.

        Reserved and unused dates are intentionally never exposed as datasets. All
        source dates must nevertheless be accounted for, so a stale/incomplete
        manifest cannot silently drop examples or leak an omitted date.
        """
        path = Path(self.split_manifest)
        if not path.is_file():
            raise FileNotFoundError(f"split manifest not found: {path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("split manifest must contain a JSON object")
        parts = {}
        for key in ("train", "validation", "test", "reserved", "unused"):
            values = payload.get(key, [])
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError(f"manifest {key} dates must be a list of YYYY-MM-DD strings")
            normalized = _dates(pd.Series(values, dtype="string")).tolist() if values else []
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"manifest {key} contains duplicate dates")
            parts[key] = normalized
        flattened = [day for values in parts.values() for day in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError("manifest date partitions overlap")
        present = set(frame.delivery_date.unique())
        if set(flattened) != present:
            missing = sorted(present - set(flattened))
            unknown = sorted(set(flattened) - present)
            raise ValueError(f"manifest must partition all present source dates (missing={missing[:3]}, unknown={unknown[:3]})")
        if not parts["train"] or not (parts["validation"] or parts["test"]):
            raise ValueError("manifest needs training dates and at least one evaluation partition")
        return parts
