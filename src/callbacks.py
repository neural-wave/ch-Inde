"""Local, reproducible run bookkeeping for the daily transformer CLI."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)
SOURCE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RunSettings:
    experiment: str = "daily-transformer"
    run_id: str = "full-mask-global-zone"
    entity: str = ""
    project: str = ""
    group: str = ""
    resume: bool = False
    smoke_test: bool = False
    output_root: str = "artifacts/runs"
    selection_metric: str = "final_epoch"
    save_last: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0


def _sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_revision() -> dict[str, Any]:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE_ROOT, text=True).strip())
        return {"revision": rev, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        manifest_path = SOURCE_ROOT / "source_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            return {"revision": manifest["git_revision"], "dirty": manifest["git_dirty"]}
        return {"revision": None, "dirty": None}


def _source_fingerprint() -> str:
    """Fingerprint neural source so changed dirty trees cannot pass a resume check."""
    root = SOURCE_ROOT / "src/neural"
    h = hashlib.sha256()
    if root.exists():
        for path in sorted(root.rglob("*.py")):
            h.update(path.relative_to(SOURCE_ROOT).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _plain(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        value = vars(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class RunArtifactsCallback(Callback):
    """Persist resolved settings and keyed stage outputs below one run leaf."""

    def __init__(self, config: dict[str, Any], settings: RunSettings, stage: str = "fit"):
        super().__init__()
        self.config = _plain(config)
        self.settings = settings
        self.root = Path(settings.output_root) / settings.experiment / settings.run_id
        self.stage = stage
        self.started = False
        self.owns_run = False
        self.predictions: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.resolved_yaml = yaml.safe_dump(self.config, sort_keys=True, allow_unicode=True)
        self.identity_config = json.loads(json.dumps(self.config))
        self.identity_config.get("run", {}).pop("resume", None)
        self.identity_config.pop("ckpt_path", None)
        if isinstance(self.identity_config.get("trainer"), dict): self.identity_config["trainer"].pop("ckpt_path", None)

    def _prepare(self, trainer, stage: str) -> None:
        if self.started: return
        if getattr(trainer, "world_size", 1) != 1:
            raise RuntimeError("run artifact callbacks currently support single-process execution only (world_size must be 1)")
        self.stage = stage
        self.predictions.clear()
        for label in (self.settings.experiment, self.settings.run_id):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", label) or re.search(r"20\d{2}[-_]?\d{2}[-_]?\d{2}", label):
                raise ValueError("run experiment and run_id must be stable, timestamp-free path-safe labels")
        self.root.mkdir(parents=True, exist_ok=True)
        config_path = self.root / "config.yaml"
        scientific_yaml = yaml.safe_dump(self.identity_config, sort_keys=True, allow_unicode=True)
        digest = hashlib.sha256(scientific_yaml.encode()).hexdigest()
        dm = trainer.datamodule
        train_path = getattr(dm, "train_path", None)
        predict_path = getattr(dm, "predict_path", None)
        split_manifest = getattr(dm, "split_manifest", None)
        identity = {
            "config_sha256": digest,
            "source": _git_revision(),
            "source_fingerprint": _source_fingerprint(),
            "data": {str(p): _sha256(p) for p in (train_path, predict_path, split_manifest) if p},
        }
        if stage != "fit" and self.config.get("ckpt_path"):
            checkpoint = Path(self.config["ckpt_path"]).expanduser().resolve()
            identity["evaluation_checkpoint"] = {
                "path": str(checkpoint), "sha256": _sha256(checkpoint),
            }
        meta_path = self.root / "run.json"
        if meta_path.exists():
            previous = json.loads(meta_path.read_text())
            if not self.settings.resume:
                raise FileExistsError(f"run directory already exists; use run.resume=true only for an exact resume: {self.root}")
            if previous.get("identity") != identity:
                raise ValueError("cannot resume run: resolved configuration, source revision, or data identity differs")
        elif self.settings.resume:
            raise FileNotFoundError(f"cannot resume because run metadata does not exist: {meta_path}")
        if getattr(dm, "split_dates", None):
            (self.root / "split_dates.json").write_text(json.dumps({
                "protocol": dm.split_protocol, "fold": dm.cv_fold if dm.split_protocol == "cv" else None,
                "dates": dm.split_dates}, indent=2) + "\n")
        config_path.write_text(self.resolved_yaml)
        metadata = {
            "experiment": self.settings.experiment,
            "run_id": self.settings.run_id,
            "status": "running",
            "stage": stage,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "seed": self.config.get("seed_everything", self.config.get("seed", 42)),
            "evaluation_checkpoint": self.config.get("ckpt_path"),
            "selection_metric": self.settings.selection_metric if stage == "fit" else None,
            "score_eur_mwh": None,
        }
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self.started = True
        self.owns_run = True
        logger = trainer.logger
        if logger:
            # W&B's run config stays nested under the requested `config` key.
            logger.log_hyperparams({"config": self.config, "run": asdict(self.settings), **_git_revision()})
            dm = trainer.datamodule
            summary = {
                "train_day_zone_samples": len(getattr(dm, "_sets", {}).get("train", [])),
                "validation_day_zone_samples": len(getattr(dm, "_sets", {}).get("val", [])),
                "test_day_zone_samples": len(getattr(dm, "_sets", {}).get("test", [])),
                "input_hours": 24,
                "output_quarters": 96,
                "zones": 4,
            }
            logger.log_hyperparams({"data_summary": summary})
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "save"):
                # W&B reserves its run-root config.yaml for its own metadata.
                # A symlink uploaded under that name lets W&B overwrite our source.
                upload_path = self.root / "resolved" / "config.yaml"
                upload_path.parent.mkdir(exist_ok=True)
                upload_path.write_text(self.resolved_yaml)
                experiment.save(str(upload_path), base_path=str(self.root), policy="now")
        self._trainer = trainer
        log.info(json.dumps({"event": "run_started", "experiment": self.settings.experiment, "run_id": self.settings.run_id, "stage": stage, "config_sha256": digest}, sort_keys=True))

    def on_fit_start(self, trainer, pl_module): self._prepare(trainer, "fit")
    def on_validation_start(self, trainer, pl_module):
        if self.stage == "validate": self._prepare(trainer, "validate")
    def on_test_start(self, trainer, pl_module): self._prepare(trainer, "test")
    def on_predict_start(self, trainer, pl_module): self._prepare(trainer, "predict")

    def on_validation_end(self, trainer, pl_module):
        for key, value in trainer.callback_metrics.items():
            try: self.metrics[str(key)] = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            except (TypeError, ValueError): continue
        if self.metrics:
            p = self.root / "metrics.json"; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.metrics, indent=2, sort_keys=True) + "\n")
        if self.stage == "fit":
            self._save_selection(trainer)
        if self.stage == "validate":
            self._save_stage_score(trainer, "val/mae")
            self._finish("complete")

    def _save_selection(self, trainer):
        if self.settings.selection_metric == "final_epoch":
            path = self.root / "run.json"
            if path.exists():
                meta = json.loads(path.read_text())
                meta.update(evaluation_checkpoint=str(self.root / "checkpoints" / "last.ckpt"),
                            selection_metric="final_epoch", score_eur_mwh=None)
                path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            return
        for callback in trainer.callbacks:
            if not isinstance(callback, ModelCheckpoint) or getattr(callback, "monitor", None) != self.settings.selection_metric: continue
            path = self.root / "run.json"
            if not path.exists(): return
            meta = json.loads(path.read_text())
            score = getattr(callback, "best_model_score", None)
            meta["evaluation_checkpoint"] = getattr(callback, "best_model_path", "") or None
            meta["selection_metric"] = self.settings.selection_metric
            meta["selection_mode"] = "min"
            meta["score_eur_mwh"] = float(score.detach().cpu()) if hasattr(score, "detach") else (float(score) if score is not None else None)
            path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            return

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._collect_predictions(outputs, batch)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._collect_predictions(outputs, batch)

    def _collect_predictions(self, outputs, batch):
        if not isinstance(outputs, dict) or "predictions" not in outputs:
            raise ValueError("test_step/predict_step must return a mapping containing predictions")
        pred = outputs["predictions"].detach().cpu().numpy() if hasattr(outputs["predictions"], "detach") else np.asarray(outputs["predictions"])
        raw_hours = batch.get("hourly_original", batch["hourly"])
        hours = raw_hours.detach().cpu().numpy() if hasattr(raw_hours, "detach") else np.asarray(raw_hours)
        dates, zones = outputs.get("delivery_date", batch["delivery_date"]), outputs.get("zone", batch["zone"])
        if hasattr(zones, "detach"): zones = zones.detach().cpu().numpy()
        self.predictions.append({"predictions": pred, "hourly": hours, "dates": list(dates), "zones": np.asarray(zones).tolist()})

    def on_exception(self, trainer, pl_module, exception):
        if self.owns_run: self._finish("failed", error=f"{type(exception).__name__}: {exception}")

    def on_train_epoch_end(self, trainer, pl_module):
        if self.settings.selection_metric == "val/mae":
            # Some Lightning versions refresh last only when the top-k improves.
            # Save the actual epoch state independently of validation selection.
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint) and callback.monitor == "val/mae":
                    last = Path(callback.dirpath) / "last.ckpt"
                    callback.last_model_path = str(last)
                    trainer.save_checkpoint(str(last))
                    break

    def on_fit_end(self, trainer, pl_module):
        self._save_selection(trainer)
        self._finish("complete")
    def on_test_end(self, trainer, pl_module):
        self._capture_metrics(trainer)
        self._log_final_test_metrics(trainer)
        if self.predictions: save_prediction_batches(self.predictions, self.root / "predictions.csv")
        self._log_predictions(trainer)
        self._save_stage_score(trainer, "test/mae")
        self._finish("complete")
    def on_predict_end(self, trainer, pl_module):
        if self.predictions:
            save_prediction_batches(self.predictions, self.root / "predictions.csv")
            self._log_predictions(trainer)
        self._finish("complete")

    def _capture_metrics(self, trainer):
        for key, value in trainer.callback_metrics.items():
            try: self.metrics[str(key)] = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            except (TypeError, ValueError): continue
        if self.metrics:
            p = self.root / "metrics.json"
            p.write_text(json.dumps(self.metrics, indent=2, sort_keys=True) + "\n")

    def _log_final_test_metrics(self, trainer):
        """Send finalized scalar test metrics to the active tracker and its summary."""
        logger = getattr(trainer, "logger", None)
        if not logger:
            return
        test_metrics = {key: value for key, value in self.metrics.items()
                        if key.startswith("test/") and np.isscalar(value)}
        if not test_metrics:
            return
        # Explicit logging here ensures aggregated test_epoch_end values are sent
        # even when the logger did not receive them during the test loop.
        logger.log_metrics(test_metrics, step=getattr(trainer, "global_step", None))
        experiment = getattr(logger, "experiment", None)
        summary = getattr(experiment, "summary", None)
        if summary is not None:
            for key, value in test_metrics.items():
                summary[key] = value

    def _save_stage_score(self, trainer, key):
        score = trainer.callback_metrics.get(key)
        path = self.root / "run.json"
        meta = json.loads(path.read_text())
        meta["selection_metric"] = self.settings.selection_metric
        meta["evaluation_metric"] = key
        meta["score_eur_mwh"] = float(score.detach().cpu()) if hasattr(score, "detach") else (float(score) if score is not None else None)
        path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    def _log_predictions(self, trainer):
        if not trainer.logger or not self.predictions: return
        path = self.root / "predictions.csv"
        if path.exists():
            frame = pd.read_csv(path)
            trainer.logger.log_table("predictions", columns=list(frame.columns), data=frame.values.tolist())

    def _finish(self, status: str, error: str | None = None):
        if not self.owns_run: return
        path = self.root / "run.json"
        if not path.exists(): return
        meta = json.loads(path.read_text())
        meta.update(status=status, finished_at_utc=datetime.now(timezone.utc).isoformat(), error=error)
        path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def save_prediction_batches(batches: list[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    """Validate and serialize model prediction batches, including CSV round-trip."""
    if not batches: raise ValueError("prediction stage produced no batches")
    pred = np.concatenate([np.asarray(b["predictions"]) for b in batches])
    hourly = np.concatenate([np.asarray(b.get("hourly_original", b["hourly"])) for b in batches])
    dates = sum((list(b["dates"]) for b in batches), [])
    zones = sum((list(b["zones"]) for b in batches), [])
    if pred.shape != (len(dates), 96) or hourly.shape != (len(dates), 24): raise ValueError("prediction output must have shapes [N,96] and [N,24]")
    if not np.isfinite(pred).all() or not np.isfinite(hourly).all(): raise ValueError("prediction arrays must be finite")
    from src.constraints import write_predictions
    write_predictions(pred, hourly.astype(np.float64), dates, zones, path)
    return pd.read_csv(path)
