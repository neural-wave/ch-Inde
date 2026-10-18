"""LightningCLI entry point for reproducible daily transformer experiments."""
from __future__ import annotations

import logging
import json
import math
from pathlib import Path
from typing import Any

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from src.callbacks import RunArtifactsCallback, RunSettings
from src.datamodule import DailyPriceDataModule
from src.transformer import DailyPriceTransformer

log = logging.getLogger(__name__)


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_dict"): value = value.as_dict()
    elif hasattr(value, "__dict__"): value = vars(value)
    return dict(value) if isinstance(value, dict) else {}


def _validate_fit_selection(settings: RunSettings, cfg: dict[str, Any], trainer) -> None:
    """Validate that validation-based selection has a legal validation partition."""
    if settings.selection_metric not in ("final_epoch", "val/mae"):
        raise ValueError("run.selection_metric must be 'final_epoch' or 'val/mae'; test metrics cannot select checkpoints")
    patience = settings.early_stopping_patience
    delta = settings.early_stopping_min_delta
    if type(patience) is not int or patience < 1:
        raise ValueError("run.early_stopping_patience must be a positive integer")
    if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError("run.early_stopping_min_delta must be finite and nonnegative")
    if trainer.max_epochs is None or trainer.max_epochs <= 0:
        raise ValueError("training requires a positive trainer.max_epochs cap")
    if trainer.max_steps != -1 or _mapping(cfg.get("trainer", {})).get("max_time") is not None:
        raise ValueError("training requires max_steps=-1 and max_time=null")
    if settings.selection_metric == "val/mae":
        data = _mapping(cfg.get("data", {}))
        manifest = data.get("split_manifest")
        if manifest:
            path = Path(manifest)
            if not path.is_file():
                raise ValueError(f"run.selection_metric='val/mae' requires a readable validation manifest: {path}")
            try:
                dates = json.loads(path.read_text()).get("validation", [])
            except (json.JSONDecodeError, AttributeError) as exc:
                raise ValueError("run.selection_metric='val/mae' requires a valid validation manifest") from exc
            if not dates:
                raise ValueError("run.selection_metric='val/mae' requires nonempty validation dates")
        elif data.get("split_protocol", "static") == "cv":
            raise ValueError("run.selection_metric='val/mae' requires validation dates; CV has no validation partition")


class DailyTransformerCLI(LightningCLI):
    def _run_subcommand(self, subcommand: str):
        success = False
        try:
            super()._run_subcommand(subcommand)
            success = True
        finally:
            # Lightning's WandbLogger.finalize does not close the W&B run.
            # Close only an already initialized experiment; never start a run here.
            from lightning.pytorch.loggers import WandbLogger
            logger = getattr(getattr(self, "trainer", None), "logger", None)
            if isinstance(logger, WandbLogger) and logger._experiment is not None:
                logger._experiment.finish(exit_code=0 if success else 1)

    def add_arguments_to_parser(self, parser):
        parser.add_class_arguments(RunSettings, "run")

    def _prepare_run(self, stage: str):
        full_cfg = _mapping(self.config)
        cfg = _mapping(full_cfg.get(stage, full_cfg.get("fit", full_cfg)))
        run_values = _mapping(cfg.get("run", {}))
        settings = RunSettings(**{k: run_values[k] for k in RunSettings.__dataclass_fields__ if k in run_values})
        if stage == "fit":
            _validate_fit_selection(settings, cfg, self.trainer)
        if not settings.smoke_test:
            if not settings.entity or not settings.project:
                raise ValueError("configure run.entity and run.project explicitly for online W&B runs")
            from lightning.pytorch.loggers import WandbLogger
            # Stable identities are deliberate: W&B resume is permitted only for exact local identity matches.
            self.trainer.logger = WandbLogger(
                entity=settings.entity, project=settings.project,
                name=settings.run_id, id=f"{settings.experiment}--{settings.run_id}",
                group=settings.group or f"{settings.experiment}/{settings.run_id}", log_model=False,
                offline=False,
                save_dir=str(Path(settings.output_root) / settings.experiment / settings.run_id / "wandb"),
                resume="allow" if settings.resume else None,
            )
        elif self.trainer.logger not in (None, False):
            raise ValueError("run.smoke_test=true requires trainer.logger=false")
        root = Path(settings.output_root) / settings.experiment / settings.run_id
        if stage == "fit":
            ckpt_dir = root / "checkpoints"
            # Remove YAML-provided selection callbacks and install only the declared policy.
            self.trainer.callbacks = [cb for cb in self.trainer.callbacks
                                      if not isinstance(cb, (ModelCheckpoint, EarlyStopping))]
            if settings.selection_metric == "val/mae":
                self.trainer.callbacks.extend([
                    ModelCheckpoint(dirpath=str(ckpt_dir), filename="best", monitor="val/mae",
                                    mode="min", save_top_k=1, save_last=True,
                                    auto_insert_metric_name=False),
                    EarlyStopping(monitor="val/mae", mode="min", patience=settings.early_stopping_patience,
                                  min_delta=settings.early_stopping_min_delta, strict=True, check_finite=True),
                ])
            else:
                self.trainer.callbacks.append(ModelCheckpoint(
                    dirpath=str(ckpt_dir), filename="epoch={epoch:03d}", auto_insert_metric_name=False,
                    monitor=None, save_top_k=0, save_last=True,
                ))
        elif cfg.get("ckpt_path"):
            source_ckpt_dir = str(Path(cfg["ckpt_path"]).expanduser().resolve().parent)
            self.trainer.callbacks.append(ModelCheckpoint(
                dirpath=source_ckpt_dir, monitor=None, save_top_k=0, save_last=False))
        artifacts = RunArtifactsCallback(cfg, settings, stage=stage)
        self.trainer.callbacks.append(artifacts)
        if stage == "fit" and settings.resume:
            last = root / "checkpoints" / "last.ckpt"
            if not last.is_file(): raise FileNotFoundError(f"run.resume=true requires local checkpoint: {last}")
            # LightningCLI constructs trainer kwargs after before_fit; set the parsed
            # subcommand argument there so Trainer.fit receives the supported ckpt_path.
            if isinstance(self.config_init, dict):
                self.config_init.setdefault("fit", {})["ckpt_path"] = str(last)
            elif hasattr(self.config_init, "fit"):
                self.config_init.fit.ckpt_path = str(last)
        log.info("cli_ready stage=%s experiment=%s run_id=%s smoke_test=%s", stage, settings.experiment, settings.run_id, settings.smoke_test)

    def before_fit(self): self._prepare_run("fit")
    def before_validate(self): self._prepare_run("validate")
    def before_test(self): self._prepare_run("test")
    def before_predict(self): self._prepare_run("predict")


def main():
    DailyTransformerCLI(
        model_class=DailyPriceTransformer,
        datamodule_class=DailyPriceDataModule,
        save_config_callback=None,
        run=True,
        parser_kwargs={"default_config_files": []},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
