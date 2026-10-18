"""Refit both delivered components on every supplied date — the shipped checkpoints.

`checkpoints/a0-r3-cnn.ckpt` and `checkpoints/b2-base.ckpt` were produced by the full-data refit:
seed 42, exactly 30 epochs, final epoch-30 checkpoint, all 363 official dates (1 October 2025 –
30 September 2026, October included), no validation or test partition, all metrics in-sample.
This script replays that refit from the delivery folder itself:

  * recipe      configs/a0-r3-cnn.yaml / configs/b2-base.yaml — byte-for-byte copies of the
                configurations saved by the completed fits;
  * dates       configs/full-data-split.json — every supplied date in `train`, no holdout;
  * code        src/neural/context/full_refit_cli.py — the same entry point the shipped runs
                used (FullDataCLI + FullRefitDataModule + full-data diagnostics and the A0 gate).

Documented overrides only: the split manifest and training CSV paths, a fresh run identity and
output root under `trained/`, offline execution (`run.smoke_test: true`: no W&B, no network) and
the accelerator. The shipped checkpoints are left untouched.

Usage:
  python train.py --dry-run              # write trained/configs/*.yaml and stop
  python train.py                        # refit both components (minutes on a GPU)
  python train.py --accelerator cpu      # CPU works, slower
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 12):  # fail before the scientific stack is imported
    raise SystemExit(
        "This package requires Python 3.12 or newer (this interpreter is "
        f"{sys.version.split()[0]}). On macOS use an explicit interpreter, e.g. "
        "'python3.13 -m venv .venv' — the system 'python3' is 3.9 and will not work."
    )

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

COMPONENTS = (
    {"name": "a0", "config": "configs/a0-r3-cnn.yaml", "checkpoint": "checkpoints/a0-r3-cnn.ckpt",
     "needs_gate": True, "delivered_gate": "checkpoints/a0-r3-cnn-gate.json"},
    {"name": "b2", "config": "configs/b2-base.yaml", "checkpoint": "checkpoints/b2-base.ckpt",
     "needs_gate": False, "delivered_gate": None},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None,
                        help="Official training CSV (default: data/train_prices_15min.csv, then ../data/)")
    parser.add_argument("--output", type=Path, default=HERE / "trained", help="Output root for the refit runs")
    parser.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu", "mps"))
    parser.add_argument("--dry-run", action="store_true", help="Write the resolved configs and stop")
    return parser.parse_args()


def resolve_data(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [HERE / "data/train_prices_15min.csv",
                                              HERE.parent / "data/train_prices_15min.csv"]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "official training CSV not found; put it at data/train_prices_15min.csv or pass --data"
    )


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_full_split(split_path: Path, data_path: Path) -> dict:
    import pandas as pd

    split = json.loads(split_path.read_text())
    if split.get("strategy") != "full_data_refit":
        raise ValueError(f"{split_path} is not a full_data_refit manifest")
    train = split.get("train")
    if not isinstance(train, list) or any(not isinstance(day, str) for day in train):
        raise ValueError("full-data training dates must be a list of date strings")
    if any(split.get(key) for key in ("validation", "test", "reserved", "unused")):
        raise ValueError("the full-data refit has no holdout or reserved dates")
    supplied = sorted(pd.read_csv(data_path).delivery_date.unique())
    if sorted(train) != supplied:
        raise ValueError("the manifest must list every supplied date exactly once")
    return split


def run_config(component: dict, args: argparse.Namespace, data_path: Path, split_path: Path) -> dict:
    config = yaml.safe_load((HERE / component["config"]).read_text())
    data = config.setdefault("data", {})
    data.update({"train_path": str(data_path), "predict_path": None, "split_manifest": str(split_path)})
    trainer = config.setdefault("trainer", {})
    trainer.update({"accelerator": args.accelerator, "devices": 1, "max_epochs": 30,
                    "logger": False, "limit_val_batches": 0.0, "num_sanity_val_steps": 0})
    run = config.setdefault("run", {})
    run.update({"experiment": "full-data-refit", "run_id": f"{component['name']}-s42-e30",
                "output_root": str(args.output), "selection_metric": "final_epoch",
                "smoke_test": True,  # offline: no W&B, no network
                "group": f"full-data-refit/{component['name']}-s42-e30"})
    run.pop("resume", None)
    return config


def main() -> None:
    args = parse_args()
    data_path = resolve_data(args.data)
    split_path = (HERE / "configs/full-data-split.json").resolve()
    split = check_full_split(split_path, data_path)
    print(f"data:  {data_path}  (sha256 {sha256(data_path)})")
    print(f"dates: {len(split['train'])} full-data training dates {split['train'][0]} .. {split['train'][-1]}")

    configs_dir = args.output / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for component in COMPONENTS:
        path = configs_dir / f"{component['name']}-s42-e30.yaml"
        path.write_text(yaml.safe_dump(run_config(component, args, data_path, split_path), sort_keys=False))
        written[component["name"]] = path
        print(f"wrote {path}")
    if args.dry_run:
        print("dry run: configs written, training not started")
        return

    recorded = {}
    ensemble_path = HERE / "checkpoints/ensemble.json"
    if ensemble_path.is_file():
        recorded = {entry["checkpoint"]: entry["checkpoint_sha256"]
                    for entry in json.loads(ensemble_path.read_text())["components"]}

    produced = {}
    for component in COMPONENTS:
        run_root = args.output / "full-data-refit" / f"{component['name']}-s42-e30"
        command = [sys.executable, "-m", "src.context.full_refit_cli", "fit", "--config",
                   str(written[component["name"]])]
        print("+", " ".join(command))
        subprocess.run(command, cwd=HERE, check=True, env=_env())
        checkpoint = run_root / "checkpoints" / "last.ckpt"
        if not checkpoint.is_file():
            raise RuntimeError(f"training did not produce {checkpoint}")
        gate = run_root / "gate.json"
        if component["needs_gate"] and not gate.is_file():
            raise RuntimeError(f"the A0 training-fit gate was not produced under {run_root}")
        digest = sha256(checkpoint)
        reference = recorded.get(Path(component["checkpoint"]).name)
        print(f"{component['name']}: {checkpoint} sha256 {digest}")
        if reference:
            print(f"{component['name']}: shipped {Path(component['checkpoint']).name} sha256 {reference}"
                  + ("  (identical)" if reference == digest else "  (different — expected: bit-exact"
                     " reproduction depends on the hardware and library stack)"))
        produced[component["name"]] = {"run_root": str(run_root), "checkpoint": str(checkpoint),
                                       "checkpoint_sha256": digest,
                                       "gate": str(gate) if gate.is_file() else None,
                                       "gate_sha256": sha256(gate)}

    provenance = {
        "protocol": {"epochs": 30, "seed": 42, "selection_metric": "final_epoch", "offline": True,
                     "split_manifest": str(split_path), "training_dates": len(split["train"]),
                     "interpretation": "Full-data deployment refit; every recorded metric is in-sample"},
        "data": {"path": str(data_path), "sha256": sha256(data_path)},
        "recipes": {component["name"]: {"source": component["config"],
                                        "resolved": str(written[component["name"]]),
                                        "sha256": sha256(written[component["name"]])}
                    for component in COMPONENTS},
        "produced": produced,
    }
    (args.output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    print("training complete; provenance written to trained/PROVENANCE.json")
    print("the shipped checkpoints in checkpoints/ were not modified")


def _env() -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(HERE) + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


if __name__ == "__main__":
    main()
