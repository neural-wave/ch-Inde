# Checkpoint training configurations

| Checkpoint | Resolved training configuration |
| --- | --- |
| `../checkpoints/a0-r3-cnn.ckpt` | `a0-r3-cnn.yaml` |
| `../checkpoints/b2-base.ckpt` | `b2-base.yaml` |

These YAML files are byte-for-byte copies of the configurations saved by the
completed full-data fits: seed 42, exactly 30 epochs, final checkpoint selection.
`full-data-split.json` records all 363 training dates and empty evaluation
partitions. The supplied split manifest overrides the YAML's legacy static
date-range defaults; those defaults were not used to partition the full refit.

The original absolute server path in `data.split_manifest` is retained as
provenance. To replay from the project root, override that field with
`delivery/configs/full-data-split.json` using
`python -m src.context.full_refit_cli fit --config ...` and use a fresh
run identity and output location. The model checkpoints already embed their
model hyperparameters; inference does not rerun these training configurations.

`../checkpoints/ensemble.json` links each checkpoint to its config and SHA-256
hash. Its fixed weights are 0.5 for each component. A0's frozen output gate is
stored separately in `../checkpoints/a0-r3-cnn-gate.json`.
