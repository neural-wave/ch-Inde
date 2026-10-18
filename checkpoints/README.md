# Full-data checkpoints

Both checkpoints were downloaded from the completed Vast.ai refit and copied
byte for byte from the verified local model bundle. They were trained from
scratch at seed 42 for 30 epochs on all 363 official dates (1 October 2025 to
30 September 2026), including October. No validation checkpoint selection.

| File | Model | Ensemble weight |
| --- | --- | --- |
| `a0-r3-cnn.ckpt` | A0-base, R3 retrieval, CNN, additive history injection | 0.5 |
| `b2-base.ckpt` | B2-base Original, no historical context | 0.5 |
| `a0-r3-cnn-gate.json` | Frozen full-training calibration for A0 | Applied before blending |
| `ensemble.json` | Checkpoint hashes, provenance and frozen ensemble recipe | 50/50 |

Both checkpoint binaries, the frozen gate and ensemble manifest are versioned
in this delivery. Their corresponding resolved training configurations and
full-data date assignments are under `../configs/`.

From the delivery directory, predict with both models:

```sh
python inference.py \
  --hourly-input data/hourly_input.csv \
  --history-input data/train_prices_15min.csv \
  --output data/predictions/ensemble.csv
```

The supplied hourly grid contains all 24 hours and four zones. Historical
retrieval excludes current/future dates; final serialized predictions are
validated against the exact hourly means. Pass `--model b2` to run only the B2
component; the default is the frozen 50/50 ensemble.
