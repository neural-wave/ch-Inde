# AET Neural Wave 2026 Challenge - Team Inde

This is Team Inde's competition delivery for reconstructing four 15-minute
prices from each supplied hourly price. The shipped recipe is a fixed 50/50
ensemble of:

- `A0-base/R3-CNN`: five same-zone, hour-local historical matches from the
  preceding 28 calendar days; their MAKIMA corrections are encoded by a CNN and
  injected into the A0 model.
- `B2-base/ORIGINAL`: a context-free, 24-token Transformer (width 64, two
  layers) with a signed reconstruction head.

Both components predict zero-mean within-hour residuals, so each four-quarter
mean equals the supplied hourly price. The A0 and B2 checkpoint binaries, the
frozen A0 gate, and the ensemble manifest are bundled in `checkpoints/`.

## Package contents

- `inference.py` runs the shipped 50/50 ensemble by default or B2 alone with
  `--model b2`.
- `train.py` reproduces the full-data refit without modifying the shipped
  checkpoints.
- `configs/` contains the resolved component configurations and the 363-date
  full-data training manifest.
- `splits/` contains the shipped 11 monthly-CV, 1 October-holdout, and 8
  chronological split manifests. `scripts/create_validation_splits.py`
  regenerates them.
- `src/` contains the B2 model and inference utilities; `src/context/` holds
  the retrieval/context model. `src/classical/` provides MAKIMA, while
  `src/gradient/` and `src/evaluation/` remain for split generation and refit
  evaluation.
- `data/train_prices_15min.csv` is the supplied historical package;
  `data/predictions/` is the default location for generated outputs.
- `images/`, `report/`, and `notebooks/dataprocess.ipynb` retain the submitted
  supporting material.

## Setup

Use Python 3.12 or newer and `uv`. Run the following commands from the
`delivery/` directory:

```sh
uv sync --locked
source .venv/bin/activate
```

The command uses the local `pyproject.toml` and `uv.lock`. It is the supported
environment for the ensemble inference script. This `uv` setup does not require
`pip install -r requirements.txt`; that command belongs only to the plain-Python
setup below.

### Plain Python environment

For an environment without `uv`, create a virtual environment in `delivery/`
and install the delivery inference dependencies:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains the runtime dependencies for inference and the
optional refit: NumPy, pandas, PyTorch, Lightning, SciPy, scikit-learn, PyYAML,
and jsonargparse. Keep the virtual environment active while running the commands
below. The plain-Python route is not lockfile-resolved; use `uv sync --locked`
for an exact project environment.

The verified model files are bundled under `checkpoints/`:

```text
checkpoints/
  a0-r3-cnn.ckpt
  b2-base.ckpt
  a0-r3-cnn-gate.json
  ensemble.json
```

`ensemble.json` records checkpoint SHA-256 values and the frozen 50/50 recipe.
`checkpoints/README.md` links the files to their training configurations.

## Run inference

The ensemble requires an hourly input CSV and may use the supplied historical
quarter-hour CSV. It never reads quarter-hour labels for a delivery date being
predicted.

```sh
python inference.py \
  --hourly-input data/hourly_input.csv \
  --history-input data/train_prices_15min.csv \
  --output data/predictions/ensemble.csv
```

`--history-input` is optional. When present, retrieval only considers
strictly earlier same-zone delivery dates; missing history uses the implemented
fallback. Use `--device cuda` to request CUDA explicitly; the default is CPU.

The hourly input must contain:

```text
delivery_date,market_id,hour_index,hourly_price_eur_mwh
```

It must provide all 24 hours for every zone and delivery date. The output has
`delivery_date,market_id,hour_index,quarter_index,predicted_price_eur_mwh` and
96 rows per date-zone. Inference validates the final serialized CSV: keys must
be complete and unique, values finite, and each four-quarter mean must match
the supplied hourly price within `1e-4 EUR/MWh`.

### B2-only inference

Add `--model b2` to run only the B2 component:

```sh
python inference.py \
  --model b2 \
  --hourly-input data/hourly_input.csv \
  --checkpoint checkpoints/b2-base.ckpt \
  --output data/predictions/b2-only.csv
```

This has the same input and output checks but does not use historical context.

## Selected model and recorded scores

The selected delivery recipe is `A0-R3-CNN + B2 Original`, with fixed 50/50
weights. HARVEST favored this recipe, whereas HORIZON favored a local A0 plus
global B2 retrieval choice. The final selection explicitly uses the HARVEST
seasonal-transfer result: **4.181 EUR/MWh** on October 2025, a **19.31%**
improvement over MAKIMA. Its HORIZON macro-month and pooled-date MAEs are
**4.064** and **4.072 EUR/MWh**, respectively.

| Method | HORIZON MAE | HARVEST MAE |
| --- | ---: | ---: |
| Flat hourly repeat | 8.228 | 8.944 |
| MAKIMA | 4.780 | 5.181 |
| LightGBM | 4.213 | 4.487 |
| XGBoost | 4.234 | 4.553 |
| GBM average | 4.200 | 4.508 |
| Selected `A0-R3-CNN + B2 Original` (50/50) | 4.064 macro-month / 4.072 pooled-date | 4.181 |

All values are EUR/MWh. HORIZON values are macro-month means unless the
pooled-date value is explicitly given. Scores use daily, equal-zone MAE: score
each date-zone over 96 quarters, average dates within each zone, then average
the four zones. MAKIMA is the fixed amplitude-2, explicitly extrapolated,
hour-centered reference.

## Evaluation protocols

The maintained protocol names are:

- **HORIZON:** expanding training windows over November 2025--September 2026;
  validate on the next complete month, from February through September. It is
  the temporal robustness and model-selection view.
- **HARVEST:** train on November 2025--September 2026 and evaluate October
  2025 once. It is a seasonal-transfer test; it never selects architecture,
  gate, weights, or checkpoint.

All four zones remain together in every date split. Learned statistics and gate
calibration fit training dates only. Sliding windows cannot cross a split
boundary, missing date, or DST gap. Exact assignments are in `splits/`.

To generate a new copy of the shipped protocol manifests, choose an empty
output directory:

```sh
python scripts/create_validation_splits.py \
  --data data/train_prices_15min.csv --output splits-new --no-enrich
```

The generator creates 20 folds: 11 monthly-CV selection folds with October
reserved, one October 2025 holdout, and eight expanding chronological folds.
It refuses to overwrite an existing output directory.

## Reproduce the full-data refit

The shipped checkpoints were trained from scratch at seed 42 for 30 epochs on
all 363 supplied dates, with no validation or test partition. The resulting
metrics are necessarily in-sample. `train.py` uses the resolved configurations
and `configs/full-data-split.json`; it writes new outputs under `trained/` and
does not replace `checkpoints/`.

```sh
python train.py --dry-run
python train.py
python train.py --accelerator cpu
```

Each component refit writes `checkpoints/last.ckpt`, diagnostics, and, for A0,
`gate.json`. The script prints each produced checkpoint hash alongside the
hash recorded in `checkpoints/ensemble.json`; exact bytes can depend on the
hardware and library stack.

## Model history

Era 2 established the validation and Transformer development path: a fixed
July validation / August--September test split, weekly CV, and fixed-epoch
checkpoint selection. Per-day normalization slightly outperformed global
normalization, while weekday conditioning and sliding-window training did not
help. Of six early architectures, the hourly four-output Transformer (`E2-B`)
led July validation at 4.0492 MAE, although MAKIMA still outperformed all six
neural models on the August--September test.

Teacher forcing improved masked models, and a flat gate with local features
improved the early neural variants. This work was consolidated into the shared
20-fold MOSAIC, HARVEST, and HORIZON protocol used for subsequent comparisons.

Later experiments evolved the Transformer into the A and B families, then
tested historical context. CNN encoding led the A-family context models; the
final retrieval recipe selects five same-zone, same-clock-hour observations
within 28 days (`R3`). Fixed 50/50 A/B ensembles beat their individual
components in the matched replication comparisons, leading to contextual
`A0-R3-CNN + B2 Original` as the frozen delivery model.
