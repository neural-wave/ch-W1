# AET Neural Wave: official-data reconstruction

Reconstruct 96 quarter-hour prices from 24 supplied hourly prices per zone, preserving every
hourly average within 0.0001 EUR/MWh. This is constrained reconstruction, not hourly forecasting.

The only data source is the official participant package, already included in data/ (see [data/README.md](data/README.md)).

## Team ApplePi

- Cristina Fernández-Simal Bernard
- Paola Hernandez Llamas
- Andrea Mero Plaza
- Laura Jimena Tagle Muñoz

## Current model and evidence

The current configuration (I3) and supplied `artifacts_official/` use:
- cumulative PCHIP interpolation;
- three L1 LightGBM regressors (1000 rounds) for zero-sum slope, curvature and zigzag residual coefficients;
- residual scale 0.8;
- historical and cross-zone features; **prev_day_edge: true**;
- fixed mean-preserving post-processing: zones within 0.1 EUR/MWh share one quarter-hour shape,
  zones within 5 EUR/MWh move 40% toward their cluster shape, and quarters keep the sign of their
  hour (floor -0.01 for positive hours, ceiling 0 for negative hours);
- no analogue memory, graph smoothing, quantile decision or magnitude-feature extension.

| Method | Official-data development MAE | September backtest MAE |
|---|---:|---:|
| Flat | 8.1276 | 10.8988 |
| Linear ramp ×1 | 5.7580 | — |
| PCHIP | 5.6185 | 6.9270 |
| Linear ramp ×2 | 4.8322 | — |
| Unscaled coefficient model with history | 3.9838 | 5.4270 |
| I2, previous final (no post-processing, 500 rounds) | 3.9479 | 5.0444 |
| **I3, current final** | **3.8883** | **5.0032** |

Development: eight Jan–Aug monthly folds, days 19–month-end. 
September: train through August, assess 19–30 September. September aggregate EDA had already been inspected,
and September was scored a second time when switching from I2 to I3, so it is not an untouched
holdout for I3 (docs/METHODOLOGY.md section 4). Exported inference scores 5.0035 rather than
5.0032 due to tiny hourly-input rounding differences affecting tree branches and tie tolerances;
both paths improve over flat by approximately 54.1%.

On these development folds, PCHIP improves over flat by 30.87%, slope ×2 by 40.55%,
and the final model by 52.16%. Final improves over slope ×2 by 19.53%.
The full-year descriptive EDA uses a different sample (PCHIP approximately 29%);
do not mix its percentages with this table. Dashes mean no assessment reported here.
New local-geometry and extrema-distance features did not pass the development promotion
gate and remain disabled; see [experiment details](docs/experiments/GEOMETRY.md).


## Install and test

Use Python 3.12, matching the recorded artifact environment. If using uv:

```bash
uv python install 3.12
uv venv --python 3.12 .venv312
source .venv312/bin/activate
uv pip install -r requirements.txt
python -m pytest -q
```

Dependencies must already be installed for offline execution.

## Run the supplied model

From this repo root, with organiser-supplied input files:

```bash
python -m aet.infer --hourly test_hourly.csv --history history_15min.csv --model artifacts_official --out predictions.csv
```

No training is needed. All required model files are in artifacts_official/. See
[portable execution instructions](docs/SUBMISSION.md).

## Retrain or reproduce development

The official participant dataset is already included in data/ (see [data/README.md](data/README.md)).

```bash
export TRAIN_CSV="data/train_prices_15min.csv"
python -m aet.train --train "$TRAIN_CSV" --config configs/final.yaml --out artifacts_retrained
python -m aet.backtest --train "$TRAIN_CSV" --config configs/final.yaml --folds monthly --out results/reproduced_development
```

Training writes to artifacts_retrained and does not overwrite artifacts_official.
Reopening September is for audit/reproduction only, never another tuning round.
The inference output uses the official schema and is checked before writing and after reloading.

## Safeguards and limits

The mean-preserving baseline plus zero-sum correction enforces consistency algebraically.
A final float64 projection removes numerical drift. Calendar-based history excludes the
target date and all future dates. Model weights are never refitted at inference.

With valid hourly input and installed dependencies, artifact/history/prediction failures fall
back to hourly repetition and emit an emergency diagnostic. Invalid hourly references cannot
be repaired safely; malformed inputs, missing runtime dependencies and write failures remain
errors. Do not claim the pipeline always writes a compliant file under every failure.

## Optional analysis

- EDA: see [analysis/README.md](analysis/README.md).
- Report figures: `python -m tools.report_assessment --assessment results/september_assessment --out results/september_report`.
- Volatility: [docs/experiments/VOLATILITY.md](docs/experiments/VOLATILITY.md). The risk indicator is a separate
  heuristic, not calibrated confidence. Added magnitude features were not promoted.
- Dashboard: `python -m streamlit run dashboard/app.py --server.address localhost`; replays held-out
  September with `artifacts_september_assessment/` (see [dashboard/README.md](dashboard/README.md)). Not part of the inference path.
  To see your own test days instead, see [below](#view-your-own-test-results-in-the-dashboard).
- Tests: `python -m pytest -q`.

## View your own test results in the dashboard

You tested the model on your own days and want to see the results as charts. It takes two steps.

**What you need:** three CSV files, all in the repository folder (the folder this README is in).

| File | What it contains |
|---|---|
| `test_hourly.csv` | The 24 hourly prices of each day you want to test. |
| `history_15min.csv` | The real 15-minute prices of the days *before* your test days. |
| `truth_15min.csv` | The real 15-minute prices of your test days, so the dashboard can tell how close the model came. Same columns as the training file: `delivery_date, market_id, hour_index, quarter_index, price_eur_mwh`. Every test day must be complete: 96 quarter-hours for each of the 4 zones. |

**Step 1: let the model make its predictions.** Run this in the terminal, from the repository folder:

```bash
python -m aet.infer --hourly test_hourly.csv --history history_15min.csv --model artifacts_official --out predictions.csv
```

This creates a new file, `predictions.csv`, with the model's 15-minute prices.

**Step 2: open the dashboard with your files.**

```bash
python -m streamlit run dashboard/app.py --server.address localhost -- \
  --hourly test_hourly.csv --history history_15min.csv --predictions predictions.csv --truth truth_15min.csv
```

The dashboard opens in your browser and shows only your test days. You can see the model's error compared with
simply repeating the hourly price four times (the flat baseline), how each prediction was built, and the hours
flagged as risky.

**What the parts of the step 2 command mean**

- The lone `--` in the middle: everything after it is for our dashboard, everything before it is for Streamlit,
  the program that runs the dashboard. Don't leave it out.
- The `\` at the end of the first line only means "the command continues on the next line".
- After the `--`, each option is followed by the file you want to use:

| Option | Must I give it? | What it means |
|---|---|---|
| `--truth` | Yes | Your file of real 15-minute prices for the test days. The dashboard shows the days in this file. Giving this option is what makes the dashboard show your days instead of our September example. |
| `--history` | Yes | The same history file you used in step 1. The model looks at recent days to make its predictions. |
| `--predictions` | No | The `predictions.csv` file made in step 1. If you leave it out, the dashboard makes the predictions again itself. |
| `--hourly` | No | The same hourly file you used in step 1. If you leave it out, the dashboard averages the real prices in `--truth` instead. Because of rounding, the predictions can then differ very slightly. |
| `--model` | No | The model to use. Leave it out to use our submitted model (`artifacts_official/`). |
| `--train` | No | Only needed if you moved `train_prices_15min.csv` out of the `data/` folder: give its new location. |

**Tips**

- If your files are in another folder, write the full path, for example
  `--truth /Users/me/Downloads/truth_15min.csv`.
- To stop the dashboard, press `Ctrl+C` in the terminal.
- The dashboard only reads your files. It does not change them, and it does not retrain the model.

## Repository layout

| Path | Content |
|---|---|
| `aet/` | Model package: training (`aet.train`), inference (`aet.infer`), backtest, features, history |
| `configs/` | `core.yaml` (folds, candidates), `final.yaml` (frozen submission), `experiments/` (not promoted) |
| `artifacts_official/` | Trained model used for submission (full training year) |
| `artifacts_september_assessment/` | Same recipe trained through 2026-08-31; used only by the dashboard |
| `data/` | Official training CSV (`train_prices_15min.csv`) and its documentation |
| `analysis/` | EDA notebook, reports and figures |
| `dashboard/` | Optional Streamlit dashboard |
| `docs/` | Methodology, submission contract, provenance, AI usage, references, experiment notes |
| `tests/` | Unit and end-to-end tests (synthetic data, no network) |
| `tools/` | Official-metric scorer and report figures |
| `results/` | Curated results table and reproduction commands |
| `report/`, `video/` | Deliverables: 3-page report and video link |
