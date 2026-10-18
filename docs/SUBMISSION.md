# Running the official-data reconstruction artifacts

The supplied model (I3) uses PCHIP + 0.8 times three zero-sum LightGBM coefficient corrections,
with history, cross-zone and previous-day boundary features enabled, followed by fixed
mean-preserving zone-coupling, partial-tie and sign floor/ceiling rules.

## Environment

Use Python 3.12. Create an environment and install `requirements.txt` before moving offline;
the dependency file alone is not an offline wheel cache. Inference makes no network requests.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Predict using the supplied artifacts (no training required)

From the extracted bundle root, with the jury's inputs present:

```bash
python -m aet.infer --hourly test_hourly.csv --history history_15min.csv --model artifacts_official --out predictions.csv
```

The output is validated before writing and after reload. Valid hourly input is required.
An artifact-loading, history-processing or reconstruction failure produces the flat baseline
with `emergency_flat: true` in the JSON diagnostic. Missing dependencies, invalid hourly inputs,
and filesystem failures cannot be guaranteed to produce a compliant file and are reported as errors.

## Regenerate artifacts from the organisers' official package

The official dataset is committed at data/train_prices_15min.csv; set TRAIN_CSV to use another copy:

```bash
export TRAIN_CSV="data/train_prices_15min.csv"
python -m aet.train --train "$TRAIN_CSV" --config configs/final.yaml --out artifacts_retrained
python -m aet.infer --hourly test_hourly.csv --history history_15min.csv --model artifacts_retrained --out predictions.csv
```

This uses the same frozen recipe and leaves the supplied artifacts unchanged. Numeric reproducibility
requires the documented dependency versions; the existing model's manifest records its environment.
Do not perform model selection on evaluation-period observations or retrain during jury inference.
