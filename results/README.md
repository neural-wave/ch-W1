# Results

[summary.csv](summary.csv) is the curated results table (MAE in EUR/MWh, official metric:
MAE per day and zone → mean over days per zone → mean over the four zones).

| Window | Definition |
|---|---|
| `dev_jan_aug_monthly_folds` | Eight monthly folds Jan–Aug 2026: train on everything before the month, score days 19–month-end. Used for model selection. |
| `sep_19_30_holdout` | Train through 2026-08-31, score 19–30 Sep 2026 with exported inference. Held out from the development comparisons; scored once for I2 and once more when switching from I2 to the current final (see [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) section 4). |

`final` is the current model (I3); `i2_previous_final` is the model it replaced. September figures
are exported-inference scores (I3 5.0035, I2 5.0442); the backtest path gives 5.0032 and 5.0444
because of tiny hourly-input rounding differences (see [docs/METHODOLOGY.md](../docs/METHODOLOGY.md)).
`coef_hist_unscaled` on September (5.4270) is the backtest-path score from the locked run
(`E3_coef_hist`), since no exported predictions exist for it.

## Reproduce

```bash
export TRAIN_CSV=data/train_prices_15min.csv
# Development folds (flat, PCHIP and model ladder are candidates in the config)
python -m aet.backtest --train "$TRAIN_CSV" --config configs/core.yaml --folds monthly --out results/development
python -m aet.backtest --train "$TRAIN_CSV" --config configs/final.yaml --folds monthly --out results/final
# Experiments that were not promoted
python -m aet.backtest --train "$TRAIN_CSV" --config configs/experiments/ablations.yaml --reference E3_coef_hist --out results/ablations
```

Everything under `results/` except this README and `summary.csv` is generated and gitignored.
