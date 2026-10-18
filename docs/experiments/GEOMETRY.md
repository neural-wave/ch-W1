# EDA-driven development experiment

> Run before the switch to the current final model (I3). "Incumbent" here is the previous final
> model I2 (3.9479); I3 scores 3.8883 on the same folds and is 19.53% better than slope ×2.
> See [METHODOLOGY.md](../METHODOLOGY.md) section 4.

The experiment uses official data and the existing eight January–August monthly development
folds, with the unchanged incumbent as reference. No September assessment was run and no
frozen model artifacts were overwritten. These are reused development folds, not a new
independent test set.

## Benchmarks and results

| Candidate | MAE (EUR/MWh) | Improvement over incumbent | Better folds |
|---|---:|---:|---:|
| Flat | 8.1276 | — | — |
| Linear ramp ×1 | 5.7580 | — | — |
| PCHIP | 5.6185 | — | — |
| Linear ramp ×2 | 4.8322 | — | — |
| Incumbent | 3.9479 | reference | — |
| Four local geometry features | 3.9497 | −0.0446% | 4/8 |
| Two extrema distances | 3.9500 | −0.0538% | 3/8 |

Promotion requires at least 0.5% lower MAE, at least six of eight folds better, and no zone
more than 2% worse. Neither feature family qualifies. Keep the incumbent. Slope ×2 is a
stronger simple benchmark: the incumbent reduces its MAE by 18.30%. Keep PCHIP in the
comparison because it is also the incumbent's reconstruction baseline.

## Definitions and edge handling

Let b = H[h] − H[h−1], f = H[h+1] − H[h], and R be the supplied daily hourly range.
The optional local_geometry flag adds:

- local_movement = (|b| + |f|)/2;
- turning_point = 1 when b*f < 0;
- ramp_asymmetry = (f−b)/(|b|+|f|+1e−6), a normalized signed curvature;
- norm_movement = (|b|+|f|)/max(R, 1 EUR/MWh).

The last denominator deliberately floors the daily range to prevent near-flat days from
producing extreme ratios. Missing neighboring hours remain missing, including the turning
flag; the existing previous-day edge policy applies and no following-day prices are used.

The separate extrema_distance flag adds absolute hour-index distances to the day's maximum
and minimum supplied hourly price. Ties use the first occurrence. Constant days have missing
distances because they have no distinct extrema. Both flags default to false, preserving the
frozen model's feature set.

The linear_x2 baseline doubles the existing linear ramp's within-hour deviations and
preserves each supplied hourly mean. Its scale is fixed, not fitted.

## Reproduce

From the repository root with the project environment activated:

```bash
python -m pytest -q
python -m aet.backtest \
  --train data/train_prices_15min.csv \
  --config configs/experiments/geometry.yaml \
  --folds monthly --reference incumbent \
  --out results/official_geometry
```

Recorded run: results/official_geometry/geometry_20261017_174341. Tests: 59 passed.
Do not add --include-locked for model selection.
