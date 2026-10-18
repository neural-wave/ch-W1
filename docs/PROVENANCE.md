# Data provenance

## Data used

The submitted model was developed, selected and trained only on the official participant package
(`train_prices_15min.csv`, 2025-10-01 to 2026-09-30). `artifacts_official/manifest.json` records
the training file's SHA-256, the training dates and the config hash. Inference uses only the jury's
`test_hourly.csv` and `history_15min.csv`, makes no network requests, and `aet/schema.py` rejects
October 2026 dates in training input.

## Each design choice is justified on official data

| Choice | Official-data EDA (analysis/reports) | Development folds, Jan–Aug (MAE, EUR/MWh) |
|---|---|---|
| Mean-preserving PCHIP baseline | Intra-hour slope tracks the hourly curve; slope rule removes ~40% of flat error (EDA §13) | Flat 8.1276 → PCHIP 5.6185 |
| Slope / curvature / zigzag basis | Hours tilt almost linearly; Q2/Q3 ≈ ⅓ of Q1/Q4 (EDA §10) | Haar basis 4.0024 vs poly 3.9838 |
| LightGBM (L1) with history features | Errors concentrate at ramps, turning points and day edges (EDA §13–14) | 3.9838 |
| Cross-zone features | Zones share a daily rhythm | Removing them: 4.0406 |
| Previous-day edge features | Largest unexplained error at the day edge, e.g. ZONE_3 00h (EDA §14) | Removing them after scaling: −0.16% |
| Residual scale 0.8 | — | 3.9479, better on 6/8 folds (I2, previous final) |
| Zone coupling, partial tie, sign floor/ceiling, 1000 rounds | Zones share a daily rhythm; quarters rarely change sign within an hour | 3.8883, better on 8/8 folds and all zones (I3, current final) |

Full results: [METHODOLOGY.md](METHODOLOGY.md) and [results/summary.csv](../results/summary.csv).
