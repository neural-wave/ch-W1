# Data

The official participant dataset is committed here and is used for training, the EDA notebook and the
dashboard:

```
data/train_prices_15min.csv
```

- Source: AET Neural Wave 2026 participant package (ENTSO-E Transparency Platform, processed by AET).
  The original documentation is in [README_participants.txt](README_participants.txt).
- Schema: `delivery_date, market_id, hour_index, quarter_index, price_eur_mwh`
- Zones `ZONE_0`–`ZONE_3`, 2025-10-01 to 2026-09-30, 363 days per zone (DST days removed),
  96 quarter-hours per day, prices in EUR/MWh (can be negative).

Inference does **not** need this file: `artifacts_official/` already contains the trained model and
the history store. At inference the jury supplies `test_hourly.csv` and `history_15min.csv`
(see [docs/SUBMISSION.md](../docs/SUBMISSION.md)).

No other data source is used anywhere in this repository.
