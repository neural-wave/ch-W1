# Dashboard (optional)

Interactive replay of the reconstruction on the September 2026 hold-out: hourly input, PCHIP
baseline, learned slope/curvature/zigzag corrections, the fixed post-processing rules (zone coupling,
partial tie, sign floor/ceiling), the final 15-minute curve and a heuristic
risk flag per hour. It is not part of the inference path.

## Required resources

- `data/train_prices_15min.csv` (see [data/README.md](../data/README.md)), or pass `-- --train <path>`
  or set `AET_TRAIN_CSV`.
- `artifacts_september_assessment/` (committed): the final configuration trained **through 2026-08-31
  only**, so September is genuinely held out. The submission model `artifacts_official/` (trained through
  2026-09-30) is *not* used here, because it has already seen September. To regenerate the folder from
  scratch (gives identical predictions):

```bash
python -c "import pandas as pd; d = pd.read_csv('data/train_prices_15min.csv'); \
d[d.delivery_date <= '2026-08-31'].to_csv('data/train_through_august.csv', index=False)"
python -m aet.train --train data/train_through_august.csv --config configs/final.yaml --out artifacts_september_assessment
```

## Launch

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py --server.address localhost
```

## Your own test days

To see your own test days instead of the September replay, follow
[View your own test results in the dashboard](../README.md#view-your-own-test-results-in-the-dashboard)
in the main README.

Nothing is trained, tuned or written to disk by the dashboard.
