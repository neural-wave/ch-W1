# Official-data volatility extension

This is separate development research. It does not change `configs/final.yaml`, saved submission
artifacts, or the required prediction schema. No external inputs, pretrained weights, synthetic
training examples or September-based model selection are used.

Set `TRAIN_CSV` to the included official file before running these commands:

```bash
export TRAIN_CSV="data/train_prices_15min.csv"
```

## 1. Diagnose extreme hours and validate a simple risk index

```bash
python -m aet.volatility diagnose \
  --train "$TRAIN_CSV" \
  --config configs/final.yaml \
  --out results/official_volatility
```

Replays the frozen recipe over the eight Jan–Aug monthly development folds, days 19 onward.
Every fold fits zone-specific p95 extreme thresholds on its training slice, and uses observed
quarter-hour extremes only as retrospective labels. Reports counts, MAE, Q1/Q4 signed biases,
range underestimation and illustrative valuation errors for flat, PCHIP and the model.

The separate shape-risk index combines absolute hourly slope, daily hourly-price standard deviation
and recent absolute residual magnitudes, scaled by each zone's training p95. Training-score median
and p90 define relative low/medium/high labels. Missing historical signals are omitted, counts are
exported, and limited history is flagged. The score is a transparent heuristic, NOT a probability,
calibrated interval, volatility forecast or guarantee about hidden quarter-hour spikes. Validate
whether high-risk groups actually have larger held-out errors; inspect each zone and sample sizes.

Read `results/official_volatility/report.md`, `risk_validation.csv`, `regime_summary.csv` and
`training_thresholds.csv`. Conditional regime MAE does not replace the official unconditional score.

## 2. Test magnitude features separately

```bash
python -m aet.backtest \
  --train "$TRAIN_CSV" \
  --config configs/experiments/volatility.yaml \
  --folds monthly --reference incumbent \
  --out results/official_volatility_features
```

The `volatility_history` feature flag defaults to false. Opting in adds 7-/28-calendar-day mean
absolute residual coefficients (absolute value taken BEFORE averaging), mean historical observed
within-hour price ranges and counts. These are computed strictly before the target day. Signed
historical means remain available separately. No values are invented when history is missing:
LightGBM receives NaN and counts. No new standard-deviation forecast or distribution is claimed.

Do not add `--include-locked`. September was already examined and cannot validate a newly selected
extension independently. Compare on development folds using the same promotion rule, report that
these folds have been used repeatedly, and keep the incumbent if the evidence is weak. A favourable
development result alone does not justify an independent final-performance claim.

## 3. Export risk diagnostics without changing prices

```bash
python -m aet.volatility infer \
  --hourly results/inference_smoke_test/test_hourly.csv \
  --history results/inference_smoke_test/history_15min.csv \
  --risk-model results/official_volatility/risk_reference.json \
  --out results/risk_diagnostics.csv
```

The saved risk reference comes from the last development fold (training through July), as recorded
in the JSON. Inference requires later dates and complete hourly curves for the configured four
zones; it filters history strictly before each date, performs no fitting, and writes a separate
one-row-per-hour CSV. It does not overwrite or alter `predictions.csv`. For a later production
freeze, reference fitting and its provenance must be explicitly documented, not silently updated.

## Initial official-data development results

The opt-in magnitude-feature model scored 3.9590 EUR/MWh versus the incumbent's (I2) 3.9479,
0.28% worse, improving only 2/8 folds. It was not promoted. Leave `volatility_history: false`
in the final model; the diagnostic risk indicator is independent of that model choice.

The heuristic's low/medium/high groups have increasing observed average error in all four zones
(model errors from the current final model I3, replayed over the same folds; overall MAE 3.8883):

| Zone | Low-risk MAE | Medium-risk MAE | High-risk MAE |
|---|---:|---:|---:|
| ZONE_0 | 1.510 | 2.373 | 5.895 |
| ZONE_1 | 2.799 | 3.764 | 8.315 |
| ZONE_2 | 2.034 | 3.400 | 5.681 |
| ZONE_3 | 2.220 | 3.932 | 7.203 |

With the previous final model (I2) the same groups scored 1.524/2.387/5.926, 2.833/3.815/8.497,
2.042/3.446/5.754 and 2.297/4.036/7.322; the high/low ratios (2.8–3.9×) are unchanged.

These are development-period group averages in EUR/MWh, not probabilities or error guarantees.
The index and its 50th/90th-percentile categories were predefined for this experiment, not tuned
against these results. Repeated use of the development folds limits independent generalisation claims.

## Software verification

```bash
python -m pytest -q
```

Checks cover absolute-value cancellation, current/future target poisoning, calendar gaps, empty
history, default-feature compatibility, risk-reference persistence and magnitude-model constraint
and save/load behaviour. Synthetic observations appear in software tests only.
