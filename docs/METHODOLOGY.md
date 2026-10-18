# Official-data methodology and results

This note describes the configuration in configs/final.yaml and artifacts_official/manifest.json.

## 1. Data

Development, training, validation and model selection use only the official training file
(2025-10-01 to 2026-09-30). Inference uses only the jury's hourly prices and earlier 15-minute
history. See PROVENANCE.md.

## 2. Target and representation

For each supplied hourly value H, predict four quarter-hour prices whose mean is H.
Cumulative interpolation uses C[0]=0, C[h+1]=C[h]+H[h]. A PCHIP interpolator F through these
knots gives quarter q as 4*(F(h+(q+1)/4)-F(h+q/4)). The increments telescope, preserving H.

Observed-minus-PCHIP residuals are encoded in an orthonormal zero-sum basis:
slope [-3,-1,1,3]/sqrt(20), curvature [1,-1,-1,1]/2, zigzag [-1,3,-3,1]/sqrt(20).
Three pooled LightGBM regressors predict the coefficients. Each uses regression_l1,
1000 rounds (500 in the previous final I2), learning rate 0.05, 15 leaves, minimum leaf size 100
and seed 42. Coefficient L1 is a surrogate; model selection scores reconstructed prices.

Raw prices are PCHIP + 0.8*decoded_residual. The current final (I3) then applies three fixed,
mean-preserving post-processing rules per hour, in this order:
1. Zone coupling: zones whose hourly prices are within 0.1 EUR/MWh (connected components) take
   the mean of their intra-hour deviations, i.e. one shared quarter-hour shape.
2. Partial tie: zones within 5 EUR/MWh move 40% of the way toward their cluster's mean deviation.
3. Sign floor/ceiling: in hours with H > -0.01, quarters below -0.01 are raised to -0.01 by the
   Euclidean projection that keeps the hourly mean; mirrored, in hours with H < 0 quarters above
   0 are lowered to 0. Hours already satisfying the bound are unchanged.
Deviations are zero-sum, so rules 1-2 keep every hourly mean; rule 3 is a constrained projection.
A float64 mean projection follows. The rules have no fitted parameters. They use fixed basis vectors,
not learned external embeddings. The sign rule is a heuristic: in the official training data
218 positive-mean hours contain a quarter below -0.01 and 133 of 1354 negative-mean hours
contain a quarter above 0.

## 3. Current features and history

The current official model has **prev_day_edge: true**.
Features include zone, hour, weekday/weekend, seasonal sin/cos, local hourly levels,
backward/forward slopes and curvature, daily statistics, cross-zone spreads/slopes,
baseline coefficients, prior residual coefficients at calendar lags 1/7 and 7-/28-day
means with counts. The first hours may use the previous calendar day's known last hours.
The last hours never read next-day prices. Missing history stays missing (NaN).

All target-day hourly curves are supplied together. Quarter-hour lookups are strictly
earlier than the target date. Supplied history is merged over saved official training history,
then sliced before each prediction date. Weights and calibration are not updated at inference.
Optional magnitude features default off and are not in the shipped model.

## 4. Official-data validation

Eight rolling-origin development folds cover January–August 2026. Each model trains through
the preceding month-end and scores days 19 through month-end with earlier-history replay.
The predefined promotion gate requires at least 0.5% lower MAE, improvement on at least
two-thirds of folds, and no zone more than 2% worse. This is an engineering rule, not a
statistical significance guarantee; repeated development comparisons can overfit those folds.

Score = mean over zones of mean daily MAE. Each zone has 25% weight.

| Model | Development MAE | September backtest MAE |
|---|---:|---:|
| Flat | 8.1276 | 10.8988 |
| PCHIP | 5.6185 | 6.9270 |
| Unscaled coefficients + history | 3.9838 | 5.4270 |
| I2: scale 0.8, previous-day edge on (previous final) | 3.9479 | 5.0444 |
| **I3: I2 + coupling, partial tie, sign floor/ceiling, 1000 rounds (current final)** | **3.8883** | **5.0032** |

September assessment trained through 31 August and scored 19–30 September only (12 dates).
September aggregate EDA had already been inspected. The official-data ledger records a
locked scoring run of I2 after the core/ablation/memory/E6 comparisons. The I3 post-processing
rules were found by an automated search over the development folds on 2026-10-17 and evaluated
on top of I2 on 2026-10-18; September was then scored a second time before switching, with the
pre-stated rule to keep I2 unless I3 also beat 5.0444. September is therefore no longer an
untouched holdout for I3, and development gains are inflated by repeated selection on the same
folds. Do not restart model selection based on these results.

I3 versus I2 on development: 1.51% lower MAE, 8/8 folds, all four zones better, 89 of 98 days
better. On September: 0.82% lower MAE, 9 of 12 days better (paired day bootstrap 95% interval for
the gain -0.002..+0.090 EUR/MWh), ZONE_3 0.30% worse.

Final (I3) September backtest zone MAEs: ZONE_0 3.8262, ZONE_1 6.1266, ZONE_2 5.4917,
ZONE_3 4.5682 (I2: 3.8588, 6.2573, 5.5072, 4.5544). I3 reduces September MAE by 54.1% versus
flat and 27.8% versus PCHIP.

## 5. Official-data experiment decisions

- Scale 0.8 improved 0.90% over unscaled, 6/8 folds: promoted.
- Removing previous-day edge features after scaling worsened 0.16%: retained.
- Removing cross-zone features from the unscaled model worsened 1.43%: retained.
- Haar, Akima and spline substitutions did not improve the complete learned pipeline.
- Analogue memory K=8 improved 0.33%, only 5/8 folds: below gate, off.
- Graph smoothing lambda=0.1 improved 0.19%: below gate; lambda=0.3 worsened 0.75%.
- Quantile alternatives worsened 1.67–2.70%: off.
- Later magnitude-feature extension scored 3.9590 versus 3.9479: off.
- Risk diagnostics are separate development research, not part of the required prediction CSV.
- Post-processing on top of I2 (development, versus I2): zone coupling alone +0.32% (below gate);
  sign floor/ceiling +0.53%, 7/8 folds; partial tie + 1000 rounds +0.86%, 8/8; all three
  combined +1.51%, 8/8: promoted as I3.
- A forward-fitted residual scale scored +1.82% on development but selected exactly 0.8 on the
  full training set and gave identical September predictions: not adopted (fixed 0.8 kept).

configs/experiments/e6.yaml describes the official experiment with the edge on.

## 6. Backtest versus exported inference

The exported I3 September assessment predictions score 5.003526 EUR/MWh versus 5.003181 in
the backtest. The saved boosters and a fresh fit both reproduce 5.003181 with backtest inputs;
replacing only the hourly inputs by the supplied CSV values (NumPy versus CSV-input means differ
by at most 1.14e-13 EUR/MWh) reproduces 5.003526 exactly. Such differences can cross tree
thresholds and zone-tie tolerances. The maximum individual prediction difference is about
0.609 EUR/MWh. (I2: 5.044184 exported versus 5.044430 backtest, same mechanism.) Do not claim
bit-identical backtest and CSV paths. The official inference smoke run is repeatable for
identical files.

Representative/difficult plots and valuation results use exported predictions from a model
trained through August, never the full-year model. Illustrative 1 MWh Q1/Q4 positions have
mean absolute errors about EUR 5.88/6.15 (I2: 5.96/6.21) versus flat EUR 15.60/16.13. These are
numerical translations of quarter-specific errors, not actual AET positions, profits or savings.

## 7. Artifacts and reproducibility

artifacts_official/ contains the three I3 boosters, history_store.npz and a manifest recording
training dates 2025-10-01..2026-09-30, data/config hashes, versions and the post-processing
settings. The separate artifacts_september_assessment/ (committed; regenerate as in
dashboard/README.md) ends 2026-08-31. The I2 artifacts remain in Git history. Current source
defaults the optional volatility_history, geometry, couple_tol, partial_tie, forward_scale,
price_floor and price_ceiling settings to off, so I2 manifests load unchanged.

Use Python 3.12 and requirements.txt, then run python -m pytest -q.
See SUBMISSION.md for portable inference and retraining commands. The official dataset is
provided separately; no code fetches it during inference. Replication is conditional on runtime
versions and valid inputs, not a guarantee across all hardware or environments.

## 8. Robustness and limitations

Outputs require four finite prices per supplied hour, unique quarter keys, no extra hours
and mean error <=0.0001. Checks run before and after CSV writing. With valid hourly input,
model-loading/history/reconstruction failures produce flat predictions and an explicit diagnostic.
Malformed hourly input is rejected rather than silently inventing missing prices or choosing
among duplicates. Missing dependencies and filesystem failures can prevent output.

Rare events remain harder,
and 12 dates cannot establish universal generalisation. No calibrated uncertainty is claimed.
