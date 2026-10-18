# Feature engineering recommendations

Based on [EDA_final.ipynb](../notebooks/EDA_final.ipynb) and its write-up [EDA_analysis.md](EDA_analysis.md). Every feature is tied to the EDA finding that motivates it, and compared with what the AET pipeline already builds ([aet/features.py](../../aet/features.py)).

## Ground rules (apply to every feature)

- **Allowed inputs only:** the 24 hourly prices of the evaluated day (all zones, all hours) and 15-min history from **strictly earlier dates**. No load, weather, generation or external prices. The 15-min prices of the evaluated day are never used.
- **The next day is unknown.** Hours 23–24 cannot look forward. Hour 1 can look back, because the previous day is in history.
- **Same code in training and inference.** Features are computed by one function, and lags are counted in calendar days, so the DST gaps give NaN, not the wrong day.
- **Fixed rules, no refitting at inference.** Statistics taken from history (for example, rolling means) are feature calculations. Anything *fitted*, such as k or a model, is frozen at training time.
- **Robust to short history.** Every history feature must fall back to NaN or a training-time default when the jury's history is short.

## Target design (before features)

| EDA finding | Recommendation | Status in AET |
|---|---|---|
| The hourly level explains 93–96% of the variance; only the shape inside the hour is scored (§5) | Predict the **residual** relative to a mean-preserving baseline, then re-center | ✅ Done |
| Every hour tilts almost linearly; Q2/Q3 ≈ ⅓ of Q1/Q4 (§10) | Encode the 4 residuals as **slope / curvature / zigzag** coefficients (zero-sum basis). Slope carries most of the signal. | ✅ Done (`poly` basis) |
| The slope rule alone cuts the error by 40% (§13) | Use a smooth interpolation (PCHIP) as the **baseline**, and let the model learn the correction | ✅ Done (`pchip`) |
| Fat-tailed, Laplace-like residuals; rare synchronized extremes (§5, §7) | **L1 / MAE objective**, and shrink the predictions (scale < 1) | ✅ Done (L1, `residual_scale: 0.8`) |

---

## Tier 1: core features (keep, they carry most of the signal)

| Feature | Definition | Why (EDA) | Status |
|---|---|---|---|
| `slope_m1`, `slope_p1` | H(h) − H(h−1), H(h+1) − H(h) | Slope vs tilt r = 0.76–0.91 (§13) | ✅ |
| `slope_m2`, `slope_p2` | Same, one hour further out | The ramp shape over ±2 hours | ✅ |
| `curvature` | slope_p1 − slope_m1 | Peaks and valleys bend the hour (curvature coefficient) | ✅ |
| `level`, `relative_level` | H(h), H(h) − day mean | Where the hour sits in the day's curve | ✅ |
| `hour` | 0–23 | The direction of the tilt follows the time of day: drops at 07–12h and 20–03h, climbs at 04–06h and 13–19h (§10) | ✅ |
| `zone` (categorical) | ZONE_0–3 | The tilt size differs ~2× between zones (ZONE_1 22.7 vs ZONE_2 12.6) (§10) | ✅ |
| `day_range`, `day_std` | max − min and std of the 24 hourly prices | Movement inside the hour tracks the daily range (ρ = 0.68–0.84), much more than the level (§3) | ✅ |
| `prev_day_edge` | Hour 1 sees the previous day's hours 23–24 | The day edge has the largest unexplained error (ZONE_3 at 00h) (§14) | ✅ |
| `base_c1..3` | PCHIP baseline coefficients | Gives the model the interpolated shape it is correcting | ✅ |

## Tier 2: new features the EDA supports

These are not in the current pipeline. Each one targets a specific gap where the slope rule leaves error (§14).

### A. Turning points
The average tilt flips sign at **about 03h, 07h, 13h and 20h**, where the day's curve changes direction, and the direction there varies from day to day (§10).

| Feature | Definition |
|---|---|
| `is_turning_point` | `slope_m1 × slope_p1 < 0` (the curve changes direction at this hour) |
| `turn_sharpness` | \|curvature\| / (\|slope_m1\| + \|slope_p1\| + ε) |
| `is_local_max`, `is_local_min` | H(h) above (below) both neighbours |
| `hours_to_day_max`, `hours_to_day_min` | Signed distance in hours to the day's peak and to the solar valley |

Why: at a turn, the slope averages out to ≈0 and the slope rule predicts flat, while the real hour is often a peak or a U shape (curvature).

### B. Ramp intensity relative to the day
Movement scales with the day's range (§3), and the remaining error concentrates on the steepest ramps (06–09h, 15–19h) (§14).

| Feature | Definition |
|---|---|
| `slope_norm` | (H(h+1) − H(h−1)) / 2 / (day_range + ε) |
| `abs_slope` | \|H(h+1) − H(h−1)\| / 2 |
| `ramp_asymmetry` | \|slope_p1\| − \|slope_m1\| (is the ramp accelerating or ending?) |

Why: a 30 EUR step means a steep ramp on a calm winter day but a minor wiggle on a 300 EUR-range summer day. Tree models can't easily build ratios themselves.

### C. Solar valley and price floor
Negative prices are a spring/summer midday event (up to 30% of quarters at 13h), the floor (−500) hits several zones at once, and 0.5–4.4% of hours are completely flat, mostly in ZONE_3 (§6, §7, §8).

| Feature | Definition |
|---|---|
| `valley_depth` | Day mean − min of H over 10–16h |
| `near_zero_hours` | Count of hours with \|H\| < 5 in the day (extends the existing `day_negative_hours`) |
| `at_floor` | H(h) and both neighbours ≤ 0 (or \|H\| < 1) |
| `dist_to_zero` | H(h) itself, but capped, so the model sees "stuck at 0" clearly |

Why: when prices sit at 0 or at the floor, they can't drift further, so the shape is flat. The slope rule over-tilts these hours. `valley_depth` also works as a data-driven **solar-season flag** that works on forward curves without a calendar assumption.

### D. Regime: winter vs solar season
Winter moves ~40% less inside the hour, and the other three seasons are nearly identical (§8).

| Feature | Definition |
|---|---|
| `season_sin`, `season_cos` | Day-of-year cycle (✅ already in AET) |
| `solar_regime` | `valley_depth > threshold` (threshold fixed at training time), or simply month ∈ Mar–Sep |

Recommendation: prefer the **price-based** `valley_depth` over calendar features where possible. It generalizes to unusual days (a sunny winter day, a cloudy summer day) and to other years.

### E. Night vs day
The day moves ~2× more inside the hour than the night, in every month (§12).

| Feature | Definition |
|---|---|
| `is_night` | Hour in 22–05h |

This is cheap and mostly covered by `hour`. Add it only if it helps a linear or ridge model; trees learn it from `hour`.

### F. Upper end of the day (hours 23–24)
The next day is unknown, so hour 24 only has a one-sided slope.

| Feature | Definition |
|---|---|
| `hist_next_step_24` | Mean over the last 28 days (from history) of H(next day, hour 1) − H(hour 24), per zone |
| `hist_tilt_24` | Mean tilt at hour 24 over the last 28 days, per zone |

Why: this replaces the missing future neighbour with its typical value. It is a fixed rule over history, not refitting.

## Tier 3: history features (keep, extend carefully)

| Feature | Definition | Status |
|---|---|---|
| `lag1_c*`, `lag7_c*` | Residual coefficients for the same zone and hour, 1 and 7 days ago | ✅ |
| `mean7_c*`, `mean28_c*` | Rolling mean coefficients, same zone and hour | ✅ |
| `hist_k_zone_hour` | **New:** rolling ratio of tilt to slope over the last 28 days, per zone and hour, i.e. a "local k" | ⬜ |
| `mean_abs7/28`, `mean_range7/28` | Volatility from history | ⚠️ Exists as `volatility_history`, **not promoted** (didn't pass the gate) |

`hist_k_zone_hour` is motivated by the best k differing by zone (1.5–2.5) (§13) and probably by season. Letting it drift with recent history captures that without refitting.

## Tier 4: cross-zone (keep)

| Feature | Definition | Status |
|---|---|---|
| `z{k}_spread`, `z{k}_slope_m1/p1` | Other zones' hourly level and slopes on the same day | ✅ |

The four zones move almost in lockstep in the sawtooth (§11), and extremes hit several zones at once (§7). Another zone's steeper ramp is a strong hint for this zone.

## What not to add

- **Weekday-specific shapes beyond `weekday` / `weekend`.** The EDA shows that hour of day and season dominate, and weekday is already a feature.
- **Separate models per season.** Three of the four seasons behave the same. A regime feature is simpler.
- **Price-level bins.** Movement correlates weakly with the level (ρ 0.28–0.45) compared with the range.
- **Anything using the next day's prices,** even for hour 24. They are not available at inference.

## How to test them

1. Add **one group at a time** (A, B, C, D/E, F, `hist_k`) as a new candidate in `configs/`, starting from the frozen `final` config.
2. Run the monthly backtest folds (`python -m aet.backtest ... --folds monthly`) and compare against `final` using the existing **promotion gate**: ≥0.5% relative improvement, wins on ≥66% of folds, no zone worse by more than 2%.
3. Report MAE **per zone and per hour**. The groups are designed for specific gaps: A should help 03/07/13/20h, B the 06–09h and 15–19h ramps, C midday ZONE_3, and F hours 23–24. If the target hours don't improve, drop the group, even if the total moves slightly.
4. Keep the September block locked. Never tune on it.

**Suggested order:** B (ramp intensity) → A (turning points) → C (valley and floor) → F (day end) → `hist_k`. B and A target the hours with the most error left; C and F are smaller, more specific fixes.
