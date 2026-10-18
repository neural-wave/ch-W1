"""Official-data development diagnostics and a separate, heuristic shape-risk indicator.

No synthetic data, external inputs, September assessment, or frozen-model modification.
`risk_score` is a relative input-magnitude index, NOT a probability or prediction interval.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import central_slope, get_baseline
from .basis import encode, get_basis
from .config import final_config, folds_from_config, load_config
from .cube import HourlyCube, QuarterCube
from .features import build_features
from .history import HistoryStore
from .pipeline import PipelineConfig, ShapePipeline
from .schema import HOURS, check_training_window, read_hourly, read_quarter_hourly

SIGNALS = ("hourly_slope", "daily_std", "historical_residual_magnitude")


def markdown_table(frame):
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda x: f"{x:.3f}")
    return "\n".join(["| " + " | ".join(map(str, shown.columns)) + " |",
                      "| " + " | ".join(["---"]*len(shown.columns)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |"
                        for row in shown.itertuples(index=False, name=None)]])


def risk_inputs(dates, H, store, config):
    """Only supplied hourly curves and strictly earlier residuals; no target quarters."""
    basis = get_basis(config.basis)
    base = get_baseline(config.baseline)(H)
    features = build_features(
        dates, H, encode(base - H[..., None], basis), store,
        replace(config.features, volatility_history=True))
    n = H.shape[0]
    magnitude = features[[f"mean_abs{w}_c{j}" for w in (7, 28) for j in (1, 2, 3)]].to_numpy()
    finite = np.isfinite(magnitude)
    magnitude = np.divide(np.where(finite, magnitude, 0).sum(axis=1), finite.sum(axis=1),
                          out=np.full(len(features), np.nan), where=finite.sum(axis=1) > 0)
    return pd.DataFrame({
        "delivery_date": np.repeat(np.asarray(dates, dtype="datetime64[D]"), len(config.zones)*HOURS),
        "market_id": np.tile(np.repeat(config.zones, HOURS), n),
        "hour_index": np.tile(np.arange(1, HOURS+1), n*len(config.zones)),
        "hourly_slope": np.abs(central_slope(H)).reshape(-1),
        "daily_std": np.broadcast_to(H.std(axis=-1, keepdims=True), H.shape).reshape(-1),
        "historical_residual_magnitude": magnitude,
        "historical_days_7": features.n_abs7.to_numpy(),
        "historical_days_28": features.n_abs28.to_numpy(),
    })


@dataclass
class ShapeRisk:
    """Frozen per-zone training references; no parameters are updated during inference."""
    reference: dict
    training_end: str
    config: dict

    @classmethod
    def fit(cls, training: QuarterCube, config: PipelineConfig):
        store = HistoryStore.build(training, config.baseline, get_basis(config.basis))
        inputs = risk_inputs(training.dates, training.hourly(), store, config)
        reference = {}
        for zone, part in inputs.groupby("market_id"):
            scales = {}
            for signal in SIGNALS:
                values = part[signal].dropna()
                # Floors in EUR/MWh avoid division by zero in constant-price histories.
                scales[signal] = max(float(values.quantile(.95)) if len(values) else 0., 1.)
            score = cls._score(part, scales)
            reference[zone] = {"scales": scales, "medium_threshold": float(np.quantile(score, .5)),
                               "high_threshold": float(np.quantile(score, .9))}
        return cls(reference, str(training.dates[-1]), config.to_dict())

    @staticmethod
    def _score(inputs, scales):
        ratios = inputs[list(SIGNALS)].to_numpy(float) / np.array([scales[s] for s in SIGNALS])
        finite = np.isfinite(ratios)
        return np.divide(np.where(finite, np.log1p(np.maximum(ratios, 0)), 0).sum(axis=1),
                         finite.sum(axis=1), out=np.zeros(len(inputs)), where=finite.sum(axis=1) > 0)

    def predict(self, inputs):
        parts = []
        for zone, part in inputs.groupby("market_id", sort=False):
            if zone not in self.reference:
                raise ValueError(f"No frozen risk reference for zone {zone}")
            part = part.copy()
            ref = self.reference[zone]
            part["risk_score"] = self._score(part, ref["scales"])
            # Strict >: identical constant inputs must not all be labelled high risk.
            part["risk_label"] = np.where(part.risk_score > ref["high_threshold"], "high",
                                           np.where(part.risk_score > ref["medium_threshold"], "medium", "low"))
            ratios = part[list(SIGNALS)].div(pd.Series(ref["scales"]))
            part["dominant_signal"] = ratios.idxmax(axis=1)
            part["limited_history"] = part.historical_days_7 < 3
            part["reference_training_end"] = self.training_end
            parts.append(part)
        return pd.concat(parts).sort_values(["delivery_date", "market_id", "hour_index"]).reset_index(drop=True)

    def save(self, path):
        Path(path).write_text(json.dumps({"reference": self.reference, "training_end": self.training_end,
                                         "config": self.config, "type": "heuristic_not_probability"}, indent=2)+"\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(data["reference"], data["training_end"], data["config"])


def summarise_regimes(hours):
    """Report both zone-specific results and equal-weight means of available zone summaries."""
    rows = []
    groups = [("all", np.ones(len(hours), bool))]
    for name in ("extreme_range", "extreme_pchip_correction"):
        groups += [(name, hours[name].to_numpy()), ("ordinary_"+name, ~hours[name].to_numpy())]
    groups += [("risk_"+label, hours.risk_label.eq(label).to_numpy()) for label in ("low", "medium", "high")]
    for regime, mask in groups:
        part = hours[mask]
        for zone, z in part.groupby("market_id"):
            for method in ("flat", "pchip", "model"):
                rows.append({"regime": regime, "market_id": zone, "method": method,
                             "hours": len(z), "dates": z.delivery_date.nunique(),
                             "mae": z[f"{method}_mae"].mean(),
                             "q1_bias": z[f"{method}_q1_bias"].mean(),
                             "q4_bias": z[f"{method}_q4_bias"].mean(),
                             "range_bias": z[f"{method}_range_bias"].mean(),
                             "early_1MWh_error_EUR": z[f"{method}_q1_abs"].mean(),
                             "late_1MWh_error_EUR": z[f"{method}_q4_abs"].mean()})
    return pd.DataFrame(rows)


def diagnose(train_path, config_path, output):
    raw = load_config(config_path)
    config = final_config(raw)
    folds, _ = folds_from_config(raw, "monthly")
    if any(f.score_end >= np.datetime64("2026-09-01") for f in folds):
        raise ValueError("This research diagnostic must not evaluate September or later")
    df = read_quarter_hourly(train_path)
    check_training_window(df)
    # Explicit cutoff before any target statistics, feature selection, or training.
    df = df[df.delivery_date < pd.Timestamp("2026-09-01")]
    cube = QuarterCube.from_long(df, config.zones)
    H = cube.hourly()
    frames, threshold_rows = [], []
    for fold in folds:
        train = cube.until(fold.train_end)
        pipe = ShapePipeline(config).fit(train)
        risk = ShapeRisk.fit(train, config)
        observed_range = np.ptp(train.values, axis=-1)
        corrections = np.abs(train.values - pipe.baseline(train.hourly())).mean(axis=-1)
        range_threshold = np.nanquantile(observed_range, .95, axis=(0, 2))
        correction_threshold = np.nanquantile(corrections, .95, axis=(0, 2))
        for z, zone in enumerate(config.zones):
            threshold_rows.append({"fold": fold.name, "market_id": zone, "training_end": str(fold.train_end),
                                   "range_p95": range_threshold[z], "pchip_correction_p95": correction_threshold[z]})
        for i in np.flatnonzero((cube.dates >= fold.score_start) & (cube.dates <= fold.score_end)):
            day, obs = cube.dates[i], cube.values[i]
            prediction, _ = pipe.predict_day(day, H[i], cube)
            base = pipe.baseline(H[i])
            store = HistoryStore.build(cube.before(day), config.baseline, pipe.basis)
            inputs = risk_inputs(np.array([day]), H[i:i+1], store, config)
            frame = risk.predict(inputs)
            frame["fold"] = fold.name
            actual_range = np.ptp(obs, axis=-1)
            correction_size = np.abs(obs-base).mean(axis=-1)
            frame["observed_range"] = actual_range.reshape(-1)
            frame["pchip_correction_magnitude"] = correction_size.reshape(-1)
            frame["extreme_range"] = (actual_range > range_threshold[:, None]).reshape(-1)
            frame["extreme_pchip_correction"] = (correction_size > correction_threshold[:, None]).reshape(-1)
            for name, pred in [("flat", np.repeat(H[i, ..., None], 4, axis=-1)), ("pchip", base), ("model", prediction)]:
                error = pred-obs
                frame[f"{name}_mae"] = np.abs(error).mean(axis=-1).reshape(-1)
                frame[f"{name}_q1_bias"] = error[..., 0].reshape(-1)
                frame[f"{name}_q4_bias"] = error[..., 3].reshape(-1)
                frame[f"{name}_q1_abs"] = np.abs(error[..., 0]).reshape(-1)
                frame[f"{name}_q4_abs"] = np.abs(error[..., 3]).reshape(-1)
                frame[f"{name}_range_bias"] = (np.ptp(pred, axis=-1)-actual_range).reshape(-1)
            frames.append(frame)
        print(f"Completed {fold.name}: training through {fold.train_end}", flush=True)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    hours = pd.concat(frames, ignore_index=True)
    hours.to_csv(out/"hourly_diagnostics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out/"training_thresholds.csv", index=False)
    summary = summarise_regimes(hours)
    summary.to_csv(out/"regime_summary.csv", index=False)
    last = folds[-1]
    # Demonstration artifact: last DEVELOPMENT fold only, not a submission-model change.
    risk.save(out/"risk_reference.json")
    risk_rows = summary[(summary.method == "model") & summary.regime.str.startswith("risk_")]
    display = risk_rows.pivot(index="market_id", columns="regime", values="mae")
    display.to_csv(out/"risk_validation.csv")
    ordered = (display.get("risk_high", np.nan) > display.get("risk_low", np.nan))
    report = f"""# Volatility and shape-risk diagnostics: development only

Uses the official supplied CSV only. All price analysis stops before September 2026.
The selected reconstruction recipe is replayed over eight monthly development folds,
with frozen weights per fold and strictly earlier quarter-hour history per date.
No existing model artifacts or final configuration are modified.

## What is measured

- Extreme shape range: observed Q1–Q4 maximum minus minimum exceeds its zone's training p95.
- Extreme PCHIP correction: mean absolute observed-minus-PCHIP correction exceeds training p95.
- These labels are retrospective diagnostics, never target-day model features.
- Mean absolute historical coefficients and observed historical ranges use exact 7-/28-calendar-day
  windows strictly before the target date, with counts and missing values rather than invented history.
- Regime MAEs are conditional per-hour averages; they are not the unconditional official score.
  Always inspect sample sizes and dates. Zone-specific thresholds can denote different absolute magnitudes.

## Separate risk indicator

The index averages log(1 + signal/training_p95) for absolute central hourly slope,
daily hourly-price standard deviation, and recent mean absolute residual coefficients.
Missing signals are omitted; fewer than three available prior days are flagged as limited history.
Each zone's training median and p90 of that index define low/medium/high groups.
References are fitted on each fold's training slice only, then frozen.
The index is NOT an estimated probability, calibrated confidence interval, predicted error magnitude,
or guarantee that an extreme event is observable from the hourly curve.

Across this replay, high-risk MAE exceeds low-risk MAE in {int(np.sum(ordered))} of {len(display)} zones.
See `risk_validation.csv` and counts in `regime_summary.csv`; this is retrospective development
evidence, not a new independent final assessment. Input-magnitude ranking may miss hidden quarter-hour spikes.

{markdown_table(display.reset_index())}

The values above are observed group MAEs in EUR/MWh, not the numerical risk scores themselves.

## Outputs and interpretation

- `hourly_diagnostics.csv`: risk score, reasons, retrospective extremes and all per-hour errors.
- `regime_summary.csv`: counts, MAE, Q1/Q4 signed biases, predicted-minus-observed range,
  and illustrative 1 MWh Q1/Q4 absolute valuation errors by regime, zone and method.
- Negative range bias means movements were underestimated. Signed biases averaged across all
  four quarters would be zero by construction, so Q1 and Q4 are reported separately.
- `training_thresholds.csv`: provenance of extreme thresholds for every fold.
- `risk_reference.json`: separate heuristic risk reference from training through {last.train_end}.
  It can annotate new hourly inputs; it does not modify price predictions or refit at inference.

## Optional model experiment

`configs/experiments/volatility.yaml` compares the incumbent with opt-in magnitude features on the original
development folds. Do not add `--include-locked` or select the extension on September.
The previous final model remains frozen; no synthetic augmentation is implemented.
"""
    (out/"report.md").write_text(report)
    print(display.to_string())
    print(f"Report: {out/'report.md'}")


def infer(hourly_path, history_path, reference_path, output):
    risk = ShapeRisk.load(reference_path)
    config = PipelineConfig.from_dict(risk.config)
    hourly = read_hourly(hourly_path)
    if not hourly.market_id.isin(config.zones).all():
        raise ValueError("Risk input contains zones not in the frozen reference")
    cube = HourlyCube.from_long(hourly, config.zones)
    if not np.isfinite(cube.values).all():
        raise ValueError("Risk indicator requires complete 24-hour curves for all configured zones")
    if (cube.dates <= np.datetime64(risk.training_end)).any():
        raise ValueError("Risk inference dates must follow reference training_end")
    history = QuarterCube.from_long(read_quarter_hourly(history_path, strict=False), config.zones)
    frames = []
    for i, day in enumerate(cube.dates):
        store = HistoryStore.build(history.before(day), config.baseline, get_basis(config.basis))
        frames.append(risk.predict(risk_inputs(np.array([day]), cube.values[i:i+1], store, config)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(f"Separate risk diagnostics: {output}; no prices modified, no fitting performed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("diagnose")
    dev.add_argument("--train", required=True)
    dev.add_argument("--config", default="configs/final.yaml")
    dev.add_argument("--out", default="results/official_volatility")
    pred = sub.add_parser("infer")
    pred.add_argument("--hourly", required=True)
    pred.add_argument("--history", required=True)
    pred.add_argument("--risk-model", required=True)
    pred.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "diagnose":
        diagnose(args.train, args.config, args.out)
    else:
        infer(args.hourly, args.history, args.risk_model, args.out)


if __name__ == "__main__":
    main()
