"""Report existing held-out predictions; never train, tune, or change model artifacts.

python -m tools.report_assessment --assessment results/september_assessment
Requires matplotlib (included in requirements.txt).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aet.baselines import cumulative
from aet.schema import HOUR_KEYS, QH_KEYS, clean_quarter_hourly, validate_predictions

NAMES = {"flat": "Hourly repetition", "pchip": "PCHIP", "final": "Final model"}
COLORS = {"flat": "#92999f", "pchip": "#dd953d", "final": "#2266a5"}


def table(frame):
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    return "\n".join([
        "| " + " | ".join(map(str, shown.columns)) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)],
    ])


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate(assessment, output, backtest=None):
    assessment, output = Path(assessment), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    truth = clean_quarter_hourly(pd.read_csv(assessment / "truth_15min.csv"))
    predictions = pd.read_csv(assessment / "predictions.csv", parse_dates=["delivery_date"])
    hours = truth.groupby(HOUR_KEYS, as_index=False).price_eur_mwh.mean().rename(
        columns={"price_eur_mwh": "hourly_price_eur_mwh"})
    checks = validate_predictions(predictions, hours)
    # No dropping, clipping or silently filling observations during reporting.
    counts = truth.groupby(["delivery_date", "market_id"]).size()
    if not counts.eq(96).all():
        raise ValueError("Reporting expects complete 96-quarter zone-days")
    if not truth.groupby("delivery_date").market_id.nunique().eq(4).all():
        raise ValueError("Reporting expects four zones for every date")
    data = truth.merge(predictions[QH_KEYS + ["predicted_price_eur_mwh"]],
                       on=QH_KEYS, validate="one_to_one").merge(hours, on=HOUR_KEYS, validate="many_to_one")
    data = data.sort_values(QH_KEYS).reset_index(drop=True)
    data["flat"] = data.hourly_price_eur_mwh
    data["final"] = data.predicted_price_eur_mwh
    ordered_hours = hours.sort_values(HOUR_KEYS).copy()
    curve = ordered_hours.hourly_price_eur_mwh.to_numpy().reshape(-1, 24)
    baseline = cumulative(curve, "pchip").reshape(-1)
    pchip_frame = data[QH_KEYS].copy()
    pchip_frame["pchip"] = baseline
    data = data.merge(pchip_frame, on=QH_KEYS, validate="one_to_one")
    records, valuations = [], []
    for method in NAMES:
        check_frame = data[QH_KEYS].copy()
        check_frame["predicted_price_eur_mwh"] = data[method]
        validate_predictions(check_frame, hours)
        data[f"error_{method}"] = (data[method] - data.price_eur_mwh).abs()
        daily = data.groupby(["delivery_date", "market_id"])[f"error_{method}"].mean()
        zone = daily.groupby("market_id").mean()
        records.append({"method": method, "MAE_EUR_per_MWh": zone.mean(), **zone.to_dict()})
        for quarter, label in [(1, "Early: 1 MWh in Q1"), (4, "Late: 1 MWh in Q4")]:
            subset = data[data.quarter_index.eq(quarter)]
            # One MWh in the selected quarter of each hour: numerical EUR error = price error.
            err = subset.groupby(["delivery_date", "market_id"])[f"error_{method}"].mean()
            valuations.append({"method": method, "profile": label,
                               "mean_absolute_hourly_valuation_error_EUR": err.groupby("market_id").mean().mean()})
    scores = pd.DataFrame(records)
    flat_score = scores.loc[scores.method.eq("flat"), "MAE_EUR_per_MWh"].iloc[0]
    scores["improvement_vs_flat_pct"] = 100 * (1 - scores.MAE_EUR_per_MWh / flat_score)
    scores.to_csv(output / "model_comparison.csv", index=False)
    valuation = pd.DataFrame(valuations)
    valuation.to_csv(output / "valuation_comparison.csv", index=False)
    daily = data.groupby(["delivery_date", "market_id"])[[f"error_{m}" for m in NAMES]].mean().reset_index()
    daily.to_csv(output / "daily_zone_errors.csv", index=False)
    quarter = data.groupby("quarter_index")[[f"error_{m}" for m in NAMES]].mean()
    quarter.to_csv(output / "quarter_errors.csv")
    data.to_csv(output / "assessment_details.csv", index=False, float_format="%.10f")
    by_date = daily.groupby("delivery_date").error_final.mean()
    representative = (by_date - by_date.median()).abs().idxmin()
    hardest = by_date.idxmax()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for label, date in [("representative", representative), ("hardest", hardest)]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for ax, zone in zip(axes.flat, sorted(data.market_id.unique())):
            day = data[data.delivery_date.eq(date) & data.market_id.eq(zone)]
            x = day.hour_index - 1 + (day.quarter_index - 1) / 4
            ax.plot(x, day.price_eur_mwh, color="#202020", lw=1.5, label="Observed")
            for method in NAMES:
                if method == "flat":
                    ax.step(x, day[method], where="post", color=COLORS[method], lw=1, label=NAMES[method])
                else:
                    ax.plot(x, day[method], color=COLORS[method], lw=1.1, alpha=.9, label=NAMES[method])
            ax.set(title=zone, xlabel="Hours since midnight", ylabel="EUR/MWh", xticks=[0, 6, 12, 18, 24])
            ax.legend(fontsize=8)
        fig.suptitle(f"{label.capitalize()} assessment day: {date:%Y-%m-%d}")
        save(fig, figures / f"{label}_reconstruction.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = list(NAMES)
    axes[0].bar([NAMES[m] for m in methods], scores.MAE_EUR_per_MWh,
                color=[COLORS[m] for m in methods])
    axes[0].set(title="Equal-zone MAE", ylabel="EUR/MWh", ylim=(0, flat_score*1.18))
    for i, value in enumerate(scores.MAE_EUR_per_MWh):
        axes[0].text(i, value+.12, f"{value:.3f}", ha="center")
    for i, method in enumerate(methods):
        row = scores[scores.method.eq(method)].iloc[0]
        axes[1].bar(np.arange(4)+(i-1)*.25, [row[z] for z in sorted(data.market_id.unique())],
                    width=.25, color=COLORS[method], label=NAMES[method])
    axes[1].set(xticks=range(4), xticklabels=sorted(data.market_id.unique()), ylabel="EUR/MWh", title="MAE by zone", ylim=(0, None))
    axes[1].legend()
    save(fig, figures / "accuracy_comparison.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    profiles = valuation.profile.unique()
    for i, method in enumerate(methods):
        values = valuation[valuation.method.eq(method)].mean_absolute_hourly_valuation_error_EUR.to_numpy()
        bars = ax.bar(np.arange(2)+(i-1)*.25, values, width=.25, color=COLORS[method], label=NAMES[method])
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(xticks=range(2), xticklabels=profiles, ylabel="Mean absolute valuation error (EUR per hour)",
           title="Illustrative 1 MWh delivery profiles; not actual AET positions", ylim=(0, valuation.mean_absolute_hourly_valuation_error_EUR.max()*1.25))
    ax.legend()
    save(fig, figures / "valuation_comparison.png")
    fig, ax = plt.subplots(figsize=(11, 4))
    date_means = daily.groupby("delivery_date")[[f"error_{m}" for m in methods]].mean()
    for method in methods:
        ax.plot(date_means.index, date_means[f"error_{method}"], marker="o", color=COLORS[method], label=NAMES[method])
    ax.set(title="Assessment error by date (equal zone weights)", ylabel="MAE (EUR/MWh)", xlabel="Delivery date", ylim=(0, None))
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    save(fig, figures / "daily_errors.png")
    comparison_note = "Backtest reconciliation was not requested."
    if backtest:
        previous = pd.read_csv(backtest, parse_dates=["delivery_date"])
        previous = previous[previous.candidate.eq("final")]
        if previous.empty:
            raise ValueError("No 'final' candidate in the supplied backtest file")
        check = daily.merge(previous[["delivery_date", "zone", "mae"]],
                           left_on=["delivery_date", "market_id"], right_on=["delivery_date", "zone"],
                           how="outer", validate="one_to_one", indicator=True)
        if not check._merge.eq("both").all():
            raise ValueError("Backtest and exported predictions cover different zone-days")
        check["export_minus_backtest_MAE"] = check.error_final - check.mae
        check.to_csv(output / "backtest_reconciliation.csv", index=False)
        old = check.groupby("market_id").mae.mean().mean()
        new = scores.loc[scores.method.eq("final"), "MAE_EUR_per_MWh"].iloc[0]
        comparison_note = (f"Backtest MAE: {old:.8f}; exported-inference MAE: {new:.8f}; "
                           f"difference: {new-old:+.8f} EUR/MWh. Maximum absolute zone-day MAE "
                           f"difference: {check.export_minus_backtest_MAE.abs().max():.8f}. "
                           "The source of a nonzero difference is not established by this report. "
                           "Do not describe the two paths as exactly identical. See the reconciliation CSV.")
        audit_path = assessment / "roundtrip_audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            comparison_note = comparison_note.replace(
                "The source of a nonzero difference is not established by this report. ", "")
            comparison_note += "\n\nAdditional diagnostic: " + audit["explanation"]
    final_row = scores[scores.method.eq("final")].iloc[0]
    start, end = data.delivery_date.min(), data.delivery_date.max()
    metadata = {"start": str(start.date()), "end": str(end.date()), "dates": int(data.delivery_date.nunique()),
                "rows": len(data), "representative_date": str(representative.date()), "hardest_date": str(hardest.date()),
                "output_validation": checks, "score_EUR_per_MWh": float(final_row.MAE_EUR_per_MWh),
                "improvement_vs_flat_pct": float(final_row.improvement_vs_flat_pct)}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2)+"\n")
    report = f"""# Official-data September assessment

## Result

On {start:%d %B}–{end:%d %B %Y} ({metadata['dates']} dates, four zones), the final exported
predictions achieve **{final_row.MAE_EUR_per_MWh:.4f} EUR/MWh MAE**, a
**{final_row.improvement_vs_flat_pct:.2f}% reduction** against hourly repetition.
The assessment model was trained through 31 August. The full-year submission model must not
be used to claim held-out September accuracy. These are existing assessment outputs, not a new test.

{table(scores)}

![Accuracy](figures/accuracy_comparison.png)
![Daily errors](figures/daily_errors.png)

## Reconstruction examples

The representative date ({representative:%Y-%m-%d}) is closest to the median final-model daily
MAE averaged equally across zones. The difficult example ({hardest:%Y-%m-%d}) has the highest
such daily MAE. Dates are selected automatically after assessment for illustration, not tuning.

![Representative reconstruction](figures/representative_reconstruction.png)
![Difficult reconstruction](figures/hardest_reconstruction.png)

## Illustrative valuation accuracy

Each profile delivers 1 MWh in the selected quarter of each hour, zero elsewhere.
Errors are calculated as absolute monetary errors per hour, then averaged over dates and zones.
They are numerical translations of Q1/Q4 price errors, not independent financial evidence,
actual AET exposures, trading profit or demonstrated savings. Do not interpret 1 MWh as 1 MW.

{table(valuation)}

![Valuation](figures/valuation_comparison.png)

## Output eligibility and reproducibility

Exported predictions pass schema, key, finiteness and hourly-mean validation against observed
hourly averages: **{checks['rows']} rows**, **{checks['hours']} hours**,
maximum hourly-mean error **{checks['max_abs_mean_error']:.3g} EUR/MWh**
(tolerance 0.0001). Flat and PCHIP comparison outputs also pass the same validation.

{comparison_note}

## Interpretation and limitations

- The final method is PCHIP plus a learned zero-sum correction. This report does not retrain,
  modify configuration, select hyperparameters or access external data.
- Scores use daily MAE, averaged over dates for each zone, then equally over zones.
- The evaluation covers 12 September dates, not the entire month or all future market conditions.
- September aggregate EDA was previously examined; this block was held out from the subsequent
  official-data model comparisons, rather than completely unseen in every respect.
- No calibrated confidence intervals are claimed. These figures show point reconstructions.

## Files

`assessment_details.csv` contains joined observed and reconstructed prices and absolute errors.
`model_comparison.csv`, `daily_zone_errors.csv`, `quarter_errors.csv`, and
`valuation_comparison.csv` provide the numerical evidence behind the figures.
"""
    (output / "report.md").write_text(report)
    print(scores.to_string(index=False))
    print("\n"+comparison_note)
    print(f"\nReport: {output / 'report.md'}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, default=Path("results/september_assessment"))
    parser.add_argument("--out", type=Path, default=Path("results/september_report"))
    parser.add_argument("--backtest", type=Path, help="Optional locked_per_day CSV to reconcile")
    args = parser.parse_args()
    generate(args.assessment, args.out, args.backtest)


if __name__ == "__main__":
    main()
