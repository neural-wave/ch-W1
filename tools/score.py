"""Score predictions against the hidden quarter-hours with the official metric.

    python -m tools.score --pred predictions.csv --truth truth_15min.csv

Eligibility is checked first (as the organisers do), then: daily MAE per zone ->
mean over days per zone -> mean over zones, next to the flat baseline.
"""
from __future__ import annotations

import argparse

import pandas as pd

from aet.schema import HOUR_KEYS, QH_KEYS, clean_quarter_hourly, to_timestamps, validate_predictions


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    truth = clean_quarter_hourly(truth, "truth")
    hourly = (truth.groupby(HOUR_KEYS, as_index=False)["price_eur_mwh"].mean()
                   .rename(columns={"price_eur_mwh": "hourly_price_eur_mwh"}))
    validate_predictions(pred, hourly)
    pred = pred.assign(delivery_date=to_timestamps(pred["delivery_date"]).to_numpy(),
                       market_id=pred["market_id"].astype(str).str.strip())
    merged = truth.merge(pred, on=QH_KEYS, how="left")
    merged["model"] = (merged["predicted_price_eur_mwh"] - merged["price_eur_mwh"]).abs()
    merged["flat"] = (merged.groupby(HOUR_KEYS)["price_eur_mwh"].transform("mean") - merged["price_eur_mwh"]).abs()
    by_zone = merged.groupby(["delivery_date", "market_id"])[["model", "flat"]].mean().groupby("market_id").mean()
    by_zone["gain_pct"] = 100 * (1 - by_zone["model"] / by_zone["flat"])
    by_quarter = merged.groupby("quarter_index")[["model", "flat"]].mean()
    overall = by_zone[["model", "flat"]].mean()
    return by_zone, by_quarter, {"score": float(overall["model"]), "flat": float(overall["flat"]),
                                 "gain_pct": float(100 * (1 - overall["model"] / overall["flat"]))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--truth", required=True)
    args = parser.parse_args(argv)
    by_zone, by_quarter, overall = score(pd.read_csv(args.pred), pd.read_csv(args.truth))
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(by_zone.to_string())
        print(by_quarter.to_string())
    print(f"\nscore {overall['score']:.4f} EUR/MWh | flat {overall['flat']:.4f} | gain {overall['gain_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
