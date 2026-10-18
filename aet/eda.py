"""Descriptive analysis of the official training set (nothing is fitted here).

    python -m aet.eda --train data/train_prices_15min.csv --out results/eda

Writes four tables:
  baselines.csv    MAE of each parameter-free baseline by zone (E0/E1 in one shot)
  components.csv   share of residual energy in slope / curvature / zigzag
  shape_atlas.csv  mean intra-hour deviation by zone, hour and direction of the curve
  drift.csv        monthly amplitude per unit of hourly slope: is the sawtooth changing?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import BASELINES, central_slope, get_baseline
from .basis import COMPONENT_NAMES, encode, get_basis
from .cube import QuarterCube
from .schema import check_training_window, read_quarter_hourly

DIRECTION_THRESHOLD = 1.0   # EUR/MWh per hour; |slope| below this counts as "flat"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", default="results/eda")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = read_quarter_hourly(args.train, strict=True)
    check_training_window(df)
    zones = tuple(sorted(df["market_id"].unique()))
    cube = QuarterCube.from_long(df, zones)
    H, obs = cube.hourly(), cube.values
    valid = np.isfinite(obs).all(axis=(2, 3))                       # [date, zone]

    def zone_mean(day_zone: np.ndarray) -> np.ndarray:
        return np.array([day_zone[valid[:, z], z].mean() for z in range(len(zones))])

    # 1. parameter-free baselines
    baselines = pd.DataFrame(
        {name: zone_mean(np.abs(get_baseline(name)(H) - obs).mean(axis=(2, 3))) for name in BASELINES},
        index=zones,
    ).T
    baselines["official"] = baselines.mean(axis=1)
    baselines.to_csv(out / "baselines.csv")

    # 2. where the residual energy lives
    basis, names = get_basis("poly"), COMPONENT_NAMES["poly"]
    rows = []
    for ref in ("flat", "pchip"):
        energy = np.nanmean(encode(obs - get_baseline(ref)(H), basis) ** 2, axis=(0, 2))   # [zone, 3]
        for z, zone in enumerate(zones):
            rows.append({"reference": ref, "zone": zone,
                         **{f"{n}_share": energy[z, j] / energy[z].sum() for j, n in enumerate(names)},
                         "rms": float(np.sqrt(energy[z].sum()))})
    components = pd.DataFrame(rows)
    components.to_csv(out / "components.csv", index=False)

    # 3. shape atlas: mean deviation from the hourly price, by direction of the hourly curve
    slope = central_slope(H)                                        # [date, zone, hour]
    direction = np.where(slope > DIRECTION_THRESHOLD, "up", np.where(slope < -DIRECTION_THRESHOLD, "down", "flat"))
    dev = obs - H[..., None]
    d, z, h = np.meshgrid(np.arange(len(cube.dates)), np.arange(len(zones)), np.arange(24), indexing="ij")
    atlas = pd.DataFrame({"zone": np.asarray(zones)[z.ravel()], "hour": h.ravel() + 1,
                          "direction": direction.ravel(), **{f"q{q + 1}": dev[..., q].ravel() for q in range(4)}})
    atlas = atlas.dropna().groupby(["zone", "hour", "direction"]).agg(
        n=("q1", "size"), **{f"q{q}": (f"q{q}", "mean") for q in range(1, 5)}).reset_index()
    atlas.to_csv(out / "shape_atlas.csv", index=False)

    # 4. drift: amplitude of the residual over PCHIP per unit of hourly slope, by month
    month = pd.DatetimeIndex(cube.dates.astype("datetime64[ns]")).to_period("M").astype(str)
    resid = np.abs(obs - get_baseline("pchip")(H)).mean(axis=(2, 3))   # [date, zone]
    flat_mae = np.abs(dev).mean(axis=(2, 3))
    mean_slope = np.abs(slope).mean(axis=2)
    rows = []
    for z, zone in enumerate(zones):
        frame = pd.DataFrame({"month": month, "flat_mae": flat_mae[:, z], "pchip_mae": resid[:, z],
                              "abs_slope": mean_slope[:, z]})[valid[:, z]]
        g = frame.groupby("month").mean()
        g["zone"], g["amplitude_per_slope"] = zone, g["pchip_mae"] / g["abs_slope"]
        rows.append(g.reset_index())
    drift = pd.concat(rows, ignore_index=True)
    drift.to_csv(out / "drift.csv", index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nMAE of parameter-free baselines (EUR/MWh)\n", baselines.to_string())
        print("\nResidual energy by component\n", components.to_string(index=False))
        print("\nPCHIP residual per unit of hourly slope, by month (a trend here = drift)\n",
              drift.pivot(index="month", columns="zone", values="amplitude_per_slope").to_string())
    print(f"\ntables in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
