"""Synthetic prices for SOFTWARE TESTS ONLY.

Never used for training, validation, calibration or hyperparameter selection of
the submitted model. It only checks invariants and wiring: a smooth daily curve
(demand peaks, a seasonal solar trough that can go negative) plus an amplified
intra-hour deviation that produces a sawtooth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aet.cube import QuarterCube

ZONES = ("ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3")


def synthetic_values(dates: pd.DatetimeIndex, n_zones: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(dates)
    t = (np.arange(96) + 0.5) / 4.0
    doy = dates.dayofyear.to_numpy()[:, None, None]
    weekend = (dates.dayofweek.to_numpy() >= 5)[:, None, None]
    zone = np.arange(n_zones)[None, :, None]
    season = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 172) / 365.25)            # 1 in June
    solar = np.clip(np.sin(np.pi * (t - 7.0) / 12.0), 0, None)[None, None, :]
    demand = (1 + 0.35 * np.exp(-((t - 8.5) ** 2) / 2) + 0.45 * np.exp(-((t - 20) ** 2) / 3))[None, None, :]
    level = 90 + 10 * zone + rng.normal(0, 8, size=(n, 1, 1)) - 12 * weekend
    x = level * demand - (60 + 25 * zone) * season * solar + rng.normal(0, 2, size=(n, n_zones, 96))
    q = x.reshape(n, n_zones, 24, 4)
    gamma = (0.5 + 0.25 * np.arange(n_zones))[None, :, None, None]            # zone-specific sawtooth
    return q + gamma * (q - q.mean(axis=-1, keepdims=True))


def synthetic_long(start: str, end: str, zones=ZONES, seed: int = 0, skip=()) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    dates = dates[~dates.strftime("%Y-%m-%d").isin(list(skip))]
    cube = QuarterCube(dates.to_numpy().astype("datetime64[D]"), tuple(zones),
                       synthetic_values(dates, len(zones), seed))
    df = cube.to_long()
    df["delivery_date"] = df["delivery_date"].dt.strftime("%Y-%m-%d")
    return df


def hourly_from(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["delivery_date", "market_id", "hour_index"], as_index=False)["price_eur_mwh"].mean()
              .rename(columns={"price_eur_mwh": "hourly_price_eur_mwh"}))
