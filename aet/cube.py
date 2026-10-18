"""Dense array views of the long tables, indexed [date, zone, hour(, quarter)].

NaN marks missing data, so DST-excluded days and absent zones need no special
casing downstream: they simply propagate as missing values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import HOURS, QUARTERS


def as_days(values) -> np.ndarray:
    return np.asarray(values).astype("datetime64[D]")


def _index(df: pd.DataFrame, zones: tuple):
    days = as_days(df["delivery_date"].to_numpy())
    dates = np.unique(days)
    zone_pos = df["market_id"].map({z: i for i, z in enumerate(zones)}).to_numpy(dtype=np.int64)
    return dates, np.searchsorted(dates, days), zone_pos


@dataclass(frozen=True)
class QuarterCube:
    """Quarter-hour prices as float64 [date, zone, hour, quarter]."""

    dates: np.ndarray
    zones: tuple
    values: np.ndarray

    def __post_init__(self):
        if self.values.shape != (len(self.dates), len(self.zones), HOURS, QUARTERS):
            raise ValueError(f"cube shape {self.values.shape} does not match dates/zones")
        if len(self.dates) > 1 and not (np.diff(self.dates) > np.timedelta64(0, "D")).all():
            raise ValueError("cube dates must be strictly increasing")

    @classmethod
    def empty(cls, zones) -> "QuarterCube":
        zones = tuple(zones)
        return cls(np.array([], dtype="datetime64[D]"), zones, np.empty((0, len(zones), HOURS, QUARTERS)))

    @classmethod
    def from_long(cls, df: pd.DataFrame, zones, value_col: str = "price_eur_mwh") -> "QuarterCube":
        zones = tuple(zones)
        df = df[df["market_id"].isin(zones)]
        if df.empty:
            return cls.empty(zones)
        dates, d, z = _index(df, zones)
        values = np.full((len(dates), len(zones), HOURS, QUARTERS), np.nan)
        h = df["hour_index"].to_numpy() - 1
        q = df["quarter_index"].to_numpy() - 1
        values[d, z, h, q] = df[value_col].to_numpy(dtype=float)
        return cls(dates, zones, values)

    def hourly(self) -> np.ndarray:
        """[date, zone, hour] hourly means (NaN if any quarter is missing)."""
        return self.values.mean(axis=-1)

    def _select(self, mask) -> "QuarterCube":
        return QuarterCube(self.dates[mask], self.zones, self.values[mask])

    def before(self, day) -> "QuarterCube":
        """Strictly earlier dates: exactly the history visible when `day` is reconstructed."""
        return self._select(self.dates < np.datetime64(day, "D"))

    def until(self, day) -> "QuarterCube":
        return self._select(self.dates <= np.datetime64(day, "D"))

    def merge(self, other: "QuarterCube") -> "QuarterCube":
        """Union of both cubes; finite values of `other` win (jury history over stored history)."""
        if other.zones != self.zones:
            raise ValueError("cannot merge cubes with different zones")
        dates = np.union1d(self.dates, other.dates).astype("datetime64[D]")
        values = np.full((len(dates), len(self.zones), HOURS, QUARTERS), np.nan)
        values[np.searchsorted(dates, self.dates)] = self.values
        idx = np.searchsorted(dates, other.dates)
        block = values[idx]
        finite = np.isfinite(other.values)
        block[finite] = other.values[finite]
        values[idx] = block
        return QuarterCube(dates, self.zones, values)

    def to_long(self, value_col: str = "price_eur_mwh") -> pd.DataFrame:
        d, z, h, q = np.meshgrid(
            np.arange(len(self.dates)), np.arange(len(self.zones)),
            np.arange(HOURS), np.arange(QUARTERS), indexing="ij",
        )
        return pd.DataFrame({
            "delivery_date": self.dates[d.ravel()].astype("datetime64[ns]"),
            "market_id": np.asarray(self.zones, dtype=object)[z.ravel()],
            "hour_index": h.ravel() + 1,
            "quarter_index": q.ravel() + 1,
            value_col: self.values.ravel(),
        })

    def save(self, path) -> None:
        np.savez_compressed(
            path, dates=self.dates.astype("int64"), zones=np.array(self.zones, dtype=str), values=self.values
        )

    @classmethod
    def load(cls, path) -> "QuarterCube":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["dates"].astype("datetime64[D]"),
                tuple(str(z) for z in data["zones"]),
                data["values"].copy(),
            )


@dataclass(frozen=True)
class HourlyCube:
    """Hourly prices as float64 [date, zone, hour]."""

    dates: np.ndarray
    zones: tuple
    values: np.ndarray

    @classmethod
    def from_long(cls, df: pd.DataFrame, zones, value_col: str = "hourly_price_eur_mwh") -> "HourlyCube":
        zones = tuple(zones)
        df = df[df["market_id"].isin(zones)]
        if df.empty:
            return cls(np.array([], dtype="datetime64[D]"), zones, np.empty((0, len(zones), HOURS)))
        dates, d, z = _index(df, zones)
        values = np.full((len(dates), len(zones), HOURS), np.nan)
        values[d, z, df["hour_index"].to_numpy() - 1] = df[value_col].to_numpy(dtype=float)
        return cls(dates, zones, values)
