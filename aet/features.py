"""Feature construction shared by training, backtesting and inference.

Per target date the inputs are: the 24 hourly prices of every zone (allowed by
the brief), the baseline shape, and a HistoryStore that only exposes earlier
dates. Rows are ordered (date, zone, hour) everywhere in the package.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from .history import HistoryStore
from .schema import HOURS

CATEGORICAL = ["zone"]
LAGS = (1, 7)
WINDOWS = (7, 28)


@dataclass(frozen=True)
class FeatureConfig:
    history: bool = True         # lagged residual shapes from strictly earlier dates
    cross_zone: bool = True      # same-day hourly curves of every zone
    prev_day_edge: bool = True   # hour 1 sees the previous day's last hours (from history)
    volatility_history: bool = False  # opt-in research extension; old/frozen features unchanged
    local_geometry: bool = False
    extrema_distance: bool = False

    @classmethod
    def from_dict(cls, data) -> "FeatureConfig":
        data = dict(data or {})
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown feature options {sorted(unknown)}")
        return cls(**{k: bool(v) for k, v in data.items()})


def build_features(dates, H: np.ndarray, base_coeffs: np.ndarray, store: HistoryStore, cfg: FeatureConfig,
                   memory: dict | None = None) -> pd.DataFrame:
    """H: [date, zone, 24] hourly prices; base_coeffs: [date, zone, 24, k];
    memory: optional analogue-memory arrays [date, zone, 24] (AnalogueMemory.lookup)."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    n_dates, n_zones, _ = H.shape
    shape = (n_dates, n_zones, HOURS)
    cols: dict[str, np.ndarray] = {}

    def put(name: str, value) -> None:
        cols[name] = np.broadcast_to(np.asarray(value, dtype=float), shape).reshape(-1)

    calendar = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    weekday = calendar.dayofweek.to_numpy()[:, None, None]
    season = 2 * np.pi * calendar.dayofyear.to_numpy()[:, None, None] / 365.25
    put("zone", np.arange(n_zones)[None, :, None])
    put("hour", np.arange(HOURS)[None, None, :])
    put("weekday", weekday)
    put("weekend", weekday >= 5)
    put("season_sin", np.sin(season))
    put("season_cos", np.cos(season))

    # Neighbouring hours. The previous day is known history, so hour 1 is not an edge;
    # the next day is unknown, so hours 23-24 look forward into NaN (LightGBM handles it).
    prev = store.lag("hourly", dates, 1) if cfg.prev_day_edge else np.full(shape, np.nan)
    ext = np.concatenate([prev[..., -2:], H, np.full((n_dates, n_zones, 2), np.nan)], axis=-1)
    hm2, hm1, h0, hp1, hp2 = (ext[..., i:i + HOURS] for i in range(5))
    slope_m2, slope_m1, slope_p1, slope_p2 = hm1 - hm2, h0 - hm1, hp1 - h0, hp2 - hp1
    put("level", h0)
    put("slope_m2", slope_m2)
    put("slope_m1", slope_m1)
    put("slope_p1", slope_p1)
    put("slope_p2", slope_p2)
    put("curvature", slope_p1 - slope_m1)

    day_mean = H.mean(axis=-1, keepdims=True)
    put("day_mean", day_mean)
    put("day_range", H.max(axis=-1, keepdims=True) - H.min(axis=-1, keepdims=True))
    put("day_std", H.std(axis=-1, keepdims=True))
    put("day_negative_hours", (H < 0).sum(axis=-1, keepdims=True))
    put("relative_level", h0 - day_mean)

    if cfg.local_geometry:
        b, f = slope_m1, slope_p1
        total = np.abs(b) + np.abs(f)
        put("local_movement", total / 2.0)
        put("turning_point", np.where(np.isfinite(b) & np.isfinite(f), (b*f < 0).astype(float), np.nan))
        # Normalised signed curvature: not the same as |f|-|b|.
        put("ramp_asymmetry", (f-b)/(total + 1e-6))
        day_range = H.max(axis=-1, keepdims=True) - H.min(axis=-1, keepdims=True)
        put("norm_movement", total/np.maximum(day_range, 1.0))

    if cfg.extrema_distance:
        # First occurrence breaks ties; constant days get no unique extrema location.
        idx = np.arange(HOURS)[None, None, :]
        varying = (H.max(axis=-1, keepdims=True)-H.min(axis=-1, keepdims=True)) > 0
        put("dist_to_peak", np.where(varying, np.abs(idx-H.argmax(axis=-1, keepdims=True)), np.nan))
        put("dist_to_trough", np.where(varying, np.abs(idx-H.argmin(axis=-1, keepdims=True)), np.nan))

    if cfg.cross_zone:
        for k in range(n_zones):
            put(f"z{k}_spread", H[:, k:k + 1] - H)
            put(f"z{k}_slope_m1", slope_m1[:, k:k + 1])
            put(f"z{k}_slope_p1", slope_p1[:, k:k + 1])

    for j in range(base_coeffs.shape[-1]):
        put(f"base_c{j + 1}", base_coeffs[..., j])

    if cfg.history:
        for k in LAGS:
            lagged = store.lag("coeffs", dates, k)
            for j in range(lagged.shape[-1]):
                put(f"lag{k}_c{j + 1}", lagged[..., j])
        for w in WINDOWS:
            mean, count = store.window_mean("coeffs", dates, w)
            for j in range(mean.shape[-1]):
                put(f"mean{w}_c{j + 1}", mean[..., j])
            put(f"n{w}", count[..., 0])

    if cfg.volatility_history:
        for w in WINDOWS:
            magnitude, count = store.window_mean("coeffs", dates, w, absolute=True)
            for j in range(magnitude.shape[-1]):
                put(f"mean_abs{w}_c{j + 1}", magnitude[..., j])
            mean_range, range_count = store.window_mean("within_range", dates, w)
            put(f"mean_range{w}", mean_range[..., 0])
            put(f"n_abs{w}", count[..., 0])
            put(f"n_range{w}", range_count[..., 0])

    for name, value in (memory or {}).items():
        if name != "explain":
            put(name, value)

    frame = pd.DataFrame(cols)
    frame["zone"] = frame["zone"].astype(np.int32)
    return frame
