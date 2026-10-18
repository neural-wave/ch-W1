"""Calendar-indexed history of past residual shapes.

Lookups for a target date only ever read calendar slots strictly before that
date, and lags are counted in calendar days, never in rows: the day after a
DST-excluded date sees a missing lag, not the wrong day. Nothing here is fitted,
so recomputing it on fresher jury history is feature calculation, not
recalibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import get_baseline
from .basis import encode
from .cube import QuarterCube
from .schema import HOURS


@dataclass(frozen=True)
class HistoryStore:
    start: np.datetime64 | None
    zones: tuple
    hourly: np.ndarray   # [slot, zone, hour]
    coeffs: np.ndarray   # [slot, zone, hour, k] residual coefficients vs the baseline
    within_range: np.ndarray | None = None  # [slot, zone, hour, 1], observed historical price range

    @classmethod
    def build(cls, cube: QuarterCube, baseline: str, basis: np.ndarray) -> "HistoryStore":
        n_zones, k = len(cube.zones), basis.shape[1]
        if len(cube.dates) == 0:
            return cls(None, cube.zones, np.empty((0, n_zones, HOURS)), np.empty((0, n_zones, HOURS, k)),
                       np.empty((0, n_zones, HOURS, 1)))
        start = cube.dates[0]
        slots = int((cube.dates[-1] - start).astype(np.int64)) + 1
        pos = (cube.dates - start).astype(np.int64)
        H = cube.hourly()
        hourly = np.full((slots, n_zones, HOURS), np.nan)
        hourly[pos] = H
        coeffs = np.full((slots, n_zones, HOURS, k), np.nan)
        coeffs[pos] = encode(cube.values - get_baseline(baseline)(H), basis)
        within_range = np.full((slots, n_zones, HOURS, 1), np.nan)
        within_range[pos, ..., 0] = np.ptp(cube.values, axis=-1)
        return cls(start, cube.zones, hourly, coeffs, within_range)

    @property
    def n_slots(self) -> int:
        return self.hourly.shape[0]

    def _positions(self, dates) -> np.ndarray:
        return (np.asarray(dates, dtype="datetime64[D]") - self.start).astype(np.int64)

    def lag(self, field: str, dates, k: int) -> np.ndarray:
        """Value of `field` exactly k calendar days before each date (NaN if absent)."""
        if k < 1:
            raise ValueError("history lookups must be strictly in the past (k >= 1)")
        arr = getattr(self, field)
        out = np.full((len(dates),) + arr.shape[1:], np.nan)
        if self.start is None:
            return out
        src = self._positions(dates) - k
        ok = (src >= 0) & (src < self.n_slots)
        out[ok] = arr[src[ok]]
        return out

    def window_mean(self, field: str, dates, window: int, absolute: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Mean of `field` over the `window` calendar days before each date, ignoring gaps.

        Arithmetic means of zero-sum shapes stay zero-sum (medians would not).
        Returns (mean, count of available days).
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        arr = getattr(self, field)
        if arr is None:
            raise ValueError(f"history field {field!r} is not available")
        if absolute:
            arr = np.abs(arr)  # absolute BEFORE averaging: opposite signs must not cancel
        shape = (len(dates),) + arr.shape[1:]
        if self.start is None:
            return np.full(shape, np.nan), np.zeros(shape, dtype=np.int64)
        pos = self._positions(dates)
        finite = np.isfinite(arr)
        zero = np.zeros((1,) + arr.shape[1:])
        csum = np.concatenate([zero, np.cumsum(np.where(finite, arr, 0.0), axis=0)])
        ccnt = np.concatenate([zero, np.cumsum(finite, axis=0)])
        hi = np.clip(pos, 0, self.n_slots)
        lo = np.clip(pos - window, 0, self.n_slots)
        total = csum[hi] - csum[lo]
        count = (ccnt[hi] - ccnt[lo]).astype(np.int64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        return mean, count


# --------------------------------------------------------------------------- analogue memory (E4)
def curve_descriptors(dates, hourly: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Describe the local hourly curve around every (date, zone, hour): [n, zone, 24, d].

    Used identically for queries and stored candidates: slopes at -2..+2 (hours 1-2 read the
    previous day, hours 23-24 never see the next day, which is unknown at inference), day level
    and range, the current-hour spread to every zone, and the weekend flag.
    """
    n, n_zones, _ = hourly.shape
    ext = np.concatenate([prev[..., -2:], hourly, np.full((n, n_zones, 2), np.nan)], axis=-1)
    hm2, hm1, h0, hp1, hp2 = (ext[..., i:i + HOURS] for i in range(5))
    shape = (n, n_zones, HOURS)
    weekend = (np.asarray(dates, dtype="datetime64[D]").astype(np.int64) + 3) % 7 >= 5   # day 0 = Thursday
    parts = [hm1 - hm2, h0 - hm1, hp1 - h0, hp2 - hp1,
             np.broadcast_to(hourly.mean(axis=-1, keepdims=True), shape),
             np.broadcast_to(hourly.max(axis=-1, keepdims=True) - hourly.min(axis=-1, keepdims=True), shape)]
    parts += [hourly[:, k:k + 1] - hourly for k in range(n_zones)]
    parts.append(np.broadcast_to(weekend[:, None, None].astype(float), shape))
    return np.stack(parts, axis=-1)


@dataclass(frozen=True)
class AnalogueMemory:
    """Curve-conditioned retrieval of past residual shapes (roadmap section B).

    For each zone and hour, the K earlier dates (within `window_days`, same zone and hour) whose
    standardised curve descriptors are closest to the query are averaged with one Gaussian weight
    per neighbour (bandwidth = distance of the K-th neighbour). The scaler is fitted once on the
    training slice and frozen; at inference this is a fixed lookup over the supplied history.
    """

    k: int
    window_days: int | None
    center: np.ndarray   # [d]
    scale: np.ndarray    # [d]

    @classmethod
    def fit(cls, store: HistoryStore, k: int = 16, window_days: int | None = 180) -> "AnalogueMemory":
        desc = cls._store_descriptors(store)
        flat = desc.reshape(-1, desc.shape[-1])
        center = np.nanmean(flat, axis=0)
        scale = np.nanstd(flat, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        return cls(int(k), None if window_days is None else int(window_days), center, scale)

    @staticmethod
    def _store_descriptors(store: HistoryStore) -> np.ndarray:
        prev = np.concatenate([np.full((1,) + store.hourly.shape[1:], np.nan), store.hourly[:-1]])
        dates = store.start + np.arange(store.n_slots) if store.start is not None else np.array([], "datetime64[D]")
        return curve_descriptors(dates, store.hourly, prev)

    def _standardise(self, desc: np.ndarray) -> np.ndarray:
        z = (desc - self.center) / self.scale
        return np.where(np.isfinite(z), z, 0.0)                    # missing descriptor -> training mean

    def lookup(self, store: HistoryStore, dates, H: np.ndarray, explain: tuple | None = None) -> dict:
        """Memory features for queries (dates, H [n, zone, 24]); every candidate is strictly earlier.

        Returns arrays [n, zone, 24]: mem_c1..mem_ck (weighted residual coefficients, zeros when no
        candidate exists), mem_dist (nearest distance, NaN if none), mem_count (candidates in the
        window; 0 is the missing flag) and mem_eff (1 / sum w^2). With explain=(i, zone, hour),
        also returns the selected dates and weights for that query (diagnostics).
        """
        dates = np.asarray(dates, dtype="datetime64[D]")
        n, n_zones, _ = H.shape
        n_coef = store.coeffs.shape[-1]
        out = {f"mem_c{j + 1}": np.zeros((n, n_zones, HOURS)) for j in range(n_coef)}
        out["mem_dist"] = np.full((n, n_zones, HOURS), np.nan)
        out["mem_count"] = np.zeros((n, n_zones, HOURS))
        out["mem_eff"] = np.zeros((n, n_zones, HOURS))
        if store.start is None or n == 0:
            return out
        query = self._standardise(curve_descriptors(dates, H, store.lag("hourly", dates, 1)))
        cand = self._standardise(self._store_descriptors(store))
        valid = np.isfinite(store.coeffs).all(axis=-1) & np.isfinite(store.hourly)      # [slot, zone, hour]
        age = store._positions(dates)[:, None] - np.arange(store.n_slots)[None, :]       # [n, slot]
        allowed = age >= 1
        if self.window_days is not None:
            allowed &= age <= self.window_days
        qq = (query ** 2).sum(-1)                                                       # [n, zone, hour]
        cc = (cand ** 2).sum(-1)                                                        # [slot, zone, hour]
        for z in range(n_zones):
            d2 = qq[:, z, :, None] + cc[None, :, z, :].transpose(0, 2, 1) \
                - 2.0 * np.einsum("nhd,shd->nhs", query[:, z], cand[:, z])           # [n, hour, slot]
            ok = allowed[:, None, :] & valid[:, z, :].T[None]
            d2 = np.where(ok, np.maximum(d2, 0.0), np.inf)
            count = ok.sum(-1)
            kk = min(self.k, store.n_slots)
            idx = np.argpartition(d2, kk - 1, axis=-1)[..., :kk]                       # [n, hour, kk]
            dist = np.sqrt(np.take_along_axis(d2, idx, axis=-1))
            live = np.isfinite(dist)
            band = np.maximum(np.where(live, dist, 0.0).max(-1, keepdims=True), 1e-6)
            w = np.where(live, np.exp(-(dist / band) ** 2), 0.0)
            wsum = w.sum(-1, keepdims=True)
            w = np.divide(w, wsum, out=np.zeros_like(w), where=wsum > 0)
            coef = store.coeffs[:, z].transpose(1, 0, 2)                                # [hour, slot, k]
            picked = coef[np.arange(HOURS)[None, :, None], idx]                          # [n, hour, kk, k]
            mem = np.einsum("nhk,nhkc->nhc", w, np.nan_to_num(picked))
            has = count > 0
            for j in range(n_coef):
                out[f"mem_c{j + 1}"][:, z] = np.where(has, mem[..., j], 0.0)
            out["mem_dist"][:, z] = np.where(has, np.where(live, dist, np.inf).min(-1), np.nan)
            out["mem_count"][:, z] = count
            out["mem_eff"][:, z] = np.where(has, 1.0 / np.maximum((w ** 2).sum(-1), 1e-12), 0.0)
            if explain is not None and explain[1] == z:
                i, _, h = explain
                keep = live[i, h]
                out["explain"] = {"dates": [str(store.start + int(s)) for s in idx[i, h][keep]],
                                  "weights": w[i, h][keep].round(4).tolist(),
                                  "distances": dist[i, h][keep].round(4).tolist()}
        return out

    def to_dict(self) -> dict:
        return {"k": self.k, "window_days": self.window_days,
                "center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "AnalogueMemory":
        return cls(int(data["k"]), data["window_days"], np.asarray(data["center"], float),
                   np.asarray(data["scale"], float))
