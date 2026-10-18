"""Parameter-free, mean-preserving baselines: hourly curve [..., 24] -> [..., 24, 4].

They need no fitting, so they are also the fallback when a learned residual is
unavailable.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

from .schema import HOURS, QUARTERS

Baseline = Callable[[np.ndarray], np.ndarray]
_OFFSETS = (np.arange(QUARTERS) + 0.5) / QUARTERS - 0.5   # [-0.375, -0.125, 0.125, 0.375]


def flat(H: np.ndarray) -> np.ndarray:
    """Official comparison baseline: the hourly price repeated four times."""
    H = np.asarray(H, dtype=float)
    return np.repeat(H[..., None], QUARTERS, axis=-1)


def central_slope(H: np.ndarray) -> np.ndarray:
    """Hourly slope in EUR/MWh per hour: central difference, one-sided at the day edges."""
    H = np.asarray(H, dtype=float)
    slope = np.empty_like(H)
    slope[..., 1:-1] = (H[..., 2:] - H[..., :-2]) / 2.0
    slope[..., 0] = H[..., 1] - H[..., 0]
    slope[..., -1] = H[..., -1] - H[..., -2]
    return slope


def linear_ramp(H: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Centred ramp following the central hourly slope; mean-preserving because the offsets sum to zero."""
    H = np.asarray(H, dtype=float)
    return H[..., None] + scale * central_slope(H)[..., None] * _OFFSETS


_INTERPOLATORS = {
    "pchip": lambda x, y: PchipInterpolator(x, y, axis=0),
    "akima": lambda x, y: Akima1DInterpolator(x, y, axis=0),
    "spline": lambda x, y: CubicSpline(x, y, axis=0, bc_type="natural"),
}


def cumulative(H: np.ndarray, method: str = "pchip") -> np.ndarray:
    """Interpolate the cumulative hourly curve and difference it on the quarter grid.

    C[0] = 0, C[h+1] = C[h] + H[h]; F interpolates (h, C[h]); quarter q of hour h is
    4 * (F(h + (q+1)/4) - F(h + q/4)). The four values telescope to 4 * H[h], so the
    hourly mean is preserved algebraically for any interpolator through the knots.
    Days with a missing hour come back as NaN (the fallback chain handles them).
    """
    H = np.asarray(H, dtype=float)
    lead = H.shape[:-1]
    days = H.reshape(-1, HOURS)
    out = np.full((days.shape[0], HOURS * QUARTERS), np.nan)
    ok = np.isfinite(days).all(axis=1)
    if ok.any():
        C = np.zeros((HOURS + 1, int(ok.sum())))
        C[1:] = np.cumsum(days[ok], axis=1).T
        F = _INTERPOLATORS[method](np.arange(HOURS + 1, dtype=float), C)
        grid = F(np.arange(HOURS * QUARTERS + 1) / QUARTERS)
        out[ok] = (QUARTERS * np.diff(grid, axis=0)).T
    return out.reshape(*lead, HOURS, QUARTERS)


BASELINES: dict[str, Baseline] = {
    "flat": flat,
    "linear": linear_ramp,
    "linear_x2": partial(linear_ramp, scale=2.0),
    "pchip": partial(cumulative, method="pchip"),
    "akima": partial(cumulative, method="akima"),
    "spline": partial(cumulative, method="spline"),
}


def get_baseline(name: str) -> Baseline:
    try:
        return BASELINES[name]
    except KeyError:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}") from None
