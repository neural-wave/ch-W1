"""Orthonormal zero-sum bases for the four quarter-hours of an hour.

Every combination of these vectors sums to zero, so adding it to a
mean-preserving baseline cannot move the hourly average: the challenge
constraint holds by construction, not by post-hoc correction.
"""
from __future__ import annotations

import numpy as np

_POLY = np.array([
    [-3.0, -1.0, 1.0, 3.0],   # slope: positive = rising inside the hour
    [1.0, -1.0, -1.0, 1.0],   # curvature: U (positive) or peak (negative)
    [-1.0, 3.0, -3.0, 1.0],   # zigzag
]).T
_HAAR = np.array([
    [1.0, 1.0, -1.0, -1.0],   # first half vs second half
    [1.0, -1.0, 0.0, 0.0],    # detail inside the first half
    [0.0, 0.0, 1.0, -1.0],    # detail inside the second half
]).T

BASES = {
    "poly": _POLY / np.linalg.norm(_POLY, axis=0),
    "haar": _HAAR / np.linalg.norm(_HAAR, axis=0),
}
COMPONENT_NAMES = {
    "poly": ("slope", "curvature", "zigzag"),
    "haar": ("halves", "first_half", "second_half"),
}


def get_basis(name: str) -> np.ndarray:
    """4x3 matrix with orthonormal, zero-sum columns."""
    try:
        return BASES[name].copy()
    except KeyError:
        raise ValueError(f"unknown basis {name!r}; choose from {sorted(BASES)}") from None


def encode(residual: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """[..., 4] residual -> [..., 3] coefficients (drops any non-zero-sum part)."""
    return residual @ basis


def decode(coeffs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """[..., 3] coefficients -> [..., 4] zero-sum residual."""
    return coeffs @ basis.T


def center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)
