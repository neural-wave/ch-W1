"""Data contracts for every file that enters or leaves the pipeline.

After these functions the rest of the package can assume normalised dates,
1-based integer indices, stripped zone ids, float64 prices and unique keys.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HOURS = 24
QUARTERS = 4
OFFICIAL_TOLERANCE = 1e-4                          # EUR/MWh, brief section 9
OFFICIAL_TRAIN_END = np.datetime64("2026-09-30")   # 1-18 Oct 2026 is off limits before the freeze

QH_KEYS = ["delivery_date", "market_id", "hour_index", "quarter_index"]
HOUR_KEYS = ["delivery_date", "market_id", "hour_index"]
QH_COLUMNS = QH_KEYS + ["price_eur_mwh"]
HOURLY_COLUMNS = HOUR_KEYS + ["hourly_price_eur_mwh"]
PRED_COLUMNS = QH_KEYS + ["predicted_price_eur_mwh"]


class SchemaError(ValueError):
    """An input or output file violates the challenge contract."""


def to_timestamps(values) -> pd.Series:
    """Any date-like column -> midnight timestamps at a single (ns) resolution."""
    return pd.Series(pd.to_datetime(values)).dt.normalize().astype("datetime64[ns]")


def _require(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing} (found {list(df.columns)})")


def _normalise_keys(df: pd.DataFrame, name: str, with_quarter: bool) -> pd.DataFrame:
    out = df.copy()
    try:
        out["delivery_date"] = to_timestamps(out["delivery_date"]).to_numpy()
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"{name}: unparseable delivery_date ({exc})") from exc
    if out["delivery_date"].isna().any():
        raise SchemaError(f"{name}: empty delivery_date values")
    out["market_id"] = out["market_id"].astype(str).str.strip()
    checks = [("hour_index", HOURS)] + ([("quarter_index", QUARTERS)] if with_quarter else [])
    for column, upper in checks:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise SchemaError(f"{name}: non-integer values in {column}")
        if not values.between(1, upper).all():
            raise SchemaError(f"{name}: {column} outside 1..{upper}")
        out[column] = values.astype(np.int64)
    return out


def clean_quarter_hourly(df: pd.DataFrame, name: str = "quarter-hourly data", strict: bool = True) -> pd.DataFrame:
    """Training/history schema. strict=True raises on duplicates or non-finite prices;
    strict=False (jury history) keeps the last duplicate and drops bad rows, with a warning."""
    _require(df, QH_COLUMNS, name)
    out = _normalise_keys(df[QH_COLUMNS], name, with_quarter=True)
    out["price_eur_mwh"] = pd.to_numeric(out["price_eur_mwh"], errors="coerce").astype(np.float64)
    duplicated = out.duplicated(QH_KEYS, keep="last").to_numpy()
    non_finite = ~np.isfinite(out["price_eur_mwh"].to_numpy())
    if duplicated.any() or non_finite.any():
        message = f"{name}: {int(duplicated.sum())} duplicated keys, {int(non_finite.sum())} non-finite prices"
        if strict:
            raise SchemaError(message)
        log.warning("%s; keeping the last duplicate and dropping non-finite rows", message)
        out = out[~duplicated & ~non_finite]
    return out.sort_values(QH_KEYS).reset_index(drop=True)


def clean_hourly(df: pd.DataFrame, name: str = "hourly input") -> pd.DataFrame:
    _require(df, HOURLY_COLUMNS, name)
    out = _normalise_keys(df[HOURLY_COLUMNS], name, with_quarter=False)
    out["hourly_price_eur_mwh"] = pd.to_numeric(out["hourly_price_eur_mwh"], errors="coerce").astype(np.float64)
    if out.duplicated(HOUR_KEYS).any():
        raise SchemaError(f"{name}: duplicated (delivery_date, market_id, hour_index) keys")
    if not np.isfinite(out["hourly_price_eur_mwh"].to_numpy()).all():
        raise SchemaError(f"{name}: non-finite hourly prices cannot be reconstructed")
    return out.sort_values(HOUR_KEYS).reset_index(drop=True)


def read_quarter_hourly(path, strict: bool = True) -> pd.DataFrame:
    return clean_quarter_hourly(pd.read_csv(path), name=str(path), strict=strict)


def read_hourly(path) -> pd.DataFrame:
    return clean_hourly(pd.read_csv(path), name=str(path))


def incomplete_days(df: pd.DataFrame) -> pd.DataFrame:
    """Zone-days without exactly 96 quarter-hours (they are excluded from targets)."""
    counts = df.groupby(["delivery_date", "market_id"]).size()
    return counts[counts != HOURS * QUARTERS].rename("rows").reset_index()


def check_training_window(df: pd.DataFrame, name: str = "training data") -> None:
    """Refuse dates after the official training period (compliance guard)."""
    days = df["delivery_date"].to_numpy().astype("datetime64[D]")
    late = days > OFFICIAL_TRAIN_END
    if late.any():
        raise SchemaError(
            f"{name}: {int(late.sum())} rows after {OFFICIAL_TRAIN_END}; "
            "October 2026 prices must not enter training, validation or selection"
        )


def validate_predictions(pred: pd.DataFrame, hourly: pd.DataFrame, tolerance: float = OFFICIAL_TOLERANCE) -> dict:
    """Mirror of the organisers' eligibility check (brief, section 9)."""
    _require(pred, PRED_COLUMNS, "predictions")
    p = _normalise_keys(pred[PRED_COLUMNS], "predictions", with_quarter=True)
    price = pd.to_numeric(p["predicted_price_eur_mwh"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(price).all():
        raise SchemaError("predictions: non-finite prices")
    p["predicted_price_eur_mwh"] = price
    if p.duplicated(QH_KEYS).any():
        raise SchemaError("predictions: duplicated keys")
    per_hour = (
        p.groupby(HOUR_KEYS)
        .agg(n=("quarter_index", "size"), mean=("predicted_price_eur_mwh", "mean"))
        .reset_index()
    )
    merged = clean_hourly(hourly).merge(per_hour, on=HOUR_KEYS, how="outer", indicator=True)
    missing = int((merged["_merge"] == "left_only").sum())
    extra = int((merged["_merge"] == "right_only").sum())
    if missing or extra:
        raise SchemaError(f"predictions: {missing} required hours missing, {extra} unexpected hours")
    if (merged["n"] != QUARTERS).any():
        raise SchemaError("predictions: some hours do not have exactly four quarter-hours")
    error = (merged["mean"] - merged["hourly_price_eur_mwh"]).abs()
    if error.max() > tolerance:
        raise SchemaError(f"predictions: hourly-mean error {error.max():.3g} exceeds {tolerance}")
    return {"rows": int(len(p)), "hours": int(len(merged)), "max_abs_mean_error": float(error.max())}
