"""Empirical (split-conformal) uncertainty ranges around the point prediction.

The point model is unchanged. Ranges come from honest out-of-sample errors: the monthly
development folds are replayed exactly like `aet.backtest` (each fold trained only on earlier
months, history strictly before each scored date), the signed errors actual - predicted are
pooled per zone x hour x quarter, and their lower/upper quantiles are stored. At display time
the range is prediction + [q_lo, q_hi]. Nothing is refitted at inference.

The ranges are display-only. They are not part of the submission schema and do not touch the
hourly-mean constraint, which applies to the point prediction alone.

    PYTHONPATH=. python dashboard/intervals.py --train data/train_prices_15min.csv --config configs/final.yaml \
        --out dashboard/september_80.json

Default folds score Jan-Aug 2026, so every calibration error predates September: the table is
valid for the September assessment model without peeking at September.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from aet.config import final_config, folds_from_config, load_config
from aet.cube import QuarterCube
from aet.pipeline import ShapePipeline
from aet.schema import HOURS, QUARTERS, check_training_window, read_quarter_hourly

log = logging.getLogger("dashboard.intervals")
DEFAULT_LEVEL = 0.8


def fold_errors(cube: QuarterCube, pipe_config, folds) -> np.ndarray:
    """Signed errors actual - predicted, stacked as [scored date, zone, 24, 4] over all folds."""
    H_all = cube.hourly()
    errors = []
    for fold in folds:
        scored = np.flatnonzero((cube.dates >= fold.score_start) & (cube.dates <= fold.score_end))
        if len(scored) == 0:
            log.warning("fold %s has no scored dates in the data; skipped", fold.name)
            continue
        pipe = ShapePipeline(pipe_config).fit(cube.until(fold.train_end))
        for i in scored:
            P, _ = pipe.predict_day(cube.dates[i], H_all[i], cube)   # history is sliced to < day inside
            errors.append(cube.values[i] - P)
        log.info("fold %s: %d scored dates", fold.name, len(scored))
    return np.stack(errors) if errors else np.empty((0, len(cube.zones), HOURS, QUARTERS))


def interval_table(errors: np.ndarray, level: float = DEFAULT_LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed quantiles of the errors per zone x hour x quarter -> (lo, hi), each [zone, 24, 4]."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    tail = (1 - level) / 2
    lo = np.nanquantile(errors, tail, axis=0)
    hi = np.nanquantile(errors, 1 - tail, axis=0)
    return lo, hi


PERCENTILES = np.arange(1, 100)   # the stored error distribution: percentiles 1..99


def percentile_table(errors: np.ndarray) -> np.ndarray:
    """Error percentiles 1..99 per zone x hour x quarter -> [99, zone, 24, 4] (the empirical distribution)."""
    return np.nanpercentile(errors, PERCENTILES, axis=0)


def band(percentiles: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed range at `level` (0.5, 0.8, 0.9, ...) from the stored percentiles."""
    tail = round((1 - level) / 2 * 100)
    if not 1 <= tail <= 49:
        raise ValueError("level must be between 0.02 and 0.98")
    return percentiles[tail - 1], percentiles[99 - tail]


def chance_within(percentiles: np.ndarray, distance: float) -> np.ndarray:
    """Probability (1% resolution) that |actual - predicted| <= distance, per zone x hour x quarter."""
    return (np.abs(percentiles) <= distance).mean(axis=0)


def save_table(path, zones, lo, hi, level: float, meta: dict, percentiles: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"level": level, "zones": list(zones), "lo": np.round(lo, 4).tolist(),
               "hi": np.round(hi, 4).tolist(), **meta}
    if percentiles is not None:
        payload["percentiles"] = np.round(percentiles, 3).tolist()
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def load_table(path) -> dict:
    """{'level', 'zones', 'lo'/'hi' [zone,24,4], optional 'percentiles' [99,zone,24,4], ...} as numpy arrays."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("lo", "hi", "percentiles"):
        if key in raw:
            raw[key] = np.asarray(raw[key], dtype=float)
    return raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="official train_prices_15min.csv")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--candidate", help="default: the config's `final: true` candidate")
    parser.add_argument("--level", type=float, default=DEFAULT_LEVEL, help="coverage of the range, e.g. 0.8")
    parser.add_argument("--out", default="dashboard/september_80.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw = load_config(args.config)
    config = final_config(raw, args.candidate)
    folds, _ = folds_from_config(raw, "monthly")
    df = read_quarter_hourly(args.train, strict=True)
    check_training_window(df)
    cube = QuarterCube.from_long(df, config.zones)

    errors = fold_errors(cube, config, folds)
    if len(errors) == 0:
        parser.error("no scored dates: check the folds against the training data")
    lo, hi = interval_table(errors, args.level)
    inside = ((errors >= lo) & (errors <= hi)).mean()
    save_table(args.out, cube.zones, lo, hi, args.level, percentiles=percentile_table(errors), meta={
        "candidate": config.name, "config": raw["_path"], "config_sha256": raw["_sha256"],
        "folds": [f.name for f in folds], "scored_dates": int(len(errors)),
        "calibration_coverage": round(float(inside), 4), "created": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"{len(errors)} scored dates, level {args.level:.0%}, in-sample coverage {inside:.1%}, "
          f"mean width {np.mean(hi - lo):.2f} EUR/MWh -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
