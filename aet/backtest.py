"""Rolling-origin backtest with history replay and the official metric.

For every scored date the fold model (frozen at the fold's train_end) receives
only that date's hourly curves and the quarter-hour history strictly before it,
exactly as the jury will run it.

    python -m aet.backtest --train data/train_prices_15min.csv --config configs/core.yaml [--folds monthly]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Fold, candidate_configs, folds_from_config, gate_from_config, load_config
from .cube import QuarterCube
from .pipeline import PipelineConfig, ShapePipeline
from .schema import QUARTERS, check_training_window, read_quarter_hourly

log = logging.getLogger("aet.backtest")
KEYS = ["fold", "delivery_date", "zone"]


def evaluate(cube: QuarterCube, config: PipelineConfig, folds: list[Fold]) -> tuple[pd.DataFrame, pd.DataFrame]:
    H_all = cube.hourly()
    rows, timings = [], []
    for fold in folds:
        started = time.perf_counter()
        pipe = ShapePipeline(config).fit(cube.until(fold.train_end))
        train_s = time.perf_counter() - started
        started = time.perf_counter()
        scored = np.flatnonzero((cube.dates >= fold.score_start) & (cube.dates <= fold.score_end))
        for i in scored:
            day = cube.dates[i]
            P, info = pipe.predict_day(day, H_all[i], cube)   # history is sliced to < day inside
            for z, zone in enumerate(cube.zones):
                obs = cube.values[i, z]
                if not np.isfinite(obs).all():
                    continue
                err = np.abs(P[z] - obs)
                rows.append({
                    "candidate": config.name, "fold": fold.name, "delivery_date": str(day), "zone": zone,
                    "mae": err.mean(),
                    "flat_mae": np.abs(H_all[i, z][:, None] - obs).mean(),
                    **{f"q{q + 1}_mae": err[:, q].mean() for q in range(QUARTERS)},
                    "max_mean_error": float(np.abs(P[z].mean(axis=-1) - H_all[i, z]).max()),
                    "fallback_hours": int(info["fallback_hours"][z]),
                    **({"bracket_fail_hours": int(info["bracket_fail_hours"][z])}
                       if "bracket_fail_hours" in info else {}),
                })
        timings.append({"candidate": config.name, "fold": fold.name, "train_s": train_s,
                        "infer_s": time.perf_counter() - started, "scored_dates": int(len(scored))})
    return pd.DataFrame(rows), pd.DataFrame(timings)


def summarize(per_day: pd.DataFrame, timings: pd.DataFrame, reference: str, gate: dict) -> pd.DataFrame:
    """Official score = mean over zones of (mean over days of daily MAE), pooled over folds."""
    zone_scores = per_day.groupby(["candidate", "zone"])[["mae", "flat_mae"]].mean()
    overall = zone_scores.groupby(level="candidate").mean()
    fold_scores = per_day.groupby(["candidate", "fold", "zone"])["mae"].mean().groupby(level=["candidate", "fold"]).mean()
    ref_zone, ref_fold = zone_scores.loc[reference, "mae"], fold_scores.loc[reference]
    ref_daily = per_day[per_day["candidate"] == reference].set_index(KEYS)["mae"]
    rows = []
    for name in per_day["candidate"].unique():
        sub = per_day[per_day["candidate"] == name]
        score, flat = overall.loc[name, "mae"], overall.loc[name, "flat_mae"]
        rel = 1.0 - score / overall.loc[reference, "mae"]
        cand_fold, cand_zone = fold_scores.loc[name], zone_scores.loc[name, "mae"]
        folds_better = int((cand_fold < ref_fold).sum())
        worst_zone = float((cand_zone / ref_zone - 1.0).max())
        paired = (sub.set_index(KEYS)["mae"] - ref_daily).dropna()
        is_ref = name == reference
        passed = (not is_ref and rel >= gate["min_rel_improvement"]
                  and folds_better / len(ref_fold) >= gate["min_fold_share"]
                  and worst_zone <= gate["max_zone_degradation"])
        t = timings[timings["candidate"] == name]
        rows.append({
            "candidate": name, "score": score, "flat": flat,
            "vs_flat_pct": 100.0 * (1.0 - score / flat), "vs_ref_pct": 100.0 * rel,
            "folds_better": f"{folds_better}/{len(ref_fold)}", "worst_zone_pct": 100.0 * worst_zone,
            "day_win_rate": np.nan if is_ref else float((paired < 0).mean()),
            "gate": "ref" if is_ref else ("PASS" if passed else "-"),
            **{zone: cand_zone.loc[zone] for zone in cand_zone.index},
            **{f"q{q}": sub[f"q{q}_mae"].mean() for q in range(1, QUARTERS + 1)},
            "max_mean_err": sub["max_mean_error"].max(), "fallback_hours": int(sub["fallback_hours"].sum()),
            **({"bracket_fail_hours": int(sub["bracket_fail_hours"].sum())}
               if "bracket_fail_hours" in sub and sub["bracket_fail_hours"].notna().any() else {}),
            "train_s": t["train_s"].sum(), "infer_s": t["infer_s"].sum(),
        })
    return pd.DataFrame(rows)


def _print(title: str, summary: pd.DataFrame) -> None:
    head = ["candidate", "score", "vs_flat_pct", "vs_ref_pct", "folds_better", "worst_zone_pct", "day_win_rate", "gate"]
    detail = ["candidate"] + [c for c in summary.columns if c.startswith("ZONE_") or c in {"q1", "q2", "q3", "q4"}] \
        + ["max_mean_err", "fallback_hours", "bracket_fail_hours", "train_s", "infer_s"]
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.float_format", "{:.4f}".format):
        print(f"\n=== {title} ===")
        print(summary[head].to_string(index=False))
        print(summary[[c for c in detail if c in summary.columns]].to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="official train_prices_15min.csv")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--folds", choices=["explicit", "monthly"], help="override the config's fold scheme")
    parser.add_argument("--candidates", help="comma-separated subset of candidates")
    parser.add_argument("--reference", help="incumbent for the promotion gate (default: first candidate)")
    parser.add_argument("--include-locked", action="store_true",
                        help="also score the locked block: do this once, after freezing, never to tune")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw = load_config(args.config)
    configs = candidate_configs(raw, args.candidates.split(",") if args.candidates else None)
    folds, locked = folds_from_config(raw, args.folds)
    reference = args.reference or configs[0].name
    if reference not in {c.name for c in configs}:
        parser.error(f"reference {reference!r} is not among the evaluated candidates")
    gate = gate_from_config(raw)

    df = read_quarter_hourly(args.train, strict=True)
    check_training_window(df)
    cube = QuarterCube.from_long(df, configs[0].zones)
    log.info("%d dates, %d zones, %d folds: %s", len(cube.dates), len(cube.zones), len(folds),
             ", ".join(f.name for f in folds))

    run_dir = Path(args.out) / f"{Path(args.config).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    blocks = [("development", folds)] + ([("locked", [locked])] if args.include_locked and locked else [])
    for label, block in blocks:
        if label == "locked":
            log.warning("scoring the LOCKED block: report it, do not start another tuning cycle")
        results = [evaluate(cube, cfg, block) for cfg in configs]
        per_day = pd.concat([r[0] for r in results], ignore_index=True)
        timings = pd.concat([r[1] for r in results], ignore_index=True)
        summary = summarize(per_day, timings, reference, gate)
        per_day.to_csv(run_dir / f"{label}_per_day.csv", index=False)
        summary.to_csv(run_dir / f"{label}_summary.csv", index=False)
        _print(f"{label} ({len(block)} folds, reference {reference})", summary)
        with open(Path(args.out) / "ledger.jsonl", "a", encoding="utf-8") as ledger:
            for row in summary.to_dict("records"):
                ledger.write(json.dumps({
                    "time": datetime.now().isoformat(timespec="seconds"), "block": label,
                    "config": raw["_path"], "config_sha256": raw["_sha256"],
                    "folds": [f.name for f in block], "run_dir": str(run_dir),
                    **{k: row[k] for k in ("candidate", "score", "vs_flat_pct", "vs_ref_pct", "folds_better", "gate")},
                }, default=float) + "\n")
    (run_dir / "run.json").write_text(json.dumps({
        "config": raw["_path"], "config_sha256": raw["_sha256"], "reference": reference, "gate": gate,
        "folds": [vars(f) for f in folds], "locked": vars(locked) if locked else None,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nresults in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
