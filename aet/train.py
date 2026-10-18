"""Fit the selected configuration on the official training set and save artifacts.

    python -m aet.train --train data/train_prices_15min.csv --config configs/final.yaml --out artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from .config import final_config, load_config
from .cube import QuarterCube
from .pipeline import ShapePipeline
from .schema import check_training_window, incomplete_days, read_quarter_hourly

log = logging.getLogger("aet.train")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="official train_prices_15min.csv")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="artifact directory")
    parser.add_argument("--candidate", help="candidate name (default: the one marked final)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw = load_config(args.config)
    config = final_config(raw, args.candidate)
    df = read_quarter_hourly(args.train, strict=True)
    check_training_window(df)
    gaps = incomplete_days(df)
    if len(gaps):
        log.warning("%d incomplete zone-days are excluded from the targets", len(gaps))
    cube = QuarterCube.from_long(df, config.zones)

    started = time.perf_counter()
    pipe = ShapePipeline(config).fit(cube)
    seconds = time.perf_counter() - started
    pipe.save(args.out, cube, extra={
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_seconds": round(seconds, 2),
        "training_csv": Path(args.train).name,
        "training_csv_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        "training_rows": int(len(df)),
        "training_dates": int(len(cube.dates)),
        "config_file": raw["_path"],
        "config_sha256": raw["_sha256"],
    })
    print(json.dumps({"artifacts": str(args.out), "candidate": config.name,
                      "training_range": pipe.training_range, "train_seconds": round(seconds, 2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
