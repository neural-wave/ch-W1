"""Offline inference entry point for the jury. No network, no refitting.

    python -m aet.infer --hourly test_hourly.csv --history history_15min.csv \
                        --model artifacts_official --out predictions.csv

The jury's history is merged over the training prices stored in the artifacts
(jury values win), then fed to rules that were frozen at training time.
The output is validated before writing and again after reloading the CSV.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from .cube import QuarterCube
from .pipeline import ShapePipeline, flat_predictions
from .schema import read_hourly, read_quarter_hourly, validate_predictions

log = logging.getLogger("aet.infer")


def run(hourly_path, history_path, model_dir, out_path) -> dict:
    started = time.perf_counter()
    # A valid hourly reference is indispensable: do not invent prices or keys when it is invalid.
    hourly = read_hourly(hourly_path)
    candidate = "unavailable"
    try:
        pipe, stored = ShapePipeline.load(model_dir)
        candidate = pipe.config.name
        history = stored
        if history_path:
            jury = read_quarter_hourly(history_path, strict=False)
            history = stored.merge(QuarterCube.from_long(jury, pipe.zones))
        pred, diag = pipe.predict_frame(hourly, history)
        validate_predictions(pred, hourly)
    except Exception as exc:
        log.exception("reconstruction failed; writing the flat baseline so the output stays compliant")
        pred, diag = flat_predictions(hourly), {"emergency_flat": True, "failure_type": type(exc).__name__}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False, float_format="%.10f")
    report = validate_predictions(pd.read_csv(out_path), hourly)
    return {"output": str(out_path), "candidate": candidate, **report, **diag,
            "seconds": round(time.perf_counter() - started, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hourly", required=True, help="test_hourly.csv")
    parser.add_argument("--history", help="history_15min.csv (earlier dates only)")
    parser.add_argument("--model", required=True, help="artifact directory written by aet.train")
    parser.add_argument("--out", required=True, help="predictions CSV")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(run(args.hourly, args.history, args.model, args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
