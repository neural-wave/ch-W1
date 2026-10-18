"""Offline train -> infer smoke run on synthetic data (wiring checks, not performance claims)."""
import filecmp

import numpy as np
import pandas as pd
import pytest

from aet import backtest, infer, train
from aet.cube import QuarterCube
from aet.pipeline import ShapePipeline
from aet.schema import read_hourly, validate_predictions
from tests.synthetic import ZONES, hourly_from, synthetic_long

CONFIG = """
seed: 7
zones: [ZONE_0, ZONE_1, ZONE_2, ZONE_3]
folds:
  scheme: explicit
  list:
    - {name: f_jan, train_end: 2025-12-31, score_start: 2026-01-19, score_end: 2026-01-31}
    - {name: f_feb, train_end: 2026-01-31, score_start: 2026-02-19, score_end: 2026-02-28}
defaults:
  baseline: pchip
  model: lgbm_coef
  num_boost_round: 80
  model_params: {min_data_in_leaf: 50, num_threads: 2}
candidates:
  - {name: flat, baseline: flat, model: none}
  - {name: pchip, model: none}
  - {name: smoke, final: true}
  - {name: smoke_quarter, model: lgbm_quarter}
  - {name: smoke_memory, memory: {k: 8, window_days: 60}}
  - {name: smoke_quantile, model: lgbm_quantile}
  - {name: smoke_shrink, shrinkage: {lambda: 0.3}}
"""


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    full = synthetic_long("2025-10-01", "2026-02-28", skip=("2025-10-26",))
    paths = {k: root / f"{k}.csv" for k in ("train", "history", "hourly", "hourly_multi", "truth")}
    full[full["delivery_date"] <= "2026-01-31"].to_csv(paths["train"], index=False)
    full[full["delivery_date"] < "2026-02-20"].to_csv(paths["history"], index=False)
    target = full[full["delivery_date"] == "2026-02-20"]
    target.to_csv(paths["truth"], index=False)
    hourly_from(target).to_csv(paths["hourly"], index=False)
    multi = hourly_from(full[full["delivery_date"].isin(["2026-02-20", "2026-02-21"])])
    extra = hourly_from(target[target["market_id"] == "ZONE_0"]).assign(market_id="ZONE_9")  # unseen zone
    pd.concat([multi, extra]).to_csv(paths["hourly_multi"], index=False)
    (root / "config.yaml").write_text(CONFIG)
    paths.update(root=root, config=root / "config.yaml", artifacts=root / "artifacts")
    assert train.main(["--train", str(paths["train"]), "--config", str(paths["config"]),
                       "--out", str(paths["artifacts"])]) == 0
    return paths


def _infer(ws, out, hourly="hourly", history=True):
    args = ["--hourly", str(ws[hourly]), "--model", str(ws["artifacts"]), "--out", str(ws["root"] / out)]
    if history:
        args += ["--history", str(ws["history"])]
    assert infer.main(args) == 0
    return pd.read_csv(ws["root"] / out)


def test_inference_is_compliant_and_deterministic(ws):
    first = _infer(ws, "pred_a.csv")
    _infer(ws, "pred_b.csv")
    report = validate_predictions(first, read_hourly(ws["hourly"]))
    assert report["max_abs_mean_error"] < 1e-6
    assert filecmp.cmp(ws["root"] / "pred_a.csv", ws["root"] / "pred_b.csv", shallow=False)


def test_inference_without_jury_history_and_with_extra_dates_and_zones(ws):
    no_history = _infer(ws, "pred_nohist.csv", history=False)
    validate_predictions(no_history, read_hourly(ws["hourly"]))
    multi = _infer(ws, "pred_multi.csv", hourly="hourly_multi")
    validate_predictions(multi, read_hourly(ws["hourly_multi"]))
    assert set(multi["market_id"]) == set(ZONES) | {"ZONE_9"}


def test_learned_shape_beats_flat_on_synthetic_sawtooth(ws):
    pred = _infer(ws, "pred_score.csv")
    truth = pd.read_csv(ws["truth"])
    merged = pred.merge(truth, on=["delivery_date", "market_id", "hour_index", "quarter_index"])
    hourly = merged.groupby(["market_id", "hour_index"])["price_eur_mwh"].transform("mean")
    learned = (merged["predicted_price_eur_mwh"] - merged["price_eur_mwh"]).abs().mean()
    flat = (hourly - merged["price_eur_mwh"]).abs().mean()
    assert learned < 0.8 * flat


def test_prediction_ignores_future_history(ws):
    pipe, stored = ShapePipeline.load(ws["artifacts"])
    day = np.datetime64("2026-01-20")
    H = stored.hourly()[stored.dates == day][0]
    poisoned = QuarterCube(stored.dates, stored.zones,
                           np.where((stored.dates >= day)[:, None, None, None], -1e6, stored.values))
    clean, _ = pipe.predict_day(day, H, stored)
    dirty, _ = pipe.predict_day(day, H, poisoned)
    np.testing.assert_array_equal(clean, dirty)


def test_fallback_chain_keeps_output_valid(ws, monkeypatch):
    pipe, stored = ShapePipeline.load(ws["artifacts"])
    day = np.datetime64("2026-01-20")
    H = stored.hourly()[stored.dates == day][0]
    monkeypatch.setattr(pipe.model, "predict", lambda X, basis: np.full((len(X), 4), np.nan))
    P, info = pipe.predict_day(day, H, stored)
    np.testing.assert_allclose(P, pipe.baseline(H[None])[0], atol=1e-9)    # learned -> baseline
    assert info["fallback_hours"].sum() == 4 * 24
    monkeypatch.setattr(pipe, "baseline", lambda h: np.full(h.shape + (4,), np.nan))
    P, info = pipe.predict_day(day, H, stored)
    np.testing.assert_allclose(P, np.repeat(H[..., None], 4, axis=-1))    # baseline -> flat
    assert info["flat_hours"].sum() == 4 * 24


def test_backtest_runs_and_ranks_candidates(ws, capsys):
    out = ws["root"] / "results"
    assert backtest.main(["--train", str(ws["train"]), "--config", str(ws["config"]), "--out", str(out)]) == 0
    summary = pd.read_csv(next(out.glob("*/development_summary.csv"))).set_index("candidate")
    assert summary.loc["smoke", "score"] < summary.loc["flat", "score"]
    assert summary["max_mean_err"].max() < 1e-9
    assert summary.loc["smoke_quantile", "bracket_fail_hours"] >= 0
    assert summary.loc[["smoke_memory", "smoke_quantile", "smoke_shrink"], "score"].lt(summary.loc["flat", "score"]).all()
    assert (out / "ledger.jsonl").exists()
