"""Causality/constraint checks only; synthetic observations never select real models."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from aet.baselines import get_baseline
from aet.basis import encode, get_basis
from aet.cube import QuarterCube
from aet.features import FeatureConfig, build_features
from aet.history import HistoryStore
from aet.pipeline import PipelineConfig, ShapePipeline
from aet.schema import clean_quarter_hourly
from aet.volatility import ShapeRisk, risk_inputs
from tests.synthetic import ZONES, synthetic_long


def cube():
    return QuarterCube.from_long(clean_quarter_hourly(synthetic_long(
        "2026-01-01", "2026-03-31", skip=("2026-03-29",))), ZONES)


def features(c, day, H, enabled):
    basis = get_basis("poly")
    store = HistoryStore.build(c, "pchip", basis)
    base = get_baseline("pchip")(H)
    return build_features(np.array([day]), H, encode(base-H[..., None], basis), store,
                          FeatureConfig(volatility_history=enabled))


def test_absolute_magnitudes_do_not_cancel():
    coefficients = np.array([[[[2., -4., 6.]]], [[[-2., 4., -6.]]]])
    store = HistoryStore(np.datetime64("2026-01-01"), ("ZONE_0",),
                         np.ones((2, 1, 1)), coefficients)
    dates = np.array(["2026-01-03"], dtype="datetime64[D]")
    signed, count = store.window_mean("coeffs", dates, 7)
    magnitude, _ = store.window_mean("coeffs", dates, 7, absolute=True)
    np.testing.assert_allclose(signed, 0.)
    np.testing.assert_allclose(magnitude[0, 0, 0], [2., 4., 6.])
    assert (count == 2).all()


def test_magnitude_features_ignore_target_and_future_quarters():
    c = cube()
    day = np.datetime64("2026-03-15")
    H = c.hourly()[c.dates == day]
    poisoned = QuarterCube(c.dates, c.zones, np.where(
        (c.dates >= day)[:, None, None, None], 1e9, c.values))
    pd.testing.assert_frame_equal(features(c, day, H, True), features(poisoned, day, H, True))


def test_opt_in_features_preserve_default_feature_values():
    c = cube()
    day = np.datetime64("2026-03-15")
    H = c.hourly()[c.dates == day]
    old, new = features(c, day, H, False), features(c, day, H, True)
    assert not any(col.startswith("mean_abs") for col in old)
    assert "mean_abs7_c1" in new and "mean_range28" in new
    pd.testing.assert_frame_equal(old, new[old.columns])


def test_magnitude_windows_respect_calendar_gap_and_empty_history():
    c = cube()
    day = np.datetime64("2026-03-30")
    H = c.hourly()[c.dates == day]
    result = features(c, day, H, True)
    assert (result.n_range7 == 6).all()
    assert (result.n_abs7 == 6).all()
    assert result.mean_range7.notna().all()
    empty = features(QuarterCube.empty(ZONES), day, H, True)
    assert (empty.n_abs7 == 0).all()
    assert empty.mean_abs7_c1.isna().all()
    assert empty.mean_range7.isna().all()


def test_risk_reference_roundtrip_and_inference_causality(tmp_path):
    c = cube()
    config = PipelineConfig(residual_scale=.8)
    training = c.until("2026-02-28")
    risk = ShapeRisk.fit(training, config)
    path = tmp_path/"risk.json"
    risk.save(path)
    restored = ShapeRisk.load(path)
    day = np.datetime64("2026-03-15")
    H = c.hourly()[c.dates == day]
    store = HistoryStore.build(c, "pchip", get_basis("poly"))
    poisoned = QuarterCube(c.dates, c.zones, np.where(
        (c.dates >= day)[:, None, None, None], 1e9, c.values))
    other = HistoryStore.build(poisoned, "pchip", get_basis("poly"))
    x = risk_inputs(np.array([day]), H, store, config)
    y = risk_inputs(np.array([day]), H, other, config)
    pd.testing.assert_frame_equal(risk.predict(x), restored.predict(y))
    result = risk.predict(x)
    assert len(result) == 96
    assert set(result.risk_label) <= {"low", "medium", "high"}
    assert np.isfinite(result.risk_score).all()


def test_magnitude_pipeline_preserves_means_and_save_load(tmp_path):
    c = cube().until("2026-01-31")
    config = PipelineConfig(num_boost_round=8,
        model_params={"min_data_in_leaf": 10, "num_threads": 1},
        features=FeatureConfig(volatility_history=True))
    pipeline = ShapePipeline(config).fit(c)
    day = np.datetime64("2026-02-01")
    H = c.hourly()[-1]
    first, _ = pipeline.predict_day(day, H, c)
    np.testing.assert_allclose(first.mean(axis=-1), H, rtol=0, atol=1e-9)
    pipeline.save(tmp_path, c)
    loaded, history = ShapePipeline.load(tmp_path)
    second, _ = loaded.predict_day(day, H, history)
    np.testing.assert_array_equal(first, second)
