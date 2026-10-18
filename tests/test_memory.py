"""E4 analogue memory: causality, frozen scaler, missing history and persistence."""
import numpy as np
import pandas as pd

from aet.basis import decode, get_basis
from aet.cube import QuarterCube
from aet.history import AnalogueMemory, HistoryStore, curve_descriptors
from aet.pipeline import PipelineConfig, ShapePipeline
from aet.schema import clean_quarter_hourly
from tests.synthetic import ZONES, synthetic_long

B = get_basis("poly")


def _cube(start="2025-10-01", end="2026-03-31", skip=()):
    return QuarterCube.from_long(clean_quarter_hourly(synthetic_long(start, end, skip=skip)), ZONES)


def _memory(cube, **kw):
    return AnalogueMemory.fit(HistoryStore.build(cube, "pchip", B), **kw)


def test_weekend_flag_matches_calendar():
    dates = pd.date_range("2026-03-02", periods=7).to_numpy().astype("datetime64[D]")   # Monday..Sunday
    desc = curve_descriptors(dates, np.zeros((7, 1, 24)), np.zeros((7, 1, 24)))
    assert desc[:, 0, 0, -1].tolist() == [0, 0, 0, 0, 0, 1, 1]


def test_future_sentinel_rows_cannot_change_memory():
    cube = _cube()
    memory = _memory(cube.until("2026-01-31"))
    day = np.datetime64("2026-03-01")
    H = cube.hourly()[cube.dates == day]
    poisoned = QuarterCube(cube.dates, cube.zones,
                           np.where((cube.dates >= day)[:, None, None, None], 1e6, cube.values))
    clean = memory.lookup(HistoryStore.build(cube, "pchip", B), [day], H)
    dirty = memory.lookup(HistoryStore.build(poisoned, "pchip", B), [day], H)
    for name in clean:
        np.testing.assert_array_equal(clean[name], dirty[name])


def test_neighbours_are_strictly_earlier_and_inside_the_window():
    cube = _cube()
    memory = _memory(cube, k=8, window_days=30)
    day = cube.dates[-1]
    out = memory.lookup(HistoryStore.build(cube, "pchip", B), [day], cube.hourly()[-1:], explain=(0, 2, 10))
    picked = np.array(out["explain"]["dates"], dtype="datetime64[D]")
    assert len(picked) == 8 and (picked < day).all() and (picked >= day - 30).all()
    assert np.isclose(sum(out["explain"]["weights"]), 1.0)
    assert (out["mem_count"] == 30).all() and (out["mem_eff"] <= 8 + 1e-9).all()
    # training-time queries on the same store: the first date has nothing earlier
    train = memory.lookup(HistoryStore.build(cube, "pchip", B), cube.dates[:2], cube.hourly()[:2])
    assert (train["mem_count"][0] == 0).all() and (train["mem_count"][1] == 1).all()


def test_missing_history_gives_zero_shape_and_zero_count():
    cube = _cube()
    memory = _memory(cube)
    empty = HistoryStore.build(QuarterCube.empty(ZONES), "pchip", B)
    out = memory.lookup(empty, cube.dates[-1:], cube.hourly()[-1:])
    assert (out["mem_count"] == 0).all() and np.isnan(out["mem_dist"]).all()
    coeffs = np.stack([out[f"mem_c{j}"] for j in (1, 2, 3)], axis=-1)
    assert (coeffs == 0).all()
    np.testing.assert_allclose(decode(coeffs, B).sum(-1), 0.0)


def test_memory_pipeline_round_trip(tmp_path):
    cube = _cube()
    train = cube.until("2026-02-28")
    cfg = PipelineConfig(name="mem", num_boost_round=30, memory={"k": 8, "window_days": 60},
                         model_params={"min_data_in_leaf": 50, "num_threads": 2})
    pipe = ShapePipeline(cfg).fit(train)
    pipe.save(tmp_path, train)
    loaded, stored = ShapePipeline.load(tmp_path)
    np.testing.assert_array_equal(loaded.memory.scale, pipe.memory.scale)
    day = np.datetime64("2026-03-10")
    H = cube.hourly()[cube.dates == day][0]
    a, _ = pipe.predict_day(day, H, cube)
    b, _ = loaded.predict_day(day, H, stored.merge(cube.before(day)))
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(a.mean(-1), H, atol=1e-9)
