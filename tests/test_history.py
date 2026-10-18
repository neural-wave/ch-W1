import numpy as np
import pandas as pd

from aet.baselines import get_baseline
from aet.basis import encode, get_basis
from aet.cube import QuarterCube
from aet.features import FeatureConfig, build_features
from aet.history import HistoryStore
from aet.schema import clean_quarter_hourly
from tests.synthetic import ZONES, synthetic_long

B = get_basis("poly")


def _cube(skip=()):
    return QuarterCube.from_long(clean_quarter_hourly(synthetic_long("2026-01-01", "2026-03-31", skip=skip)), ZONES)


def _features(history: QuarterCube, day, H):
    store = HistoryStore.build(history, "pchip", B)
    base_coeffs = encode(get_baseline("pchip")(H) - H[..., None], B)
    return build_features(np.array([day], dtype="datetime64[D]"), H, base_coeffs, store, FeatureConfig())


def test_future_sentinel_rows_cannot_change_features():
    cube = _cube()
    day = np.datetime64("2026-03-01")
    H = cube.hourly()[cube.dates == day]                       # the allowed same-day input
    poisoned = QuarterCube(cube.dates, cube.zones,
                           np.where((cube.dates >= day)[:, None, None, None], 1e6, cube.values))
    pd.testing.assert_frame_equal(_features(cube, day, H), _features(poisoned, day, H))


def test_calendar_lags_skip_excluded_dates_instead_of_shifting_rows():
    store = HistoryStore.build(_cube(skip=("2026-03-29",)), "pchip", B)
    after_gap = np.array(["2026-03-30"], dtype="datetime64[D]")
    assert np.isnan(store.lag("coeffs", after_gap, 1)).all()           # not 28 March
    assert np.isfinite(store.lag("coeffs", after_gap, 7)).all()        # 23 March exists
    mean, count = store.window_mean("coeffs", after_gap, 7)
    assert (count == 6).all() and np.isfinite(mean).all()


def test_store_beyond_its_last_date_still_serves_recent_lags():
    store = HistoryStore.build(_cube(), "pchip", B)
    later = np.array(["2026-04-01", "2026-05-01"], dtype="datetime64[D]")
    lag1 = store.lag("hourly", later, 1)
    assert np.isfinite(lag1[0]).all() and np.isnan(lag1[1]).all()
