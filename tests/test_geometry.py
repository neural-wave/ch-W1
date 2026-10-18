import numpy as np
import pandas as pd

from aet.baselines import flat, get_baseline
from aet.basis import get_basis, encode
from aet.cube import QuarterCube
from aet.features import FeatureConfig, build_features
from aet.history import HistoryStore


def build(H, config):
    dates = np.array(["2026-01-02"], dtype="datetime64[D]")
    store = HistoryStore.build(QuarterCube.empty(("ZONE_0",)), "pchip", get_basis("poly"))
    coeff = encode(get_baseline("pchip")(H)-H[..., None], get_basis("poly"))
    return build_features(dates, H, coeff, store, config)


def test_scaled_ramp_preserves_means_and_doubles_shape():
    H = np.linspace(-100, 100, 24)[None, None, :]
    once, twice = get_baseline("linear")(H), get_baseline("linear_x2")(H)
    np.testing.assert_allclose(twice.mean(axis=-1), H, atol=1e-12)
    np.testing.assert_allclose(twice-flat(H), 2*(once-flat(H)), atol=1e-12)


def test_geometry_preserves_missing_neighbours_and_marks_reversal():
    H = np.zeros((1, 1, 24))
    H[0, 0, :3] = [0, 10, 0]
    f = build(H, FeatureConfig(local_geometry=True, extrema_distance=True))
    assert np.isnan(f.turning_point.iloc[0]) and np.isnan(f.turning_point.iloc[-1])
    assert f.turning_point.iloc[1] == 1
    assert f.local_movement.iloc[1] == 10
    assert np.isclose(f.ramp_asymmetry.iloc[1], -1)
    assert f.dist_to_peak.iloc[1] == 0
    assert f.dist_to_trough.iloc[1] == 1  # first of tied troughs


def test_flat_curve_does_not_explode_or_invent_unique_extrema():
    f = build(np.zeros((1, 1, 24)), FeatureConfig(local_geometry=True, extrema_distance=True))
    assert (f.norm_movement.dropna() == 0).all()
    assert (f.turning_point.iloc[1:-1] == 0).all()
    assert f.dist_to_peak.isna().all() and f.dist_to_trough.isna().all()


def test_disabled_features_leave_existing_feature_values_unchanged():
    H = np.linspace(-10, 20, 24)[None, None, :]
    old = build(H, FeatureConfig())
    new = build(H, FeatureConfig(local_geometry=True, extrema_distance=True))
    assert "local_movement" not in old
    pd.testing.assert_frame_equal(old, new[old.columns])
