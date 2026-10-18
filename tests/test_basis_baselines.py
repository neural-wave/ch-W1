import numpy as np
import pytest

from aet.baselines import BASELINES, get_baseline
from aet.basis import center, decode, encode, get_basis


@pytest.mark.parametrize("name", ["poly", "haar"])
def test_basis_is_orthonormal_and_zero_sum(name):
    B = get_basis(name)
    assert B.shape == (4, 3)
    np.testing.assert_allclose(B.T @ B, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(B.sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("name", ["poly", "haar"])
def test_decode_of_encode_is_the_centred_residual(name):
    B = get_basis(name)
    r = np.random.default_rng(0).normal(size=(50, 4))
    np.testing.assert_allclose(decode(encode(r, B), B), center(r), atol=1e-12)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_preserve_hourly_mean_with_negative_prices(name):
    H = np.random.default_rng(1).normal(40, 80, size=(5, 3, 24))   # plenty of negative hours
    P = get_baseline(name)(H)
    assert P.shape == (5, 3, 24, 4)
    assert np.isfinite(P).all()
    np.testing.assert_allclose(P.mean(axis=-1), H, atol=1e-9)


def test_cumulative_interpolation_of_a_constant_day_is_flat():
    np.testing.assert_allclose(get_baseline("pchip")(np.full((1, 24), 55.0)), 55.0, atol=1e-9)


def test_missing_hour_only_invalidates_its_own_day():
    H = np.full((2, 24), 50.0)
    H[0, 5] = np.nan
    P = get_baseline("pchip")(H)
    assert np.isnan(P[0]).all() and np.isfinite(P[1]).all()


def test_coefficient_model_can_learn_only_leading_components():
    import pandas as pd

    from aet.models import LGBMCoefficientModel

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"zone": rng.integers(0, 4, 400).astype(np.int32), "x": rng.normal(size=400)})
    R = np.outer(X["x"], [-3, -1, 1, 3]) + rng.normal(size=(400, 4))
    R -= R.mean(axis=1, keepdims=True)
    basis = get_basis("poly")
    model = LGBMCoefficientModel({"n_components": 1, "num_threads": 1, "min_data_in_leaf": 20}, 20).fit(X, R, basis)
    out = model.predict(X, basis)
    assert len(model.boosters) == 1 and "n_components" not in model.params
    np.testing.assert_allclose(out @ basis[:, 1:], 0.0, atol=1e-9)      # curvature and zigzag stay zero
    np.testing.assert_allclose(out.sum(axis=1), 0.0, atol=1e-9)
