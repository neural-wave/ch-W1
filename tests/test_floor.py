"""R-016: in hours whose mean is above the price floor, no quarter is predicted below it."""
import numpy as np

from aet.pipeline import PipelineConfig, ShapePipeline


def _bisection(p, h, floor):
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.maximum(p - mid, floor).mean() > h else (lo, mid)
    return np.maximum(p - (lo + hi) / 2, floor)


def test_floor_projection_keeps_means_bound_and_untouched_hours():
    rng = np.random.default_rng(0)
    H = rng.uniform(-5, 10, size=(4, 24))
    H[0, 0] = -0.01                                # at the floor: unchanged
    dev = rng.normal(scale=8, size=(4, 24, 4))
    P = H[..., None] + dev - dev.mean(axis=-1, keepdims=True)
    P[1, 3] = np.nan                               # non-finite hours are left to the fallback chain
    F = ShapePipeline._floor(P, H, -0.01)
    ok = np.isfinite(P).all(-1)
    np.testing.assert_allclose(F[ok].mean(-1), H[ok], atol=1e-9)
    act = ok & (H > -0.01) & (P.min(-1) < -0.01)
    assert act.sum() > 10
    assert (F[act] >= -0.01 - 1e-12).all()
    np.testing.assert_array_equal(F[~act & ok], P[~act & ok])
    assert np.isnan(F[1, 3]).all()
    for z, h in zip(*np.nonzero(act)):
        np.testing.assert_allclose(F[z, h], _bisection(P[z, h], H[z, h], -0.01), atol=1e-8)


def test_floor_and_ceiling_in_predict_day_bound_both_sides_and_keep_means():
    H = np.linspace(-60, 60, 24)[None] + np.array([[0.0], [2.1], [-2.1], [1.5]])
    ramp = np.array([-3.0, -1.0, 1.0, 3.0])         # zero-sum shape that crosses zero near H = 0

    def predict(**bounds):
        pipe = ShapePipeline(PipelineConfig(model="none", **bounds))
        pipe.baseline = lambda h: h[..., None] + ramp
        return pipe.predict_day("2026-01-05", H, None)[0]

    P0, P = predict(), predict(price_floor=-0.01, price_ceiling=0.0)
    np.testing.assert_allclose(P.mean(-1), H, atol=1e-9)
    assert (P[H > -0.01] >= -0.01 - 1e-9).all()
    assert (P[H < 0.0] <= 1e-9).all()
    assert ((P0[H > -0.01] < -0.01).any() and (P0[H < 0.0] > 0.0).any())   # both bounds were active
    np.testing.assert_allclose(P[np.abs(H) > 29], P0[np.abs(H) > 29])
