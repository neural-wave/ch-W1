"""E6 options: constrained common-quantile decision and cross-zone graph shrinkage."""
import numpy as np
from scipy.optimize import minimize

from aet.cube import QuarterCube
from aet.models import common_quantile
from aet.pipeline import PipelineConfig, ShapePipeline
from aet.schema import clean_quarter_hourly
from tests.synthetic import ZONES, synthetic_long

LEVELS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def test_common_quantile_is_an_exact_root_with_a_shared_tau():
    rng = np.random.default_rng(0)
    Q = np.sort(rng.normal(size=(200, 4, 5)) * 5 + rng.normal(size=(200, 4, 1)), axis=-1)
    a, ok = common_quantile(Q)
    np.testing.assert_allclose(a[ok].sum(axis=1), 0.0, atol=1e-9)
    i = np.flatnonzero(ok)[0]                       # all four quarters sit at the same tau
    taus = [np.interp(a[i, q], Q[i, q], LEVELS) for q in range(4)]
    assert np.ptp(taus) < 1e-9


def test_common_quantile_matches_numerical_constrained_l1():
    # Piecewise-uniform marginals defined by the five quantiles: the expected L1 is convex and
    # its constrained minimum must coincide with the common-tau rule.
    Q = np.array([[[-6, -3, -1, 1, 2], [-2, -1, 0, 1, 3], [-1, 0, 1, 2, 5], [-4, -2, 0, 2, 4]]], float)
    a, ok = common_quantile(Q)
    grid = np.linspace(0.1, 0.9, 4001)
    samples = np.stack([np.interp(grid, LEVELS, Q[0, q]) for q in range(4)])

    def loss(x):
        return np.abs(samples - x[:, None]).mean(axis=1).sum()

    res = minimize(loss, np.zeros(4), constraints={"type": "eq", "fun": lambda x: x.sum()}, method="SLSQP")
    assert ok[0]
    np.testing.assert_allclose(a[0], res.x, atol=2e-2)


def test_common_quantile_repairs_crossings_and_falls_back_without_bracket():
    crossed = np.array([[[1.0, -1.0, 0.0, 2.0, -2.0]] * 4])        # unsorted levels
    a, ok = common_quantile(crossed)
    assert ok[0] and np.isclose(a.sum(), 0.0)
    positive = np.full((1, 4, 5), 1.0) + np.arange(4)[None, :, None]   # sum > 0 at every level
    a, ok = common_quantile(positive)
    assert not ok[0]
    np.testing.assert_allclose(a[0], [-1.5, -0.5, 0.5, 1.5])          # centred medians


def _cube():
    return QuarterCube.from_long(clean_quarter_hourly(synthetic_long("2026-01-01", "2026-02-28")), ZONES)


def test_graph_shrinkage_keeps_means_and_pulls_shapes_together():
    cube = _cube()
    day, H = cube.dates[-1], cube.hourly()[-1]
    plain = ShapePipeline(PipelineConfig(model="none")).fit(cube)
    zero = ShapePipeline(PipelineConfig(model="none", shrinkage={"lambda": 0.0})).fit(cube)
    strong = ShapePipeline(PipelineConfig(model="none", shrinkage={"lambda": 50.0, "floor": 1e6})).fit(cube)
    P0, _ = plain.predict_day(day, H, cube)
    Pz, _ = zero.predict_day(day, H, cube)
    Ps, _ = strong.predict_day(day, H, cube)
    np.testing.assert_allclose(Pz, P0)
    np.testing.assert_allclose(Ps.mean(-1), H, atol=1e-9)
    dev0, devs = P0 - H[..., None], Ps - H[..., None]
    spread = lambda d: np.abs(d - d.mean(axis=0, keepdims=True)).mean()
    assert spread(devs) < 0.1 * spread(dev0)                          # huge scale + lambda: near consensus
    np.testing.assert_allclose(devs.mean(axis=0), dev0.mean(axis=0), atol=1e-9)   # L conserves the zone sum
