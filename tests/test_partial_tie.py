"""R-017: nearly coupled zones move a fraction alpha toward their cluster-mean shape."""
import numpy as np

from aet.cube import QuarterCube
from aet.pipeline import PipelineConfig, ShapePipeline
from aet.schema import clean_quarter_hourly
from tests.synthetic import ZONES, synthetic_long


def test_partial_tie_interpolates_between_no_tie_and_full_tie_and_keeps_means():
    rng = np.random.default_rng(1)
    H = rng.normal(80, 20, size=(4, 24))
    H[1] = H[0] + 2.0                           # zones 0-1 within tol 3 in every hour
    dev = rng.normal(size=(4, 24, 4))
    P = H[..., None] + dev - dev.mean(axis=-1, keepdims=True)
    pipe = ShapePipeline(PipelineConfig(model="none"))
    np.testing.assert_allclose(pipe._tie(P, H, 3.0, 0.0), P)
    full = pipe._tie(P, H, 3.0, 1.0)
    half = pipe._tie(P, H, 3.0, 0.4)
    np.testing.assert_allclose(half.mean(-1), H, atol=1e-9)
    np.testing.assert_allclose(half, P + 0.4 * (full - P), atol=1e-9)
    D = full - H[..., None]
    np.testing.assert_allclose(D[0], D[1])


def test_partial_tie_runs_in_predict_day_and_keeps_means():
    cube = QuarterCube.from_long(clean_quarter_hourly(synthetic_long("2026-01-01", "2026-01-06")), ZONES)
    pipe = ShapePipeline(PipelineConfig(model="none", partial_tie={"tol": 50.0, "alpha": 0.5})).fit(cube)
    plain = ShapePipeline(PipelineConfig(model="none")).fit(cube)
    H = cube.hourly()[-1]
    P, _ = pipe.predict_day(cube.dates[-1], H, cube)
    P0, _ = plain.predict_day(cube.dates[-1], H, cube)
    np.testing.assert_allclose(P.mean(-1), H, atol=1e-9)
    np.testing.assert_allclose(P, plain._tie(P0, H, 50.0, 0.5), atol=1e-9)
    assert not np.allclose(P, P0)
