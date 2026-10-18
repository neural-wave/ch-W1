"""R-008c: one shared residual scale fitted on forward cross-fitted predictions of the training slice."""
import numpy as np

from aet.cube import QuarterCube
from aet.pipeline import PipelineConfig, ShapePipeline
from aet.schema import clean_quarter_hourly
from tests.synthetic import ZONES, synthetic_long


def test_forward_scale_is_fitted_stored_and_keeps_hourly_means(tmp_path):
    cube = QuarterCube.from_long(clean_quarter_hourly(synthetic_long("2026-01-01", "2026-02-28")), ZONES)
    train, day = cube.before(cube.dates[-1]), cube.dates[-1]
    pipe = ShapePipeline(PipelineConfig(num_boost_round=20, residual_scale=1.0, forward_scale={"blocks": 4}))
    pipe.fit(train)
    assert 0.0 <= pipe.fitted_scale <= 1.5

    H = cube.hourly()[-1]
    P, _ = pipe.predict_day(day, H, cube)
    np.testing.assert_allclose(P.mean(-1), H, atol=1e-9)

    pipe.save(tmp_path, train)
    loaded, _ = ShapePipeline.load(tmp_path)
    assert loaded.fitted_scale == pipe.fitted_scale
    np.testing.assert_allclose(loaded.predict_day(day, H, cube)[0], P)

    pipe.fitted_scale = 1.0                       # a unit scale leaves the learned residual unchanged
    plain = ShapePipeline(PipelineConfig(num_boost_round=20, residual_scale=1.0)).fit(train)
    np.testing.assert_allclose(pipe.predict_day(day, H, cube)[0], plain.predict_day(day, H, cube)[0])
