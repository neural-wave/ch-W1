"""R-011: zones cleared at the same hourly price share one intra-hour shape."""
import numpy as np

from aet.pipeline import PipelineConfig, ShapePipeline


def test_tie_shares_shapes_of_coupled_zones_only_and_keeps_means():
    rng = np.random.default_rng(0)
    H = rng.normal(80, 20, size=(4, 24))
    H[1, :12] = H[0, :12]                       # zones 0-1 coupled in hours 1-12
    H[2, 5] = H[1, 5]                           # 0-1-2 in hour 6 (component through zone 1)
    dev = rng.normal(size=(4, 24, 4))
    P = H[..., None] + dev - dev.mean(axis=-1, keepdims=True)
    pipe = ShapePipeline(PipelineConfig(model="none", couple_tol=1e-6))
    T = pipe._tie(P, H)
    np.testing.assert_allclose(T.mean(-1), H, atol=1e-9)
    D, D0 = T - H[..., None], P - H[..., None]
    np.testing.assert_allclose(D[0, :12], D[1, :12])
    np.testing.assert_allclose(D[2, 5], D[0, 5])
    np.testing.assert_allclose(D[0, 5], D0[:3, 5].mean(axis=0))
    np.testing.assert_allclose(D[3], D0[3])     # never coupled: unchanged
    np.testing.assert_allclose(D[:2, 12:], D0[:2, 12:])
