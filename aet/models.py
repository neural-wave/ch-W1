"""Residual models: interchangeable strategies mapping features to a zero-sum
intra-hour residual of shape [n, 4]."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .basis import center, decode, encode
from .features import CATEGORICAL
from .schema import QUARTERS

DEFAULT_LGBM_PARAMS = {
    "objective": "regression_l1",   # the official metric is MAE
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 100,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l2": 0.0,
    "seed": 42,
    "deterministic": True,          # with force_row_wise: repeatable training on one machine
    "force_row_wise": True,
    "num_threads": 4,
    "verbosity": -1,
}


class ResidualModel(ABC):
    kind = "abstract"

    def __init__(self, params: dict | None = None, num_boost_round: int = 500):
        self.params = dict(params or {})
        self.num_boost_round = int(num_boost_round)

    @abstractmethod
    def fit(self, X: pd.DataFrame, R: np.ndarray, basis: np.ndarray, weight: np.ndarray | None = None) -> "ResidualModel":
        """R: [n, 4] zero-sum residual targets (observed minus baseline)."""

    @abstractmethod
    def predict(self, X: pd.DataFrame, basis: np.ndarray) -> np.ndarray:
        """[n, 4] zero-sum residuals."""

    def save(self, directory: Path) -> dict:
        return {}

    @classmethod
    def load(cls, directory: Path, meta: dict, params: dict, num_boost_round: int) -> "ResidualModel":
        return cls(params, num_boost_round)


class NoResidual(ResidualModel):
    """Baseline only (E0/E1)."""

    kind = "none"

    def fit(self, X, R, basis, weight=None):
        return self

    def predict(self, X, basis):
        return np.zeros((len(X), QUARTERS))


class _LightGBMModel(ResidualModel):
    def __init__(self, params=None, num_boost_round=500):
        super().__init__({**DEFAULT_LGBM_PARAMS, **(params or {})}, num_boost_round)
        self.boosters: list[lgb.Booster] = []
        self.feature_names: list[str] = []

    def _train(self, X: pd.DataFrame, y: np.ndarray, weight, **overrides) -> lgb.Booster:
        data = lgb.Dataset(X, label=y, weight=weight, categorical_feature=CATEGORICAL, free_raw_data=True)
        return lgb.train({**self.params, **overrides}, data, num_boost_round=self.num_boost_round)

    def save(self, directory):
        files = []
        for i, booster in enumerate(self.boosters):
            name = f"{self.kind}_{i}.txt"
            booster.save_model(str(Path(directory) / name))
            files.append(name)
        return {"files": files, "feature_names": self.feature_names}

    @classmethod
    def load(cls, directory, meta, params, num_boost_round):
        model = cls(params, num_boost_round)
        model.boosters = [lgb.Booster(model_file=str(Path(directory) / f)) for f in meta["files"]]
        model.feature_names = list(meta["feature_names"])
        return model


class LGBMCoefficientModel(_LightGBMModel):
    """One L1 regressor per basis coefficient; decoding is zero-sum by construction.
    Coefficient L1 is a surrogate for price L1, so selection uses reconstructed prices.
    `n_components` in model_params learns only the leading basis components (others = 0)."""

    kind = "lgbm_coef"

    def __init__(self, params=None, num_boost_round=500):
        params = dict(params or {})
        self.n_components = params.pop("n_components", None)
        super().__init__(params, num_boost_round)

    def fit(self, X, R, basis, weight=None):
        self.feature_names = list(X.columns)
        C = encode(R, basis)
        n = C.shape[1] if self.n_components is None else int(self.n_components)
        self.boosters = [self._train(X, C[:, j], weight) for j in range(n)]
        return self

    def predict(self, X, basis):
        X = X[self.feature_names]
        C = np.zeros((len(X), basis.shape[1]))
        for j, booster in enumerate(self.boosters):
            C[:, j] = booster.predict(X)
        return decode(C, basis)


class LGBMQuarterModel(_LightGBMModel):
    """One pooled L1 regressor over the four quarters (loss in price space), then centring.
    Natural stepping stone to the constrained-quantile variant (same rows, other alphas)."""

    kind = "lgbm_quarter"

    @staticmethod
    def _stack(X: pd.DataFrame) -> pd.DataFrame:
        stacked = pd.concat([X] * QUARTERS, ignore_index=True)
        stacked["quarter"] = np.repeat(np.arange(QUARTERS), len(X))
        return stacked

    def fit(self, X, R, basis, weight=None):
        self.feature_names = list(X.columns)
        y = R.T.reshape(-1)                       # quarter-major, matching _stack
        w = None if weight is None else np.tile(weight, QUARTERS)
        self.boosters = [self._train(self._stack(X), y, w)]
        return self

    def predict(self, X, basis):
        raw = self.boosters[0].predict(self._stack(X[self.feature_names]))
        return center(raw.reshape(QUARTERS, len(X)).T)


def common_quantile(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constrained L1 point decision from conditional quantiles (roadmap section E).

    Q: [n, 4, L] predicted quantiles of the four quarter residuals at increasing levels.
    Minimising sum_q E|R_q - a_q| subject to sum_q a_q = 0 gives a_q = Q_q(tau) with one common
    tau such that sum_q Q_q(tau) = 0. Quantiles are sorted (crossing repair) and interpolated
    linearly between levels, so the sum is piecewise linear in tau and the root is exact.
    Rows whose summed curve does not bracket zero fall back to the centred medians.
    Returns (a [n, 4], bracketed [n] bool).
    """
    Q = np.sort(Q, axis=-1)
    S = Q.sum(axis=1)                                             # [n, L], non-decreasing in tau
    levels = Q.shape[-1]
    ok = np.isfinite(S).all(axis=1) & (S[:, 0] <= 0) & (S[:, -1] >= 0)
    lo = np.clip(np.argmax(S >= 0, axis=1) - 1, 0, levels - 2)
    rows = np.arange(len(Q))
    s_lo, s_hi = S[rows, lo], S[rows, lo + 1]
    gap = s_hi - s_lo
    t = np.clip(np.divide(-s_lo, gap, out=np.zeros_like(gap), where=gap > 0), 0.0, 1.0)
    a = Q[rows, :, lo] + t[:, None] * (Q[rows, :, lo + 1] - Q[rows, :, lo])
    median = center(Q[:, :, levels // 2])
    return np.where(ok[:, None], a, median), ok


class LGBMQuantileModel(LGBMQuarterModel):
    """E6 option: five pooled quantile regressors over the stacked quarters, combined with the
    common-tau decision above. `last_bracketed` is False for rows that used the median fallback."""

    kind = "lgbm_quantile"
    ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)

    def fit(self, X, R, basis, weight=None):
        self.feature_names = list(X.columns)
        stacked, y = self._stack(X), R.T.reshape(-1)
        w = None if weight is None else np.tile(weight, QUARTERS)
        self.boosters = [self._train(stacked, y, w, objective="quantile", alpha=a) for a in self.ALPHAS]
        return self

    def predict(self, X, basis):
        stacked = self._stack(X[self.feature_names])
        Q = np.stack([b.predict(stacked).reshape(QUARTERS, len(X)).T for b in self.boosters], axis=-1)
        a, self.last_bracketed = common_quantile(Q)
        return center(a)


MODELS: dict[str, type[ResidualModel]] = {
    cls.kind: cls for cls in (NoResidual, LGBMCoefficientModel, LGBMQuarterModel, LGBMQuantileModel)
}
