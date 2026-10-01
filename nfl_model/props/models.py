"""Distributional models for a count-like stat (QB passing yards).

mean model (ridge / gradient boosting) + heteroscedastic spread (sigma = c0 + c1*mean, fit on
|residual|) + empirical standardized-residual shape. Gives quantiles and P(stat > line).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def make_mean_model(kind: str):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30))
    if kind == "gbm":
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=200,
                                                           min_samples_leaf=60, l2_regularization=5.0))
    raise ValueError(kind)


@dataclass
class DistModel:
    feats: list
    mean_model: object
    c0: float
    c1: float
    z: np.ndarray          # sorted standardized residuals

    def mean(self, X: pd.DataFrame) -> np.ndarray:
        return self.mean_model.predict(X[self.feats])

    def sigma(self, mean: np.ndarray) -> np.ndarray:
        return np.maximum(self.c0 + self.c1 * mean, 15.0)

    def quantile(self, mean: np.ndarray, q: float) -> np.ndarray:
        return mean + self.sigma(mean) * np.quantile(self.z, q)

    def prob_over(self, mean: np.ndarray, line) -> np.ndarray:
        """P(stat > line); half-integer lines make pushes impossible, whole-number lines split mass at the line."""
        zl = (np.asarray(line, float) - mean) / self.sigma(mean)
        return 1.0 - np.searchsorted(self.z, zl, side="right") / len(self.z)


def fit_dist(train: pd.DataFrame, feats: list, target: str = "passing_yards", kind: str = "ridge") -> DistModel:
    m = make_mean_model(kind).fit(train[feats], train[target])
    mu = m.predict(train[feats]); res = train[target].values - mu
    A = np.column_stack([np.ones(len(mu)), mu]); c, *_ = np.linalg.lstsq(A, np.abs(res) * np.sqrt(np.pi / 2), rcond=None)
    dm = DistModel(feats, m, float(c[0]), float(c[1]), np.array([0.0]))
    z = np.sort(res / dm.sigma(mu)); dm.z = z
    return dm


def _prob_under(self: DistModel, mean: np.ndarray, line) -> np.ndarray:
    """P(stat < line)."""
    zl = (np.asarray(line, float) - mean) / self.sigma(mean)
    return np.searchsorted(self.z, zl, side="left") / len(self.z)


DistModel.prob_under = _prob_under
