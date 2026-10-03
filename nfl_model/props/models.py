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
    if kind == "blend":
        return Blend(make_mean_model("ridge"), make_mean_model("gbm"))
    raise ValueError(kind)


class Blend:
    """Average of a linear model and a boosted-tree model. On 2019-2025 the average beat either one alone
    for every prop: the linear half is steady, the boosted half picks up interactions such as role changes."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def fit(self, X, y):
        self.a.fit(X, y); self.b.fit(X, y)
        return self

    def predict(self, X):
        return 0.5 * self.a.predict(X) + 0.5 * self.b.predict(X)

    def predict_proba(self, X):
        return 0.5 * self.a.predict_proba(X) + 0.5 * self.b.predict_proba(X)


@dataclass
class DistModel:
    feats: list
    mean_model: object
    c0: float
    c1: float
    z: np.ndarray          # sorted standardized residuals
    floor: float = 15.0    # minimum sigma

    def mean(self, X: pd.DataFrame) -> np.ndarray:
        return self.mean_model.predict(X[self.feats])

    def sigma(self, mean: np.ndarray) -> np.ndarray:
        return np.maximum(self.c0 + self.c1 * mean, self.floor)

    def quantile(self, mean: np.ndarray, q: float) -> np.ndarray:
        return mean + self.sigma(mean) * np.quantile(self.z, q)

    def prob_over(self, mean: np.ndarray, line) -> np.ndarray:
        """P(stat > line); half-integer lines make pushes impossible, whole-number lines split mass at the line."""
        zl = (np.asarray(line, float) - mean) / self.sigma(mean)
        return 1.0 - np.searchsorted(self.z, zl, side="right") / len(self.z)


def fit_dist(train: pd.DataFrame, feats: list, target: str = "passing_yards", kind: str = "ridge",
             floor: float = 15.0) -> DistModel:
    m = make_mean_model(kind).fit(train[feats], train[target])
    mu = m.predict(train[feats]); res = train[target].values - mu
    A = np.column_stack([np.ones(len(mu)), mu]); c, *_ = np.linalg.lstsq(A, np.abs(res) * np.sqrt(np.pi / 2), rcond=None)
    dm = DistModel(feats, m, float(c[0]), float(c[1]), np.array([0.0]), floor)
    z = np.sort(res / dm.sigma(mu)); dm.z = z
    return dm


def _prob_under(self: DistModel, mean: np.ndarray, line) -> np.ndarray:
    """P(stat < line)."""
    zl = (np.asarray(line, float) - mean) / self.sigma(mean)
    return np.searchsorted(self.z, zl, side="left") / len(self.z)


DistModel.prob_under = _prob_under


# ---------------------------------------------------------------- count and binary models ----
from scipy.stats import nbinom, poisson                           # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier       # noqa: E402
from sklearn.linear_model import LogisticRegression, PoissonRegressor  # noqa: E402


@dataclass
class CountModel:
    """Negative-binomial counts: log-link Poisson regression for the mean, dispersion from training."""
    feats: list
    model: object
    disp: float            # var = mu + disp * mu^2

    def mean(self, X):
        return self.model.predict(X[self.feats])

    def _dist(self, mu):
        mu = np.maximum(mu, 1e-6)
        if self.disp < 1e-6:
            return poisson(mu)
        n = 1.0 / self.disp
        return nbinom(n, n / (n + mu))

    def prob_over(self, mu, line):
        return self._dist(np.asarray(mu)).sf(np.floor(np.asarray(line, float)))

    def prob_under(self, mu, line):
        return self._dist(np.asarray(mu)).cdf(np.ceil(np.asarray(line, float)) - 1)

    def pmf(self, mu, k):
        return self._dist(np.asarray(mu)).pmf(k)


def fit_count(train, feats, target, kind="poisson") -> CountModel:
    lin = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
    gbm = lambda: make_pipeline(SimpleImputer(strategy="median"),
                                HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.04, max_iter=200,
                                                              min_samples_leaf=60, l2_regularization=5.0))
    m = lin() if kind == "poisson" else Blend(lin(), gbm()) if kind == "blend" else gbm()
    m.fit(train[feats], train[target])
    mu = m.predict(train[feats]); y = train[target].values
    disp = max(0.0, float(np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2)))
    return CountModel(feats, m, disp)


@dataclass
class BinaryModel:
    feats: list
    model: object

    def prob(self, X):
        return self.model.predict_proba(X[self.feats])[:, 1]


def fit_binary(train, feats, target, kind="logit") -> BinaryModel:
    lin = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
    gbm = lambda: make_pipeline(SimpleImputer(strategy="median"),
                                HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=200,
                                                               min_samples_leaf=80, l2_regularization=5.0))
    m = lin() if kind == "logit" else Blend(lin(), gbm()) if kind == "blend" else gbm()
    m.fit(train[feats], train[target])
    return BinaryModel(feats, m)
