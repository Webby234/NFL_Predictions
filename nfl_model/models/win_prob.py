"""Win-probability models. Each spec = (feature list, estimator factory).

Names are referenced by the backtest, the betting simulator and the app:
  logit_no_market    pure team-strength model, knows nothing about betting lines
  gbm_no_market      same features, gradient boosting
  logit_no_market_qb adds current-starting-QB features (significantly better than logit_no_market)
  logit_with_spread  same features + closing spread (can only add to the market)
  gbm_with_spread    same, gradient boosting
  market_spread      spread alone mapped to a probability (benchmark)
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import INJURY_TRAIN_START, TRAIN_START
from ..features import INJURY_FEATURES, NO_MARKET, QB_FEATURES, WITH_SPREAD


class ModelSpec(NamedTuple):
    features: list[str]
    factory: Callable[[], object]
    train_start: int = TRAIN_START      # injury features only exist from 2014 (snap counts start 2013)


def make_logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))


def make_gbm(kind: str = "sklearn"):
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss"))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def model_zoo(gbm_kind: str = "sklearn") -> dict[str, ModelSpec]:
    return {
        "logit_no_market": ModelSpec(NO_MARKET, make_logit),
        "gbm_no_market": ModelSpec(NO_MARKET, lambda: make_gbm(gbm_kind)),
        "logit_no_market_qb": ModelSpec(NO_MARKET + QB_FEATURES, make_logit),
        "logit_with_spread": ModelSpec(WITH_SPREAD, make_logit),
        "gbm_with_spread": ModelSpec(WITH_SPREAD, lambda: make_gbm(gbm_kind)),
        "market_spread": ModelSpec(["spread_line"], make_logit),
    }


def fit(spec: ModelSpec, train: pd.DataFrame):
    mdl = spec.factory()
    mdl.fit(train[spec.features], train["home_win"].astype(int))
    return mdl


def predict_home_prob(mdl, spec: ModelSpec, games: pd.DataFrame):
    return mdl.predict_proba(games[spec.features])[:, 1]


def ablation_zoo() -> dict[str, ModelSpec]:
    """Logistic models adding QB then injury features, each paired with a same-window baseline.

    *_w14 models are trained from INJURY_TRAIN_START so they are comparable with the injury models.
    """
    zoo = {}
    for flavor, base in (("no_market", NO_MARKET), ("with_spread", WITH_SPREAD)):
        zoo[f"base_{flavor}"] = ModelSpec(base, make_logit, TRAIN_START)
        zoo[f"base_{flavor}_w14"] = ModelSpec(base, make_logit, INJURY_TRAIN_START)
        zoo[f"qb_{flavor}"] = ModelSpec(base + QB_FEATURES, make_logit, TRAIN_START)
        zoo[f"qb_{flavor}_w14"] = ModelSpec(base + QB_FEATURES, make_logit, INJURY_TRAIN_START)
        zoo[f"qb_inj_{flavor}_w14"] = ModelSpec(base + QB_FEATURES + INJURY_FEATURES, make_logit, INJURY_TRAIN_START)
    return zoo
