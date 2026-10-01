"""Do QB and injury features add anything? Same-window paired comparisons.

PRE-DECLARED RULE: a feature set counts as an improvement only if its paired log-loss difference against
the same-window baseline is below -2 standard errors. Everything else is reported but not claimed.
Test seasons start at ABLATION_FIRST_TEST so every model (incl. injury ones, trained from 2014) has
>= 3 seasons of training data before its first test season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ABLATION_FIRST_TEST
from ..models import ablation_zoo
from .backtest import walk_forward
from .metrics import EPS, paired_logloss_diff, score

# (candidate, baseline) pairs; candidate - baseline < 0 means the added features help
COMPARISONS = [
    ("qb_no_market", "base_no_market", "+QB (train 2006+)"),
    ("qb_no_market_w14", "base_no_market_w14", "+QB (train 2014+)"),
    ("qb_inj_no_market_w14", "qb_no_market_w14", "+injuries on top of QB (2014+)"),
    ("qb_inj_no_market_w14", "base_no_market_w14", "+QB+injuries (2014+)"),
    ("qb_with_spread", "base_with_spread", "+QB, spread included (2006+)"),
    ("qb_with_spread_w14", "base_with_spread_w14", "+QB, spread included (2014+)"),
    ("qb_inj_with_spread_w14", "qb_with_spread_w14", "+injuries on top of QB, spread included"),
    ("qb_inj_with_spread_w14", "base_with_spread_w14", "+QB+injuries, spread included (2014+)"),
]


def run_ablation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zoo = ablation_zoo()
    out = walk_forward(df, zoo=zoo, first_test=ABLATION_FIRST_TEST)
    y = out["home_win"].to_numpy(float)
    names = list(zoo) + ["elo", "market_ml"]
    rows = []
    for n in names:
        p = out[f"p_{n}"].to_numpy(float)
        d, se = paired_logloss_diff(y, p, out["p_market_ml"].to_numpy(float))
        rows.append({"model": n, **score(y, p), "ll_vs_market": d, "ll_vs_market_se": se})
    models = pd.DataFrame(rows).set_index("model")

    comps = []
    for cand, base, label in COMPARISONS:
        d, se = paired_logloss_diff(y, out[f"p_{cand}"].to_numpy(float), out[f"p_{base}"].to_numpy(float))
        comps.append({"comparison": label, "candidate": cand, "baseline": base, "ll_diff": d, "se": se,
                      "z": d / se, "verdict": "IMPROVES" if d < -2 * se else ("worse" if d > 2 * se else "no clear change")})
    return models, pd.DataFrame(comps), out
