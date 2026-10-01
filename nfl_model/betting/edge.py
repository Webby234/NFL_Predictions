"""Turn model probabilities + market prices into bets: edge, expected value, stake size."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .odds import american_to_decimal, devig_two_way


def moneyline_candidates(df: pd.DataFrame, p_home_col: str) -> pd.DataFrame:
    """Two candidate bets per game (home side, away side) with edge and EV.

    Needs columns: home_moneyline, away_moneyline and `p_home_col` (model P(home wins)).
      market_prob = vig-free probability implied by the two prices (or `market_home_prob` if supplied)
      edge        = model_prob - market_prob        (probability points)
      ev          = model_prob * decimal_odds - 1   (expected profit per 1 staked, at the price offered)
      kelly       = full-Kelly fraction of bankroll ( = ev / (decimal_odds - 1) ), floored at 0
    """
    d = df.copy()
    mh, ma = devig_two_way(d["home_moneyline"], d["away_moneyline"])
    if "market_home_prob" in d.columns:     # e.g. consensus across books (live odds) overrides same-price devig
        mh = d["market_home_prob"].fillna(pd.Series(np.asarray(mh), index=d.index))
        ma = 1 - mh
    sides = []
    for side, ml_col, p_model, p_mkt, team_col, opp_col in (
        ("home", "home_moneyline", d[p_home_col], mh, "home_team", "away_team"),
        ("away", "away_moneyline", 1 - d[p_home_col], ma, "away_team", "home_team"),
    ):
        x = d.copy()
        x["side"], x["team"], x["opponent"] = side, x[team_col], x[opp_col]
        x["moneyline"] = x[ml_col]
        x["model_prob"], x["market_prob"] = p_model.to_numpy(), np.asarray(p_mkt)
        x["decimal_odds"] = american_to_decimal(x["moneyline"])
        x["edge"] = x["model_prob"] - x["market_prob"]
        x["ev"] = x["model_prob"] * x["decimal_odds"] - 1.0
        x["kelly"] = (x["ev"] / (x["decimal_odds"] - 1.0)).clip(lower=0)
        sides.append(x)
    return pd.concat(sides, ignore_index=True)


def select_bets(cands: pd.DataFrame, min_edge: float = 0.0, min_ev: float = 0.0) -> pd.DataFrame:
    """Best side per game, kept only if it clears BOTH thresholds.
    min_edge is in probability points (0.03 = model is 3 points above the vig-free market)."""
    c = cands[(cands["edge"] >= min_edge) & (cands["ev"] > min_ev)]
    if c.empty:
        return c
    best = c.sort_values("ev", ascending=False).groupby("game_id", sort=False).head(1)
    return best.sort_values("ev", ascending=False).reset_index(drop=True)


def stake_units(kelly: pd.Series, bankroll: float = 100.0, fraction: float = 0.25, cap: float = 0.05) -> pd.Series:
    """Fractional Kelly stake in bankroll units, capped per bet (full Kelly is far too aggressive
    when the probability estimate itself is noisy)."""
    return (kelly * fraction).clip(upper=cap) * bankroll
