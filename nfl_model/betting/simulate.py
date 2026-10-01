"""Historical betting simulation on walk-forward predictions.

Bets are placed at the CLOSING moneyline (the only historical price we have) on the side
with the best expected value, using only probabilities produced by models that never saw
the game. Closing lines are the sharpest prices available, so this is a demanding test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .edge import moneyline_candidates, select_bets, stake_units


def settle(bets: pd.DataFrame) -> pd.DataFrame:
    """Add win flag and 1-unit profit to selected bets."""
    b = bets.copy()
    b["won"] = np.where(b["side"] == "home", b["home_win"] == 1, b["home_win"] == 0)
    b["profit"] = np.where(b["won"], b["decimal_odds"] - 1.0, -1.0)
    return b


def summarize_bets(b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    if b.empty:
        return {"bets": 0}
    prof = b["profit"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(prof, size=(n_boot, len(prof)), replace=True).mean(axis=1)
    return {
        "bets": len(b),
        "win_rate": b["won"].mean(),
        "avg_decimal_odds": b["decimal_odds"].mean(),
        "avg_model_prob": b["model_prob"].mean(),
        "units": prof.sum(),
        "roi": prof.mean(),
        "roi_ci_low": float(np.percentile(boots, 2.5)),
        "roi_ci_high": float(np.percentile(boots, 97.5)),
    }


def threshold_table(out: pd.DataFrame, p_col: str, edges=(0.0, 0.02, 0.04, 0.06, 0.08)) -> pd.DataFrame:
    """Flat 1-unit bets on the best side of each game whenever edge >= threshold (and EV > 0)."""
    cands = moneyline_candidates(out, p_col)
    rows = []
    for e in edges:
        bets = settle(select_bets(cands, min_edge=e))
        rows.append({"min_edge": e, **summarize_bets(bets)})
    return pd.DataFrame(rows)


def naive_strategies(out: pd.DataFrame) -> pd.DataFrame:
    """Context rows: what blind betting costs. ROI of these ~= -(bookmaker vig)."""
    cands = moneyline_candidates(out.assign(_p=0.5), "_p")
    cands["is_fav"] = cands["moneyline"] < 0
    cands["home_win"] = cands["home_win"].astype(float)
    rows = {}
    for name, sel in (("bet_every_home_team", cands.side == "home"),
                      ("bet_every_favorite", cands.is_fav),
                      ("bet_every_underdog", ~cands.is_fav)):
        rows[name] = summarize_bets(settle(cands[sel]))
    return pd.DataFrame(rows).T


def kelly_simulation(out: pd.DataFrame, p_col: str, min_edge: float, bankroll0: float = 100.0,
                     fraction: float = 0.25, cap: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Sequential bankroll simulation with fractional Kelly staking (games settled by date)."""
    bets = settle(select_bets(moneyline_candidates(out, p_col), min_edge=min_edge)).sort_values("gameday")
    bank, curve = bankroll0, []
    for r in bets.itertuples():
        stake = float(stake_units(pd.Series([r.kelly]), bank, fraction, cap).iloc[0])
        bank += stake * (r.decimal_odds - 1.0) if r.won else -stake
        curve.append(bank)
    bets["bankroll"] = curve
    peak = pd.Series(curve).cummax() if curve else pd.Series(dtype=float)
    stats = {"bets": len(bets), "final_bankroll": bank if curve else bankroll0,
             "max_drawdown": float(((peak - pd.Series(curve)) / peak).max()) if curve else 0.0}
    return bets, stats
