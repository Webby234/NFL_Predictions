"""Upcoming-game picks: train on everything finished, predict what's next, compare to the market."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRAIN_START
from ..data import load_games
from ..features import load_feature_table
from ..models import fit, model_zoo, predict_home_prob
from .edge import moneyline_candidates, select_bets, stake_units

DEFAULT_MODEL = "logit_with_spread"


def qb_name_map() -> dict[str, str]:
    """gsis id -> display name, from the schedule file."""
    g = load_games()
    m = {**dict(zip(g.home_qb_id, g.home_qb_name)), **dict(zip(g.away_qb_id, g.away_qb_name))}
    return {k: v for k, v in m.items() if isinstance(k, str)}


def add_qb_names(games: pd.DataFrame) -> pd.DataFrame:
    names = qb_name_map()
    g = games.copy()
    g["home_qb"] = g["qb_id_h"].map(names) if "qb_id_h" in g else None
    g["away_qb"] = g["qb_id_a"].map(names) if "qb_id_a" in g else None
    for side in ("home", "away"):
        changed = g[f"qb_changed_{'h' if side == 'home' else 'a'}"] == 1
        g[f"{side}_qb"] = g[f"{side}_qb"].fillna("unknown").where(~changed, g[f"{side}_qb"].fillna("unknown") + " (new)")
    return g


def upcoming_games(df: pd.DataFrame, week: int | None = None) -> pd.DataFrame:
    """Unplayed regular-season games of the current season that already have lines."""
    up = df[(df.game_type == "REG") & df.home_score.isna() & df.home_moneyline.notna() & df.spread_line.notna()]
    if up.empty:
        return up
    season = up.season.min()
    up = up[up.season == season]
    week = int(week if week is not None else up.week.min())
    return up[up.week == week].copy()


def predict_upcoming(df: pd.DataFrame, games: pd.DataFrame, model: str = DEFAULT_MODEL,
                     gbm_kind: str = "sklearn") -> pd.DataFrame:
    """Fit on all finished regular-season games (features are pre-game, so this is leak-free)."""
    spec = model_zoo(gbm_kind)[model]
    train = df[(df.game_type == "REG") & df.home_win.notna() & (df.season >= TRAIN_START)]
    train = train[train.season >= spec.train_start]
    train = train.dropna(subset=["spread_line"]) if "spread_line" in spec.features else train
    g = games.copy()
    g["p_home"] = predict_home_prob(fit(spec, train), spec, g)
    return add_qb_names(g)


def picks_table(games_with_p: pd.DataFrame, min_edge: float = 0.03, bankroll: float = 100.0,
                kelly_fraction: float = 0.25, cap: float = 0.05,
                live: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (every game with both sides, recommended bets only).

    live: optional output of live_odds.best_and_consensus; replaces the schedule-file prices with
    best-available prices and uses the consensus fair probability as the market estimate."""
    g = games_with_p.copy()
    if live is not None and not live.empty:
        lv = live[["home_team", "away_team", "home_moneyline", "home_book", "away_moneyline", "away_book",
                   "market_home_prob", "n_books"]].rename(columns={"home_moneyline": "live_home_ml",
                                                                   "away_moneyline": "live_away_ml"})
        g = g.merge(lv, on=["home_team", "away_team"], how="left")
        # live best prices where available, schedule-file prices otherwise
        g["home_moneyline"] = g["live_home_ml"].fillna(g["home_moneyline"])
        g["away_moneyline"] = g["live_away_ml"].fillna(g["away_moneyline"])
    cands = moneyline_candidates(g, "p_home")
    bets = select_bets(cands, min_edge=min_edge)
    if not bets.empty:
        bets["stake"] = stake_units(bets["kelly"], bankroll, kelly_fraction, cap).round(2)
    return cands, bets
