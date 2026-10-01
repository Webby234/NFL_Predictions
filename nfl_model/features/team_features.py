"""Leak-free pre-game features.

Every feature for a game is computed from games that finished BEFORE it.
The builder walks games in date order, takes a snapshot of each team's state
(Elo + exponentially-weighted stats) *before* the game, and only then updates
the state with the game's result. Future/unplayed games get a snapshot but no
update, so the same code produces features for upcoming games.

`test_no_leakage.py` verifies this by blanking out a week's results and
confirming the features for that week do not change.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---- Elo settings (FiveThirtyEight-style) -----------------------------------
ELO_START = 1500.0
ELO_K = 20.0
ELO_HFA = 48.0          # home-field advantage in Elo points
ELO_REVERT = 0.25       # fraction regressed to the mean each new season

# Franchise relocations: map old codes to the modern one so Elo/EWMA carry over.
TEAM_CODE_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}

# Per-team per-game metrics we keep an exponentially-weighted average of.
METRICS = ["pts", "yds", "epa", "giveaways"]          # "for" side; "against" = opponent's
MODEL_FEATURES_BASE = (
    ["elo_diff"]
    + [f"d_{m}_{side}" for m in METRICS for side in ("for", "against")]
    + ["rest_diff", "div_game", "neutral", "min_games"]
)


def _team_game_table(team_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team) with the raw offensive numbers we need."""
    t = team_stats.dropna(subset=["team"]).copy()
    t["team"] = t["team"].replace(TEAM_CODE_FIX)
    for c in ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
              "passing_interceptions", "sack_yards_lost", "sacks_suffered"]:
        t[c] = t[c].fillna(0)
    t["yds"] = t["passing_yards"].fillna(0) - t["sack_yards_lost"] + t["rushing_yards"].fillna(0)
    plays = t["attempts"].fillna(0) + t["carries"].fillna(0) + t["sacks_suffered"]
    epa = t["passing_epa"].fillna(0) + t["rushing_epa"].fillna(0)
    t["epa"] = np.where(plays > 0, epa / plays.replace(0, np.nan), np.nan)
    t["giveaways"] = (t["passing_interceptions"] + t["sack_fumbles_lost"]
                      + t["rushing_fumbles_lost"] + t["receiving_fumbles_lost"])
    return t[["game_id", "team", "yds", "epa", "giveaways"]]


def _elo_expected(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def build_features(games: pd.DataFrame, team_stats: pd.DataFrame,
                   halflife: float = 6.0) -> pd.DataFrame:
    """Return one row per game with pre-game features (and outcomes/odds).

    games      : nflverse games.csv (may include unplayed games, NaN scores)
    team_stats : nflverse weekly team stats (rows for unplayed games absent)
    halflife   : EWMA half-life in games (chosen a priori, not tuned on results)
    """
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    tg = _team_game_table(team_stats).set_index(["game_id", "team"])

    g = games.copy()
    g["home_team"] = g["home_team"].replace(TEAM_CODE_FIX)
    g["away_team"] = g["away_team"].replace(TEAM_CODE_FIX)
    g = g.sort_values(["gameday", "gametime", "game_id"], na_position="last").reset_index(drop=True)

    elo: dict[str, float] = {}
    last_season: dict[str, int] = {}
    n_games: dict[str, int] = {}
    # ewma[team][metric_side] -> float
    ewma: dict[str, dict[str, float]] = {}

    def state_for(team: str, season: int):
        # season rollover: regress Elo toward the mean once per team per season
        if team not in elo:
            elo[team], n_games[team], ewma[team] = ELO_START, 0, {}
        if last_season.get(team) is not None and season != last_season[team]:
            elo[team] = elo[team] * (1 - ELO_REVERT) + ELO_START * ELO_REVERT
        last_season[team] = season

    rows = []
    for r in g.itertuples(index=False):
        h, a, season = r.home_team, r.away_team, int(r.season)
        state_for(h, season)
        state_for(a, season)
        neutral = 1 if getattr(r, "location", "Home") == "Neutral" else 0
        hfa = 0.0 if neutral else ELO_HFA

        # ---------- SNAPSHOT (pre-game) ----------
        row = {"game_id": r.game_id, "elo_h": elo[h], "elo_a": elo[a]}
        row["elo_diff"] = elo[h] - elo[a]
        row["elo_prob"] = _elo_expected(elo[h] - elo[a] + hfa)
        for m in METRICS:
            for side in ("for", "against"):
                key = f"{m}_{side}"
                hv = ewma[h].get(key, np.nan)
                av = ewma[a].get(key, np.nan)
                row[f"h_{key}"], row[f"a_{key}"] = hv, av
                row[f"d_{key}"] = hv - av
        row["min_games"] = min(n_games[h], n_games[a])

        # ---------- UPDATE (only if the game has been played) ----------
        played = pd.notna(r.home_score) and pd.notna(r.away_score)
        if played:
            hs, as_ = float(r.home_score), float(r.away_score)
            obs = {}
            for team, opp, pf, pa in ((h, a, hs, as_), (a, h, as_, hs)):
                o = {"pts_for": pf, "pts_against": pa}
                mine = tg.loc[(r.game_id, team)] if (r.game_id, team) in tg.index else None
                theirs = tg.loc[(r.game_id, opp)] if (r.game_id, opp) in tg.index else None
                for m in ("yds", "epa", "giveaways"):
                    o[f"{m}_for"] = float(mine[m]) if mine is not None and pd.notna(mine[m]) else np.nan
                    o[f"{m}_against"] = float(theirs[m]) if theirs is not None and pd.notna(theirs[m]) else np.nan
                obs[team] = o
            for team in (h, a):
                for k, v in obs[team].items():
                    if np.isnan(v):
                        continue
                    prev = ewma[team].get(k)
                    ewma[team][k] = v if prev is None else alpha * v + (1 - alpha) * prev
                n_games[team] += 1
            # Elo update with margin-of-victory multiplier
            mov = hs - as_
            exp_h = _elo_expected(elo[h] - elo[a] + hfa)
            actual_h = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
            winner_diff = (elo[h] - elo[a] + hfa) if mov > 0 else (elo[a] - elo[h] - hfa)
            mult = math.log(abs(mov) + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2) if mov != 0 else 1.0
            delta = ELO_K * mult * (actual_h - exp_h)
            elo[h] += delta
            elo[a] -= delta
        rows.append(row)

    feats = pd.DataFrame(rows)
    out = g.merge(feats, on="game_id", how="left")

    # Non-rolling pre-game info known before kickoff
    out["rest_diff"] = (out["home_rest"] - out["away_rest"]).clip(-7, 7)
    out["neutral"] = (out.get("location", "Home") == "Neutral").astype(int)
    out["div_game"] = out["div_game"].fillna(0).astype(int)

    # Outcomes and market-implied probabilities (benchmarks, not model inputs
    # unless a feature set explicitly includes them)
    out["home_win"] = np.where(out["result"].isna(), np.nan,
                               np.where(out["result"] > 0, 1.0, np.where(out["result"] < 0, 0.0, np.nan)))
    imp_h, imp_a = _implied(out["home_moneyline"]), _implied(out["away_moneyline"])
    out["market_ml_prob"] = imp_h / (imp_h + imp_a)      # vig removed (proportional)
    return out


def _implied(ml: pd.Series) -> pd.Series:
    """American moneyline -> implied probability (still includes the vig)."""
    ml = pd.to_numeric(ml, errors="coerce")
    p = np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))
    return pd.Series(p, index=ml.index).where(ml.notna())
