"""Injury-burden features (leak-free, 2013+).

For each team-week: sum over injured players of  P(misses game | report status) x importance,
split by position group. importance = the player's average snap share over his last 6 games
BEFORE this week (so a star out counts ~1, a backup ~0.2).

Report status is the final pre-game report. Constants come from 2013-2025 history of how often
players with each status actually took a snap (Out/Doubtful ~0%, Questionable ~58%, Probable ~64%
of listed players got a snap - the rest are special-teamers / inactive).
QBs are excluded here; they are handled by qb.py.

Caveat for live use: early-week reports are incomplete (Wed practice reports have no game status),
so run picks late in the week for the injury features to mean what they meant in training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .team_features import TEAM_CODE_FIX

P_MISS = {"Out": 1.0, "Doubtful": 1.0, "Questionable": 0.4, "Probable": 0.1}
DEFAULT_IMPORTANCE = 0.25      # no snap history (rookie / newly signed)
GROUPS = {
    "skill": {"RB", "FB", "WR", "TE"},
    "ol": {"T", "G", "C", "OL", "OT", "OG"},
    "front7": {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB"},
    "db": {"CB", "S", "SS", "FS", "DB"},
}
INJURY_FEATURES = [f"inj_{g}_diff" for g in GROUPS]


def _importance(injuries: pd.DataFrame, snaps: pd.DataFrame, players: pd.DataFrame) -> pd.Series:
    """Rolling-6-game snap share as of just before each injury row's week."""
    sn = snaps.merge(players, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    sn["share"] = sn[["offense_pct", "defense_pct"]].max(axis=1).fillna(0.0)
    sn["key"] = sn["season"] * 100 + sn["week"]
    sn = sn.sort_values(["gsis_id", "key"])
    sn["roll6"] = sn.groupby("gsis_id")["share"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    inj = injuries.assign(key=injuries["season"] * 100 + injuries["week"], _row=np.arange(len(injuries)))
    m = pd.merge_asof(inj.sort_values("key"), sn[["gsis_id", "key", "roll6"]].sort_values("key"),
                      on="key", by="gsis_id", direction="backward", allow_exact_matches=False, tolerance=120)
    return m.sort_values("_row")["roll6"].fillna(DEFAULT_IMPORTANCE).to_numpy()


def add_injury_features(out: pd.DataFrame, injuries: pd.DataFrame, snaps: pd.DataFrame,
                        players: pd.DataFrame) -> pd.DataFrame:
    inj = injuries.dropna(subset=["gsis_id"]).copy()
    inj["team"] = inj["team"].replace(TEAM_CODE_FIX)
    inj["p_miss"] = inj["report_status"].map(P_MISS).fillna(0.0)
    inj = inj[inj["p_miss"] > 0].reset_index(drop=True)
    inj["group"] = inj["position"].map({p: g for g, ps in GROUPS.items() for p in ps})
    inj = inj.dropna(subset=["group"]).reset_index(drop=True)
    inj["burden"] = inj["p_miss"].to_numpy() * _importance(inj, snaps, players)

    burden = inj.pivot_table(index=["season", "week", "team"], columns="group", values="burden",
                             aggfunc="sum", fill_value=0.0)
    burden = burden.reindex(columns=list(GROUPS), fill_value=0.0)
    # weeks for which any injury report exists at all (absence of a report is not "healthy")
    reported = set(zip(injuries["season"], injuries["week"]))
    has_snaps = snaps["season"].min() if len(snaps) else 9999

    res = out.copy()
    keys_h = pd.MultiIndex.from_frame(res[["season", "week", "home_team"]])
    keys_a = pd.MultiIndex.from_frame(res[["season", "week", "away_team"]])
    ok = np.array([(s, w) in reported and s > has_snaps for s, w in zip(res["season"], res["week"])])
    for g in GROUPS:
        h = burden[g].reindex(keys_h).fillna(0.0).to_numpy()
        a = burden[g].reindex(keys_a).fillna(0.0).to_numpy()
        res[f"inj_{g}_diff"] = np.where(ok, h - a, np.nan)    # + means home is MORE injured
    return res
