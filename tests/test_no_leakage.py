"""Leakage test for the whole feature pipeline (team + QB + injuries).

For a sample of weeks, erase everything that happens on/after that week - scores, results, team and
QB box scores, snap counts, and injury reports for LATER weeks - then rebuild all features. Pre-game
features for that week must not change. (The week's own injury report and announced starters are
legitimately known before kickoff, so they are kept.)

Run:  python -m tests.test_no_leakage   (or pytest)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data import (load_games, load_injuries, load_players, load_qb_games, load_snap_counts,
                            load_team_stats)
from nfl_model.features.injuries import INJURY_FEATURES
from nfl_model.features.pipeline import build_feature_table
from nfl_model.features.qb import QB_FEATURES
from nfl_model.features.team_features import MODEL_FEATURES_BASE

CHECK_COLS = MODEL_FEATURES_BASE + ["elo_prob", "qb_rating_h", "qb_rating_a"] + QB_FEATURES + INJURY_FEATURES
SAMPLES = [(2008, 1), (2012, 10), (2016, 4), (2019, 9), (2021, 17), (2023, 3), (2024, 12), (2025, 18), (2026, 3)]


def erase_from(tables: dict, season: int, week: int, cutoff: pd.Timestamp) -> dict:
    key = lambda d: d["season"] * 100 + d["week"]
    k0 = season * 100 + week
    g = tables["games"].copy()
    future = g["gameday"] >= cutoff
    for c in ["home_score", "away_score", "result", "total", "overtime"]:
        g.loc[future, c] = np.nan
    return {
        "games": g,
        "team_stats": tables["team_stats"][~tables["team_stats"]["game_id"].isin(g.loc[future, "game_id"])],
        "qb_games": tables["qb_games"][key(tables["qb_games"]) < k0],
        "snaps": tables["snaps"][key(tables["snaps"]) < k0],
        "injuries": tables["injuries"][key(tables["injuries"]) <= k0],   # same-week report is pre-game info
        "players": tables["players"],
    }


def run() -> int:
    tables = {"games": load_games(), "team_stats": load_team_stats(1999), "qb_games": load_qb_games(1999),
              "injuries": load_injuries(2009), "snaps": load_snap_counts(2013), "players": load_players()}
    full = build_feature_table(**tables).set_index("game_id")
    failures = 0
    for season, week in SAMPLES:
        wk = tables["games"][(tables["games"].season == season) & (tables["games"].week == week)]
        if wk.empty:
            print(f"skip {season} wk{week}: no games")
            continue
        trunc = build_feature_table(**erase_from(tables, season, week, wk["gameday"].min())).set_index("game_id")
        ids = wk["game_id"]
        a, b = full.loc[ids, CHECK_COLS].to_numpy(float), trunc.loc[ids, CHECK_COLS].to_numpy(float)
        ok = np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9)
        bad = [c for c, x, y in zip(CHECK_COLS, a.T, b.T) if not np.allclose(x, y, equal_nan=True, atol=1e-9)]
        print(f"{'PASS' if ok else 'FAIL'}  {season} week {week:>2}: {len(ids)} games, {len(CHECK_COLS)} features"
              + ("" if ok else f"  CHANGED: {bad}"))
        failures += (not ok)
    print("\nALL PASSED - no look-ahead detected" if not failures else f"\n{failures} FAILED")
    return failures


def test_no_leakage():
    assert run() == 0


def test_check_has_teeth():
    """Next week's features MUST change when this week's results are erased (otherwise the test proves nothing)."""
    tables = {"games": load_games(), "team_stats": load_team_stats(1999), "qb_games": load_qb_games(1999),
              "injuries": load_injuries(2009), "snaps": load_snap_counts(2013), "players": load_players()}
    full = build_feature_table(**tables).set_index("game_id")
    g = tables["games"]
    wk = g[(g.season == 2023) & (g.week == 6)]
    trunc = build_feature_table(**erase_from(tables, 2023, 6, wk["gameday"].min())).set_index("game_id")
    nxt = g[(g.season == 2023) & (g.week == 7)]["game_id"]
    for cols in (["elo_diff"], ["qb_diff"], INJURY_FEATURES):
        a, b = full.loc[nxt, cols].to_numpy(float), trunc.loc[nxt, cols].to_numpy(float)
        assert not np.allclose(a, b, equal_nan=True), f"{cols} unaffected by erased history - test has no teeth"


if __name__ == "__main__":
    rc = run()
    test_check_has_teeth()
    print("teeth check passed: later-week features do change when history is erased")
    raise SystemExit(rc)
