"""Hand-checkable tests for the QB and injury features. Run: python -m tests.test_features (or pytest)"""
import numpy as np
import pandas as pd

from nfl_model.features.injuries import DEFAULT_IMPORTANCE, add_injury_features
from nfl_model.features.qb import QB_PRIOR, QB_PRIOR_STRENGTH, add_qb_features


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, (a, b)


def games(rows):
    cols = ["season", "week", "game_type", "home_team", "away_team", "home_score", "away_score", "home_qb_id", "away_qb_id"]
    return pd.DataFrame(rows, columns=cols)


def qbg(rows):
    return pd.DataFrame(rows, columns=["player_id", "season", "week", "season_type", "team", "attempts",
                                       "sacks_suffered", "passing_epa"])


def test_rating_formula_and_new_starter_flag():
    g = games([(2020, 1, "REG", "AAA", "BBB", 20, 10, "a1", "b1"),
               (2020, 2, "REG", "AAA", "CCC", 20, 10, "a1", "c1"),
               (2020, 3, "REG", "AAA", "BBB", 20, 10, "a2", "b1")])
    q = qbg([("a1", 2020, 1, "REG", "AAA", 30, 0, 6.0), ("b1", 2020, 1, "REG", "BBB", 30, 0, -3.0),
             ("a1", 2020, 2, "REG", "AAA", 30, 0, 6.0), ("c1", 2020, 2, "REG", "CCC", 30, 0, 0.0),
             ("a2", 2020, 3, "REG", "AAA", 10, 0, -5.0), ("b1", 2020, 3, "REG", "BBB", 30, 0, 0.0)])
    r = add_qb_features(g, q)
    prior_only = QB_PRIOR                                           # week 1: nobody has history
    approx(r.qb_rating_h[0], prior_only)
    wk2 = (6.0 + QB_PRIOR_STRENGTH * QB_PRIOR) / (30 + QB_PRIOR_STRENGTH)   # a1 after one game, hand formula
    approx(r.qb_rating_h[1], wk2)
    assert r.qb_changed_h[1] == 0 and r.qb_changed_h[2] == 1        # a2 is a new starter in week 3
    # week 3: a2 has no history -> prior; drop = prior - a1's rating after two games
    decay = 0.5 ** (1 / 24)
    a1_after2 = ((6.0 * decay + 6.0) + QB_PRIOR_STRENGTH * QB_PRIOR) / ((30 * decay + 30) + QB_PRIOR_STRENGTH)
    away_zero_drop = 0.0
    b1_rating = r.qb_rating_a[2]
    expected_drop_home = QB_PRIOR - a1_after2
    approx(r.qb_drop_diff[2], expected_drop_home - away_zero_drop)
    assert r.qb_exp_diff[2] < 0                                      # a2 has 0 prior starts vs b1's 2


def test_unannounced_starter_replaced_when_injured():
    g = games([(2020, 1, "REG", "AAA", "BBB", 20, 10, "a1", "b1"),
               (2020, 2, "REG", "AAA", "BBB", 20, 10, "a2", "b1"),
               (2020, 3, "REG", "AAA", "BBB", np.nan, np.nan, np.nan, np.nan)])   # not announced yet
    q = qbg([("a1", 2020, 1, "REG", "AAA", 30, 0, 0.0), ("b1", 2020, 1, "REG", "BBB", 30, 0, 0.0),
             ("a2", 2020, 2, "REG", "AAA", 30, 0, 0.0), ("b1", 2020, 2, "REG", "BBB", 30, 0, 0.0)])
    inj_healthy = pd.DataFrame(columns=["season", "week", "team", "gsis_id", "position", "report_status"])
    assert add_qb_features(g, q, inj_healthy).qb_id_h[2] == "a2"     # last starter carries forward
    inj = pd.DataFrame([(2020, 3, "AAA", "a2", "QB", "Out")],
                       columns=["season", "week", "team", "gsis_id", "position", "report_status"])
    assert add_qb_features(g, q, inj).qb_id_h[2] == "a1"             # a2 out -> most recent other QB


def test_same_week_games_do_not_see_each_other():
    """QB x leaves T2 for T1; T2's feature must not depend on x's same-week game with T1 (the 2008 Favre bug)."""
    rows = [(2020, 1, "REG", "T2", "T3", 20, 10, "x", "z"),
            (2020, 2, "REG", "T1", "T4", 20, 10, "x", "w"),      # x now plays for T1 (earlier in the week's order)
            (2020, 2, "REG", "T2", "T3", 20, 10, "y", "z")]      # T2 starts new QB y; prev starter x
    q = qbg([("x", 2020, 1, "REG", "T2", 30, 0, 9.0), ("z", 2020, 1, "REG", "T3", 30, 0, 0.0),
             ("x", 2020, 2, "REG", "T1", 30, 0, -9.0), ("w", 2020, 2, "REG", "T4", 30, 0, 0.0),
             ("y", 2020, 2, "REG", "T2", 30, 0, 0.0), ("z", 2020, 2, "REG", "T3", 30, 0, 0.0)])
    full = add_qb_features(games(rows), q)
    rows_unplayed = [rows[0], (2020, 2, "REG", "T1", "T4", np.nan, np.nan, "x", "w"), rows[2]]
    part = add_qb_features(games(rows_unplayed), q)
    approx(full.qb_drop_diff[2], part.qb_drop_diff[2])
    approx(full.qb_rating_h[2], part.qb_rating_h[2])


def test_injury_burden_weights_and_no_lookahead():
    out = pd.DataFrame({"season": [2020] * 4, "week": [1, 2, 3, 4], "home_team": ["AAA"] * 4, "away_team": ["BBB"] * 4})
    players = pd.DataFrame({"gsis_id": ["g1", "g2", "g3"], "pfr_id": ["p1", "p2", "p3"]})
    snaps = pd.DataFrame(
        [(2019, 17, "AAA", "p1", 0.9, 0.0)] + [(2020, w, "AAA", "p1", 0.9, 0.0) for w in (1, 2, 3)]
        + [(2020, 4, "AAA", "p1", 0.1, 0.0),                       # SAME-week snap share must be ignored
           (2020, 3, "BBB", "p3", 0.0, 0.7)],
        columns=["season", "week", "team", "pfr_player_id", "offense_pct", "defense_pct"])
    inj = pd.DataFrame([
        (2020, 4, "AAA", "g1", "T", "Out"),            # OL starter out:   1.0 * 0.9
        (2020, 4, "AAA", "g2", "WR", "Questionable"),  # no history:       0.4 * DEFAULT_IMPORTANCE
        (2020, 4, "BBB", "g3", "CB", "Doubtful"),      # DB, 0.7 share:    1.0 * 0.7
        (2020, 4, "AAA", "gq", "QB", "Out"),           # QBs handled elsewhere -> ignored here
    ], columns=["season", "week", "team", "gsis_id", "position", "report_status"])
    r = add_injury_features(out, inj, snaps, players)
    wk4 = r.iloc[3]
    approx(wk4.inj_ol_diff, 0.9 - 0.0)                         # home minus away
    approx(wk4.inj_skill_diff, 0.4 * DEFAULT_IMPORTANCE)
    approx(wk4.inj_db_diff, 0.0 - 0.7)
    approx(wk4.inj_front7_diff, 0.0)
    assert r.iloc[:3][["inj_ol_diff", "inj_skill_diff"]].isna().all().all()   # no report published for weeks 1-3 -> NaN, not "healthy"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
