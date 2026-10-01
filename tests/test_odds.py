"""Unit tests for odds math and bet selection. Run: python -m tests.test_odds  (or pytest)"""
import numpy as np
import pandas as pd

from nfl_model.betting.edge import moneyline_candidates, select_bets, stake_units
from nfl_model.betting.odds import (american_to_decimal, devig_two_way, implied_prob,
                                    overround, prob_to_american, profit_per_unit)


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, (a, b)


def test_conversions():
    approx(american_to_decimal(150), 2.5)
    approx(american_to_decimal(-200), 1.5)
    approx(american_to_decimal(100), 2.0)
    approx(american_to_decimal(-110), 1 + 100 / 110)
    approx(implied_prob(-200), 2 / 3)
    approx(implied_prob(150), 0.4)
    approx(implied_prob(100), 0.5)
    approx(profit_per_unit(150), 1.5)
    approx(profit_per_unit(-200), 0.5)


def test_vectorised_matches_scalar():
    mls = [-300, -110, 100, 130, 500]
    assert np.allclose(implied_prob(pd.Series(mls)), [implied_prob(m) for m in mls])
    assert np.allclose(american_to_decimal(np.array(mls)), [american_to_decimal(m) for m in mls])


def test_devig_sums_to_one_and_vig_positive():
    pa, pb = devig_two_way(-110, -110)
    approx(pa, 0.5); approx(pb, 0.5)
    approx(overround(-110, -110), 2 * 110 / 210 - 1)
    pa, pb = devig_two_way(-200, 170)
    approx(pa + pb, 1.0)
    assert pa > implied_prob(-200) - 1e-12 - 0.05 and pa < implied_prob(-200)   # vig removed shrinks it


def test_roundtrip_prob_to_american():
    for p in (0.3, 0.5, 0.65, 0.8):
        ml = prob_to_american(p)
        approx(implied_prob(float(ml)), p, 1e-9)


def test_ev_and_edge_hand_computed():
    # model says home 60%; prices home +100 / away -120. Fair home prob = 0.5/(0.5+0.54545)=0.47826
    df = pd.DataFrame({"game_id": ["g"], "home_team": ["A"], "away_team": ["B"],
                       "home_moneyline": [100.0], "away_moneyline": [-120.0], "p": [0.60]})
    c = moneyline_candidates(df, "p").set_index("side")
    approx(c.loc["home", "ev"], 0.60 * 2.0 - 1)                      # +0.20
    approx(c.loc["home", "edge"], 0.60 - 0.5 / (0.5 + 120 / 220), 1e-9)
    approx(c.loc["away", "ev"], 0.40 * (1 + 100 / 120) - 1)           # -0.2667
    approx(c.loc["home", "kelly"], 0.20 / 1.0)                         # ev/(dec-1)
    assert c.loc["away", "kelly"] == 0.0
    sel = select_bets(moneyline_candidates(df, "p"), min_edge=0.05)
    assert len(sel) == 1 and sel.loc[0, "team"] == "A"
    assert select_bets(moneyline_candidates(df, "p"), min_edge=0.20).empty


def test_no_bet_when_model_agrees_with_market():
    df = pd.DataFrame({"game_id": ["g"], "home_team": ["A"], "away_team": ["B"],
                       "home_moneyline": [-150.0], "away_moneyline": [130.0]})
    from nfl_model.betting.odds import devig_two_way as dv
    df["p"] = dv(-150.0, 130.0)[0]
    # at fair probability every side has negative EV because of the vig
    assert select_bets(moneyline_candidates(df, "p")).empty


def test_stake_cap_and_fraction():
    k = pd.Series([0.0, 0.10, 0.50])
    s = stake_units(k, bankroll=100, fraction=0.25, cap=0.05)
    assert list(s) == [0.0, 2.5, 5.0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"\nall {len(fns)} odds tests passed")
