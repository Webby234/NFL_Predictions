import numpy as np, pandas as pd
from nfl_model.data import load_games, load_passing_games
from nfl_model.props.features import GAMES_CAP, build_qb_prop_table
from nfl_model.props.models import DistModel
from nfl_model.props.predict import price_lines
from sklearn.dummy import DummyRegressor


def _dist(mean=250.0):
    z = np.sort(np.random.default_rng(0).normal(size=5000))
    m = DummyRegressor(strategy="constant", constant=mean).fit(np.zeros((2, 1)), [mean, mean])
    return DistModel(["x"], m, 60.0, 0.0, z)


def test_prob_over_under_push_partition_and_monotone():
    d = _dist(); mu = np.full(5, 250.0); lines = np.array([150.0, 200.5, 250.0, 300.5, 350.0])
    po, pu = d.prob_over(mu, lines), d.prob_under(mu, lines)
    assert np.all(po + pu <= 1 + 1e-9) and np.all(np.diff(po) < 0) and np.all(np.diff(pu) > 0)
    assert abs(po[2] - 0.5) < 0.03                      # line at the mean is a coin flip
    q = [d.quantile(mu[:1], x)[0] for x in (0.1, 0.5, 0.9)]
    assert q[0] < q[1] < q[2] and abs(q[1] - 250) < 5


def test_price_lines_ev_matches_hand_computation():
    d = _dist()
    rows = pd.DataFrame(dict(player_display_name=["Test QB"], team=["AAA"], opponent_team=["BBB"], x=[0.0], qb_yds=[250.0]))
    pred = pd.DataFrame(); pred.attrs["model"] = d; pred.attrs["rows"] = rows
    lines = pd.DataFrame(dict(player=["test qb"], line=[200.5], over_odds=[-110], under_odds=[-110]))
    out = price_lines(pred, lines).iloc[0]
    po = d.prob_over(np.array([250.0]), 200.5)[0]
    assert abs(out.p_over - round(po, 3)) < 1e-9
    assert abs(out.ev_over - round(po * (1 + 100 / 110) - 1, 3)) < 1e-3   # no push on a half line
    assert out.pick == "OVER"


def test_features_do_not_see_the_future_and_career_games_capped():
    games, passing = load_games(), load_passing_games()
    base = build_qb_prop_table(games, passing)
    wk = (2022, 9)
    cut = (passing.season > wk[0]) | ((passing.season == wk[0]) & (passing.week >= wk[1]))
    changed = passing.copy()
    changed.loc[cut, ["passing_yards", "attempts"]] = [999, 99]       # scramble this week and everything later
    alt = build_qb_prop_table(games, changed)
    cols = ["qb_yds", "qb_att", "qb_ypa", "opp_yds_allowed", "lg_yds", "qb_games"]
    a = base[(base.season == wk[0]) & (base.week == wk[1])].set_index("player_id")[cols]
    b = alt[(alt.season == wk[0]) & (alt.week == wk[1])].set_index("player_id")[cols]
    common = a.index.intersection(b.index)
    assert len(common) > 20
    assert np.allclose(a.loc[common].values.astype(float), b.loc[common].values.astype(float))
    assert base.qb_games.max() <= GAMES_CAP


if __name__ == "__main__":
    bad = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("ok  ", n)
            except Exception as e: bad += 1; print("FAIL", n, repr(e))
    raise SystemExit(bad)
