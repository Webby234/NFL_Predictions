import numpy as np, pandas as pd
from sklearn.dummy import DummyRegressor
from nfl_model.data import load_games, load_skill_games
from nfl_model.props import skill_predict as SP
from nfl_model.props.models import CountModel
from nfl_model.props.skill import build_skill_table
from nfl_model.props.skill_eval import STATS, eligible


def test_skill_features_ignore_this_week_and_future():
    games, skill = load_games(), load_skill_games()
    base = build_skill_table(games, skill)
    wk = (2022, 9)
    cut = (skill.season > wk[0]) | ((skill.season == wk[0]) & (skill.week >= wk[1]))
    scr = skill.copy()
    for c in ["carries", "rushing_yards", "targets", "receptions", "receiving_yards", "rushing_tds",
              "receiving_tds", "receiving_air_yards", "attempts", "passing_tds", "passing_yards"]:
        scr.loc[cut, c] = 777
    from nfl_model.data import load_injuries, load_players, load_snap_counts
    snaps, players, inj = load_snap_counts(), load_players(), load_injuries()
    base = build_skill_table(games, skill, snaps, players, inj)
    s2 = snaps.copy()
    s2.loc[(s2.season > wk[0]) | ((s2.season == wk[0]) & (s2.week >= wk[1])), "offense_pct"] = 0.123
    alt = build_skill_table(games, scr, s2, players, inj)
    assert base[(base.season == wk[0]) & (base.week == wk[1])].p_snap.notna().mean() > 0.8
    cols = [c for c in base.columns if c.startswith(("p_", "pf_", "ps_", "last_", "vac_", "qb_", "team_", "opp_"))
            or c in ("games", "total_r", "inj_q", "n_out")]
    assert {"vac_tgt", "pf_targets", "ps_targets", "last_targets", "qb_pass_dev", "inj_q"} <= set(cols)
    wkrows = base[(base.season == wk[0]) & (base.week == wk[1])]
    assert wkrows.vac_tgt.max() > 0 and wkrows.inj_q.sum() > 0          # the injury report is actually being used
    a = base[(base.season == wk[0]) & (base.week == wk[1])].set_index(["player_id", "team"])[cols]
    b = alt[(alt.season == wk[0]) & (alt.week == wk[1])].set_index(["player_id", "team"])[cols]
    idx = a.index.intersection(b.index)
    assert len(idx) > 200
    assert np.allclose(a.loc[idx].values.astype(float), b.loc[idx].values.astype(float), equal_nan=True)


def test_eligibility_uses_only_pregame_columns():
    t = pd.DataFrame(dict(position=["RB"] * 3, games=[5, 5, 5], team_implied_r=[0.0] * 3, p_carries=[8.0, 2.0, 8.0],
                          p_targets=[1.0, 1.0, 1.0], p_pass_att=[0.0] * 3, rushing_yards=[0.0, 200.0, 50.0],
                          carries=[0.0, 30.0, 12.0]))      # same-game outcomes must not change who is eligible
    assert eligible(t, STATS["rush_yds"]).index.tolist() == [0, 2]


def test_blend_is_the_average_of_its_two_models():
    from nfl_model.props.models import Blend
    class Const:
        def __init__(self, v): self.v = v
        def fit(self, X, y): return self
        def predict(self, X): return np.full(len(X), self.v)
        def predict_proba(self, X): return np.column_stack([np.full(len(X), 1 - self.v), np.full(len(X), self.v)])
    b = Blend(Const(0.2), Const(0.6)).fit(None, None)
    assert np.allclose(b.predict(np.zeros((3, 1))), 0.4) and np.allclose(b.predict_proba(np.zeros((3, 1)))[:, 1], 0.4)


def test_every_prop_input_exists_in_the_table():
    from nfl_model.props.skill import XCOLS, METRIC_NAMES, BASE
    from nfl_model.props.skill_eval import feats_for
    have = set(XCOLS) | {f"p_{m}" for m in METRIC_NAMES} | set(BASE) | {"is_home", "dome", "wind_eff", "temp_eff",
                                                                        "team_implied_r", "opp_implied_r", "spread_team", "total_r"}
    for st in STATS.values():
        assert not [f for f in feats_for(st, True) if f not in have], st.name


def test_count_model_probabilities():
    m = DummyRegressor(strategy="constant", constant=3.0).fit(np.zeros((2, 1)), [3, 3])
    cm = CountModel(["x"], m, 0.2); mu = np.full(4, 3.0); lines = np.array([0.5, 1.5, 2.5, 3.5])
    po, pu = cm.prob_over(mu, lines), cm.prob_under(mu, lines)
    assert np.allclose(po + pu, 1.0) and np.all(np.diff(po) < 0)       # half lines: no push
    assert abs(sum(cm.pmf(mu[:1], k)[0] for k in range(200)) - 1) < 1e-6
    assert abs(cm.prob_over(mu[:1], 2.0)[0] + cm.prob_under(mu[:1], 2.0)[0] + cm.pmf(mu[:1], 2)[0] - 1) < 1e-9


def test_anytime_td_price_hand_computed_and_missing_no_price():
    class M:
        def prob(self, X): return np.array([0.6, 0.6])
    rows = pd.DataFrame(dict(player_display_name=["A One", "B Two"], team=["X", "Y"], opponent_team=["Y", "X"],
                             games=[5.0, 5.0]))
    pred = pd.DataFrame(); pred.attrs.update(stat="anytime_td", model=M(), rows=rows)
    lines = pd.DataFrame(dict(player=["a one", "b two"], odds=[-150, 100], no_odds=[130, np.nan]))
    out = SP.price_lines(pred, lines).set_index("player")
    assert abs(out.loc["a one", "ev"] - round(0.6 * (1 + 100 / 150) - 1, 3)) < 1e-3
    assert abs(out.loc["b two", "mkt_yes"] - 0.5) < 1e-9 and not out.mkt_yes.isna().any()


def test_placeholders_only_listed_qb_and_skip_non_upcoming():
    games = pd.DataFrame(dict(game_type=["REG"], season=[2026], week=[4], home_score=[np.nan], home_team=["AAA"],
                              away_team=["BBB"], home_qb_id=["q1"], away_qb_id=["q9"]))
    sk = pd.DataFrame(dict(player_id=["q1", "q2", "r1", "w1"], player_display_name=list("abcd"),
                           position=["QB", "QB", "RB", "WR"], season=2026, week=3, team=["AAA", "AAA", "AAA", "BBB"],
                           opponent_team="ZZZ"))
    ph = SP.placeholder_rows(games, sk)
    assert sorted(ph.player_id) == ["q1", "r1", "w1"]                   # backup QB q2 excluded
    assert ph.set_index("player_id").loc["w1", "opponent_team"] == "AAA"


if __name__ == "__main__":
    bad = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("ok  ", n)
            except Exception as e: bad += 1; print("FAIL", n, repr(e))
    raise SystemExit(bad)
