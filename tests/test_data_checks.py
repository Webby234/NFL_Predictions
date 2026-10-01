import numpy as np, pandas as pd
from nfl_model.data.validate import check
from nfl_model.features import INJURY_FEATURES, NO_MARKET, QB_FEATURES, WITH_SPREAD
from nfl_model.features.availability import AVAILABILITY, usable_at
from nfl_model.evaluation import lines_backtest as L


def _games(n=4):
    return pd.DataFrame(dict(game_id=[f"g{i}" for i in range(n)], season=2020, game_type="REG", week=1,
                             gameday="2020-09-13", home_score=20.0, away_score=17.0, spread_line=3.0,
                             total_line=45.0, home_moneyline=-150, home_spread_odds=-110,
                             away_spread_odds=-110, over_odds=-110, under_odds=-110))


def test_every_model_feature_is_tagged():
    allf = set(NO_MARKET + WITH_SPREAD + QB_FEATURES + INJURY_FEATURES + L.MARGIN_FEATS + L.TOTAL_FEATS)
    assert not [f for f in allf if f not in AVAILABILITY]


def test_early_drops_same_week_news_but_gameday_keeps_it():
    early = usable_at(L.MARGIN_FEATS + L.TOTAL_FEATS, "early")
    for f in QB_FEATURES + INJURY_FEATURES + ["wind_eff", "temp_eff"]:
        assert f not in early
    assert "elo_diff" in early and "dome" in early
    assert "wind_eff" in usable_at(L.TOTAL_FEATS, "gameday")


def test_untagged_feature_raises():
    try:
        usable_at(["brand_new_feature"])
    except KeyError:
        return
    raise AssertionError("should have raised")


def test_validator_flags_duplicates_and_bad_lines_and_passes_clean():
    clean = check(_games())
    assert not any(l == "ERROR" for l, _ in clean)
    dup = pd.concat([_games(2), _games(2)])
    assert any(l == "ERROR" and "duplicate" in m for l, m in check(dup))
    bad = _games(); bad.loc[0, "spread_line"] = 99.0
    assert any(l == "ERROR" and "implausible" in m for l, m in check(bad))


def test_validator_flags_blank_feature_and_missing_lines():
    g = _games(); g["feat"] = np.nan; g.loc[:1, "spread_line"] = np.nan
    msgs = [m for _, m in check(g, ["feat"])]
    assert any("blank" in m for m in msgs) and any("spread_line missing" in m for m in msgs)


if __name__ == "__main__":
    bad = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("ok  ", n)
            except Exception as e: bad += 1; print("FAIL", n, repr(e))
    raise SystemExit(bad)
