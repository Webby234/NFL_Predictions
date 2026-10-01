"""Parser/aggregation test on a hand-written sample shaped like The Odds API v4 h2h response."""
import pandas as pd

from nfl_model.betting.live_odds import best_and_consensus, parse_events
from nfl_model.betting.odds import devig_two_way
from nfl_model.betting.picks import picks_table

SAMPLE = [{
    "id": "x1", "commence_time": "2026-10-04T17:00:00Z",
    "home_team": "Houston Texans", "away_team": "Dallas Cowboys",
    "bookmakers": [
        {"key": "bk1", "title": "BookOne", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Houston Texans", "price": -160}, {"name": "Dallas Cowboys", "price": 140}]}]},
        {"key": "bk2", "title": "BookTwo", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Dallas Cowboys", "price": 150}, {"name": "Houston Texans", "price": -175}]}]},
        {"key": "bk3", "title": "SpreadsOnly", "markets": [{"key": "spreads", "outcomes": []}]},
    ]}, {"id": "x2", "home_team": "Unknown FC", "away_team": "Dallas Cowboys", "bookmakers": []}]


def test_parse_and_best_prices():
    books = parse_events(SAMPLE)
    assert len(books) == 2 and set(books.book) == {"BookOne", "BookTwo"}
    assert set(books.home_team) == {"HOU"} and set(books.away_team) == {"DAL"}
    g = best_and_consensus(books).iloc[0]
    assert g.home_moneyline == -160 and g.home_book == "BookOne"      # least negative = best for home
    assert g.away_moneyline == 150 and g.away_book == "BookTwo"       # larger positive = best for away
    f1 = devig_two_way(-160, 140)[0]; f2 = devig_two_way(-175, 150)[0]
    assert abs(g.market_home_prob - (f1 + f2) / 2) < 1e-12 and g.n_books == 2


def test_live_prices_flow_into_picks():
    games = pd.DataFrame({"game_id": ["g1"], "gameday": [pd.Timestamp("2026-10-04")], "home_team": ["HOU"],
                          "away_team": ["DAL"], "home_moneyline": [-200.0], "away_moneyline": [170.0],
                          "p_home": [0.50]})
    live = best_and_consensus(parse_events(SAMPLE))
    cands, bets = picks_table(games, min_edge=0.0, live=live)
    assert set(cands.moneyline) == {-160.0, 150.0}                    # live best prices replaced file prices
    assert (cands.market_prob.sum() - 2.0) < 1e-9 and abs(cands.groupby("side").market_prob.first().sum() - 1) < 1e-9


if __name__ == "__main__":
    test_parse_and_best_prices(); print("PASS test_parse_and_best_prices")
    test_live_prices_flow_into_picks(); print("PASS test_live_prices_flow_into_picks")
