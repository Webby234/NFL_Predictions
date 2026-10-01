"""Live moneyline odds from The Odds API (https://the-odds-api.com), with line shopping.

Set the key in the ODDS_API_KEY environment variable. The free tier is enough for weekly use.
NOTE: parse_events() is unit-tested on a hand-written sample shaped like the API's documented
v4 response; the HTTP call itself has not been exercised against the real service.

For each game we return
  * the BEST available price on each side across books (that is what you would bet), and
  * a consensus fair probability (average of each book's vig-free probability), which is a
    better estimate of the true market than any single book's two prices.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from ..config import TEAM_NAME_TO_CODE
from .odds import devig_two_way

API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


def fetch_events(api_key: str | None = None, regions: str = "us", timeout: int = 30) -> list[dict]:
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set the ODDS_API_KEY environment variable (or pass api_key).")
    qs = urllib.parse.urlencode({"apiKey": api_key, "regions": regions, "markets": "h2h", "oddsFormat": "american"})
    with urllib.request.urlopen(f"{API_URL}?{qs}", timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_events(events: list[dict]) -> pd.DataFrame:
    """One row per (game, bookmaker) with home/away American prices."""
    rows = []
    for ev in events:
        home, away = TEAM_NAME_TO_CODE.get(ev["home_team"]), TEAM_NAME_TO_CODE.get(ev["away_team"])
        if home is None or away is None:
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                price = {o["name"]: o["price"] for o in mk.get("outcomes", [])}
                if ev["home_team"] in price and ev["away_team"] in price:
                    rows.append({"home_team": home, "away_team": away, "commence_time": ev.get("commence_time"),
                                 "book": bk.get("title", bk.get("key")), "home_ml": float(price[ev["home_team"]]),
                                 "away_ml": float(price[ev["away_team"]])})
    return pd.DataFrame(rows, columns=["home_team", "away_team", "commence_time", "book", "home_ml", "away_ml"])


def best_and_consensus(books: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-book rows to one row per game: best prices + consensus fair probability."""
    if books.empty:
        return pd.DataFrame(columns=["home_team", "away_team", "home_moneyline", "home_book", "away_moneyline",
                                     "away_book", "market_home_prob", "n_books"])
    b = books.copy()
    b["fair_home"], _ = devig_two_way(b["home_ml"], b["away_ml"])
    out = []
    for (home, away), g in b.groupby(["home_team", "away_team"]):
        hi, ai = g["home_ml"].idxmax(), g["away_ml"].idxmax()     # higher American number = better payout
        out.append({"home_team": home, "away_team": away, "home_moneyline": g.loc[hi, "home_ml"],
                    "home_book": g.loc[hi, "book"], "away_moneyline": g.loc[ai, "away_ml"],
                    "away_book": g.loc[ai, "book"], "market_home_prob": g["fair_home"].mean(), "n_books": len(g)})
    return pd.DataFrame(out)


def live_odds_table(api_key: str | None = None) -> pd.DataFrame:
    return best_and_consensus(parse_events(fetch_events(api_key)))
