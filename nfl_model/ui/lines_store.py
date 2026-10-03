"""Sportsbook prop lines typed into the app. Kept outside data/ (a cache) so they are never wiped:
they are also the start of a real price history for testing the prop models later."""
from __future__ import annotations
import pandas as pd
from ..config import ROOT

PATH = ROOT / "lines" / "prop_lines.csv"
COLS = ["season", "week", "stat", "player", "line", "over_odds", "under_odds", "saved_at"]


def load(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(PATH)
    if season is not None:
        d = d[(d.season == season) & (d.week == week)]
    return d.reset_index(drop=True)


def save(season: int, week: int, stat: str, rows: pd.DataFrame) -> int:
    """Replace this week's saved lines for one stat. `rows`: player, line, over_odds, under_odds (blank rows dropped)."""
    rows = rows.copy()
    for c in ("line", "over_odds", "under_odds"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce") if c in rows else float("nan")
    has = rows.over_odds.notna() if stat == "anytime_td" else rows.line.notna()
    rows = rows[has & rows.player.notna()]
    rows = rows.assign(season=season, week=week, stat=stat, saved_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))[COLS]
    old = load()
    keep = old[~((old.season == season) & (old.week == week) & (old.stat == stat))]
    PATH.parent.mkdir(exist_ok=True)
    pd.concat([keep, rows], ignore_index=True).to_csv(PATH, index=False)
    return len(rows)


GAME_PATH = ROOT / "lines" / "game_lines.csv"
GAME_FIELDS = ["away_ml", "home_ml", "home_spread", "away_spread_odds", "home_spread_odds", "total", "over_odds", "under_odds"]
GAME_COLS = ["season", "week", "game_id"] + GAME_FIELDS + ["saved_at"]


def load_game_lines(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    if not GAME_PATH.exists():
        return pd.DataFrame(columns=GAME_COLS)
    d = pd.read_csv(GAME_PATH)
    if season is not None:
        d = d[(d.season == season) & (d.week == week)]
    return d.reset_index(drop=True)


def save_game_lines(season: int, week: int, edited: pd.DataFrame, listed: pd.DataFrame) -> int:
    """Keep only the numbers you changed from the listed ones (so untouched games keep following the listed line).
    Both frames: game_id + GAME_FIELDS. Returns how many games have at least one of your own numbers."""
    e = edited.set_index("game_id")[GAME_FIELDS].apply(pd.to_numeric, errors="coerce")
    l = listed.set_index("game_id")[GAME_FIELDS].apply(pd.to_numeric, errors="coerce").reindex(e.index)
    changed = e.where((e - l).abs() > 1e-9)                       # blank where equal to the listed number or left empty
    rows = changed.dropna(how="all").reset_index()
    rows = rows.assign(season=season, week=week, saved_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))[GAME_COLS]
    old = load_game_lines()
    keep = old[~((old.season == season) & (old.week == week))]
    GAME_PATH.parent.mkdir(exist_ok=True)
    pd.concat([keep, rows], ignore_index=True).to_csv(GAME_PATH, index=False)
    return len(rows)
