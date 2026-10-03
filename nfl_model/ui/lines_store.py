"""Sportsbook lines typed into the app.

On your own computer they are saved under lines/ (kept outside data/, which is a cache), and double as the start
of a real price history. On a shared host every function takes `store` (the visitor's session), so one visitor's
numbers never change the board for anyone else and nothing is written to the shared disk."""
from __future__ import annotations
import pandas as pd
from ..config import ROOT

PATH = ROOT / "lines" / "prop_lines.csv"
COLS = ["season", "week", "stat", "player", "line", "over_odds", "under_odds", "saved_at"]


def _read(path, cols, store, key) -> pd.DataFrame:
    """All saved rows: from `store` (a dict such as a visitor's session) when given, otherwise from the file."""
    if store is not None:
        d = store.get(key)
        return d.copy() if d is not None else pd.DataFrame(columns=cols)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)


def _write(path, store, key, d: pd.DataFrame) -> None:
    if store is not None:
        store[key] = d.reset_index(drop=True)
        return
    path.parent.mkdir(exist_ok=True)
    d.to_csv(path, index=False)


def load(season: int | None = None, week: int | None = None, store=None) -> pd.DataFrame:
    d = _read(PATH, COLS, store, "prop_lines")
    if season is not None:
        d = d[(d.season == season) & (d.week == week)]
    return d.reset_index(drop=True)


def save(season: int, week: int, stat: str, rows: pd.DataFrame, store=None) -> int:
    """Replace this week's saved lines for one stat. `rows`: player, line, over_odds, under_odds (blank rows dropped)."""
    rows = rows.copy()
    for c in ("line", "over_odds", "under_odds"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce") if c in rows else float("nan")
    has = rows.over_odds.notna() if stat == "anytime_td" else rows.line.notna()
    rows = rows[has & rows.player.notna()]
    rows = rows.assign(season=season, week=week, stat=stat, saved_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))[COLS]
    old = load(store=store)
    keep = old[~((old.season == season) & (old.week == week) & (old.stat == stat))]
    _write(PATH, store, "prop_lines", pd.concat([keep, rows], ignore_index=True))
    return len(rows)


GAME_PATH = ROOT / "lines" / "game_lines.csv"
GAME_FIELDS = ["away_ml", "home_ml", "home_spread", "away_spread_odds", "home_spread_odds", "total", "over_odds", "under_odds"]
GAME_COLS = ["season", "week", "game_id"] + GAME_FIELDS + ["saved_at"]


def load_game_lines(season: int | None = None, week: int | None = None, store=None) -> pd.DataFrame:
    d = _read(GAME_PATH, GAME_COLS, store, "game_lines")
    if season is not None:
        d = d[(d.season == season) & (d.week == week)]
    return d.reset_index(drop=True)


def save_game_lines(season: int, week: int, edited: pd.DataFrame, listed: pd.DataFrame, store=None) -> int:
    """Keep only the numbers you changed from the listed ones (so untouched games keep following the listed line).
    Both frames: game_id + GAME_FIELDS. Returns how many games have at least one of your own numbers."""
    e = edited.set_index("game_id")[GAME_FIELDS].apply(pd.to_numeric, errors="coerce")
    l = listed.set_index("game_id")[GAME_FIELDS].apply(pd.to_numeric, errors="coerce").reindex(e.index)
    changed = e.where((e - l).abs() > 1e-9)                       # blank where equal to the listed number or left empty
    rows = changed.dropna(how="all").reset_index()
    rows = rows.assign(season=season, week=week, saved_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))[GAME_COLS]
    old = load_game_lines(store=store)
    keep = old[~((old.season == season) & (old.week == week))]
    _write(GAME_PATH, store, "game_lines", pd.concat([keep, rows], ignore_index=True))
    return len(rows)
