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
