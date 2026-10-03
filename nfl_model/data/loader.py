"""Download and cache nflverse data as plain CSVs.

Big per-season player files (player stats, injuries, snap counts) are slimmed to the columns/rows
we need as they are downloaded, so the cache stays small (the raw files are ~200 MB in total).

Why not nfl_data_py? It has been archived upstream. nflverse publishes the same
data as CSV release assets on GitHub, so pandas alone is enough.

Cached files live in ./data (git-ignored). The schedule and the in-progress
season are re-downloaded when older than `max_age_hours`; finished seasons are
downloaded once.
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Columns we use from the weekly team stats file (keeps memory small).
TEAM_STAT_COLS = [
    "season", "week", "team", "season_type", "game_id", "opponent_team",
    "attempts", "carries", "sacks_suffered", "sack_yards_lost",
    "passing_yards", "rushing_yards", "passing_epa", "rushing_epa",
    "passing_interceptions", "sack_fumbles_lost", "rushing_fumbles_lost",
    "receiving_fumbles_lost",
]


def _download(url: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            path.write_bytes(resp.read())
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def _fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def load_games(max_age_hours: float = 6) -> pd.DataFrame:
    """All games 1999-present incl. future schedule, results and closing odds."""
    path = DATA_DIR / "games.csv"
    if not _fresh(path, max_age_hours):
        if not _download(f"{BASE}/schedules/games.csv", path) and not path.exists():
            raise RuntimeError("Could not download games.csv")
    games = pd.read_csv(path)
    games["gameday"] = pd.to_datetime(games["gameday"])
    return games


def load_team_stats(first_season: int = 1999, last_season: int | None = None,
                    max_age_hours: float = 6) -> pd.DataFrame:
    """Weekly team box-score stats (one row per team per game)."""
    last_season = last_season or date.today().year
    frames = []
    for season in range(first_season, last_season + 1):
        path = DATA_DIR / "team_stats" / f"stats_team_week_{season}.csv"
        finished = season < date.today().year - (0 if date.today().month >= 3 else 1)
        if not (path.exists() and (finished or _fresh(path, max_age_hours))):
            _download(f"{BASE}/stats_team/stats_team_week_{season}.csv", path)
        if path.exists():
            frames.append(pd.read_csv(path, usecols=lambda c: c in TEAM_STAT_COLS))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Player-level data (slimmed on download)
# --------------------------------------------------------------------------------------
def _season_is_final(season: int) -> bool:
    today = date.today()
    return season < today.year - (0 if today.month >= 3 else 1)


def _load_slim(release: str, name_fmt: str, subdir: str, first: int, last: int, slim,
               max_age_hours: float = 6) -> pd.DataFrame:
    """Download each season's file once, keep only `slim(raw_df)`, cache the slim CSV."""
    frames = []
    for season in range(first, last + 1):
        path = DATA_DIR / subdir / f"{season}.csv"
        if not (path.exists() and (_season_is_final(season) or _fresh(path, max_age_hours))):
            try:
                with urllib.request.urlopen(f"{BASE}/{release}/{name_fmt.format(season=season)}", timeout=180) as r:
                    raw = r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                raise
            if not raw.strip():
                continue                                  # some seasons are published as empty files
            path.parent.mkdir(parents=True, exist_ok=True)
            slim(pd.read_csv(io.BytesIO(raw), low_memory=False)).to_csv(path, index=False)
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_qb_games(first_season: int = 1999, last_season: int | None = None) -> pd.DataFrame:
    """One row per QB per game with dropbacks and passing EPA (QBs with >=1 attempt)."""
    def slim(d):
        d = d[(d["position"] == "QB") & (d["attempts"] > 0)]
        return d[["player_id", "season", "week", "season_type", "team", "attempts", "sacks_suffered", "passing_epa"]]
    return _load_slim("stats_player", "stats_player_week_{season}.csv", "qb_games", first_season,
                      last_season or date.today().year, slim)


def load_passing_games(first_season: int = 2006, last_season: int | None = None) -> pd.DataFrame:
    """One row per QB per game (attempts > 0): passing and rushing volume/yards, for props."""
    def slim(d):
        d = d[(d["position"] == "QB") & (d["attempts"] > 0) & (d["season_type"] == "REG")]
        return d[["player_id", "player_display_name", "season", "week", "team", "opponent_team", "completions",
                  "attempts", "passing_yards", "passing_tds", "passing_interceptions", "sacks_suffered",
                  "carries", "rushing_yards"]]
    return _load_slim("stats_player", "stats_player_week_{season}.csv", "passing_games", first_season,
                      last_season or date.today().year, slim)


def load_skill_games(first_season: int = 2006, last_season: int | None = None) -> pd.DataFrame:
    """One row per QB/RB/WR/TE per regular-season game: rushing, receiving, passing volume, yards, TDs."""
    def slim(d):
        d = d[(d["season_type"] == "REG") & d["position"].isin(["QB", "RB", "FB", "WR", "TE"])].copy()
        d["position"] = d["position"].replace({"FB": "RB"})
        return d[["player_id", "player_display_name", "position", "season", "week", "team", "opponent_team",
                  "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards",
                  "receiving_tds", "receiving_air_yards", "attempts", "passing_tds", "passing_yards"]]
    # cache folder name carries a version: bump it whenever the kept columns change, or old caches go stale
    return _load_slim("stats_player", "stats_player_week_{season}.csv", "skill_games_v2", first_season,
                      last_season or date.today().year, slim)


def load_injuries(first_season: int = 2009, last_season: int | None = None) -> pd.DataFrame:
    """Weekly NFL injury reports (final status published before each game), regular season only."""
    def slim(d):
        d = d[d["game_type"] == "REG"]
        return d[["season", "week", "team", "gsis_id", "position", "report_status"]]
    return _load_slim("injuries", "injuries_{season}.csv", "injuries", first_season,
                      last_season or date.today().year, slim)


def load_snap_counts(first_season: int = 2013, last_season: int | None = None) -> pd.DataFrame:
    """Per-player snap shares by game (regular season). Published from 2013."""
    def slim(d):
        d = d[d["game_type"] == "REG"]
        return d[["season", "week", "team", "pfr_player_id", "offense_pct", "defense_pct"]]
    return _load_slim("snap_counts", "snap_counts_{season}.csv", "snap_counts", first_season,
                      last_season or date.today().year, slim)


def load_players() -> pd.DataFrame:
    """ID crosswalk: gsis_id (injuries, stats) <-> pfr_id (snap counts)."""
    path = DATA_DIR / "players_xwalk.csv"
    if not path.exists():
        with urllib.request.urlopen(f"{BASE}/players/players.csv", timeout=180) as r:
            raw = pd.read_csv(io.BytesIO(r.read()), usecols=["gsis_id", "pfr_id"], low_memory=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.dropna().drop_duplicates().to_csv(path, index=False)
    return pd.read_csv(path)
