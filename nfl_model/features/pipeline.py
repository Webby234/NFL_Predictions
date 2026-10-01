"""Assemble the full pre-game feature table (team + QB + injuries)."""
from __future__ import annotations

import pandas as pd

from ..data import (load_games, load_injuries, load_players, load_qb_games, load_snap_counts,
                    load_team_stats)
from .injuries import add_injury_features
from .qb import add_qb_features
from .team_features import build_features


def build_feature_table(games: pd.DataFrame, team_stats: pd.DataFrame, qb_games: pd.DataFrame,
                        injuries: pd.DataFrame, snaps: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    out = build_features(games, team_stats)
    out = add_qb_features(out, qb_games, injuries)
    return add_injury_features(out, injuries, snaps, players)


def load_feature_table() -> pd.DataFrame:
    """Download (cached) everything and build the table."""
    return build_feature_table(load_games(), load_team_stats(1999), load_qb_games(1999), load_injuries(2009),
                               load_snap_counts(2013), load_players())
