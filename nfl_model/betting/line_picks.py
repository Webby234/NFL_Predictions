"""Spread and total picks for upcoming games (model vs listed line).

Uses the ridge models WITH the line as an input (the only variants that came close to the
market in backtests). Edge = model number - line, in points. Backtests show no reliable ROI,
so treat these as disagreements to investigate.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ..config import TRAIN_START
from ..evaluation import lines_backtest as L


def upcoming_line_picks(df: pd.DataFrame, week: int | None = None, min_edge: float = 1.0) -> pd.DataFrame:
    df = L.prep(df)
    up = df[(df.game_type == "REG") & df.home_score.isna() & df.spread_line.notna() & df.total_line.notna()]
    if up.empty:
        return up
    up = up[up.season == up.season.min()]
    up = up[up.week == int(week if week is not None else up.week.min())].copy()
    done = df[(df.game_type == "REG") & df.home_score.notna() & (df.season >= TRAIN_START)]
    rows = []
    for name, target, feats, col in (("spread", "result", L.MARGIN_FEATS, "spread_line"),
                                     ("total", "total", L.TOTAL_FEATS, "total_line")):
        cols = feats + [col]
        tr = done.dropna(subset=[target, col])
        m = L.ridge().fit(tr[cols], tr[target])
        up[f"{name}_model"] = m.predict(up[cols])
        up[f"{name}_edge"] = up[f"{name}_model"] - up[col]
    up["spread_pick"] = np.where(up.spread_edge > min_edge, "HOME " + up.home_team,
                         np.where(up.spread_edge < -min_edge, "AWAY " + up.away_team, "-"))
    up["total_pick"] = np.where(up.total_edge > min_edge, "OVER", np.where(up.total_edge < -min_edge, "UNDER", "-"))
    cols = ["gameday", "away_team", "home_team", "spread_line", "spread_model", "spread_edge", "spread_pick",
            "total_line", "total_model", "total_edge", "total_pick"]
    out = up[cols].copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out
