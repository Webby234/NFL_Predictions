"""Predict upcoming QB passing yards and price manual prop lines."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..betting.odds import american_to_decimal, devig_two_way
from ..data import load_games, load_passing_games
from .evaluate import TRAIN_START
from .features import WITH_LINES, build_qb_prop_table, upcoming_qb_rows
from .models import QUANTILES, fit_dist


def fit_production(table: pd.DataFrame | None = None, feats=WITH_LINES, kind: str = "ridge"):
    t = build_qb_prop_table() if table is None else table
    tr = t[t.passing_yards.notna() & (t.season >= TRAIN_START)].dropna(subset=["team_implied_r"])
    return fit_dist(tr, feats, kind=kind)


def predict_upcoming(week: int | None = None, kind: str = "ridge") -> pd.DataFrame:
    games, passing = load_games(), load_passing_games()
    rows = upcoming_qb_rows(games, passing, week)
    if rows.empty:
        return rows
    model = fit_production(kind=kind)
    rows = rows.dropna(subset=["team_implied_r"]).copy()
    mu = model.mean(rows)
    rows["pred_yards"] = mu
    for q in QUANTILES:
        rows[f"q{int(q*100)}"] = model.quantile(mu, q)
    rows["new_to_model"] = rows.qb_games == 0
    cols = ["season", "week", "gameday", "player_display_name", "team", "opponent_team", "pred_yards",
            "q10", "q25", "q50", "q75", "q90", "qb_games", "team_implied", "spread_team"]
    out = rows[cols].copy()
    num = out.select_dtypes("number").columns.difference(["season", "week"])
    out[num] = out[num].round(1)
    out.attrs["model"] = model
    out.attrs["rows"] = rows
    return out


def price_lines(pred: pd.DataFrame, lines: pd.DataFrame, min_edge: float = 0.03) -> pd.DataFrame:
    """Price manual prop lines. `lines` columns: player, line, over_odds, under_odds (American).

    Players are matched on display name (case-insensitive); unmatched players are dropped with a warning.
    Pushes (whole-number lines) refund the stake.
    """
    model, rows = pred.attrs["model"], pred.attrs["rows"]
    r = rows.assign(key=rows.player_display_name.str.lower().str.strip())
    ln = lines.assign(key=lines.player.str.lower().str.strip()).merge(r.drop_duplicates("key"), on="key", how="left")
    unmatched = ln[ln.qb_yds.isna()].player.tolist()
    if unmatched:
        import warnings
        warnings.warn(f"no upcoming-game match for: {unmatched}")
    ln = ln[ln.qb_yds.notna()].copy()
    mu = model.mean(ln)
    ln["pred_yards"] = mu
    ln["p_over"] = model.prob_over(mu, ln.line.values)
    ln["p_under"] = model.prob_under(mu, ln.line.values)
    p_push = 1 - ln.p_over - ln.p_under
    ln["mkt_over"], ln["mkt_under"] = devig_two_way(ln.over_odds, ln.under_odds)
    ln["ev_over"] = ln.p_over * american_to_decimal(ln.over_odds) - 1 + p_push
    ln["ev_under"] = ln.p_under * american_to_decimal(ln.under_odds) - 1 + p_push
    ln["edge_over"], ln["edge_under"] = ln.p_over - ln.mkt_over, ln.p_under - ln.mkt_under
    ln["pick"] = np.where((ln.edge_over >= min_edge) & (ln.ev_over > 0), "OVER",
                  np.where((ln.edge_under >= min_edge) & (ln.ev_under > 0), "UNDER", "-"))
    return ln[["player", "team", "opponent_team", "line", "over_odds", "under_odds", "pred_yards", "p_over",
               "p_under", "mkt_over", "edge_over", "edge_under", "ev_over", "ev_under", "pick"]].round(3)
