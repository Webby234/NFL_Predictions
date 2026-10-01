"""Diagnostic: what news moves lines, and does the market move enough?

For each game: open = earliest snapshot 5-8 days out, close = nflverse final.
  move   = close - open            (what the market adjusted for)
  missed = actual - open           (what the open line missed; actual = margin or total)
Regress both on the same news features. If missed/move > 1 the market
under-adjusted for that news on average; < 1 it over-adjusted.
Spread sign: + = home favored. Features are the pre-game values incl. same-week news.
"""
import numpy as np, pandas as pd
from ..config import REPORT_DIR
from ..features import load_feature_table
from .early_lines import early_snapshot
from .lines_backtest import prep


def ols(X, y):
    X = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; s2 = r @ r / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))); return b[1:], se[1:]


def run():
    f = prep(load_feature_table()); f = f[(f.game_type == "REG") & f.home_score.notna()]
    s = early_snapshot()
    d = f.merge(s[["game_id", "spread_line", "total_line"]].rename(
        columns={"spread_line": "o_spread", "total_line": "o_total"}), on="game_id")
    d["qb_swap"] = d.qb_changed_h.astype(float) - d.qb_changed_a.astype(float)
    sets = {"spread": ("spread_line", "o_spread", d.home_score - d.away_score,
                       ["qb_swap", "qb_diff", "inj_skill_diff", "inj_ol_diff", "inj_front7_diff", "inj_db_diff"]),
            "total": ("total_line", "o_total", d.home_score + d.away_score,
                      ["dome", "wind_eff", "temp_eff"])}
    rows = []
    for mk, (close, op, actual, feats) in sets.items():
        x = d.dropna(subset=feats + [op, close]); a = actual[x.index]
        move = (x[close] - x[op]).values; miss = (a - x[op]).values
        X = x[feats].values.astype(float)
        bm, sm = ols(X, move); bx, sx = ols(X, miss)
        for i, ft in enumerate(feats):
            rows.append(dict(market=mk, feature=ft, n=len(x), market_moved_pts_per_unit=bm[i], se_move=sm[i],
                             open_missed_pts_per_unit=bx[i], se_missed=sx[i],
                             market_share_of_news=bm[i] / bx[i] if abs(bx[i]) > 1e-9 else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = run(); r.to_csv(REPORT_DIR / "lines_move_attribution.csv", index=False)
    print(r.round(3).to_string(index=False))
