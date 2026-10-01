"""python -m nfl_model.evaluation.run_lines  -> reports/lines_*.csv"""
import pandas as pd
from ..config import DATA_DIR, REPORT_DIR
from ..data import load_games
from ..features import load_feature_table
from . import lines_backtest as L


def open_vs_close(hist_csv=DATA_DIR / "line_history.csv"):
    g = load_games()
    h = pd.read_csv(hist_csv, parse_dates=["snap_ts", "kickoff"]).sort_values("snap_ts")
    h = h.dropna(subset=["spread_line"])
    h["lead"] = (h.kickoff - h.snap_ts).dt.total_seconds() / 86400
    rows = []
    for lo, hi in [(5, 8), (3, 5), (1, 3)]:
        s = h[(h.lead >= lo) & (h.lead < hi)].groupby("game_id").first().reset_index()
        d = g.merge(s[["game_id", "spread_line", "total_line"]].rename(
            columns=lambda c: "o_" + c if c != "game_id" else c), on="game_id")
        d = d[d.home_score.notna()]
        for nm, y, c in [("spread", d.home_score - d.away_score, "spread_line"),
                         ("total", d.home_score + d.away_score, "total_line")]:
            dm = (y - d["o_" + c]) ** 2 - (y - d[c]) ** 2
            rows.append(dict(market=nm, days_before_kickoff=f"{lo}-{hi}", n=len(d),
                             mse_open_minus_close=dm.mean(), se=dm.std() / len(d) ** .5,
                             pct_moved=(d["o_" + c] != d[c]).mean()))
    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    f = load_feature_table(); f = f[f.game_type == "REG"]
    ov = open_vs_close(); ov.to_csv(REPORT_DIR / "lines_open_vs_close.csv", index=False); print(ov.round(3))
    mse, roi = [], []
    for kind in ["ridge", "gbm"]:
        for mk, target, feats, col, expr in [
                ("spread", "result", L.MARGIN_FEATS, "spread_line", lambda d: d.result),
                ("total", "total", L.TOTAL_FEATS, "total_line", lambda d: d.total)]:
            for wl in [None, col]:
                d = L.walk_forward(f, target, feats, wl, kind)
                name = f"{kind}_{'with_line' if wl else 'no_market'}"
                mse.append(dict(market=mk, model=name, **L.mse_vs_market(d, expr(d), col)))
                t = L.roi_table(d, mk, col); t.insert(0, "model", name); t.insert(0, "market", mk)
                roi.append(t)
    pd.DataFrame(mse).to_csv(REPORT_DIR / "lines_mse.csv", index=False)
    pd.concat(roi).to_csv(REPORT_DIR / "lines_roi.csv", index=False)


if __name__ == "__main__":
    main()
