"""Test: does a model fed an early-week line beat early-week prices? (and CLV)

Approximation (documented): models are trained on CLOSING lines, then fed the
early snapshot line (5-8 days out, earliest in that window) as the line input,
and bets are settled at that snapshot's line and odds. CLV = how far the line
later moved toward our side (close - open, in bet direction).
"""
import numpy as np, pandas as pd
from ..config import DATA_DIR, REPORT_DIR
from ..features import load_feature_table
from . import lines_backtest as L
from ..features.availability import usable_at


def early_snapshot(hist_csv=DATA_DIR / "line_history.csv", lo=5, hi=8):
    h = pd.read_csv(hist_csv, parse_dates=["snap_ts", "kickoff"]).sort_values("snap_ts")
    h["lead"] = (h.kickoff - h.snap_ts).dt.total_seconds() / 86400
    h = h[(h.lead >= lo) & (h.lead < hi)]
    return h.groupby("game_id").first().reset_index()


def run(kind="ridge", lo=5, hi=8, seed=0, asof="early"):
    """asof="early" (default) drops same-week-news features; asof="gameday" keeps them (leaky here)."""
    f = load_feature_table(); f = f[f.game_type == "REG"]
    snap = early_snapshot(lo=lo, hi=hi)
    odds = ["home_spread_odds", "away_spread_odds", "over_odds", "under_odds"]
    rows = []
    for mk, target, feats, col in [("spread", "result", L.MARGIN_FEATS, "spread_line"),
                                   ("total", "total", L.TOTAL_FEATS, "total_line")]:
        # closing-trained predictions with CLOSING line as input -> reuse fitted pipeline
        # by refitting per season but predicting with the early line substituted.
        d0 = L.prep(f); d0 = d0[d0.home_score.notna() & d0[target].notna()]
        feats = usable_at(feats, asof)
        cols = feats + [col]
        out = []
        for s in sorted(d0.season.unique()):
            if s < max(2020, 2015):
                continue
            tr = d0[(d0.season >= L.TRAIN_START) & (d0.season < s)]
            te = d0[d0.season == s].merge(
                snap[["game_id", col] + odds].rename(columns={c: "e_" + c for c in [col] + odds}),
                on="game_id").reset_index(drop=True)
            if te.empty:
                continue
            m = L.ridge() if kind == "ridge" else None
            m.fit(tr[cols], tr[target])
            t = te.copy()
            x = t[cols].copy(); x[col] = t["e_" + col]
            t["pred"] = m.predict(x)
            out.append(t)
        d = pd.concat(out).reset_index(drop=True)
        # settle vs early line & early odds
        e = d.copy(); e[col] = d["e_" + col]
        for o in odds:
            e[o] = d["e_" + o]
        for t in (0, 1, 2, 3):
            b = L.settle_lines(e, mk, col, t)
            if len(b) < 20:
                continue
            clv = np.where(b.up, d.loc[b.index, col] - d.loc[b.index, "e_" + col],
                           d.loc[b.index, "e_" + col] - d.loc[b.index, col])
            p = b.profit.values
            rng = np.random.default_rng(seed)
            bs = [rng.choice(p, len(p)).mean() for _ in range(2000)]
            rows.append(dict(market=mk, threshold=t, bets=len(b), roi=p.mean(),
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             mean_clv_pts=clv.mean(), clv_se=clv.std() / len(clv) ** .5,
                             pct_line_moved_our_way=(clv > 0).mean(), pct_moved_against=(clv < 0).mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = run()
    REPORT_DIR.mkdir(exist_ok=True)
    r.to_csv(REPORT_DIR / "lines_early_clv.csv", index=False)
    print(r.round(3).to_string(index=False))
