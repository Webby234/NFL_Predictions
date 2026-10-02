"""Recompute the TRUST factors in board.py:  python -m nfl_model.ui.calibrate

For each market, a one-parameter fit on walk-forward test seasons: how much of the gap between the model's
chance and the sportsbook's chance shows up in results. Needs reports/predictions.csv (python -m nfl_model backtest).
The factor is fitted on the same seasons it is reported on, so read it as a rough guide, not a guarantee.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from ..config import REPORT_DIR, TRAIN_START
from ..evaluation import lines_backtest as L
from ..features import load_feature_table
from .board import _two_way

lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def _fit(X, y, col, n_boot=300, seed=0):
    f = lambda X_, y_: LogisticRegression(C=1e6, fit_intercept=False).fit(X_, y_).coef_[0][col]
    rng = np.random.default_rng(seed)
    bs = [f(X[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return dict(trust=f(X, y), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n=len(y))


def run() -> pd.DataFrame:
    rows = []
    d = pd.read_csv(REPORT_DIR / "predictions.csv").dropna(subset=["p_logit_with_spread", "p_market_ml", "home_win"])
    X = np.column_stack([lg(d.p_market_ml), lg(d.p_logit_with_spread) - lg(d.p_market_ml)])
    rows.append(dict(market="Moneyline", **_fit(X, d.home_win.values, 1)))
    f = L.prep(load_feature_table()); f = f[(f.game_type == "REG") & f.home_score.notna()]
    for name, tgt, feats, line in (("Spread", "result", L.MARGIN_FEATS, "spread_line"),
                                   ("Total", "total", L.TOTAL_FEATS, "total_line")):
        P, Y = [], []
        for s in sorted(f.season.unique()):
            tr = f[(f.season >= TRAIN_START) & (f.season < s)].dropna(subset=[tgt, line])
            te = f[f.season == s].dropna(subset=[tgt, line])
            if s < 2015 or te.empty:
                continue
            cols = feats + [line]; m = L.ridge().fit(tr[cols], tr[tgt]); res = (tr[tgt] - m.predict(tr[cols])).values
            for p, l, y in zip(m.predict(te[cols]), te[line], te[tgt]):
                o, u = _two_way(p, res, l)
                if y != l:
                    P.append(o / (o + u)); Y.append(float(y > l))
        rows.append(dict(market=name, **_fit(lg(np.array(P)).reshape(-1, 1), np.array(Y), 0)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = run(); r.round(3).to_csv(REPORT_DIR / "ui_trust.csv", index=False); print(r.round(2).to_string(index=False))
