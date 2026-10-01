"""Walk-forward spread (margin) and totals models, scored vs market lines.

Targets: margin = home_score-away_score (market: spread_line, + = home favored);
total = home+away (market: total_line). Bets are settled at the listed side odds
(non-standard/NaN odds -> -110); pushes refund.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..config import TRAIN_START, FIRST_TEST
from ..features import NO_MARKET, QB_FEATURES
from ..betting.odds import american_to_decimal

MARGIN_FEATS = NO_MARKET + QB_FEATURES
TOTAL_FEATS = ["d_pts_for", "d_pts_against", "h_pts_for", "a_pts_for", "h_pts_against",
               "a_pts_against", "h_epa_for", "a_epa_for", "h_epa_against", "a_epa_against",
               "dome", "wind_eff", "temp_eff", "min_games"]


def prep(df):
    df = df.copy()
    df["dome"] = df["roof"].isin(["dome", "closed"]).astype(float)
    df["wind_eff"] = np.where(df.dome == 1, 0.0, df["wind"])
    df["temp_eff"] = np.where(df.dome == 1, 70.0, df["temp"])
    return df


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=50))


def gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=200,
                                         min_samples_leaf=40, l2_regularization=5.0)


def _dec(odds):
    o = pd.to_numeric(odds, errors="coerce")
    o = o.where((o.abs() >= 100) & (o.abs() <= 400), -110)
    return american_to_decimal(o)


def walk_forward(df, target, feats, with_line=None, kind="ridge", first_test=FIRST_TEST):
    """Return df with column pred (+ resid sigma) for each test game."""
    df = prep(df)
    df = df[df.home_score.notna() & df[target].notna()].copy()
    cols = list(feats) + ([with_line] if with_line else [])
    out = []
    for s in sorted(df.season.unique()):
        if s < first_test:
            continue
        tr = df[(df.season >= TRAIN_START) & (df.season < s) & df[cols[0]].notna()]
        te = df[df.season == s]
        m = ridge() if kind == "ridge" else make_pipeline(SimpleImputer(strategy="median"), gbm())
        m.fit(tr[cols], tr[target])
        sig = float(np.std(tr[target] - m.predict(tr[cols])))
        t = te.copy()
        t["pred"] = m.predict(te[cols])
        t["sigma"] = sig
        out.append(t)
    return pd.concat(out)


def settle_lines(d, kind, market_col, thresh, pred_col="pred"):
    """kind='spread' or 'total'. Returns per-bet frame with profit (1 unit stake)."""
    diff = d[pred_col] - d[market_col]
    if kind == "spread":
        outcome = d["home_score"] - d["away_score"] - d[market_col]   # >0 home covers
        side_up, side_dn = "home_spread_odds", "away_spread_odds"
    else:
        outcome = d["home_score"] + d["away_score"] - d[market_col]   # >0 over
        side_up, side_dn = "over_odds", "under_odds"
    up = diff > thresh
    dn = diff < -thresh
    b = d[up | dn].copy()
    b["up"] = up[up | dn]
    o = outcome[b.index]
    win = np.where(b.up, o > 0, o < 0)
    push = o == 0
    dec = np.where(b.up, _dec(b[side_up]), _dec(b[side_dn]))
    b["profit"] = np.where(push, 0.0, np.where(win, dec - 1, -1.0))
    return b


def roi_table(d, kind, market_col, thresholds=(0, 1, 2, 3, 4), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in thresholds:
        b = settle_lines(d, kind, market_col, t)
        if len(b) < 20:
            continue
        p = b.profit.values
        bs = [rng.choice(p, len(p)).mean() for _ in range(2000)]
        rows.append(dict(threshold=t, bets=len(b), roi=p.mean(), lo=np.percentile(bs, 2.5),
                         hi=np.percentile(bs, 97.5), win_rate=(p > 0).sum() / max((p != 0).sum(), 1)))
    return pd.DataFrame(rows)


def mse_vs_market(d, target_expr, market_col):
    err_m = target_expr - d["pred"]
    err_k = target_expr - d[market_col]
    dd = err_m**2 - err_k**2
    return dict(mse_model=(err_m**2).mean(), mse_market=(err_k**2).mean(),
                diff=dd.mean(), se=dd.std() / len(dd) ** .5, n=len(dd))
