"""Walk-forward evaluation of QB passing-yards models. Scored on outcomes, not betting returns."""
from __future__ import annotations
import numpy as np, pandas as pd
from .features import BEST, NO_LINES, WITH_LINES, build_qb_prop_table
from .models import QUANTILES, fit_dist

TRAIN_START, FIRST_TEST = 2008, 2015
ZOO = {"ridge_no_lines": (NO_LINES, "ridge"), "ridge_with_lines": (WITH_LINES, "ridge"),
       "gbm_with_lines": (WITH_LINES, "gbm"), "blend_with_lines": (BEST, "blend")}
APP_MODEL = "blend_with_lines"          # what the app shows


def pinball(y, q_pred, q):
    d = y - q_pred
    return np.maximum(q * d, (q - 1) * d)


def walk_forward(t: pd.DataFrame, first_test: int = FIRST_TEST, zoo=ZOO):
    t = t[t.passing_yards.notna()].copy()
    preds, dists = [], {}
    for s in sorted(t.season.unique()):
        if s < first_test:
            continue
        tr = t[(t.season >= TRAIN_START) & (t.season < s)].dropna(subset=["team_implied_r"])
        te = t[t.season == s].copy()
        te["pred_baseline_qb"] = te.qb_yds
        for name, (feats, kind) in zoo.items():
            d = fit_dist(tr, feats, kind=kind)
            mu = d.mean(te)
            te[f"pred_{name}"] = mu
            te[f"sig_{name}"] = d.sigma(mu)
            for q in QUANTILES:
                te[f"q{int(q*100)}_{name}"] = d.quantile(mu, q)
            # prob over a naive "book-like" line: baseline mean rounded to .5
            te[f"pover_{name}"] = d.prob_over(mu, np.round(te.qb_yds * 2 - 0.5) / 2 + 0.5 * 0)
            dists[(s, name)] = d
        preds.append(te)
    return pd.concat(preds), dists


def summarize(out: pd.DataFrame, models=("baseline_qb",) + tuple(ZOO)) -> pd.DataFrame:
    y = out.passing_yards.values; base_se = (y - out.pred_baseline_qb) ** 2
    rows = []
    for m in models:
        mu = out[f"pred_{m}"]; se = (y - mu) ** 2
        r = dict(model=m, n=len(out), mae=np.abs(y - mu).mean(), rmse=np.sqrt(se.mean()))
        dse = se - base_se
        r["mse_diff_vs_baseline"] = dse.mean(); r["se"] = dse.std() / np.sqrt(len(dse))
        if m != "baseline_qb":
            r["pinball_avg"] = np.mean([pinball(y, out[f"q{int(q*100)}_{m}"].values, q).mean() for q in QUANTILES])
            r["cover_80"] = ((y >= out[f"q10_{m}"]) & (y <= out[f"q90_{m}"])).mean()
            r["cover_50"] = ((y >= out[f"q25_{m}"]) & (y <= out[f"q75_{m}"])).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def over_calibration(out: pd.DataFrame, m: str, bins=(0, .2, .35, .45, .55, .65, .8, 1.0)) -> pd.DataFrame:
    line = np.round(out.qb_yds * 2 - 0.5) / 2 + 0.0
    over = (out.passing_yards > line).astype(float); keep = out.passing_yards != line
    p = out[f"pover_{m}"][keep]; o = over[keep]
    b = pd.cut(p, list(bins), include_lowest=True)
    return pd.DataFrame(dict(n=o.groupby(b, observed=True).size(), predicted=p.groupby(b, observed=True).mean(),
                             actual=o.groupby(b, observed=True).mean())).round(3)


def by_season(out, m=APP_MODEL):
    return out.groupby("season").apply(lambda d: pd.Series(dict(
        n=len(d), mae_model=np.abs(d.passing_yards - d[f"pred_{m}"]).mean(),
        mae_baseline=np.abs(d.passing_yards - d.pred_baseline_qb).mean())), include_groups=False).round(2)
