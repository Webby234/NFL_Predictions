"""Stat configs and walk-forward evaluation for rushing, receiving, receptions, passing-TD and anytime-TD props.
Scored on outcomes only (no free historical prop prices): error, likelihood, calibration, vs a form-only baseline."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from .models import QUANTILES, fit_binary, fit_count, fit_dist
from .skill import CTX, build_skill_table

TRAIN_START, FIRST_TEST = 2008, 2015
LINE_CTX = ["team_implied_r", "opp_implied_r", "spread_team", "total_r"]
VENUE = ["is_home", "dome", "wind_eff", "temp_eff"]
POSF = ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]


@dataclass
class Stat:
    name: str
    kind: str                # cont | count | binary
    target: str
    base: str                # form-only baseline column
    feats: list
    elig: callable
    floor: float = 5.0


STATS = {
    "rush_yds": Stat("rush_yds", "cont", "rushing_yards", "p_rush_yds",
                     ["p_rush_yds", "p_carries", "p_targets", "team_carries_rel", "opp_rush_allowed_rel", "games"] + POSF,
                     lambda d: d.p_carries >= 5, 6.0),
    "rec_yds": Stat("rec_yds", "cont", "receiving_yards", "p_rec_yds",
                    ["p_rec_yds", "p_targets", "p_receptions", "team_att_rel", "opp_rec_allowed_rel", "games"] + POSF,
                    lambda d: d.p_targets >= 3.5, 6.0),
    "receptions": Stat("receptions", "count", "receptions", "p_receptions",
                       ["p_receptions", "p_targets", "p_rec_yds", "team_att_rel", "opp_rec_allowed_rel", "games"] + POSF,
                       lambda d: d.p_targets >= 3.5),
    "pass_tds": Stat("pass_tds", "count", "passing_tds", "p_pass_tds",
                     ["p_pass_tds", "p_pass_att", "team_att_rel", "opp_rec_allowed_rel", "games"],
                     lambda d: (d.position == "QB") & (d.p_pass_att >= 15)),
    "anytime_td": Stat("anytime_td", "binary", "any_td", "p_any_td",
                       ["p_any_td", "p_carries", "p_targets", "p_rush_yds", "p_rec_yds", "team_carries_rel",
                        "team_att_rel", "games"] + POSF,
                       lambda d: (d.p_carries >= 5) | (d.p_targets >= 3.5)),
}


def feats_for(stat: Stat, with_lines: bool):
    return stat.feats + VENUE + (LINE_CTX if with_lines else [])


def eligible(table: pd.DataFrame, stat: Stat) -> pd.DataFrame:
    t = table[stat.elig(table) & (table.games >= 2)]
    return t.dropna(subset=["team_implied_r"])


def naive_line(stat: Stat, base):
    base = np.asarray(base, float)
    return np.round(base * 2 - 0.5) / 2 if stat.kind == "cont" else np.floor(np.maximum(base, 0)) + 0.5


def walk_forward(table: pd.DataFrame, stat: Stat, first_test=FIRST_TEST, cont_kind="ridge") -> pd.DataFrame:
    t = eligible(table, stat)
    t = t[t[stat.target].notna()]
    outs = []
    for s in sorted(t.season.unique()):
        if s < first_test:
            continue
        tr = t[(t.season >= TRAIN_START) & (t.season < s)]; te = t[t.season == s].copy()
        for tag, wl in (("no_lines", False), ("with_lines", True)):
            f = feats_for(stat, wl)
            if stat.kind == "cont":
                m = fit_dist(tr, f, stat.target, cont_kind, floor=stat.floor)
                mu = m.mean(te); te[f"mean_{tag}"] = mu
                for q in QUANTILES:
                    te[f"q{int(q*100)}_{tag}"] = m.quantile(mu, q)
                te[f"pover_{tag}"] = m.prob_over(mu, naive_line(stat, te[stat.base]))
            elif stat.kind == "count":
                m = fit_count(tr, f, stat.target); mu = m.mean(te); te[f"mean_{tag}"] = mu
                te[f"pover_{tag}"] = m.prob_over(mu, naive_line(stat, te[stat.base]))
                te[f"disp_{tag}"] = m.disp
            else:
                te[f"prob_{tag}"] = fit_binary(tr, f, stat.target).prob(te)
        outs.append(te)
    return pd.concat(outs)


def _pinball(y, q_pred, q):
    d = y - q_pred
    return np.maximum(q * d, (q - 1) * d)


def summarize(out: pd.DataFrame, stat: Stat) -> pd.DataFrame:
    y = out[stat.target].values.astype(float); rows = []
    if stat.kind == "binary":
        base = np.clip(out[stat.base].values, 0.01, 0.95); const = np.full(len(y), y.mean())
        ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
        l0 = ll(base)
        for name, p in [("baseline_form", base), ("no_lines", out.prob_no_lines.values),
                        ("with_lines", out.prob_with_lines.values)]:
            d = ll(p) - l0
            rows.append(dict(stat=stat.name, model=name, n=len(y), log_loss=ll(p).mean(), brier=((p - y) ** 2).mean(),
                             auc=roc_auc_score(y, p), logloss_diff_vs_form=d.mean(), se=d.std() / np.sqrt(len(d))))
        return pd.DataFrame(rows)
    base = out[stat.base].values
    for name, mu in [("baseline_form", base), ("no_lines", out.mean_no_lines.values), ("with_lines", out.mean_with_lines.values)]:
        r = dict(stat=stat.name, model=name, n=len(y), mae=np.abs(y - mu).mean(), bias=(y - mu).mean())
        d = (y - mu) ** 2 - (y - base) ** 2
        r["mse_diff_vs_form"] = d.mean(); r["se"] = d.std() / np.sqrt(len(d))
        if stat.kind == "cont" and name != "baseline_form":
            tag = name
            r["pinball_avg"] = np.mean([_pinball(y, out[f"q{int(q*100)}_{tag}"].values, q).mean() for q in QUANTILES])
            r["cover_80"] = ((y >= out[f"q10_{tag}"]) & (y <= out[f"q90_{tag}"])).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def calibration(out: pd.DataFrame, stat: Stat, tag="with_lines", bins=None) -> pd.DataFrame:
    if stat.kind == "binary":
        p = out[f"prob_{tag}"]; o = out[stat.target].astype(float)
        bins = bins or [0, .1, .2, .3, .4, .5, .7, 1.0]
    else:
        line = naive_line(stat, out[stat.base]); y = out[stat.target]
        keep = (y != line).values; p = out[f"pover_{tag}"][keep]; o = (y > line)[keep].astype(float)
        bins = bins or [0, .2, .35, .45, .55, .65, .8, 1.0]
    b = pd.cut(p, bins, include_lowest=True)
    return pd.DataFrame(dict(n=o.groupby(b, observed=True).size(), predicted=p.groupby(b, observed=True).mean(),
                             actual=o.groupby(b, observed=True).mean())).round(3)


def run_all(table=None, stats=STATS):
    table = build_skill_table() if table is None else table
    res = {}
    for k, st in stats.items():
        out = walk_forward(table, st)
        res[k] = (out, summarize(out, st), calibration(out, st))
    return res
