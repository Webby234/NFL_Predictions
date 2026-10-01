"""Walk-forward backtest for win-probability models.

For each test season S: train on regular-season games from TRAIN_START to S-1,
predict every game in S. Features are pre-game only (see features/), so nothing
from the future reaches a prediction. All models and baselines are scored on the
SAME games.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from ..features import WITH_SPREAD
from ..config import FIRST_TEST, REPORT_DIR, TRAIN_START
from ..models import fit, model_zoo, predict_home_prob
from .metrics import EPS, paired_logloss_diff, score

def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Regular-season, finished, non-tie games with everything needed by every row."""
    e = df[(df.game_type == "REG") & df.home_win.notna()].copy()
    return e.dropna(subset=["spread_line", "market_ml_prob", "elo_prob"])


def walk_forward(df: pd.DataFrame, gbm_kind: str = "sklearn", zoo: dict | None = None,
                 first_test: int = FIRST_TEST) -> pd.DataFrame:
    e = eligible(df)
    zoo = zoo if zoo is not None else model_zoo(gbm_kind)
    preds = []
    for season in sorted(e.season[e.season >= first_test].unique()):
        test = e[e.season == season].copy()
        for name, spec in zoo.items():
            train = e[(e.season >= spec.train_start) & (e.season < season)]
            test[f"p_{name}"] = predict_home_prob(fit(spec, train), spec, test)
        train_all = e[(e.season >= TRAIN_START) & (e.season < season)]
        test["p_home_rate"] = train_all["home_win"].mean()
        preds.append(test)
    out = pd.concat(preds)
    out["p_elo"] = out["elo_prob"]
    out["p_market_ml"] = out["market_ml_prob"]
    return out


PRED_ORDER = ["home_rate", "elo", "logit_no_market", "logit_no_market_qb", "gbm_no_market", "market_spread",
              "market_ml", "logit_with_spread", "gbm_with_spread"]


def summarize(out: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = out["home_win"].to_numpy(float)
    rows = []
    for name in PRED_ORDER:
        s = score(y, out[f"p_{name}"].to_numpy(float))
        d, se = paired_logloss_diff(y, out[f"p_{name}"].to_numpy(float), out["p_market_ml"].to_numpy(float))
        s.update(model=name, ll_vs_market=d, ll_vs_market_se=se)
        rows.append(s)
    pooled = pd.DataFrame(rows).set_index("model")

    by_season = []
    for season, grp in out.groupby("season"):
        yy = grp["home_win"].to_numpy(float)
        r = {"season": int(season), "n": len(grp)}
        for name in PRED_ORDER:
            r[name] = log_loss(yy, np.clip(grp[f"p_{name}"], EPS, 1 - EPS))
        by_season.append(r)
    return pooled, pd.DataFrame(by_season).set_index("season")


def calibration_table(out: pd.DataFrame, name: str, bins=10) -> pd.DataFrame:
    p, y = out[f"p_{name}"].to_numpy(float), out["home_win"].to_numpy(float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = [{"bin": f"{b/bins:.1f}-{(b+1)/bins:.1f}", "n": int((idx == b).sum()),
             "avg_predicted": p[idx == b].mean(), "actual_win_rate": y[idx == b].mean()}
            for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def plot_calibration(out: pd.DataFrame, path: Path, names=("logit_no_market", "market_ml", "gbm_no_market")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name in names:
        t = calibration_table(out, name, bins=8)
        ax.plot(t.avg_predicted, t.actual_win_rate, "o-", label=name)
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Actual home-win rate")
    ax.set_title(f"Calibration, walk-forward {out.season.min()}-{out.season.max()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


