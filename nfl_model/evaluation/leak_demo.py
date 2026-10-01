"""One-off: reproduce the ORIGINAL evaluation style (random split, team averages that
include the games being predicted) to check whether it inflated results."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from ..features.team_features import TEAM_CODE_FIX, _team_game_table
from ..models import make_gbm
from .metrics import EPS


def leak_demo(games: pd.DataFrame, stats: pd.DataFrame, gbm_kind: str) -> None:
    """Re-create the ORIGINAL main-branch evaluation style to show how it inflates results:
    team averages computed over 2022-2024 (including the games being predicted, and
    applied to 2020-2021 games too) + a random 80/20 split. Compared with an honest
    walk-forward on the same 2022-2024 games."""
    from sklearn.model_selection import train_test_split
    from features import TEAM_CODE_FIX, _team_game_table

    g = games[(games.game_type == "REG") & games.season.between(2020, 2024)].copy()
    g[["home_team", "away_team"]] = g[["home_team", "away_team"]].replace(TEAM_CODE_FIX)
    g = g[g.result.notna() & (g.result != 0)]
    g["home_win"] = (g.result > 0).astype(int)

    # team-level averages over 2022-2024 that INCLUDE the games being predicted
    season_of = stats.assign(team=stats.team.replace(TEAM_CODE_FIX))[["game_id", "team", "season"]]
    tg = _team_game_table(stats).merge(season_of, on=["game_id", "team"])
    avg = tg[tg.season.between(2022, 2024)].groupby("team")[["yds", "giveaways"]].mean()
    reg = games[games.season.between(2022, 2024) & (games.game_type == "REG")]
    pts = pd.concat([
        reg[["home_team", "home_score"]].set_axis(["team", "pts"], axis=1),
        reg[["away_team", "away_score"]].set_axis(["team", "pts"], axis=1)])
    avg["pts"] = pts.assign(team=pts.team.replace(TEAM_CODE_FIX)).groupby("team")["pts"].mean()
    for side in ("home", "away"):
        for c in ("yds", "giveaways", "pts"):
            g[f"{side}_{c}"] = g[f"{side}_team"].map(avg[c])
    g["yard_diff"] = g.home_yds - g.away_yds
    g["turnover_diff"] = g.away_giveaways - g.home_giveaways
    cols = ["home_pts", "away_pts", "yard_diff", "turnover_diff", "spread_line"]
    g = g.dropna(subset=cols)
    for label, use in (("with spread", cols), ("team stats only", cols[:-1])):
        tr, te = train_test_split(g, test_size=0.2, random_state=42)
        m = make_gbm(gbm_kind).fit(tr[use], tr.home_win)
        p = m.predict_proba(te[use])[:, 1]
        print(f"  ORIGINAL-STYLE ({label:15s}): accuracy {accuracy_score(te.home_win, p > .5):.3f}  log loss {log_loss(te.home_win, np.clip(p, EPS, 1-EPS)):.4f}")


