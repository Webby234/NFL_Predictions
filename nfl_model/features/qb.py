"""Quarterback features (leak-free).

Rating = decayed EPA per dropback, shrunk toward a prior so a backup with one good game is not
over-trusted. State is snapshotted BEFORE each game and updated afterwards, like the team features.

Which QB starts?
  * played games / announced upcoming games: the starter listed in games.csv (known at kickoff)
  * unannounced upcoming games: the team's last starter, unless the injury report lists him
    Out/Doubtful, in which case the team's most recent other QB (or a replacement-level unknown).

Constants are fixed a priori from 1999-2005 data (before any test season), not tuned on results:
starters averaged +0.015 EPA/dropback and low-volume backups -0.149.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .team_features import TEAM_CODE_FIX

QB_PRIOR = -0.05             # EPA/dropback assumed for a QB with no history (new starters skew below average)
QB_PRIOR_STRENGTH = 200.0    # pseudo-dropbacks behind the prior (~5-6 games)
QB_HALFLIFE = 24.0           # games (of that QB's own) for his old games to lose half their weight
QB_FEATURES = ["qb_diff", "qb_exp_diff", "qb_drop_diff"]
OUT_STATUSES = {"Out", "Doubtful"}


def _season_type(game_type: str) -> str:
    return "REG" if game_type == "REG" else "POST"


def add_qb_features(out: pd.DataFrame, qb_games: pd.DataFrame, injuries: pd.DataFrame | None = None) -> pd.DataFrame:
    """`out` = chronologically sorted output of build_features (needs home/away_qb_id, game_type, scores)."""
    decay = 0.5 ** (1.0 / QB_HALFLIFE)

    q = qb_games.copy()
    q["team"] = q["team"].replace(TEAM_CODE_FIX)
    q["db"] = q["attempts"] + q["sacks_suffered"].fillna(0)
    q["epa"] = q["passing_epa"].fillna(0.0)
    by_team_week = {k: list(zip(g["player_id"], g["epa"], g["db"]))
                    for k, g in q.groupby(["team", "season", "week", "season_type"])}

    qb_out: dict[tuple, set] = defaultdict(set)         # (season, week, team) -> QBs listed Out/Doubtful
    if injuries is not None and not injuries.empty:
        i = injuries[(injuries["position"] == "QB") & injuries["report_status"].isin(OUT_STATUSES)]
        for r in i.itertuples(index=False):
            qb_out[(int(r.season), int(r.week), TEAM_CODE_FIX.get(r.team, r.team))].add(r.gsis_id)

    s_epa: dict[str, float] = defaultdict(float)
    s_db: dict[str, float] = defaultdict(float)
    starts: dict[str, int] = defaultdict(int)
    last_qb: dict[str, str] = {}
    team_qbs: dict[str, dict[str, int]] = defaultdict(dict)    # team -> {qb: order of last game}

    def rating(qid):
        if qid is None:
            return QB_PRIOR
        return (s_epa[qid] + QB_PRIOR_STRENGTH * QB_PRIOR) / (s_db[qid] + QB_PRIOR_STRENGTH)

    def pick_starter(team, listed, season, week):
        if isinstance(listed, str):
            return listed
        last = last_qb.get(team)
        if last is not None and last in qb_out[(season, week, team)]:
            others = [(o, k) for k, o in team_qbs[team].items() if k != last and k not in qb_out[(season, week, team)]]
            return max(others)[1] if others else None
        return last

    cols = {k: [] for k in ("qb_rating_h", "qb_rating_a", "qb_diff", "qb_exp_diff", "qb_drop_diff",
                            "qb_changed_h", "qb_changed_a", "qb_id_h", "qb_id_a")}
    order = 0

    def apply_update(r):
        """Fold a finished game into the QB state."""
        nonlocal order
        season, week = int(r.season), int(r.week)
        order += 1
        st = _season_type(r.game_type)
        for side, team in (("h", r.home_team), ("a", r.away_team)):
            for qid, epa, db in by_team_week.get((team, season, week, st), []):
                s_epa[qid] = s_epa[qid] * decay + epa
                s_db[qid] = s_db[qid] * decay + db
                team_qbs[team][qid] = order
            listed = r.home_qb_id if side == "h" else r.away_qb_id
            if isinstance(listed, str):
                starts[listed] += 1
                last_qb[team] = listed
                team_qbs[team].setdefault(listed, order)

    # Snapshot every game of a (season, week) BEFORE applying any of that week's results. A QB's state can
    # cross teams (e.g. a QB who just changed teams), so same-week games must not see each other.
    pending, current_week = [], None
    for r in out.itertuples(index=False):
        season, week = int(r.season), int(r.week)
        if (season, week) != current_week:
            for done in pending:
                apply_update(done)
            pending, current_week = [], (season, week)
        snap = {}
        for side, team, listed in (("h", r.home_team, r.home_qb_id), ("a", r.away_team, r.away_qb_id)):
            qid = pick_starter(team, listed, season, week)
            last = last_qb.get(team)
            snap[side] = dict(qid=qid, rating=rating(qid), exp=starts[qid] if qid else 0,
                              prev=rating(last) if last else rating(qid), changed=int(last is not None and qid != last))
        h, a = snap["h"], snap["a"]
        cols["qb_rating_h"].append(h["rating"]); cols["qb_rating_a"].append(a["rating"])
        cols["qb_diff"].append(h["rating"] - a["rating"])
        cols["qb_exp_diff"].append(math.log1p(h["exp"]) - math.log1p(a["exp"]))
        cols["qb_drop_diff"].append((h["rating"] - h["prev"]) - (a["rating"] - a["prev"]))
        cols["qb_changed_h"].append(h["changed"]); cols["qb_changed_a"].append(a["changed"])
        cols["qb_id_h"].append(h["qid"]); cols["qb_id_a"].append(a["qid"])
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            pending.append(r)
    res = out.copy()
    for k, v in cols.items():
        res[k] = v
    return res
