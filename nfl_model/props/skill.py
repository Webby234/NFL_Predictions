"""Leak-free pre-game features for rushing / receiving / touchdown props (QB, RB, WR, TE).

One row per player-game. Everything is measured relative to the league level of that era and snapshotted
BEFORE the week's results update the state (same design as the QB passing model):
  p_<metric>   player's shrunk recent form (EWMA over games, prior = position-group league level)
  team_*_rel   own team's recent carries / pass attempts vs league
  opp_*_rel    yards the opponent has been allowing (rush / receiving) vs league
  context      implied points, spread, total (relative), home, dome, wind, temperature
Eligibility for a prop is decided from PRE-game expected volume (not same-game volume, which would
select on the outcome). Players who did not appear in the stats file that week (DNP) have no row.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ..data import load_games, load_skill_games
from ..features.team_features import TEAM_CODE_FIX
from .features import GAMES_CAP, team_game_context

HALFLIFE = 8.0
PRIOR_GAMES = 3.0
LG_WEEK_HL = 12.0
TEAM_HL = 8.0
LAM, LG_D, TLAM = 0.5 ** (1 / HALFLIFE), 0.5 ** (1 / LG_WEEK_HL), 0.5 ** (1 / TEAM_HL)
POS = ["QB", "RB", "WR", "TE"]
METRICS = {"rush_yds": "rushing_yards", "carries": "carries", "rec_yds": "receiving_yards", "targets": "targets",
           "receptions": "receptions", "pass_tds": "passing_tds", "pass_att": "attempts"}
METRIC_NAMES = list(METRICS) + ["any_td"]
CTX = ["is_home", "dome", "wind_eff", "temp_eff", "team_implied_r", "opp_implied_r", "spread_team", "total_r"]
BASE = ["games", "pos_QB", "pos_RB", "pos_WR", "pos_TE", "team_carries_rel", "team_att_rel",
        "opp_rush_allowed_rel", "opp_rec_allowed_rel"]


def _prep(skill: pd.DataFrame) -> pd.DataFrame:
    d = skill.copy()
    d["team"] = d.team.replace(TEAM_CODE_FIX); d["opponent_team"] = d.opponent_team.replace(TEAM_CODE_FIX)
    d["any_td"] = ((d.rushing_tds + d.receiving_tds) > 0).astype(float)
    return d


def build_skill_table(games: pd.DataFrame | None = None, skill: pd.DataFrame | None = None) -> pd.DataFrame:
    games = load_games() if games is None else games
    d = _prep(load_skill_games() if skill is None else skill)
    ctx = team_game_context(games)
    d = d.merge(ctx[["season", "week", "team", "is_home", "dome", "wind_eff", "temp_eff", "total_line",
                     "spread_team", "team_implied", "opp_implied"]], on=["season", "week", "team"], how="left")
    d = d.sort_values(["season", "week"]).reset_index(drop=True)
    vals = {m: d[c].values.astype(float) for m, c in METRICS.items()}
    vals["any_td"] = d.any_td.values.astype(float)
    pos = d.position.values; pid = d.player_id.values; team = d.team.values; opp = d.opponent_team.values
    wk_idx = d.groupby(["season", "week"]).indices

    player = {}                                     # id -> [w, n, {m: dev}]
    lgp = {p: [0.0, {m: 0.0 for m in METRIC_NAMES}] for p in POS}       # per pos: decayed rows, decayed sums
    tstate, ostate = {}, {}                          # team -> {carries:[s,w], att:[s,w]};  defense -> {rush,rec:[s,w]}
    lgt = {"carries": [0.0, 0.0], "att": [0.0, 0.0], "rush": [0.0, 0.0], "rec": [0.0, 0.0], "tot": [0.0, 0.0]}
    feats = {k: np.full(len(d), np.nan) for k in [f"p_{m}" for m in METRIC_NAMES] + BASE + ["total_r"]}
    team_implied_r = np.full(len(d), np.nan); opp_implied_r = np.full(len(d), np.nan)
    lvl = lambda a: a[0] / a[1] if a[1] else np.nan

    for (s, w), idx in sorted(wk_idx.items()):
        lg_t = {k: lvl(v) for k, v in lgt.items()}
        lgm = {p: {m: (lgp[p][1][m] / lgp[p][0] if lgp[p][0] else 0.0) for m in METRIC_NAMES} for p in POS}
        # ---- snapshot features for every row of this week
        for i in idx:
            st = player.get(pid[i]); p = pos[i]
            n_eff = st[0] if st else 0.0
            for m in METRIC_NAMES:
                feats[f"p_{m}"][i] = lgm[p][m] + (st[2][m] / (n_eff + PRIOR_GAMES) if st else 0.0)
            feats["games"][i] = min(st[1], GAMES_CAP) if st else 0
            for q in POS:
                feats[f"pos_{q}"][i] = 1.0 if p == q else 0.0
            ts = tstate.get(team[i]); os_ = ostate.get(opp[i])
            sh = lambda a, lg: (lg + (a[0] / (a[1] + PRIOR_GAMES)) - 0.0) if a else lg
            feats["team_carries_rel"][i] = (ts["carries"][0] / (ts["carries"][1] + PRIOR_GAMES)) if ts else 0.0
            feats["team_att_rel"][i] = (ts["att"][0] / (ts["att"][1] + PRIOR_GAMES)) if ts else 0.0
            feats["opp_rush_allowed_rel"][i] = (os_["rush"][0] / (os_["rush"][1] + PRIOR_GAMES)) if os_ else 0.0
            feats["opp_rec_allowed_rel"][i] = (os_["rec"][0] / (os_["rec"][1] + PRIOR_GAMES)) if os_ else 0.0
            tot = lg_t["tot"] if not np.isnan(lg_t["tot"]) else 44.0
            feats["total_r"][i] = d.total_line.values[i] - tot
            team_implied_r[i] = d.team_implied.values[i] - tot / 2
            opp_implied_r[i] = d.opp_implied.values[i] - tot / 2
        # ---- update state with this week's results (placeholder rows have NaN yards -> skipped)
        done = [i for i in idx if not np.isnan(vals["rush_yds"][i])]
        for q in POS:                                # league level first decays, then adds this week
            lgp[q][0] *= LG_D
            for m in METRIC_NAMES:
                lgp[q][1][m] *= LG_D
        for i in done:
            p = pos[i]; lgp[p][0] += 1
            for m in METRIC_NAMES:
                lgp[p][1][m] += vals[m][i]
        for i in done:
            st = player.setdefault(pid[i], [0.0, 0, {m: 0.0 for m in METRIC_NAMES}])
            st[0] = LAM * st[0] + 1; st[1] += 1
            for m in METRIC_NAMES:
                st[2][m] = LAM * st[2][m] + (vals[m][i] - lgm[pos[i]][m])
        # team volume and defense allowed (per team-week sums over all rows)
        tw, ow = {}, {}
        for i in done:
            t = tw.setdefault(team[i], [0.0, 0.0]); t[0] += vals["carries"][i]; t[1] += vals["pass_att"][i]
            o = ow.setdefault(opp[i], [0.0, 0.0]); o[0] += vals["rush_yds"][i]; o[1] += vals["rec_yds"][i]
        for k in ("carries", "att", "rush", "rec"):
            lgt[k][0] *= LG_D; lgt[k][1] *= LG_D
        if tw:
            for t_, (c, a) in tw.items():
                lgt["carries"][0] += c; lgt["carries"][1] += 1; lgt["att"][0] += a; lgt["att"][1] += 1
            for t_, (r, c) in ow.items():
                lgt["rush"][0] += r; lgt["rush"][1] += 1; lgt["rec"][0] += c; lgt["rec"][1] += 1
        for t_, (c, a) in tw.items():
            ts = tstate.setdefault(t_, {"carries": [0.0, 0.0], "att": [0.0, 0.0]})
            ts["carries"][0] = TLAM * ts["carries"][0] + (c - lg_t["carries"] if not np.isnan(lg_t["carries"]) else 0.0)
            ts["carries"][1] = TLAM * ts["carries"][1] + 1
            ts["att"][0] = TLAM * ts["att"][0] + (a - lg_t["att"] if not np.isnan(lg_t["att"]) else 0.0)
            ts["att"][1] = TLAM * ts["att"][1] + 1
        for t_, (r, c) in ow.items():
            os_ = ostate.setdefault(t_, {"rush": [0.0, 0.0], "rec": [0.0, 0.0]})
            os_["rush"][0] = TLAM * os_["rush"][0] + (r - lg_t["rush"] if not np.isnan(lg_t["rush"]) else 0.0)
            os_["rush"][1] = TLAM * os_["rush"][1] + 1
            os_["rec"][0] = TLAM * os_["rec"][0] + (c - lg_t["rec"] if not np.isnan(lg_t["rec"]) else 0.0)
            os_["rec"][1] = TLAM * os_["rec"][1] + 1
        # league average total from this week's games
        g = ctx[(ctx.season == s) & (ctx.week == w)]
        lgt["tot"][0] = lgt["tot"][0] * LG_D + g.total_line.sum() / 2   # each game appears twice in ctx
        lgt["tot"][1] = lgt["tot"][1] * LG_D + len(g) / 2
    out = d.copy()
    for k, v in feats.items():
        out[k] = v
    out["team_implied_r"], out["opp_implied_r"] = team_implied_r, opp_implied_r
    return out
