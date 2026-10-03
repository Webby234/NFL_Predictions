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
           "receptions": "receptions", "pass_tds": "passing_tds", "pass_att": "attempts",
           "air_yds": "receiving_air_yards"}
DERIVED = ["any_td", "tgt_share", "carry_share"]     # computed per row in _prep
METRIC_NAMES = list(METRICS) + DERIVED
DEF_KEYS = ["rec_RB", "rec_WR", "rec_TE", "rush_RB", "td"]   # what a defense allows, split by who gained it
CTX = ["is_home", "dome", "wind_eff", "temp_eff", "team_implied_r", "opp_implied_r", "spread_team", "total_r"]
BASE = ["games", "pos_QB", "pos_RB", "pos_WR", "pos_TE", "team_carries_rel", "team_att_rel",
        "opp_rush_allowed_rel", "opp_rec_allowed_rel", "opp_pos_rec_rel", "opp_rb_rush_rel", "opp_td_rel", "p_snap"]


def _prep(skill: pd.DataFrame) -> pd.DataFrame:
    d = skill.copy()
    d["team"] = d.team.replace(TEAM_CODE_FIX); d["opponent_team"] = d.opponent_team.replace(TEAM_CODE_FIX)
    d["any_td"] = ((d.rushing_tds + d.receiving_tds) > 0).astype(float)
    if "receiving_air_yards" not in d:
        d["receiving_air_yards"] = 0.0
    tm = d.groupby(["season", "week", "team"])[["targets", "carries"]].transform("sum")
    d["tgt_share"] = (d.targets / tm.targets.where(tm.targets > 0)).fillna(0.0)      # share of the team's targets
    d["carry_share"] = (d.carries / tm.carries.where(tm.carries > 0)).fillna(0.0)
    return d


def _snap_pct(d: pd.DataFrame, snaps, players) -> np.ndarray:
    """Offensive snap share per row (NaN before 2013 or when the player can't be matched)."""
    if snaps is None or players is None or not len(snaps):
        return np.full(len(d), np.nan)
    s = snaps.merge(players, left_on="pfr_player_id", right_on="pfr_id")[["season", "week", "gsis_id", "offense_pct"]]
    s = s.drop_duplicates(["season", "week", "gsis_id"])
    return d[["season", "week", "player_id"]].merge(s, left_on=["season", "week", "player_id"],
                                                    right_on=["season", "week", "gsis_id"], how="left").offense_pct.values


def build_skill_table(games: pd.DataFrame | None = None, skill: pd.DataFrame | None = None,
                      snaps: pd.DataFrame | None = None, players: pd.DataFrame | None = None) -> pd.DataFrame:
    games = load_games() if games is None else games
    if skill is None and snaps is None:
        from ..data import load_players, load_snap_counts
        snaps, players = load_snap_counts(), load_players()
    d = _prep(load_skill_games() if skill is None else skill)
    ctx = team_game_context(games)
    d = d.merge(ctx[["season", "week", "team", "is_home", "dome", "wind_eff", "temp_eff", "total_line",
                     "spread_team", "team_implied", "opp_implied"]], on=["season", "week", "team"], how="left")
    d = d.sort_values(["season", "week"]).reset_index(drop=True)
    vals = {m: d[c].values.astype(float) for m, c in METRICS.items()}
    for m in DERIVED:
        vals[m] = d[m].values.astype(float)
    snap = _snap_pct(d, snaps, players)
    psnap = {}                                       # id -> [decayed sum, decayed weight]
    pos = d.position.values; pid = d.player_id.values; team = d.team.values; opp = d.opponent_team.values
    wk_idx = d.groupby(["season", "week"]).indices

    player = {}                                     # id -> [w, n, {m: dev}]
    lgp = {p: [0.0, {m: 0.0 for m in METRIC_NAMES}] for p in POS}       # per pos: decayed rows, decayed sums
    tstate, ostate = {}, {}                          # team -> {carries:[s,w], att:[s,w]};  defense -> {rush,rec:[s,w]}
    lgt = {"carries": [0.0, 0.0], "att": [0.0, 0.0], "rush": [0.0, 0.0], "rec": [0.0, 0.0], "tot": [0.0, 0.0]}
    lgt.update({k: [0.0, 0.0] for k in DEF_KEYS})
    dstate = {}                                      # defense -> {DEF_KEY: [dev sum, weight]}
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
            ds = dstate.get(opp[i])
            dv = lambda k: (ds[k][0] / (ds[k][1] + PRIOR_GAMES)) if ds else 0.0
            feats["opp_pos_rec_rel"][i] = dv(f"rec_{p}") if p != "QB" else 0.0
            feats["opp_rb_rush_rel"][i] = dv("rush_RB")
            feats["opp_td_rel"][i] = dv("td")
            sn = psnap.get(pid[i])
            feats["p_snap"][i] = sn[0] / sn[1] if sn else np.nan
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
        for i in done:
            if not np.isnan(snap[i]):
                sn = psnap.setdefault(pid[i], [0.0, 0.0])
                sn[0] = LAM * sn[0] + snap[i]; sn[1] = LAM * sn[1] + 1
        # what each defense allowed this week, split by the position that gained it
        dw = {}
        for i in done:
            a = dw.setdefault(opp[i], dict.fromkeys(DEF_KEYS, 0.0))
            if pos[i] != "QB":
                a[f"rec_{pos[i]}"] += vals["rec_yds"][i]; a["td"] += vals["any_td"][i]
            if pos[i] == "RB":
                a["rush_RB"] += vals["rush_yds"][i]
        for k in DEF_KEYS:
            lgt[k][0] *= LG_D; lgt[k][1] *= LG_D
        for t_, a in dw.items():
            st = dstate.setdefault(t_, {k: [0.0, 0.0] for k in DEF_KEYS})
            for k in DEF_KEYS:
                lgt[k][0] += a[k]; lgt[k][1] += 1
                st[k][0] = TLAM * st[k][0] + (a[k] - lg_t[k] if not np.isnan(lg_t[k]) else 0.0)
                st[k][1] = TLAM * st[k][1] + 1
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
