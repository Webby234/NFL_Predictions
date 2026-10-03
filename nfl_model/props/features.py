"""Leak-free pre-game features for QB passing-yards props.

One row per QB-game (attempts >= MIN_ATT, regular season). State (QB form, opponent pass defense,
league average) is snapshotted for every game of a week BEFORE that week's results update it.
Starter identity is the QB who actually played (known by game day; same as using the announced starter).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ..data import load_games, load_passing_games
from ..features.team_features import TEAM_CODE_FIX

MIN_ATT = 10
HALFLIFE = 8.0          # QB form, in games
OPP_HALFLIFE = 8.0
LG_HALFLIFE = 150.0     # league level, in QB-games (~a quarter season): tracks era drift
GAMES_CAP = 32          # cap career games: uncapped it grows with calendar time and drifts predictions
PRIOR_GAMES = 3.0       # pseudo-games of league-average shrinkage
LG_LAM = 0.5 ** (1 / LG_HALFLIFE)
LAM = 0.5 ** (1 / HALFLIFE)
OPP_LAM = 0.5 ** (1 / OPP_HALFLIFE)


QB_FEATS = ["qb_yds", "qb_att", "qb_ypa", "qb_games", "opp_yds_allowed", "is_home", "dome", "wind_eff", "temp_eff", "lg_yds"]
LINE_FEATS = ["team_implied_r", "opp_implied_r", "spread_team", "total_r"]
NO_LINES = ["qb_yds", "qb_att", "qb_ypa", "qb_games", "opp_yds_allowed", "is_home", "dome", "wind_eff", "temp_eff", "lg_yds"]
WITH_LINES = NO_LINES + LINE_FEATS
# Inputs borrowed from the skill-player table (teammates out, fast/slow form, injury tag...). Tested on 2019-2024
# and 2025: with the blended model they lowered squared error by about 100 in both periods.
QB_EXTRA = ["vac_tgt", "vac_tgt_share", "vac_pos_tgt", "n_out", "inj_q", "pf_pass_yds", "ps_pass_yds", "pf_pass_tds",
            "ps_pass_tds", "p_pass_att", "last_snap", "opp_pos_rec_rel", "team_att_rel"]
BEST = WITH_LINES + QB_EXTRA


def add_skill_inputs(qb_table: pd.DataFrame, skill_table: pd.DataFrame | None) -> pd.DataFrame:
    """Attach QB_EXTRA columns from the skill-player table (blank when no table is given)."""
    out = qb_table.drop(columns=[c for c in QB_EXTRA if c in qb_table], errors="ignore")
    if skill_table is None or not len(out):
        return out.assign(**{c: np.nan for c in QB_EXTRA})
    x = skill_table[["season", "week", "player_id"] + QB_EXTRA].drop_duplicates(["season", "week", "player_id"])
    return out.merge(x, on=["season", "week", "player_id"], how="left")


def team_game_context(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with game context from that team's perspective."""
    g = games[games.game_type == "REG"].copy()
    dome = g.roof.isin(["dome", "closed"]).astype(float)
    wind = np.where(dome == 1, 0.0, g.wind); temp = np.where(dome == 1, 70.0, g.temp)
    rows = []
    for side, other, sign in (("home", "away", 1.0), ("away", "home", -1.0)):
        sp = sign * g.spread_line          # + = this team favoured
        rows.append(pd.DataFrame(dict(
            season=g.season, week=g.week, game_id=g.game_id, team=g[f"{side}_team"].replace(TEAM_CODE_FIX),
            opp=g[f"{other}_team"].replace(TEAM_CODE_FIX), is_home=1.0 if side == "home" else 0.0,
            dome=dome, wind_eff=wind, temp_eff=temp, total_line=g.total_line, spread_team=sp,
            team_implied=(g.total_line + sp) / 2, opp_implied=(g.total_line - sp) / 2, gameday=g.gameday)))
    return pd.concat(rows, ignore_index=True)


def build_qb_prop_table(games: pd.DataFrame | None = None, passing: pd.DataFrame | None = None,
                        skill_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """QB-game rows with pre-game inputs. Called with no arguments it also builds the skill-player table and
    attaches QB_EXTRA; pass `skill_table` to reuse one you already have."""
    if games is None and passing is None and skill_table is None:
        from .skill import build_skill_table
        skill_table = build_skill_table()
    return add_skill_inputs(_build_qb_rows(games, passing), skill_table)


def _build_qb_rows(games: pd.DataFrame | None = None, passing: pd.DataFrame | None = None) -> pd.DataFrame:
    games = load_games() if games is None else games
    passing = load_passing_games() if passing is None else passing
    p = passing.copy()
    p["team"] = p.team.replace(TEAM_CODE_FIX); p["opponent_team"] = p.opponent_team.replace(TEAM_CODE_FIX)
    p = p[p.attempts >= MIN_ATT].copy()
    ctx = team_game_context(games)
    p = p.merge(ctx, left_on=["season", "week", "team"], right_on=["season", "week", "team"], how="left")
    # yards allowed to all QBs by defending team, per game (all QBs, any attempts)
    allp = passing.copy(); allp["opponent_team"] = allp.opponent_team.replace(TEAM_CODE_FIX)
    allowed = allp.groupby(["season", "week", "opponent_team"]).passing_yards.sum().rename("allowed").reset_index()

    qb = {}      # id -> [s_yds, s_att, s_w, n]
    opp = {}     # team -> [s_allowed, w]
    lg = [0.0, 0.0, 0.0, 0.0]  # decayed: sum yds, n, sum attempts, sum total_line
    out = []
    p = p.sort_values(["season", "week"]).reset_index(drop=True)
    wk = allowed.groupby(["season", "week"])
    allowed_by_week = {k: v for k, v in wk}
    played_weeks = sorted(set(zip(p.season, p.week)) | set(allowed_by_week))
    by_week = {k: v for k, v in p.groupby(["season", "week"])}
    for (s, w) in played_weeks:
        lg_mean = lg[0] / lg[1] if lg[1] else 225.0
        lg_tot = lg[3] / lg[1] if lg[1] else 44.0
        lg_att = lg[2] / lg[1] if lg[1] else 33.0
        rows = by_week.get((s, w))
        if rows is not None:
            for r in rows.itertuples(index=False):
                st = qb.get(r.player_id)
                if st:
                    n_eff = st[2]
                    yds = lg_mean + st[0] / (n_eff + PRIOR_GAMES)     # league level now + QB's shrunk deviation
                    att = lg_att + st[1] / (n_eff + PRIOR_GAMES)
                    games_n = min(st[3], GAMES_CAP)
                else:
                    yds, att, games_n = lg_mean, lg_att, 0
                o = opp.get(r.opponent_team)
                oy = lg_mean + o[0] / (o[1] + PRIOR_GAMES) if o else lg_mean
                d = r._asdict(); d.update(qb_yds=yds, qb_att=att, qb_ypa=yds / max(att, 1), qb_games=games_n,
                                          opp_yds_allowed=oy, lg_yds=lg_mean,
                                          team_implied_r=r.team_implied - lg_tot / 2, opp_implied_r=r.opp_implied - lg_tot / 2,
                                          total_r=r.total_line - lg_tot)
                out.append(d)
            # update QB form + league mean AFTER snapshotting the week
            for r in rows.itertuples(index=False):
                st = qb.setdefault(r.player_id, [0.0, 0.0, 0.0, 0])
                st[0] = LAM * st[0] + (r.passing_yards - lg_mean); st[1] = LAM * st[1] + (r.attempts - lg_att)
                st[2] = LAM * st[2] + 1; st[3] += 1
                lg[0] = LG_LAM * lg[0] + r.passing_yards; lg[1] = LG_LAM * lg[1] + 1; lg[2] = LG_LAM * lg[2] + r.attempts; lg[3] = LG_LAM * lg[3] + r.total_line
        a = allowed_by_week.get((s, w))
        if a is not None:
            for r in a.itertuples(index=False):
                o = opp.setdefault(r.opponent_team, [0.0, 0.0])
                o[0] = OPP_LAM * o[0] + (r.allowed - lg_mean); o[1] = OPP_LAM * o[1] + 1
    return pd.DataFrame(out)


def upcoming_qb_rows(games: pd.DataFrame, passing: pd.DataFrame, week: int | None = None) -> pd.DataFrame:
    """Feature rows for the expected starters of upcoming games (uses schedule's listed QBs)."""
    up = games[(games.game_type == "REG") & games.home_score.isna()]
    if up.empty:
        return up
    up = up[up.season == up.season.min()]
    week = int(week if week is not None else up.week.min()); up = up[up.week == week]
    # append placeholder result rows so the builder snapshots state for these games
    fake = []
    for g in up.itertuples():
        for side, qid, nm, other in (("home", g.home_qb_id, g.home_qb_name, g.away_team),
                                     ("away", g.away_qb_id, g.away_qb_name, g.home_team)):
            if isinstance(qid, str):
                fake.append(dict(player_id=qid, player_display_name=nm, season=g.season, week=g.week,
                                 team=getattr(g, f"{side}_team"), opponent_team=other, completions=0, attempts=MIN_ATT,
                                 passing_yards=np.nan, passing_tds=0, passing_interceptions=0, sacks_suffered=0,
                                 carries=0, rushing_yards=0))
    if not fake:
        return pd.DataFrame()
    both = pd.concat([passing, pd.DataFrame(fake)], ignore_index=True)
    t = _build_qb_rows(games, both)
    return t[(t.season == up.season.iloc[0]) & (t.week == week) & t.passing_yards.isna()].copy()
