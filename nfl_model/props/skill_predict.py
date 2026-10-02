"""Upcoming-game predictions and manual-line pricing for rushing/receiving/TD props."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..betting.odds import american_to_decimal, devig_two_way, implied_prob
from ..data import load_games, load_skill_games
from ..features.team_features import TEAM_CODE_FIX
from .models import QUANTILES, fit_binary, fit_count, fit_dist
from .skill import build_skill_table
from .skill_eval import STATS, TRAIN_START, eligible, feats_for, naive_line

RECENT_WEEKS = 4
RAW = ["player_id", "player_display_name", "position", "season", "week", "team", "opponent_team", "carries",
       "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "attempts",
       "passing_tds", "passing_yards"]


def _ruled_out(season: int, week: int) -> set:
    """Players listed Out/Doubtful on this week's injury report (empty if the report isn't published yet)."""
    try:
        from ..data import load_injuries
        inj = load_injuries(season, season)
        inj = inj[(inj.season == season) & (inj.week == week) & inj.report_status.isin(["Out", "Doubtful"])]
        return set(inj.gsis_id.dropna())
    except Exception:
        return set()


def placeholder_rows(games: pd.DataFrame, skill: pd.DataFrame, week: int | None = None) -> pd.DataFrame:
    """Expected participants for upcoming games: players whose latest team is playing and who appeared in
    their team's last few games. Injury status is NOT applied (check inactives yourself)."""
    up = games[(games.game_type == "REG") & games.home_score.isna()]
    if up.empty:
        return pd.DataFrame(columns=RAW)
    up = up[up.season == up.season.min()]
    week = int(week if week is not None else up.week.min()); up = up[up.week == week]
    sk = skill.copy(); sk["team"] = sk.team.replace(TEAM_CODE_FIX)
    order = sk.season * 100 + sk.week
    recent_cut = np.sort(order.unique())[-min(RECENT_WEEKS, order.nunique())]
    last = sk[order >= recent_cut].sort_values(["season", "week"]).groupby("player_id").tail(1)
    out_ids = _ruled_out(int(up.season.iloc[0]), week)
    rows = []
    for g in up.itertuples():
        for side, other in (("home", g.away_team), ("away", g.home_team)):
            team = TEAM_CODE_FIX.get(getattr(g, f"{side}_team"), getattr(g, f"{side}_team"))
            opp = TEAM_CODE_FIX.get(other, other)
            starter = getattr(g, f"{side}_qb_id", None)
            for r in last[last.team == team].itertuples():
                if r.position == "QB" and r.player_id != starter:      # only the listed starting QB
                    continue
                if r.player_id in out_ids:
                    continue
                rows.append({**{c: np.nan for c in RAW}, "player_id": r.player_id,
                             "player_display_name": r.player_display_name, "position": r.position,
                             "season": g.season, "week": g.week, "team": team, "opponent_team": opp})
    return pd.DataFrame(rows, columns=RAW)


def fit_production(table: pd.DataFrame, stat, with_lines=True):
    t = eligible(table, stat)
    tr = t[t[stat.target].notna() & (t.season >= TRAIN_START)]
    f = feats_for(stat, with_lines)
    if stat.kind == "cont":
        return fit_dist(tr, f, stat.target, "ridge", floor=stat.floor)
    if stat.kind == "count":
        return fit_count(tr, f, stat.target)
    return fit_binary(tr, f, stat.target)


def upcoming_table(week: int | None = None):
    games, skill = load_games(), load_skill_games()
    ph = placeholder_rows(games, skill, week)
    if ph.empty:
        return None, None
    table = build_skill_table(games, pd.concat([skill, ph], ignore_index=True))
    up = table[(table.season == ph.season.iloc[0]) & (table.week == ph.week.iloc[0]) & table.rushing_yards.isna()]
    return table, up


def predict_stat(table, up, name: str) -> pd.DataFrame:
    st = STATS[name]
    rows = up[st.elig(up) & (up.games >= 2)].dropna(subset=["team_implied_r"]).copy()
    if rows.empty:
        return rows
    m = fit_production(table, st)
    out = rows[["player_display_name", "position", "team", "opponent_team", "gameday"]].copy() \
        if "gameday" in rows else rows[["player_display_name", "position", "team", "opponent_team"]].copy()
    if st.kind == "binary":
        out["p_td"] = m.prob(rows)
    else:
        mu = m.mean(rows); out["pred"] = mu
        if st.kind == "cont":
            for q in QUANTILES:
                out[f"q{int(q*100)}"] = m.quantile(mu, q)
        else:
            ln = naive_line(st, mu); out["line"] = ln; out["p_over"] = m.prob_over(mu, ln)
    out.attrs.update(model=m, rows=rows, stat=name)
    shown = out.copy()
    for c in shown.select_dtypes("number").columns:          # chances keep 3 decimals, everything else 1
        shown[c] = shown[c].round(3 if c.startswith("p_") else 1)
    shown.attrs = out.attrs
    return shown.sort_values(shown.columns[-1] if st.kind == "binary" else "pred", ascending=False)


def price_lines(pred: pd.DataFrame, lines: pd.DataFrame, min_edge: float = 0.03) -> pd.DataFrame:
    """cont/count: columns player, line, over_odds, under_odds.  anytime_td: player, odds (yes) [, no_odds]."""
    name, m, rows = pred.attrs["stat"], pred.attrs["model"], pred.attrs["rows"]
    st = STATS[name]
    r = rows.assign(key=rows.player_display_name.str.lower().str.strip()).drop_duplicates("key")
    ln = lines.assign(key=lines.player.str.lower().str.strip()).merge(r, on="key", how="left")
    miss = ln[ln.games.isna()].player.tolist()
    if miss:
        import warnings; warnings.warn(f"no eligible upcoming match for: {miss}")
    ln = ln[ln.games.notna()].copy()
    if st.kind == "binary":
        p = m.prob(ln); ln["p_yes"] = p
        ln["mkt_yes"] = implied_prob(ln.odds)
        if "no_odds" in ln:
            both = ln.no_odds.notna()          # devig only where the "no" price was given
            ln.loc[both, "mkt_yes"] = devig_two_way(ln.odds[both], ln.no_odds[both])[0]
        ln["edge"] = ln.p_yes - ln.mkt_yes
        ln["ev"] = ln.p_yes * american_to_decimal(ln.odds) - 1
        ln["pick"] = np.where((ln.edge >= min_edge) & (ln.ev > 0), "YES", "-")
        return ln[["player", "team", "opponent_team", "odds", "p_yes", "mkt_yes", "edge", "ev", "pick"]].round(3)
    mu = m.mean(ln); ln["pred"] = mu
    ln["p_over"] = m.prob_over(mu, ln.line.values); ln["p_under"] = m.prob_under(mu, ln.line.values)
    push = 1 - ln.p_over - ln.p_under
    ln["mkt_over"], ln["mkt_under"] = devig_two_way(ln.over_odds, ln.under_odds)
    ln["ev_over"] = ln.p_over * american_to_decimal(ln.over_odds) - 1 + push
    ln["ev_under"] = ln.p_under * american_to_decimal(ln.under_odds) - 1 + push
    ln["edge_over"], ln["edge_under"] = ln.p_over - ln.mkt_over, ln.p_under - ln.mkt_under
    ln["pick"] = np.where((ln.edge_over >= min_edge) & (ln.ev_over > 0), "OVER",
                  np.where((ln.edge_under >= min_edge) & (ln.ev_under > 0), "UNDER", "-"))
    return ln[["player", "team", "opponent_team", "line", "over_odds", "under_odds", "pred", "p_over", "p_under",
               "edge_over", "edge_under", "ev_over", "ev_under", "pick"]].round(3)
