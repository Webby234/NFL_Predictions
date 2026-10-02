"""Assemble everything the app shows for the upcoming week, in plain data (no Streamlit here).

Every bet, whatever the market, is reduced to the same four numbers so they can share one list:
  p_win      model chance the bet wins (pushes removed)
  breakeven  chance needed to break even at the offered price
  edge       p_win - breakeven
  ev         expected profit per $1 staked
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ..betting.odds import american_to_decimal, implied_prob, prob_to_american
from ..betting.picks import DEFAULT_MODEL, predict_upcoming, upcoming_games
from ..config import TEAM_NAME_TO_CODE, TRAIN_START
from ..evaluation import lines_backtest as L
from ..features import load_feature_table
from ..features.team_features import TEAM_CODE_FIX

CODE_TO_NAME = {v: k for k, v in TEAM_NAME_TO_CODE.items()}
NICK = {c: n.split()[-1] for c, n in CODE_TO_NAME.items()}
PROP_LABEL = {"qb_pass_yds": "Passing yards", "rush_yds": "Rushing yards", "rec_yds": "Receiving yards",
              "receptions": "Receptions", "pass_tds": "Passing touchdowns", "anytime_td": "Anytime touchdown"}
TOP10_EXCLUDED = {"pass_tds"}      # over/under chances for passing TDs were overconfident in testing


def nick(code: str) -> str:
    code = TEAM_CODE_FIX.get(code, code)
    return NICK.get(code, code)


def clean_odds(o, default=-110):
    """American odds as int; blanks and impossible values fall back to the standard -110."""
    try:
        o = float(o)
    except (TypeError, ValueError):
        return default
    if np.isnan(o) or abs(o) < 100 or abs(o) > 5000:
        return default
    return int(round(o))


# How much of the model's disagreement with the sportsbook has held up on past seasons (walk-forward 2015+).
# 1.0 = believe the model fully, 0 = believe the sportsbook. Recompute with: python -m nfl_model.ui.calibrate
# Props have no price history to test against, so they get a cautious assumed value.
TRUST = {"Moneyline": 0.40, "Spread": 0.20, "Total": 0.55, "prop": 0.50}


def _logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return np.log(p / (1 - p))


def side(p_win: float, p_loss: float, odds, other_odds=None, trust: float = 1.0) -> dict:
    """Common yardstick for one side of any market.

    The raw model chance is pulled toward the sportsbook's own (vig-free) chance by `trust`, because past
    seasons show the book is right about most of the gap. `breakeven` still includes the vig you pay.
    """
    odds = clean_odds(odds)
    live = p_win + p_loss
    raw = p_win / live if live > 0 else 0.0
    be = float(implied_prob(odds))
    fair = be
    if other_odds is not None and not (isinstance(other_odds, float) and np.isnan(other_odds)):
        ob = float(implied_prob(clean_odds(other_odds)))
        fair = be / (be + ob)
    p = float(1 / (1 + np.exp(-(_logit(fair) + trust * (_logit(raw) - _logit(fair))))))
    return dict(odds=odds, p_win=p, p_raw=float(raw), breakeven=be, edge=p - be,
                ev=float(live * (p * american_to_decimal(odds) - 1)))


def kickoff(row) -> str:
    try:
        t = pd.to_datetime(f"{row.gameday} {row.gametime}")
        return f"{t:%a} {t.hour % 12 or 12}:{t:%M} {'PM' if t.hour >= 12 else 'AM'}"
    except Exception:
        return str(getattr(row, "gameday", ""))


def _outcome_samples(df: pd.DataFrame, up: pd.DataFrame, target: str, feats: list, line: str):
    """Model number for each upcoming game plus a spread of plausible final results around it."""
    done = df[(df.game_type == "REG") & df.home_score.notna() & (df.season >= TRAIN_START)].dropna(subset=[target, line])
    cols = feats + [line]
    m = L.ridge().fit(done[cols], done[target])
    resid = (done[target] - m.predict(done[cols])).values
    pred = m.predict(up[cols])
    return pred, resid


def _two_way(pred: float, resid: np.ndarray, line: float):
    """Chances the final number lands above / below the line (scores are whole numbers, so ties can happen)."""
    sims = np.round(pred + resid)
    return float((sims > line).mean()), float((sims < line).mean())


def build_games(df: pd.DataFrame, week: int | None = None) -> list[dict]:
    up = upcoming_games(df, week)
    if up.empty:
        return []
    scored = predict_upcoming(df, up, DEFAULT_MODEL)
    full = L.prep(df)
    upp = L.prep(scored)
    sp_pred, sp_res = _outcome_samples(full, upp, "result", L.MARGIN_FEATS, "spread_line")
    has_tot = upp.total_line.notna().values
    tt_pred = np.full(len(upp), np.nan); tt_res = np.array([0.0])
    if has_tot.any():
        p, tt_res = _outcome_samples(full, upp[has_tot], "total", L.TOTAL_FEATS, "total_line")
        tt_pred[has_tot] = p
    games = []
    for i, g in enumerate(upp.itertuples(index=False)):
        home, away = g.home_team, g.away_team
        label = f"{nick(away)} at {nick(home)}"
        base = dict(game_id=g.game_id, matchup=label, kickoff=kickoff(g))
        ph = float(g.p_home)
        ml = [dict(base, market="Moneyline", team=away, pick=f"{nick(away)} to win", **side(1 - ph, ph, g.away_moneyline, g.home_moneyline, TRUST["Moneyline"])),
              dict(base, market="Moneyline", team=home, pick=f"{nick(home)} to win", **side(ph, 1 - ph, g.home_moneyline, g.away_moneyline, TRUST["Moneyline"]))]
        s = float(g.spread_line)                       # + means home favoured by s
        over, under = _two_way(sp_pred[i], sp_res, s)  # "over" = home covers
        fmt = lambda x: "PK" if x == 0 else f"{x:+g}"
        sp = [dict(base, market="Spread", team=away, pick=f"{nick(away)} {fmt(s)}", **side(under, over, g.away_spread_odds, g.home_spread_odds, TRUST["Spread"])),
              dict(base, market="Spread", team=home, pick=f"{nick(home)} {fmt(-s)}", **side(over, under, g.home_spread_odds, g.away_spread_odds, TRUST["Spread"]))]
        tot = []
        if has_tot[i]:
            o, u = _two_way(tt_pred[i], tt_res, float(g.total_line))
            tot = [dict(base, market="Total", team=None, pick=f"Over {g.total_line:g}", **side(o, u, g.over_odds, g.under_odds, TRUST["Total"])),
                   dict(base, market="Total", team=None, pick=f"Under {g.total_line:g}", **side(u, o, g.under_odds, g.over_odds, TRUST["Total"]))]
        games.append(dict(base, home=home, away=away, home_name=nick(home), away_name=nick(away),
                          season=int(g.season), week=int(g.week), gameday=str(g.gameday),
                          moneyline=ml, spread=sp, total=tot,
                          model_margin=float(sp_pred[i]), model_total=float(tt_pred[i])))
    return sorted(games, key=lambda x: (x["gameday"], x["game_id"]))


def game_bets(games: list[dict]) -> list[dict]:
    return [b for g in games for mk in ("moneyline", "spread", "total") for b in g[mk]]


# ------------------------------------------------------------------ player props ----
def build_props(week: int | None = None) -> dict:
    """Projections for each prop market. Returns {stat: DataFrame}; frames keep the fitted model in .attrs."""
    from ..props import skill_predict as SP
    from ..props.predict import predict_upcoming as predict_qb
    out = {}
    qb = predict_qb(week)
    if len(qb):
        out["qb_pass_yds"] = qb.rename(columns={"pred_yards": "pred"})
        out["qb_pass_yds"].attrs = dict(qb.attrs, stat="qb_pass_yds")
    table, up = SP.upcoming_table(week)
    if up is not None and len(up):
        for name in ("rush_yds", "rec_yds", "receptions", "pass_tds", "anytime_td"):
            p = SP.predict_stat(table, up, name)
            if len(p):
                out[name] = p
    for k, p in list(out.items()):
        if "pred" in p:                                    # biggest projections first
            attrs = p.attrs; out[k] = p = p.sort_values("pred", ascending=False); p.attrs = attrs
        if "p_td" in p:
            p["fair_odds"] = [int(prob_to_american(min(max(x, 0.01), 0.99))) for x in p.p_td]
    return out


def price_prop_lines(props: dict, lines: pd.DataFrame) -> list[dict]:
    """Turn saved sportsbook lines into bets on the common yardstick. `lines`: stat, player, line, over_odds, under_odds."""
    from ..props import skill_predict as SP
    bets = []
    if lines is None or lines.empty:
        return bets
    for stat, ln in lines.groupby("stat"):
        pred = props.get(stat)
        if pred is None or not len(pred):
            continue
        model, rows = pred.attrs["model"], pred.attrs["rows"]
        r = rows.assign(key=rows.player_display_name.str.lower().str.strip()).drop_duplicates("key").set_index("key")
        for x in ln.itertuples(index=False):
            key = str(x.player).lower().strip()
            if key not in r.index:
                continue
            row = r.loc[[key]]
            sub = f"{nick(row.team.iloc[0])} vs {nick(row.opponent_team.iloc[0])}"
            base = dict(market=PROP_LABEL[stat], stat=stat, team=row.team.iloc[0], matchup=sub, kickoff="",
                        player=row.player_display_name.iloc[0], game_id=None)
            if stat == "anytime_td":
                p = float(model.prob(row)[0])
                bets.append(dict(base, pick=f"{base['player']} to score", **side(p, 1 - p, x.over_odds, getattr(x, "under_odds", None), TRUST["prop"])))
                continue
            if pd.isna(x.line):
                continue
            mu = model.mean(row)
            po = float(np.ravel(model.prob_over(mu, np.array([x.line])))[0])
            pu = float(np.ravel(model.prob_under(mu, np.array([x.line])))[0])
            unit = {"qb_pass_yds": "pass yds", "rush_yds": "rush yds", "rec_yds": "rec yds",
                    "receptions": "receptions", "pass_tds": "pass TDs"}[stat]
            o = dict(base, pick=f"{base['player']} over {x.line:g} {unit}", **side(po, pu, x.over_odds, x.under_odds, TRUST["prop"]))
            u = dict(base, pick=f"{base['player']} under {x.line:g} {unit}", **side(pu, po, x.under_odds, x.over_odds, TRUST["prop"]))
            bets += [o, u]
    return bets


def top_bets(bets: list[dict], n: int = 10) -> list[dict]:
    """Best `n` by edge (they may still fall short of break-even: check b["ev"] > 0). A team's moneyline and
    spread are the same opinion, so only the stronger of the two is kept per game; likewise one side per over/under."""
    good = sorted([b for b in bets if b.get("stat") not in TOP10_EXCLUDED],
                  key=lambda b: b["edge"], reverse=True)
    seen, out = set(), []
    for b in good:
        if b["market"] in ("Moneyline", "Spread"):
            key = ("side", b["game_id"])
        elif b["market"] == "Total":
            key = ("total", b["game_id"])
        else:
            key = (b["market"], b.get("player"))
        if key in seen:
            continue
        seen.add(key); out.append(b)
        if len(out) == n:
            break
    return out


def _stamp(t) -> str:
    """'Oct 2, 7:09 PM' without the %-d / %-I codes, which Windows does not support."""
    return f"{t:%b} {t.day}, {t.hour % 12 or 12}:{t:%M} {'PM' if t.hour >= 12 else 'AM'}"


def build_board(week: int | None = None) -> dict:
    df = load_feature_table()
    games = build_games(df, week)
    if not games:
        return dict(games=[], props={}, week=None, season=None)
    wk = games[0]["week"]
    return dict(games=games, props=build_props(wk), week=wk, season=games[0]["season"],
                built=_stamp(pd.Timestamp.now()))
