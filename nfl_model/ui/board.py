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
TOP10_EXCLUDED: set = set()       # props kept off the Home list (none: passing-TD chances are calibrated since the blend)


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


def side(p_win: float, p_loss: float, odds, other_odds=None, trust: float = 1.0, fair: float | None = None) -> dict:
    """Common yardstick for one side of any market.

    The raw model chance is pulled toward the sportsbook's own (vig-free) chance by `trust`, because past
    seasons show the book is right about most of the gap. `breakeven` still includes the vig you pay.
    """
    odds = clean_odds(odds)
    live = p_win + p_loss
    raw = p_win / live if live > 0 else 0.0
    be = float(implied_prob(odds))
    if fair is not None:
        fair = min(max(float(fair), 0.01), 0.99)
    elif other_odds is None or (isinstance(other_odds, float) and np.isnan(other_odds)):
        fair = be
    else:
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
    return score_games(df, upcoming_games(df, week))


def model_pick(sides: list[dict]) -> dict:
    """The side the model predicts: the likelier winner of the bet (ties go to the better price)."""
    return max(sides, key=lambda b: (round(b["p_win"], 6), b["edge"]))


def score_games(df: pd.DataFrame, up: pd.DataFrame) -> list[dict]:
    """Price every market of the games in `up`, learning only from the finished games in `df`.
    (For the live board `df` is everything played so far; for past weeks it is the seasons before.)"""
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
        base = dict(game_id=g.game_id, matchup=f"{nick(away)} at {nick(home)}", kickoff=kickoff(g))
        listed = dict(away_ml=g.away_moneyline, home_ml=g.home_moneyline, home_spread=-float(g.spread_line),
                      away_spread_odds=g.away_spread_odds, home_spread_odds=g.home_spread_odds,
                      total=float(g.total_line) if has_tot[i] else np.nan, over_odds=g.over_odds, under_odds=g.under_odds)
        listed = {k: (clean_odds(v) if k.endswith(("_ml", "_odds")) else v) for k, v in listed.items()}
        sim = dict(p_home=float(g.p_home), margin=float(sp_pred[i]), margin_res=sp_res,
                   total=float(tt_pred[i]), total_res=tt_res)
        game = dict(base, home=home, away=away, home_name=nick(home), away_name=nick(away),
                    season=int(g.season), week=int(g.week), gameday=str(g.gameday), listed=listed, sim=sim,
                    yours=False, model_margin=float(sp_pred[i]), model_total=float(tt_pred[i]))
        game.update(price_game(game, listed))
        games.append(game)
    return sorted(games, key=lambda x: (x["gameday"], x["game_id"]))


LINE_FIELDS = ["away_ml", "home_ml", "home_spread", "away_spread_odds", "home_spread_odds", "total", "over_odds", "under_odds"]


def _fair(odds, other) -> float:
    a, b = float(implied_prob(clean_odds(odds))), float(implied_prob(clean_odds(other)))
    return a / (a + b)


def _shift(center: float, res: np.ndarray, listed_line: float, line: float) -> float:
    """How much the market's own chance of 'over' changes when the line moves from the listed number to `line`
    (the market is taken to expect the listed number; the spread of results around it comes from history)."""
    def p(x):
        o, u = _two_way(center, res, x)
        return o / (o + u) if o + u else 0.5
    return p(line) - p(listed_line)


def price_game(game: dict, lines: dict) -> dict:
    """Both sides of the moneyline, spread and total for one game at the given lines and prices.
    `lines["home_spread"]` is the home team's number as a sportsbook shows it (-2.5 = home favoured by 2.5).

    The sportsbook's side of the argument always comes from the LISTED lines (the wider market), so a better
    number or price at your own book shows up as extra edge instead of being argued away."""
    base = dict(game_id=game["game_id"], matchup=game["matchup"], kickoff=game["kickoff"])
    home, away, sim, L0 = game["home"], game["away"], game["sim"], game["listed"]
    ph = sim["p_home"]
    f_home = _fair(L0["home_ml"], L0["away_ml"])
    ml = [dict(base, market="Moneyline", team=away, side="away", line=None, pick=f"{nick(away)} to win",
               **side(1 - ph, ph, lines["away_ml"], None, TRUST["Moneyline"], fair=1 - f_home)),
          dict(base, market="Moneyline", team=home, side="home", line=None, pick=f"{nick(home)} to win",
               **side(ph, 1 - ph, lines["home_ml"], None, TRUST["Moneyline"], fair=f_home))]
    s, s0 = -float(lines["home_spread"]), -float(L0["home_spread"])       # + means home favoured by s
    over, under = _two_way(sim["margin"], sim["margin_res"], s)            # "over" = home covers
    f_cover = _fair(L0["home_spread_odds"], L0["away_spread_odds"]) + _shift(s0, sim["margin_res"], s0, s)
    fmt = lambda x: "PK" if x == 0 else f"{x:+g}"
    sp = [dict(base, market="Spread", team=away, side="away", line=s, pick=f"{nick(away)} {fmt(s)}",
               **side(under, over, lines["away_spread_odds"], None, TRUST["Spread"], fair=1 - f_cover)),
          dict(base, market="Spread", team=home, side="home", line=s, pick=f"{nick(home)} {fmt(-s)}",
               **side(over, under, lines["home_spread_odds"], None, TRUST["Spread"], fair=f_cover))]
    tot = []
    if not pd.isna(lines["total"]) and not np.isnan(sim["total"]):
        t, t0 = float(lines["total"]), float(L0["total"])
        o, u = _two_way(sim["total"], sim["total_res"], t)
        f_over = _fair(L0["over_odds"], L0["under_odds"]) + _shift(t0, sim["total_res"], t0, t)
        tot = [dict(base, market="Total", team=None, side="over", line=t, pick=f"Over {t:g}",
                    **side(o, u, lines["over_odds"], None, TRUST["Total"], fair=f_over)),
               dict(base, market="Total", team=None, side="under", line=t, pick=f"Under {t:g}",
                    **side(u, o, lines["under_odds"], None, TRUST["Total"], fair=1 - f_over))]
    return dict(moneyline=ml, spread=sp, total=tot)


def apply_book_lines(games: list[dict], saved: pd.DataFrame) -> list[dict]:
    """Re-price games at the lines you entered from your own sportsbook (blank fields keep the listed number).
    Returns new game dicts; the originals are left alone."""
    if saved is None or saved.empty:
        return games
    by_id = saved.drop_duplicates("game_id", keep="last").set_index("game_id")
    out = []
    for g in games:
        if g["game_id"] not in by_id.index:
            out.append(g); continue
        row = by_id.loc[g["game_id"]]
        lines = dict(g["listed"])
        for k in LINE_FIELDS:
            v = row.get(k)
            if v is not None and not pd.isna(v):
                lines[k] = clean_odds(v) if k.endswith(("_ml", "_odds")) else float(v)
        g2 = dict(g, yours=lines != g["listed"], lines=lines)
        g2.update(price_game(g2, lines))
        out.append(g2)
    return out


# ------------------------------------------------------------------ teasers ----
TEASE = 6.0
DOG_RANGE = (1.5, 2.5)        # underdogs in this range, teased 6, cross both 3 and 7
MAX_TOTAL = 49.0


def teaser_legs(games: list[dict]) -> list[dict]:
    """This week's qualifying legs: short underdogs in lower-scoring games, moved up six points."""
    legs = []
    for g in games:
        s = g["spread"][0]["line"]                      # + = home favoured
        dog, pts = (g["away"], s) if s > 0 else (g["home"], -s)
        total = g["total"][0]["line"] if g["total"] else np.nan
        if DOG_RANGE[0] <= pts <= DOG_RANGE[1] and (np.isnan(total) or total <= MAX_TOTAL):
            legs.append(dict(game_id=g["game_id"], team=dog, matchup=g["matchup"], kickoff=g["kickoff"],
                             pick=f"{nick(dog)} +{pts + TEASE:g}", was=f"+{pts:g}", total=total))
    return legs


def teaser_history(games_df: pd.DataFrame, first_season: int = 2006) -> pd.DataFrame:
    """Every past qualifying leg at closing lines: season, week, won (1/0, NaN for a tie at the teased number)."""
    g = games_df[(games_df.game_type == "REG") & games_df.home_score.notna() & games_df.spread_line.notna()
                 & (games_df.season >= first_season)]
    pts = g.spread_line.abs()
    q = g[(pts >= DOG_RANGE[0]) & (pts <= DOG_RANGE[1]) & (g.total_line.isna() | (g.total_line <= MAX_TOTAL))]
    dog_margin = -(q.home_score - q.away_score) * np.sign(q.spread_line)        # underdog's final margin
    cover = dog_margin + q.spread_line.abs() + TEASE
    return pd.DataFrame(dict(season=q.season.values, week=q.week.values,
                             won=np.where(cover > 0, 1.0, np.where(cover < 0, 0.0, np.nan))))


def teaser_stats(hist: pd.DataFrame) -> dict:
    """Leg win rate and what it means for a two-team teaser at common prices."""
    w = hist.won.dropna()
    p, n = float(w.mean()), int(len(w))
    half = 1.96 * (p * (1 - p) / n) ** 0.5
    roi = lambda rate, odds: rate * rate * american_to_decimal(odds) - 1
    return dict(legs=n, rate=p, low=p - half, high=p + half, first=int(hist.season.min()),
                roi={o: roi(p, o) for o in (-110, -120, -130, -135)},
                roi_low={o: roi(p - half, o) for o in (-110, -120, -130, -135)},
                breakeven=int(prob_to_american(p * p)))


def game_bets(games: list[dict]) -> list[dict]:
    return [b for g in games for mk in ("moneyline", "spread", "total") for b in g[mk]]


# ------------------------------------------------------------------ player props ----
def build_props(week: int | None = None) -> dict:
    """Projections for each prop market. Returns {stat: DataFrame}; frames keep the fitted model in .attrs."""
    from ..props import skill_predict as SP
    from ..props.predict import predict_upcoming as predict_qb
    out = {}
    table, up = SP.upcoming_table(week)
    qb = predict_qb(week, skill_table=table)
    if len(qb):
        out["qb_pass_yds"] = qb.rename(columns={"pred_yards": "pred"})
        out["qb_pass_yds"].attrs = dict(qb.attrs, stat="qb_pass_yds")
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


def price_prop_lines(props: dict, lines: pd.DataFrame, games: list[dict] | None = None) -> list[dict]:
    """Turn saved sportsbook lines into bets on the common yardstick. `lines`: stat, player, line, over_odds, under_odds."""
    from ..props import skill_predict as SP
    bets = []
    team_game = {TEAM_CODE_FIX.get(g[k], g[k]): g["game_id"] for g in (games or []) for k in ("home", "away")}
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
                        player=row.player_display_name.iloc[0], player_id=row.player_id.iloc[0],
                        game_id=team_game.get(TEAM_CODE_FIX.get(row.team.iloc[0], row.team.iloc[0])))
            if stat == "anytime_td":
                p = float(model.prob(row)[0])
                bets.append(dict(base, side="yes", line=None, pick=f"{base['player']} to score", **side(p, 1 - p, x.over_odds, getattr(x, "under_odds", None), TRUST["prop"])))
                continue
            if pd.isna(x.line):
                continue
            mu = model.mean(row)
            po = float(np.ravel(model.prob_over(mu, np.array([x.line])))[0])
            pu = float(np.ravel(model.prob_under(mu, np.array([x.line])))[0])
            unit = {"qb_pass_yds": "pass yds", "rush_yds": "rush yds", "rec_yds": "rec yds",
                    "receptions": "receptions", "pass_tds": "pass TDs"}[stat]
            o = dict(base, side="over", line=float(x.line), pick=f"{base['player']} over {x.line:g} {unit}", **side(po, pu, x.over_odds, x.under_odds, TRUST["prop"]))
            u = dict(base, side="under", line=float(x.line), pick=f"{base['player']} under {x.line:g} {unit}", **side(pu, po, x.under_odds, x.over_odds, TRUST["prop"]))
            bets += [o, u]
    return bets


def is_moneyline_underdog(b: dict) -> bool:
    """Plus-money moneylines are kept off the Home list: rebuilt Home lists for 2015-2026 returned -5.8% on them
    (455 bets) against +4.3% on favourites, and dropping them helped in both halves of that history
    (+3.8% in 2015-2020, +2.8% in 2021-2026, versus +2.1% and -1.2% with them). They still show on the games tab."""
    return b["market"] == "Moneyline" and b["odds"] > 0


def top_bets(bets: list[dict], n: int = 10) -> list[dict]:
    """Best `n` by edge (they may still fall short of break-even: check b["ev"] > 0). A team's moneyline and
    spread are the same opinion, so only the stronger of the two is kept per game; likewise one side per over/under."""
    good = sorted([b for b in bets if b.get("stat") not in TOP10_EXCLUDED and not is_moneyline_underdog(b)],
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
    from ..data import load_games
    return dict(games=games, props=build_props(wk), week=wk, season=games[0]["season"],
                teasers=teaser_stats(teaser_history(load_games())),
                built=_stamp(pd.Timestamp.now()))
