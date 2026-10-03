"""How the model did in past weeks, for each page of the app.

Past weeks are rebuilt honestly: for each season the models learn only from EARLIER seasons, then make the same
picks and projections the app would have shown, using the sportsbook's closing lines. Nothing here has seen the
results it is graded on. (Props are graded on outcomes, not prices: there is no free prop price history.)
"""
from __future__ import annotations
import pickle
import numpy as np, pandas as pd
from ..config import DATA_DIR
from ..data import load_games
from ..features import load_feature_table
from . import board as B, tracker

CACHE = DATA_DIR / "ui_history.pkl"
YARD_STATS = {"qb_pass_yds": "Passing yards", "rush_yds": "Rushing yards", "rec_yds": "Receiving yards"}
NO_SKILL = pd.DataFrame(columns=["season", "week", "player_id", "rushing_tds", "receiving_tds"])


def past_line_bets(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Every moneyline/spread/total side for the finished games of `season`, graded. Flags the model's pick in each
    market and the bets that would have made that week's Home top 10."""
    done = df[(df.game_type == "REG") & (df.season == season) & df.home_score.notna()
              & df.home_moneyline.notna() & df.spread_line.notna()]
    if done.empty:
        return pd.DataFrame()
    games = B.score_games(df[df.season < season], done)
    rows = []
    for wk in sorted({g["week"] for g in games}):
        wk_games = [g for g in games if g["week"] == wk]
        top = {id(b): i for i, b in enumerate(B.top_bets(B.game_bets(wk_games)), 1)}
        for g in wk_games:
            for key in ("moneyline", "spread", "total"):
                if not g[key]:
                    continue
                best = B.model_pick(g[key])
                for b in g[key]:
                    rows.append(dict(b, season=season, week=wk, stat=None, player_id=None,
                                     is_pick=b is best, rank=top.get(id(b), np.nan)))
    return tracker.grade(pd.DataFrame(rows), load_games(), NO_SKILL)


def past_props(seasons: list[int], progress=None) -> dict:
    """Out-of-sample projections vs what happened, per prop. {stat: frame with season, week, actual, pred, ...}"""
    from ..props import evaluate as E, skill_eval as S
    from ..props.features import build_qb_prop_table
    say = progress or (lambda f, t: None)
    out = {}
    say(0.0, "Checking past weeks: passing yards")
    qb, _ = E.walk_forward(build_qb_prop_table(), first_test=min(seasons))
    m = E.APP_MODEL
    out["qb_pass_yds"] = pd.DataFrame(dict(season=qb.season, week=qb.week, actual=qb.passing_yards, pred=qb[f"pred_{m}"],
                                           base=qb.pred_baseline_qb, q10=qb[f"q10_{m}"], q25=qb[f"q25_{m}"],
                                           q75=qb[f"q75_{m}"], q90=qb[f"q90_{m}"]))
    table = S.build_skill_table()
    for i, (name, st) in enumerate(S.STATS.items()):
        say(0.25 + 0.75 * i / len(S.STATS), f"Checking past weeks: {B.PROP_LABEL.get(name, name).lower()}")
        w = S.walk_forward(table, st, first_test=min(seasons))
        d = dict(season=w.season, week=w.week, actual=w[st.target].astype(float), base=w[st.base])
        if st.kind == "binary":
            d["p"] = w.prob_with_lines
        else:
            d["pred"] = w.mean_with_lines
            if st.kind == "cont":
                d.update({q: w[f"{q}_with_lines"] for q in ("q10", "q25", "q75", "q90")})
            else:
                d["line"] = S.naive_line(st, w[st.base]); d["p_over"] = w.pover_with_lines
        out[name] = pd.DataFrame(d)
    return {k: v[v.season.isin(seasons)].reset_index(drop=True) for k, v in out.items()}


# ------------------------------------------------------------------ display tables ----
def _open_weeks() -> set:
    """(season, week) pairs that still have games to play."""
    g = load_games()
    g = g[(g.game_type == "REG") & g.home_score.isna()]
    return set(zip(g.season.astype(int), g.week.astype(int)))


def _by_period(d: pd.DataFrame, current: int, fn) -> pd.DataFrame:
    """One row per finished week of the current season, then the season so far, then last season as a yardstick."""
    rows = []
    cur = d[d.season == current]
    for wk, x in cur.groupby("week"):
        tag = " (in progress)" if (current, int(wk)) in _open_weeks() else ""
        rows.append({"": f"Week {int(wk)}{tag}", **fn(x)})
    if len(cur):
        rows.append({"": "This season", **fn(cur)})
    prev = d[d.season == current - 1]
    if len(prev):
        rows.append({"": "Last season", **fn(prev)})
    return pd.DataFrame(rows)


def _record(x: pd.DataFrame) -> str:
    w, l = int((x.result == "win").sum()), int((x.result == "loss").sum())
    return f"{w}-{l} ({100 * w / (w + l):.0f}%)" if w + l else "-"


def lines_table(bets: pd.DataFrame, current: int) -> pd.DataFrame:
    p = bets[bets.is_pick]
    return _by_period(p, current, lambda x: {m: _record(x[x.market == k]) for m, k in
                                             (("Moneyline picks", "Moneyline"), ("Spread picks", "Spread"), ("Total picks", "Total"))})


def home_table(bets: pd.DataFrame, current: int) -> pd.DataFrame:
    t = bets[bets["rank"].notna()]
    def fn(x):
        s = x[x.result.isin(["win", "loss", "push"])]
        return {"Best 3 record": _record(s[s["rank"] <= 3]), "Best 3 profit": f"{s[s['rank'] <= 3].profit.sum():+.1f}",
                "Top 10 record": _record(s), "Pushes": int((s.result == "push").sum()),
                "Profit (units)": f"{s.profit.sum():+.1f}", "Return per bet": f"{100 * s.profit.mean():+.1f}%" if len(s) else "-"}
    return _by_period(t, current, fn)


def yards_table(d: pd.DataFrame, current: int) -> pd.DataFrame:
    def fn(x):
        return {"Players": len(x), "Average miss": f"{(x.actual - x.pred).abs().mean():.1f}",
                "Miss using recent form only": f"{(x.actual - x.base).abs().mean():.1f}",
                "Inside the thick bar (aim 50%)": f"{100 * ((x.actual >= x.q25) & (x.actual <= x.q75)).mean():.0f}%",
                "Inside the full range (aim 80%)": f"{100 * ((x.actual >= x.q10) & (x.actual <= x.q90)).mean():.0f}%"}
    return _by_period(d, current, fn)


def count_table(d: pd.DataFrame, current: int) -> pd.DataFrame:
    def fn(x):
        live = x[x.actual != x.line]
        right = ((live.p_over > 0.5) == (live.actual > live.line)).mean() if len(live) else np.nan
        return {"Players": len(x), "Average miss": f"{(x.actual - x.pred).abs().mean():.2f}",
                "Miss using recent form only": f"{(x.actual - x.base).abs().mean():.2f}",
                "Over/under call right": f"{100 * right:.0f}%" if len(live) else "-"}
    return _by_period(d, current, fn)


def td_table(d: pd.DataFrame, current: int) -> pd.DataFrame:
    def fn(x):
        hits, n = 0, 0
        for _, wk in x.groupby(["season", "week"]):
            top = wk.nlargest(10, "p"); hits += int(top.actual.sum()); n += len(top)
        return {"Players": len(x), "Scorers expected": f"{x.p.sum():.0f}", "Scorers actual": int(x.actual.sum()),
                "10 most likely each week: scored": f"{hits} of {n} ({100 * hits / n:.0f}%)" if n else "-"}
    return _by_period(d, current, fn)


def teaser_table(legs: pd.DataFrame, current: int) -> pd.DataFrame:
    def fn(x):
        w = x.won.dropna()
        p = w.mean() if len(w) else np.nan
        ret = lambda odds: f"{round(100 * (p * p * (1 + 100 / odds) - 1)) + 0:+d}%" if len(w) else "-"
        return {"Legs": int(len(w)), "Legs won": f"{int(w.sum())} ({100 * p:.0f}%)" if len(w) else "-",
                "Two-team teaser at -120": ret(120), "At -130": ret(130)}
    t = _by_period(legs, current, fn)
    return pd.concat([t, pd.DataFrame([{"": f"Since {int(legs.season.min())}", **fn(legs)}])], ignore_index=True)


def build(current: int | None = None, progress=None) -> dict:
    say = progress or (lambda f, t: None)
    say(0.0, "Checking past weeks: games")
    df = load_feature_table()
    done = df[(df.game_type == "REG") & df.home_score.notna()]
    current = int(current or done.season.max())
    seasons = [current - 1, current]
    bets = pd.concat([past_line_bets(df, s) for s in seasons], ignore_index=True)
    props = past_props(seasons, lambda f, t: say(0.15 + 0.85 * f, t))
    out = {"season": current, "Home": home_table(bets, current), "Moneyline & Spread": lines_table(bets, current),
           "Touchdowns": td_table(props["anytime_td"], current), "Passing touchdowns": count_table(props["pass_tds"], current),
           "Receptions": count_table(props["receptions"], current),
           "Teasers": teaser_table(B.teaser_history(load_games()), current)}
    for k, title in YARD_STATS.items():
        out[title] = yards_table(props[k], current)
    return out


def load_or_build(progress=None) -> dict:
    """Rebuilt only when a new game has finished; otherwise read from the cache on disk."""
    g = load_games()
    done = g[(g.game_type == "REG") & g.home_score.notna()]
    key = (int(done.season.max()), int(len(done)), 5)          # last number: bump when the tables change shape
    if CACHE.exists():
        try:
            saved = pickle.loads(CACHE.read_bytes())
            if saved.get("key") == key:
                return saved["tables"]
        except Exception:
            pass
    tables = build(key[0], progress)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(dict(key=key, tables=tables)))
    return tables
