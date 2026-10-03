"""HTML for the app. Pure strings, no Streamlit, so the look can be previewed and tested on its own.

Every builder returns one line of HTML (no blank lines or indentation), which keeps Streamlit's
markdown renderer from mangling it.
"""
from __future__ import annotations
from html import escape
import pandas as pd
from ..config import logo_url
from .board import PROP_LABEL, model_pick, nick

WINDOW = 0.08          # the meter shows break-even +/- this many points of win chance

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
:root{--chalk:#F4F6F3;--card:#FFFFFF;--ink:#12241B;--turf:#1E6B46;--turf-soft:#E4F0E9;--flag:#F5C518;--muted:#5E6B63;--rule:#DCE2DC;
--display:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif;--body:'Barlow','Segoe UI',system-ui,sans-serif}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--chalk);color:var(--ink);font-family:var(--body)}
[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none}
header[data-testid="stHeader"]{background:transparent}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1080px;padding-top:2.4rem;padding-bottom:4rem}
.stTabs [data-baseweb="tab-list"]{gap:30px;border-bottom:1px solid var(--rule)}
.stTabs [data-baseweb="tab"]{padding:10px 0;background:transparent;color:var(--muted)}
.stTabs [data-baseweb="tab"] p{font-family:var(--display);font-size:1.25rem;font-weight:600;letter-spacing:.01em}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--ink)}
.stTabs [data-baseweb="tab-highlight"]{background:var(--flag);height:4px}
.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs .stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:0;margin:6px 0 4px}
.stTabs .stTabs [data-baseweb="tab"]{padding:6px 18px;border:1px solid var(--rule);border-radius:999px;background:var(--card)}
.stTabs .stTabs [data-baseweb="tab"] p{font-family:var(--body);font-size:.95rem;font-weight:600}
.stTabs .stTabs [data-baseweb="tab"][aria-selected="true"]{background:var(--turf);border-color:var(--turf)}
.stTabs .stTabs [data-baseweb="tab"][aria-selected="true"] p{color:#fff}
.stTabs .stTabs [data-baseweb="tab-highlight"]{display:none}
.nb *{box-sizing:border-box}
.nb{font-family:var(--body);color:var(--ink);line-height:1.4}
.nb-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin:0 0 18px}
.nb-week{font-family:var(--display);font-weight:700;font-size:3.4rem;line-height:.95;letter-spacing:-.01em}
.nb-sub{color:var(--muted);font-size:1rem;margin-top:6px;max-width:72ch}
.nb-stamp{color:var(--muted);font-size:.85rem}
.nb-h2{font-family:var(--display);font-weight:600;font-size:1.7rem;margin:26px 0 4px}
.nb-lead{color:var(--muted);margin:0 0 14px;max-width:70ch}
.nb-panel{background:var(--card);border:1px solid var(--rule);border-radius:14px;overflow:hidden}
.nb-row{display:grid;grid-template-columns:44px 44px minmax(0,1fr) 78px 230px 70px;align-items:center;gap:14px;padding:14px 18px;border-top:1px solid var(--rule)}
.nb-row:first-child{border-top:0}
.nb-rank{font-family:var(--display);font-weight:700;font-size:1.9rem;color:var(--muted);text-align:center}
.nb-row.clears .nb-rank{color:var(--ink)}
.nb img{overflow:hidden;font-size:0;color:transparent;flex:none}
.nb-logo{width:40px;height:40px;object-fit:contain}
.nb-logos{display:flex;flex-direction:column;align-items:center;gap:2px}
.nb-logos img{width:22px;height:22px;object-fit:contain}
.nb-pick{font-family:var(--display);font-weight:600;font-size:1.45rem;line-height:1.1}
.nb-meta{color:var(--muted);font-size:.88rem;margin-top:3px}
.nb-tag{display:inline-block;font-size:.76rem;font-weight:600;color:var(--turf);background:var(--turf-soft);border-radius:4px;padding:1px 7px;margin-right:8px}
.nb-odds{font-family:var(--display);font-weight:600;font-size:1.5rem;text-align:right}
.nb-edge{text-align:right}
.nb-edge b{font-family:var(--display);font-weight:700;font-size:1.5rem;display:block;line-height:1}
.nb-edge span{font-size:.78rem;color:var(--muted)}
.nb-row.short .nb-edge b{color:var(--muted)}
.nb-meter .track{position:relative;height:14px;border-radius:3px;background:repeating-linear-gradient(90deg,#E9EEE9 0,#E9EEE9 calc(12.5% - 1px),#fff calc(12.5% - 1px),#fff 12.5%)}
.nb-meter .gain{position:absolute;top:0;bottom:0;background:var(--turf);border-radius:2px}
.nb-meter .gain.short{background:#B9C3BC}
.nb-meter .need{position:absolute;top:-4px;bottom:-4px;left:50%;width:4px;margin-left:-2px;background:var(--flag);border-radius:2px}
.nb-meter .ball{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:var(--ink);border:2px solid #fff}
.nb-meter .lbl{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);margin-top:5px}
.nb-meter .lbl b{color:var(--ink);font-weight:600}
.nb-note{color:var(--muted);font-size:.9rem;margin:16px 2px 0;max-width:78ch}
.nb-note b{color:var(--ink);font-weight:600}
.nb-empty{padding:28px 22px;color:var(--muted)}
.nb-game{background:var(--card);border:1px solid var(--rule);border-radius:14px;margin:0 0 14px;padding:16px 18px 14px}
.nb-ghead{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:10px}
.nb-gname{font-family:var(--display);font-weight:600;font-size:1.55rem}
.nb-gtime{color:var(--muted);font-size:.9rem}
.nb-mk{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.nb-mkcol h4{font-family:var(--body);font-size:.82rem;font-weight:600;color:var(--muted);margin:0 0 6px}
.nb-opt{display:grid;grid-template-columns:24px minmax(0,1fr) auto;align-items:center;gap:8px;padding:8px 10px;border:1px solid var(--rule);border-radius:9px;margin-bottom:6px;position:relative}
.nb-opt img{width:22px;height:22px;object-fit:contain}
.nb-opt .what{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nb-opt .price{color:var(--muted);font-weight:500;margin-left:6px}
.nb-opt .chance{font-family:var(--display);font-weight:600;font-size:1.2rem}
.nb-opt.pick{border-color:var(--turf);background:var(--turf-soft)}
.nb-opt.value::before{content:"";position:absolute;left:-1px;top:8px;bottom:8px;width:4px;border-radius:2px;background:var(--flag)}
.nb-opt .right{display:flex;align-items:center;gap:8px}
.nb-chip{font-size:.72rem;font-weight:600;border-radius:4px;padding:1px 6px;background:var(--flag);color:var(--ink)}
.nb-key{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:.88rem;margin:0 0 14px}
.nb-key span{display:inline-flex;align-items:center;gap:7px}
.nb-key i{width:26px;height:16px;border-radius:5px;border:1px solid var(--turf);background:var(--turf-soft);display:inline-block}
.nb-gfoot{color:var(--muted);font-size:.88rem;margin-top:6px}
.nb-table{width:100%;border-collapse:collapse}
.nb-table th{font-size:.8rem;font-weight:600;color:var(--muted);text-align:left;padding:10px 14px;border-bottom:1px solid var(--rule);white-space:nowrap}
.nb-table td{padding:11px 14px;border-top:1px solid var(--rule);vertical-align:middle}
.nb-table tr:first-child td{border-top:0}
.nb-table .num{font-family:var(--display);font-weight:600;font-size:1.35rem;white-space:nowrap}
.nb-table .r{text-align:right}
.nb-player{display:flex;align-items:center;gap:10px}
.nb-player img{width:26px;height:26px;object-fit:contain}
.nb-player b{font-weight:600;display:block;line-height:1.15}
.nb-player span{color:var(--muted);font-size:.84rem}
.nb-range{position:relative;height:14px;min-width:170px}
.nb-range .w{position:absolute;top:6px;height:2px;background:#B9C3BC}
.nb-range .b{position:absolute;top:3px;height:8px;border-radius:2px;background:var(--turf)}
.nb-range .m{position:absolute;top:0;width:3px;height:14px;margin-left:-1px;background:var(--ink);border-radius:1px}
.nb-bar{position:relative;height:10px;min-width:150px;background:#E9EEE9;border-radius:2px}
.nb-bar i{position:absolute;left:0;top:0;bottom:0;background:var(--turf);border-radius:2px}
.nb-verdict{font-size:.88rem;white-space:nowrap}
.nb-verdict .yes{font-weight:600;color:var(--turf)}
.nb-verdict .no{color:var(--muted)}
.nb-table tr.sum td{background:#F7F9F6;font-weight:600}
.nb-table td.c,.nb-table th.c{text-align:right}
.nb-table td.plain{font-size:1rem;white-space:nowrap}
.nb-leg{display:grid;grid-template-columns:44px minmax(0,1fr) auto;align-items:center;gap:14px;padding:14px 18px;border-top:1px solid var(--rule)}
.nb-leg:first-child{border-top:0}
.nb-leg .was{color:var(--muted);font-size:.9rem;text-align:right}
.nb-facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 16px}
.nb-fact{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:12px 14px}
.nb-fact b{font-family:var(--display);font-weight:700;font-size:1.7rem;display:block;line-height:1.1}
.nb-fact span{color:var(--muted);font-size:.84rem}
@media (max-width:760px){.nb-facts{grid-template-columns:repeat(2,minmax(0,1fr))}}
.nb-load{text-align:center;margin:16vh auto 18px;max-width:680px}
.nb-load h1{font-family:var(--display);font-weight:700;font-size:2.6rem;line-height:1;margin:0 0 10px;color:var(--ink)}
.nb-load p{color:var(--muted);margin:0;font-size:1rem}
.nb-step{text-align:center;color:var(--muted);font-size:.92rem;margin-top:8px;min-height:1.4em}
.nb-axis{display:flex;justify-content:space-between;font-size:.74rem;color:var(--muted);font-weight:400}
@media (max-width:760px){
.nb-week{font-size:2.6rem}
.nb-pick{font-size:1.2rem}
.nb-odds{font-size:1.25rem}
.nb-row{grid-template-columns:30px 34px minmax(0,1fr) auto;row-gap:10px;padding:14px}
.nb-row .nb-meter{grid-column:2 / 4}
.nb-row .nb-edge{grid-column:4}
.nb-logo{width:32px;height:32px}
.nb-mk{grid-template-columns:1fr}
.nb-table th.hide,.nb-table td.hide{display:none}
}
"""


def style_tag() -> str:
    return f"<style>{CSS}</style>"


def _one_line(html: str) -> str:
    return "".join(part.strip() for part in html.splitlines())


def fmt_odds(o) -> str:
    return f"{int(o):+d}"


def pct(p, digits=0) -> str:
    return f"{100 * p:.{digits}f}%"


def logo(code, cls="nb-logo") -> str:
    if not code:
        return ""
    return f'<img class="{cls}" src="{logo_url(code)}" alt="{escape(nick(code))} logo" loading="lazy">'


def header(board: dict, title: str | None = None, sub: str = "") -> str:
    wk = f"Week {board['week']}" if board.get("week") else "No games yet"
    stamp = f'<div class="nb-stamp">Lines and projections updated {escape(board.get("built", ""))}</div>' if board.get("built") else ""
    return _one_line(f'''<div class="nb"><div class="nb-head"><div><div class="nb-week">{escape(title or wk)}</div>
<div class="nb-sub">{escape(sub)}</div></div>{stamp}</div></div>''')


def meter(b: dict) -> str:
    """Break-even sits on the yellow line in the middle; the marker is the model's win chance."""
    gap = max(-WINDOW, min(WINDOW, b["p_win"] - b["breakeven"]))
    pos = 50 + 50 * gap / WINDOW
    left, width = (50, pos - 50) if gap >= 0 else (pos, 50 - pos)
    short = "" if gap >= 0 else " short"
    return _one_line(f'''<div class="nb-meter" role="img" aria-label="Model {pct(b['p_win'])}, needs {pct(b['breakeven'])} to break even">
<div class="track"><div class="gain{short}" style="left:{left:.1f}%;width:{width:.1f}%"></div><div class="need"></div>
<div class="ball" style="left:{pos:.1f}%"></div></div>
<div class="lbl"><span>Need {pct(b['breakeven'], 1)}</span><span>Model <b>{pct(b['p_win'], 1)}</b></span></div></div>''')


def bet_row(rank: int, b: dict, games_by_id: dict | None = None) -> str:
    clears = b["ev"] > 0
    if b.get("team"):
        art = logo(b["team"])
    else:
        g = (games_by_id or {}).get(b.get("game_id"))
        art = f'<div class="nb-logos">{logo(g["away"], "")}{logo(g["home"], "")}</div>' if g else ""
    where = " ".join(x for x in (b.get("matchup", ""), b.get("kickoff", "")) if x)
    edge = f"{100 * b['edge']:+.1f}"
    return _one_line(f'''<div class="nb-row {'clears' if clears else 'short'}"><div class="nb-rank">{rank}</div><div>{art}</div>
<div><div class="nb-pick">{escape(b['pick'])}</div><div class="nb-meta"><span class="nb-tag">{escape(b['market'])}</span>{escape(where)}</div></div>
<div class="nb-odds">{fmt_odds(b['odds'])}</div>{meter(b)}
<div class="nb-edge"><b>{edge}</b><span>{'edge' if clears else 'short'}</span></div></div>''')


def top_list(bets: list[dict], games: list[dict], start: int = 1) -> str:
    if not bets and start > 1:
        return ""
    if not bets:
        return '<div class="nb"><div class="nb-panel"><div class="nb-empty">No games with posted lines yet. Lines usually appear early in the week; reload then.</div></div></div>'
    by_id = {g["game_id"]: g for g in games}
    rows = "".join(bet_row(i, b, by_id) for i, b in enumerate(bets, start))
    return f'<div class="nb"><div class="nb-panel">{rows}</div></div>'


def note(html: str) -> str:
    return f'<div class="nb"><p class="nb-note">{html}</p></div>'


def section(title: str, lead: str = "") -> str:
    lead_html = f'<p class="nb-lead">{escape(lead)}</p>' if lead else ""
    return f'<div class="nb"><div class="nb-h2">{escape(title)}</div>{lead_html}</div>'


def _opt(b: dict, label: str, code=None, pick: bool = False) -> str:
    """One side of a market. `pick` = the side the model predicts to win the bet; the Value chip marks any
    side whose price pays more than the model's chance requires (on a moneyline that can be the underdog)."""
    value = b["ev"] > 0
    cls = "nb-opt" + (" pick" if pick else "") + (" value" if value else "")
    art = logo(code, "") if code else "<span></span>"
    chip = '<span class="nb-chip">Value</span>' if value else ""
    return (f'<div class="{cls}">{art}<div class="what">{escape(label)}'
            f'<span class="price">{fmt_odds(b["odds"])}</span></div>'
            f'<div class="right">{chip}<div class="chance">{pct(b["p_win"])}</div></div></div>')


def lines_key() -> str:
    return ('<div class="nb"><div class="nb-key"><span><i></i>Model\'s pick in each market</span>'
            '<span><span class="nb-chip">Value</span>The price pays more than the model\'s chance requires</span></div></div>')


def game_card(g: dict) -> str:
    cols = []
    for title, key, label in (("Moneyline", "moneyline", lambda b: nick(b["team"])), ("Spread", "spread", lambda b: b["pick"]),
                              ("Total points", "total", lambda b: b["pick"])):
        if not g[key]:
            continue
        best = model_pick(g[key])
        cols.append(f"<h4>{title}</h4>" + "".join(_opt(b, label(b), b.get("team"), b is best) for b in g[key]))
    m = g["model_margin"]
    lean = "a toss-up" if abs(m) < 0.25 else f"{g['home_name'] if m > 0 else g['away_name']} by {abs(m):.1f}"
    tot = f" and {g['model_total']:.1f} total points" if g["total"] else ""
    body = "".join(f'<div class="nb-mkcol">{c}</div>' for c in cols)
    return _one_line(f'''<div class="nb"><div class="nb-game"><div class="nb-ghead"><div class="nb-gname">{escape(g['away_name'])} at {escape(g['home_name'])}</div>
<div class="nb-gtime">{'<span class="nb-tag">Your lines</span>' if g.get('yours') else ''}{escape(g['kickoff'])}</div></div><div class="nb-mk">{body}</div>
<div class="nb-gfoot">Model sees {escape(lean)}{tot}.</div></div></div>''')


def _player_cell(r) -> str:
    return (f'<div class="nb-player">{logo(r.team, "")}<div><b>{escape(str(r.player_display_name))}</b>'
            f'<span>{escape(nick(r.team))} vs {escape(nick(r.opponent_team))}</span></div></div>')


def _verdict(bets: list[dict]) -> str:
    """Best side of a saved sportsbook line, in plain words. Empty when no line is saved for the player."""
    if not bets:
        return ""
    b = max(bets, key=lambda x: x["edge"])
    if " over " in b["pick"]:
        what = "Over " + b["pick"].split(" over ")[-1].split(" ")[0]
    elif " under " in b["pick"]:
        what = "Under " + b["pick"].split(" under ")[-1].split(" ")[0]
    else:
        what = "Yes"
    txt = f"{what} at {fmt_odds(b['odds'])}"
    if b["ev"] > 0:
        return f'<span class="yes">{escape(txt)}, {100 * b["edge"]:+.1f} edge</span>'
    return f'<span class="no">{escape(txt)}, no edge</span>'


def _book_col(bets: list[dict]):
    """(header cell, cell-builder) for the sportsbook column; the column disappears when nothing is saved."""
    by_player = {}
    for b in bets:
        by_player.setdefault(b["player"], []).append(b)
    if not by_player:
        return "", lambda name: ""
    return ('<th class="hide">Your sportsbook</th>',
            lambda name: f'<td class="nb-verdict hide">{_verdict(by_player.get(name, []))}</td>')


def yards_table(df: pd.DataFrame, bets: list[dict], unit: str = "yards") -> str:
    """Projection, likely range and fair line for a yardage prop. `bets`: priced saved lines for these players."""
    if df is None or not len(df):
        return '<div class="nb"><div class="nb-panel"><div class="nb-empty">No players to show for this week yet.</div></div></div>'
    lo, hi = max(0.0, float(df.q10.min())), float(df.q90.max())
    span = max(hi - lo, 1.0)
    pos = lambda v: 100 * (min(max(v, lo), hi) - lo) / span
    book_th, book_td = _book_col(bets)
    rows = []
    for r in df.itertuples(index=False):
        fair = round(r.q50 - 0.5) + 0.5
        rng = (f'<div class="nb-range" title="Middle half of outcomes: {r.q25:.0f} to {r.q75:.0f}">'
               f'<div class="w" style="left:{pos(r.q10):.1f}%;width:{pos(r.q90) - pos(r.q10):.1f}%"></div>'
               f'<div class="b" style="left:{pos(r.q25):.1f}%;width:{pos(r.q75) - pos(r.q25):.1f}%"></div>'
               f'<div class="m" style="left:{pos(r.q50):.1f}%"></div></div>')
        rows.append(f'<tr><td>{_player_cell(r)}</td><td class="num r">{r.pred:.0f}</td><td class="hide">{rng}</td>'
                    f'<td class="num r">{fair:.1f}</td>{book_td(r.player_display_name)}</tr>')
    axis = f'<div class="nb-axis"><span>{lo:.0f}</span><span>Likely range</span><span>{hi:.0f} {unit}</span></div>'
    head = (f'<tr><th>Player</th><th class="r">Projection</th><th class="hide">{axis}</th>'
            f'<th class="r">Fair line</th>{book_th}</tr>')
    return f'<div class="nb"><div class="nb-panel"><table class="nb-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div></div>'


def count_table(df: pd.DataFrame, bets: list[dict], noun: str) -> str:
    """Counting props (receptions, passing TDs): projection and the chance of clearing the nearest line."""
    if df is None or not len(df):
        return '<div class="nb"><div class="nb-panel"><div class="nb-empty">No players to show for this week yet.</div></div></div>'
    book_th, book_td = _book_col(bets)
    rows = []
    for r in df.itertuples(index=False):
        bar = f'<div class="nb-bar"><i style="width:{100 * r.p_over:.0f}%"></i></div>'
        rows.append(f'<tr><td>{_player_cell(r)}</td><td class="num r">{r.pred:.1f}</td><td class="num r">{r.line:g}</td>'
                    f'<td class="num r">{pct(r.p_over)}</td><td class="hide">{bar}</td>{book_td(r.player_display_name)}</tr>')
    head = (f'<tr><th>Player</th><th class="r">Projected {escape(noun)}</th><th class="r">Line</th>'
            f'<th class="r">Chance to go over</th><th class="hide"></th>{book_th}</tr>')
    return f'<div class="nb"><div class="nb-panel"><table class="nb-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div></div>'


def td_table(df: pd.DataFrame, bets: list[dict]) -> str:
    if df is None or not len(df):
        return '<div class="nb"><div class="nb-panel"><div class="nb-empty">No players to show for this week yet.</div></div></div>'
    book_th, book_td = _book_col(bets)
    rows = []
    for r in df.itertuples(index=False):
        bar = f'<div class="nb-bar"><i style="width:{100 * r.p_td:.0f}%"></i></div>'
        rows.append(f'<tr><td>{_player_cell(r)}</td><td class="num r">{pct(r.p_td)}</td><td class="hide">{bar}</td>'
                    f'<td class="num r">{fmt_odds(r.fair_odds)}</td>'
                    f'{book_td(r.player_display_name)}</tr>')
    head = ('<tr><th>Player</th><th class="r">Chance to score</th><th class="hide"></th>'
            f'<th class="r">Fair price</th>{book_th}</tr>')
    return f'<div class="nb"><div class="nb-panel"><table class="nb-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div></div>'


def simple_table(df: pd.DataFrame) -> str:
    """A plain table of already-formatted text. Rows whose first cell doesn't start with 'Week' are summary rows."""
    if df is None or not len(df):
        return '<div class="nb"><div class="nb-panel"><div class="nb-empty">No finished weeks to show yet.</div></div></div>'
    head = "".join(f'<th{"" if i == 0 else " class=c"}>{escape(str(c))}</th>' for i, c in enumerate(df.columns))
    rows = []
    for row in df.itertuples(index=False):
        cls = "" if str(row[0]).startswith("Week") else ' class="sum"'
        cells = "".join(f'<td class="{"plain" if i == 0 else "num c"}">{escape(str(v))}</td>' for i, v in enumerate(row))
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f'<div class="nb"><div class="nb-panel"><table class="nb-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>'


def teaser_facts(t: dict) -> str:
    """The record behind the teaser page, with what it means at the prices sportsbooks usually charge."""
    cell = lambda big, small: f'<div class="nb-fact"><b>{big}</b><span>{escape(small)}</span></div>'
    return ('<div class="nb"><div class="nb-facts">'
            + cell(pct(t["rate"], 1), f'of {t["legs"]} legs like these have won since {t["first"]}')
            + cell(fmt_odds(t["breakeven"]), "break-even price for a two-team teaser at that rate")
            + cell(f'{100 * t["roi"][-120]:+.0f}%', "past return per teaser at -120")
            + cell(f'{100 * t["roi"][-130]:+.0f}%', "past return per teaser at -130")
            + '</div></div>')


def teaser_list(legs: list[dict]) -> str:
    if not legs:
        return ('<div class="nb"><div class="nb-panel"><div class="nb-empty">No games qualify this week. '
                'A leg needs an underdog of +1.5 to +2.5 in a game with a total of 49 or less.</div></div></div>')
    rows = "".join(
        f'<div class="nb-leg"><div>{logo(l["team"])}</div><div><div class="nb-pick">{escape(l["pick"])}</div>'
        f'<div class="nb-meta">{escape(l["matchup"])} {escape(l["kickoff"])}</div></div>'
        f'<div class="was">teased from {escape(l["was"])}<br>total {l["total"]:g}</div></div>' for l in legs)
    return f'<div class="nb"><div class="nb-panel">{rows}</div></div>'


def loading(title: str = "Setting up this week's board", sub: str = "") -> str:
    """Centered heading for the loading screen; the progress bar and step text sit under it."""
    return f'<div class="nb"><div class="nb-load"><h1>{escape(title)}</h1><p>{escape(sub)}</p></div></div>'


def loading_step(text: str) -> str:
    return f'<div class="nb"><div class="nb-step">{escape(text)}</div></div>'
