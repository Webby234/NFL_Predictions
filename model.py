"""The betting board.  Run:  streamlit run model.py

Four pages for someone placing bets, not studying the model:
  Home                 the ten best prices of the week, any market
  Moneyline & Spread   every game: moneyline, spread and total
  Player Props         passing, rushing and receiving projections
  Touchdowns           anytime-touchdown chances and passing touchdowns
  Teasers              underdog teaser legs that have paid historically
  Accuracy             how each of those pages did in past weeks

All numbers come from nfl_model.ui.board; all markup from nfl_model.ui.render.
Backtests and diagnostics live in the command line (python -m nfl_model ...), not here.
"""
from __future__ import annotations

import threading
import time

import pandas as pd
import streamlit as st

from nfl_model.config import HOSTED
from nfl_model.ui import board as B, lines_store, render as R, tracker

st.set_page_config(page_title="NFL betting board", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(R.style_tag(), unsafe_allow_html=True)

SHOW = 25      # players listed before "show all"
# On a shared host, typed-in lines live in the visitor's own session; on your computer they are saved to lines/.
STORE = st.session_state if HOSTED else None
KEPT = " Your lines stay in this browser tab only and are cleared when you close it." if HOSTED else ""


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


REFRESH_SECONDS = 3600     # rebuild the board at most once an hour
RETRY_SECONDS = 600        # after a failed rebuild, keep showing the last good board and try again in 10 minutes


@st.cache_resource
def _shared() -> dict:
    """One copy for every visitor: the current board, the Accuracy tables, and a lock so only one visitor builds."""
    return {"lock": threading.Lock(), "board": None, "hist": None, "at": 0.0}


def load_board_and_history():
    """The week's board and the Accuracy tables. Shows a centered progress screen while they are being built."""
    box = _shared()
    fresh = lambda: box["board"] is not None and time.time() - box["at"] < REFRESH_SECONDS
    if fresh():
        return box["board"], box["hist"]
    slot = st.empty()
    with slot.container():
        html(R.loading("Setting up this week's board",
                       "Downloading the latest games and running the numbers. This takes a minute or two."))
        mid = st.columns([1, 2, 1])[1]
        bar = mid.progress(0.0)
        step = mid.empty()

    def say(fraction: float, text: str) -> None:
        bar.progress(min(max(float(fraction), 0.0), 1.0))
        step.markdown(R.loading_step(text), unsafe_allow_html=True)

    say(0.0, "Getting started")
    with box["lock"]:
        if not fresh():                    # another visitor may have finished the build while we waited
            try:
                board = B.build_board(progress=lambda f, t: say(0.65 * f, t))
                hist = None
                if board["games"]:
                    try:
                        from nfl_model.ui import history
                        hist = history.load_or_build(lambda f, t: say(0.65 + 0.35 * f, t))
                    except Exception:
                        hist = None            # the Accuracy tab says so; the rest of the app still works
                box.update(board=board, hist=hist, at=time.time())
            except Exception:
                if box["board"] is None:
                    slot.empty()
                    raise
                box["at"] = time.time() - REFRESH_SECONDS + RETRY_SECONDS
    slot.empty()
    return box["board"], box["hist"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_record() -> str:
    """How the Home list has done so far (empty until logged picks have been played)."""
    from nfl_model.data import load_games, load_skill_games
    picks = tracker.load()
    return tracker.record_line(tracker.grade(picks, load_games(), load_skill_games())) if len(picks) else ""


def find(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Search box plus a show-all switch; returns the rows to list."""
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Find a player or team", key=f"q_{key}", placeholder="Find a player or team",
                      label_visibility="collapsed").strip().lower()
    if q:
        hit = df.player_display_name.str.lower().str.contains(q, regex=False) | \
              df.team.map(lambda t: q in B.nick(t).lower() or q == str(t).lower())
        return df[hit]
    show_all = c2.checkbox(f"Show all {len(df)}", key=f"all_{key}") if len(df) > SHOW else True
    return df if show_all else df.head(SHOW)


def line_editor(stat: str, df: pd.DataFrame, saved: pd.DataFrame, yes_only: bool = False) -> None:
    """Type in sportsbook lines for one market. Saved lines are priced and can reach the Home list."""
    mine = saved[saved.stat == stat].drop_duplicates("player").set_index("player") if len(saved) else pd.DataFrame()
    look = lambda col: df.player_display_name.map(mine[col]) if len(mine) else pd.Series([None] * len(df), index=df.index)
    table = pd.DataFrame({"Player": df.player_display_name, "Team": df.team.map(B.nick)})
    if yes_only:
        table["Odds"] = look("over_odds")
    else:
        table["Line"], table["Over odds"], table["Under odds"] = look("line"), look("over_odds"), look("under_odds")
    with st.expander("Add your sportsbook's lines"):
        st.caption("Type the odds your sportsbook offers for a touchdown (for example -130 or +145), then save." + KEPT
                   if yes_only else
                   "Type your sportsbook's line next to a player, then save. Leave the odds blank to use -110." + KEPT)
        num = {c: st.column_config.NumberColumn(c, step=0.5 if c == "Line" else 1) for c in table.columns[2:]}
        edited = st.data_editor(table.reset_index(drop=True), hide_index=True, disabled=["Player", "Team"],
                                column_config=num, use_container_width=True, key=f"ed_{stat}")
        if st.button("Save lines", key=f"save_{stat}"):
            rows = pd.DataFrame({"player": edited["Player"],
                                 "line": None if yes_only else edited["Line"],
                                 "over_odds": edited["Odds"] if yes_only else edited["Over odds"],
                                 "under_odds": None if yes_only else edited["Under odds"]})
            n = lines_store.save(board["season"], board["week"], stat, rows, store=STORE)
            st.session_state["saved_msg"] = f"Saved {n} line{'s' if n != 1 else ''}."
            st.rerun()


def game_line_editor(board: dict, games: list[dict]) -> None:
    """Type in the lines your own sportsbook offers. A better number than the listed one shows up as extra edge."""
    names = {"away_ml": "Away moneyline", "home_ml": "Home moneyline", "home_spread": "Home spread",
             "away_spread_odds": "Away spread odds", "home_spread_odds": "Home spread odds", "total": "Total",
             "over_odds": "Over odds", "under_odds": "Under odds"}
    current = pd.DataFrame([{"Game": g["matchup"], **{names[k]: g.get("lines", g["listed"])[k] for k in names}} for g in games])
    listed = pd.DataFrame([{"game_id": g["game_id"], **g["listed"]} for g in games])
    with st.expander("Enter your sportsbook's lines"):
        st.caption("These start as the listed lines. Change any number to what your sportsbook offers, then save. "
                   "Home spread is the home team's number, for example -2.5. Half a point in your favor is worth a lot." + KEPT)
        cfg = {v: st.column_config.NumberColumn(v, step=0.5 if k in ("home_spread", "total") else 1) for k, v in names.items()}
        edited = st.data_editor(current, hide_index=True, disabled=["Game"], column_config=cfg,
                                use_container_width=True, key="ed_games")
        c1, c2 = st.columns([1, 5])
        if c1.button("Save lines", key="save_games"):
            out = edited.rename(columns={v: k for k, v in names.items()}).assign(game_id=listed.game_id.values)
            n = lines_store.save_game_lines(board["season"], board["week"], out, listed, store=STORE)
            st.session_state["saved_msg"] = f"Using your lines for {n} game{'s' if n != 1 else ''}."
            st.rerun()
        if any(g.get("yours") for g in games) and c2.button("Go back to the listed lines", key="reset_games"):
            lines_store.save_game_lines(board["season"], board["week"], listed, listed, store=STORE)
            st.session_state["saved_msg"] = "Back to the listed lines."
            st.rerun()


board, hist = load_board_and_history()
if not board["games"]:
    html(R.header(board, sub="No upcoming games have posted lines yet. Lines usually appear early in the week."))
    st.stop()

props = board["props"]
games = B.apply_book_lines(board["games"], lines_store.load_game_lines(board["season"], board["week"], store=STORE))
saved = lines_store.load(board["season"], board["week"], store=STORE)
prop_bets = B.price_prop_lines(props, saved, games)
of = lambda stat: [b for b in prop_bets if b.get("stat") == stat]
if "saved_msg" in st.session_state:
    st.toast(st.session_state.pop("saved_msg"))

home, lines_tab, props_tab, td_tab, tease_tab, acc_tab = st.tabs(
    ["Home", "Moneyline & Spread", "Player Props", "Touchdowns", "Teasers", "Accuracy"])

with home:
    html(R.header(board, sub="The model's best-priced bets this week, across every market. The first three are "
                             "the ones that have held up best on past seasons."))
    top = B.top_bets(B.game_bets(games) + prop_bets)
    html(R.top_list(top[:3], games))
    if len(top) > 3:
        html(R.section("The next seven", "Smaller edges than the three above."))
        html(R.top_list(top[3:], games, start=4))
    try:                                   # keep a record of what was recommended; never let it break the page
        if not HOSTED:                 # a shared host has no private disk to keep a log on
            tracker.log(board, top, prop_bets)
        record = get_record()
    except Exception:
        record = ""
    if record:
        html(R.note(f"<b>Track record.</b> {record}"))
    html(R.note("<b>How to read it.</b> The yellow line is how often a bet has to win to break even at that price. "
                "The dot is how often the model expects it to win. Green past the line is edge; grey means it falls short."))
    if not prop_bets:
        html(R.note("Player props and touchdowns join this list once you add your sportsbook's lines on those tabs."))
    html(R.note("Moneyline underdogs are left off this list. You can still see them on the Moneyline & Spread tab."))

with lines_tab:
    html(R.section("Every game this week",
                   "The model's pick for every moneyline, spread and total. Percentages are its chance each bet wins."))
    html(R.lines_key())
    for g in games:
        html(R.game_card(g))
    game_line_editor(board, games)

with props_tab:
    passing, rushing, receiving = st.tabs(["Passing", "Rushing", "Receiving"])
    YARDS_LEAD = ("Projection is the average outcome. The bar shows the likely range, and the fair line is the number "
                  "the model sees as a coin flip: lean over if your sportsbook's line is lower, under if it is higher.")
    for tab, stat, title in ((passing, "qb_pass_yds", "Passing yards"), (rushing, "rush_yds", "Rushing yards"),
                             (receiving, "rec_yds", "Receiving yards")):
        with tab:
            df = props.get(stat)
            html(R.section(title, YARDS_LEAD))
            if df is None or not len(df):
                html(R.yards_table(None, []))
            else:
                html(R.yards_table(find(df, stat), of(stat)))
                line_editor(stat, df, saved)
            if stat == "rec_yds" and props.get("receptions") is not None:
                rc = props["receptions"]
                html(R.section("Receptions", "Projected catches and the chance of going over the nearest line."))
                html(R.count_table(find(rc, "receptions"), of("receptions"), "catches"))
                line_editor("receptions", rc, saved)
    html(R.note("Player lists come from who has played recently. Check injury reports and inactives before you bet."))

with td_tab:
    td = props.get("anytime_td")
    html(R.section("Anytime touchdown",
                   "Chance each player scores a rushing or receiving touchdown. Fair price is the break-even odds: "
                   "a sportsbook paying more than that is a price the model likes."))
    if td is None or not len(td):
        html(R.td_table(None, []))
    else:
        html(R.td_table(find(td, "anytime_td"), of("anytime_td")))
        line_editor("anytime_td", td, saved, yes_only=True)
    ptd = props.get("pass_tds")
    if ptd is not None and len(ptd):
        html(R.section("Passing touchdowns", "Projected touchdown passes for each starting quarterback."))
        html(R.count_table(ptd, of("pass_tds"), "touchdown passes"))
    html(R.note("Player lists come from who has played recently. Check injury reports and inactives before you bet."))

with tease_tab:
    t = board["teasers"]
    html(R.section("Teaser legs worth a look",
                   "A teaser moves the spread six points your way, and both teams must cover. Short underdogs in "
                   "lower-scoring games are the one kind of leg that has paid: six points takes them past both 3 and 7, "
                   "the two most common winning margins."))
    html(R.teaser_facts(t))
    legs = B.teaser_legs(games)
    html(R.teaser_list(legs))
    if len(legs) == 1:
        html(R.note("Only one game qualifies this week, and a teaser needs two legs. Pairing it with a leg that "
                    "does not qualify gives up the advantage."))
    html(R.note(f"<b>Check the price first.</b> Take a two-team, six-point teaser only at -120 or better. If the real "
                f"win rate is at the low end of its range ({100 * t['low']:.0f}%), -120 only breaks even and -130 loses. "
                "Many sportsbooks now charge -130 or more, or change the rules for ties, so read the terms."))
    html(R.note("This page does not use the model. It is a pricing pattern in the sportsbook's own lines, "
                "measured on closing spreads."))

with acc_tab:
    html(R.section("How the model has done",
                   "Past weeks, rebuilt the way this app would have shown them. The model only learned from earlier "
                   "seasons, so it had not seen any of the results it is graded on."))
    view = st.selectbox("Show accuracy for", ["Home", "Moneyline & Spread", "Player Props", "Touchdowns", "Teasers"],
                        key="acc_view")
    if not hist:
        html(R.note("Past weeks could not be loaded this time. The other tabs are unaffected."))
    if hist and view == "Home":
        html(R.section("Home list", "The best-priced game bets each week, 1 unit on each, at the sportsbook's closing price."))
        html(R.simple_table(hist["Home"]))
        html(R.note("Past Home lists cover moneylines, spreads and totals only. Player props need a sportsbook line to "
                    "be ranked, and there is no free record of past prop lines."))
        if get_record():
            html(R.note(f"<b>Picks logged by this app.</b> {get_record()}"))
    elif hist and view == "Moneyline & Spread":
        html(R.section("Model's pick in every game", "Wins and losses for the green-box pick in each market. Pushes are left out."))
        html(R.simple_table(hist["Moneyline & Spread"]))
        html(R.note("A spread or total bet at the usual -110 has to win about 52.4% of the time to break even."))
    elif hist and view == "Player Props":
        for title in ("Passing yards", "Rushing yards", "Receiving yards"):
            html(R.section(title, "Average miss is how far the projection was from the real number, in yards."))
            html(R.simple_table(hist[title]))
        html(R.section("Receptions", "Over/under call is against the nearest line to the player's recent average."))
        html(R.simple_table(hist["Receptions"]))
        html(R.note("Props are graded on what players actually did, not against sportsbook prices."))
    elif hist and view == "Teasers":
        html(R.section("Qualifying teaser legs", "Underdogs of +1.5 to +2.5 in games with a total of 49 or less, moved up six points."))
        html(R.simple_table(hist["Teasers"]))
        html(R.note("Returns assume two qualifying legs per teaser that win or lose independently."))
    elif hist:
        html(R.section("Anytime touchdown", "Expected scorers is the sum of the model's chances. If it is close to the "
                                            "actual count, the chances are about right."))
        html(R.simple_table(hist["Touchdowns"]))
        html(R.section("Passing touchdowns", "Average miss is in touchdown passes per quarterback."))
        html(R.simple_table(hist["Passing touchdowns"]))
    if hist:
        html(R.note("A single week is mostly luck. The season and last-season rows are the ones to read."))
