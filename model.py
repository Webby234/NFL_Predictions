"""The betting board.  Run:  streamlit run model.py

Four pages for someone placing bets, not studying the model:
  Home                 the ten best prices of the week, any market
  Moneyline & Spread   every game: moneyline, spread and total
  Player Props         passing, rushing and receiving projections
  Touchdowns           anytime-touchdown chances and passing touchdowns

All numbers come from nfl_model.ui.board; all markup from nfl_model.ui.render.
Backtests and diagnostics live in the command line (python -m nfl_model ...), not here.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl_model.ui import board as B, lines_store, render as R

st.set_page_config(page_title="NFL betting board", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(R.style_tag(), unsafe_allow_html=True)

SHOW = 25      # players listed before "show all"


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


@st.cache_resource(ttl=3600, show_spinner="Getting this week's lines and projections...")
def get_board() -> dict:
    return B.build_board()


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
        st.caption("Type the odds your sportsbook offers for a touchdown (for example -130 or +145), then save."
                   if yes_only else
                   "Type your sportsbook's line next to a player, then save. Leave the odds blank to use -110.")
        num = {c: st.column_config.NumberColumn(c, step=0.5 if c == "Line" else 1) for c in table.columns[2:]}
        edited = st.data_editor(table.reset_index(drop=True), hide_index=True, disabled=["Player", "Team"],
                                column_config=num, use_container_width=True, key=f"ed_{stat}")
        if st.button("Save lines", key=f"save_{stat}"):
            rows = pd.DataFrame({"player": edited["Player"],
                                 "line": None if yes_only else edited["Line"],
                                 "over_odds": edited["Odds"] if yes_only else edited["Over odds"],
                                 "under_odds": None if yes_only else edited["Under odds"]})
            n = lines_store.save(board["season"], board["week"], stat, rows)
            st.session_state["saved_msg"] = f"Saved {n} line{'s' if n != 1 else ''}."
            st.rerun()


board = get_board()
if not board["games"]:
    html(R.header(board, sub="No upcoming games have posted lines yet. Lines usually appear early in the week."))
    st.stop()

games, props = board["games"], board["props"]
saved = lines_store.load(board["season"], board["week"])
prop_bets = B.price_prop_lines(props, saved)
of = lambda stat: [b for b in prop_bets if b.get("stat") == stat]
if "saved_msg" in st.session_state:
    st.toast(st.session_state.pop("saved_msg"))

home, lines_tab, props_tab, td_tab = st.tabs(["Home", "Moneyline & Spread", "Player Props", "Touchdowns"])

with home:
    html(R.header(board, sub="The ten bets where the model likes the price most this week, across every market."))
    html(R.top_list(B.top_bets(B.game_bets(games) + prop_bets), games))
    html(R.note("<b>How to read it.</b> The yellow line is how often a bet has to win to break even at that price. "
                "The dot is how often the model expects it to win. Green past the line is edge; grey means it falls short."))
    html(R.note("<b>Keep it in proportion.</b> On past seasons this model has not beaten sportsbook prices after the "
                "vig, so the win chances shown are already pulled most of the way toward the sportsbook's. "
                "Use the list as leads to look into, not sure things."))
    if not prop_bets:
        html(R.note("Player props and touchdowns join this list once you add your sportsbook's lines on those tabs."))

with lines_tab:
    html(R.section("Every game this week",
                   "Percentages are the model's chance each bet wins. A green box marks a price the model sees value in."))
    for g in games:
        html(R.game_card(g))

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
        html(R.note("Passing-touchdown chances have run too confident in testing, so they are left out of the Home list."))
    html(R.note("Player lists come from who has played recently. Check injury reports and inactives before you bet."))
