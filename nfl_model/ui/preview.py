"""Static preview of the app's pages (no Streamlit needed):  python -m nfl_model.ui.preview [out.html]"""
import sys
import pandas as pd
from . import board as B, lines_store, render as R


def build_html(board: dict, lines: pd.DataFrame) -> str:
    games, props = board["games"], board["props"]
    prop_bets = B.price_prop_lines(props, lines)
    top = B.top_bets(B.game_bets(games) + prop_bets)
    of = lambda stat: [b for b in prop_bets if b.get("stat") == stat]
    parts = [R.header(board, sub="The ten spots where the model likes the price most this week."),
             R.top_list(top, games), R.section("Moneyline & Spread")]
    parts += [R.game_card(g) for g in games[:3]]
    for stat, title in (("qb_pass_yds", "Passing yards"), ("rush_yds", "Rushing yards"), ("rec_yds", "Receiving yards")):
        if stat in props:
            parts += [R.section(title), R.yards_table(props[stat].head(8), of(stat))]
    if "receptions" in props:
        parts += [R.section("Receptions"), R.count_table(props["receptions"].head(6), of("receptions"), "catches")]
    if "anytime_td" in props:
        parts += [R.section("Anytime touchdown"), R.td_table(props["anytime_td"].head(8), of("anytime_td"))]
    body = "".join(parts)
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'{R.style_tag()}</head><body style="margin:0;padding:32px 20px"><div style="max-width:1080px;margin:0 auto">{body}</div></body></html>')


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "board_preview.html"
    bd = B.build_board()
    open(out, "w", encoding="utf-8").write(build_html(bd, lines_store.load(bd["season"], bd["week"])))
    print("wrote", out)
