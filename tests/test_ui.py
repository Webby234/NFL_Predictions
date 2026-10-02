import numpy as np, pandas as pd
from nfl_model.ui import board as B, lines_store, render as R


def test_side_trust_pulls_between_book_and_model():
    raw = B.side(0.60, 0.40, -110, -110, trust=1.0)
    none = B.side(0.60, 0.40, -110, -110, trust=0.0)
    half = B.side(0.60, 0.40, -110, -110, trust=0.5)
    assert abs(raw["p_win"] - 0.60) < 1e-9 and abs(none["p_win"] - 0.50) < 1e-9
    assert none["p_win"] < half["p_win"] < raw["p_win"]
    assert abs(raw["breakeven"] - 110 / 210) < 1e-9
    assert abs(raw["ev"] - (0.60 * (1 + 100 / 110) - 1)) < 1e-9
    assert none["ev"] < 0 < raw["ev"]                      # at the book's own number you just pay the vig


def test_side_handles_pushes_and_blank_odds():
    s = B.side(0.45, 0.45, None, None, trust=1.0)          # 10% push, blank odds -> -110
    assert s["odds"] == -110 and abs(s["p_win"] - 0.5) < 1e-9
    assert abs(s["ev"] - 0.9 * (0.5 * (1 + 100 / 110) - 1)) < 1e-9
    assert B.clean_odds(50) == -110 and B.clean_odds("+145") == 145 and B.clean_odds(float("nan")) == -110


def test_two_way_counts_ties_as_neither():
    over, under = B._two_way(3.0, np.array([-1.0, 0.0, 0.2, 1.0]), 3)     # results 2, 3, 3, 4 against a line of 3
    assert over == 0.25 and under == 0.25


def _bet(market, game, edge, **kw):
    return dict(market=market, game_id=game, edge=edge, ev=edge, pick=market, odds=-110, p_win=.5, breakeven=.5, **kw)


def test_top_bets_one_opinion_per_game_and_skips_passing_tds():
    bets = [_bet("Moneyline", "g1", .05), _bet("Spread", "g1", .04), _bet("Total", "g1", .03), _bet("Total", "g1", -.03),
            _bet("Spread", "g2", .02), _bet("Passing touchdowns", None, .30, stat="pass_tds", player="QB One"),
            _bet("Rushing yards", None, .01, stat="rush_yds", player="RB One"),
            _bet("Rushing yards", None, -.05, stat="rush_yds", player="RB One")]
    top = B.top_bets(bets)
    assert [(b["market"], b["game_id"]) for b in top] == [("Moneyline", "g1"), ("Total", "g1"), ("Spread", "g2"),
                                                           ("Rushing yards", None)]
    assert len(B.top_bets(bets, n=2)) == 2


def test_html_is_one_line_and_escapes_names():
    b = dict(B.side(0.6, 0.4, -110, -110, 1.0), market="Spread", pick="<script>x</script> -3", team="KC",
             matchup="A at B", kickoff="Sun 1:00 PM", game_id="g")
    out = R.top_list([b], [])
    assert "\n" not in out and "<script>" not in out and "&lt;script&gt;" in out
    assert "No games with posted lines" in R.top_list([], [])
    assert "\n\n" not in R.CSS.strip()


def test_meter_stays_inside_the_track():
    far = dict(p_win=0.95, breakeven=0.50); short = dict(p_win=0.10, breakeven=0.50)
    assert "left:100.0%" in R.meter(far) and "left:0.0%" in R.meter(short) and "gain short" in R.meter(short)


def test_saved_lines_round_trip_and_replace(tmp_path=None):
    import pathlib, tempfile
    old = lines_store.PATH
    lines_store.PATH = pathlib.Path(tmp_path or tempfile.mkdtemp()) / "lines" / "prop_lines.csv"
    try:
        rows = pd.DataFrame(dict(player=["A", "B", "C"], line=[70.5, None, "x"], over_odds=[-115, None, None], under_odds=[None] * 3))
        assert lines_store.save(2026, 4, "rush_yds", rows) == 1                     # blank and junk rows dropped
        lines_store.save(2026, 4, "anytime_td", pd.DataFrame(dict(player=["A"], line=[None], over_odds=[-120], under_odds=[None])))
        lines_store.save(2026, 4, "rush_yds", pd.DataFrame(dict(player=["B"], line=[55.5], over_odds=[None], under_odds=[None])))
        got = lines_store.load(2026, 4)
        assert sorted(zip(got.stat, got.player)) == [("anytime_td", "A"), ("rush_yds", "B")]
        assert lines_store.load(2026, 5).empty
    finally:
        lines_store.PATH = old


def test_timestamp_uses_no_platform_specific_codes():
    import inspect
    assert B._stamp(pd.Timestamp("2026-10-02 19:09")) == "Oct 2, 7:09 PM"
    assert B._stamp(pd.Timestamp("2026-10-12 00:05")) == "Oct 12, 12:05 AM"
    for mod in (B, R, lines_store):
        assert "%-" not in inspect.getsource(mod).replace("%-d / %-I", "")     # these break strftime on Windows


if __name__ == "__main__":
    bad = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("ok  ", n)
            except Exception as e: bad += 1; print("FAIL", n, repr(e))
    raise SystemExit(bad)
