"""Command line entry point:  python -m nfl_model <command>

  backtest   walk-forward probability backtest vs baselines + market   (writes reports/)
  bets       betting simulation: ROI by edge threshold, Kelly sim       (writes reports/)
  ablation   do QB / injury features help? (same-window paired comparisons)   (writes reports/)
  picks      edges and recommended bets for the next slate of games
  test       run the unit + leakage tests
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import DATA_DIR, REPORT_DIR

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def cmd_backtest(args) -> None:
    from .data import load_games, load_team_stats
    from .evaluation.backtest import PRED_ORDER, calibration_table, plot_calibration, summarize, walk_forward
    from .features import load_feature_table

    games, stats = load_games(), load_team_stats(1999)
    df = load_feature_table()
    out = walk_forward(df, args.gbm)
    pooled, by_season = summarize(out)

    REPORT_DIR.mkdir(exist_ok=True)
    pooled.round(5).to_csv(REPORT_DIR / "backtest_summary.csv")
    by_season.round(5).to_csv(REPORT_DIR / "backtest_by_season.csv")
    keep = (["game_id", "season", "week", "gameday", "home_team", "away_team", "home_win", "spread_line",
             "home_moneyline", "away_moneyline"] + [f"p_{n}" for n in PRED_ORDER])
    out[keep].to_csv(REPORT_DIR / "predictions.csv", index=False)
    plot_calibration(out, REPORT_DIR / "calibration.png")

    print(f"\nWalk-forward backtest: {len(out)} games, seasons {out.season.min()}-{out.season.max()}; GBM = {args.gbm}")
    print("Lower log loss / Brier / ECE is better; 'll_vs_market' negative = beats the market.\n")
    print(pooled[["n", "log_loss", "brier", "accuracy", "auc", "ece", "ll_vs_market", "ll_vs_market_se"]].round(4).to_string())
    print("\nLog loss by season:\n", by_season.round(4).to_string())
    print("\nCalibration (logit_no_market):\n", calibration_table(out, "logit_no_market").round(3).to_string(index=False))
    if args.leak_demo:
        from .evaluation.leak_demo import leak_demo
        print("\nLeak reproduction of the original evaluation style:")
        leak_demo(df, stats, args.gbm)


def cmd_bets(args) -> None:
    import numpy as np
    from .betting.simulate import kelly_simulation, naive_strategies, threshold_table
    from .evaluation.backtest import walk_forward
    from .features import load_feature_table

    df = load_feature_table()
    out = walk_forward(df, args.gbm)
    REPORT_DIR.mkdir(exist_ok=True)

    models = ["logit_with_spread", "gbm_with_spread", "logit_no_market", "logit_no_market_qb", "gbm_no_market"]
    tables = [threshold_table(out, f"p_{m}").assign(model=m) for m in models]
    thr = pd.concat(tables)[["model"] + list(tables[0].columns[:-1])]
    naive = naive_strategies(out)
    thr.round(4).to_csv(REPORT_DIR / "betting_thresholds.csv", index=False)
    naive.round(4).to_csv(REPORT_DIR / "betting_naive.csv")

    n_looks = len(models) * len(thr.min_edge.unique())
    print(f"\nFlat 1-unit moneyline bets at closing prices, {len(out)} games, seasons {out.season.min()}-{out.season.max()}")
    print("\nReference - blind strategies (cost of the vig):")
    print(naive[["bets", "win_rate", "roi", "roi_ci_low", "roi_ci_high"]].round(4).to_string())
    for m in models:
        print(f"\n{m}:")
        print(thr[thr.model == m].drop(columns="model")[["min_edge", "bets", "win_rate", "units", "roi", "roi_ci_low", "roi_ci_high"]].round(4).to_string(index=False))
    print(f"\nCAUTION: {n_looks} model/threshold combinations are shown. With that many looks, one or two positive "
          f"ROI cells are expected by chance; a CI that spans 0 is not evidence of an edge.")

    bets, stats = kelly_simulation(out, "p_logit_with_spread", min_edge=0.04)
    print(f"\nQuarter-Kelly (5% cap) on logit_with_spread, edge>=4%: {stats['bets']} bets, "
          f"$100 -> ${stats['final_bankroll']:.0f}, max drawdown {stats['max_drawdown']:.0%}")


def cmd_ablation(args) -> None:
    from .evaluation.ablation import run_ablation
    from .features import load_feature_table

    models, comps, out = run_ablation(load_feature_table())
    REPORT_DIR.mkdir(exist_ok=True)
    models.round(5).to_csv(REPORT_DIR / "ablation_models.csv")
    comps.round(5).to_csv(REPORT_DIR / "ablation_comparisons.csv", index=False)
    print(f"\nFeature ablation: {len(out)} games, seasons {out.season.min()}-{out.season.max()}. Log loss, lower is better.\n"
          "Rule set in advance: a feature set 'IMPROVES' only if its paired difference is below -2 standard errors.\n")
    print(models[["n", "log_loss", "brier", "accuracy", "ece", "ll_vs_market", "ll_vs_market_se"]].round(4).to_string())
    print("\n", comps[["comparison", "ll_diff", "se", "z", "verdict"]].round(4).to_string(index=False))


def cmd_picks(args) -> None:
    from .betting.picks import DEFAULT_MODEL, picks_table, predict_upcoming, upcoming_games
    from .features import load_feature_table

    df = load_feature_table()
    games = upcoming_games(df, args.week)
    if games.empty:
        print("No upcoming games with lines found.")
        return
    gp = predict_upcoming(df, games, model=args.model or DEFAULT_MODEL, gbm_kind=args.gbm)
    live = None
    if args.live:
        from .betting.live_odds import live_odds_table
        live = live_odds_table()
        print(f"Live odds for {len(live)} games from The Odds API")
    cands, bets = picks_table(gp, args.min_edge, args.bankroll, live=live)
    wk, season = int(games.week.iloc[0]), int(games.season.iloc[0])
    print(f"\n{season} week {wk} - model {args.model or DEFAULT_MODEL}, min edge {args.min_edge:.1%}, "
          f"{'live best prices' if live is not None else 'schedule-file prices'}")
    cols = ["gameday", "away_team", "home_team", "team", "moneyline", "model_prob", "market_prob", "edge", "ev", "stake",
            "away_qb", "home_qb"]
    if bets.empty:
        print("No bets clear the edge threshold.")
    else:
        print(bets[cols].assign(gameday=bets.gameday.dt.strftime("%a %m-%d")).round(3).to_string(index=False))
    print("\nBacktest reminder: see `python -m nfl_model bets` - historical ROI is not distinguishable from zero.")


def cmd_check(args) -> None:
    from .data.validate import check
    from .features import INJURY_FEATURES, NO_MARKET, QB_FEATURES, load_feature_table
    rows = check(load_feature_table(), NO_MARKET + QB_FEATURES + INJURY_FEATURES)
    for lvl, msg in rows:
        print(f"[{lvl}] {msg}")
    sys.exit(1 if any(l == "ERROR" for l, _ in rows) else 0)


def cmd_lines(args) -> None:
    """Spread/totals backtests, early-line test, line-move diagnostic -> reports/lines_*.csv"""
    from .evaluation import early_lines, line_moves, run_lines
    run_lines.main()
    if (DATA_DIR / "line_history.csv").exists():
        early_lines.run().to_csv(REPORT_DIR / "lines_early_clv.csv", index=False)
        line_moves.run().to_csv(REPORT_DIR / "lines_move_attribution.csv", index=False)
    else:
        print("no data/line_history.csv - skipped early-line tests (see nfl_model/data/line_history.py)")


def cmd_linepicks(args) -> None:
    from .betting.line_picks import upcoming_line_picks
    from .features import load_feature_table
    t = upcoming_line_picks(load_feature_table(), args.week, args.min_edge)
    print("No upcoming games with lines." if t.empty else t.to_string(index=False))
    print("\nBacktests: spread/total edges are not distinguishable from zero ROI. Disagreements, not sure bets.")


def cmd_props(args) -> None:
    """Player props. python -m nfl_model props {backtest|predict|price} [--stat NAME] [--file lines.csv]

    stats: qb_pass_yds (default), rush_yds, rec_yds, receptions, pass_tds, anytime_td, all (backtest only)
    """
    stat = args.stat
    if stat == "all" and args.action != "backtest":
        sys.exit("--stat all only works with backtest")
    if args.action == "backtest":
        if stat in ("qb_pass_yds", "all"):
            from .props import evaluate as E
            from .props.features import build_qb_prop_table
            out, _ = E.walk_forward(build_qb_prop_table())
            summ = E.summarize(out)
            summ.round(4).to_csv(REPORT_DIR / "props_qb_summary.csv", index=False)
            E.by_season(out).to_csv(REPORT_DIR / "props_qb_by_season.csv")
            E.over_calibration(out, E.APP_MODEL).to_csv(REPORT_DIR / "props_qb_calibration.csv")
            print("== qb_pass_yds"); print(summ.round(3).to_string(index=False))
            print(E.over_calibration(out, E.APP_MODEL))
        if stat != "qb_pass_yds":
            from .props import skill_eval as S
            names = list(S.STATS) if stat == "all" else [stat]
            table = S.build_skill_table()
            summaries = []
            for k in names:
                out = S.walk_forward(table, S.STATS[k]); summ = S.summarize(out, S.STATS[k])
                summaries.append(summ); cal = S.calibration(out, S.STATS[k])
                cal.to_csv(REPORT_DIR / f"props_{k}_calibration.csv")
                print(f"== {k}"); print(summ.round(4).to_string(index=False)); print(cal)
            pd.concat(summaries).round(5).to_csv(REPORT_DIR / "props_skill_summary.csv", index=False)
        print("\nScored on outcomes only: no historical prop prices exist for free, so ROI is untested.")
        return
    if stat == "qb_pass_yds":
        from .props.predict import predict_upcoming, price_lines
        pred = predict_upcoming(args.week)
        if pred.empty:
            print("No upcoming games with listed quarterbacks."); return
        if args.action == "predict":
            print(pred.drop(columns=["season"]).to_string(index=False))
            print("\nqb_games=0 means a QB new to the data: treat with extra caution.")
            return
    else:
        from .props import skill_predict as SP
        table, up = SP.upcoming_table(args.week)
        if up is None:
            print("No upcoming games."); return
        pred = SP.predict_stat(table, up, stat)
        if args.action == "predict":
            print(pred.to_string(index=False))
            print("\nParticipants are inferred from recent usage; injuries only if the report is published. Check inactives.")
            return
        price_lines = SP.price_lines
    if not args.file:
        sys.exit("price needs --file lines.csv (player,line,over_odds,under_odds; anytime_td: player,odds[,no_odds])")
    res = price_lines(pred, pd.read_csv(args.file), args.min_edge)
    print(res.to_string(index=False))
    print("\nUntested against real prop prices - treat as a second opinion, not a bet signal.")


def cmd_track(args) -> None:
    """Grade the picks the app has logged (lines/picks_log.csv) -> reports/picks_graded.csv"""
    from .data import load_games, load_skill_games
    from .ui import tracker
    picks = tracker.load()
    if picks.empty:
        print("No picks logged yet. Open the app (streamlit run model.py) before the games to log the week's list."); return
    graded = tracker.grade(picks, load_games(), load_skill_games())
    graded.to_csv(REPORT_DIR / "picks_graded.csv", index=False)
    summ = tracker.summarize(graded)
    print(f"{len(graded)} picks logged, {graded.result.notna().sum()} settled.")
    print(summ.to_string(index=False) if len(summ) else "Nothing settled yet.")
    print("\nA few weeks is noise: it takes hundreds of bets to tell skill from luck.")


def cmd_test(args) -> None:
    import subprocess
    sys.exit(max(subprocess.call([sys.executable, "-m", m]) for m in
                 ("tests.test_odds", "tests.test_live_odds", "tests.test_features", "tests.test_lines", "tests.test_data_checks", "tests.test_props", "tests.test_skill_props", "tests.test_ui", "tests.test_no_leakage")))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m nfl_model", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("backtest", cmd_backtest), ("bets", cmd_bets), ("ablation", cmd_ablation), ("picks", cmd_picks), ("check", cmd_check), ("lines", cmd_lines), ("linepicks", cmd_linepicks), ("props", cmd_props), ("track", cmd_track), ("test", cmd_test)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name in ("backtest", "bets", "picks"):
            p.add_argument("--gbm", choices=["sklearn", "xgboost"], default="sklearn")
        if name == "backtest":
            p.add_argument("--leak-demo", action="store_true")
        if name == "props":
            p.add_argument("action", choices=["backtest", "predict", "price"])
            p.add_argument("--stat", default="qb_pass_yds",
                           choices=["qb_pass_yds", "rush_yds", "rec_yds", "receptions", "pass_tds", "anytime_td", "all"])
            p.add_argument("--week", type=int)
            p.add_argument("--file")
            p.add_argument("--min-edge", type=float, default=0.03)
        if name == "linepicks":
            p.add_argument("--week", type=int)
            p.add_argument("--min-edge", type=float, default=1.0)
        if name == "picks":
            p.add_argument("--week", type=int)
            p.add_argument("--model")
            p.add_argument("--min-edge", type=float, default=0.03)
            p.add_argument("--bankroll", type=float, default=100.0)
            p.add_argument("--live", action="store_true", help="use The Odds API (needs ODDS_API_KEY)")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
