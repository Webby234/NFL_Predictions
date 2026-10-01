"""Streamlit dashboard.  Run:  streamlit run model.py

Thin UI over the `nfl_model` package - all modelling lives there.
Reports shown in the backtest tabs come from:  python -m nfl_model backtest  /  bets
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from nfl_model.betting.picks import DEFAULT_MODEL, picks_table, predict_upcoming, upcoming_games
from nfl_model.config import REPORT_DIR, logo_url
from nfl_model.features import load_feature_table

st.set_page_config(page_title="NFL Betting Model", page_icon="🏈", layout="wide")
st.title("🏈 NFL Model vs. Sportsbook")

MODELS = ["logit_with_spread", "gbm_with_spread", "logit_no_market_qb", "logit_no_market", "gbm_no_market"]


@st.cache_data(ttl=3600, show_spinner="Loading data and building features...")
def get_features() -> pd.DataFrame:
    return load_feature_table()


@st.cache_data(ttl=300, show_spinner="Fetching live odds...")
def get_live(api_key: str) -> pd.DataFrame:
    from nfl_model.betting.live_odds import live_odds_table
    return live_odds_table(api_key)


def read_report(name: str):
    path = REPORT_DIR / name
    return pd.read_csv(path) if path.exists() else None


# ----------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", MODELS, index=MODELS.index(DEFAULT_MODEL),
                              help="'with_spread' models start from the market line; 'no_market' models are "
                                   "independent of it and therefore disagree with it more (and, historically, lose more). "
                                   "'_qb' adds the expected starting quarterback.")
    min_edge = st.slider("Minimum edge (model prob - market prob)", 0.0, 0.10, 0.03, 0.005, format="%.3f")
    bankroll = st.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=50.0)
    kelly_fraction = st.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05,
                               help="Fraction of the full-Kelly stake. Full Kelly is too aggressive when "
                                    "probabilities are noisy.")
    odds_source = st.radio("Odds source", ["Schedule file (nflverse)", "Live - The Odds API"])
    api_key = ""
    if odds_source.startswith("Live"):
        api_key = st.text_input("Odds API key", value=os.environ.get("ODDS_API_KEY", ""), type="password")

df = get_features()
tab_bets, tab_model, tab_roi, tab_lines, tab_past = st.tabs(
    ["💰 Best Bets", "📊 Model vs Market", "📈 Betting Backtest", "📐 Spreads & Totals", "📅 Past Predictions"])

# ---------------------------------------------------------------- best bets ----
with tab_bets:
    roi_tbl = read_report("betting_thresholds.csv")
    if roi_tbl is None:
        st.warning("**Read this first.** No betting backtest found - run `python -m nfl_model bets` and check the "
                   "historical ROI (and its confidence interval) before trusting any edge shown here.")
    elif (roi_tbl["roi_ci_low"] > 0).any():
        st.warning("**Read this first.** Some backtest cells show a positive ROI interval, but many model/threshold "
                   "combinations were examined, so a few are expected by chance. Confirm on new data before staking real money.")
    else:
        st.warning("**Read this first.** In the walk-forward backtest (closing prices) no model/threshold shows an ROI "
                   "interval above zero, i.e. no moneyline edge distinguishable from luck. Treat these as model-vs-market "
                   "disagreements to investigate, not proven profitable bets.")
    pending = df[(df.game_type == "REG") & df.home_score.isna() & df.home_moneyline.notna()]
    weeks = sorted(pending[pending.season == pending.season.min()].week.unique()) if not pending.empty else []
    if not weeks:
        st.info("No upcoming games with betting lines found.")
    else:
        week = st.selectbox("Week", weeks)
        games = upcoming_games(df, week)
        scored = predict_upcoming(df, games, model=model_name)
        live = None
        if odds_source.startswith("Live"):
            try:
                live = get_live(api_key)
                st.caption(f"Live odds loaded for {len(live)} games - best available prices, consensus fair probability.")
            except Exception as exc:                     # network / key problems shouldn't crash the page
                st.error(f"Could not load live odds ({exc}). Showing schedule-file prices instead.")
        cands, bets = picks_table(scored, min_edge, bankroll, kelly_fraction, live=live)

        st.subheader(f"Recommended bets - week {week}")
        if bets.empty:
            st.write("No side clears the minimum edge at the current prices.")
        else:
            for _, r in bets.iterrows():
                c1, c2, c3 = st.columns([1, 4, 3])
                c1.image(logo_url(r["team"]), width=56)
                c2.markdown(f"**{r['team']} moneyline** ({r['moneyline']:+.0f}) &nbsp; _{r['away_team']} @ {r['home_team']}, "
                            f"{r['gameday']:%a %b %d}_")
                c3.markdown(f"Model **{r['model_prob']:.1%}** vs market {r['market_prob']:.1%} &nbsp;|&nbsp; "
                            f"edge **{r['edge']:+.1%}** &nbsp;|&nbsp; EV {r['ev']:+.1%} &nbsp;|&nbsp; stake **${r['stake']:.2f}**")

        with st.expander("All games - model vs market"):
            g = scored[["gameday", "away_team", "home_team", "away_qb", "home_qb", "p_home", "spread_line"]].copy()
            mk = cands[cands.side == "home"].set_index("game_id")
            g["market_home_prob"] = scored["game_id"].map(mk["market_prob"]).to_numpy()
            g["diff"] = g["p_home"] - g["market_home_prob"]
            st.dataframe(g.rename(columns={"p_home": "model_home_prob"}).style.format(
                {"model_home_prob": "{:.1%}", "market_home_prob": "{:.1%}", "diff": "{:+.1%}", "spread_line": "{:+.1f}"}),
                use_container_width=True, hide_index=True)

# ------------------------------------------------------------ model vs market ----
with tab_model:
    summary = read_report("backtest_summary.csv")
    if summary is None:
        st.info("Run `python -m nfl_model backtest` to generate this report.")
    else:
        st.markdown("Walk-forward probability quality: each season is predicted by a model trained only on "
                    "earlier seasons. **Lower log loss is better**; `ll_vs_market` < 0 would mean beating the "
                    "sportsbook.")
        st.dataframe(summary[["model", "n", "log_loss", "brier", "accuracy", "auc", "ece", "ll_vs_market",
                              "ll_vs_market_se"]].style.format(precision=4), use_container_width=True, hide_index=True)
        by_season = read_report("backtest_by_season.csv")
        if by_season is not None:
            st.line_chart(by_season.set_index("season")[["elo", "logit_no_market", "market_ml", "logit_with_spread"]])
        abl, abl_cmp = read_report("ablation_models.csv"), read_report("ablation_comparisons.csv")
        if abl_cmp is not None:
            st.markdown("**Do quarterback / injury features help?** Same-window paired comparisons "
                        "(rule set in advance: improvement only if below -2 standard errors).")
            st.dataframe(abl_cmp[["comparison", "ll_diff", "se", "z", "verdict"]].style.format(precision=4),
                         use_container_width=True, hide_index=True)
        if (REPORT_DIR / "calibration.png").exists():
            st.image(str(REPORT_DIR / "calibration.png"), caption="Calibration: predicted vs actual home-win rate")

# ------------------------------------------------------------ betting backtest ----
with tab_roi:
    thr, naive = read_report("betting_thresholds.csv"), read_report("betting_naive.csv")
    if thr is None:
        st.info("Run `python -m nfl_model bets` to generate this report.")
    else:
        st.markdown("Flat 1-unit bets at **closing** moneylines on the best side whenever the edge clears the "
                    "threshold. ROI = profit per unit staked; the 95% interval is a bootstrap over bets.")
        st.dataframe(thr.style.format(precision=4), use_container_width=True, hide_index=True)
        if naive is not None:
            st.markdown("**Reference - blind strategies** (their ROI is roughly the bookmaker's vig):")
            st.dataframe(naive.rename(columns={naive.columns[0]: "strategy"}).style.format(precision=4),
                         use_container_width=True, hide_index=True)
        st.caption(f"{len(thr)} model/threshold combinations are shown; some positive cells are expected by chance.")

# ------------------------------------------------------------ spreads & totals ----
with tab_lines:
    from nfl_model.betting.line_picks import upcoming_line_picks
    lmse, lroi = read_report("lines_mse.csv"), read_report("lines_roi.csv")
    st.subheader("This week: model number vs listed line")
    edge_pts = st.slider("Minimum edge (points)", 0.0, 4.0, 1.0, 0.5, key="line_edge")
    picks = upcoming_line_picks(df, week=None, min_edge=edge_pts)
    if picks.empty:
        st.info("No upcoming games with spreads and totals yet.")
    else:
        st.dataframe(picks, use_container_width=True, hide_index=True)
    st.warning("Backtested spread/total edges are not distinguishable from zero ROI. Treat these as "
               "disagreements to investigate, not sure bets.")
    if lmse is None:
        st.info("Run `python -m nfl_model lines` to generate the backtest reports.")
    else:
        st.markdown("**Accuracy vs the closing line** (squared error; `diff` > 0 means the model is worse than the line)")
        st.dataframe(lmse.style.format(precision=3), use_container_width=True, hide_index=True)
        if lroi is not None:
            st.markdown("**Walk-forward ROI by edge threshold** (flat stakes at listed odds, 95% bootstrap interval)")
            st.dataframe(lroi.style.format(precision=3), use_container_width=True, hide_index=True)
    for nm, title in (("lines_early_clv.csv", "Early-week prices (5-8 days out) and closing-line value"),
                      ("lines_move_attribution.csv", "What moves lines (diagnostic)")):
        t = read_report(nm)
        if t is not None:
            st.markdown(f"**{title}**")
            st.dataframe(t.style.format(precision=3), use_container_width=True, hide_index=True)

# ----------------------------------------------------------- past predictions ----
with tab_past:
    preds = read_report("predictions.csv")
    if preds is None:
        st.info("Run `python -m nfl_model backtest` to generate walk-forward predictions.")
    else:
        season = st.selectbox("Season", sorted(preds.season.unique(), reverse=True))
        ps = preds[preds.season == season]
        week = st.selectbox("Week ", sorted(ps.week.unique()), key="past_week")
        w = ps[ps.week == week].copy()
        pcol = f"p_{model_name}"
        w["model_pick"] = w.apply(lambda r: r.home_team if r[pcol] > 0.5 else r.away_team, axis=1)
        w["winner"] = w.apply(lambda r: r.home_team if r.home_win == 1 else r.away_team, axis=1)
        w["correct"] = w.model_pick == w.winner
        st.subheader(f"Model {int(w.correct.sum())}-{int((~w.correct).sum())}  ({w.correct.mean():.0%})   |   "
                     f"market favourite {int(((w.p_market_ml > 0.5) == (w.home_win == 1)).sum())}-"
                     f"{int(((w.p_market_ml > 0.5) != (w.home_win == 1)).sum())}")
        show = w[["away_team", "home_team", pcol, "p_market_ml", "model_pick", "winner", "correct"]]
        st.dataframe(show.rename(columns={pcol: "model_home_prob", "p_market_ml": "market_home_prob"}).style.format(
            {"model_home_prob": "{:.1%}", "market_home_prob": "{:.1%}"}), use_container_width=True, hide_index=True)
