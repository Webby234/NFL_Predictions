# NFL Model vs. Sportsbook

Machine-learning win probabilities for NFL games, compared against sportsbook odds to surface
bets where the model disagrees with the market. Player-prop models are planned (see Roadmap).

> **Status:** the honest walk-forward backtest (2015-2026) shows the model *matches* but does not
> *beat* the closing market, and moneyline ROI is not distinguishable from zero. Current-starting-QB
> features measurably improve a market-independent model (log loss 0.634 -> 0.628) but the market already
> prices that in; injury-burden features add nothing clear. The value of the project right now is a
> trustworthy evaluation harness - improve the model, re-run, see if it moves.

## Layout

```
nfl_model/
  config.py            paths, constants, team-name maps
  data/loader.py       nflverse CSV downloads (cached in ./data)
  features/            leak-free pre-game features
    team_features.py     Elo + exponentially weighted team stats, rest, division
    qb.py                current-starter rating (shrunk EPA/dropback), new-starter flag, expected starter
    injuries.py          injury burden by position group, weighted by snap share (2014+)
    pipeline.py          assembles the full table
  models/win_prob.py   model zoo (logistic / gradient boosting, with or without the spread)
  evaluation/          walk-forward backtest, log loss / Brier / calibration, feature ablation
  betting/
    odds.py            American odds <-> probability, vig removal
    edge.py            edge, expected value, Kelly stake sizing
    simulate.py        historical betting simulation (ROI + bootstrap CI)
    picks.py           train on finished games, score the next slate
    live_odds.py       optional live odds + line shopping (The Odds API)
  cli.py               python -m nfl_model <command>
tests/                 odds math, live-odds parsing, look-ahead-leak test
model.py               Streamlit betting board (4 tabs; markup in nfl_model/ui)
reports/               backtest outputs (CSVs, calibration plot)
```

## Setup

```
pip install -r requirements.txt
```

## Commands

```
python -m nfl_model backtest [--gbm xgboost] [--leak-demo]   # probability quality vs baselines and market
python -m nfl_model ablation                                  # do QB / injury features help? (paired tests)
python -m nfl_model bets                                      # ROI by edge threshold, Kelly simulation
python -m nfl_model picks [--week N] [--min-edge 0.03] [--live]
python -m nfl_model test                                      # unit tests + leakage test
streamlit run model.py                                        # dashboard
```

`--live` uses The Odds API for best-available prices across books; set `ODDS_API_KEY` first.
(Its response parser is tested on a sample in the documented format; the HTTP call itself has not
been run against the real service.) Without it, prices come from the nflverse schedule file.

## How evaluation avoids cheating

* Features for a game use only games played **before** it. State is snapshotted for every game of a
  week before any of that week's results are applied. `tests/test_no_leakage.py` erases a week's results
  (and later injury reports, snap counts, QB box scores) and checks the week's features do not change.
  It already caught one real (tiny) same-week leak, now fixed with a regression test in `test_features.py`.
* Feature changes are judged by paired log-loss differences against a same-training-window baseline,
  with the pass rule (below -2 standard errors) fixed in advance.
* Seasons are predicted by models trained only on earlier seasons (walk-forward, no random split).
* Every model and baseline is scored on the same games with log loss, Brier score and calibration,
  and compared to the vig-free market probability with a paired standard error.
* Betting simulations use closing prices (the sharpest available) and report bootstrap intervals;
  many model/threshold combinations are shown, so isolated positive cells are expected by chance.

## Roadmap

1. ~~Leak-free features and honest evaluation~~
2. ~~Odds comparison, edge/EV/Kelly, betting backtest, dashboard~~
3. ~~Quarterback and injury features~~ (QB helps a market-free model; neither beats the market)
4. Where an edge could plausibly live: line movement / opening-vs-closing prices (needs historical odds
   snapshots), pace/weather/totals, spread and totals markets (margin/total models)
5. Player props (receiving/rushing/passing yards) with real prop lines

## Known limits

* Historical QB/injury features use the actual starter and the final injury report (what the market knew at
  kickoff). Live picks before the report is final see less - run them late in the week and check the
  starters shown in the picks table.
* Injury features need snap counts, which only exist from 2013, so they are tested on 2017+.
* Live odds parsing is tested on a sample in the documented format but not against the real API.


## Spreads, totals and data checks

| Command | What it does |
|---|---|
| `python -m nfl_model check` | data sanity checks (duplicate games, line plausibility, stale cache, blank features) |
| `python -m nfl_model lines` | spread/totals backtests, early-week-line test + CLV, line-move diagnostic (`reports/lines_*.csv`) |
| `python -m nfl_model linepicks [--week N --min-edge PTS]` | model number vs listed spread/total for the upcoming week |

* `nfl_model/features/availability.py` tags every feature `early` (known from completed games) or `gameday`
  (needs same-week injury/QB news or game-day weather). Tests against early-week lines must use `usable_at(features, "early")`;
  a test fails if a feature is untagged.
* `data/line_history.csv` is built from nflverse/nfldata git history (`nfl_model/data/line_history.py`, needs a blob-less
  bare clone). Early-line "open" = earliest snapshot 5-8 days before kickoff, not the true first posting.
* Findings so far: no spread/total model beats the closing line; apparent early-week edges came from same-week news features.

## Player props (QB passing yards first)

| Command | What it does |
|---|---|
| `python -m nfl_model props backtest` | walk-forward 2015+ accuracy/calibration -> `reports/props_qb_*.csv` |
| `python -m nfl_model props predict [--week N]` | predicted mean + 10/25/50/75/90% quantiles for upcoming starters |
| `python -m nfl_model props price --file lines.csv` | P(over/under), edge and EV for lines you type in (`player,line,over_odds,under_odds`) |

* Features (`nfl_model/props/features.py`) are leak-free, snapshotted before each week: QB form, attempts and
  opponent pass defense are measured **relative to the league level at the time** (passing volume drifts by era;
  absolute levels and uncapped career-game counts caused year-over-year bias, found and fixed), plus game-script
  context from the market (implied points, spread, total relative to league level) and venue/weather.
* Scored on outcomes only (MAE, pinball loss, interval coverage, calibration of P(over) at a naive line) - there is
  no free historical prop pricing, so betting ROI is untested. Real lines can only be compared as you collect them.

### Rushing, receiving, receptions, passing TDs, anytime TD

`python -m nfl_model props {backtest|predict|price} --stat {rush_yds|rec_yds|receptions|pass_tds|anytime_td}`
(`backtest --stat all` runs everything). Code: `nfl_model/props/skill.py` (features), `skill_eval.py` (configs + walk-forward),
`skill_predict.py` (upcoming predictions + manual-line pricing). Yards use a mean + spread model, counts a negative-binomial,
anytime TD a logistic model. Eligibility comes from *pre-game* expected volume; players with no stats row that week (DNP) are absent.
Predictions infer participants from the last 4 weeks of usage (QBs from the schedule's listed starter) and drop players
listed Out/Doubtful when that week's report exists - check inactives yourself before betting anything.
Lines files: `player,line,over_odds,under_odds` (anytime_td: `player,odds[,no_odds]`).
Backtest findings: small accuracy gains over form-only baselines for yards and anytime TD, none for receptions or passing TDs.

## The app (`streamlit run model.py`)

Built for someone placing bets, not studying the model. No sidebar, four tabs:

| Tab | What it shows |
|---|---|
| Home | the ten bets where the model likes the price most this week, any market |
| Moneyline & Spread | every game: moneyline, spread and total, with the model's win chance for each side |
| Player Props | Passing / Rushing / Receiving: projection, likely range, fair line |
| Touchdowns | anytime-touchdown chance and fair price; passing touchdowns |

* `nfl_model/ui/board.py` puts every bet on one yardstick (win chance, break-even, edge, expected value).
  Win chances are pulled toward the sportsbook's by the `TRUST` factors, because on past seasons most of the model's
  disagreement with the book did not hold up (`python -m nfl_model.ui.calibrate` recomputes them -> `reports/ui_trust.csv`).
  Props have no price history, so they use an assumed factor of 0.5.
* Player props only reach the Home list after you type your sportsbook's lines into the "Add your sportsbook's lines"
  box on each tab. They are saved to `lines/prop_lines.csv`, which doubles as the start of a real prop price history.
* `nfl_model/ui/render.py` is plain HTML/CSS; `python -m nfl_model.ui.preview out.html` writes a static copy of the pages.
* Colours come from `.streamlit/config.toml` plus the CSS in `render.py`. Backtests and diagnostics stay in the CLI.
