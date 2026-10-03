# NFL Betting Board

A local web app that predicts NFL games and player stats, compares them with sportsbook lines, and shows
where the price looks best each week: moneylines, spreads, totals, player props, touchdowns and teasers.
It runs on your own computer, uses free public data, and needs no account or API key.

> **Read this first.** On past seasons the model has **not** beaten sportsbook closing lines. The app says so on
> its own pages and shows its record in the Accuracy tab. Treat everything it shows as a lead to check, not a
> sure thing. Nothing here is betting advice. Only bet what you can afford to lose, and only where it is legal for you.

## Contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [Using the app each week](#using-the-app-each-week)
- [How to read the pages](#how-to-read-the-pages)
- [Command line reference](#command-line-reference)
- [Where your files live](#where-your-files-live)
- [How it works](#how-it-works)
- [How accurate is it](#how-accurate-is-it)
- [Project layout](#project-layout)
- [Running the tests](#running-the-tests)
- [Troubleshooting](#troubleshooting)
- [Known limits](#known-limits)
- [Data and credits](#data-and-credits)

## What you get

| Tab | What it shows |
|---|---|
| **Home** | The best-priced bets of the week across every market. The best three come first, then the next seven. |
| **Moneyline & Spread** | Every game: moneyline, spread and total, with the model's pick and win chance for each. |
| **Player Props** | Passing, rushing and receiving: projection, likely range and a fair line for each player. |
| **Touchdowns** | Each player's chance to score and the break-even odds, plus passing touchdowns. |
| **Teasers** | This week's underdog teaser legs that have paid historically, and the price you need. |
| **Accuracy** | How every page above did in past weeks, this season and last season. |

## Quick start

### 1. What you need

- **Python 3.10 or newer** (built and tested on 3.11). Check with `python --version`.
  Download from [python.org](https://www.python.org/downloads/). On Windows, tick "Add Python to PATH" in the installer.
- **Git**, to download the project ([git-scm.com](https://git-scm.com/downloads)). Or use GitHub's "Download ZIP" button.
- **An internet connection.** The app downloads game and player data, team logos and fonts.
- About **60 MB** of free disk space for the data, plus room for the Python packages.

### 2. Download the project

```
git clone https://github.com/Webby234/NFL_Predicitions.git
cd NFL_Predicitions
```

### 3. Install the dependencies

A virtual environment keeps these packages separate from the rest of your computer. It is optional but recommended.

Windows:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell says "running scripts is disabled on this system", run this first. It applies only to the
current window:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

macOS or Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Start the app

```
streamlit run model.py
```

Your browser opens at <http://localhost:8501>. If it does not, paste that address into your browser.
To stop the app, press `Ctrl+C` in the terminal.

**The first start is slow.** It downloads about 40 MB of data and builds the week's projections, which takes
two to three minutes. Later starts take about half a minute, and the data refreshes itself as games are played.
The first start after each week's games also takes about a minute longer while the Accuracy tab is rebuilt.

## Using the app each week

1. **Open the app before the games.** Opening it saves that week's Home list so it can be graded later.
2. **Enter your own sportsbook's lines** (optional, and worth doing). On the Moneyline & Spread tab, open
   "Enter your sportsbook's lines". The table starts with the listed lines. Change any number to what your
   sportsbook offers and press **Save lines**. A better number than the listed one shows up as extra edge.
3. **Enter prop lines** (optional). On Player Props and Touchdowns, open "Add your sportsbook's lines" and type the
   line and odds next to a player. Props only appear on the Home list after you do this, because there is no free
   source of prop prices.
4. **Check the Teasers tab** for qualifying legs and the price you need.
5. **After the games,** open the Accuracy tab, or run `python -m nfl_model track`, to see how the picks did.

## How to read the pages

**The meter on Home.** The yellow line is how often a bet must win to break even at that price. The dot is how
often the model expects it to win. Green past the line is edge. Grey means the bet falls short of break-even.

**Odds** are American odds. `-110` means you bet 110 to win 100. `+150` means you bet 100 to win 150.

**Green boxes on Moneyline & Spread** mark the model's pick in each market. A yellow **Value** chip marks any
side whose price pays more than the model's win chance requires. On a moneyline the pick and the value can be
different teams: the model can expect the favorite to win while the underdog's price is the better bet.

**Fair line** on Player Props is the number the model sees as a coin flip. Lean over if your sportsbook's line is
lower, under if it is higher.

**Fair price** on Touchdowns is the break-even odds. A sportsbook paying more than that is a price the model likes.

**Home spread** in the lines table is the home team's number as a sportsbook shows it: `-2.5` means the home team
is favored by 2.5.

## Command line reference

Everything the app shows, and the tests behind it, can also be run from a terminal in the project folder.

| Command | What it does |
|---|---|
| `streamlit run model.py` | Start the app. |
| `python -m nfl_model track` | Grade the picks the app has logged. Writes `reports/picks_graded.csv`. |
| `python -m nfl_model check` | Check the downloaded data for problems. |
| `python -m nfl_model test` | Run every test (about five minutes on a fresh download). |
| `python -m nfl_model picks [--week N] [--min-edge 0.03]` | Moneyline picks for the upcoming week. |
| `python -m nfl_model linepicks [--week N] [--min-edge 1.0]` | The model's number against each spread and total. |
| `python -m nfl_model props predict --stat STAT` | Player projections for the upcoming week. |
| `python -m nfl_model props price --stat STAT --file lines.csv` | Price prop lines from a file. |
| `python -m nfl_model props backtest --stat all` | Test the prop models on past seasons. |
| `python -m nfl_model backtest` | Test the win-probability models on past seasons. |
| `python -m nfl_model bets` | Simulate moneyline betting on past seasons. |
| `python -m nfl_model lines` | Test the spread and total models on past seasons. |
| `python -m nfl_model ablation` | Test whether quarterback and injury inputs help. |
| `python -m nfl_model.ui.calibrate` | Recompute how far win chances are pulled toward the sportsbook. |
| `python -m nfl_model.ui.preview out.html` | Write a static copy of the app's pages. |

`STAT` is one of `qb_pass_yds`, `rush_yds`, `rec_yds`, `receptions`, `pass_tds`, `anytime_td`.

A prop lines file is a CSV with the columns `player,line,over_odds,under_odds`. For `anytime_td` the columns are
`player,odds` with an optional `no_odds`.

## Where your files live

| Folder | What is in it | Safe to delete? |
|---|---|---|
| `data/` | Downloaded game and player data. | Yes. It downloads again on the next start. |
| `lines/` | Lines you typed in and the log of the app's picks. | **No.** This is your record; it cannot be rebuilt. |
| `reports/` | Results of the tests on past seasons. | Yes. The commands above rebuild them. |

`data/` is not stored in the repository, so a fresh download of the project starts with an empty one.

## How it works

1. **Data.** Game results, sportsbook lines, player stats, injury reports and snap counts come from
   [nflverse](https://github.com/nflverse), a free public source. Files are downloaded once and refreshed as games finish.
2. **Inputs.** For every game the app builds a picture of each team and player using only what was known before
   kickoff: team strength, recent form, the starting quarterback, injuries, rest, weather and the sportsbook's own line.
3. **Models.**
   - Games: a win-probability model, a score-margin model and a total-points model.
   - Player props: a projection with a likely range for yards, a count model for receptions and passing
     touchdowns, and a chance-to-score model for touchdowns. Each averages a linear model with a boosted-tree
     model, and uses recent form at two speeds, the player's last game, teammates ruled out, his own injury
     tag and the starting quarterback.
4. **Comparing with the sportsbook.** Every bet is reduced to the same numbers: the model's win chance, the win
   rate needed to break even, and the gap between them (the edge).
5. **Staying humble.** On past seasons most of the model's disagreement with the sportsbook did not hold up. So
   the win chances shown are pulled toward the sportsbook's, by an amount measured for each market.
6. **Testing without hindsight.** Every result in the Accuracy tab and in `reports/` comes from a model that had
   only seen earlier seasons. A dedicated test fails if any input uses information from the game it is predicting.

## How accurate is it

Tested on seasons the models had not seen (2015 to 2026).

| Area | Result |
|---|---|
| Picking winners | About as accurate as the sportsbook's own odds, not better. |
| Spreads | Model picks win about 50%. Break-even at `-110` is 52.4%. |
| Totals | Model picks win about 52%. Not clearly different from break-even. |
| Home list, top ten | +3.3% per bet since 2015, with a margin of error of about 2 points either way. |
| Home list, best three | +7.4% per bet since 2015, with a margin of error of about 4 points either way. |
| Passing yards | Average miss about 58 yards, against about 60 for a recent-form average. |
| Rushing yards | Average miss about 23.7 yards, against 24.8 for a recent-form average. |
| Receiving yards | Average miss about 24.6 yards, against 25.0 for a recent-form average. |
| Receptions | Average miss about 1.72 catches, against 1.75 for a recent-form average. |
| Anytime touchdown | Chances match what happened closely. |
| Teaser legs | 77.5% of 555 qualifying legs since 2006. Worth betting only at `-120` or better. |

Single seasons swing widely, from about -13% to +15% on the Home list, so a few weeks tell you very little.
The Accuracy tab shows the current numbers.

## Project layout

```
model.py                 the app (start it with: streamlit run model.py)
requirements.txt         packages to install
.streamlit/config.toml   app colors
nfl_model/
  config.py              folders, constants, team names
  cli.py                 the python -m nfl_model commands
  data/                  downloads and data checks
  features/              inputs for the game models (team form, quarterback, injuries)
  models/                win-probability models
  evaluation/            tests on past seasons
  betting/               odds math, edge and stake sizing
  props/                 player prop inputs, models and tests
  ui/                    everything the app shows
    board.py               the week's bets, your lines, teasers
    render.py              page layout and styling
    history.py             the Accuracy tab
    tracker.py             the log of picks and its grading
    lines_store.py         lines you type in
tests/                   automated tests
data/                    downloaded data (created on first run)
lines/                   your saved lines and pick log (created when you use the app)
reports/                 results of tests on past seasons
```

## Running the tests

```
python -m nfl_model test
```

This runs the unit tests and the check that no input uses information from the future. It takes about five
minutes when the data has to be downloaded first.

## Troubleshooting

| Problem | What to do |
|---|---|
| `streamlit` is not recognized | Run `python -m streamlit run model.py`. If that fails, repeat the install step. |
| `python` is not recognized (Windows) | Reinstall Python with "Add Python to PATH" ticked, or use `py` in place of `python`. |
| "running scripts is disabled on this system" (Windows) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. Or skip activation and use `.venv\Scripts\python -m streamlit run model.py`. |
| `No module named ...` | Activate the virtual environment, then run `pip install -r requirements.txt` again. |
| The page shows an error after an update | Stop the app with `Ctrl+C` and start it again. A browser refresh can keep old code loaded. |
| The first load seems stuck | The first run downloads data and can take two minutes. Watch the terminal for progress or errors. |
| A download fails | Check your internet connection. Some work networks block GitHub downloads. |
| "No upcoming games have posted lines yet" | Lines for the next week usually appear early in the week. Try again later. |
| Numbers look out of date | Delete the `data/` folder and start the app again. |
| Team logos are missing | They load from the web. Check your connection or ad blocker. |
| Port 8501 is already in use | Run `streamlit run model.py --server.port 8502`. |

## Known limits

- **No proven edge.** Nothing in the app has been shown to beat sportsbooks to the usual statistical standard.
- **One source of lines.** Listed lines come from one free source, not from your sportsbook. Enter your own to compare.
- **No prop prices.** There is no free history of prop lines, so props are tested on outcomes, not on profit.
- **Who is playing.** Player lists come from recent games. Check injury reports and inactives yourself.
- **Regular season only.** Playoff games are not covered.
- **No live odds feed.** Code for an optional feed exists in `nfl_model/betting/live_odds.py`, but it needs an
  account key and has not been run against the real service.

## Data and credits

- Game, player, injury and snap data: [nflverse](https://github.com/nflverse/nflverse-data).
- Team logos load from ESPN's public image server.
- Built with [Streamlit](https://streamlit.io/), [pandas](https://pandas.pydata.org/) and
  [scikit-learn](https://scikit-learn.org/).
