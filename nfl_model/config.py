"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"          # cached downloads (git-ignored)
REPORT_DIR = ROOT / "reports"     # backtest outputs

TRAIN_START = 2006     # first season used for training (Elo/EWMA warm up from 1999)
FIRST_TEST = 2015      # first walk-forward test season
INJURY_TRAIN_START = 2014   # snap-count-based injury features exist from 2013 (+1 season of history)
ABLATION_FIRST_TEST = 2017  # 3 seasons of injury-feature training data before the first test season

# Full team names (as used by odds APIs) -> nflverse team codes
TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

def logo_url(code: str) -> str:
    code = {"LA": "lar", "WAS": "wsh"}.get(code, code.lower())
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"
