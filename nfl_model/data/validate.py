"""Data sanity checks. `python -m nfl_model check` prints them; returns (level, message) rows."""
from __future__ import annotations
import pandas as pd
from ..config import DATA_DIR


def check(df: pd.DataFrame, features: list[str] | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    add = lambda lvl, msg: out.append((lvl, msg))
    if df.game_id.duplicated().any():
        add("ERROR", f"{df.game_id.duplicated().sum()} duplicate game_id rows")
    reg = df[df.game_type == "REG"]
    for s, g in reg.groupby("season"):
        exp = 272 if s >= 2021 else 256 if s >= 2002 else None
        if s == 2022:
            exp = 271   # BUF-CIN cancelled after Hamlin collapse (real, not a data error)
        n = len(g)
        if exp and s < df.season.max() and n != exp:
            add("ERROR", f"{s}: {n} regular-season games, expected {exp}")
    last = reg[reg.home_score.notna()]
    if not last.empty:
        age = (pd.Timestamp.now() - pd.to_datetime(last.gameday).max()).days
        if age > 10 and last.season.max() == df.season.max():
            add("WARN", f"newest completed game is {age} days old - cache may be stale")
    recent = reg[(reg.season >= 2015) & reg.home_score.notna()]   # completed games only
    for c in ("spread_line", "total_line", "home_moneyline"):
        miss = recent[c].isna().mean()
        if miss > 0.02:
            add("WARN", f"{c} missing for {miss:.1%} of 2015+ games")
    bad = recent[(recent.spread_line.abs() > 30) | (recent.total_line.fillna(45).between(25, 70) == False)]
    if len(bad):
        add("ERROR", f"{len(bad)} games with implausible spread/total lines")
    for c in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds"):
        odd = recent[c].dropna()
        frac = ((odd.abs() < 100) | (odd.abs() > 400)).mean()
        if frac > 0.01:
            add("WARN", f"{c}: {frac:.1%} of values are non-standard American odds (treated as -110)")
    for f in features or []:
        if f not in df:
            add("ERROR", f"feature {f} missing from table"); continue
        by = reg[reg.season >= 2015].groupby("season")[f].apply(lambda x: x.isna().mean())
        for s, m in by.items():
            if m > 0.9:
                add("WARN", f"feature {f} is {m:.0%} blank in {s}")
    hp = DATA_DIR / "line_history.csv"
    if hp.exists():
        h = pd.read_csv(hp, usecols=["snap_ts"], parse_dates=["snap_ts"])
        add("INFO", f"line history: {len(h)} rows through {h.snap_ts.max():%Y-%m-%d}")
    else:
        add("INFO", "line history not built (optional; early-line analysis only)")
    if not out or all(l == "INFO" for l, _ in out):
        add("OK", "no data problems found")
    return out
