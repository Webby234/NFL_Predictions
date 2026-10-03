"""Extract line snapshots from the git history of nflverse/nfldata data/games.csv.

Requires a blob-less bare clone:
    git clone --bare --filter=blob:none https://github.com/nflverse/nfldata.git
Snapshots: Mon/Wed/Fri/Sun 12:00 UTC (oldest commit at/after, within 1 day).
Output long table: snap_ts, game_id, spread_line, total_line, moneylines, odds.
Only pre-kickoff snapshots are kept (leak safety).
"""
import io, subprocess, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

ODDS = ["spread_line", "total_line", "away_moneyline", "home_moneyline",
        "away_spread_odds", "home_spread_odds", "over_odds", "under_odds"]
KEEP = ["game_id", "season", "week", "gameday", "gametime"] + ODDS


def commits(bare):
    out = subprocess.run(["git", "-C", bare, "log", "--format=%H %cI", "--", "data/games.csv"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    rows = [l.split() for l in out if l.strip()]
    df = pd.DataFrame(rows, columns=["sha", "ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def grid(first=2020, last=2026):
    for s in range(first, last + 1):
        d = datetime(s, 8, 25, 12, tzinfo=timezone.utc)
        end = datetime(s + 1, 2, 15, tzinfo=timezone.utc)
        while d < end:
            if d.weekday() in (0, 2, 4, 6):
                yield pd.Timestamp(d)
            d += timedelta(days=1)


def kickoff_utc(df):
    t = pd.to_datetime(df["gameday"] + " " + df["gametime"].fillna("13:00"), errors="coerce")
    return t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def extract(bare, out_csv, first=2020, last=2026):
    out_csv = Path(out_csv)
    cm = commits(bare)
    done = set()
    parts = []
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        parts.append(old)
        done = set(old["sha"])
    seen = set(done)
    for ts in grid(first, last):
        if ts > pd.Timestamp.now(tz="UTC"):
            break
        c = cm[(cm.ts >= ts) & (cm.ts < ts + pd.Timedelta(days=1))]
        if c.empty:
            continue
        r = c.iloc[0]
        if r.sha in seen:
            continue
        seen.add(r.sha)
        blob = subprocess.run(["git", "-C", bare, "cat-file", "-p", f"{r.sha}:data/games.csv"],
                              capture_output=True, check=True).stdout
        g = pd.read_csv(io.BytesIO(blob), usecols=lambda c: c in KEEP)
        g = g[g["season"].between(first, last)]
        g = g.dropna(subset=["spread_line", "total_line"], how="all")
        g["snap_ts"] = r.ts
        g["sha"] = r.sha
        g["kickoff"] = kickoff_utc(g)
        g = g[g["snap_ts"] < g["kickoff"]]
        parts.append(g)
        print(ts.date(), len(g), flush=True)
        pd.concat(parts).to_csv(out_csv, index=False)
    return pd.concat(parts) if parts else pd.DataFrame()


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
