import numpy as np, pandas as pd
from nfl_model.evaluation import lines_backtest as L
from nfl_model.evaluation import early_lines as E
from nfl_model.data import line_history as H


def _g(**kw):
    base = dict(pred=[0, 0, 0], spread_line=[3.0, 3.0, 3.0], home_score=[24, 20, 10],
                away_score=[17, 17, 17], home_spread_odds=[-110, -110, np.nan],
                away_spread_odds=[-110, -110, 900])
    base.update(kw)
    return pd.DataFrame(base)


def test_spread_settle_push_loss_and_odds_fallback():
    d = _g(pred=[6.0, 6.0, -6.0])          # bet home, home, away
    b = L.settle_lines(d, "spread", "spread_line", 1)
    # g1: margin 7 vs 3 -> home covers, win at -110
    assert abs(b.profit.iloc[0] - (1 / 1.1)) < 1e-9
    # g2: margin 3 vs 3 -> push
    assert b.profit.iloc[1] == 0
    # g3: bet away, margin -7 -> away covers; odds 900 invalid -> -110 fallback
    assert abs(b.profit.iloc[2] - (1 / 1.1)) < 1e-9


def test_threshold_filters_bets():
    d = _g(pred=[3.5, 3.0, 0.0])
    assert len(L.settle_lines(d, "spread", "spread_line", 1)) == 1   # only the -3 edge
    assert len(L.settle_lines(d, "spread", "spread_line", 0)) == 2


def test_total_settle_over_under():
    d = pd.DataFrame(dict(pred=[50, 40], total_line=[45.0, 45.0], home_score=[30, 20],
                          away_score=[20, 20], over_odds=[-110, -110], under_odds=[-110, -110]))
    b = L.settle_lines(d, "total", "total_line", 1)
    assert (b.profit > 0).all() and len(b) == 2   # over wins (50>45), under wins (40<45)


def test_walk_forward_never_trains_on_test_season(monkeypatch):
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(dict(season=np.repeat(np.arange(2006, 2018), 50), x=rng.normal(size=n),
                           roof="outdoors", wind=5.0, temp=60.0))
    df["result"] = 10 * df.x + rng.normal(size=n)
    df["home_score"] = 20.0; df["away_score"] = 20.0
    seen = []
    orig = L.ridge
    def spy():
        m = orig(); f = m.fit
        def fit(X, y, *a, **k):
            seen.append(len(X)); return f(X, y, *a, **k)
        m.fit = fit; return m
    monkeypatch.setattr(L, "ridge", spy)
    out = L.walk_forward(df, "result", ["x"], first_test=2015)
    assert sorted(out.season.unique()) == [2015, 2016, 2017]
    assert seen == [450, 500, 550]          # seasons 2006..s-1 only


def test_early_snapshot_lead_window_and_first(tmp_path):
    k = pd.Timestamp("2024-09-08 17:00", tz="UTC")
    rows = [("a", k - pd.Timedelta(days=d), 3.0 + i) for i, d in enumerate([10, 7, 6, 2])]
    h = pd.DataFrame([dict(game_id=g, snap_ts=t, kickoff=k, spread_line=s) for g, t, s in rows])
    p = tmp_path / "h.csv"; h.to_csv(p, index=False)
    s = E.early_snapshot(p, 5, 8)
    assert len(s) == 1 and s.spread_line.iloc[0] == 4.0   # earliest within 5-8d, not 10d


def test_history_grid_days_and_kickoff_conversion():
    assert all(t.weekday() in (0, 2, 4, 6) for t in H.grid(2023, 2023))
    d = pd.DataFrame(dict(gameday=["2024-09-08"], gametime=["13:00"]))
    assert H.kickoff_utc(d).iloc[0] == pd.Timestamp("2024-09-08 17:00", tz="UTC")


if __name__ == "__main__":
    import inspect, pathlib, sys, tempfile

    class _MP:
        def setattr(self, o, n, v):
            setattr(o, n, v)

    bad = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            params = inspect.signature(fn).parameters
            try:
                fn(*([_MP()] if "monkeypatch" in params else [pathlib.Path(tempfile.mkdtemp())] if "tmp_path" in params else []))
                print("ok  ", name)
            except Exception as e:  # noqa
                bad += 1; print("FAIL", name, repr(e))
    sys.exit(bad)
