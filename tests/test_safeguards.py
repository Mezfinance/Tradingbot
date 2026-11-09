"""safeguards: causal pause / stop rules scored on replayed trades (bad months saved vs good months kept)."""
import numpy as np

from tradingbot import safeguards as sg
from tradingbot.watcher import WatchStore

D = 86400.0
T0 = 1_780_000_000.0   # 2026-05-28


def _rows(days=60, per_day=200, crash=(30, 40), seed=1):
    """Every pool +10% except a crash stretch (-40%, half rugged); rw_pct random."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        for k in range(per_day):
            t = T0 + d * D + k * (D / per_day)
            bad = crash[0] <= d < crash[1]
            ret = (-1.0 if rng.random() < 0.5 else 0.2) if bad else 0.10
            rows.append({"pool": f"P{d}_{k}", "create_ts": t, "ret": ret, "liq_usd": 10_000.0,
                         "rw": 0.0, "rw_pct": float(rng.integers(0, 100)), "rwb_pct": None, "w": 1.0})
    return rows


def _trades(rows):
    return sorted([{"ret": r["ret"], "cost": 100.0, "ts": r["create_ts"] + 360, "opened": r["create_ts"] + 10,
                    "liq": r["liq_usd"], "w": 1.0} for r in rows if r["rw_pct"] < 50], key=sg._decide)


def test_market_pause_is_causal_and_cuts_the_crash():
    rows = _rows()
    mkt = sg.market_window(rows)
    trades = _trades(rows)
    f = sg.apply(trades, mkt, "market", 6, -10)
    day = lambda t: int((t["opened"] - T0) // D)
    assert all(v == 1.0 for t, v in zip(trades, f) if day(t) < 30)           # good days: never paused
    first_hour = [v for t, v in zip(trades, f) if day(t) == 30 and t["opened"] - T0 - 30 * D < 3600]
    assert all(v == 1.0 for v in first_hour)               # the first crash hour: too few finished crash pools yet
    assert np.mean([v for t, v in zip(trades, f) if 31 <= day(t) < 40]) < 0.05   # paused inside the crash
    assert np.mean([v for t, v in zip(trades, f) if day(t) >= 41]) > 0.95    # back on once the window clears
    base = sg.score(trades, np.ones(len(trades)), {})
    r = sg.score(trades, f, base["months"])
    assert r["pnl_usd"] > base["pnl_usd"] and r["max_dd_usd"] > base["max_dd_usd"]


def test_daystop_stops_after_the_days_loss_and_resets_next_day():
    t = T0 + 3600
    trades = [{"ret": -1.0, "cost": 100.0, "ts": t + i * 600 + 360, "opened": t + i * 600 + 10, "w": 1.0}
              for i in range(10)]
    trades += [{"ret": 0.1, "cost": 100.0, "ts": t + D + i * 600 + 360, "opened": t + D + i * 600 + 10, "w": 1.0}
               for i in range(10)]
    f = sg.apply(trades, None, "daystop", 0, 20)   # median day traded $1,000 -> stop past a $200 loss
    assert list(f[:10]) == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]   # 3 sold losers (-$300) before the 4th decision
    assert all(v == 1 for v in f[10:])
    half = sg.apply(trades, None, "daystop", 0, 20, half=True)
    assert list(half[3:10]) == [0.5] * 7


def test_own_performance_pause_and_edge_rule_run():
    rows = _rows()
    mkt = sg.market_window(rows)
    trades = _trades(rows)
    own = sg.apply(trades, mkt, "own", 24, -5)
    day = lambda t: int((t["opened"] - T0) // D)
    assert np.mean([v for t, v in zip(trades, own) if 32 <= day(t) < 40]) < 0.1
    edge = sg.apply(trades, mkt, "edge", 24, 0)    # random rw_pct: no edge, so it pauses about half the time
    assert 0.1 < edge.mean() < 0.95


def test_study_book_scores_every_rule_and_round_trips_the_saved_replay():
    rows = _rows()
    out = sg.study_book("x", _trades(rows), sg.market_window(rows))
    assert out["bad months"] and "no safeguard" in out["rules"] and "day stop 10%" in out["rules"]
    base = out["rules"]["no safeguard"]
    assert base["kept_pct"] == 100 and base["saved_pct"] == 0
    assert out["rules"]["market 6h < -10%"]["saved_pct"] > 50
    st = WatchStore(":memory:")
    sg.save_replay(st, rows, {"fits": 3})
    back, meta = sg.load_replay(st)
    assert len(back) == len(rows) and meta["fits"] == 3 and back[0]["rw_pct"] == rows[0]["rw_pct"]
    assert "SAFEGUARDS" in sg.render({"source": "t", "took_s": 0, "pools": 1, "books": {"x": out}, "note": "n"})


def test_safeguards_command_runs_through_the_cli(cfg, tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    cli.main(["safeguards"])
    assert "no hist.db" in capsys.readouterr().out
