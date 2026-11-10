"""canary-package: the canary's proposed per-band rules replayed over the year (11-10)."""
from tradingbot import canary_package as cp
from tradingbot import risk_history as rh
from tradingbot import safeguards as sg
from tradingbot.watcher import WatchStore

D = 86400.0
T0 = 1_780_000_000.0 // D * D


def _row(i, t, liq, rw_pct, ret, big=None, **kw):
    return {"pool": f"P{i}", "create_ts": t, "ret": ret, "liq_usd": liq, "rw": 0.1, "rw_pct": rw_pct, "rwb_pct": None,
            "w": 1.0, "pulled_ts": None, "token_mint": f"M{i}", "entry_px": 1.0, "peak_sol": 100.0, "raw": False,
            "src": "hist", "big": big, **kw}


class Cfg:
    model_book_liq_share = 0.02


def _store(rows, m10=None, m15=None):
    st = WatchStore(":memory:")
    sg.save_replay(st, rows, {})
    st.execute(rh.HOLD_PATH_SCHEMA)
    for pool, px in (m10 or {}).items():   # pool state at minute 10: price = sol / tokens
        st.execute("INSERT INTO hist_path(pool, sec, sol, tok_ui, tok_raw, ts) VALUES (?,?,?,?,?,?)",
                   (pool, cp.HOLD10_S, 100.0 * px, 100.0, 100.0, 0.0))
    for pool, px in (m15 or {}).items():   # and at minute 15
        st.execute("INSERT INTO hist_path(pool, sec, sol, tok_ui, tok_raw, ts) VALUES (?,?,?,?,?,?)",
                   (pool, cp.HOLD15_S, 100.0 * px, 100.0, 100.0, 0.0))
    return st


def test_each_band_uses_its_own_rule_size_and_exit():
    rows, m10, m15 = [], {}, {}
    t = T0
    for i in range(80):   # $15-25k: sz skips every pool (rw_pct 80); the +50% model ranks winners high
        good = i % 2 == 0
        rows.append(_row(f"m{i}", t, 20_000.0, 80.0, 0.5 if good else -0.5, big=0.5 + i / 1000 if good else 0.1))
        t += 600
    for i in range(10):   # $25-40k kept half: flat at minute 6, +50% price by minute 10
        rows.append(_row(f"h{i}", t, 30_000.0, 10.0, 0.0)); m10[f"Ph{i}"] = 1.5; m15[f"Ph{i}"] = 2.0; t += 600
    for i in range(10):   # thin and deep sz picks
        rows.append(_row(f"t{i}", t, 12_000.0, 10.0, 0.2)); t += 600
        rows.append(_row(f"d{i}", t, 60_000.0, 10.0, -0.1)); t += 600
    out = cp.study(_store(rows, m10, m15), None, None, Cfg(), now=t + D)
    gap = out["gap"]
    prop, cur = out["packages"]["proposed"], out["packages"]["current"]
    pb, cb = prop["bands"], cur["bands"]
    assert "$15-25k" not in cb                                   # sz's rule buys none of them
    assert pb["$15-25k"]["trades"] == 15                          # best 30% of the 30 pools ranked after 50
    assert abs(pb["$15-25k"]["mean_pct"] - 100 * (0.5 - gap - 0.30 / 10)) < 0.1   # winners only, $10 pays a 3% fee
    assert abs(pb["$25-40k"]["mean_pct"] - 100 * (1.5 - 1 - 0.06 - gap - 0.03)) < 0.1   # held to minute 10
    p15 = out["packages"]["proposed m15"]["bands"]
    assert abs(p15["$25-40k"]["mean_pct"] - 100 * (2.0 - 1 - 0.06 - gap - 0.03)) < 0.1  # held to minute 15
    assert p15["$15-25k"] == pb["$15-25k"] and p15["under $15k"] == pb["under $15k"]
    assert abs(cb["$25-40k"]["mean_pct"] - 100 * (0.0 - gap - 0.03)) < 0.1             # current: sold at minute 6
    assert pb["under $15k"] == cb["under $15k"] and pb["$40k+"] == cb["$40k+"]
    assert abs(pb["under $15k"]["mean_pct"] - 100 * (0.2 - gap - 0.30 / 25)) < 0.1
    assert out["info"]["$15-25k pools without a big rank yet (thin window: not bought)"] == 50
    text = cp.render(out)
    assert "WHOLE PACKAGE" in text and "BY BAND" in text and "BY MONTH" in text


def test_the_twins_drawdown_pauses_the_canary_and_a_missing_minute_10_keeps_minute_6():
    rows = []
    t = T0
    for i in range(5):   # the twin buys $240 of each (2% of $12k) and loses it all: -$1,200 inside 24 h
        rows.append(_row(f"r{i}", t, 12_000.0, 10.0, -1.0)); t += 600
    t += 3600
    for i in range(5):   # bought within the next day: paused
        rows.append(_row(f"a{i}", t, 12_000.0, 10.0, 0.3)); t += 600
    rows.append(_row("h0", t, 30_000.0, 10.0, 0.1, big=0.2))   # $25-40k pick with no minute-10 state
    t += 2 * D
    for i in range(5):   # two days later the window has cleared
        rows.append(_row(f"b{i}", t, 12_000.0, 10.0, 0.3)); t += 600
    out = cp.study(_store(rows), None, None, Cfg(), now=t + D)
    p, q = out["packages"]["proposed"], out["packages"]["proposed + pause"]
    assert p["bands"]["under $15k"]["trades"] == 15 and q["bands"]["under $15k"]["trades"] == 10   # 5 losers + 5 later
    assert "$25-40k" not in q["bands"]
    assert out["info"]["$25-40k minute-10 states missing (minute 6 used)"] == 1
    assert out["info"]["$25-40k minute-15 states missing (minute 6 used)"] == 1
    assert abs(p["bands"]["$25-40k"]["mean_pct"] - 100 * (0.1 - out["gap"] - 0.03)) < 0.1


def test_an_old_replay_without_big_scores_asks_for_a_fresh_one():
    st = _store([_row("x", T0, 12_000.0, 10.0, 0.1)])
    out = cp.study(st, None, None, Cfg(), now=T0 + D)
    assert "--fresh" in out["error"] and "canary-package --fresh" in cp.render(out)


def test_cli_without_history(cfg, tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    cli.main(["canary-package"])
    assert "no hist.db" in capsys.readouterr().out
