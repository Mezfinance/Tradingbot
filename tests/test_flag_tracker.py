"""Flag tracker: per-book results in each flag's action window, pass bars, what-if responses (11-09)."""
import json
import random

from tradingbot import flag_tracker as ft
from tradingbot import market_monitor as mm
from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

H = 3600.0
T0 = 1_780_000_000.0 // H * H
SZ = "launchrw50_sz"


def _flag(st, metric, opened, closed, direction="up"):
    st.execute("INSERT INTO market_flags(metric, direction, opened_hour, closed_hour, peak_z, data) VALUES(?,?,?,?,?,?)",
               (metric, direction, opened, closed, 2.5, json.dumps({"seen_hour": opened + H})))


def _trade(j, ts, ret, book=SZ, cost=100.0):
    j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) VALUES(?,?,?,?,?,?)",
              (f"T{ts}", cost, ts, ts + 360, cost * (1 + ret), book))


def _setup(days=10, rug_eps=9, thin_eps=3, seed=1):
    st, j = WatchStore(":memory:"), Journal(":memory:")
    st.execute(mm.SCHEMA)
    st.execute(mm.FLAG_SCHEMA)
    rug = []
    for d in range(rug_eps):
        o = T0 + d * 86400 + 6 * H
        _flag(st, "rug_pct", o, o + 8 * H)
        rug.append(ft.windows({"opened_hour": o, "closed_hour": o + 8 * H, "seen_hour": o + H}, 0))
    for d in range(thin_eps):
        o = T0 + d * 86400 + 18 * H
        _flag(st, "thin_pct", o, o + 3 * H)
    rng = random.Random(seed)
    t = T0
    while t < T0 + days * 86400:
        bad = any(a <= t < b for a, b in rug)
        _trade(j, t, -0.5 if bad else rng.uniform(-0.2, 0.4))
        t += 600
    return st, j


def test_window_starts_when_the_flag_is_confirmed_and_stored():
    a, b = ft.windows({"opened_hour": T0, "closed_hour": T0 + 5 * H, "seen_hour": T0 + H}, now=0)
    assert a == T0 + 2 * H + mm.HOUR_DELAY_S          # the streak's 2nd hour finished and was stored
    assert b == T0 + 6 * H + mm.HOUR_DELAY_S          # the close is known the same way
    a2, b2 = ft.windows({"opened_hour": T0, "closed_hour": None}, now=T0 + 10 * H)
    assert a2 == T0 + 2 * H + mm.HOUR_DELAY_S and b2 == T0 + 10 * H   # still open: up to now


def test_bad_period_rule_passes_and_a_short_history_is_only_watching():
    st, j = _setup()
    out = ft.study(st, j, Config(), now=T0 + 10 * 86400, book_list=[SZ])
    rules = {(r["metric"], r["direction"]): r for r in out["books"][SZ]["rules"]}
    rug, thin = rules[("rug_pct", "up")], rules[("thin_pct", "up")]
    assert rug["usable"] == 9 and rug["passing"] and rug["verdict"] == "signal: bad periods"
    assert rug["gap"] < -40 and rug["ci"][1] < 0 and rug["halves"][0] < 0 and rug["halves"][1] < 0
    assert rug["whatif_usd"]["skip"] > 0 and rug["whatif_usd"]["half size"] == round(rug["whatif_usd"]["skip"] / 2)
    assert rug["whatif_usd"]["double"] < 0
    assert thin["verdict"].startswith("watching (3/") and not thin["passing"]
    assert out["passing"] == [p for p in out["passing"] if p["metric"] == "rug_pct"] and len(out["passing"]) == 1
    text = ft.render(out, SZ)
    assert "PASSING RULES" in text and "signal: bad periods" in text and "watching" in text


def test_trades_before_the_flag_is_confirmed_do_not_count():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    st.execute(mm.FLAG_SCHEMA)
    _flag(st, "rug_pct", T0, T0 + 4 * H)
    for k in range(12):                                 # the two hot hours before confirmation: disasters
        _trade(j, T0 + k * 600, -1.0)
    for k in range(12):                                 # after confirmation: fine
        _trade(j, T0 + 2 * H + mm.HOUR_DELAY_S + k * 600, 0.1)
    for k in range(12):                                 # long after: fine
        _trade(j, T0 + 30 * H + k * 600, 0.1)
    out = ft.study(st, j, Config(), now=T0 + 40 * H, book_list=[SZ])
    r = out["books"][SZ]["rules"][0]
    assert r["trades_in"] == 12 and abs(r["in_pct"] - 10.0) < 1e-6
    assert r["base_pct"] < 0                             # the pre-confirmation losses sit outside the window


def test_monitor_keeps_the_summary_for_the_dashboard():
    st, j = _setup(days=3, rug_eps=2, thin_eps=1)
    cfg = Config(launch_live=True, launch_sized_book=True)
    ft.update(st, j, cfg, now=T0 + 3 * 86400)
    got = ft.read(st.query)
    assert SZ in got["books"] and "episodes_detail" not in got["books"][SZ]["rules"][0]
    page = mm.payload([], now=T0 + 3 * 86400, tracker=got)
    assert page["tracker"]["books"][SZ]["trades"] > 0
    assert ft.read(lambda sql, p: (_ for _ in ()).throw(__import__("sqlite3").OperationalError("x"))) == {}


def test_flag_tracker_runs_through_the_cli(cfg, tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    cli.main(["flag-tracker"])
    assert "FLAG TRACKER" in capsys.readouterr().out
