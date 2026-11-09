"""missed-signals: a launch book's signals split into bought and missed, missed grouped by skip reason (11-09)."""
from tradingbot import launch_live as ll
from tradingbot import missed_signals as ms
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

T0 = 1_780_000_000.0


def _setup():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    # (rw_pct, ret, liq, outcome): rw_pct < 50 = an rw50 signal for launchrw50_sz
    pools = [(10, 0.20, 10_000, "buy"), (20, -1.0, 12_000, "buy"),
             (30, 0.80, 11_000, "slippage exceeded (reverted)"), (40, 0.50, 9_000, "slippage exceeded (reverted)"),
             (15, -1.0, 20_000, "no live price"), (25, 0.10, 30_000, None),
             (80, 0.90, 10_000, "skip")]  # rw_pct 80: no signal at all
    for i, (rw, ret, liq, what) in enumerate(pools):
        c = T0 + i * 1_000
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, decided_ts, liquid, status, ret, "
                   "liq_usd, rw_pct) VALUES(?,?,?,?,?,?,?,?,?)", (f"P{i}", f"M{i}", c, c + 5, 1, "done", ret, liq, rw))
        if what == "buy":
            j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) "
                      "VALUES(?,?,?,?,?,?)", (f"M{i}", 200.0, c + 11, c + 370, 200.0 * (1 + ret - 0.05), "launchrw50_sz"))
        elif what and what != "skip":
            reason = "transaction failed" if "reverted" in what else what
            j.execute("INSERT INTO events(ts, kind, token, detail, strategy) VALUES(?,?,?,?,?)",
                      (c + 14, "entry_skipped", f"M{i}", reason, "launchrw50_sz"))
    return st, j


def test_signals_split_into_bought_and_missed_by_reason():
    st, j = _setup()
    out = ms.study(st, j, hours=10, now=T0 + 7_000 + 3_600)
    assert out["signals"] == 6 and out["bought"] == 2 and out["missed"] == 4
    r = out["by_reason"]
    assert r["transaction failed"]["n"] == 2 and r["transaction failed"]["mean_pct"] == 65.0
    assert r["no live price"]["n"] == 1 and r[ms.NO_EVENT]["n"] == 1
    assert r["transaction failed"]["skip_at_pool_second_median"] == 14.0
    assert out["bought_real"]["n"] == 2 and out["real_minus_ideal_pts"]["mean"] == -5.0
    assert out["fill_at_pool_second_median"] == 11.0
    text = ms.render(out)
    assert "WHY MISSED" in text and "transaction failed" in text and "M6" not in text


def test_reason_key_groups_numbers():
    assert ms.reason_key("no real liquidity (live price 16x the snapshot)") == "no real liquidity"
    assert ms.reason_key("round-trip loss 99.4% (honeypot/tax/thin pool)") == "round-trip loss #"
    assert ms.reason_key("executor declined: price impact 93.12%") == "executor declined: price impact #"


def test_missed_signals_runs_through_the_cli(cfg, tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    cli.main(["missed-signals"])
    assert "no scorer table yet" in capsys.readouterr().out
