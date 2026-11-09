"""sz-stops: drawdown limits and per-trade stop losses replayed on a book's real trades (11-09)."""
from tradingbot import launch_live as ll
from tradingbot import sz_stops as ss
from tradingbot.data import pumpswap as ps
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

D = 86400.0
T0 = 1_780_000_000.0 // D * D  # midnight UTC


def _pos(j, token, opened, ret, cost=100.0, hold=360.0):
    j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) VALUES(?,?,?,?,?,?)",
              (token, cost, opened, opened + hold, cost * (1 + ret), ss.BOOK))


def test_day_loss_limit_skips_the_rest_of_the_day_and_only_counts_trades_already_sold():
    j = Journal(":memory:")
    # day 1: three -50% trades on $1,000 sold by 00:35, then ten +10% trades; day 2: ten +10% trades
    for i in range(3):
        _pos(j, f"L{i}", T0 + i * 600, -0.5, cost=1000.0, hold=900)
    for i in range(10):
        _pos(j, f"W{i}", T0 + 3600 + i * 600, 0.10, cost=1000.0)
        _pos(j, f"N{i}", T0 + D + i * 600, 0.10, cost=1000.0)
    tr = ss.trades(j, None, ss.BOOK, 0, T0 + 2 * D)
    rets = [t["ret"] for t in tr]
    m = ss.apply_rule(tr, rets, {"kind": "day", "k": 10}, ss.CAPITAL)   # $1,000 day loss
    taken = {t["token"] for t, x in zip(tr, m) if x}
    assert {"L0", "L1", "L2"} <= taken                    # L1 was bought before L0 was sold
    assert not any(t.startswith("W") for t in taken)      # day 1 stopped after -$1,500
    assert all(f"N{i}" in taken for i in range(10))       # a new day trades again
    half = ss.apply_rule(tr, rets, {"kind": "half", "k": 10}, ss.CAPITAL)
    assert all(x == 0.5 for t, x in zip(tr, half) if t["token"].startswith("W"))


def _tx(t, sol, tok, pool, mint):
    bal = lambda s, k: [{"owner": pool, "mint": ps.WSOL, "uiTokenAmount": {"uiAmount": s}},  # noqa: E731
                        {"owner": pool, "mint": mint, "uiTokenAmount": {"amount": str(k)}}]
    return {"blockTime": t, "meta": {"err": None, "preTokenBalances": bal(sol, tok), "postTokenBalances": bal(sol, tok)}}


def test_a_stop_sells_a_slow_fall_but_not_a_one_transaction_pull():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    st.execute(ss.SCHEMA)
    for name, pulled in (("slow", None), ("pull", T0 + 200)):
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, status, pulled_ts) VALUES(?,?,?,?,?)",
                   (f"P{name}", f"M{name}", T0, "done", pulled))
        _pos(j, f"M{name}", T0 + 10, -1.0 if name == "pull" else -0.66, hold=350)
    # slow: price 1.0 falls 1% every 5 s from second 10 (pool SOL drains, tokens constant)
    slow = ss.path_rows([_tx(T0 + s, 100 * (1 - 0.01 * max(0, (s - 10) / 5)), 100, "Pslow", "Mslow")
                         for s in range(5, 365, 5)], "Pslow", "Mslow")
    # pull: flat at 1.0 until second 200, then empty in one tx
    pull = ss.path_rows([_tx(T0 + s, 100 if s < 200 else 0, 100, "Ppull", "Mpull") for s in range(5, 365, 5)],
                        "Ppull", "Mpull")
    import json
    for pool, rows in (("Pslow", slow), ("Ppull", pull)):
        st.execute("INSERT INTO watch_trade_path(pool, data, ts) VALUES(?,?,?)",
                   (pool, json.dumps({"rows": rows, "complete": True}), T0))
    out = ss.study(st, j, now=T0 + 3600)
    assert out["covered"] == 2
    g = {(x["stop"], x["limit"]): x for x in out["grid"]}
    base, s20 = g[("no stop", "no limit")], g[("stop -20%", "no limit")]
    assert base["usd"] == -166
    # slow fall: sold near -20% pool move instead of -68% at the end -> about +45 pts on $100
    assert -125 < s20["usd"] < -110
    i = out["stop_info"]["stop -20%"]
    assert i["fired"] == 2 and i["fired_rugged"] == 1        # it fired on the pull too, but sold at nothing
    assert "STOP LOSSES PER TRADE" in ss.render(out)


def test_cli_runs_without_data(cfg, tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    cli.main(["sz-stops"])
    assert "STOPS AND DRAWDOWN LIMITS" in capsys.readouterr().out


def test_fetch_reads_each_hold_once_within_the_credit_cap():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    for i in range(3):
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, status) VALUES(?,?,?,?)",
                   (f"P{i}", f"M{i}", T0 + i * 1000, "done"))
        _pos(j, f"M{i}", T0 + i * 1000 + 10, 0.1)

    class FakeHelius:
        credits_used = 0
        calls = []

        def _rpc(self, method, params, credits):
            self.credits_used += credits
            self.calls.append(params)
            pool = params[0]
            lo = params[1]["filters"]["blockTime"]["gte"]
            return {"data": [_tx(lo + s, 50, 100, pool, "M" + pool[1:]) for s in range(0, 360, 30)]}

    h = FakeHelius()
    tr = ss.trades(j, st, ss.BOOK, 0, T0 + 9000)
    got = ss.fetch_paths(st, h, tr, max_credits=95)
    assert got["fetched"] == 2 and got["stopped_by_cap"]            # room for a full 80-credit hold is kept back
    assert ss.fetch_paths(st, h, tr, max_credits=95)["fetched"] == 1   # cached ones are not read again
    assert len(ss.load_paths(st)) == 3
    assert {c[1]["filters"]["blockTime"]["gte"] for c in h.calls} == {int(t["opened"]) - ss.PRE_S for t in tr}
