"""canary-trades: every real canary trade, every form, with the pool path after our buy (11-10)."""
import json

from tradingbot import canary_trades as ct
from tradingbot import launch_live as ll
from tradingbot import model_book as mb
from tradingbot.data import pumpswap as ps
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

T0 = 1_780_000_000.0


def _tx(t, sol, tok, pool, mint):
    bal = [{"owner": pool, "mint": ps.WSOL, "uiTokenAmount": {"uiAmount": sol}},
           {"owner": pool, "mint": mint, "uiTokenAmount": {"amount": str(tok)}}]
    return {"blockTime": t, "meta": {"err": None, "preTokenBalances": bal, "postTokenBalances": bal}}


def _setup():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    # (book, price function of seconds after creation, pulled second, net result)
    spec = {"a": ("canary_rw50_sz@2026-11-10-canary-live-2", lambda s: 1.0 if s < 60 else 0.0, 60, -1.0),
            "b": ("canary_rw50_sz", lambda s: 1.0 + 0.001 * (s - 10), None, 0.02),
            "c": ("canary_launch4_2", lambda s: 1.0, None, -0.05)}
    for i, (k, (book, fn, pulled, ret)) in enumerate(spec.items()):
        c = T0 + i * 2000
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, status, pulled_ts, liq_usd, "
                   "sol_at_decide, ret) VALUES(?,?,?,?,?,?,?,?)",
                   (f"P{k}", f"M{k}", c, "done", c + pulled if pulled else None, 10_000.0 if k != "c" else 30_000.0,
                    50.0, ret))
        j.execute("INSERT INTO positions(token, symbol, cost_usd, opened_at, closed_at, realized_usd, "
                        "exit_reason, strategy) VALUES(?,?,?,?,?,?,?,?)",
                        (f"M{k}", k.upper(), 2.5, c + 9, c + 360, 2.5 * (1 + ret), "time_stop", book))
        pid = j.query("SELECT MAX(id) i FROM positions")[0]["i"]
        j.execute("INSERT INTO fills(ts, position_id, side, qty, price, usd, fee_usd) VALUES(?,?,?,?,?,?,?)",
                  (c + 10, pid, "buy", 100.0, 0.025, -2.5, 0.2))
        j.execute("INSERT INTO fills(ts, position_id, side, qty, price, usd, fee_usd) VALUES(?,?,?,?,?,?,?)",
                  (c + 360, pid, "sell", 100.0, 0.025 * (1 + ret), 2.5 * (1 + ret), 0.05))
        st.execute(ct.SCHEMA)
        rows = [[c + s, 100 * fn(s), 1_000_000] for s in range(5, 900, 5)]
        st.execute("INSERT INTO watch_canary_path(pool, data, ts) VALUES(?,?,?)",
                   (f"P{k}", json.dumps({"rows": rows, "complete": True, "hi": c + 910}), T0))
    j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) "
              "VALUES(?,?,?,?,?,?)", ("Mb", 25.0, T0 + 2008, T0 + 2360, 26.0, mb.CANARY_TWIN))
    return st, j


def test_every_canary_form_is_exported_with_its_pool_path():
    st, j = _setup()
    out = ct.study(st, j, now=T0 + 10_000)
    assert len(out["trades"]) == 3 and len(out["books"]) == 3        # archived reset and old canary books included
    by = {t["symbol"]: t for t in out["trades"]}
    a, b = by["A"], by["B"]
    assert a["pulled_after_fill_s"] == 50 and a["curve"]["pct_at"][60] == -100.0   # pulled 50 s after our fill
    assert a["curve"]["pct_at"][30] > -1                                             # flat before the pull
    assert b["curve"]["pct_at"][300] > 25 and abs(b["twin"]["ret"] - 0.04) < 1e-9               # paper twin on the same token
    assert b["fill_s"] == 10 and b["thin"] and not by["C"]["thin"]
    s = out["summary"]["all"]
    assert s["with_path"] == 3 and s["sell at offset (pool return before fees, % of our buy)"][60]["n"] == 3
    assert "thin (< $15k)" in out["summary"] and "size $2.5" in out["summary"]
    text = ct.render(out)
    assert "CANARY TRADES" in text and "canary_launch4_2" in text


def test_fetch_reads_each_pool_once_within_the_cap():
    st, j = _setup()
    st.execute("DELETE FROM watch_canary_path")

    class FakeHelius:
        credits_used = 0

        def _rpc(self, method, params, credits):
            self.credits_used += credits
            pool, lo = params[0], params[1]["filters"]["blockTime"]["gte"]
            return {"data": [_tx(lo + s, 50, 100, pool, "M" + pool[1:]) for s in range(0, 600, 30)]}

    h = FakeHelius()
    got = ct.fetch_paths(st, h, ct.trades(j, st, T0 + 10_000), max_credits=115)
    assert got["fetched"] == 2 and got["stopped_by_cap"]
    assert ct.fetch_paths(st, h, ct.trades(j, st, T0 + 10_000), max_credits=115)["fetched"] == 1
    assert len(ct.load_paths(st)) == 3


def test_cli_writes_the_export(tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    dest = tmp_path / "out.json"
    cli.main(["canary-trades", str(dest)])
    assert "CANARY TRADES" in capsys.readouterr().out and json.loads(dest.read_text())["trades"] == []
