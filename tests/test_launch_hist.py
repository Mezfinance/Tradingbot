"""Chain-only history of hand-made pools: listing candidates, the creation-time window, parsing, the test."""
import base64
import json
import pytest
from tradingbot import launch_hist as lh
from tradingbot.config import Config
from tradingbot.data import pumpswap as ps
from tradingbot.watcher import WatchStore
from tests.test_launch_live import b58decode

NOW = 2_000_000_000.0
TOKEN = "6RnYgrbQL7HBDyB16VW9zvi1hTrTR9vowxsf29eA6Ks2"
LP = "3VrG6oUA8STG9gHUXVafni6V729q6suVR93pt1Xd2ygn"


def slice_bytes(base, quote, lp):
    return b58decode(base) + b58decode(quote) + b58decode(lp)


def tx(t, payer, tok_before, tok_after, sol_before, sol_after, pool):
    bal = lambda mint, amt: {"owner": pool, "mint": mint, "uiTokenAmount": {"uiAmount": amt}}
    return {"blockTime": t, "slot": int(t), "meta": {"err": None,
            "preTokenBalances": [bal(TOKEN, tok_before), bal(ps.WSOL, sol_before)],
            "postTokenBalances": [bal(TOKEN, tok_after), bal(ps.WSOL, sol_after)]},
            "transaction": {"message": {"accountKeys": [{"pubkey": payer}]}}}


def pool_history(pool, t0, up):
    """Creation (seed 1000 tokens / 50 SOL), buyers in seconds 1-3, then the price ends up x `up` at minute 6."""
    txs = [tx(t0, "OP", 0, 1000, 0, 50, pool)]
    tok, sol = 1000.0, 50.0
    for j in range(3):
        txs.append(tx(t0 + 1 + j, f"W{j}", tok, tok - 10, sol, sol + 0.5, pool))
        tok, sol = tok - 10, sol + 0.5
    # last trade before minute 6 sets the price sol/tok; keep tokens fixed and scale SOL
    txs.append(tx(t0 + 300, "W9", tok, tok, sol, sol * up, pool))
    txs.append(tx(t0 + 420, "W10", tok, tok, sol * up, sol * up, pool))
    return txs


class FakeHelius:
    enabled, api_key = True, "k"

    def __init__(self, pools):
        self.pools, self.credits_used, self.history_calls = pools, 0, 0

    def slot(self):
        return 400_000_000

    def program_accounts_changed(self, program, filters, data_slice, since_slot, **kw):
        assert {"memcmp": {"offset": 211, "bytes": ps.NO_KEY}} in filters and data_slice == {"offset": 43, "length": 96}
        self.credits_used += 10
        return [(p, slice_bytes(ps.WSOL, TOKEN, LP)) for p in self.pools] + \
            [("NOTSOL", slice_bytes(TOKEN, TOKEN, LP))]

    def first_time(self, address):
        self.credits_used += 1
        return self.t_for(address)

    def history_from_start(self, address, span_s, max_pages=6):
        self.history_calls += 1
        self.credits_used += 10
        t0, up = self.pools[address]
        return pool_history(address, t0, up), True, True


def make(pools):
    hel = FakeHelius(pools)
    order = iter([pools[p][0] for p in pools])
    hel.t_for = lambda lp: next(order)
    return hel


def test_backfill_keeps_pools_created_in_the_window():
    st = WatchStore(":memory:")
    pools = {"P1": (NOW - 3600, 2.0), "OLD": (NOW - 10 * 86400, 1.0), "P2": (NOW - 7200, 0.5)}
    out = lh.backfill(st, make(pools), 3, 10_000, now=NOW)
    assert out["candidates"] == 3 and out["fetched"] == 2 and out["outside window"] == 1
    rows = {r["pool"]: r["status"] for r in st.query("SELECT * FROM watch_launch_hist")}
    assert rows == {"P1": "ok", "P2": "ok", "OLD": "outside window"}
    again = make(pools)
    assert lh.backfill(st, again, 3, 10_000, now=NOW)["already fetched"] == 3 and again.history_calls == 0


def test_backfill_stops_at_the_credit_cap():
    st = WatchStore(":memory:")
    pools = {f"P{i}": (NOW - 3600 - i, 1.0) for i in range(5)}
    out = lh.backfill(st, make(pools), 3, 10 + 11 + (10 * lh.MAX_PAGES + 1), now=NOW)  # listing + 1 pool + worst case
    assert out["fetched"] <= 2 and out["stopped: credit cap"] >= 3


def test_rows_use_chain_prices_and_the_live_features():
    st = WatchStore(":memory:")
    lh.backfill(st, make({"UP": (NOW - 3600, 2.0), "PULL": (NOW - 3700, 0.05)}), 3, 10_000, now=NOW)
    st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts) VALUES('UP', 'M', 'pumpswap', 'X', ?, ?)",
               (NOW - 3600, NOW - 3500))
    rows, dropped = lh.rows(st, Config())
    r = {x["pool"]: x for x in rows}
    # entry at the second-6 chain price = after the third buy (51.5 SOL / 970 tokens); exit at minute 6 = x2
    assert r["UP"]["ret"][6] == pytest.approx(2.0 - 1 - 0.06) and r["UP"]["listed"] and r["UP"]["liquid"]
    assert r["PULL"]["ret"][6] == -1.0 and not r["PULL"]["listed"]  # SOL fell below 10% of its peak
    assert "log_buyers" in r["UP"]["feat"][4] and r["UP"]["feat"][4].get("name_len", 0) == 0


def test_test_reports_listed_and_unlisted():
    st = WatchStore(":memory:")
    pools = {f"P{i}": (NOW - 3 * 86400 + 3600 + i * 600, 2.0 if i % 3 == 0 else 0.9) for i in range(300)}
    lh.backfill(st, make(pools), 4, 100_000, now=NOW)
    out = lh.test(st, Config(db_path=":memory:"))
    assert out["liquid at second 4"] == 300 and out["every liquid pool"]["never listed"]["n"] == 300
    wf = out["walk-forward on these rows (train on earlier days, no names)"]
    assert "top 2%" in wf and "by day" in wf
