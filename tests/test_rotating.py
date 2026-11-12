"""Wallet rotation for the launch canary: one open trade per wallet, freshest wallet first, sells route back."""
from __future__ import annotations

import json

import pytest

from tradingbot.execution.base import Fill
from tradingbot.execution.rotating import RotatingExecutor


class FakeSub:
    """A stand-in for one wallet's DirectPumpswapExecutor."""

    route = "direct"

    def __init__(self, name: str):
        self.name = name
        self.last_note = ""
        self.pools: dict[str, str] = {}
        self.bought: list[str] = []
        self.sold: list[str] = []
        self.balances: dict[str, int] = {}
        self.fail_buy = False
        self.fail_sell = False  # True = the pool rugged, the sale finds no route
        self.feinted: list = []
        self.feint_verdict = "alive"
        self.last_feint: dict = {}

    def feint(self, token: str, usd: float, commit: bool = True) -> str:
        self.feinted.append((token, commit))
        return self.feint_verdict

    def remember_pool(self, mint: str, pool: str) -> None:
        self.pools[mint] = pool

    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        if self.fail_buy:
            self.last_note = "boom"
            return Fill(qty=0.0, price=0.0, usd=0.0, fee_usd=0.0, ok=False, note="boom")
        self.bought.append(token)
        self.balances[token] = 1000
        return Fill(qty=1000.0, price=ref_price or 1e-6, usd=usd, fee_usd=0.0, tx="buysig", ok=True)

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        self.sold.append(token)
        if self.fail_sell or self.balances.get(token, 0) <= 0:
            return Fill(qty=0.0, price=0.0, usd=0.0, fee_usd=0.0, ok=False, note="no route")
        self.balances[token] = 0
        return Fill(qty=qty, price=ref_price or 1e-6, usd=5.0, fee_usd=0.0, tx="sellsig", ok=True)

    def token_balance(self, mint: str) -> int:
        return self.balances.get(mint, 0)

    def sweep(self, keep=frozenset(), apply: bool = True) -> dict:
        return {"closed": 1, "usd_back": 0.5, "held": [self.name]}


def rot(n=3, **kw):
    return RotatingExecutor([FakeSub(f"w{i}") for i in range(n)], **kw)


def test_each_trade_gets_a_distinct_wallet():
    r = rot(3)
    for m in ("A", "B", "C"):
        assert r.buy(m, 5.0, 1e-6) is not None
    assert sorted(r._assign.values()) == [0, 1, 2]  # three mints, three different wallets
    assert all(len(s.bought) == 1 for s in r.subs)


def test_one_trade_per_wallet_blocks_when_all_are_busy():
    r = rot(2)
    assert r.buy("A", 5.0, 1e-6) is not None
    assert r.buy("B", 5.0, 1e-6) is not None
    assert r.buy("C", 5.0, 1e-6) is None  # both wallets hold an open trade
    assert "no free wallet (all 2 busy)" in r.last_note


def test_a_successful_sell_frees_the_wallet_for_reuse():
    r = rot(2)
    r.buy("A", 5.0, 1e-6)  # wallet 0
    r.buy("B", 5.0, 1e-6)  # wallet 1
    assert r.buy("C", 5.0, 1e-6) is None  # full
    assert r.sell("A", 1000.0, 1e-6).ok  # frees wallet 0
    assert "A" not in r._assign
    assert r.buy("C", 5.0, 1e-6) is not None  # wallet 0 reused


def test_sell_routes_to_the_wallet_that_holds_the_token():
    r = rot(3)
    r.buy("A", 5.0, 1e-6)  # wallet 0
    r.buy("B", 5.0, 1e-6)  # wallet 1
    r.sell("B", 1000.0, 1e-6)
    assert r.subs[1].sold == ["B"]
    assert r.subs[0].sold == [] and r.subs[2].sold == []


def test_the_freshest_wallet_is_used_first():
    clock = {"t": 100.0}
    r = rot(3, clock=lambda: clock["t"])
    r.buy("A", 5.0, 1e-6)        # wallet 0, used at t=100
    clock["t"] = 200.0
    r.sell("A", 1000.0, 1e-6)    # wallet 0 free again but now "stale"
    clock["t"] = 300.0
    r.buy("B", 5.0, 1e-6)        # wallets 0 (used), 1 and 2 (never used) free -> picks a never-used one
    assert r._assign["B"] in (1, 2)
    assert r.subs[0].bought == ["A"]  # wallet 0 not reused while fresh wallets remain


def test_a_failed_buy_releases_the_wallet():
    r = rot(2)
    r.subs[0].fail_buy = True
    fill = r.buy("A", 5.0, 1e-6)  # tries wallet 0 (freshest), fails
    assert fill is not None and not fill.ok
    assert "A" not in r._assign  # wallet released, nothing stuck
    assert "boom" in r.last_note
    r.subs[0].fail_buy = False
    assert r.buy("B", 5.0, 1e-6) is not None  # both wallets still available


def test_a_stuck_rug_is_reclaimed_after_the_timeout():
    clock = {"t": 0.0}
    r = rot(2, reclaim_s=600.0, clock=lambda: clock["t"])
    for m in ("A", "B"):  # both wallets rug: sell finds no route, so they stay assigned
        r.buy(m, 5.0, 1e-6)
        r.subs[r._assign[m]].fail_sell = True
    assert not r.sell("A", 1000.0, 1e-6).ok  # stuck, wallet still busy
    assert r.buy("C", 5.0, 1e-6) is None      # all busy, nothing reclaimable yet
    clock["t"] = 700.0                         # past reclaim_s
    assert r.buy("C", 5.0, 1e-6) is not None   # a stuck wallet is reclaimed


def test_assignment_survives_a_restart(tmp_path):
    state = str(tmp_path / "canary_rotation.json")
    r = RotatingExecutor([FakeSub(f"w{i}") for i in range(3)], state_path=state)
    r.remember_pool("A", "poolA")
    r.buy("A", 5.0, 1e-6)  # wallet 0, persisted
    assert json.load(open(state))["assign"]["A"] == 0
    # a fresh process: the tokens live in wallet 0, so a reloaded rotator sells from wallet 0
    subs = [FakeSub(f"w{i}") for i in range(3)]
    subs[0].balances["A"] = 1000
    r2 = RotatingExecutor(subs, state_path=state)
    assert r2._assign.get("A") == 0
    assert r2.subs[0].pools.get("A") == "poolA"  # pool re-applied so the sell can route
    assert r2.sell("A", 1000.0, 1e-6).ok
    assert r2.subs[0].sold == ["A"]


def test_sweep_merges_every_wallet():
    r = rot(3)
    res = r.sweep()
    assert res["closed"] == 3  # one per wallet
    assert res["usd_back"] == pytest.approx(1.5)
    assert set(res["held"]) == {"w0", "w1", "w2"}


def test_read_canary_wallets_parses_and_caps(tmp_path, monkeypatch):
    from solders.keypair import Keypair

    from tradingbot.config import Config
    from tradingbot.execution.rotating import MAX_WALLETS, read_canary_wallets

    keys = [Keypair() for _ in range(12)]
    arr = [list(bytes(k)) for k in keys]
    # a JSON array of solana-keygen int arrays, with one duplicate, more than MAX_WALLETS
    monkeypatch.setenv("TB_CANARY_WALLETS", json.dumps(arr + [arr[0]]))
    cfg = Config(launch_canary_wallets_path="")
    out = read_canary_wallets(cfg)
    assert len(out) == MAX_WALLETS  # capped at 10
    assert len({str(k.pubkey()) for k in out}) == MAX_WALLETS  # deduped
    monkeypatch.delenv("TB_CANARY_WALLETS")
    assert read_canary_wallets(cfg) == []  # nothing configured -> empty


def test_feint_keeps_the_wallet_for_a_following_commit():
    r = rot(2)
    assert r.feint("A", 25.0, commit=True) == "alive"  # commit coming: hold the wallet for buy()
    assert list(r._assign) == ["A"]
    assert r.subs[r._assign["A"]].feinted == [("A", True)]


def test_feint_only_frees_the_wallet_each_time():
    r = rot(2)
    assert r.feint("A", 25.0, commit=False) == "alive"  # feint-only: no commit, so release the wallet
    assert r._assign == {}
    assert r.feint("B", 25.0, commit=False) == "alive"  # a second pool still finds a free wallet
    assert r._assign == {}


def test_feint_only_rotates_across_the_fresh_wallets():
    clock = {"t": 100.0}
    r = rot(3, clock=lambda: clock["t"])
    for m in ("A", "B", "C"):  # each freed after its feint, but marked used so the next picks a different wallet
        clock["t"] += 1
        r.feint(m, 25.0, commit=False)
    assert sorted(len(s.feinted) for s in r.subs) == [1, 1, 1]  # one feint each, not three on one wallet
    clock["t"] += 1
    r.feint("D", 25.0, commit=False)  # a fourth wraps back to the least-recently-used wallet
    assert sorted(len(s.feinted) for s in r.subs) == [1, 1, 2]
