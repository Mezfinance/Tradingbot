"""Direct pumpswap swaps for the launch canary (11-10): account lists pinned to real mainnet swaps, the transaction the
executor sends, fills read from it, and the engine handing the signal's pool over."""
import base64
import time
from types import SimpleNamespace

import pytest

from tests.test_analyst_engine import FakeDex
from tests.test_launch_canary import FakeJup, FakeRpc
from tradingbot import model_book as mb
from tradingbot.config import Config
from tradingbot.data.pumpswap import WSOL, parse_pool
from tradingbot.engine import Engine
from tradingbot.execution import pumpswap_direct as d
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal

solders = pytest.importorskip("solders")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

T22, TOK = d.TOKEN_2022, d.TOKEN


def pool_bytes(base, quote, base_vault, quote_vault, coin_creator=d.NO_KEY, lp="LPmint1111111111111111111111111111111111111"):
    k = lambda a: bytes(Pubkey.from_string(a))
    raw = bytes(8) + bytes(3) + k(d.SYSTEM) + k(base) + k(quote) + k(lp) + k(base_vault) + k(quote_vault)
    raw += (10**9).to_bytes(8, "little") + k(coin_creator) + bytes(2) + bytes(16)
    return raw + bytes(301 - len(raw))


def global_bytes(protocol, buyback, lp=20, proto=5, creator=5, disable=0):
    k = lambda a: bytes(Pubkey.from_string(a))
    pad = lambda keys, n: b"".join(k(x) for x in keys) + bytes(32 * (n - len(keys)))
    raw = bytes(8) + bytes(32) + lp.to_bytes(8, "little") + proto.to_bytes(8, "little") + bytes([disable])
    raw += pad(protocol, 8) + creator.to_bytes(8, "little") + bytes(32 + 32 + 32 + 1 + 224 + 1) + pad(buyback, 8)
    return raw + bytes(949 - len(raw))


# A real sell of SOL for a hand-made pool's token (tx 4GfbuC6s..., 11-10): 21 IDL accounts + buyback recipient + its ATA.
HANDMADE_SELL = [
    "5nx51XqswLBiyKLchz8gTNhYXS9pbyjmAAERHzejyddu", "6boJKB8D3bv9Lee6sS6dGyTKsb3wucxh19svcD49vu7o", d.GLOBAL_CONFIG,
    WSOL, "4yPNCd9Y1XnH3T1Srkzh8z4YQU4sZbtnptFDaxKU1Z6M", "H6ziReBdgdwXhsYQ1fqmbuLvFD8HGKPDvTfUbaDkKEAV",
    "CmRPyfPn4G25ie155twU3erXLPx135CVdKJAABzKSVfM", "8YYzTCF4NQH6aJPaNutpQZZchZG5KGuyyMmc3TmnUisw",
    "GTdrjB2X4H9hmS2ZxPz5YeQmT2oYo8KzjoW8wAXF3LLh", "FWsW1xNtWscwNmKv6wVsU1iTzRN6wmmk3MjxRP5tT7hz",
    "HdZzHQzQkWk9fbcYpPDpjJbYjSzUbHYPNY9A3C54bE47", TOK, T22, d.SYSTEM, d.ATA_PROGRAM, d.EVENT_AUTHORITY, d.PROGRAM,
    "2KMUCkVA3FWGPcY5x7ArLepeeQqmmqMGbt5oGkFqj7v8", "8N3GDaZ2iwN65oxVatKTLPNooAVUJTbfiVJ1ahyqwjSk", d.FEE_CONFIG,
    d.FEE_PROGRAM, "5YxQFdt3Tr9zJLvkFccqXVUwhdTWJQc1fFg2YPbxvxeD", "ATsupyFqd7dkicTaLdoFNCusNfBxwGgsDSo7zCWGLogb"]
# A real buy_exact_quote_in on a pump.fun coin's pool (tx UQyYMSXt..., 11-10): + volume accounts and PDA('pool-v2').
PUMP_BUY = [
    "3iQnKd3DFtAG1hvT9kW9tG8aWKt3RisNtjZo3bV95FYf", "EfUQzXaPWtAARwo7NvyepuqwCwL2GuyQkmQfAKLeDmzT", d.GLOBAL_CONFIG,
    "DtXzHu5uPgt9qPJ3y2L7b8UaLhWtZc78RRDBzBAspump", WSOL, "Cfw3NHtJc239rEUuk5tMtAfjFzYKsKri3PoctBgggKHw",
    "49kfnV9FsDMCAaWdcRE1etTLKqNf6hYUmV1DtW1A3eCM", "FBnS8iJvA1x3267LR5BRqZ8vAJptywb9DkCnhMNAGmuG",
    "6d5GMps4oMRJTQcifT7Pr9JNb6cv3r2CG4W8vDjD1HKA", "62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV",
    "94qWNrtmfn42h3ZjUZwWvK1MEo9uVmmrBPd2hpNjYDjb", T22, TOK, d.SYSTEM, d.ATA_PROGRAM, d.EVENT_AUTHORITY, d.PROGRAM,
    "B5hCwfrBtsk2bRwsGaAFw634z1em9Rzf7TYpy1yiXs5", "9kCTNpkWmwhb8e78nML1Gp3TMYxNwrSdDoreSfYnN6Nc", d.GLOBAL_VOLUME,
    "48gxyRbyUMvUyDzCACvxHMnFurGJSB4YxwqSo5ubdJsP", d.FEE_CONFIG, d.FEE_PROGRAM,
    "H6Q1erXPHDD5wKRqG4CHQopGPZYcup1ExmWytub1Etsa", "GXPFM2caqTtQYC2cJ5yJRi9VDkpsYZXzYdwYpGnLmtDL",
    "AktftA98kSWAxn6kVSoqBXBELUArjKu2H9WmKB48ULFY"]


def test_derived_addresses_match_the_program():
    assert d.pda([b"__event_authority"], d.PROGRAM) == d.EVENT_AUTHORITY
    assert d.pda([b"global_volume_accumulator"], d.PROGRAM) == d.GLOBAL_VOLUME
    assert d.pda([b"fee_config", d.key(d.PROGRAM)], d.FEE_PROGRAM) == d.FEE_CONFIG


def test_account_lists_match_real_mainnet_swaps():
    r = HANDMADE_SELL
    pool = parse_pool(pool_bytes(r[3], r[4], r[7], r[8]))
    accs = d.swap_accounts("sell", r[0], pool, r[1], TOK, T22, r[9], r[21], pump_coin=False)
    assert [a for a, _, _ in accs] == r
    assert "".join("W" if w else "-" for _, _, w in accs) == "WW---WWWW-W------W----W"  # flags of the real tx
    r = PUMP_BUY
    creator = "8duuapjsz4JE7jfV6YadcsLKJwPgyRJFBxQZZLdr3ikV"  # any non-zero coin creator: only its vault PDA is checked
    pool = parse_pool(pool_bytes(r[3], r[4], r[7], r[8], coin_creator=creator))
    accs = d.swap_accounts("buy_exact_quote_in", r[0], pool, r[1], T22, TOK, r[9], r[24], pump_coin=True)
    got = [a for a, _, _ in accs]
    assert got[:17] == r[:17] and got[19:] == r[19:]  # 17-18 are the coin creator's vault (a different creator here)
    assert "".join("W" if w else "-" for _, _, w in accs) == "WW---WWWW-W------W--W----W"


def test_plan_picks_the_instruction_by_which_side_sol_is_on():
    assert d.plan("buy", token_is_base=True) == "buy_exact_quote_in" and d.plan("sell", True) == "sell"
    assert d.plan("buy", token_is_base=False) == "sell" and d.plan("sell", False) == "buy_exact_quote_in"
    assert d.swap_data("sell", 5, 7) == d.SELL + (5).to_bytes(8, "little") + (7).to_bytes(8, "little")
    assert d.swap_data("buy_exact_quote_in", 5, 7)[-1:] == bytes([0]) and len(d.swap_data("buy_exact_quote_in", 5, 7)) == 25
    assert d.amount_out(10**9, 100 * 10**9, 10**15, 30) == pytest.approx(10**15 * 0.997 / 100.997, rel=1e-6)


MINT = "Tok1111111111111111111111111111111111111111"
POOL = "Poo1111111111111111111111111111111111111111"
BV, QV = "BVau111111111111111111111111111111111111111", "QVau111111111111111111111111111111111111111"
PROTO, BUYBACK = "Prot111111111111111111111111111111111111111", "Buyb111111111111111111111111111111111111111"


def acct(raw: bytes, owner: str) -> dict:
    return {"data": [base64.b64encode(raw).decode(), "base64"], "owner": owner}


def vault(amount: int) -> bytes:
    return bytes(64) + amount.to_bytes(8, "little") + bytes(93)


class PoolRpc(FakeRpc):
    """FakeRpc plus the pool, global config, mints and vaults of a hand-made pool with SOL as the base."""

    def __init__(self, owner, sol_reserve=10 * 10**9, tok_reserve=10**12, **kw):
        super().__init__(owner, **kw)
        self.accts = {POOL: acct(pool_bytes(WSOL, MINT, BV, QV), d.PROGRAM),
                      d.GLOBAL_CONFIG: acct(global_bytes([PROTO], [BUYBACK]), d.PROGRAM),
                      WSOL: acct(bytes(82), TOK), MINT: acct(bytes(82), T22),
                      BV: acct(vault(sol_reserve), TOK), QV: acct(vault(tok_reserve), T22)}
        self.raw = []

    def post(self, url, json=None):
        m, p = json["method"], json["params"]
        if m == "getMultipleAccounts":
            body = {"result": {"value": [self.accts.get(a) for a in p[0]]}}
            return SimpleNamespace(json=lambda: body)
        if m == "getPriorityFeeEstimate":
            body = {"result": {"priorityFeeEstimate": 1000}}
            return SimpleNamespace(json=lambda: body)
        if m == "sendTransaction" and p[1].get("skipPreflight") is False:
            self.raw.append(p[0])
        return super().post(url, json)


def direct(**kw):
    kp = Keypair()
    rpc = PoolRpc(str(kp.pubkey()), **kw)
    ex = d.DirectPumpswapExecutor(Config(max_price_impact_pct=5.0), FakeJup(), http=rpc, keypair=kp,
                                  sell_slippage_bps=1000, sleep=lambda s: None)
    ex.remember_pool(MINT, POOL)
    return ex, rpc


def sol_meta(owner, token, owner_lamports, new_rent=0, fee=5000, err=None):
    rows = lambda t: [{"mint": MINT, "owner": owner, "uiTokenAmount": {"amount": str(t)}}]
    pre, post = [10**9, 0, 5 * 10**8], [10**9 + owner_lamports, new_rent, 5 * 10**8]
    return {"err": err, "fee": fee, "preBalances": pre, "postBalances": post,
            "preTokenBalances": rows(token[0]), "postTokenBalances": rows(token[1])}


def decoded(raw_b64):
    tx = VersionedTransaction.from_bytes(base64.b64decode(raw_b64))
    keys = [str(k) for k in tx.message.account_keys]
    return [(keys[ix.program_id_index], bytes(ix.data), [keys[i] for i in ix.accounts]) for ix in tx.message.instructions]


def test_direct_buy_sells_sol_into_the_pool_and_reads_the_fill():
    ex, rpc = direct()
    # $10 at $200/SOL = 0.05 SOL into the swap; the tx also paid 5,000 lamports and 0.002 SOL of new-account rent
    rpc.next_meta = sol_meta(ex.owner, (0, 4_800_000_000), -(50_000_000 + 5_000 + 2_000_000), new_rent=2_000_000)
    fill = ex.buy(MINT, 10.0, 0.0)
    assert fill.ok and fill.qty == pytest.approx(4_800.0)  # 6 decimals
    assert fill.fee_usd == pytest.approx((5_000 + 2_000_000) / 1e9 * 200)
    assert fill.usd == pytest.approx(10.0 + fill.fee_usd) and fill.price == pytest.approx(10.0 / 4_800)
    ixs = decoded(rpc.raw[0])
    programs = [p for p, _, _ in ixs]
    assert programs.count("ComputeBudget111111111111111111111111111111") == 2 and d.PROGRAM in programs
    assert "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4" not in programs  # no Jupiter anywhere
    _, data, accounts = next(x for x in ixs if x[0] == d.PROGRAM)
    assert data[:8] == d.SELL and int.from_bytes(data[8:16], "little") == 50_000_000  # SOL is the base: sell SOL
    expected = d.amount_out(50_000_000, 10 * 10**9, 10**12, 30)
    assert int.from_bytes(data[16:24], "little") == int(expected * 0.97)  # default 3% slippage
    assert accounts[0] == POOL and accounts[3] == WSOL and accounts[4] == MINT and accounts[-2] == BUYBACK
    assert programs[-1] == TOK and ixs[-1][1] == bytes([9])  # the wrapped SOL account is closed again


def test_direct_buy_declines_without_enough_sol_or_on_a_thin_pool():
    ex, rpc = direct(sol=40_000_000)  # 0.04 SOL: short of the 0.05 SOL buy plus the 0.01 SOL reserve
    assert ex.buy(MINT, 10.0, 0.0) is None and "wallet SOL low" in ex.last_note and not rpc.sent
    ex, rpc = direct(sol_reserve=10**9)  # 0.05 SOL into a 1 SOL pool is ~4.8% impact
    ex.cfg.max_price_impact_pct = 3.0
    assert ex.buy(MINT, 10.0, 0.0) is None and ex.last_note.startswith("price impact") and not rpc.sent


def test_direct_sale_spends_tokens_for_sol_and_closes_the_account():
    ex, rpc = direct(token=4_800_000_000)
    rpc.next_meta = sol_meta(ex.owner, (4_800_000_000, 0), 49_000_000 - 5_000)
    rpc.bal[MINT] = 4_800_000_000
    closed = []
    ex.close_empty = lambda mint: closed.append(mint) or 0.4  # the token account's rent back
    fill = ex.sell(MINT, 4_800.0, 0.0)
    assert fill.ok and fill.qty == pytest.approx(4_800.0) and closed == [MINT]
    assert fill.usd == pytest.approx(49_000_000 / 1e9 * 200 - (5_000 / 1e9 * 200 - 0.4))
    _, data, _ = next(x for x in decoded(rpc.raw[0]) if x[0] == d.PROGRAM)
    assert data[:8] == d.BUY_EXACT_QUOTE_IN and int.from_bytes(data[8:16], "little") == 4_800_000_000
    expected = d.amount_out(4_800_000_000, 10**12, 10 * 10**9, 30)
    assert int.from_bytes(data[16:24], "little") == int(expected * 0.9)  # the canary's 10% sell slippage


def test_direct_sale_on_an_emptied_pool_sends_nothing():
    ex, rpc = direct(token=10**6, sol_reserve=0)
    rpc.bal[MINT] = 10**6
    fill = ex.sell(MINT, 1.0, 0.0)
    assert not fill.ok and fill.usd == 0.0 and fill.note == "no route (pool emptied)" and not rpc.sent


def test_a_reverted_direct_swap_costs_only_its_fee():
    ex, rpc = direct()
    rpc.next_meta = sol_meta(ex.owner, (0, 0), -5_000, err={"InstructionError": [6, {"Custom": 6004}]})
    fill = ex.buy(MINT, 10.0, 0.0)
    assert not fill.ok and fill.fee_usd == pytest.approx(5_000 / 1e9 * 200) and "reverted" in fill.note


def test_engine_hands_the_signals_pool_to_the_direct_executor(cfg, good_pair):
    class Live(PaperExecutor):
        def __init__(self, c):
            super().__init__(c)
            self.pools = {}

        def remember_pool(self, mint, pool):
            self.pools[mint] = pool

    cfg.launch_book = cfg.launch_live = cfg.launch_canary = True
    try:
        j = Journal(":memory:")
        live = Live(cfg)
        eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                     model_executor=PaperExecutor(cfg), launch_canary_executor=live)
        eng.model_book.signals.put({"pool": "PoolA", "mint": "A", "dex": "pumpswap", "created_ts": time.time() - 6,
                                    "name": "", "price": 1e-5, "liq": 12_000.0, "score": 1.0, "ts": time.time(),
                                    "decided_ts": time.time(), "kind": "launch", "rw50": True})
        eng.buy_model_signals()
        assert live.pools == {"A": "PoolA"} and j.open_positions(mb.CANARY4)
    finally:
        cfg.launch_book = cfg.launch_live = cfg.launch_canary = False


def test_canary_test_run_stops_after_its_trade_cap(cfg, good_pair):
    cfg.launch_book = cfg.launch_live = cfg.launch_canary = True
    cfg.launch_canary_max_trades = 2
    try:
        j = Journal(":memory:")
        eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                     model_executor=PaperExecutor(cfg), launch_canary_executor=PaperExecutor(cfg))
        for m in ("A", "B", "C"):
            eng.model_book.signals.put({"pool": "P" + m, "mint": m, "dex": "pumpswap", "created_ts": time.time() - 6,
                                        "name": "", "price": 1e-5, "liq": 12_000.0, "score": 1.0, "ts": time.time(),
                                        "decided_ts": time.time(), "kind": "launch", "rw50": True})
            eng.buy_model_signals()
        assert [p.token_address for p in j.open_positions(mb.CANARY4)] == ["A", "B"]
        assert eng._model_skip_reason(eng.books[mb.CANARY4], {"mint": "D", "liq": 12_000.0, "ts": time.time()}) == \
            "canary test run done (2 of 2 trades)"
    finally:
        cfg.launch_book = cfg.launch_live = cfg.launch_canary = False
        cfg.launch_canary_max_trades = 0


def test_the_canary_route_setting_picks_the_executor(monkeypatch):
    from tradingbot import cli
    monkeypatch.setattr(d.DirectPumpswapExecutor, "__init__", lambda self, *a, **k: None)
    from tradingbot.execution import jupiter as jx
    monkeypatch.setattr(jx.JupiterExecutor, "__init__", lambda self, *a, **k: None)
    assert type(cli.canary_executor(Config(launch_canary_route="direct"), None)) is d.DirectPumpswapExecutor
    assert type(cli.canary_executor(Config(), None)) is jx.JupiterExecutor
