"""LP route for the launch canary (11-10): enter a pumpswap pool as a liquidity provider instead of a buyer. Layouts
pinned to the IDL; the buy/sell transactions are decoded to check they deposit and withdraw and never touch Jupiter."""
import base64
from types import SimpleNamespace

import pytest

from tests.test_launch_canary import FakeJup
from tests.test_pumpswap_direct import (
    BUYBACK,
    PROTO,
    acct,
    global_bytes,
    pool_bytes,
    vault,
)
from tradingbot.config import Config
from tradingbot.data.pumpswap import WSOL
from tradingbot.execution import lp_pumpswap as L
from tradingbot.execution import pumpswap_direct as D

solders = pytest.importorskip("solders")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

MINT = "Tok1111111111111111111111111111111111111111"
POOL = "Poo1111111111111111111111111111111111111111"
LP = "LPmint1111111111111111111111111111111111111"
BV, QV = "BVau111111111111111111111111111111111111111", "QVau111111111111111111111111111111111111111"
T22, TOK = D.TOKEN_2022, D.TOKEN
JUPITER = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"


def test_layouts_and_discriminators():
    assert L.DEPOSIT.hex() == "f223c68952e1f2b6" and L.WITHDRAW.hex() == "b712469c946da122"
    pool = {"base_mint": WSOL, "quote_mint": MINT, "lp_mint": LP, "base_vault": BV, "quote_vault": QV}
    user = str(Keypair().pubkey())
    accs = L.lp_accounts(L.DEPOSIT, POOL, pool, user, TOK, T22)
    assert len(accs) == 15
    names = [POOL, L.GLOBAL_CONFIG, None, WSOL, MINT, LP, None, None, None, BV, QV, TOK, T22, L.EVENT_AUTHORITY, L.PROGRAM]
    for (addr, _, _), want in zip(accs, names):
        if want is not None:
            assert addr == want
    assert accs[2][1] is True  # user signs
    assert L.lp_data(L.DEPOSIT, 5, 7, 9) == L.DEPOSIT + (5).to_bytes(8, "little") + (7).to_bytes(8, "little") + (9).to_bytes(8, "little")


def lp_mint_bytes(supply: int, decimals: int = 6) -> bytes:
    raw = (1).to_bytes(4, "little") + bytes(Pubkey.from_string(POOL)) + supply.to_bytes(8, "little") + bytes([decimals])
    return raw + bytes(82 - len(raw))


class LpRpc:
    """Answers the LP executor's RPC: pool state, LP mint, balances, and a settle transaction."""

    def __init__(self, owner, sol_reserve=50 * 10**9, tok_reserve=10**15, lp_supply=10**12, wallet_sol=5 * 10**9):
        self.owner = owner
        self.accts = {POOL: acct(pool_bytes(WSOL, MINT, BV, QV), D.PROGRAM),
                      D.GLOBAL_CONFIG: acct(global_bytes([PROTO], [BUYBACK]), D.PROGRAM),
                      WSOL: acct(bytes(82), TOK), MINT: acct(bytes(82), T22),
                      BV: acct(vault(sol_reserve), TOK), QV: acct(vault(tok_reserve), T22),
                      LP: acct(lp_mint_bytes(lp_supply), T22)}
        self.wallet_sol = wallet_sol
        self.bal = {LP: 0, MINT: 0}
        self.raw = []
        self.meta = None  # settle meta
        self.revert_first = False  # first sendTransaction's settle reverts (dead-pool exit)
        self.revert_raw = set()  # 1-based send numbers whose settle reverts (sweep retry test)
        self.reject_atomic = False  # reject (in preflight, before sending) any tx carrying a sale leg

    def post(self, url, json=None):
        m, p = json["method"], json["params"]
        res = None
        if m == "getMultipleAccounts":
            res = {"value": [self.accts.get(a) for a in p[0]]}
        elif m == "getTokenAccountsByOwner":
            amt = self.bal.get(p[1]["mint"], 0)
            res = {"value": [{"pubkey": "Acc", "account": {"owner": TOK,
                   "data": {"parsed": {"info": {"tokenAmount": {"amount": str(amt)}}}}}}]}
        elif m == "getBalance":
            res = {"value": self.wallet_sol}
        elif m == "getPriorityFeeEstimate":
            res = {"priorityFeeEstimate": 1000}
        elif m == "getLatestBlockhash":
            from solders.hash import Hash
            res = {"value": {"blockhash": str(Hash.default()), "lastValidBlockHeight": 1}}
        elif m == "sendTransaction":
            if self.reject_atomic:
                discs = {d for _, d in decoded(p[0])}
                if D.SELL.hex() in discs or D.BUY_EXACT_QUOTE_IN.hex() in discs:  # a drained pool reverts the sale leg
                    return SimpleNamespace(json=lambda: {"error": {"code": -32002, "message": "preflight: custom 6001"}})
            self.raw.append(p[0])
            res = f"sig{len(self.raw)}"
        elif m == "getSignatureStatuses":
            res = {"value": [{"confirmationStatus": "confirmed", "err": (self.meta or {}).get("err")}]}
        elif m == "getTransaction":
            n = int(p[0][3:])  # sig is "sig<N>", N = the send's 1-based order
            if (self.revert_first and n == 1) or n in self.revert_raw:
                res = {"meta": {**self.meta, "err": {"InstructionError": [7, {"Custom": 6001}]}}}
            else:
                res = {"meta": self.meta}
        return SimpleNamespace(json=lambda: {"result": res})


def executor(**kw):
    kp = Keypair()
    rpc = LpRpc(str(kp.pubkey()), **kw)
    ex = L.LpPumpswapExecutor(Config(max_price_impact_pct=5.0, db_path=":memory:"), FakeJup(), http=rpc, keypair=kp,
                              sell_slippage_bps=1000, sleep=lambda s: None)
    ex.remember_pool(MINT, POOL)
    return ex, rpc


def meta(owner_delta, lp_delta=0, tok_delta=0, fee=5000):
    rows = lambda lp, tk: ([{"mint": LP, "owner": None, "uiTokenAmount": {"amount": str(lp)}}]
                           + [{"mint": MINT, "owner": None, "uiTokenAmount": {"amount": str(tk)}}])
    return {"err": None, "fee": fee, "preBalances": [10**9, 0], "postBalances": [10**9 + owner_delta, 0],
            "preTokenBalances": rows(0, 0), "postTokenBalances": rows(lp_delta, tok_delta)}


def decoded(raw_b64):
    tx = VersionedTransaction.from_bytes(base64.b64decode(raw_b64))
    keys = [str(k) for k in tx.message.account_keys]
    return [(keys[ix.program_id_index], bytes(ix.data)[:8].hex()) for ix in tx.message.instructions]


def _fill_owner(meta_dict, ex):
    # the settle reads this wallet's own token rows; stamp the owner in now that we know it
    for r in meta_dict["preTokenBalances"] + meta_dict["postTokenBalances"]:
        r["owner"] = ex.owner


def test_lp_buy_deposits_and_never_touches_jupiter():
    ex, rpc = executor()
    rpc.meta = meta(owner_delta=-(25 * 10**9 // 150 + 5000), lp_delta=4_000_000, tok_delta=100)  # SOL spent, LP minted
    _fill_owner(rpc.meta, ex)
    fill = ex.buy(MINT, 25.0, 0.0)  # FakeJup SOL price 200
    assert fill is not None and fill.ok and fill.qty > 0  # LP tokens received
    progs = decoded(rpc.raw[0])
    discs = [d for _, d in progs]
    assert L.DEPOSIT.hex() in discs                                   # a deposit instruction is in the buy
    assert D.SELL.hex() in discs or D.BUY_EXACT_QUOTE_IN.hex() in discs  # and the buy swap
    assert JUPITER not in [pid for pid, _ in progs]                   # no Jupiter anywhere


def test_lp_buy_sells_the_excess_naked_tokens():
    # The atomic buy+deposit leaves ~15% of the bought tokens naked (it can't read the exact buy output in-tx). A
    # follow-up transaction SELLS them back to SOL (not a deposit), so nothing is held outside the pool and the
    # proceeds stay as safe wallet SOL rather than being pushed into a pool that may be drained.
    ex, rpc = executor()
    rpc.bal[MINT] = 2 * 10**12  # naked tokens the atomic buy left behind, read by the excess sell
    rpc.meta = meta(owner_delta=-(25 * 10**9 // 150 + 5000), lp_delta=4_000_000, tok_delta=100)
    _fill_owner(rpc.meta, ex)
    fill = ex.buy(MINT, 25.0, 0.0)
    assert fill is not None and fill.ok
    assert len(rpc.raw) == 2  # the atomic buy+deposit, then the excess sell
    excess = [d for _, d in decoded(rpc.raw[1])]
    assert D.SELL.hex() in excess or D.BUY_EXACT_QUOTE_IN.hex() in excess  # the excess tx is a sell ...
    assert L.DEPOSIT.hex() not in excess                                   # ... not a deposit
    assert fill.qty == 4_000_000 / 10**6  # qty is the LP from the one deposit; the excess sell only nets SOL back


def test_lp_buy_declines_without_enough_sol():
    ex, rpc = executor(wallet_sol=10**8)  # 0.1 SOL: short of a $25 (~0.125 SOL) LP plus slack
    assert ex.buy(MINT, 25.0, 0.0) is None and "wallet SOL low" in ex.last_note and not rpc.raw


def test_lp_sell_withdraws_and_sells_in_one_transaction():
    ex, rpc = executor()
    rpc.bal[LP] = 4_000_000  # we hold LP
    rpc.bal[MINT] = 10**6    # naked tokens the buy left in the wallet, to be swept by the sale
    rpc.meta = meta(owner_delta=12 * 10**9 // 150)  # SOL recovered
    _fill_owner(rpc.meta, ex)
    fill = ex.sell(MINT, 4.0, 0.0)
    assert fill.ok and fill.usd > 0 and len(rpc.raw) == 1  # one atomic transaction
    discs = [d for _, d in decoded(rpc.raw[0])]
    assert L.WITHDRAW.hex() in discs                                    # withdraw ...
    assert D.SELL.hex() in discs or D.BUY_EXACT_QUOTE_IN.hex() in discs  # ... and the sale, in the same tx
    assert JUPITER not in [pid for pid, _ in decoded(rpc.raw[0])]


def test_lp_sell_falls_back_to_withdraw_only_when_the_sale_reverts():
    ex, rpc = executor()
    rpc.bal[LP] = 4_000_000
    rpc.bal[MINT] = 10**6
    rpc.revert_first = True  # the atomic withdraw+sell reverts; the fallback is a withdraw-only tx
    rpc.meta = meta(owner_delta=6 * 10**9 // 150)
    _fill_owner(rpc.meta, ex)
    fill = ex.sell(MINT, 4.0, 0.0)
    assert fill.ok and fill.usd > 0 and len(rpc.raw) == 2  # the reverted atomic tx, then withdraw-only
    assert all(L.WITHDRAW.hex() in [d for _, d in decoded(r)] for r in rpc.raw)
    fallback = [d for _, d in decoded(rpc.raw[1])]
    assert D.SELL.hex() not in fallback and D.BUY_EXACT_QUOTE_IN.hex() not in fallback  # no sale in the fallback


def test_lp_sell_falls_back_when_the_atomic_exit_is_rejected_in_preflight():
    # The 9w1vtt stuck-at-$0 bug: on a fully-drained pool the sale leg reverts, so the atomic withdraw+sell is
    # rejected in preflight (never sent). The exit must still fall back to withdraw-only and recover the SOL share,
    # not bail with a "not sent:" note that the engine retries silently forever.
    ex, rpc = executor()
    rpc.bal[LP] = 4_000_000
    rpc.bal[MINT] = 10**6
    rpc.reject_atomic = True  # the withdraw+sell tx is rejected before sending; withdraw-only (no sale) still sends
    rpc.meta = meta(owner_delta=6 * 10**9 // 150)  # the withdraw-only returns our residual SOL
    _fill_owner(rpc.meta, ex)
    fill = ex.sell(MINT, 4.0, 0.0)
    assert fill is not None and fill.ok and fill.usd > 0
    assert len(rpc.raw) == 1  # only the withdraw-only landed (the atomic was rejected, never sent)
    discs = [d for _, d in decoded(rpc.raw[0])]
    assert L.WITHDRAW.hex() in discs
    assert D.SELL.hex() not in discs and D.BUY_EXACT_QUOTE_IN.hex() not in discs  # no sale leg in the fallback


def test_route_setting_picks_the_lp_executor(monkeypatch):
    from tradingbot import cli
    monkeypatch.setattr(L.LpPumpswapExecutor, "__init__", lambda self, *a, **k: None)
    assert type(cli.canary_executor(Config(launch_canary_route="lp"), None)) is L.LpPumpswapExecutor
