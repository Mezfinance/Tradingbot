"""Live executor that swaps straight on the pumpswap pool, with no Jupiter route (11-10).

Real money. Built for the launch canary to test whether operators react to Jupiter-routed buys: every real canary buy
so far went through Jupiter's program, which marks the transaction as an outside buyer; operator bots call pumpswap
directly. This executor sends the same shape of transaction those bots send (compute budget, token-account create,
wrap SOL, pumpswap `buy_exact_quote_in` / `sell`, unwrap), signed by the same wallet.

It trades SOL, not USDC: the wallet's SOL pays for buys and sales come back as SOL (the book still counts in USD at
Jupiter's SOL price). Layouts are from pump.fun's published IDL (pump-public-docs/idl/pump_amm.json) checked against
real mainnet swaps on 11-10:
  - v1 `sell` (21 IDL accounts) and `buy_exact_quote_in` (23) are what direct traders use, each followed by remaining
    accounts the IDL doesn't list: a buyback fee recipient from the global config (read-only) and its token account
    for the quote mint (writable); pools of pump.fun coins put PDA('pool-v2', base_mint) first;
  - hand-made pools put SOL on either side, so a token buy is `buy_exact_quote_in` (spend SOL) when the token is the
    base and `sell` (sell SOL) when SOL is the base; a sale is the mirror image.
Everything else (RPC, sending, resending, confirmation, closing accounts, the sweep) is the Jupiter executor's.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import time

from tradingbot.data.helius import b58encode
from tradingbot.data.pumpswap import PROGRAM, WSOL, parse_pool
from tradingbot.execution.base import Fill, failed
from tradingbot.execution.jupiter import DUST_SHARE, MIN_SOL, NOT_SENT, JupiterExecutor

log = logging.getLogger(__name__)

GLOBAL_CONFIG = "ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw"
FEE_PROGRAM = "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ"
ATA_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
SYSTEM = "11111111111111111111111111111111"
TOKEN = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
NO_KEY = SYSTEM  # 32 zero bytes
SELL = bytes([51, 230, 133, 164, 1, 127, 131, 173])
BUY_EXACT_QUOTE_IN = bytes([198, 46, 21, 82, 180, 217, 232, 112])
CU_LIMIT = 300_000  # real direct swaps use ~80-115k compute units; a first buy on a fresh pool also creates up to 3 fee accounts (~170k simulated 11-10)
GLOBAL_TTL_S = 600
ROUTE = "direct"


def pda(seeds: list[bytes], program: str) -> str:
    from solders.pubkey import Pubkey
    return str(Pubkey.find_program_address(seeds, Pubkey.from_string(program))[0])


def key(addr: str) -> bytes:
    from solders.pubkey import Pubkey
    return bytes(Pubkey.from_string(addr))


def ata(owner: str, token_program: str, mint: str) -> str:
    return pda([key(owner), key(token_program), key(mint)], ATA_PROGRAM)


EVENT_AUTHORITY = "GS4CU59F31iL7aR2Q8zVS8DRrcRnXX1yjQ66TqNVQnaR"  # pda([b"__event_authority"], PROGRAM)
GLOBAL_VOLUME = "C2aFPdENg4A2HQsmrd5rTw5TaYBX5Ku887cWjbFKtZpw"  # pda([b"global_volume_accumulator"], PROGRAM)
FEE_CONFIG = "5PHirr8joyTMp9JMm6nW7hNDVyEYdkzDqazxPD7RaTjx"  # pda([b"fee_config", PROGRAM], FEE_PROGRAM)


def parse_global(raw: bytes) -> dict:
    """GlobalConfig: admin, lp/protocol bps, disable flags, 8 protocol fee recipients, creator bps, ..., 8 buyback
    fee recipients (Anchor layout from the IDL; offsets checked against the live account, 949 bytes)."""
    u64 = lambda o: int.from_bytes(raw[o:o + 8], "little")
    keys = lambda o, n: [b58encode(raw[o + 32 * i:o + 32 * i + 32]) for i in range(n)]
    o = 8 + 32
    lp, protocol, disable = u64(o), u64(o + 8), raw[o + 16]
    o += 17
    protocol_recipients = keys(o, 8)
    o += 256
    creator = u64(o)
    o += 8 + 32 + 32 + 32 + 1 + 224 + 1  # set-creator authority, whitelist, reserved recipient, mayhem, reserved x7, cashback
    buyback = keys(o, 8)
    return {"lp_bps": lp, "protocol_bps": protocol, "creator_bps": creator, "disable": disable,
            "protocol_recipients": [k for k in protocol_recipients if k != NO_KEY],
            "buyback_recipients": [k for k in buyback if k != NO_KEY]}


def pool_extra(raw: bytes) -> dict:
    """Pool fields after coin_creator: is_mayhem_mode @243, is_cashback_coin @244, virtual_quote_reserves i128 @245
    (0 for pools that aren't boosted)."""
    if len(raw) < 261:
        return {"cashback": False, "virtual_quote": 0}
    return {"cashback": bool(raw[244]), "virtual_quote": int.from_bytes(raw[245:261], "little", signed=True)}


def amount_out(amount_in: int, reserve_in: int, reserve_out: int, fee_bps: int) -> int:
    """Constant-product output with the pool's total fee taken off the input (pumpswap charges buys on the quote in
    and sells on the quote out; the difference is inside the slippage tolerance)."""
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return 0
    net = amount_in * (10_000 - fee_bps) // 10_000
    return reserve_out * net // (reserve_in + net)


def swap_accounts(name: str, pool_addr: str, pool: dict, user: str, base_prog: str, quote_prog: str,
                  protocol_recipient: str, buyback_recipient: str, pump_coin: bool) -> list[tuple[str, bool, bool]]:
    """(address, signer, writable) in the order the program expects: the IDL's accounts, then the remaining ones seen
    on real swaps."""
    base, quote = pool["base_mint"], pool["quote_mint"]
    creator_auth = pda([b"creator_vault", key(pool["coin_creator"])], PROGRAM)
    acc = [(pool_addr, False, True), (user, True, True), (GLOBAL_CONFIG, False, False), (base, False, False),
           (quote, False, False), (ata(user, base_prog, base), False, True), (ata(user, quote_prog, quote), False, True),
           (pool["base_vault"], False, True), (pool["quote_vault"], False, True), (protocol_recipient, False, False),
           (ata(protocol_recipient, quote_prog, quote), False, True), (base_prog, False, False),
           (quote_prog, False, False), (SYSTEM, False, False), (ATA_PROGRAM, False, False),
           (EVENT_AUTHORITY, False, False), (PROGRAM, False, False),
           (ata(creator_auth, quote_prog, quote), False, True), (creator_auth, False, False)]
    if name == "buy_exact_quote_in":
        acc += [(GLOBAL_VOLUME, False, False),
                (pda([b"user_volume_accumulator", key(user)], PROGRAM), False, True)]
    acc += [(FEE_CONFIG, False, False), (FEE_PROGRAM, False, False)]
    if pump_coin:
        acc.append((pda([b"pool-v2", key(base)], PROGRAM), False, False))
    acc += [(buyback_recipient, False, False), (ata(buyback_recipient, quote_prog, quote), False, True)]
    return acc


def swap_data(name: str, amount_in: int, min_out: int) -> bytes:
    le = lambda n: int(n).to_bytes(8, "little")
    if name == "sell":  # base_amount_in, min_quote_amount_out
        return SELL + le(amount_in) + le(min_out)
    return BUY_EXACT_QUOTE_IN + le(amount_in) + le(min_out) + bytes([0])  # track_volume: OptionBool(false)


def plan(side: str, token_is_base: bool) -> str:
    """Which instruction a token buy or sale is on this pool."""
    if side == "buy":
        return "buy_exact_quote_in" if token_is_base else "sell"
    return "sell" if token_is_base else "buy_exact_quote_in"


def _pick(items: list[str], seed: str) -> str:
    """A stable choice among the fee recipients (any of them is accepted)."""
    return items[int(hashlib.sha1(seed.encode()).hexdigest(), 16) % len(items)]


class DirectPumpswapExecutor(JupiterExecutor):
    route = ROUTE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pools: dict[str, str] = {}  # token mint -> pumpswap pool, from the signal (or looked up)
        self._global: tuple[float, dict] | None = None

    def remember_pool(self, mint: str, pool: str) -> None:
        self.pools[mint] = pool

    # --- pool state -------------------------------------------------------------
    def _accounts(self, addrs: list[str]) -> list[dict | None]:
        res = self._rpc("getMultipleAccounts", [addrs, {"encoding": "base64", "commitment": "confirmed"}]) or {}
        return res.get("value") or [None] * len(addrs)

    def _global_config(self) -> dict | None:
        if self._global and time.time() - self._global[0] < GLOBAL_TTL_S:
            return self._global[1]
        acct = self._accounts([GLOBAL_CONFIG])[0]
        if not acct:
            return None
        g = parse_global(base64.b64decode(acct["data"][0]))
        self._global = (time.time(), g)
        return g

    def _find_pool(self, mint: str) -> str | None:
        """The SOL pool of `mint` with the most SOL, when the signal's pool isn't known (after a restart)."""
        found = []
        for offset in (43, 75):  # base_mint, quote_mint
            res = self._rpc("getProgramAccounts", [PROGRAM, {"encoding": "base64", "commitment": "confirmed",
                            "filters": [{"memcmp": {"offset": offset, "bytes": mint}}]}]) or []
            for r in res if isinstance(res, list) else []:
                p = parse_pool(base64.b64decode(r["account"]["data"][0]))
                if p and p["token_mint"] == mint:
                    found.append((r["pubkey"], p))
        if not found:
            return None
        vaults = self._accounts([p["sol_vault"] for _, p in found])
        sol = [_token_amount(v) for v in vaults]
        return max(zip(sol, [a for a, _ in found]))[1]

    def _state(self, mint: str) -> dict | None:
        pool_addr = self.pools.get(mint) or self._find_pool(mint)
        if not pool_addr:
            self.last_note = "no pumpswap pool found"
            return None
        self.pools[mint] = pool_addr
        g = self._global_config()
        accts = self._accounts([pool_addr])
        if not g or not accts[0]:
            self.last_note = "pool or global config unreadable"
            return None
        raw = base64.b64decode(accts[0]["data"][0])
        pool = parse_pool(raw)
        if not pool or pool["token_mint"] != mint:
            self.last_note = "not a SOL pool of this token"
            return None
        if g["disable"] & 0b11000:
            self.last_note = "pumpswap trading disabled"
            return None
        extra = pool_extra(raw)
        mints = self._accounts([pool["base_mint"], pool["quote_mint"], pool["base_vault"], pool["quote_vault"]])
        if not all(mints):
            self.last_note = "pool accounts unreadable"
            return None
        token_is_base = pool["base_mint"] == mint
        base_amt, quote_amt = _token_amount(mints[2]), _token_amount(mints[3]) + max(0, extra["virtual_quote"])
        return {"addr": pool_addr, "pool": pool, "global": g, "token_is_base": token_is_base,
                "base_prog": mints[0]["owner"], "quote_prog": mints[1]["owner"],
                "token_prog": mints[0]["owner"] if token_is_base else mints[1]["owner"],
                "reserves": (base_amt, quote_amt),
                "pump_coin": pool["coin_creator"] != NO_KEY,
                "fee_bps": g["lp_bps"] + g["protocol_bps"] + g["creator_bps"]}

    # --- transactions -------------------------------------------------------------
    def _priority_micro(self, accounts: list[str]) -> int:
        """Micro-lamports per compute unit: Helius's 'VeryHigh' estimate for these accounts, capped so the priority
        fee never passes `priority_fee_sol`; half the cap when there is no estimate."""
        cap = int(self.priority_fee_sol * 1e9 * 1e6 / CU_LIMIT)
        res = self._rpc("getPriorityFeeEstimate", [{"accountKeys": accounts, "options": {"priorityLevel": "VeryHigh"}}])
        try:
            est = int(float(res["priorityFeeEstimate"]))
        except (TypeError, KeyError, ValueError):
            return cap // 2
        return max(1, min(est, cap))

    def _instructions(self, st: dict, side: str, amount_in: int, min_out: int) -> list:
        from solders.compute_budget import (
            set_compute_unit_limit,
            set_compute_unit_price,
        )
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey
        from solders.system_program import TransferParams, transfer

        P = Pubkey.from_string
        me, g, pool = self.owner, st["global"], st["pool"]
        name = plan(side, st["token_is_base"])
        accs = swap_accounts(name, st["addr"], pool, me, st["base_prog"], st["quote_prog"],
                             _pick(g["protocol_recipients"], st["addr"]), _pick(g["buyback_recipients"], me),
                             st["pump_coin"])
        swap = Instruction(P(PROGRAM), swap_data(name, amount_in, min_out),
                           [AccountMeta(P(a), s, w) for a, s, w in accs])
        wsol = ata(me, TOKEN, WSOL)

        def create(owner_ata: str, mint: str, prog: str):
            return Instruction(P(ATA_PROGRAM), bytes([1]), [
                AccountMeta(P(me), True, True), AccountMeta(P(owner_ata), False, True), AccountMeta(P(me), False, False),
                AccountMeta(P(mint), False, False), AccountMeta(P(SYSTEM), False, False), AccountMeta(P(prog), False, False)])

        ixs = [set_compute_unit_limit(CU_LIMIT),
               set_compute_unit_price(self._priority_micro([st["addr"], PROGRAM])),
               create(wsol, WSOL, TOKEN)]
        if side == "buy":
            token = st["pool"]["token_mint"]
            ixs += [create(ata(me, st["token_prog"], token), token, st["token_prog"]),
                    transfer(TransferParams(from_pubkey=P(me), to_pubkey=P(wsol), lamports=amount_in)),
                    Instruction(P(TOKEN), bytes([17]), [AccountMeta(P(wsol), False, True)])]  # SyncNative
        ixs += [swap, Instruction(P(TOKEN), bytes([9]), [  # CloseAccount: unwrap every lamport back to the wallet
            AccountMeta(P(wsol), False, True), AccountMeta(P(me), False, True), AccountMeta(P(me), True, False)])]
        return ixs

    def _send_ixs(self, ixs: list) -> str:
        from solders.hash import Hash
        from solders.message import MessageV0
        from solders.transaction import VersionedTransaction

        latest = (self._rpc("getLatestBlockhash", [{"commitment": "confirmed"}]) or {}).get("value") or {}
        if not latest.get("blockhash"):
            return f"{NOT_SENT} no blockhash"
        msg = MessageV0.try_compile(self.keypair.pubkey(), ixs, [], Hash.from_string(latest["blockhash"]))
        raw = base64.b64encode(bytes(VersionedTransaction(msg, [self.keypair]))).decode()
        sig, err = self._rpc_err("sendTransaction", [raw, {"encoding": "base64", "skipPreflight": False, "maxRetries": 0}])
        if not sig:
            return f"{NOT_SENT} {str(err)[:120] if err else 'send failed'}"
        self._pending[sig] = (raw, latest.get("lastValidBlockHeight"))
        log.info("direct swap sent %s", sig)
        return sig

    def _settle_direct(self, sig: str, mint: str) -> dict:
        """{ok, note, token (raw delta), swap_lamports (SOL into (-) or out of (+) the swap), fee_lamports (network fee
        plus rent of accounts this transaction created and left open)}."""
        status = self._await(sig)
        tx = self._fetch_tx(sig) if status != "dropped" else None
        if tx is None:
            note = f"not confirmed (dropped) {sig}" if status == "dropped" else "confirmed but transaction not readable"
            return {"ok": False, "note": note, "token": 0, "swap_lamports": 0,
                    "fee_lamports": 0 if status == "dropped" else int(self.priority_fee_sol * 1e9)}
        meta = tx.get("meta") or {}
        pre, post = meta.get("preBalances") or [0], meta.get("postBalances") or [0]
        fee = int(meta.get("fee") or 0)
        if meta.get("err"):
            return {"ok": False, "note": f"reverted: {meta['err']}", "token": 0, "swap_lamports": 0,
                    "fee_lamports": pre[0] - post[0]}
        rent = sum(b - a for a, b in zip(pre[1:], post[1:]) if a == 0 and b > 0)
        swap = (post[0] - pre[0]) + fee + rent
        return {"ok": True, "note": "ok", "token": self._delta(meta, mint), "swap_lamports": swap,
                "fee_lamports": fee + rent}

    # --- public -------------------------------------------------------------------
    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        sol_usd = self.jup.sol_price()
        if not sol_usd:
            self.last_note = "SOL price unknown"
            return None
        decimals = self._decimals(token)
        if decimals is None:
            self.last_note = "token decimals unknown"
            return None
        lamports = int(usd / sol_usd * 1e9)
        sol = self.sol_balance()
        if sol is None or sol < lamports / 1e9 + MIN_SOL:
            self.last_note = f"wallet SOL low ({sol if sol is not None else '?'}) for a {lamports / 1e9:.4f} SOL buy"
            return None
        st = self._state(token)
        if st is None:
            return None
        base_amt, quote_amt = st["reserves"]
        sol_res, tok_res = (quote_amt, base_amt) if st["token_is_base"] else (base_amt, quote_amt)
        impact = lamports / (sol_res + lamports) * 100 if sol_res else 100.0
        if impact > self.cfg.max_price_impact_pct:
            self.last_note = f"price impact {impact:.2f}%"
            return None
        expected = amount_out(lamports, sol_res, tok_res, st["fee_bps"])
        min_out = int(expected * (10_000 - self.cfg.slippage_bps) / 10_000)
        if min_out <= 0:
            self.last_note = "pool has no tokens to sell"
            return None
        sent = self._send_ixs(self._instructions(st, "buy", lamports, min_out))
        if sent.startswith(NOT_SENT):
            self.last_note = sent
            return None
        res = self._settle_direct(sent, token)
        fee_usd = res["fee_lamports"] / 1e9 * sol_usd
        if not res["ok"]:
            self.last_note = res["note"]
            return failed(fee_usd, res["note"])
        got, spent = res["token"], -res["swap_lamports"]
        if got <= 0 or spent <= 0:
            self.last_note = "confirmed but no tokens received"
            return failed(fee_usd, self.last_note)
        qty = got / 10**decimals
        swap_usd = spent / 1e9 * sol_usd
        self.last_note = "ok"
        return Fill(qty=qty, price=swap_usd / qty, usd=swap_usd + fee_usd, fee_usd=fee_usd, tx=sent, note="ok")

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        sol_usd = self.jup.sol_price()
        if not sol_usd:
            return failed(0.0, f"{NOT_SENT} SOL price unknown")
        decimals = self._decimals(token)
        if decimals is None:
            return failed(0.0, "token decimals unknown")
        held = self.token_balance(token)
        raw_in = min(round(qty * 10**decimals), held)
        if held - raw_in <= max(1, held * DUST_SHARE):
            raw_in = held
        if raw_in <= 0:
            return failed(0.0, "no balance")
        st = self._state(token)
        if st is None:
            return failed(0.0, f"{NOT_SENT} {self.last_note}")
        base_amt, quote_amt = st["reserves"]
        sol_res, tok_res = (quote_amt, base_amt) if st["token_is_base"] else (base_amt, quote_amt)
        expected = amount_out(raw_in, tok_res, sol_res, st["fee_bps"])
        if expected <= 0:
            # An emptied (pulled) pool pays nothing: the program refuses with ZeroBaseAmount (simulated 11-10), so a
            # send would only pay a fee on every retry. Same note as Jupiter's no route: the engine marks the exit
            # blocked and retries every few minutes.
            return failed(0.0, "no route (pool emptied)")
        slip = self.sell_slippage_bps if self.sell_slippage_bps is not None else self.cfg.slippage_bps
        min_out = max(1, int(expected * (10_000 - slip) / 10_000))  # buy_exact_quote_in refuses a 0 minimum (ZeroBaseAmount)
        sent = self._send_ixs(self._instructions(st, "sell", raw_in, min_out))
        if sent.startswith(NOT_SENT):
            return failed(0.0, sent)
        res = self._settle_direct(sent, token)
        fee = res["fee_lamports"] / 1e9 * sol_usd
        if not res["ok"]:
            return failed(fee, res["note"])
        sold_raw, proceeds = -res["token"], res["swap_lamports"] / 1e9 * sol_usd
        if sold_raw <= 0:
            return failed(fee, "confirmed but no tokens sold")
        sold = sold_raw / 10**decimals
        if raw_in >= held:
            fee -= self.close_empty(token)
        return Fill(qty=sold, price=proceeds / sold, usd=proceeds - fee, fee_usd=fee, tx=sent, note="ok")


def _token_amount(acct: dict | None) -> int:
    if not acct:
        return 0
    raw = base64.b64decode(acct["data"][0])
    return int.from_bytes(raw[64:72], "little") if len(raw) >= 72 else 0
