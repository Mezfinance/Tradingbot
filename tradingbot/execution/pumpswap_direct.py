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
import json
import logging
import os
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
FEINT_CU_LIMIT = 500_000  # the feint is two swaps (buy + sell) plus the fee-account creation in one transaction
PROBE_SLOT_WAIT_S = 2.5   # how long to wait for the probe's buy to be visible on-chain before firing the sell
PROBE_POLL_S = 0.1        # how often to poll the buy's status while waiting
GLOBAL_TTL_S = 600
ROUTE = "direct"
POOL_FILE = "canary_pools.json"  # next to the journal: mint -> pool of the canary's buys
POOL_FILE_KEEP = 500
LARGEST_CHECKED = 8  # largest holders checked for the pool's vault
SWEEP_QTY = 1e18  # a sentinel qty for the dust sweep: sell() clamps raw_in to the held balance, so this sells it all


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
        db = getattr(self.cfg, "db_path", ":memory:")
        self._pool_file = None if db == ":memory:" else os.path.join(os.path.dirname(db) or ".", POOL_FILE)
        if self._pool_file and os.path.exists(self._pool_file):
            try:
                with open(self._pool_file) as fh:
                    self.pools.update(json.load(fh))
            except (OSError, ValueError):
                log.warning("could not read %s", self._pool_file)

    def remember_pool(self, mint: str, pool: str) -> None:
        """Kept on disk too (11-10): a position bought before a restart must still find its pool to sell."""
        self.pools[mint] = pool
        if not self._pool_file:
            return
        keep = dict(list(self.pools.items())[-POOL_FILE_KEEP:])
        try:
            tmp = self._pool_file + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(keep, fh)
            os.replace(tmp, self._pool_file)
        except OSError:
            log.warning("could not save %s", self._pool_file, exc_info=True)

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
        """The SOL pool of `mint` with the most SOL, when its pool wasn't saved: a pool's token vault is almost
        always among the token's largest holders, and the vault's owner is the pool (getProgramAccounts over all
        ~10M pumpswap accounts is refused by Helius, 11-10)."""
        res = self._rpc("getTokenLargestAccounts", [mint, {"commitment": "confirmed"}]) or {}
        top = [a["address"] for a in (res.get("value") or [])[:LARGEST_CHECKED]]
        if not top:
            return None
        owners = [b58encode(base64.b64decode(a["data"][0])[32:64]) for a in self._accounts(top) if a]
        found = []
        for addr, acct in zip(owners, self._accounts(owners) if owners else []):
            if acct and acct.get("owner") == PROGRAM:
                p = parse_pool(base64.b64decode(acct["data"][0]))
                if p and p["token_mint"] == mint:
                    found.append((addr, p))
        if not found:
            return None
        sol = [_token_amount(v) for v in self._accounts([p["sol_vault"] for _, p in found])]
        return max(zip(sol, [a for a, _ in found]))[1]

    def _state(self, mint: str) -> dict | None:
        pool_addr = self.pools.get(mint) or self._find_pool(mint)
        if not pool_addr:
            self.last_note = "no pumpswap pool found"
            return None
        if self.pools.get(mint) != pool_addr:
            self.remember_pool(mint, pool_addr)
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
    def _priority_micro(self, accounts: list[str], cu: int = CU_LIMIT) -> int:
        """Micro-lamports per compute unit: Helius's 'VeryHigh' estimate for these accounts, capped so the priority
        fee never passes `priority_fee_sol` over `cu` compute units; half the cap when there is no estimate."""
        cap = int(self.priority_fee_sol * 1e9 * 1e6 / cu)
        res = self._rpc("getPriorityFeeEstimate", [{"accountKeys": accounts, "options": {"priorityLevel": "VeryHigh"}}])
        try:
            est = int(float(res["priorityFeeEstimate"]))
        except (TypeError, KeyError, ValueError):
            return cap // 2
        return max(1, min(est, cap))

    def _swap_ix(self, st: dict, side: str, amount_in: int, min_out: int):
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey
        P = Pubkey.from_string
        me, g = self.owner, st["global"]
        name = plan(side, st["token_is_base"])
        accs = swap_accounts(name, st["addr"], st["pool"], me, st["base_prog"], st["quote_prog"],
                             _pick(g["protocol_recipients"], st["addr"]), _pick(g["buyback_recipients"], me),
                             st["pump_coin"])
        return Instruction(P(PROGRAM), swap_data(name, amount_in, min_out),
                           [AccountMeta(P(a), s, w) for a, s, w in accs])

    def _create_ata_ix(self, owner_ata: str, mint: str, prog: str):
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey
        P, me = Pubkey.from_string, self.owner
        return Instruction(P(ATA_PROGRAM), bytes([1]), [
            AccountMeta(P(me), True, True), AccountMeta(P(owner_ata), False, True), AccountMeta(P(me), False, False),
            AccountMeta(P(mint), False, False), AccountMeta(P(SYSTEM), False, False), AccountMeta(P(prog), False, False)])

    def _close_wsol_ix(self):
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey
        P, me = Pubkey.from_string, self.owner
        wsol = ata(me, TOKEN, WSOL)
        return Instruction(P(TOKEN), bytes([9]), [  # CloseAccount: unwrap every lamport back to the wallet
            AccountMeta(P(wsol), False, True), AccountMeta(P(me), False, True), AccountMeta(P(me), True, False)])

    def _instructions(self, st: dict, side: str, amount_in: int, min_out: int, bump: int = 0) -> list:
        from solders.compute_budget import (
            set_compute_unit_limit,
            set_compute_unit_price,
        )
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey
        from solders.system_program import TransferParams, transfer

        P, me = Pubkey.from_string, self.owner
        wsol = ata(me, TOKEN, WSOL)
        ixs = [set_compute_unit_limit(CU_LIMIT),
               # bump distinguishes otherwise-identical clips (same wallet, amount, accounts) so each is its own tx
               set_compute_unit_price(self._priority_micro([st["addr"], PROGRAM]) + bump),
               self._create_ata_ix(wsol, WSOL, TOKEN)]
        if side == "buy":
            token = st["pool"]["token_mint"]
            ixs += [self._create_ata_ix(ata(me, st["token_prog"], token), token, st["token_prog"]),
                    transfer(TransferParams(from_pubkey=P(me), to_pubkey=P(wsol), lamports=amount_in)),
                    Instruction(P(TOKEN), bytes([17]), [AccountMeta(P(wsol), False, True)])]  # SyncNative
        ixs += [self._swap_ix(st, side, amount_in, min_out), self._close_wsol_ix()]
        return ixs

    def _feint_instructions(self, st: dict, lamports: int, buy_min_out: int) -> list:
        """One atomic transaction: buy `lamports` of the token, then sell those (guaranteed-minimum) tokens straight
        back, so the pool ends ~unchanged and we never hold the position. The operator sees a buy swap; we carry nothing
        into a pull. A token sliver (actual received minus the guaranteed minimum) is left as dust."""
        from solders.compute_budget import (
            set_compute_unit_limit,
            set_compute_unit_price,
        )
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey
        from solders.system_program import TransferParams, transfer

        P, me = Pubkey.from_string, self.owner
        token = st["pool"]["token_mint"]
        wsol = ata(me, TOKEN, WSOL)
        return [set_compute_unit_limit(FEINT_CU_LIMIT),
                set_compute_unit_price(self._priority_micro([st["addr"], PROGRAM], FEINT_CU_LIMIT)),
                self._create_ata_ix(wsol, WSOL, TOKEN),
                self._create_ata_ix(ata(me, st["token_prog"], token), token, st["token_prog"]),
                transfer(TransferParams(from_pubkey=P(me), to_pubkey=P(wsol), lamports=lamports)),
                Instruction(P(TOKEN), bytes([17]), [AccountMeta(P(wsol), False, True)]),  # SyncNative
                self._swap_ix(st, "buy", lamports, buy_min_out),  # SOL -> tokens (at least buy_min_out)
                self._swap_ix(st, "sell", buy_min_out, 1),  # tokens -> SOL, min out 1 so it can't revert on slippage
                self._close_wsol_ix()]

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

    def _simulate_ixs(self, ixs: list) -> tuple[bool, str] | None:
        """Simulate these instructions on-chain and send NOTHING (sigVerify off, blockhash replaced). (ok, note):
        ok True = every leg executed, False = a leg reverted (e.g. a honeypot's sell). None = couldn't simulate."""
        from solders.hash import Hash
        from solders.message import MessageV0
        from solders.transaction import VersionedTransaction

        latest = (self._rpc("getLatestBlockhash", [{"commitment": "confirmed"}]) or {}).get("value") or {}
        if not latest.get("blockhash"):
            return None
        msg = MessageV0.try_compile(self.keypair.pubkey(), ixs, [], Hash.from_string(latest["blockhash"]))
        raw = base64.b64encode(bytes(VersionedTransaction(msg, [self.keypair]))).decode()
        res = self._rpc("simulateTransaction",
                        [raw, {"encoding": "base64", "replaceRecentBlockhash": True, "sigVerify": False}])
        if not isinstance(res, dict):
            return None
        err = (res.get("value") or {}).get("err")
        return (err is None, "ok" if err is None else str(err)[:100])

    def round_trip(self, mint: str, usd: float) -> tuple[float, str] | None:
        """Direct-route honeypot / sell-route gate, independent of Jupiter indexing the pool: simulate a buy of `usd`
        then a sell straight back on the actual pumpswap pool (nothing is sent). Returns (round_trip_loss_pct, note)
        when the sim succeeds (the pool can be bought AND sold), (100.0, 'honeypot ...') when a leg reverts, or None
        when the pool can't be read or simulated, so the caller skips the buy to be safe. Mirrors Jupiter.round_trip."""
        sol_usd = self.jup.sol_price()
        if not sol_usd:
            return None
        st = self._state(mint)
        if st is None:
            return None
        base_amt, quote_amt = st["reserves"]
        sol0 = quote_amt if st["token_is_base"] else base_amt
        tok_res = base_amt if st["token_is_base"] else quote_amt
        lamports = int(usd / sol_usd * 1e9)
        if not sol0 or lamports <= 0:
            return None
        tokens = amount_out(lamports, sol0, tok_res, st["fee_bps"])
        fslip = getattr(self.cfg, "launch_canary_feint_slippage_bps", 0) or self.cfg.slippage_bps
        buy_min_out = int(tokens * (10_000 - fslip) / 10_000)
        if buy_min_out <= 0:
            return 100.0, "pool has no tokens"
        sim = self._simulate_ixs(self._feint_instructions(st, lamports, buy_min_out))
        if sim is None:
            return None
        ok, note = sim
        if not ok:
            return 100.0, f"honeypot (sim: {note})"
        sol_back = amount_out(tokens, tok_res - tokens, sol0 + lamports, st["fee_bps"])  # sell the tokens back out
        return round((1 - sol_back / lamports) * 100, 2), "ok"

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
        """One direct swap, or, when `launch_canary_clips` > 1, that many equal back-to-back swaps into the same pool
        (a $5 buy as 5 x $1), accumulated into one position. The clips test whether the running total of our capital
        triggers the pull (5 x $1 rugs like $5) or each small buy is judged on its own (5 x $1 stays at the $1 rate)."""
        clips = max(1, int(getattr(self.cfg, "launch_canary_clips", 1) or 1))
        if clips <= 1:
            return self._buy_once(token, usd, ref_price)
        fills: list[Fill] = []
        last = None
        for i in range(clips):  # no deliberate spacing; each clip confirms before the next (a few seconds of network latency)
            f = self._buy_once(token, usd / clips, ref_price, bump=i)
            last = f
            if f is None or not getattr(f, "ok", False):
                break
            fills.append(f)
        if not fills:  # nothing landed: propagate the failure/decline from the first clip
            return last
        qty = sum(f.qty for f in fills)
        fee = sum(f.fee_usd for f in fills)
        usd_net = sum(f.usd for f in fills)
        self.last_note = f"{len(fills)}/{clips} clips" + ("" if len(fills) == clips else f" ({self.last_note})")
        return Fill(qty=qty, price=(usd_net - fee) / qty if qty else 0.0, usd=usd_net, fee_usd=fee,
                    tx=",".join(f.tx for f in fills if f.tx), note=self.last_note)

    def _buy_once(self, token: str, usd: float, ref_price: float, bump: int = 0) -> Fill | None:
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
        sent = self._send_ixs(self._instructions(st, "buy", lamports, min_out, bump))
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

    def feint(self, token: str, usd: float, commit: bool = True) -> str:
        """Probe before committing (11-12). `commit` is only meaningful under wallet rotation (see RotatingExecutor). poke the pool with a $usd buy and sell it straight back, then re-read the
        pool and report whether its liquidity was pulled in reaction. Returns 'alive' (commit), 'pulled' (skip) or
        'inconclusive' (the probe could not be placed; skip to be safe). `launch_canary_feint_hold_s` = 0 does one
        atomic buy+sell tx (never hold, ~fee-only cost, but invisible to an operator bot that watches holdings rather
        than the trade stream); > 0 buys, holds that long, then sells (visible, but a pull in the gap costs the $usd).
        The result is a signal only: `_sell_excess`-style dust may remain, and the real commit is a separate buy()."""
        self.last_feint = {}
        sol_usd = self.jup.sol_price()
        if not sol_usd:
            self.last_note = "feint: SOL price unknown"
            return "inconclusive"
        st = self._state(token)
        if st is None:
            return "inconclusive"
        base_amt, quote_amt = st["reserves"]
        sol0 = quote_amt if st["token_is_base"] else base_amt
        tok_res = base_amt if st["token_is_base"] else quote_amt
        lamports = int(usd / sol_usd * 1e9)
        sol = self.sol_balance()
        if sol is None or sol < lamports / 1e9 + MIN_SOL:
            self.last_note = "feint: wallet SOL low"
            return "inconclusive"
        if not sol0 or lamports / (sol0 + lamports) * 100 > self.cfg.max_price_impact_pct:
            self.last_note = "feint: impact too high"
            return "inconclusive"
        # The sell leg sells exactly buy_min_out, so a tight feint slippage = less dust left as tokens (a cheaper
        # feint). The buy still succeeds as long as its output clears buy_min_out; if the pool moved, it reverts (a
        # free inconclusive). Falls back to the general slippage when the feint slippage is 0.
        fslip = getattr(self.cfg, "launch_canary_feint_slippage_bps", 0) or self.cfg.slippage_bps
        buy_min_out = int(amount_out(lamports, sol0, tok_res, st["fee_bps"]) * (10_000 - fslip) / 10_000)
        if buy_min_out <= 0:
            self.last_note = "feint: pool has no tokens"
            return "inconclusive"
        hold = float(getattr(self.cfg, "launch_canary_feint_hold_s", 0.0) or 0.0)
        if hold <= 0:
            sent = self._send_ixs(self._feint_instructions(st, lamports, buy_min_out))
            if sent.startswith(NOT_SENT):
                self.last_note = f"feint: {sent}"
                return "inconclusive"
            self._settle_direct(sent, token)  # the P&L doesn't matter; the pool state after is the signal
        else:
            buy = self._buy_once(token, usd, 0.0)
            if buy is None or not getattr(buy, "ok", False):
                self.last_note = f"feint: buy did not land ({self.last_note})"
                return "inconclusive"
            self.sleep(hold)
            self.sell(token, buy.qty, 0.0)  # best effort; a pull in the gap returns ~0 and still shows in the re-read
        self.sleep(float(getattr(self.cfg, "launch_canary_feint_wait_s", 4.0) or 4.0))
        st1 = self._state(token)
        drop = float(getattr(self.cfg, "launch_canary_feint_drop_pct", 50.0) or 50.0)
        if st1 is None:
            self.last_feint = {"pool": st["addr"], "sol_before": sol0 / 1e9, "sol_after": None}
            return "pulled"
        b1, q1 = st1["reserves"]
        sol1 = q1 if st1["token_is_base"] else b1
        self.last_feint = {"pool": st["addr"], "sol_before": sol0 / 1e9, "sol_after": sol1 / 1e9}
        return "pulled" if sol1 < sol0 * (1 - drop / 100) else "alive"

    def pool_sol(self, mint: str) -> float | None:
        """The pool's current SOL reserve (whole SOL), or None if it can't be read. Used by the probe survival watch
        to tell a pull (SOL drained) from a survivor."""
        st = self._state(mint)
        if st is None:
            return None
        base_amt, quote_amt = st["reserves"]
        return (quote_amt if st["token_is_base"] else base_amt) / 1e9

    def _wait_visible(self, sig: str, timeout_s: float) -> tuple[bool, int | None]:
        """Poll until the transaction is seen on-chain. (ok, slot): ok False = reverted or never appeared.
        A dependent sell can't be built until the buy's tokens exist, so we fire the sell the moment the buy is
        visible (processed) — the sell then lands in the next slot at the earliest."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            res = self._rpc("getSignatureStatuses", [[sig], {"searchTransactionHistory": False}])
            val = ((res or {}).get("value") or [None])[0]
            if val is not None:
                return val.get("err") is None, val.get("slot")
            self.sleep(PROBE_POLL_S)
        return False, None

    def probe_1slot(self, token: str, usd: float) -> dict:
        """Two-transaction probe (11-13, measurement): buy `usd` of the token (slot N) so committed money LANDS in the
        pool, then sell the buy's guaranteed-minimum tokens straight back in the next slot (slot N+1). Landing money and
        pulling it out one slot later is what trips a size-reacting rugger (the atomic feint never lands money, so it
        never trips one) while our sell escapes the pull — 100% of the pulls we've measured land >=2 slots after the
        buy. A free honeypot simulation runs first, so a blocked sell is caught before a cent is committed. This opens
        NO position and decides no rugger verdict; it returns the mechanical outcome (buy/sell slots, whether we got the
        stake back). The caller then watches the pool for ~48s: a pull after this = a rugger we tripped and dodged;
        survival = a legitimate pool to enter for real."""
        out = {"token": token, "usd": usd, "verdict": "skipped", "note": "", "pool": self.pools.get(token)}
        rt = self.round_trip(token, usd)  # free sim: nothing is sent
        if rt is None:
            out["note"] = self.last_note = "probe: pool unreadable / not simulable"
            return out
        loss, note = rt
        if loss >= 100.0:
            out["verdict"], out["note"] = "honeypot", f"probe: {note}"
            self.last_note = out["note"]
            return out
        sol_usd = self.jup.sol_price()
        st = self._state(token)
        if not sol_usd or st is None:
            out["note"] = self.last_note = "probe: state unreadable"
            return out
        out["pool"] = st["addr"]
        base_amt, quote_amt = st["reserves"]
        sol_res, tok_res = (quote_amt, base_amt) if st["token_is_base"] else (base_amt, quote_amt)
        out["sol_before"] = round(sol_res / 1e9, 6)  # the pool's SOL at the probe, so the survival watch can see a pull
        lamports = int(usd / sol_usd * 1e9)
        bal = self.sol_balance()
        if bal is None or bal < lamports / 1e9 + MIN_SOL:
            out["note"] = self.last_note = "probe: wallet SOL low"
            return out
        if not sol_res or lamports / (sol_res + lamports) * 100 > self.cfg.max_price_impact_pct:
            out["note"] = self.last_note = "probe: impact too high"
            return out
        # The pre-signed sell sells exactly buy_min_out — the floor we're certain the buy delivers — so it needs no
        # read of the real balance and can go the instant the buy is visible; the sliver above the floor is dust.
        buy_min_out = int(amount_out(lamports, sol_res, tok_res, st["fee_bps"]) * (10_000 - self.cfg.slippage_bps) / 10_000)
        if buy_min_out <= 0:
            out["note"] = self.last_note = "probe: pool has no tokens"
            return out
        buy_sig = self._send_ixs(self._instructions(st, "buy", lamports, buy_min_out))
        if buy_sig.startswith(NOT_SENT):
            out["note"] = self.last_note = f"probe: buy not sent ({buy_sig})"
            return out
        out["buy_tx"] = buy_sig
        buy_ok, buy_slot = self._wait_visible(buy_sig, PROBE_SLOT_WAIT_S)
        settle_buy = None
        if not buy_ok:
            # Not visible inside the fast window: resolve it (resend + confirm). If it truly reverted / never landed,
            # no money landed and there is nothing to sell. If it landed slowly, the fast N+1 is lost but we still sell
            # the floor to get the stake out, rather than leave it exposed.
            settle_buy = self._settle_direct(buy_sig, token)
            if not (settle_buy["ok"] and settle_buy["token"] > 0):
                out["verdict"], out["note"] = "buy_failed", "probe: buy reverted / not visible"
                self.last_note = out["note"]
                return out
            buy_slot = None  # the buy landed late; its slot is no longer the thing we're racing
        sell_sig = self._send_ixs(self._instructions(st, "sell", buy_min_out, 1))  # min out 1: recover whatever is there
        out["sell_tx"] = None if sell_sig.startswith(NOT_SENT) else sell_sig
        settle_buy = settle_buy or self._settle_direct(buy_sig, token)
        settle_sell = self._settle_direct(sell_sig, token) if out["sell_tx"] else {"ok": False, "swap_lamports": 0,
                                                                                   "fee_lamports": 0, "note": sell_sig}
        spent = -settle_buy["swap_lamports"] / 1e9
        recovered = settle_sell["swap_lamports"] / 1e9 if settle_sell["ok"] else 0.0
        fees = (settle_buy["fee_lamports"] + settle_sell["fee_lamports"]) / 1e9
        out["buy_slot"], sell_tx = buy_slot, self._fetch_tx(sell_sig) if out["sell_tx"] else None
        out["sell_slot"] = (sell_tx or {}).get("slot")
        out["slot_gap"] = (out["sell_slot"] - buy_slot) if (out["sell_slot"] is not None and buy_slot is not None) else None
        out["spent_sol"], out["recovered_sol"] = round(spent, 6), round(recovered, 6)
        out["net_usd"] = round((recovered - spent - fees) * sol_usd, 4)
        out["escaped"] = settle_sell["ok"] and recovered > 0
        # "ok": money landed and we got it back out (a clean probe, cost ~fees). "trapped": the buy landed but the sell
        # did not recover — a dynamic honeypot, or a pull that beat our next-slot sell — so the stake is stuck in the
        # position and the engine must keep trying to exit it.
        out["verdict"] = "ok" if out["escaped"] else "trapped"
        out["note"] = self.last_note = "ok" if out["escaped"] else f"probe: sell did not recover ({settle_sell.get('note')})"
        return out

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
        if st is None:  # no pool to sell into: the engine marks the exit blocked and retries every few minutes
            return failed(0.0, f"no route ({self.last_note})")
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

    def sweep(self, keep: set = frozenset(), apply: bool = True) -> dict:
        """Reclaim the SOL locked in this wallet's leftover token accounts. Each feint leaves a dust token balance in
        a fresh ATA (it sells only the buy's guaranteed minimum), so the base sweep's empty-account close never frees
        it. Here each dust balance is first sold back into its pool on the direct route (a dead/pulled pool returns
        'no route' unsent, at no cost); a full sale also closes the emptied account and returns its rent. Then the
        base sweep closes any still-empty accounts. Best effort: a failure on one token never stops the rest."""
        base = super().sweep(keep, apply=False)  # list held dust + empty accounts without writing
        sold, usd = 0, 0.0
        if apply:
            for h in base.get("holding_tokens", []):
                mint = h.get("mint")
                if not mint or mint in keep:  # an open position's token: leave it alone
                    continue
                try:
                    fill = self.sell(mint, SWEEP_QTY, 0.0)  # qty >= balance -> sells the whole balance and closes it
                    if fill is not None and getattr(fill, "ok", False):
                        sold += 1
                        usd += fill.usd
                except Exception:  # noqa: BLE001 (a sweep must never disturb trading)
                    log.warning("dust sweep of %s failed", mint, exc_info=True)
        res = super().sweep(keep, apply)  # close any now-empty accounts (incl. ones the dust sale emptied): rent back
        res["dust_sold"], res["dust_usd"] = sold, round(usd, 4)
        return res


def _token_amount(acct: dict | None) -> int:
    if not acct:
        return 0
    raw = base64.b64decode(acct["data"][0])
    return int.from_bytes(raw[64:72], "little") if len(raw) >= 72 else 0
