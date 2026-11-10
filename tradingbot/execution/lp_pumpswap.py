"""Live executor that enters a pumpswap pool as a liquidity provider instead of a buyer (11-10, real money).

The launch canary's operators pull by withdrawing their own LP (on-chain: `Instruction: Withdraw`, confirmed on the
11-10 pulls). A withdrawal is proportional: it takes the operator's share of both reserves, which zeroes a *buyer*
(who holds only the token) but leaves a fellow *LP*'s share untouched. So this executor makes the canary an LP:

  buy(token, usd):  one transaction that wraps SOL, buys ~half the budget of tokens on the pool, then DEPOSITS those
                    tokens plus the matching SOL as liquidity (we receive LP tokens). We end holding an LP position
                    worth ~`usd`, half exposed to SOL and half to the token. The deposit amounts are computed from the
                    reserves read at send time; `max_base`/`max_quote` cap them, so a shifted pool reverts (fee only)
                    rather than overspends.
  sell(token):      one transaction. WITHDRAW our LP (always returns our proportional share, even in a pool the
                    operator has drained) and, in the same transaction, sell the whole token side back to SOL -- both
                    the ~15% the buy left naked (already in the wallet, read exactly) and ~95% of the share the
                    withdraw will return (estimated under, so the sale never over-sells and reverts). If the pool
                    shifted hard or is dead and the sale leg reverts the whole transaction, it falls back to a
                    withdraw-only transaction so the SOL side is still recovered.

Layouts are pump.fun's IDL `deposit` / `withdraw` (15 accounts each, no fee recipients), checked against a real
mainnet withdraw on 11-10; a wrong account order reverts in the mainnet simulation before any money moves. Everything
else (RPC, sending, resending, confirmation) is the direct executor's.
"""
from __future__ import annotations

import base64
import logging

from tradingbot.data.pumpswap import PROGRAM, WSOL, parse_lp_mint
from tradingbot.execution.base import Fill, failed
from tradingbot.execution.pumpswap_direct import (
    ATA_PROGRAM,
    EVENT_AUTHORITY,
    GLOBAL_CONFIG,
    NOT_SENT,
    SYSTEM,
    TOKEN,
    TOKEN_2022,
    DirectPumpswapExecutor,
    _pick,
    amount_out,
    ata,
    plan,
    swap_accounts,
    swap_data,
)

log = logging.getLogger(__name__)

DEPOSIT = bytes.fromhex("f223c68952e1f2b6")
WITHDRAW = bytes.fromhex("b712469c946da122")
CU_LIMIT_LP = 500_000  # buy + deposit (or withdraw + sell) with up to 4 account creates; mainnet sim showed the real use
ROUTE = "lp"
DEPOSIT_TOKEN_SHARE = 0.95  # deposit this share of the tokens the buy is guaranteed to yield, leaving a dust margin
SLACK_LAMPORTS = 5_000_000  # extra wSOL wrapped so the deposit's SOL side never falls short


def lp_accounts(name: bytes, pool_addr: str, pool: dict, user: str, base_prog: str, quote_prog: str) -> list:
    """The 15 accounts pumpswap `deposit` / `withdraw` take, in IDL order (checked against a real mainnet withdraw)."""
    base, quote, lp = pool["base_mint"], pool["quote_mint"], pool["lp_mint"]
    return [(pool_addr, False, True), (GLOBAL_CONFIG, False, False), (user, True, False),
            (base, False, False), (quote, False, False), (lp, False, True),
            (ata(user, base_prog, base), False, True), (ata(user, quote_prog, quote), False, True),
            (ata(user, TOKEN_2022, lp), False, True), (pool["base_vault"], False, True), (pool["quote_vault"], False, True),
            (TOKEN, False, False), (TOKEN_2022, False, False), (EVENT_AUTHORITY, False, False), (PROGRAM, False, False)]


def lp_data(name: bytes, lp_amount: int, min_or_max_base: int, min_or_max_quote: int) -> bytes:
    le = lambda n: int(n).to_bytes(8, "little")
    return name + le(lp_amount) + le(min_or_max_base) + le(min_or_max_quote)


class LpPumpswapExecutor(DirectPumpswapExecutor):
    route = ROUTE

    def _lp_supply(self, lp_mint: str) -> tuple[int, int] | None:
        """(supply, decimals) of the pool's LP mint, read from its account (a Token-2022 mint)."""
        acct = self._accounts([lp_mint])[0]
        if not acct:
            return None
        info = parse_lp_mint(base64.b64decode(acct["data"][0]))
        return (info["supply"], info["decimals"]) if info else None

    def _meta_ixs(self, extra_accounts: list[str]):
        from solders.compute_budget import (
            set_compute_unit_limit,
            set_compute_unit_price,
        )
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey

        price = self._priority_micro(extra_accounts)
        me = self.owner
        ixs = [set_compute_unit_limit(CU_LIMIT_LP), set_compute_unit_price(price)]

        def create(mint: str, prog: str):
            owner_ata = ata(me, prog, mint)
            return Instruction(Pubkey.from_string(ATA_PROGRAM), bytes([1]), [  # createIdempotent
                AccountMeta(Pubkey.from_string(me), True, True), AccountMeta(Pubkey.from_string(owner_ata), False, True),
                AccountMeta(Pubkey.from_string(me), False, False), AccountMeta(Pubkey.from_string(mint), False, False),
                AccountMeta(Pubkey.from_string(SYSTEM), False, False), AccountMeta(Pubkey.from_string(prog), False, False)])

        return ixs, create

    def _lp_ix(self, name: bytes, st: dict, lp_amount: int, base_amt: int, quote_amt: int):
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey

        accs = lp_accounts(name, st["addr"], st["pool"], self.owner, st["base_prog"], st["quote_prog"])
        return Instruction(Pubkey.from_string(PROGRAM), lp_data(name, lp_amount, base_amt, quote_amt),
                           [AccountMeta(Pubkey.from_string(a), s, w) for a, s, w in accs])

    def _swap_ix(self, st: dict, side: str, amount_in: int, min_out: int):
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey

        name = plan(side, st["token_is_base"])
        accs = swap_accounts(name, st["addr"], st["pool"], self.owner, st["base_prog"], st["quote_prog"],
                             _pick(st["global"]["protocol_recipients"], st["addr"]),
                             _pick(st["global"]["buyback_recipients"], self.owner), st["pump_coin"])
        return Instruction(Pubkey.from_string(PROGRAM), swap_data(name, amount_in, min_out),
                           [AccountMeta(Pubkey.from_string(a), s, w) for a, s, w in accs])

    def _wrap_ixs(self, lamports: int):
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey
        from solders.system_program import TransferParams, transfer

        wsol = ata(self.owner, TOKEN, WSOL)
        return [transfer(TransferParams(from_pubkey=Pubkey.from_string(self.owner), to_pubkey=Pubkey.from_string(wsol),
                                        lamports=lamports)),
                Instruction(Pubkey.from_string(TOKEN), bytes([17]), [AccountMeta(Pubkey.from_string(wsol), False, True)])]

    def _close_wsol_ix(self):
        from solders.instruction import AccountMeta, Instruction
        from solders.pubkey import Pubkey

        wsol = ata(self.owner, TOKEN, WSOL)
        me = Pubkey.from_string(self.owner)
        return Instruction(Pubkey.from_string(TOKEN), bytes([9]),
                           [AccountMeta(Pubkey.from_string(wsol), False, True), AccountMeta(me, False, True),
                            AccountMeta(me, True, False)])

    def _settle_lp(self, sig: str, token: str, lp_mint: str) -> dict:
        """{ok, note, owner_lamports (wallet SOL change: - on a buy, + on a sell), fee_lamports, lp (LP raw delta),
        token (token raw delta)}."""
        status = self._await(sig)
        tx = self._fetch_tx(sig) if status != "dropped" else None
        if tx is None:
            note = f"not confirmed (dropped) {sig}" if status == "dropped" else "confirmed but transaction not readable"
            return {"ok": False, "note": note, "owner_lamports": 0, "lp": 0, "token": 0,
                    "fee_lamports": 0 if status == "dropped" else int(self.priority_fee_sol * 1e9)}
        meta = tx.get("meta") or {}
        pre, post = meta.get("preBalances") or [0], meta.get("postBalances") or [0]
        owner = post[0] - pre[0]
        fee = int(meta.get("fee") or 0)
        if meta.get("err"):
            return {"ok": False, "note": f"reverted: {meta['err']}", "owner_lamports": owner, "lp": 0, "token": 0,
                    "fee_lamports": pre[0] - post[0]}
        return {"ok": True, "note": "ok", "owner_lamports": owner, "fee_lamports": fee,
                "lp": self._delta(meta, lp_mint), "token": self._delta(meta, token)}

    # --- public -------------------------------------------------------------------
    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        """Buy ~half the budget of tokens and deposit them + matching SOL as liquidity, in one transaction. The Fill's
        `qty` is the LP tokens received, `usd` the total SOL committed (cost)."""
        sol_usd = self.jup.sol_price()
        if not sol_usd:
            self.last_note = "SOL price unknown"
            return None
        decimals = self._decimals(token)
        if decimals is None:
            self.last_note = "token decimals unknown"
            return None
        st = self._state(token)
        if st is None:
            return None
        sup = self._lp_supply(st["pool"]["lp_mint"])
        if sup is None or sup[0] <= 0:
            self.last_note = "LP supply unreadable"
            return None
        lp_supply, lp_dec = sup
        budget = int(usd / sol_usd * 1e9)  # total lamports to commit (buy half + deposit's SOL half)
        sol = self.sol_balance()
        if sol is None or sol < budget / 1e9 + SLACK_LAMPORTS / 1e9 + 0.01:
            self.last_note = f"wallet SOL low ({sol if sol is not None else '?'}) for a {budget / 1e9:.4f} SOL LP"
            return None
        base_amt, quote_amt = st["reserves"]  # SOL-base pool: base = SOL (lamports), quote = token (raw)
        sol_res, tok_res = (quote_amt, base_amt) if st["token_is_base"] else (base_amt, quote_amt)
        s_buy = budget // 2
        impact = s_buy / (sol_res + s_buy) * 100 if sol_res else 100.0
        if impact > self.cfg.max_price_impact_pct:
            self.last_note = f"price impact {impact:.2f}%"
            return None
        expect_tok = amount_out(s_buy, sol_res, tok_res, st["fee_bps"])
        min_tok = int(expect_tok * 0.90)  # these pools move fast; wide buy slippage
        if min_tok <= 0:
            self.last_note = "pool has no tokens to sell"
            return None
        # Deposit the share of tokens the buy is guaranteed to yield, so the deposit never needs more than we hold.
        rq2 = max(1, tok_res - expect_tok)                       # token reserve after our buy (estimate)
        rb2 = sol_res + s_buy                                    # SOL reserve after our buy
        lp_out = int(DEPOSIT_TOKEN_SHARE * min_tok * lp_supply / rq2)
        if lp_out <= 0:
            self.last_note = "deposit too small"
            return None
        base_needed = lp_out * rb2 // lp_supply                  # SOL the deposit will pull in
        max_base = int(base_needed * 1.20)                       # cap (shifted pool reverts rather than overspends)
        wrap = s_buy + max_base + SLACK_LAMPORTS
        # token orientation for the deposit's (base, quote) amounts: base = SOL, quote = token on our pools
        if st["token_is_base"]:
            dep_base, dep_quote = min_tok, max_base              # base = token, quote = SOL
        else:
            dep_base, dep_quote = max_base, min_tok              # base = SOL, quote = token
        ixs, create = self._meta_ixs([st["addr"], PROGRAM])
        ixs += [create(WSOL, TOKEN), create(token, st["token_prog"]), create(st["pool"]["lp_mint"], TOKEN_2022)]
        ixs += self._wrap_ixs(wrap)
        ixs += [self._swap_ix(st, "buy", s_buy, min_tok),
                self._lp_ix(DEPOSIT, st, lp_out, dep_base, dep_quote),
                self._close_wsol_ix()]
        sent = self._send_ixs(ixs)
        if sent.startswith(NOT_SENT):
            self.last_note = sent
            return None
        res = self._settle_lp(sent, token, st["pool"]["lp_mint"])
        fee_usd = res["fee_lamports"] / 1e9 * sol_usd
        if not res["ok"]:
            self.last_note = res["note"]
            return failed(fee_usd, res["note"])
        lp_got, spent = res["lp"], -res["owner_lamports"]  # owner balance fell by what we committed
        if lp_got <= 0 or spent <= 0:
            self.last_note = "confirmed but no LP minted"
            return failed(fee_usd, self.last_note)
        qty = lp_got / 10**lp_dec
        cost_usd = spent / 1e9 * sol_usd
        self.last_note = "ok"
        return Fill(qty=qty, price=cost_usd / qty, usd=cost_usd, fee_usd=fee_usd, tx=sent, note="ok")

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        """Withdraw our whole LP and sell the whole token side back to SOL in one transaction (`qty` is ignored, we
        always withdraw our full LP balance). The sale covers the naked tokens the buy left in the wallet plus ~95% of
        the share the withdraw returns. If the atomic transaction can't land -- whether it was rejected in preflight
        (the sale leg reverts selling into a drained/empty pool, which is exactly the hardest rug) or it landed and
        reverted -- fall back to a withdraw-only transaction so the SOL side is still recovered."""
        sol_usd = self.jup.sol_price()
        if not sol_usd:
            return failed(0.0, f"{NOT_SENT} SOL price unknown")
        st = self._state(token)
        if st is None:
            return failed(0.0, f"no route ({self.last_note})")
        lp_mint = st["pool"]["lp_mint"]
        held_lp = self.token_balance(lp_mint)
        if held_lp <= 0:
            return failed(0.0, "no LP balance")
        # Atomic: withdraw the LP and sell the whole token side in the same transaction.
        sent = self._send_ixs(self._exit_ixs(st, token, held_lp, with_sell=True))
        # Fall back to withdraw-only when the atomic exit can't land. A fully-drained pool reverts the sale leg; with
        # preflight on that is rejected *before sending* (NOT_SENT), so the earlier code bailed here and the engine
        # silently retried forever -- the stuck-at-$0 bug on the worst rugs. Treat a preflight rejection the same as
        # an on-chain revert and withdraw our SOL share (withdraw-only has no sale leg, so it passes preflight).
        res = None if sent.startswith(NOT_SENT) else self._settle_lp(sent, token, lp_mint)
        if res is None or (not res["ok"] and res["note"].startswith("reverted")):
            log.info("lp atomic exit did not land (%s); withdrawing the SOL side only",
                     sent if res is None else res["note"])
            sent = self._send_ixs(self._exit_ixs(st, token, held_lp, with_sell=False))
            if sent.startswith(NOT_SENT):
                return failed(0.0, sent)
            res = self._settle_lp(sent, token, lp_mint)
        fee = res["fee_lamports"] / 1e9 * sol_usd
        if not res["ok"]:
            return failed(fee, res["note"])
        proceeds = res["owner_lamports"] / 1e9 * sol_usd  # wallet SOL rose by the net we recovered
        if proceeds <= 0:
            return failed(fee, "withdrew nothing")
        return Fill(qty=qty, price=proceeds / qty if qty else 0.0, usd=proceeds, fee_usd=fee, tx=sent, note="ok")

    def _exit_ixs(self, st: dict, token: str, held_lp: int, with_sell: bool) -> list:
        ixs, create = self._meta_ixs([st["addr"], PROGRAM])
        ixs += [create(WSOL, TOKEN), create(token, st["token_prog"])]
        ixs.append(self._lp_ix(WITHDRAW, st, held_lp, 0, 0))  # min base/quote 0: take our share whatever is left
        if with_sell:
            # Sell the naked tokens the buy left (in the wallet now, so exact) plus ~95% of the share the withdraw
            # returns (estimated under, so naked + 0.95*share stays <= our post-withdraw holdings and never reverts).
            naked = self.token_balance(token)
            base_amt, quote_amt = st["reserves"]
            tok_res = base_amt if st["token_is_base"] else quote_amt  # the token reserve, whichever side it is
            sup = self._lp_supply(st["pool"]["lp_mint"])
            if sup and sup[0] > 0:
                share = held_lp * tok_res // sup[0]      # tokens the withdraw will return (estimate)
                tok_back = naked + int(0.95 * share)
                if tok_back > 0:
                    ixs.append(self._swap_ix(st, "sell", tok_back, 1))
        ixs.append(self._close_wsol_ix())  # unwrap all recovered SOL back to the wallet
        return ixs
