"""LocalShred consumer: transactions -> direct top-level pumpswap swaps -> per-pool per-slot
aggregate -> frontrun_live.Runner + precursor_gate -> fires jsonl.

This is the box's DIRECT decode (parity with absfast_tight's front_swap/front_feed/front_finalize)
moved into the repo so any transaction source can drive it -- the LocalShred entry-service gRPC tap on
the Nuremberg box (SHRED_DECODER.md "Nuremberg measurement runbook"), or a replay. It decodes ONLY
top-level direct pumpswap swaps -- the ~89% of the edge the vis-experiment kept -- and emits the SAME
fires jsonl the box writes to front-fires.jsonl ({pool, slot, eta, sym, m2, tier, token_mint}), so
tradingbot.frontrun_verify checks this decode against the Helius ground truth exactly as it checks the
box. That is the measurement gate: does the complete LocalShred feed reproduce the study where the
current reassembled feed (box-exact DIVERGED, 2026-11-22) does NOT.

The precursor gate is price- AND magnitude-free (all 18 predictors are sym/m2/m3 wallet-setup
motifs), so the per-slot aggregate needs only n_buys, n_sells and the ordered buy-side fee payers;
price / buy_sol / sell_sol / max_buy are fed 0.0 (harmless -- the box feeds them 0 too).

Direction = the box's post-fix orientation flip: a pumpswap swap instruction maps to a TOKEN buy/sell
through the pool orientation (token_is_base). On a SOL-as-base pool a token BUY arrives as the `sell`
instruction, so token_buy = (inst_buy == token_is_base). base_mint is the swap instruction's account
index 3, quote_mint index 4 (tradingbot.execution.pumpswap_direct.swap_accounts); token_is_base =
(base_mint != WSOL).

What is NOT here (the box-side I/O adapters -- see the runbook): the gRPC client of the proxy's
--grpc-service-port entry service [proto to confirm from dedicatednodes/shredstream-proxy], and the
solders/solana-py parse of each VersionedTransaction into the (program_id, account_indices, data)
view `decode_tx` takes. This module is the decode + aggregate + decision CORE, unit-tested without
either, so the adapters are a thin, box-local layer over a tested core.
"""
import json
import time
from typing import Callable, NamedTuple, Optional

try:                                  # in-package and standalone-on-the-box both work
    from . import frontrun_live, precursor_gate
except ImportError:                   # noqa: E722 -- box copy sits beside the two modules
    import frontrun_live
    import precursor_gate

PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"   # pumpswap (== pumpswap_direct.PROGRAM / helius.PUMPSWAP_PROGRAM)
WSOL = "So11111111111111111111111111111111111111112"

SELL = bytes([51, 230, 133, 164, 1, 127, 131, 173])        # pump_amm `sell` (== pumpswap_direct.SELL)
BUY_EXACT_QUOTE_IN = bytes([198, 46, 21, 82, 180, 217, 232, 112])  # pump_amm `buy_exact_quote_in`
BUY_BASE_OUT = bytes([102, 6, 61, 18, 1, 218, 235, 234])   # pump_amm `buy` (buy_base_out) = anchor global:buy
# The box's front_swap decodes these THREE (sell + both buys). buy_base_out is the DOMINANT buy path;
# omitting it undercounts buys so nearly every slot reads as net-sellers (sym 'S' -> a .S/BS flood with no
# new-wallet 'N' motifs -- the 2026-11-22 live symptom). The rarer variants buy_exact_in /
# sell_exact_quote_out the box does NOT decode, so they stay out too, keeping this an exact parity of the
# box's direct view (so frontrun-verify's only variable is the feed, not the discriminator set).
BUY_DISCS = frozenset({BUY_EXACT_QUOTE_IN, BUY_BASE_OUT})

DEFAULT_FLURRY_MIN = frontrun_live.DEFAULT_FLURRY_MIN


class Swap(NamedTuple):
    pool: str      # the pumpswap pool account (swap instruction account 0)
    token: str     # the non-WSOL mint of the pool (for the fires jsonl; frontrun_verify re-derives it anyway)
    is_buy: bool   # TOKEN buy (after the orientation flip), not the raw instruction kind
    payer: str     # tx fee payer (account_keys[0]) = the trading wallet, per the study


def classify_instruction(program_id: str, accts, data, account_keys, fee_payer) -> Optional[Swap]:
    """One top-level instruction -> a Swap, or None if it is not a direct pumpswap swap we decode.

    accts is the instruction's account-index list into account_keys; data is its instruction bytes
    (first 8 = Anchor discriminator). Returns None (never raises) on anything malformed."""
    if program_id != PROGRAM:
        return None
    try:
        if len(data) < 8:
            return None
        disc = bytes(data[:8])
        if disc == SELL:
            inst_buy = False
        elif disc in BUY_DISCS:
            inst_buy = True
        else:
            return None
        if len(accts) < 5:                      # need pool (0) + base_mint (3) + quote_mint (4)
            return None
        pool = account_keys[accts[0]]
        base_mint = account_keys[accts[3]]
        quote_mint = account_keys[accts[4]]
    except (IndexError, TypeError, KeyError):
        return None
    base_is_sol = base_mint == WSOL
    quote_is_sol = quote_mint == WSOL
    if base_is_sol == quote_is_sol:              # need EXACTLY one WSOL side: a SOL-paired pool (box filter)
        return None                              # excludes USDC-paired and other non-SOL pools
    token_is_base = not base_is_sol              # the token is the non-WSOL mint
    token_buy = (inst_buy == token_is_base)      # the box flip: a SOL-base pool's token BUY is the `sell` ix
    token = base_mint if token_is_base else quote_mint
    return Swap(pool=pool, token=token, is_buy=token_buy, payer=fee_payer)


def decode_tx(account_keys, instructions, fee_payer=None):
    """Decode a transaction's TOP-LEVEL instructions into Swaps. `instructions` is a list of
    (program_id, account_indices, data); the caller (the box-side adapter) resolves each instruction's
    program-id index and passes only top-level instructions (the box's direct view excludes CPIs).
    fee_payer defaults to account_keys[0]."""
    if fee_payer is None:
        fee_payer = account_keys[0] if account_keys else None
    out = []
    for ins in instructions:
        program_id, accts, data = ins
        sw = classify_instruction(program_id, accts, data, account_keys, fee_payer)
        if sw is not None:
            out.append(sw)
    return out


class _PoolState:
    __slots__ = ("slot", "token", "n_buys", "n_sells", "buyers", "runner", "first_slot")

    def __init__(self, runner):
        self.slot = None
        self.token = None
        self.n_buys = 0
        self.n_sells = 0
        self.buyers = []
        self.runner = runner
        self.first_slot = None        # first slot we saw any swap for this pool (~creation for a fresh pool)


class Consumer:
    """Route decoded swaps into one frontrun_live.Runner PER POOL (each pool has its own wallet
    history), accumulate each pool's current active slot, and finalize (push to the runner -> gate) when
    that pool next trades in a later slot. On a gate fire, call on_fire(signal) with the box-jsonl fields
    attached. `runner_factory` is for tests; by default it builds a real Runner from the gate."""

    def __init__(self, gate, tier: str = "1", flurry_min: int = DEFAULT_FLURRY_MIN,
                 on_fire: Optional[Callable[[dict], None]] = None, runner_factory=None,
                 fresh_window_slots: int = 0, on_capture: Optional[Callable] = None):
        self.gate = gate
        self.tier = str(tier)
        self.flurry_min = flurry_min
        self.on_fire = on_fire
        # on_capture(pool, token, slot_rec) is called for EVERY finalized slot (not just fires): the per-slot
        # feature aggregate our decode produces. --capture mode writes these to fit the study on our own feed.
        self.on_capture = on_capture
        self._make_runner = runner_factory or (lambda: frontrun_live.Runner(gate, self.tier, flurry_min))
        self.pools: dict = {}
        # Freshness gate (the box's front_elig / pool_fresh equivalent). The precursor gate is fit only on
        # FRESH launch pools (first ~180s from creation); scoring mid-life pools is out-of-distribution and
        # floods on SB/BS/BB churn (the 2026-11-21 box bug), AND a mid-life fire diverges in frontrun-verify
        # for the wrong reason (our per-pool history starts mid-life, the Helius ground truth from creation).
        # So fire only on a pool that FIRST appeared after a warm-up of fresh_window_slots (confidence we
        # caught it from its start, not mid-life) and within fresh_window_slots of that first appearance.
        # 0 disables it (the raw core, for tests/replay); run_localshred.py sets ~450 (~180s at 0.4s/slot).
        self.fresh_window_slots = int(fresh_window_slots)
        self._start_slot = None

    def _slot_rec(self, st: _PoolState) -> dict:
        # price/sol fields are 0.0: the precursor gate is price- and magnitude-free (sym/m2/m3 only).
        return {"slot": st.slot, "price": 0.0, "n_buys": st.n_buys, "buy_sol": 0.0, "max_buy": 0.0,
                "n_sells": st.n_sells, "sell_sol": 0.0, "buyers": list(st.buyers), "net_sol": 0.0}

    def _fresh_ok(self, signal_slot: int, st: _PoolState) -> bool:
        """True when a fire on `signal_slot` is on a genuinely fresh pool (see __init__). Window 0 = off."""
        w = self.fresh_window_slots
        if w <= 0:
            return True
        if self._start_slot is None or st.first_slot is None:
            return False
        if (st.first_slot - self._start_slot) < w:        # pool first seen during warm-up -> maybe mid-life
            return False
        return (signal_slot - st.first_slot) <= w          # and the fire is within the fresh window

    def _flush(self, pool: str, st: _PoolState) -> Optional[dict]:
        if st.slot is None:
            return None
        signal_slot = st.slot
        rec = self._slot_rec(st)
        if self.on_capture:
            self.on_capture(pool, st.token, rec)           # record the per-slot feature aggregate (for --capture)
        sig = st.runner.on_slot(rec)                        # advance the bank (keep tracking) even if suppressed
        st.slot = None
        st.n_buys = 0
        st.n_sells = 0
        st.buyers = []
        if not sig or not self._fresh_ok(signal_slot, st):
            return None
        sig = dict(sig)
        sig["pool"] = pool
        sig["slot"] = sig["signal_slot"]
        if st.token:
            sig["token_mint"] = st.token
        if self.on_fire:
            self.on_fire(sig)
        return sig

    def on_swap(self, slot: int, pool: str, token: Optional[str], is_buy: bool, payer) -> Optional[dict]:
        if self._start_slot is None:
            self._start_slot = slot
        st = self.pools.get(pool)
        if st is None:
            st = _PoolState(self._make_runner())
            st.first_slot = slot
            self.pools[pool] = st
        fired = None
        if st.slot is not None and slot != st.slot:      # pool advanced -> close the active slot it held
            fired = self._flush(pool, st)
        if st.slot is None:
            st.slot = slot
        if token and not st.token:
            st.token = token
        if is_buy:
            st.n_buys += 1
            st.buyers.append(payer)
        else:
            st.n_sells += 1
        return fired

    def on_tx(self, slot: int, account_keys, instructions, fee_payer=None) -> list:
        fires = []
        for sw in decode_tx(account_keys, instructions, fee_payer):
            f = self.on_swap(slot, sw.pool, sw.token, sw.is_buy, sw.payer)
            if f:
                fires.append(f)
        return fires

    def flush_all(self) -> list:
        """Finalize every pool's held slot. For a bounded replay; live trading instead sweeps all pools
        at each global N->N+1 boundary so the N+1 entry window isn't missed (box-side concern)."""
        fires = []
        for pool, st in list(self.pools.items()):
            f = self._flush(pool, st)
            if f:
                fires.append(f)
        return fires


def fire_line(sig: dict) -> dict:
    """The box's front-fires.jsonl record for one fire."""
    rec = {"pool": sig.get("pool"), "slot": sig.get("signal_slot", sig.get("slot")),
           "eta": sig.get("eta"), "sym": sig.get("sym"), "m2": sig.get("m2"), "tier": sig.get("tier")}
    if sig.get("token_mint"):
        rec["token_mint"] = sig["token_mint"]
    return rec


def jsonl_writer(path: str) -> Callable[[dict], None]:
    """An on_fire callback that appends one fire_line per fire to `path` (the shape frontrun-verify reads)."""
    def _write(sig):
        with open(path, "a") as fh:
            fh.write(json.dumps(fire_line(sig)) + "\n")
    return _write


def load_gate(path: str):
    return precursor_gate.load(path)


class CaptureSink:
    """Collects the per-slot feature aggregates (Consumer.on_capture) of FRESH pools and emits one feature rec
    per pool: {pool, token_mint, create_ts, slots:[[sec, slot, n_buys, n_sells, buyers], ...]}. These are OUR
    decode's features; localshred-merge backfills the price path from Helius -> flurry-study --flows fits the gate
    on our own feed (train = serve). A pool first seen during the warm-up is skipped (it may be mid-life, so its
    history doesn't start at creation); each pool is trimmed to its first `span_s` and emitted when it ages out."""

    def __init__(self, write: Callable[[dict], None], span_s: float = 180.0, warm_s: float = 180.0,
                 min_slots: int = 3, clock: Optional[Callable[[], float]] = None,
                 wall: Optional[Callable[[], float]] = None):
        self.write = write
        self.span_s = span_s
        self.warm_s = warm_s
        self.min_slots = min_slots
        self._clock = clock or time.monotonic
        self._wall = wall or time.time
        self.start = self._clock()
        self.pools: dict = {}
        self.written = 0

    def on_capture(self, pool: str, token: Optional[str], rec: dict) -> None:
        now = self._clock()
        p = self.pools.get(pool)
        if p is None:
            if now - self.start < self.warm_s:              # warm-up: skip pools first seen early (maybe mid-life)
                return
            p = self.pools[pool] = {"first": now, "create_ts": self._wall(), "token": token, "slots": []}
        if token and not p["token"]:
            p["token"] = token
        if now - p["first"] <= self.span_s:
            p["slots"].append([round(now - p["first"], 2), rec["slot"],
                               rec["n_buys"], rec["n_sells"], list(rec["buyers"])])
        for done in [k for k, q in self.pools.items() if now - q["first"] > self.span_s + 5]:
            self._emit(done, self.pools.pop(done))

    def _emit(self, pool: str, p: dict) -> None:
        if len(p["slots"]) < self.min_slots or not p["token"]:
            return
        self.write({"pool": pool, "token_mint": p["token"], "create_ts": p["create_ts"], "slots": p["slots"]})
        self.written += 1

    def close(self) -> None:
        for pool in list(self.pools):
            self._emit(pool, self.pools.pop(pool))
