"""Per-pool absolute reserve view for the box's execution path -- the data half of the WS reserve feed.

The front-run gate fires from shreds (wallet motifs), which carry no reserves. But the EXECUTOR needs the pool's
current reserves to size a buy and set its min_out floor, and the paper NET needs them to price the fill. Shreds
cannot give reserves (pre-execution transactions have no meta / post-balances), so the box runs a SEPARATE
`accountSubscribe` WS on each active pool's two vaults at `processed` commitment: every swap pushes the vault's new
absolute balance, which this tracker holds. The view is therefore at most ~1 slot stale (the in-flight slot) --
there is NO accumulation drift, because each push is the absolute post-execution balance, not a delta.

Split like the LocalShred consumer: this pure logic (decode a vault push, hold the latest (sol, tok, slot) per
pool) is repo-tested; the gRPC/WS I/O loop that feeds it lives on the box (run_localshred.py). Reserves are RAW
units -- sol = the WSOL vault's token amount (lamports), tok = the token vault's amount (raw units) -- so the price
is the unit-free ratio sol/tok, the same basis slot_flow / launch_record use; comparing the box view to Helius
truth needs no SOL price or decimals.
"""
import base64

from .data import pumpswap


def _as_bytes(data) -> bytes | None:
    """An accountSubscribe `value.data` is `[base64_str, "base64"]` (jsonParsed can differ); also accept a raw
    base64 str or bytes. Return the account's raw bytes, or None if it cannot be decoded."""
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, (list, tuple)) and data:
        data = data[0]                      # [b64, "base64"]
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:
            return None
    return None


def decode_vault_amount(data) -> int | None:
    """An accountSubscribe push's data payload -> the SPL/Token-2022 vault balance (u64 at byte 64)."""
    raw = _as_bytes(data)
    return pumpswap.token_amount(raw) if raw is not None else None


class ReserveTracker:
    """Latest absolute reserves per pool, fed by the box's accountSubscribe pushes. One tracker holds every active
    pool. `watch` registers a pool's two vault addresses so a push keyed by vault routes to the right pool + side;
    a push updates only its side with the slot it arrived on. `get` returns the pair stamped with the OLDER of the
    two sides' slots (the honest as-of), and None until both sides have been seen."""

    def __init__(self) -> None:
        self._sol: dict[str, tuple[int, int]] = {}          # pool -> (sol_lamports, slot)
        self._tok: dict[str, tuple[int, int]] = {}          # pool -> (token_units, slot)
        self._vault: dict[str, tuple[str, str]] = {}        # vault address -> (pool, "sol"|"tok")

    def watch(self, pool: str, sol_vault: str, token_vault: str) -> None:
        self._vault[sol_vault] = (pool, "sol")
        self._vault[token_vault] = (pool, "tok")

    def unwatch(self, pool: str, sol_vault: str, token_vault: str) -> None:
        self._vault.pop(sol_vault, None)
        self._vault.pop(token_vault, None)
        self._sol.pop(pool, None)
        self._tok.pop(pool, None)

    def vaults_for(self, pool: str) -> list[str]:
        return [v for v, (p, _s) in self._vault.items() if p == pool]

    def on_vault(self, vault: str, amount: int | None, slot: int) -> None:
        """A watched vault's new absolute balance at `slot`. Unknown vaults and None amounts are ignored."""
        vp = self._vault.get(vault)
        if vp is None or amount is None:
            return
        pool, side = vp
        (self._sol if side == "sol" else self._tok)[pool] = (int(amount), int(slot))

    def on_push(self, vault: str, data, slot: int) -> None:
        """Decode an accountSubscribe data payload for a watched vault and update it."""
        self.on_vault(vault, decode_vault_amount(data), slot)

    def get(self, pool: str) -> tuple[int, int, int] | None:
        """(sol_lamports, token_units, as_of_slot) or None if either side is unseen. as_of = min of the two slots."""
        s, t = self._sol.get(pool), self._tok.get(pool)
        if not s or not t:
            return None
        return s[0], t[0], min(s[1], t[1])

    def price(self, pool: str) -> float | None:
        """The box's current view of the pool price = the raw reserve ratio sol/tok (unit-free). None if unseen
        or the token side is empty."""
        r = self.get(pool)
        if not r or not r[1]:
            return None
        return r[0] / r[1]

    def stamp(self, pool: str) -> dict:
        """The reserve fields to attach to a fire record: {sol, tok, reserve_slot} (all None if the pool is unseen)."""
        r = self.get(pool)
        if not r:
            return {"sol": None, "tok": None, "reserve_slot": None}
        return {"sol": r[0], "tok": r[1], "reserve_slot": r[2]}
