"""Transaction -> Consumer mapping for the LocalShred tap (the tested, dependency-free half).

`tx_view` turns one parsed transaction (a solders `VersionedTransaction`, or anything duck-typed the
same) into the `(account_keys, instructions, fee_payer)` view `localshred_consumer.decode_tx` takes,
where `instructions` is a list of top-level `(program_id, account_indices, data)`. This is where the
index arithmetic lives (resolving each compiled instruction's program-id index and account indices
against the account-key list), so it is unit-tested; it needs no grpc and no solders (it only reads
`tx.message.account_keys`, `tx.message.instructions`, and each instruction's `program_id_index` /
`accounts` / `data`).

LUT resolution (the "fuller decode", 2026-11-23): for a v0 message `account_keys` is the STATIC keys only;
address-table-lookup accounts sit after them, in Solana's order (all tables' WRITABLE entries, then all
tables' READONLY entries). Without the table contents a swap whose pool/mint is LUT-indexed has an
out-of-range index and is skipped (classify returns None) -- the box's historical ~3.9% `drop_lut`. Pass a
`resolve(table_pubkey) -> list[str] | None` (the box caches each lookup-table account's address list; the
offline completeness tool fetches it from Helius) and `tx_view` rebuilds the full resolved account list so
those swaps decode. Resolution is all-or-nothing per tx: if ANY referenced table is unresolved we fall back
to the static-only keys (a partial resolve would shift every later table's absolute index and corrupt the
read), so `resolve=None` (or a miss) reproduces the exact old LUT-blind behaviour -- the default, byte-for-byte.
Why bother now that 2026-11-23 showed routing is 0.2% and the gap is deeper feed incompleteness: a clean
completeness read needs LUT swaps IN, so a "missing" trade is genuine shred-feed loss, not a resolver we chose
not to run. Routed (CPI) swaps still never appear here -- a compiled message carries no inner instructions.

The box-run I/O loop (gRPC `ShredstreamProxy.SubscribeEntries` stream + bincode `Vec<Entry>` deserialize
+ solders `VersionedTransaction` parse -> these functions) is the Nuremberg measurement runbook in
SHRED_DECODER.md; it is validated on the box (and ultimately by frontrun-verify), not here, because its
correctness depends on solders' deserialize behaviour and the live feed.
"""


def resolve_account_keys(msg, resolve):
    """Full resolved account-key list for a (possibly v0) message: static keys + LUT writable + LUT readonly,
    in Solana's canonical order. `resolve(table_pubkey) -> list[str] | None` gives a lookup table's address
    list. Returns (keys, resolved_lut) where resolved_lut is True only if EVERY referenced table resolved
    (and every index was in range); otherwise (static_keys, False) -- never a partial/corrupt resolve.
    A legacy message (no address_table_lookups) returns (static_keys, True)."""
    static = [str(k) for k in msg.account_keys]
    lookups = getattr(msg, "address_table_lookups", None) or []
    if not lookups:
        return static, True
    if resolve is None:
        return static, False
    writable, readonly = [], []
    for lk in lookups:
        table = resolve(str(lk.account_key))
        if not table:
            return static, False                       # unresolved table -> can't place later tables -> bail
        try:
            writable.extend(table[int(i)] for i in lk.writable_indexes)
            readonly.extend(table[int(i)] for i in lk.readonly_indexes)
        except (IndexError, TypeError):
            return static, False                       # table shorter than an index expects -> bail
    return static + writable + readonly, True


# --- Address Lookup Table resolution (fetch + cache the table contents a shred cannot carry) ---------------

ALT_PROGRAM = "AddressLookupTab1e1111111111111111111111111"   # Solana address-lookup-table program
_ALT_META = 56   # LOOKUP_TABLE_META_SIZE: an initialized table's addresses start here, 32 bytes each
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(raw: bytes) -> str:
    n = int.from_bytes(raw, "big")
    out = ""
    while n > 0:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    pad = 0
    for b in raw:
        if b == 0:
            pad += 1
        else:
            break
    return "1" * pad + out


def parse_lookup_table(data) -> list:
    """The address list of an initialized Address Lookup Table account, from its raw data bytes: the meta is a
    fixed 56 bytes, then packed 32-byte pubkeys. [] if the data is missing or too short (resolve_account_keys
    treats [] the same as an unresolved table and bails to the static keys, so a malformed table is harmless)."""
    if not data or len(data) < _ALT_META:
        return []
    body = bytes(data)[_ALT_META:]
    return [_b58encode(body[i * 32:(i + 1) * 32]) for i in range(len(body) // 32)]


class LutResolver:
    """resolve(table_pubkey) -> list[str] | None for tx_view, backed by a `fetch(table) -> raw account data |
    None` (the box builds `fetch` from its Helius/RPC getMultipleAccounts; return the account's data bytes only
    when its owner is ALT_PROGRAM, else None). A LUT is append-only, so an address at an index never changes and
    caching the list is safe. Two resolve paths:
      resolve        -- fetch inline on a cache miss; for the OFFLINE --capture decode, where a round-trip per new
                        table is fine and every LUT-indexed direct swap should decode.
      resolve_cached -- cache-only (returns None + records the table in `pending`); for the LIVE hot path, so a
                        new table never blocks a slot. The box warms `pending` out-of-band with warm(table), and
                        the table resolves on its next sighting (an operator reuses its table across many swaps).
    A miss (fetch None, not a LUT, or an empty table) is cached as None so it is not refetched every tx."""

    def __init__(self, fetch):
        self._fetch = fetch
        self._cache: dict = {}        # table pubkey -> list[str] | None
        self.pending: set = set()     # tables seen on a cache miss (live: warm these in the background)

    def resolve(self, table):
        if table not in self._cache:
            self._cache[table] = parse_lookup_table(self._fetch(table)) or None
        return self._cache[table]

    def resolve_cached(self, table):
        if table in self._cache:
            return self._cache[table]
        self.pending.add(table)
        return None

    def warm(self, table):
        self._cache[table] = parse_lookup_table(self._fetch(table)) or None
        self.pending.discard(table)


def tx_view(tx, resolve=None):
    """One parsed transaction -> (account_keys: list[str], instructions: list[(program_id, accts, data)],
    fee_payer: str|None). Top-level instructions only (the box's direct view excludes CPIs, which a
    compiled message does not contain anyway -- inner instructions live in meta, which the entry stream
    does not carry). With `resolve` (a LUT lookup) the account list is the full resolved set so LUT-indexed
    swaps decode; without it (the default), v0 LUT accounts are unresolved and such instructions are skipped,
    reproducing the box's LUT-blind view exactly. An instruction with an out-of-range index is skipped."""
    msg = tx.message
    account_keys, _ = resolve_account_keys(msg, resolve)
    n = len(account_keys)
    fee_payer = account_keys[0] if account_keys else None
    out = []
    for ix in msg.instructions:
        pid_idx = int(ix.program_id_index)
        if pid_idx < 0 or pid_idx >= n:
            continue
        accts = [int(a) for a in ix.accounts]
        if any(a < 0 or a >= n for a in accts):        # an unresolved-LUT account index -> skip (old behaviour)
            continue
        out.append((account_keys[pid_idx], accts, bytes(ix.data)))
    return account_keys, out, fee_payer


def on_transaction(consumer, slot, tx, resolve=None):
    """Feed one parsed transaction at `slot` into the Consumer. Returns any fires it produced.
    `resolve` (optional LUT lookup) enables the fuller LUT-resolved decode; default = LUT-blind."""
    account_keys, instructions, fee_payer = tx_view(tx, resolve)
    return consumer.on_tx(slot, account_keys, instructions, fee_payer)


def deserialize_entries(entries_bytes):
    """Yield each VersionedTransaction from a LocalShred entry-service `Entry.entries` payload (bincode
    `Vec<solana_entry::Entry>`). Needs solders (the `live` extra). Each `solana_entry::Entry` is
    `{ num_hashes:u64, hash:[u8;32], transactions:Vec<VersionedTransaction> }`, bincode: a `Vec` is a u64
    LE length prefix then its elements, and a transaction is serialized inline. We deserialize the tx at
    the current offset and advance by its byte length -- `VersionedTransaction.from_bytes` tolerates
    trailing bytes and `bytes(tx)` round-trips to the exact length (verified on solders 0.29, 2026-11-22),
    so no hand-written bincode message parser is needed."""
    from solders.transaction import VersionedTransaction   # box dep (pyproject `live` extra)
    buf = bytes(entries_bytes)
    off = 0

    def u64():
        nonlocal off
        v = int.from_bytes(buf[off:off + 8], "little")
        off += 8
        return v

    n_entries = u64()
    for _ in range(n_entries):
        u64()                 # num_hashes (unused)
        off += 32             # hash
        for _ in range(u64()):
            tx = VersionedTransaction.from_bytes(buf[off:])
            off += len(bytes(tx))
            yield tx


def consume_stream(stream, consumer, resolve=None):
    """Drive the Consumer from a `ShredstreamProxy.SubscribeEntries` stream: each message has `.slot`
    and `.entries` (bincode `Vec<Entry>`). Fires go through `consumer.on_fire`. Pass `resolve` (a
    LutResolver's `resolve` for --capture, `resolve_cached` for the live hot path) to decode LUT-indexed
    direct swaps instead of dropping them. The gRPC channel + the grpcio-tools-generated stubs are the only
    box-local bit:
        stub = ShredstreamProxyStub(grpc.insecure_channel("127.0.0.1:9898"))
        lut = LutResolver(fetch)   # fetch(table) -> ALT account data via the box's Helius getMultipleAccounts
        consume_stream(stub.SubscribeEntries(SubscribeEntriesRequest()), consumer, resolve=lut.resolve)
    """
    for msg in stream:
        for tx in deserialize_entries(msg.entries):
            on_transaction(consumer, msg.slot, tx, resolve)
