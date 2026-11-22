"""Transaction -> Consumer mapping for the LocalShred tap (the tested, dependency-free half).

`tx_view` turns one parsed transaction (a solders `VersionedTransaction`, or anything duck-typed the
same) into the `(account_keys, instructions, fee_payer)` view `localshred_consumer.decode_tx` takes,
where `instructions` is a list of top-level `(program_id, account_indices, data)`. This is where the
index arithmetic lives (resolving each compiled instruction's program-id index and account indices
against the account-key list), so it is unit-tested; it needs no grpc and no solders (it only reads
`tx.message.account_keys`, `tx.message.instructions`, and each instruction's `program_id_index` /
`accounts` / `data`).

LUT note: for a v0 message `account_keys` is the STATIC keys only; address-table-lookup accounts are
unresolved, so a swap whose pool account is LUT-indexed has an out-of-range index and is skipped by
`decode_tx` (classify returns None). That is the same LUT-blindness the box has, and it is fine for the
first LocalShred verify: the box-exact result (2026-11-22) showed the gap is NOT the ~3.9% LUT drop but
deeper feed incompleteness, so LocalShred's value is a COMPLETE trade stream upstream of LUT -- a
LUT resolver is only worth adding if the LocalShred verify's residual is then the LUT swaps.

The box-run I/O loop (gRPC `ShredstreamProxy.SubscribeEntries` stream + bincode `Vec<Entry>` deserialize
+ solders `VersionedTransaction` parse -> these functions) is the Nuremberg measurement runbook in
SHRED_DECODER.md; it is validated on the box (and ultimately by frontrun-verify), not here, because its
correctness depends on solders' deserialize behaviour and the live feed.
"""


def tx_view(tx):
    """One parsed transaction -> (account_keys: list[str], instructions: list[(program_id, accts, data)],
    fee_payer: str|None). Top-level instructions only (the box's direct view excludes CPIs, which a
    compiled message does not contain anyway -- inner instructions live in meta, which the entry stream
    does not carry). An instruction whose program-id index is out of range (LUT-only) is skipped."""
    msg = tx.message
    account_keys = [str(k) for k in msg.account_keys]
    n = len(account_keys)
    fee_payer = account_keys[0] if account_keys else None
    out = []
    for ix in msg.instructions:
        pid_idx = int(ix.program_id_index)
        if pid_idx < 0 or pid_idx >= n:
            continue
        out.append((account_keys[pid_idx], [int(a) for a in ix.accounts], bytes(ix.data)))
    return account_keys, out, fee_payer


def on_transaction(consumer, slot, tx):
    """Feed one parsed transaction at `slot` into the Consumer. Returns any fires it produced."""
    account_keys, instructions, fee_payer = tx_view(tx)
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


def consume_stream(stream, consumer):
    """Drive the Consumer from a `ShredstreamProxy.SubscribeEntries` stream: each message has `.slot`
    and `.entries` (bincode `Vec<Entry>`). Fires go through `consumer.on_fire`. The gRPC channel + the
    grpcio-tools-generated stubs are the only box-local bit:
        stub = ShredstreamProxyStub(grpc.insecure_channel("127.0.0.1:9898"))
        consume_stream(stub.SubscribeEntries(SubscribeEntriesRequest()), consumer)
    """
    for msg in stream:
        for tx in deserialize_entries(msg.entries):
            on_transaction(consumer, msg.slot, tx)
