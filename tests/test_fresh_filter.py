from tradingbot import fresh_filter
from tradingbot.flurry_study import _pool_vq, flows_from_file


class FakeHelius:
    """creation_slots: pool -> creation slot (or None = unreadable). vqs: pool -> raw i128 to pack at @245."""

    def __init__(self, creation_slots, vqs=None):
        self.creation_slots = creation_slots
        self.vqs = vqs or {}
        self.sig_calls = 0
        self.acct_calls = 0

    def first_slot(self, address):
        self.sig_calls += 1
        return self.creation_slots.get(address)

    def accounts(self, addresses):
        self.acct_calls += 1
        out = {}
        for a in addresses:
            vq = self.vqs.get(a)
            if vq is None:
                out[a] = None
            else:
                data = bytearray(261)
                data[245:261] = int(vq).to_bytes(16, "little", signed=True)
                out[a] = ("owner", bytes(data))
        return out


def _rec(pool, mint, first_slot, n=4, create_ts=100.0):
    txs = [[round(0.4 * i, 3), first_slot + i, "payer", 0, 0, 1000, 10] for i in range(n)]
    return {"pool": pool, "token_mint": mint, "create_ts": create_ts, "rec": {"create_ts": create_ts, "txs": txs}}


def test_keeps_fresh_drops_stale_and_stamps_vq():
    rows = [
        _rec("Pfresh", "mintA", first_slot=1005),   # creation 1000, gap 5 -> fresh
        _rec("Pstale", "mintB", first_slot=9000),   # creation 1000, gap 8000 -> stale (mid-life)
        _rec("Pedge", "mintC", first_slot=1450),    # creation 1000, gap 450 == window -> fresh
    ]
    helius = FakeHelius(
        creation_slots={"Pfresh": 1000, "Pstale": 1000, "Pedge": 1000},
        vqs={"Pfresh": 7_000_000_000, "Pedge": 123},
    )
    kept, stats = fresh_filter.fresh_filter(rows, helius, fresh_slots=450)
    pools = {r["pool"] for r in kept}
    assert pools == {"Pfresh", "Pedge"}              # stale dropped
    assert stats["fresh"] == 2 and stats["stale"] == 1 and stats["total"] == 3
    by_pool = {r["pool"]: r for r in kept}
    assert by_pool["Pfresh"]["rec"]["vq"] == 7_000_000_000   # vq stamped onto the kept rec
    assert by_pool["Pedge"]["rec"]["vq"] == 123
    # one creation read for every pool; one vq read only for the kept (fresh) ones
    assert helius.sig_calls == 3 and helius.acct_calls == 2
    assert stats["credits"] == 3 * 1 + 2 * 1


def test_unreadable_creation_is_dropped_not_kept():
    rows = [_rec("Pok", "m1", 1002), _rec("Pbad", "m2", 2000)]
    helius = FakeHelius(creation_slots={"Pok": 1000, "Pbad": None}, vqs={"Pok": 5})
    kept, stats = fresh_filter.fresh_filter(rows, helius)
    assert [r["pool"] for r in kept] == ["Pok"]
    assert stats["unreadable"] == 1 and stats["fresh"] == 1


def test_dedup_richest_and_drop_graduations():
    rows = [
        _rec("Pdup", "m", 1001, n=3),
        _rec("Pdup", "m", 1001, n=9),                # same pool, more txs -> kept
        _rec("Pgrad", "abcpump", 1001, n=5),         # pump.fun graduation -> excluded before any Helius call
    ]
    helius = FakeHelius(creation_slots={"Pdup": 1000}, vqs={"Pdup": 1})
    kept, stats = fresh_filter.fresh_filter(rows, helius)
    assert stats["total"] == 1 and [r["pool"] for r in kept] == ["Pdup"]
    assert len(kept[0]["rec"]["txs"]) == 9           # the richer duplicate
    assert helius.sig_calls == 1                     # graduation never reached Helius


def test_max_credits_caps_and_flags():
    rows = [_rec(f"P{i}", f"m{i}", 1001) for i in range(5)]
    helius = FakeHelius(creation_slots={f"P{i}": 1000 for i in range(5)},
                        vqs={f"P{i}": 1 for i in range(5)})
    kept, stats = fresh_filter.fresh_filter(rows, helius, max_credits=5)  # 2 credits/pool -> stops before pool 3
    assert stats["capped"] is True and stats["credits"] <= 5 and len(kept) <= 2


def test_kept_rows_reload_through_flows_from_file(tmp_path):
    import json
    rows = [_rec("Pfresh", "mintA", 1003)]
    helius = FakeHelius(creation_slots={"Pfresh": 1000}, vqs={"Pfresh": 9})
    kept, _ = fresh_filter.fresh_filter(rows, helius)
    path = tmp_path / "fresh.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in kept))
    metas, have = flows_from_file(str(path))
    assert [m["pool"] for m in metas] == ["Pfresh"]
    assert have["Pfresh"]["vq"] == 9                 # the study reads vq straight off the filtered rec
    assert have["Pfresh"]["txs"][0][1] == 1003       # _SLOT index, untouched
