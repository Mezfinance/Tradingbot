from tradingbot import canary_slots as cs


class FakeHelius:
    """Serves getTransaction (buy slot) and getTransactionsForAddress (pool withdraw)."""
    def __init__(self, buy_slots, pool_txs):
        self.buy_slots = buy_slots          # sig -> (slot, blockTime)
        self.pool_txs = pool_txs            # pool -> [tx dicts]
        self.credits_used = 0
        self.min_interval_s = 0

    def _rpc(self, method, params, credits):
        self.credits_used += credits
        if method == "getTransaction":
            sig = params[0]
            if sig not in self.buy_slots:
                return None
            slot, bt = self.buy_slots[sig]
            return {"slot": slot, "blockTime": bt}
        if method == "getTransactionsForAddress":
            return {"data": self.pool_txs.get(params[0], []), "paginationToken": None}
        return None


def _wd(slot, bt, by):
    return {"slot": slot, "blockTime": bt, "meta": {"err": None, "logMessages": ["Program log: Instruction: Withdraw"]},
            "transaction": {"signatures": [f"wd{slot}"], "message": {"accountKeys": [{"pubkey": by}]}}}


def _swap(slot, bt, by):  # a normal trade: no Withdraw log, must be ignored
    return {"slot": slot, "blockTime": bt, "meta": {"err": None, "logMessages": ["Program log: Instruction: Buy"]},
            "transaction": {"signatures": [f"sw{slot}"], "message": {"accountKeys": [{"pubkey": by}]}}}


class FakeStore:
    def __init__(self, rows): self._rows = rows
    def query(self, sql, params=()):
        return self._rows if "watch_launch_live" in sql else []


class FakeJournal:
    def __init__(self, positions, fills): self._p = positions; self._f = fills
    def query(self, sql, params=()):
        if "FROM positions WHERE strategy=?" in sql:  # the twin lookup
            return []
        if "FROM positions WHERE strategy LIKE" in sql:
            return self._p
        if "FROM fills" in sql:
            return self._f.get(params[0], [])
        return []


def _row(**k):
    base = {"pool": "P", "token_mint": "MINT", "create_ts": 1000.0, "pulled_ts": None, "liq_usd": 9000.0,
            "sol_at_decide": 1.0, "ret": -1.0, "rw_pct": 10.0, "fill_ts": 1006.0}
    base.update(k)
    return base


def test_slot_gap_is_computed_and_our_own_withdraw_is_skipped(monkeypatch):
    monkeypatch.setattr(cs, "_our_wallets", lambda cfg: {"ME"})
    store = FakeStore([_row()])
    pos = [{"id": 1, "strategy": "canary_rw50_sz", "token": "MINT", "symbol": "RUG", "opened_at": 1006.0,
            "closed_at": 1366.0, "cost_usd": 25.0, "realized_usd": 0.0, "exit_reason": "pool_age_exit"}]
    fills = {1: [{"side": "buy", "price": 1.0, "usd": 25.0, "qty": 25.0, "fee_usd": 0.1, "ts": 1006.0, "tx": "BUY1"}]}
    journal = FakeJournal(pos, fills)
    # our buy at slot 100; our own withdraw (LP route exit) at slot 101 must be ignored; operator pull at slot 103
    pool_txs = {"P": [_swap(99, 1005, "X"), _wd(101, 1007, "ME"), _wd(103, 1008, "OP"), _wd(105, 1009, "OP2")]}
    helius = FakeHelius({"BUY1": (100, 1006)}, pool_txs)

    out = cs.slots(store, journal, helius, cfg=None, max_credits=10_000)
    t = out["trades"][0]
    assert t["buy_slot"] == 100 and t["pull_slot"] == 103 and t["slot_gap"] == 3 and t["by"] == "OP"
    assert t["time_gap_s"] == 2.0
    s = out["summary"]
    assert s["with_pull"] == 1 and s["median_slot_gap"] == 3
    assert s["beatable_ge2_slots_pct"] == 100.0 and s["unbeatable_le1_slot_pct"] == 0.0


def test_same_slot_pull_is_unbeatable_and_missing_buy_tx_is_handled(monkeypatch):
    monkeypatch.setattr(cs, "_our_wallets", lambda cfg: set())
    store = FakeStore([_row(pool="A", token_mint="MA"), _row(pool="B", token_mint="MB")])
    pos = [
        {"id": 1, "strategy": "canary_rw50_sz", "token": "MA", "symbol": "A", "opened_at": 1006.0,
         "closed_at": 1366.0, "cost_usd": 25.0, "realized_usd": 0.0, "exit_reason": "x"},
        {"id": 2, "strategy": "canary_rw50_sz", "token": "MB", "symbol": "B", "opened_at": 1006.0,
         "closed_at": 1366.0, "cost_usd": 25.0, "realized_usd": 0.0, "exit_reason": "x"},
    ]
    fills = {1: [{"side": "buy", "price": 1.0, "usd": 25.0, "qty": 25.0, "fee_usd": 0.1, "ts": 1006.0, "tx": "BUYA"}],
             2: [{"side": "buy", "price": 1.0, "usd": 25.0, "qty": 25.0, "fee_usd": 0.1, "ts": 1006.0, "tx": "GONE"}]}
    journal = FakeJournal(pos, fills)
    # trade A: pull in the same slot as the buy (gap 0). trade B: buy tx not found.
    pool_txs = {"A": [_wd(200, 1006, "OP")], "B": []}
    helius = FakeHelius({"BUYA": (200, 1006)}, pool_txs)  # BUYA found at slot 200; GONE (trade B) not found
    out = cs.slots(store, journal, helius, cfg=None, max_credits=10_000)
    rows = {r["symbol"]: r for r in out["trades"]}
    assert rows["A"]["slot_gap"] == 0
    assert "note" in rows["B"] and rows["B"].get("slot_gap") is None
    assert out["summary"]["slot_gap_distribution_pct"]["same_slot_0"] == 100.0


def test_by_size_reports_the_latest_pull_per_size():
    rows = [
        # three $25 trades: two pulled (at 12s via withdraw tx, 45s via scorer), one survivor
        {"symbol": "A", "cost": 25.0, "ret": -1.0, "time_gap_s": 12.0, "scorer_pull_s": None},
        {"symbol": "B", "cost": 24.98, "ret": -1.0, "time_gap_s": None, "scorer_pull_s": 45.0},
        {"symbol": "C", "cost": 25.0, "ret": -0.03, "time_gap_s": None, "scorer_pull_s": None},
        # a $5 trade pulled at 3s, and one rugged with no pull time on record
        {"symbol": "D", "cost": 5.0, "ret": -1.0, "time_gap_s": 3.0, "scorer_pull_s": None},
        {"symbol": "E", "cost": 5.0, "ret": -1.0, "time_gap_s": None, "scorer_pull_s": None},
    ]
    bs = cs._by_size(rows)
    assert bs["$25"]["n"] == 3 and bs["$25"]["pulled"] == 2 and bs["$25"]["rugged"] == 2
    assert bs["$25"]["pull_s_max"] == 45.0 and bs["$25"]["last"] == (45.0, "B")  # the latest rug
    assert bs["$25"]["pull_s_min"] == 12.0
    assert bs["$5"]["pulled"] == 1 and bs["$5"]["rugged"] == 2 and bs["$5"]["rugged_no_pull"] == 1
    assert cs._bucket(25.0) == "$25" and cs._bucket(0.01) == "<$1" and cs._bucket(None) == "?"
