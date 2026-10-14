import json
from datetime import datetime, timezone
from types import SimpleNamespace

from tradingbot.config import Config
from tradingbot.watcher import WatchStore, Watcher, compact, sampled

T0 = 1_800_000_000.0


def pool(addr, mint, dex="pumpswap", created=T0, price="1.0", vol24="0"):
    return {
        "attributes": {
            "address": addr, "name": f"{mint} / SOL",
            "pool_created_at": datetime.fromtimestamp(created, timezone.utc).isoformat().replace("+00:00", "Z"),
            "base_token_price_usd": price, "fdv_usd": "5000", "market_cap_usd": None, "reserve_in_usd": "3000",
            "transactions": {"m5": {"buys": 10, "sells": 4, "buyers": 9, "sellers": 3}},
            "volume_usd": {"m5": "800", "h24": vol24}, "price_change_percentage": {"m5": "12.5"},
        },
        "relationships": {"base_token": {"data": {"id": f"solana_{mint}"}}, "dex": {"data": {"id": dex}}},
    }


class FakeGecko:
    def __init__(self, pages=None, live=None, info=None, candles=None):
        self.pages = pages or {}
        self.live = live or {}
        self.info = info
        self.candles = candles or [[T0, 1, 3, 1, 2, 100]]
        self.calls = []

    def new_pools(self, page=1):
        self.calls.append(("new", page))
        return self.pages.get(page, [])

    def pools(self, addresses):
        self.calls.append(("multi", tuple(addresses)))
        return {a: self.live[a]["attributes"] for a in addresses if a in self.live}

    def token_info(self, mint):
        self.calls.append(("info", mint))
        return self.info

    def ohlcv(self, pool):
        self.calls.append(("ohlcv", pool))
        return self.candles


def make(gecko, **cfg_kw):
    cfg = Config(db_path=":memory:", watch_launch_sample_pct=cfg_kw.pop("pct", 100), **cfg_kw)
    return Watcher(cfg, gecko, WatchStore(":memory:")), cfg


def test_discovery_samples_launches_keeps_graduations_skips_stables():
    launches = [pool(f"L{i}", f"M{i}", dex="pump-fun") for i in range(200)]
    grads = [pool("G1", "GM", dex="pumpswap"), pool("S1", "So11111111111111111111111111111111111111112")]
    w, _ = make(FakeGecko({1: launches + grads}), pct=20)
    w.discover(T0)
    kept = {r["pool"] for r in w.store.query("SELECT pool FROM watch_pools")}
    assert "G1" in kept and "S1" not in kept
    n_launch = len(kept) - 1
    assert 15 <= n_launch <= 70  # ~20% of 200, chosen by address hash
    assert {p for p in kept if p.startswith("L")} == {f"L{i}" for i in range(200) if sampled(f"L{i}", 20)}


def test_discovery_stops_paging_once_caught_up():
    g = FakeGecko({1: [pool("A", "MA")], 2: [pool("B", "MB")], 3: [pool("C", "MC")]})
    w, _ = make(g)
    w.discover(T0)
    g.calls.clear()
    g.pages = {1: [pool("N", "MN"), pool("A", "MA")], 2: [pool("Z", "MZ")]}
    w.discover(T0 + 30)
    assert [c for c in g.calls if c[0] == "new"] == [("new", 1)]


def test_snapshots_at_each_age_with_missing_and_late_marked():
    g = FakeGecko({1: [pool("A", "MA"), pool("B", "MB")]}, live={"A": pool("A", "MA", price="2.0")})
    w, _ = make(g, watch_ages_min="5,15")
    w.discover(T0)
    w.snapshot(T0 + 6 * 60)
    snaps = {r["pool"]: r["data"] for r in w.store.query("SELECT * FROM watch_snaps WHERE age_min=5")}
    assert json.loads(snaps["A"])["price"] == 2.0 and json.loads(snaps["A"])["buyers_m5"] == 9
    assert snaps["B"] == "{}"  # not returned by the API (dead or delisted)
    w.snapshot(T0 + 15 * 60 + 3600)  # an hour late: recorded as missed, never mislabelled
    late = w.store.query("SELECT data FROM watch_snaps WHERE age_min=15")
    assert [r["data"] for r in late] == [None, None]


def test_token_info_is_compacted():
    info = {"websites": ["x"], "twitter_handle": "abc", "developer_address": "DEV",
            "developer_holding_percentage": "4.5", "launchpad_details": {"graduation_percentage": 40, "completed": False},
            "holders": {"count": 120}, "mint_authority": "no"}
    w, _ = make(FakeGecko({1: [pool("A", "MA")]}, info=info))
    w.discover(T0)
    w.fetch_info(T0 + 16 * 60)
    got = json.loads(w.store.query("SELECT info FROM watch_pools")[0]["info"])
    assert got["developer"] == "DEV" and got["dev_holding_pct"] == 4.5 and got["graduation_pct"] == 40
    assert got["holders"] == 120 and got["twitter"] is True


def test_candles_only_for_pools_that_traded():
    g = FakeGecko({1: [pool("Q", "MQ"), pool("T", "MT")]},
                  live={"Q": pool("Q", "MQ", vol24="100"), "T": pool("T", "MT", vol24="90000")})
    w, _ = make(g, watch_ages_min="1440")
    w.discover(T0)
    w.snapshot(T0 + 1440 * 60 + 10)
    w.fetch_ohlcv(T0 + 26 * 3600)
    rows = {r["pool"]: r for r in w.store.query("SELECT * FROM watch_pools")}
    assert rows["Q"]["ohlcv"] == "[]" and rows["Q"]["closed"] == 1
    assert json.loads(rows["T"]["ohlcv"]) == [[T0, 1, 3, 1, 2, 100]] and rows["T"]["closed"] == 1
    assert [c for c in g.calls if c[0] == "ohlcv"] == [("ohlcv", "T")]


def test_daily_backup_uploads_once(tmp_path, monkeypatch):
    monkeypatch.setenv("BUCKET_NAME", "bkt")
    uploads = []
    s3 = SimpleNamespace(upload_file=lambda src, bucket, key: uploads.append((bucket, key)))
    cfg = Config(db_path=str(tmp_path / "bot.db"))
    w = Watcher(cfg, FakeGecko(), WatchStore(str(tmp_path / "watch.db")), s3_factory=lambda: s3)
    day = datetime(2026, 10, 15, 1, tzinfo=timezone.utc)
    w.backup_if_due(day)
    w.backup_if_due(day)
    assert uploads == [("bkt", "watch/watch-2026-10-15.db.gz")]
    assert (tmp_path / "backups" / "watch-2026-10-15.db.gz").exists()


def test_compact_flattens_windows():
    c = compact(pool("A", "MA")["attributes"])
    assert c["buys_m5"] == 10 and c["vol_m5"] == 800.0 and c["chg_m5"] == 12.5 and c["buys_h1"] is None
