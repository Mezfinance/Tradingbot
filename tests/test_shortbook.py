"""Paper short book (Strategy 1): detection, A/B classification, sizing, funding, exits. Offline fakes only."""
import pytest
from shortbook.book import Book
from shortbook.config import Config
from shortbook.store import Store
from shortbook.venues import Contract, Quote, norm

DAY = 86400.0


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeListing:
    def __init__(self, markets, price=None):
        self.m, self.trades, self.price = dict(markets), {}, price or {}

    def markets(self):
        return dict(self.m)

    def first_trade(self, symbol):
        return self.trades.get(symbol)

    def price_usd(self, symbol):
        return self.price.get(symbol)


class FakePerp:
    def __init__(self, name, contracts, quotes, funding=None):
        self.name, self.c, self.qs, self.f = name, contracts, quotes, funding or {}

    def contracts(self):
        return {b: Contract(self.name, f"{b}USDT", b, 1.0, t) for b, t in self.c.items()}

    def quote(self, c):
        return self.qs.get(c.base)

    def funding(self, c, since):
        return [(t, r) for t, r in self.f.get(c.base, []) if t >= since]

    def limits(self, c):
        return {}


def make(clock, listing=None, perps=None, **cfg):
    cfg = Config(db_path=":memory:", cash_usd=1_000_000, **cfg)
    up = listing or FakeListing({"BTC": "KRW-BTC"})
    perps = perps or {}
    return Book(cfg, Store(":memory:"), {"upbit": up}, perps, now=clock), up


def q(price, oi=50e6, vol=500e6, fr=0.0001):
    return Quote(price, price, oi, vol, fr)


def test_norm_strips_thousand_prefix():
    assert norm("1000PEPE") == ("PEPE", 1000.0)
    assert norm("1MBABYDOGE") == ("BABYDOGE", 1_000_000.0)
    assert norm("1INCH") == ("1INCH", 1.0)


def test_first_read_is_baseline_then_new_market_is_pending():
    clock = Clock()
    book, up = make(clock)
    assert book.detect() == []                      # baseline: nothing is new
    up.m["XYZ"] = "KRW-XYZ"
    assert len(book.detect()) == 1
    r = book.s.q("SELECT * FROM listings")[0]
    assert (r["base"], r["status"]) == ("XYZ", "pending")
    assert book.detect() == []                      # seen once only


def test_b_listing_is_shorted_with_oi_cap_and_costs():
    clock = Clock()
    perp = FakePerp("binance", {"XYZ": clock.t - 30 * DAY}, {"XYZ": q(2.0, oi=4e6, vol=400e6)})
    book, up = make(clock, perps={"binance": perp})
    book.detect()
    up.m["XYZ"] = "KRW-XYZ"
    up.price["KRW-XYZ"] = 2.1
    book.detect()
    up.trades["KRW-XYZ"] = clock.t - 60
    book.check_pending()
    p = book.open_positions()[0]
    assert p["group_name"] == "B upbit"
    assert p["size_usd"] == pytest.approx(80_000)             # 2% of $4M OI < 10% of $1M NAV
    impact = 0.10 * (80_000 / 400e6) ** 0.5
    assert p["fees_usd"] == pytest.approx(80_000 * (0.002 + 2 * impact))
    assert book.s.q("SELECT status FROM listings")[0]["status"] == "entered"


def test_new_token_on_upbit_is_a_and_between_is_skipped():
    clock = Clock()
    perp = FakePerp("bybit", {"NEW": clock.t - 3600, "MID": clock.t - 5 * DAY}, {"NEW": q(1.0), "MID": q(1.0)})
    book, up = make(clock, perps={"bybit": perp})
    book.detect()
    up.m.update({"NEW": "KRW-NEW", "MID": "KRW-MID"})
    book.detect()
    up.trades.update({"KRW-NEW": clock.t, "KRW-MID": clock.t})
    book.check_pending()
    rows = {r["base"]: r for r in book.s.q("SELECT * FROM listings")}
    assert rows["NEW"]["kind"] == "A" and rows["NEW"]["status"] == "entered"
    assert rows["MID"]["kind"] is None and "between A and B" in rows["MID"]["reason"]


def test_skips_no_perp_and_ticker_mismatch():
    clock = Clock()
    perp = FakePerp("gate", {"SAME": clock.t - 60 * DAY}, {"SAME": q(5.0)})
    book, up = make(clock, perps={"gate": perp})
    book.detect()
    up.m.update({"NOPE": "KRW-NOPE", "SAME": "KRW-SAME"})
    up.price["KRW-SAME"] = 0.01                     # a different token with the same ticker
    book.detect()
    up.trades.update({"KRW-NOPE": clock.t, "KRW-SAME": clock.t})
    book.check_pending()
    rows = {r["base"]: r for r in book.s.q("SELECT * FROM listings")}
    assert rows["NOPE"]["status"] == "skipped"      # no perp: A (no age evidence) without a contract
    assert "price doesn't match" in rows["SAME"]["reason"]
    assert not book.open_positions()


def test_late_detection_and_disabled_group_are_skipped():
    clock = Clock()
    perp = FakePerp("binance", {"OLD": clock.t - 90 * DAY, "LATE": clock.t - 90 * DAY}, {"OLD": q(1.0), "LATE": q(1.0)})
    book, up = make(clock, perps={"binance": perp}, groups="A:upbit")
    book.detect()
    up.m.update({"OLD": "KRW-OLD", "LATE": "KRW-LATE"})
    book.detect()
    up.trades.update({"KRW-OLD": clock.t, "KRW-LATE": clock.t - 5 * 3600})
    book.check_pending()
    rows = {r["base"]: r for r in book.s.q("SELECT * FROM listings")}
    assert "not traded" in rows["OLD"]["reason"]
    assert "before we saw it" in rows["LATE"]["reason"]


def test_funding_marking_and_exit_after_seven_days():
    clock = Clock()
    t0 = clock.t
    perp = FakePerp("binance", {"XYZ": t0 - 30 * DAY}, {"XYZ": q(10.0, oi=100e6, vol=1e12, fr=0.0)},
                    funding={"XYZ": [(t0 - 3600, 0.01), (t0 + 8 * 3600, 0.001), (t0 + 16 * 3600, -0.002)]})
    book, up = make(clock, perps={"binance": perp})
    book.detect(); up.m["XYZ"] = "KRW-XYZ"; book.detect(); up.trades["KRW-XYZ"] = t0
    book.check_pending()
    p = book.open_positions()[0]
    assert p["size_usd"] == pytest.approx(100_000)            # 10% of NAV < 2% of $100M OI
    perp.qs["XYZ"] = q(8.0, oi=100e6, vol=1e12)               # price falls 20%
    clock.t = t0 + 20 * 3600
    book.manage()
    p = book.open_positions()[0]
    # settlements after entry only; notional marked at the current price (0.8x)
    assert p["funding_usd"] == pytest.approx(100_000 * 0.8 * (0.001 - 0.002))
    book.manage()                                             # no double counting
    assert book.open_positions()[0]["funding_usd"] == pytest.approx(p["funding_usd"])
    nav = book.nav()
    assert nav == pytest.approx(1_000_000 + 20_000 + p["funding_usd"] - p["fees_usd"])
    clock.t = t0 + 7 * DAY + 1
    book.manage()
    assert not book.open_positions()
    c = book.s.q("SELECT * FROM positions")[0]
    assert c["status"] == "closed" and c["pnl_usd"] == pytest.approx(20_000 + c["funding_usd"] - c["fees_usd"])
    assert float(book.s.get("cash")) == pytest.approx(1_000_000 + c["pnl_usd"])


def test_gross_cap_limits_new_shorts_and_one_short_per_token():
    clock = Clock()
    names = ["AAA", "BBB", "CCC"]
    perp = FakePerp("binance", {n: clock.t - 30 * DAY for n in names}, {n: q(1.0, oi=1e9, vol=1e12) for n in names})
    book, up = make(clock, perps={"binance": perp}, trade_pct=40, gross_cap_pct=100)
    book.detect()
    for n in names:
        up.m[n] = f"KRW-{n}"
    book.detect()
    for n in names:
        up.trades[f"KRW-{n}"] = clock.t
    book.check_pending()
    sizes = [p["size_usd"] for p in book.open_positions()]
    assert sum(sizes) <= 1_000_000 * 1.0 + 1e-6
    assert len(sizes) == 3 and sizes[-1] < sizes[0]           # the third gets only what's left


def test_pending_listing_expires():
    clock = Clock()
    book, up = make(clock)
    book.detect(); up.m["ZZZ"] = "KRW-ZZZ"; book.detect()
    clock.t += 73 * 3600
    book.check_pending()
    assert book.s.q("SELECT status FROM listings")[0]["status"] == "expired"


def test_step_survives_a_failing_job():
    clock = Clock()
    book, up = make(clock)
    up.markets = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    book.step({})
    assert book.s.q("SELECT kind FROM events WHERE kind='loop_error'")
    assert book.s.get("heartbeat") == clock.t


class CannedHttp:
    """Answers by which params were sent: Bybit's error when startTime comes without endTime."""
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def get(self, url, params=None):
        self.calls.append(dict(params or {}))
        if "startTime" in (params or {}) and "endTime" not in params:
            return {"retCode": 10001, "retMsg": "params error"}
        return {"retCode": 0, "result": {"list": self.rows}}


def test_bybit_funding_sends_start_and_end_and_filters():
    from shortbook.venues import BybitPerp
    now = 1_800_000_000
    rows = [{"fundingRateTimestamp": str((now - h * 3600) * 1000), "fundingRate": "0.0001"} for h in (2, 10, 30)]
    h = CannedHttp(rows)
    got = BybitPerp(h).funding(Contract("bybit", "XUSDT", "X", 1.0, None), now - 24 * 3600)
    assert "endTime" in h.calls[0] and "startTime" in h.calls[0]
    assert [t for t, _ in got] == [now - 10 * 3600, now - 2 * 3600]   # oldest first, 30 h ago dropped


def test_bybit_funding_falls_back_to_latest_on_error():
    from shortbook.venues import BybitPerp

    class ErrThenOk(CannedHttp):
        def get(self, url, params=None):
            self.calls.append(dict(params or {}))
            if "startTime" in (params or {}):
                return {"retCode": 10001}
            return {"retCode": 0, "result": {"list": self.rows}}
    now = 1_800_000_000
    h = ErrThenOk([{"fundingRateTimestamp": str((now - 3600) * 1000), "fundingRate": "-0.0002"}])
    got = BybitPerp(h).funding(Contract("bybit", "XUSDT", "X", 1.0, None), now - 7200)
    assert got == [(now - 3600, -0.0002)] and len(h.calls) == 2


def test_feed_serves_nav_positions_and_groups_over_ipv6_and_ipv4():
    """11-05: the MZF dashboard reads the short book over Fly's private network (IPv6) from /api/feed."""
    import json as _json
    import urllib.request
    from shortbook import cli
    clock = Clock()
    t0 = clock.t
    perp = FakePerp("binance", {"XYZ": t0 - 30 * DAY, "ABC": t0 - 30 * DAY},
                    {"XYZ": q(10.0, oi=100e6, vol=1e12, fr=0.0), "ABC": q(5.0, oi=100e6, vol=1e12, fr=0.0)})
    book, up = make(clock, perps={"binance": perp})
    book.detect(); up.m["XYZ"] = "KRW-XYZ"; book.detect(); up.trades["KRW-XYZ"] = t0
    book.check_pending()
    perp.qs["XYZ"] = q(8.0, oi=100e6, vol=1e12)
    clock.t = t0 + 7 * DAY + 1
    book.manage()                                             # XYZ closed with a 20% fall
    up.m["ABC"] = "KRW-ABC"; book.detect(); up.trades["KRW-ABC"] = clock.t
    book.check_pending()                                      # ABC open
    book.s.put("heartbeat", clock.t)
    f = cli.feed(book, now=clock.t)
    assert f["program"] == "MZF Short Only" and f["healthy"] and f["closed_total"] == 1
    assert [p["base"] for p in f["open"]] == ["ABC"] and f["closed"][0]["base"] == "XYZ"
    assert f["closed"][0]["price_pct"] == pytest.approx(20.0)
    g = f["strategies"][0]["groups"]["B upbit"]
    assert g["open"] == 1 and g["closed"] == 1 and g["research_pct"] == 13.3 and g["mean_pct"] > 0
    assert f["nav_usd"] == pytest.approx(book.nav(), abs=0.01)
    srv = cli.serve(book, 0)
    try:
        port = srv.server_address[1]
        import socket
        try:  # this host may have no IPv6 (the sandbox doesn't); Fly machines do
            socket.socket(socket.AF_INET6).close()
            v6 = srv.address_family == socket.AF_INET6
        except OSError:
            v6 = False
        for host in (["[::1]"] if v6 else []) + ["127.0.0.1"]:
            body = _json.loads(urllib.request.urlopen(f"http://{host}:{port}/api/feed", timeout=5).read())
            assert body["closed_total"] == 1
    finally:
        srv.shutdown()
