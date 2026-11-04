"""Live dashboard: password, trade feed, server-sent events, routes."""
import http.client
import json
import threading
import time

from tradingbot import dashboard as dash
from tradingbot import health
from tradingbot.journal import Journal
from tradingbot.models import Position


def _trade(j, book, sym, cost=100.0, back=None):
    pos = Position(None, f"M{sym}", sym, 1.0, 0.0, 100.0, cost, time.time() - 300, 0.0, 1.0, strategy=book)
    j.save_position(pos)
    j.log_fill(pos.id, "buy", 100.0, 1.0, cost, 0.1, "entry", None, trigger_price=1.0)
    if back is not None:
        pos.realized_usd, pos.closed_at, pos.exit_reason = back, time.time(), "pool_age_exit"
        j.save_position(pos)
        j.log_fill(pos.id, "sell", 100.0, back / 100, back, 0.1, "pool_age_exit", None, trigger_price=1.0)
    return pos


def test_password_by_key_or_cookie(monkeypatch):
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    assert not dash.authorised({}, {"key": ["x"]})  # no secret, no dashboard
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    assert dash.authorised({}, {"key": ["s3cret"]})
    assert dash.authorised({"Cookie": "a=1; tb_dash=s3cret"}, {})
    assert not dash.authorised({"Cookie": "tb_dash=wrong"}, {}) and not dash.authorised({}, {})


def test_feed_lists_buys_and_sells_with_returns_and_skips_archived_books():
    j = Journal(":memory:")
    _trade(j, "launchrw50", "WIN", back=137.8)
    _trade(j, "launchd12@2026-11-04-impact-probe", "OLD", back=50.0)
    _trade(j, "launch4_2", "OPEN")
    feed = dash.fills_after(j, 0)
    assert [(f["book"], f["side"]) for f in feed] == [("launchrw50", "buy"), ("launchrw50", "sell"),
                                                      ("launch4_2", "buy")]
    assert feed[1]["ret_pct"] == 37.8 and feed[1]["held_s"] >= 299 and feed[0]["ret_pct"] is None
    assert dash.fills_after(j, feed[1]["id"]) == feed[2:]
    assert [f["symbol"] for f in dash.last_fills(j, 2)] == ["WIN", "OPEN"]


def test_stream_sends_summary_backlog_and_new_trades(cfg, monkeypatch):
    monkeypatch.setattr(dash, "POLL_S", 0.01)
    j = Journal(":memory:")
    _trade(j, "launchrw50", "A", back=110.0)
    board = dash.Dashboard(j, cfg)
    board.refresh()
    assert board.current()[1]["books"] is not None
    out, stop = [], threading.Event()

    def write(data):
        out.append(data.decode())
        if any('"symbol": "B"' in o for o in out):
            stop.set()
    t = threading.Thread(target=board.stream, args=(write, stop))
    t.start()
    time.sleep(0.05)
    _trade(j, "launchrw50", "B")
    t.join(2)
    events = [o.split("\n")[0] for o in out]
    assert events[:2] == ["event: summary", "event: backlog"] and "event: fill" in events
    assert not t.is_alive()


def test_routes_need_the_password_and_health_stays_open(cfg, monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    cfg.health_port = 18766
    j = Journal(":memory:")
    board = dash.Dashboard(j, cfg)
    board.refresh()
    server = health.start_health_server(j, cfg, board)

    def get(path, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", 18766, timeout=5)
        c.request("GET", path, headers={"Cookie": cookie} if cookie else {})
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    try:
        assert get("/health")[0] == 200
        assert get("/")[0] == 401 and get("/api/summary")[0] == 401
        code, headers, _ = get("/?key=s3cret")
        assert code == 303 and headers["Location"] == "/" and "tb_dash=s3cret" in headers["Set-Cookie"]
        code, _, body = get("/", cookie="tb_dash=s3cret")
        assert code == 200 and b"STRATEGY DESK" in body and b"EventSource" in body
        code, _, body = get("/api/summary", cookie="tb_dash=s3cret")
        assert code == 200 and "books" in json.loads(body)
        assert get("/nope", cookie="tb_dash=s3cret")[0] == 404
    finally:
        server.shutdown()
