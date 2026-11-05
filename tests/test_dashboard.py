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


def test_stream_sends_rows_then_deltas_and_new_trades(cfg, monkeypatch):
    monkeypatch.setattr(dash, "POLL_S", 0.01)
    cfg.launch_book = cfg.launch_live = cfg.launch_rw_book = True
    j = Journal(":memory:")
    _trade(j, "launchrw50", "A", back=110.0)
    board = dash.Dashboard(j, cfg)
    board.refresh()
    assert board.current()[1]["expected"]
    out, stop = [], threading.Event()

    def write(data):
        out.append(data.decode())
        if any('"symbol": "B"' in o or '"symbol":"B"' in o for o in out) and any(o.startswith("event: delta") for o in out):
            stop.set()
    t = threading.Thread(target=board.stream, args=(write, stop))
    t.start()
    time.sleep(0.05)
    _trade(j, "launchrw50", "B")
    t.join(2)
    events = [o.split("\n")[0] for o in out]
    assert events[:3] == ["event: init", "event: summary", "event: backlog"]
    assert "event: fill" in events and "event: delta" in events and not t.is_alive()
    init = json.loads(out[0].split("data: ", 1)[1])
    rows = [dict(zip(init["pos_cols"], r)) for r in init["positions"]]
    assert [r["symbol"] for r in rows if r["strategy"] == "launchrw50"] == ["A"]
    assert {"name": "launchrw50", "hold_min": dash.Live(cfg).holds["launchrw50"], "flag": "rw50"} in init["books"]


def test_live_versions_rows_marks_blocked_exits_and_reads_launch_pools(cfg):
    from tradingbot import launch_live
    from tradingbot.watcher import WatchStore
    cfg.launch_book = cfg.launch_live = cfg.launch_rw_book = cfg.launch_low_book = True
    j = Journal(":memory:")
    store = WatchStore(":memory:")
    launch_live.ensure_schema(store)
    now = time.time()
    store.execute("INSERT INTO watch_launch_live(pool, token_mint, decided_ts, liquid, ret, status, low_h, rw_pct, "
                  "rw_low_pct) VALUES ('P1', 'M1', ?, 1, 0.25, 'done', 1, 10, 20)", (now,))
    live = dash.Live(cfg, j, store)
    live.poll()
    v0, snap = live.snapshot()
    pool = dict(zip(snap["launch_cols"], snap["launch"][0]))
    assert pool["ret"] == 0.25 and {"d12", "rw50", "d12rw50"} <= set(pool["flags"])
    assert live.delta(v0) == (v0, None)  # nothing changed: nothing sent
    pos = _trade(j, "launchrw50", "STUCK")
    j.log_event("exit_blocked", pos.token_address, "no route", strategy="launchrw50")
    live.poll()
    v1, d = live.delta(v0)
    row = dict(zip(snap["pos_cols"], d["positions"][0]))
    assert v1 > v0 and row["symbol"] == "STUCK" and row["blocked_since"] is not None and row["closed_at"] is None
    store.execute("UPDATE watch_launch_live SET ret = -1.0 WHERE pool = 'P1'")
    live.poll()
    _, d = live.delta(v1)
    assert d["positions"] == [] and dict(zip(snap["launch_cols"], d["launch"][0]))["ret"] == -1.0


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
        assert code == 200 and "expected" in json.loads(body)
        assert get("/nope", cookie="tb_dash=s3cret")[0] == 404
    finally:
        server.shutdown()


def test_summary_is_built_in_a_child_process_for_a_file_journal(cfg, tmp_path, monkeypatch):
    """11-05: building the summary inside the bot delayed the scorer by ~40 s; a file journal uses a child."""
    db = str(tmp_path / "bot.db")
    j = Journal(db)
    _trade(j, "launchrw50", "A", back=110.0)
    cfg.db_path, cfg.launch_live = db, False
    monkeypatch.setenv("TB_LAUNCH_LIVE", "0")
    board = dash.Dashboard(j, cfg)
    assert board.in_child
    board.refresh()
    version, summary = board.current()
    assert version == 1 and "expected" in summary and summary["downtime"] is not None
    assert (tmp_path / dash.SUMMARY_FILE).exists()
    assert not dash.Dashboard(Journal(":memory:"), cfg.with_overrides({"db_path": ":memory:"})).in_child
