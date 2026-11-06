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
    assert {"name": "launchrw50", "hold_min": dash.Live(cfg).holds["launchrw50"], "flag": "rw50",
            "family": "Rug-vs-win", "cash": cfg.model_book_cash_usd, "twin": None} in init["books"]
    assert [f["name"] for f in init["families"]] == [n for n, _ in dash.FAMILIES]


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
        assert code == 200 and b"QUANT DESK" in body and b"EventSource" in body
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


def test_short_book_feed_is_streamed_and_kept_stale_when_unreachable(cfg, monkeypatch):
    """11-05: MZF Short Only (the shortbook app) on the same page, read over Fly's private network."""
    monkeypatch.setattr(dash, "POLL_S", 0.01)
    cfg.shortbook_feed_url = "http://shortbook-paper.internal:8080/api/feed"
    board = dash.Dashboard(Journal(":memory:"), cfg)
    board.refresh()
    board.fetch_shorts(get=lambda url: {"program": "MZF Short Only", "nav_usd": 1_012_000.0, "start_usd": 1e6})
    assert board.shorts["nav_usd"] == 1_012_000.0 and board.shorts["stale"] is False

    def down(url):
        raise ConnectionError("no route")
    board.fetch_shorts(get=down)
    assert board.shorts["stale"] and board.shorts["nav_usd"] == 1_012_000.0 and "no route" in board.shorts["error"]
    out, stop = [], threading.Event()

    def write(data):
        out.append(data.decode())
        if len(out) >= 4:
            stop.set()
    board.stream(write, stop)
    events = [o.split("\n")[0] for o in out]
    assert "event: shorts" in events and json.loads(out[0].split("data: ", 1)[1])["shorts_enabled"]
    cfg.shortbook_feed_url = ""
    assert not dash.Dashboard(Journal(":memory:"), cfg).live.snapshot()[1]["shorts_enabled"]


def test_the_canary_starts_from_its_own_budget():
    from tradingbot import model_book as mb
    from tradingbot.config import Config
    cfg = Config()
    assert dash.start_cash(cfg, mb.CANARY4) == cfg.launch_canary_budget_usd == 50.0
    assert dash.start_cash(cfg, "launchrw50_sz") == cfg.model_book_cash_usd


def test_canary_positions_carry_their_swap_result_and_the_twin(cfg):
    from tradingbot import model_book as mb
    cfg.launch_book = cfg.launch_live = cfg.launch_canary = cfg.launch_sized_book = True
    j = Journal(":memory:")
    pid = j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
                    "VALUES ('T', 'T', 0.31, 0.0115, 0, 1.0, 400.0, ?)", (mb.CANARY4,))
    j.execute("INSERT INTO fills(position_id, side, qty, price, usd, ts) VALUES (?, 'buy', 474.7, 2.1065e-5, 0.31, 10)", (pid,))
    j.execute("INSERT INTO fills(position_id, side, qty, price, usd, ts) VALUES (?, 'sell', 474.7, 2.8922e-5, 0.0115, 370)", (pid,))
    live = dash.Live(cfg, j)
    live.poll()
    _, snap = live.snapshot()
    rows = [dict(zip(snap["pos_cols"], r)) for r in snap["positions"]]
    assert round(rows[0]["swap_ret"], 3) == 0.373  # the 11-06 GKRs7p trade: -96% net, +37% on the swap
    assert {b["name"]: b["twin"] for b in snap["books"]}[mb.CANARY4] == mb.CANARY_TWIN
