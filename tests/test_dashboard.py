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
        code, _, body = get("/")  # unauthed root serves the sign-in / overview page, not the dashboard
        assert code == 200 and b"Sign in with passkey" in body and b"EventSource" not in body
        assert get("/api/summary")[0] == 401
        code, headers, _ = get("/?key=s3cret")
        assert code == 303 and headers["Location"] == "/" and "tb_dash=s3cret" in headers["Set-Cookie"]
        code, _, body = get("/", cookie="tb_dash=s3cret")
        assert code == 200 and b"QUANT DESK" in body and b"EventSource" in body
        code, _, body = get("/api/summary", cookie="tb_dash=s3cret")
        assert code == 200 and "expected" in json.loads(body)
        assert get("/nope", cookie="tb_dash=s3cret")[0] == 404
    finally:
        server.shutdown()


def test_pool_paths_survive_a_restart(cfg, tmp_path):
    """A fly deploy wipes in-RAM pool paths; without the early (~60s) samples a now-old pool's conf_ret /
    best entry can never be rebuilt. Persisting them to /data keeps those columns across a restart."""
    db = str(tmp_path / "bot.db")
    cfg.db_path = db
    board = dash.Dashboard(Journal(db), cfg)
    board._pool_paths = {"P1": {"pts": [[66.0, 14000.0, 1e-7], [360.0, 18000.0, 1.2e-7]],
                                "peak_liq": 18000.0, "pulled_ts": None, "conf_age": 66.0, "conf_ret": 0.14,
                                "opt_age": 66.0, "opt_ret": 0.14, "conf_by": {6: 0.14, 15: 0.9}}}
    board._save_pool_paths()
    assert (tmp_path / "pool_paths.json").exists()
    # a fresh Dashboard (= a redeploy) loads them back, with conf_by keys coerced to int for hold_curve
    reborn = dash.Dashboard(Journal(db), cfg)
    assert reborn._pool_paths["P1"]["conf_ret"] == 0.14
    assert reborn._pool_paths["P1"]["conf_by"][15] == 0.9  # int key, not "15"
    assert dash.Dashboard(Journal(":memory:"), cfg.with_overrides({"db_path": ":memory:"}))._pool_paths == {}


def test_fetch_pools_persists_to_data_even_when_the_live_read_fails(cfg, tmp_path, monkeypatch):
    """A slow / rate-limited public RPC must never stop the reserve samples reaching /data (or a fly deploy
    wipes them): fetch_pools persists _pool_paths in a finally, even when the live read / assemble raises."""
    from tradingbot import pool_tracker as pt
    db = str(tmp_path / "bot.db")
    cfg.db_path = db
    board = dash.Dashboard(Journal(db), cfg)
    board._pool_paths = {"P1": {"pts": [[66.0, 14000.0, 1e-7]], "peak_liq": 14000.0, "pulled_ts": None}}

    def boom(*a, **k):
        raise RuntimeError("rpc down")
    monkeypatch.setattr(pt, "feint_pools", boom)  # force the try body to raise before the old save point
    board.fetch_pools(rpc=lambda m, p: {})
    assert board.pools["stale"] is True  # the poll reported stale ...
    assert (tmp_path / "pool_paths.json").exists()  # ... but the file still landed in the db dir (/data on Fly)
    import json as _json
    with open(tmp_path / "pool_paths.json") as fh:
        assert "P1" in _json.load(fh)


def test_fetch_pools_keeps_finalized_scalars_for_aged_pools_and_only_drops_the_samples(cfg):
    """A pool past the live-sampling window (but still in the tracker's window) keeps its finalized peak / conf / opt
    scalars so the 2h+ survivor row still shows them; only the big pts sample list is cleared. A pool out of the
    window entirely is dropped."""
    from tradingbot import launch_live, pool_tracker as pt
    from tradingbot.watcher import WatchStore
    cfg.launch_canary = True
    now = time.time()
    j = Journal(":memory:")
    _feint(j, "canary_rw50_sz", "MintA", "alive", pool="PoolA", sol_before=70.0)
    store = WatchStore(":memory:")
    launch_live.ensure_schema(store)
    store.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, decided_ts, liquid, liq_usd, "
                  "sol_at_decide, status) VALUES(?,?,?,?,1,?,?,?)",
                  ("PoolA", "MintA", now - 7200, now - 7196, 14000.0, 70.0, "scored"))  # 2h old: past young, in window
    store.execute("INSERT INTO watch_market(ts, sol_usd) VALUES(?, ?)", (now - 10, 100.0))
    board = dash.Dashboard(j, cfg, store)
    board._pool_paths = {
        "PoolA": {"pts": [[66.0, 14000.0, 1e-7]], "peak_liq": 34800.0, "live_liq": 35100.0,
                  "conf_ret": 0.12, "conf_age": 66.0, "opt_ret": 0.2, "opt_age": 41.0, "conf_by": {6: 0.12},
                  "pulled_ts": None},
        "GhostPool": {"pts": [[1.0, 1.0, 1.0]], "peak_liq": 1.0},  # not in feints -> dropped
    }
    board.fetch_pools(rpc=lambda m, p: {})
    assert "GhostPool" not in board._pool_paths                      # out of window: dropped entirely
    kept = board._pool_paths["PoolA"]
    assert kept["pts"] == [] and kept["peak_liq"] == 34800.0 and kept["conf_ret"] == 0.12  # scalars survive, samples go
    r = next(x for x in board.pools["rows"] if x["pool"] == "PoolA")
    assert round(r["peak_liq"]) == 34800 and round(r["pnl25"], 2) == round(pt.STAKE_USD * 0.12, 2)  # still rendered


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


def test_market_monitor_rows_are_streamed(cfg, monkeypatch):
    """11-07: the Market tab reads the monitor's hourly rows (read only) and gets them as a `market` event."""
    import time as _t
    from tradingbot import market_monitor as mm
    from tradingbot.watcher import WatchStore
    monkeypatch.setattr(dash, "POLL_S", 0.01)
    cfg.market_monitor = True
    store = WatchStore(":memory:")
    board = dash.Dashboard(Journal(":memory:"), cfg, store)
    board.fetch_market()  # no table yet: an empty payload, no error
    assert board.market["rows"] == [] and board.market["last_hour"] is None
    store.execute(mm.SCHEMA)
    hour = _t.time() // 3600 * 3600 - 3600
    for i in range(30):
        store.execute("INSERT INTO market_hours(hour, data, ts) VALUES (?,?,?)",
                      (hour - i * 3600, json.dumps({"v": mm.VERSION, "liquid": 30, "rug_pct": 20.0 + i % 4}), 0))
    board.fetch_market()
    m = board.market
    assert len(m["rows"]) == 30 and m["last_hour"] == hour and m["compare"]["rug_pct"]["now"] is not None
    assert ["rug_pct", mm.LABELS["rug_pct"], "rugs"] in m["metrics"]
    out, stop = [], threading.Event()

    def write(data):
        out.append(data.decode())
        if len(out) >= 4:
            stop.set()
    board.stream(write, stop)
    assert "event: market" in [o.split("\n")[0] for o in out]
    assert json.loads(out[0].split("data: ", 1)[1])["market_enabled"]


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


def _feint(j, book, mint, verdict, note="", pool=None, sol_before=None, sol_after=None):
    d = {"verdict": verdict, "note": note}
    if pool is not None:
        d.update(pool=pool, sol_before=sol_before, sol_after=sol_after)
    j.log_event("canary_feint", mint, json.dumps(d), strategy=book)


def test_feints_feed_parses_verdict_note_and_pool_and_lists_newest():
    j = Journal(":memory:")
    _feint(j, "canary_rw50_sz", "MintA", "alive", pool="PoolA", sol_before=600.0, sol_after=610.0)
    _feint(j, "canary_rw50_sz", "MintB", "pulled", pool="PoolB", sol_before=60.0, sol_after=1.0)
    _feint(j, "canary_rw50_sz", "MintC", "inconclusive", note="feint: wallet SOL low")
    feed = dash.feints_after(j, 0)
    assert [f["verdict"] for f in feed] == ["alive", "pulled", "inconclusive"]
    assert feed[0]["pool"] == "PoolA" and feed[0]["sol_after"] == 610.0
    assert feed[2]["note"] == "feint: wallet SOL low" and feed[2]["pool"] is None
    assert dash.feints_after(j, feed[1]["id"]) == feed[2:]  # incremental by id
    assert [f["verdict"] for f in dash.last_feints(j, 2)] == ["pulled", "inconclusive"]  # newest, oldest-first


def test_live_poll_surfaces_feints_in_the_snapshot_stream(cfg):
    cfg.launch_book = cfg.launch_live = cfg.launch_canary = cfg.launch_sized_book = True
    j = Journal(":memory:")
    _feint(j, "canary_rw50_sz", "M1", "alive", pool="P1", sol_before=1.0, sol_after=1.0)
    live = dash.Live(cfg, j)
    live.poll()
    assert [f["verdict"] for f in live.feints] == ["alive"]
    _, snap = live.snapshot()
    assert snap["canary"] is True and snap["canary_halted"] is False
    _feint(j, "canary_rw50_sz", "M2", "pulled", pool="P2", sol_before=50.0, sol_after=0.1)
    live.poll()  # a new feint is picked up incrementally
    assert [f["verdict"] for f in live.feints] == ["alive", "pulled"]


def test_canary_switch_flips_the_kill_switch():
    from tradingbot.engine import CANARY_ALLOW, CANARY_HALT
    j = Journal(":memory:")
    j.set_state(CANARY_ALLOW, "1")
    assert dash.canary_switch(j, "stop") == {"halted": True, "msg": "Canary halted: no new buys or feints."}
    assert j.get_state(CANARY_HALT) is not None
    res = dash.canary_switch(j, "start")
    assert res["halted"] is False and j.get_state(CANARY_HALT) is None and j.get_state(CANARY_ALLOW) is None
    import pytest
    with pytest.raises(ValueError):
        dash.canary_switch(j, "sideways")


def test_canary_probe_all_clears_the_gate_and_warns_when_probe_is_off():
    from tradingbot.engine import CANARY_ALLOW, CANARY_HALT
    j = Journal(":memory:")
    j.set_state(CANARY_ALLOW, "1")
    j.set_state(CANARY_HALT, "123")

    class Cfg:
        launch_canary_probe = True
    res = dash.canary_probe_all(j, Cfg())
    assert res["running"] is True and res["probe_configured"] is True
    assert j.get_state(CANARY_HALT) is None and j.get_state(CANARY_ALLOW) is None  # both gates cleared

    class Off:
        launch_canary_probe = False
    res = dash.canary_probe_all(j, Off())
    assert res["probe_configured"] is False and "OFF" in res["msg"]  # warns probe mode is not configured


def test_canary_switch_buttons_are_behind_the_password(cfg, monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    cfg.health_port = 18767
    from tradingbot.engine import CANARY_HALT
    j = Journal(":memory:")
    board = dash.Dashboard(j, cfg)
    board.refresh()
    server = health.start_health_server(j, cfg, board)

    def post(path, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", 18767, timeout=5)
        c.request("POST", path, headers={"Cookie": cookie} if cookie else {})
        r = c.getresponse()
        return r.status, r.read()
    try:
        assert post("/api/canary/stop")[0] == 401  # no password
        code, body = post("/api/canary/stop", cookie="tb_dash=s3cret")
        assert code == 200 and json.loads(body)["halted"] is True and j.get_state(CANARY_HALT) is not None
        code, body = post("/api/canary/start", cookie="tb_dash=s3cret")
        assert code == 200 and json.loads(body)["halted"] is False and j.get_state(CANARY_HALT) is None
        c = http.client.HTTPConnection("127.0.0.1", 18767, timeout=5)  # a GET on the POST-only path is a 404
        c.request("GET", "/api/canary/stop", headers={"Cookie": "tb_dash=s3cret"})
        assert c.getresponse().status == 404
    finally:
        server.shutdown()


def test_wallet_directory_numbers_the_live_set_and_lists_the_archive(cfg, monkeypatch):
    from tradingbot import health
    monkeypatch.setattr(health, "canary_wallets", lambda c: {"wallets": ["Aaa", "Bbb"], "treasury": "Ttt",
                                                             "rotate": True, "route": "direct"})
    monkeypatch.setattr(health, "address_book", lambda: {"addresses": {
        "Aaa": {"label": "wallet-1"}, "Old1": {"label": "wallet-old-3", "created": "2026-11-10T00:00:00"},
        "Ttt": {"label": "treasury-current"}}})
    d = dash.wallet_directory(cfg)
    assert [(w.get("n"), w["role"], w["address"]) for w in d["live"][:3]] == [
        (1, "trading", "Aaa"), (2, "trading", "Bbb"), (None, "treasury", "Ttt")]
    assert d["live"][2]["label"] == "treasury-current"  # label comes from the address book
    assert [w["address"] for w in d["archive"]] == ["Old1"] and d["archive"][0]["label"] == "wallet-old-3"
    assert d["rotate"] is True  # live addresses are excluded from the archive


def test_fetch_wallets_adds_balances_and_tx_counts_and_keeps_last_on_failure(cfg, monkeypatch):
    from tradingbot import health
    monkeypatch.setattr(health, "canary_wallets", lambda c: {"wallets": ["Aaa", "Bbb"], "treasury": "Ttt"})
    monkeypatch.setattr(health, "address_book", lambda: {"addresses": {"Old1": {"label": "wallet-old-3"}}})
    board = dash.Dashboard(Journal(":memory:"), cfg)
    lam = {"Aaa": 350_000_000, "Bbb": 300_000_000, "Ttt": 10_000_000, "Old1": 0}
    txn = {"Aaa": 4, "Bbb": 1, "Ttt": 9}

    def rpc(method, params):
        if method == "getMultipleAccounts":
            return {"value": [({"lamports": lam[a]} if lam[a] else None) for a in params[0]]}
        if method == "getSignaturesForAddress":
            return [{"signature": "s"}] * txn.get(params[0], 0)
        return None
    board.fetch_wallets(rpc)
    w = board.wallets
    assert w["stale"] is False and w["wallets"] == 2 and board.wallets_ver == 1
    by = {x["address"]: x for x in w["live"] + w["archive"]}
    assert by["Aaa"]["sol"] == 0.35 and by["Aaa"]["txs"] == 4 and by["Ttt"]["sol"] == 0.01
    assert by["Old1"]["sol"] == 0.0 and "txs" not in by["Old1"]  # archive gets a balance, not a tx count
    assert round(w["trading_sol"], 3) == 0.65
    board.fetch_wallets(lambda m, p: (_ for _ in ()).throw(RuntimeError("boom")))  # a failed read keeps the data
    assert board.wallets["stale"] is True and "boom" in board.wallets["error"]


def _tok_acct(amount):
    import base64
    raw = bytearray(165)
    raw[64:72] = int(amount).to_bytes(8, "little")
    return {"data": [base64.b64encode(bytes(raw)).decode(), "base64"]}


def test_fetch_pools_joins_feints_to_the_scorer_and_reads_young_reserves(cfg):
    from tradingbot import launch_live
    from tradingbot.watcher import WatchStore
    cfg.launch_book = cfg.launch_live = cfg.launch_canary = cfg.launch_sized_book = True
    now = time.time()
    j = Journal(":memory:")
    _feint(j, "canary_rw50_sz", "MintY", "alive", pool="PoolY", sol_before=70.0, sol_after=72.0)
    store = WatchStore(":memory:")
    launch_live.ensure_schema(store)
    store.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, decided_ts, liquid, liq_usd, "
                  "sol_at_decide, peak_sol, sol_vault, token_vault, exit_px, pulled_ts, ret, status) "
                  "VALUES(?,?,?,?,1,?,?,?,?,?,?,?,?,?)",
                  ("PoolY", "MintY", now - 300, now - 296, 14000.0, 70.0, 140.0, "sv", "tv", None, None, 0.2, "scored"))
    store.execute("INSERT INTO watch_market(ts, sol_usd) VALUES(?, ?)", (now - 10, 100.0))
    board = dash.Dashboard(j, cfg, store)

    rpc = lambda m, p: {"value": [_tok_acct(90_000_000_000), _tok_acct(1_000_000_000_000)]}  # 90 SOL, 1e12 tok
    board.fetch_pools(rpc)
    rows = board.pools["rows"]
    assert len(rows) == 1
    r = rows[0]
    # $25 @ m6 is the confirmation entry now: one late reserve read has no pre-confirmation sample, so there is no
    # conf price yet (pnl25 None); ret_m6 keeps the scorer's second-6 ideal for context.
    assert r["liq_probe"] == 14000.0 and r["ret_m6"] == 0.2 and r["pnl25"] is None
    assert round(r["peak_liq"]) == 28000                               # liq_usd * peak_sol/sol_at_decide
    assert round(r["live_liq"]) == 18000                               # 2 * 90 * 100 from the live read
    assert board.pools["summary"]["pools"] == 1 and board.pools["stale"] is False

    # a thrown live reserve read is best-effort now: the pools still render (not stale), only the live columns
    # are skipped -- a flaky public RPC must never hide the probe/feint pools (which come from events + watch.db).
    board.fetch_pools(lambda m, p: (_ for _ in ()).throw(RuntimeError("rpc down")))
    assert board.pools["stale"] is False and board.pools["live_read_ok"] is False
    assert board.pools["rows"] and board.pools["rows"][0]["pool"] == "PoolY"


def test_canary_rotate_validates_halts_and_launches_provision(cfg, tmp_path, monkeypatch):
    import pytest
    from tradingbot.engine import CANARY_HALT
    dash._ROTATE.update(proc=None, started=0.0, args=None)  # module global: start clean
    cfg.db_path = str(tmp_path / "j.db")
    j = Journal(cfg.db_path)
    for bad in [(0, 0.2), (11, 0.2), (5, 0.0), (5, 9.0), ("x", 0.2)]:
        with pytest.raises(ValueError):
            dash.canary_rotate(j, cfg, *bad)
    assert j.get_state(CANARY_HALT) is None  # a rejected request changes nothing

    calls = {}

    class FakeProc:
        done = False
        returncode = 0

        def poll(self):
            return 0 if self.done else None

    fp = FakeProc()

    def fake_popen(argv, **kw):
        calls["argv"] = argv
        kw["stdout"].write("provisioning the new wallets…\n")
        return fp

    monkeypatch.setattr(dash.subprocess, "Popen", fake_popen)
    res = dash.canary_rotate(j, cfg, 7, 0.25)
    assert res["started"] is True and res["wallets"] == 7
    assert j.get_state(CANARY_HALT) is not None  # halted before sweeping
    assert calls["argv"][:4] == [dash.sys.executable, "-m", "tradingbot", "provision"]
    assert "7" in calls["argv"] and "0.25" in calls["argv"] and "--apply" in calls["argv"]
    lg = dash.rotate_log(cfg)
    assert lg["running"] is True and "provisioning" in lg["log"]
    assert dash.canary_rotate(j, cfg, 7, 0.25)["started"] is False  # refused while one runs
    fp.done = True
    assert dash.rotate_log(cfg)["running"] is False and dash.rotate_log(cfg)["returncode"] == 0
    dash._ROTATE.update(proc=None, started=0.0, args=None)


def test_rotate_endpoint_is_behind_the_password(cfg, monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    cfg.health_port = 18771
    j = Journal(":memory:")
    board = dash.Dashboard(j, cfg)
    board.refresh()
    monkeypatch.setattr(dash, "canary_rotate",
                        lambda journal, c, w, s: {"started": True, "wallets": w, "sol": s, "msg": "ok"})
    server = health.start_health_server(j, cfg, board)

    def post(body, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", 18771, timeout=5)
        h = {"Content-Type": "application/json"}
        if cookie:
            h["Cookie"] = cookie
        c.request("POST", "/api/canary/rotate", body=body, headers=h)
        r = c.getresponse()
        return r.status, r.read()
    try:
        assert post(json.dumps({"wallets": 5, "sol": 0.3}).encode())[0] == 401  # no password
        code, body = post(json.dumps({"wallets": 5, "sol": 0.3}).encode(), cookie="tb_dash=s3cret")
        assert code == 200 and json.loads(body)["started"] is True and json.loads(body)["wallets"] == 5
        # a validation error comes back as a 400, not a 500
        monkeypatch.setattr(dash, "canary_rotate",
                            lambda journal, c, w, s: (_ for _ in ()).throw(ValueError("bad")))
        assert post(json.dumps({"wallets": 99, "sol": 1}).encode(), cookie="tb_dash=s3cret")[0] == 400
    finally:
        server.shutdown()


def test_canary_topup_validates_and_launches_provision_without_halting(cfg, tmp_path, monkeypatch):
    import pytest
    from tradingbot.engine import CANARY_HALT
    dash._TOPUP.update(proc=None, started=0.0, args=None)
    dash._ROTATE.update(proc=None, started=0.0, args=None)
    cfg.db_path = str(tmp_path / "j.db")
    j = Journal(cfg.db_path)
    for bad in [("x", "all"), (0.0, "all"), (9.0, "all"), (0.08, [0]), (0.08, [11]), (0.08, [])]:
        with pytest.raises(ValueError):
            dash.canary_topup(cfg, *bad)

    calls = {}

    class FakeProc:
        done = False
        returncode = 0

        def poll(self):
            return 0 if self.done else None

    fp = FakeProc()

    def fake_popen(argv, **kw):
        calls["argv"] = argv
        kw["stdout"].write("topping up wallets…\n")
        return fp

    monkeypatch.setattr(dash.subprocess, "Popen", fake_popen)
    res = dash.canary_topup(cfg, 0.08, [1, 3])
    assert res["started"] is True and res["pick"] == "1,3"
    assert j.get_state(CANARY_HALT) is None  # top-up is additive and safe mid-trade: NEVER halts the canary
    assert calls["argv"][:4] == [dash.sys.executable, "-m", "tradingbot", "provision"]
    assert "--top-up" in calls["argv"] and "0.08" in calls["argv"] and "1,3" in calls["argv"] and "--apply" in calls["argv"]
    lg = dash.topup_log(cfg)
    assert lg["running"] is True and "topping up" in lg["log"]
    assert dash.canary_topup(cfg, 0.08, "all")["started"] is False  # refused while one runs
    # a rotation is refused while the top-up runs, and vice versa
    assert dash.canary_rotate(j, cfg, 5, 0.2)["started"] is False and j.get_state(CANARY_HALT) is None
    fp.done = True
    assert dash.topup_log(cfg)["running"] is False and dash.topup_log(cfg)["returncode"] == 0
    # "all" selection passes pick=all
    res2 = dash.canary_topup(cfg, 0.1, "all")
    assert res2["pick"] == "all" and "all" in calls["argv"]
    dash._TOPUP.update(proc=None, started=0.0, args=None)


def test_topup_endpoint_is_behind_the_password(cfg, monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    cfg.health_port = 18778
    j = Journal(":memory:")
    board = dash.Dashboard(j, cfg)
    board.refresh()
    monkeypatch.setattr(dash, "canary_topup",
                        lambda c, level, wallets: {"started": True, "level": level, "pick": "all", "msg": "ok"})
    server = health.start_health_server(j, cfg, board)

    def post(body, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", 18778, timeout=5)
        h = {"Content-Type": "application/json"}
        if cookie:
            h["Cookie"] = cookie
        c.request("POST", "/api/canary/topup", body=body, headers=h)
        r = c.getresponse()
        return r.status, r.read()
    try:
        assert post(json.dumps({"level": 0.08, "wallets": "all"}).encode())[0] == 401  # no password
        code, body = post(json.dumps({"level": 0.08, "wallets": "all"}).encode(), cookie="tb_dash=s3cret")
        assert code == 200 and json.loads(body)["started"] is True
        monkeypatch.setattr(dash, "canary_topup",
                            lambda c, level, wallets: (_ for _ in ()).throw(ValueError("bad")))
        assert post(json.dumps({"level": 9, "wallets": "all"}).encode(), cookie="tb_dash=s3cret")[0] == 400
    finally:
        server.shutdown()


def test_canary_settings_endpoint_is_behind_the_password(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    cfg.health_port = 18773
    cfg.db_path = str(tmp_path / "j.db")
    j = Journal(cfg.db_path)
    board = dash.Dashboard(j, cfg)
    board.refresh()
    server = health.start_health_server(j, cfg, board)

    def req(method, body=None, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", 18773, timeout=5)
        h = {"Content-Type": "application/json"} if body is not None else {}
        if cookie:
            h["Cookie"] = cookie
        c.request(method, "/api/canary/settings", body=(json.dumps(body).encode() if body is not None else None),
                  headers=h)
        r = c.getresponse()
        return r.status, r.read()
    try:
        assert req("GET")[0] == 401  # no password
        code, body = req("GET", cookie="tb_dash=s3cret")
        menu = json.loads(body)
        assert code == 200 and "fields" in menu and "presets" in menu and "launch_canary_max_open" in menu["current"]
        # a live change applies to the shared cfg at once
        code, body = req("POST", {"changes": {"launch_canary_max_open": 9}}, cookie="tb_dash=s3cret")
        assert code == 200 and cfg.launch_canary_max_open == 9 and json.loads(body)["changed"]
        # an incoherent change is a 400, not a 500, and changes nothing
        code, _ = req("POST", {"changes": {"launch_canary_feint": True}}, cookie="tb_dash=s3cret")
        assert code == 400 and cfg.launch_canary_feint is False
        # a preset applies a coherent set
        code, body = req("POST", {"preset": "clips"}, cookie="tb_dash=s3cret")
        assert code == 200 and cfg.launch_canary_clips == 5 and json.loads(body)["preset"] == "clips"
    finally:
        server.shutdown()


def test_fly_deploy_reports_when_no_command_is_available(cfg, tmp_path, monkeypatch):
    cfg.db_path = str(tmp_path / "j.db")
    monkeypatch.setattr(dash, "_deploy_command", lambda: None)
    res = dash.fly_deploy(cfg)
    assert res["started"] is False and "git pull" in res["msg"]
    assert dash.deploy_log(cfg)["command"] is None and dash.deploy_log(cfg)["running"] is False


def test_fly_deploy_launches_the_configured_command(cfg, tmp_path, monkeypatch):
    dash._DEPLOY.update(proc=None, started=0.0)
    cfg.db_path = str(tmp_path / "j.db")
    monkeypatch.setattr(dash, "_deploy_command", lambda: ["fly", "deploy", "--detach", "-a", "supermeme"])

    class FakeProc:
        done = False
        returncode = 0

        def poll(self):
            return 0 if self.done else None

    fp = FakeProc()
    calls = {}

    def fake_popen(cmd, **kw):
        calls["cmd"] = cmd
        kw["stdout"].write("deploying…\n")
        return fp

    monkeypatch.setattr(dash.subprocess, "Popen", fake_popen)
    res = dash.fly_deploy(cfg)
    assert res["started"] is True and calls["cmd"][:2] == ["fly", "deploy"]
    lg = dash.deploy_log(cfg)
    assert lg["running"] is True and "deploying" in lg["log"] and lg["command"].startswith("fly deploy")
    assert dash.fly_deploy(cfg)["started"] is False  # refused while one runs
    fp.done = True
    assert dash.deploy_log(cfg)["running"] is False and dash.deploy_log(cfg)["returncode"] == 0
    dash._DEPLOY.update(proc=None, started=0.0)


def test_deploy_endpoint_is_behind_the_password(cfg, monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    cfg.health_port = 18775
    j = Journal(":memory:")
    board = dash.Dashboard(j, cfg)
    board.refresh()
    monkeypatch.setattr(dash, "fly_deploy", lambda c: {"started": True, "msg": "ok"})
    server = health.start_health_server(j, cfg, board)

    def req(method, path, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", 18775, timeout=5)
        c.request(method, path, headers={"Cookie": cookie} if cookie else {})
        r = c.getresponse()
        return r.status, r.read()
    try:
        assert req("POST", "/api/deploy")[0] == 401                     # no password
        assert req("GET", "/api/deploy/log")[0] == 401
        code, body = req("POST", "/api/deploy", cookie="tb_dash=s3cret")
        assert code == 200 and json.loads(body)["started"] is True
        assert req("GET", "/api/deploy/log", cookie="tb_dash=s3cret")[0] == 200
    finally:
        server.shutdown()


def test_probe_summary_counts_trips_and_survivors():
    import json as _j
    j = Journal(":memory:")
    for mint, pv, slot in (("A", "ok", 1), ("B", "ok", 2), ("C", "trapped", None)):
        j.log_event("canary_probe", mint, _j.dumps({"verdict": pv, "pool": "P" + mint, "slot_gap": slot,
                                                    "blind": True, "recovered_sol": 0.12}), strategy="canary_rw50_sz")
    j.log_event("canary_probe_verdict", "A", '{"pool": "PA", "verdict": "survivor"}', strategy="canary_rw50_sz")
    j.log_event("canary_probe_verdict", "B", '{"pool": "PB", "verdict": "rugger"}', strategy="canary_rw50_sz")
    s = dash.probe_summary(j)
    assert s["probes"] == 3 and s["survivor"] == 1 and s["rugger"] == 1 and s["decided"] == 2
    assert s["trip_rate"] == 0.5 and s["pending"] == 1  # C probed, no verdict yet
    rec = {r["token"]: r for r in s["recent"]}
    assert rec["A"]["watch"] == "survivor" and rec["C"]["watch"] is None and rec["A"]["slot_gap"] == 1
    # gap distribution for tuning the probe delay: A=gap1, B=gap2 (C has no gap); gap2_rate = trap-risk share
    assert s["gap_dist"] == {"0": 0, "1": 1, "2": 1, "3+": 0} and s["gap2_rate"] == 0.5 and s["gaps_n"] == 2


def test_canary_one_arms_exactly_one_probe():
    from tradingbot.engine import CANARY_ALLOW, CANARY_HALT
    j = Journal(":memory:")
    j.set_state(CANARY_HALT, "1")
    res = dash.canary_one(j)
    assert res["armed"] and j.get_state(CANARY_ALLOW) == "1" and j.get_state(CANARY_HALT) is None


def test_probe_endpoints_are_behind_the_password(cfg, monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    cfg.health_port = 18779
    from tradingbot.engine import CANARY_ALLOW
    j = Journal(":memory:")
    board = dash.Dashboard(j, cfg)
    board.refresh()
    server = health.start_health_server(j, cfg, board)

    def req(method, path, cookie=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", 18779, timeout=5)
        c.request(method, path, body=body, headers={"Cookie": cookie} if cookie else {})
        r = c.getresponse()
        return r.status, r.read()
    try:
        assert req("GET", "/api/canary/probes")[0] == 401  # readout needs the password
        code, body = req("GET", "/api/canary/probes", cookie="tb_dash=s3cret")
        assert code == 200 and json.loads(body)["probes"] == 0
        assert req("POST", "/api/canary/probe-one")[0] == 401
        code, _ = req("POST", "/api/canary/probe-one", cookie="tb_dash=s3cret")
        assert code == 200 and j.get_state(CANARY_ALLOW) == "1"  # one probe armed
        # probe-smoke rejects a non-direct / no-live-confirm cfg as a 400 before building any executor
        code, body = req("POST", "/api/canary/probe-smoke", cookie="tb_dash=s3cret", body=json.dumps({"mint": "X"}))
        assert code == 400 and "error" in json.loads(body)
    finally:
        server.shutdown()
