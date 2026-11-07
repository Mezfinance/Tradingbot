"""Live dashboard (11-04): the paper books' trades streamed to a browser as they happen, plus the report tables.

Served by the health server (`health.start_health_server`) on `health_port`, behind a password: the Fly secret
`DASHBOARD_TOKEN` (no secret = no dashboard; /health stays open for Fly's check). Open
`https://<app>.fly.dev/?key=<token>` once: it sets a cookie, so the plain address works afterwards.

  /             the page (no build step, one HTML file)
  /api/stream   server-sent events: `init` (every position of the running books + the launch scorer's pools of
                the last day), then `delta` (changed rows) and `fill` (new trades) every POLL_S, `summary`
                (backtest notes, downtime) every SUMMARY_S, comments as keep-alives
  /api/summary  the latest summary as JSON

11-05: the page computes every table itself (equity, return, drawdown, win rate, ideal outcomes, real vs ideal)
from the raw rows, so the tables move with each trade. One shared `Live` poller reads the rows on read-only
connections of its own (no journal lock; sqlite releases the GIL while reading) and versions each changed row,
so a page costs no database reads. The summary extras come from a low-priority child process
(`python -m tradingbot dashboard-summary PATH`)."""
from __future__ import annotations

import hmac
import json
import logging
import os
import subprocess
import sys
import threading
import time

from tradingbot.config import Config
from tradingbot.journal import Journal

log = logging.getLogger(__name__)

POLL_S = 2.0  # new fills are looked for this often
SUMMARY_S = 300  # the tables are rebuilt this often
CHILD_TIMEOUT_S = 240
SUMMARY_FILE = "dashboard-summary.json"
KEEPALIVE_S = 15
SHORTS_POLL_S = 30  # MZF Short Only feed (another Fly app) is read this often
SHORTS_TIMEOUT_S = 8
MARKET_POLL_S = 60  # the market monitor's hourly rows (watch.db market_hours) are re-read this often
FEED_BACKLOG = 60  # fills sent when a page connects
COOKIE = "tb_dash"


def token() -> str:
    return os.environ.get("DASHBOARD_TOKEN", "")


def authorised(headers, query: dict) -> bool:
    """True when the request carries the token as ?key= or in the dashboard cookie."""
    secret = token()
    if not secret:
        return False
    given = (query.get("key") or [""])[0]
    if not given:
        for part in (headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE:
                given = value
    return bool(given) and hmac.compare_digest(given, secret)


def fills_after(journal, after_id: int, limit: int = 200) -> list[dict]:
    """Buys and sells of running books with id > after_id, oldest first (archived books, `name@id`, left out)."""
    rows = journal.query(
        "SELECT f.id, f.ts, f.side, f.usd, f.reason, p.strategy, p.symbol, p.token, p.cost_usd, p.realized_usd, "
        "p.closed_at, p.opened_at FROM fills f JOIN positions p ON p.id=f.position_id "
        "WHERE f.id > ? AND instr(p.strategy, '@') = 0 ORDER BY f.id LIMIT ?", (after_id, limit))
    out = []
    for r in rows:
        ret = None
        if r["side"] == "sell" and r["closed_at"] and r["cost_usd"]:
            ret = round(100 * (r["realized_usd"] / r["cost_usd"] - 1), 1)
        out.append({"id": r["id"], "ts": r["ts"], "side": r["side"], "book": r["strategy"],
                    "symbol": r["symbol"] or (r["token"] or "")[:6], "token": r["token"],
                    "usd": round(r["usd"] or 0.0, 2), "reason": r["reason"], "ret_pct": ret,
                    "held_s": round(r["ts"] - r["opened_at"]) if r["side"] == "sell" and r["opened_at"] else None})
    return out


def last_fills(journal: Journal, n: int = FEED_BACKLOG) -> list[dict]:
    top = journal.query("SELECT MAX(id) AS m FROM fills")
    newest = top[0]["m"] if top and top[0]["m"] is not None else 0
    return fills_after(journal, max(0, newest - 5 * n), 5 * n)[-n:]


def build_summary(journal: Journal, cfg: Config, store=None, now: float | None = None) -> dict:
    """The slow-changing extras the page can't compute from the rows: backtest notes, downtime, last backup.
    (Book tables and launch outcomes are computed in the browser from the streamed rows since 11-05.)"""
    from tradingbot import model_book as mb
    from tradingbot import report
    books = mb.active_books(cfg)
    return {"built_ts": now if now is not None else time.time(),
            "expected": {b: report.EXPECTED.get(b) for b in books},
            "downtime": report.downtime(journal), "last_backup": json.loads(journal.get_state("last_backup") or "null")}


class Dashboard:
    """Keeps the latest summary (rebuilt on its own thread) and serves the stream."""

    def __init__(self, journal: Journal, cfg: Config, store=None, in_child: bool | None = None):
        self.journal, self.cfg, self.store = journal, cfg, store
        # 11-05: built inside the bot, the summary (full report + launch-status, ~1 min of Python) held the
        # GIL and the journal lock every 2 min: scorer decisions started ~40 s late, buys waited 25-50 s.
        # So it runs in its own low-priority process unless the journal is in memory (tests).
        self.in_child = (cfg.db_path != ":memory:") if in_child is None else in_child
        self.summary: dict | None = None
        self.version = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.live = Live(cfg, journal, store)
        self._external = True  # streams poll for themselves until start() runs the shared poller
        self.shorts: dict | None = None  # MZF Short Only (the shortbook app's feed), refreshed every SHORTS_POLL_S
        self.shorts_ver = 0
        self.market: dict | None = None  # the meme market monitor (market_monitor.payload), every MARKET_POLL_S
        self.market_ver = 0
        self._mdb = None

    def _build_in_child(self) -> dict:
        path = os.path.join(os.path.dirname(os.path.abspath(self.cfg.db_path)), SUMMARY_FILE)

        def low_priority():
            try:
                os.nice(19)
                with open("/proc/self/oom_score_adj", "w") as fh:
                    fh.write("1000")
            except OSError:
                pass
        res = subprocess.run([sys.executable, "-m", "tradingbot", "dashboard-summary", path],
                             env={**os.environ, "TB_DB_PATH": self.cfg.db_path},
                             preexec_fn=low_priority, timeout=CHILD_TIMEOUT_S,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"child exited {res.returncode}: {res.stderr.decode(errors='replace')[-300:]}")
        with open(path) as fh:
            return json.load(fh)

    def refresh(self) -> None:
        try:
            summary = (self._build_in_child() if self.in_child
                       else build_summary(self.journal, self.cfg, self.store))
        except Exception:  # noqa: BLE001 (keep the last good summary)
            log.exception("dashboard summary failed")
            return
        with self._lock:
            self.summary, self.version = summary, self.version + 1

    def start(self) -> "Dashboard":
        def loop():
            while True:
                self.refresh()
                if self._stop.wait(SUMMARY_S):
                    return
        def live_loop():
            while not self._stop.is_set():
                self.live.poll()
                self._stop.wait(POLL_S)

        def shorts_loop():
            while not self._stop.is_set():
                self.fetch_shorts()
                self._stop.wait(SHORTS_POLL_S)
        def market_loop():
            while not self._stop.is_set():
                self.fetch_market()
                self._stop.wait(MARKET_POLL_S)
        self._external = False
        if self.cfg.market_monitor and self.store is not None:
            threading.Thread(target=market_loop, name="dashboard-market", daemon=True).start()
        threading.Thread(target=loop, name="dashboard", daemon=True).start()
        threading.Thread(target=live_loop, name="dashboard-live", daemon=True).start()
        if self.cfg.shortbook_feed_url:
            threading.Thread(target=shorts_loop, name="dashboard-shorts", daemon=True).start()
        return self

    def fetch_shorts(self, get=None) -> None:
        """MZF Short Only: the shortbook app's read-only feed over Fly's private network. A failure keeps the
        last good data and marks it stale, so the page can say so."""
        url = self.cfg.shortbook_feed_url
        if not url:
            return
        try:
            if get is None:
                import httpx
                resp = httpx.get(url, timeout=SHORTS_TIMEOUT_S)
                resp.raise_for_status()
                data = resp.json()
            else:
                data = get(url)
            data = {**data, "fetched_ts": time.time(), "stale": False}
        except Exception as exc:  # noqa: BLE001 (the short book is another app: it may be down or redeploying)
            log.warning("dashboard: short book feed failed: %s", exc)
            data = {**(self.shorts or {}), "stale": True, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        with self._lock:
            self.shorts, self.shorts_ver = data, self.shorts_ver + 1

    def fetch_market(self) -> None:
        """The market monitor's hourly rows on a read-only connection of its own (an in-memory store in tests)."""
        from tradingbot import market_monitor as mm
        try:
            path = getattr(self.store, "path", None)
            if path and path != ":memory:":
                if self._mdb is None:
                    self._mdb = Live._open(path)
                query = lambda sql, p: self._mdb.execute(sql, p).fetchall()
            else:
                query = self.store.query
            now = time.time()
            data = mm.payload(mm.read(query, 0.0), now, since=self.cfg.market_monitor_since,
                              flags=mm.read_flags(query), status=mm.read_status(query))
        except Exception:  # noqa: BLE001 (keep the last good data)
            log.exception("dashboard: market monitor read failed")
            return
        with self._lock:
            self.market, self.market_ver = data, self.market_ver + 1

    def current(self) -> tuple[int, dict | None]:
        with self._lock:
            return self.version, self.summary

    def stream(self, write, stop: threading.Event | None = None) -> None:
        """Write server-sent events to `write(bytes)` until it raises (the page closed) or `stop` is set.

        `init` (every position of the running books + the scorer's pools of the last day), `summary` (the
        slow extras), `backlog` (recent fills), then every POLL_S: `delta` (rows that changed), `fill` (new
        buys and sells) and `summary` when rebuilt. The page computes the tables from the rows, so they move
        with every trade. All of it comes from the shared `Live` poller: a page costs no database reads."""
        def send(event: str, data) -> None:
            write(f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())

        stop = stop or threading.Event()
        live = self.live
        if live.ver == 0:
            live.poll()
        ver, init = live.snapshot()
        send("init", init)
        version, summary = self.current()
        if summary is not None:
            send("summary", summary)
        backlog = live.fills[-FEED_BACKLOG:]
        send("backlog", backlog)
        shorts_ver = 0
        if self.cfg.shortbook_feed_url and self.shorts is not None:
            send("shorts", self.shorts)
            shorts_ver = self.shorts_ver
        market_ver = 0
        if self.market is not None:
            send("market", self.market)
            market_ver = self.market_ver
        last_fill = backlog[-1]["id"] if backlog else live.last_fill
        quiet = time.time()
        while not stop.is_set():
            if self._external:  # no poller thread (tests, or start() not called): poll here
                live.poll()
            sent = False
            new_ver, delta = live.delta(ver)
            if delta:
                send("delta", delta)
                sent = True
            ver = new_ver
            for f in [f for f in live.fills if f["id"] > last_fill]:
                send("fill", f)
                last_fill = f["id"]
                sent = True
            v, summary = self.current()
            if v != version and summary is not None:
                send("summary", summary)
                version = v
                sent = True
            if self.shorts_ver != shorts_ver and self.shorts is not None:
                send("shorts", self.shorts)
                shorts_ver = self.shorts_ver
                sent = True
            if self.market_ver != market_ver and self.market is not None:
                send("market", self.market)
                market_ver = self.market_ver
                sent = True
            if sent:
                quiet = time.time()
            elif time.time() - quiet > KEEPALIVE_S:
                write(b": keep-alive\n\n")
                quiet = time.time()
            stop.wait(POLL_S)


POS_COLS = ("id", "strategy", "symbol", "token", "cost_usd", "realized_usd", "qty", "entry_price", "opened_at",
            "closed_at", "exit_reason")
LAUNCH_COLS = ("pool", "token_mint", "create_ts", "decided_ts", "liquid", "ret", "status", "pulled_ts", "liq_usd",
               "top2", "top5", "top2h", "top5h", "low_h", "risk_pct", "rw_pct", "rw_low_pct", "rw_low6",
               "rw24_pct", "rw24_low_pct")
LAUNCH_KEEP_S = 86400  # the scorer's pools sent to the page (it windows them 1/6/24 h)
STUCK_AFTER_S = 0  # as engine.STUCK_AFTER_S: a position with no route out counts as $0 at once


FAMILIES = (  # book families for the dashboard's family view, in display order
    ("Canary", "real money: launchrw50_sz picks bought for real at a fraction of the twin's band sizes, judged on swap prices against the twin"),
    ("Launch picks", "the second-4 launch scorer's top 2% / 5% picks"),
    ("Low decile", "the history model's bottom 20% (D1-D2), minus the riskiest early pullers"),
    ("Rug-vs-win", "skip the riskiest half by P(rug before the exit) - P(+50% win); study rule, 24 h rule and a size-by-band twin"),
    ("Early", "hand-made pools at first sight, out by pool minute 5-7"),
    ("model5", "the 5-minute snapshot model's short-hold books"))


def start_cash(cfg, name: str) -> float:
    """What a book started with: the canary has its own small budget, every other model book the same cash."""
    from tradingbot import model_book as mb
    return cfg.launch_canary_budget_usd if name in mb.CANARY_BOOKS else cfg.model_book_cash_usd


def family_of(name: str) -> str:
    from tradingbot import model_book as mb
    if name in mb.CANARY_BOOKS:
        return "Canary"
    if name in mb.LAUNCH_RW_CUT or name in mb.LAUNCH_RW24_CUT or name in mb.LAUNCH_SIZED_CUT:
        return "Rug-vs-win"
    if name in mb.LAUNCH_LOW_CUT:
        return "Low decile"
    if name in mb.LAUNCH_CUT:
        return "Launch picks"
    if name in mb.EARLY_BOOKS:
        return "Early"
    return "model5"


class Live:
    """Raw rows for the page, polled every POLL_S on read-only connections of their own (never the journal's
    lock, and sqlite releases the GIL while it reads). Each changed row gets a new version number, so every
    open page is sent only what changed since its last version."""

    def __init__(self, cfg: Config, journal: Journal | None = None, store=None):
        from tradingbot import model_book as mb
        self.cfg = cfg
        self.books = list(mb.active_books(cfg))
        self.flag_of = {b: mb.LAUNCH_CUT.get(b) for b in self.books}
        self.canary = set(mb.CANARY_BOOKS)
        self.twin = {b: mb.CANARY_TWIN if b in mb.CANARY_BOOKS else None for b in self.books}
        self.holds = {b: mb.hold_min(cfg, b) for b in self.books}
        self.journal = journal
        self.jdb = self._open(cfg.db_path) if cfg.db_path != ":memory:" else None
        path = getattr(store, "path", None)
        if path and path != ":memory:" and os.path.exists(path):
            self.wdb = self._open(path)
        else:  # in-memory store (tests): its own connection
            self.wdb = getattr(store, "db", None)
        self.lock = threading.Lock()
        self.ver = 0
        self.pos: dict[int, tuple[int, list]] = {}
        self.launch: dict[str, tuple[int, list]] = {}
        self.blocked: dict[tuple, list[float]] = {}
        self.fills: list[dict] = []
        self.last_fill = 0
        self.last_event = None
        self.health: dict | None = None
        self.health_ver = 0
        self._health_ts = 0.0

    @staticmethod
    def _open(path: str):
        import sqlite3
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2, check_same_thread=False)
        db.row_factory = sqlite3.Row
        return db

    def _q(self, sql: str, params: tuple = ()) -> list:
        if self.jdb is not None:
            return self.jdb.execute(sql, params).fetchall()
        return self.journal.query(sql, params) if self.journal is not None else []

    def query(self, sql: str, params: tuple = ()) -> list:  # lets fills_after read through this connection
        return self._q(sql, params)

    def _bump(self, table: dict, key, row: list) -> None:
        old = table.get(key)
        if old is None or old[1] != row:
            self.ver += 1
            table[key] = (self.ver, row)

    def poll(self) -> None:
        try:
            with self.lock:
                self._poll_events()
                self._poll_positions()
                self._poll_fills()
                self._poll_launch()
                if time.time() - self._health_ts > 10 and self.journal is not None:
                    from tradingbot import health
                    h = health.status(self.journal, self.cfg)[1]
                    self._health_ts = time.time()
                    if h != self.health:
                        self.health, self.health_ver = h, self.ver + 1
                        self.ver += 1
        except Exception:  # noqa: BLE001 (a failed read is retried next poll)
            log.exception("dashboard live poll failed")

    def _poll_events(self) -> None:
        if self.last_event is None:  # first pass: the past week's blocked exits, then only new events by id
            top = self._q("SELECT MAX(id) AS m FROM events")
            rows = self._q("SELECT strategy, token, ts FROM events WHERE kind='exit_blocked' AND ts > ?",
                           (time.time() - 7 * 86400,))
            self.last_event = (top[0]["m"] if top else None) or 0
        else:
            rows = self._q("SELECT id, strategy, token, ts FROM events WHERE id > ? AND kind='exit_blocked'",
                           (self.last_event,))
            top = self._q("SELECT MAX(id) AS m FROM events")
            self.last_event = (top[0]["m"] if top else None) or self.last_event
        for r in rows:
            self.blocked.setdefault((r["strategy"], r["token"]), []).append(r["ts"])

    def _blocked_since(self, book: str, token: str, opened: float | None) -> float | None:
        ts = [t for t in self.blocked.get((book, token), ()) if t >= (opened or 0)]
        return min(ts) if ts else None

    def _poll_positions(self) -> None:
        if not self.books:
            return
        marks = ",".join("?" * len(self.books))
        top = max(self.pos) if self.pos else 0
        recent = time.time() - 60
        rows = self._q(f"SELECT {', '.join(POS_COLS)} FROM positions WHERE strategy IN ({marks}) "
                       f"AND (id > ? OR closed_at IS NULL OR closed_at > ?)", (*self.books, top, recent))
        for r in rows:
            row = [r[c] for c in POS_COLS]
            row.append(self._blocked_since(r["strategy"], r["token"], r["opened_at"]) if r["closed_at"] is None
                       else None)
            row.append(self._swap(r["id"]) if r["strategy"] in self.canary else None)
            self._bump(self.pos, r["id"], row)

    def _swap(self, pid: int) -> float | None:
        """The canary's result from its fill prices alone: a 1-cent trade's net result is all fees and rent."""
        from tradingbot.canary_rugs import swap_return
        ret = swap_return(self._q("SELECT side, price, qty FROM fills WHERE position_id=? ORDER BY ts", (pid,)))
        return None if ret is None else round(ret, 5)

    def _poll_fills(self) -> None:
        if not self.fills and not self.last_fill:
            top = self._q("SELECT MAX(id) AS m FROM fills")
            newest = (top[0]["m"] if top else None) or 0
            self.fills = fills_after(self, max(0, newest - 5 * FEED_BACKLOG), 5 * FEED_BACKLOG)[-FEED_BACKLOG:]
            self.last_fill = self.fills[-1]["id"] if self.fills else newest
            return
        new = fills_after(self, self.last_fill)
        if new:
            self.fills = (self.fills + new)[-300:]
            self.last_fill = new[-1]["id"]

    def _poll_launch(self) -> None:
        if self.wdb is None:
            return
        from tradingbot import launch_live
        cols = {r[1] for r in self.wdb.execute("PRAGMA table_info(watch_launch_live)")}
        if not cols:
            return
        sel = ", ".join(c if c in cols else f"NULL AS {c}" for c in LAUNCH_COLS)
        rows = self.wdb.execute(f"SELECT {sel} FROM watch_launch_live WHERE decided_ts > ?",
                                (time.time() - LAUNCH_KEEP_S,)).fetchall()
        for r in rows:
            flags = [k for k in ("top2", "top5", "top2h", "top5h") if r[k]]
            for fl in (launch_live.low_flags(r["low_h"], r["risk_pct"]),
                       launch_live.rw_flags(launch_live.rw_low(r), r["rw_pct"], r["rw_low_pct"]),
                       launch_live.rw24_flags(r["low_h"], r["rw24_pct"], r["rw24_low_pct"])):
                flags += [k for k, on in fl.items() if on]
            ret = r["ret"]
            self._bump(self.launch, r["pool"], [r["pool"], r["token_mint"], r["create_ts"], r["decided_ts"],
                                                1 if r["liquid"] else 0, None if ret is None else round(ret, 4),
                                                r["status"], r["pulled_ts"], r["liq_usd"], flags])
        cut = time.time() - LAUNCH_KEEP_S - 3600
        for k in [k for k, (_, row) in self.launch.items() if (row[3] or 0) < cut]:
            del self.launch[k]

    def snapshot(self) -> tuple[int, dict]:
        with self.lock:
            return self.ver, {
                "now": time.time(), "cash": self.cfg.model_book_cash_usd, "stuck_after_s": STUCK_AFTER_S,
                "shorts_enabled": bool(self.cfg.shortbook_feed_url),
                "market_enabled": bool(self.cfg.market_monitor),
                "books": [{"name": b, "hold_min": self.holds[b], "flag": self.flag_of[b], "family": family_of(b),
                           "cash": start_cash(self.cfg, b), "twin": self.twin[b]}
                          for b in self.books],
                "families": [{"name": n, "about": a} for n, a in FAMILIES],
                "pos_cols": [*POS_COLS, "blocked_since", "swap_ret"], "positions": [row for _, row in self.pos.values()],
                "launch_cols": ["pool", "token", "create_ts", "decided_ts", "liquid", "ret", "status",
                                "pulled_ts", "liq_usd", "flags"],
                "launch": [row for _, row in self.launch.values()], "launch_keep_s": LAUNCH_KEEP_S,
                "health": self.health}

    def delta(self, since: int) -> tuple[int, dict | None]:
        with self.lock:
            if self.ver == since:
                return since, None
            out = {"now": time.time(),
                   "positions": [row for v, row in self.pos.values() if v > since],
                   "launch": [row for v, row in self.launch.values() if v > since]}
            if self.health_ver > since:
                out["health"] = self.health
            return self.ver, out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MZFq Quant Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800;900&display=swap" rel="stylesheet">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' fill='%230A0A0A'/%3E%3Cpolygon points='24,26 76,26 76,39 38,61 76,61 76,74 24,74 24,61 62,39 24,39' fill='%23C8FF4D'/%3E%3C/svg%3E">
<style>
/* MZF Protocol Design System, direction Signal: ink first, lime as a fill only, 2px rules, square corners. */
:root{--ground-app:#f2f2f0;--surface:#fff;--sunken:#f2f2f0;--band:#0a0a0a;--ink:#0a0a0a;--ink-2:#4a4a4a;--ink-3:#6b6b6b;
--on-band:#fff;--on-band-2:#b5b5b5;--lime:#c8ff4d;--lime-hover:#b5f22e;--lime-tint:#eeffc2;--on-lime:#0a0a0a;--rule:#0a0a0a;
--rule-soft:#d9d9d6;--pos:#157347;--neg:#c62828;--warn:#b45309;--focus:#0a0a0a;
--sans:"Archivo",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--ground-app:#121212;--surface:#161616;--sunken:#1e1e1e;--band:#f2f2f0;--ink:#fff;
--ink-2:#b5b5b5;--ink-3:#8a8a8a;--on-band:#0a0a0a;--on-band-2:#4a4a4a;--lime-tint:#2a3313;--rule:#fff;--rule-soft:#2e2e2e;
--pos:#5cd68a;--neg:#ff6b6b;--warn:#f5a524;--focus:#c8ff4d}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground-app);color:var(--ink);font:500 16px/1.5 var(--sans)}
.num{font-variant-numeric:tabular-nums}
.label{font-size:12px;line-height:1.2;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.mono{font:500 13px/1.4 var(--mono)}
header{display:flex;flex-wrap:wrap;gap:16px;align-items:center;justify-content:space-between;padding:20px 32px;
background:var(--surface);border-bottom:2px solid var(--rule)}
.lockup{display:flex;align-items:center;gap:12px}
.lockup svg{width:40px;height:40px;flex:none}.mk-sq{fill:#0a0a0a}.mk-z{fill:#c8ff4d}
@media (prefers-color-scheme:dark){.mk-sq{fill:#c8ff4d}.mk-z{fill:#0a0a0a}}
.word{font-weight:900;font-size:26px;letter-spacing:-.04em;line-height:1}
.desc{font-weight:700;font-size:11px;letter-spacing:.22em;line-height:1;margin-top:4px}
.status{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.tag{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border:2px solid var(--rule);background:var(--surface);
color:var(--ink);font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}
.tag.lime{background:var(--lime);color:var(--on-lime);border-color:var(--rule)}
.tag.ink{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.tag.neg{color:var(--neg);border-color:var(--neg)}
.dot{width:8px;height:8px;background:currentColor}
main{max-width:1360px;margin:0 auto;padding:32px;display:grid;gap:24px}
main[hidden],[hidden]{display:none!important}
h1{margin:0;font-size:30px;line-height:1.05;font-weight:800;letter-spacing:-.02em}
.lead{margin:6px 0 0;color:var(--ink-2);font-size:14px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.stat{padding:20px;border:2px solid var(--rule);background:var(--surface);display:flex;flex-direction:column;gap:8px;min-width:0}
.stat .label{color:var(--ink-2)}
.stat .v{font-size:36px;line-height:1;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.stat .h{font-size:12px;line-height:1.4;font-weight:600;color:var(--ink-2)}
.stat.lime{background:var(--lime);color:var(--on-lime)}.stat.lime .label,.stat.lime .h{color:var(--on-lime)}
.cols{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:24px;align-items:start}
@media (max-width:1100px){.cols{grid-template-columns:minmax(0,1fr)}}
.card{border:2px solid var(--rule);background:var(--surface);padding:28px;min-width:0}
.card h2{margin:0 0 4px;font-size:24px;line-height:1.15;font-weight:800;letter-spacing:-.02em}
.card .sub{margin:0 0 16px;color:var(--ink-2);font-size:14px}
.eyebrow{color:var(--ink-2);margin:0 0 6px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th{font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);text-align:right;
padding:8px 9px;border-bottom:2px solid var(--rule);white-space:nowrap;cursor:pointer;user-select:none}
td{padding:10px 9px;text-align:right;border-bottom:1px solid var(--rule-soft);white-space:nowrap}
th:first-child,td:first-child{text-align:left;padding-left:0}th:last-child,td:last-child{padding-right:0}
tr.lead-row td{background:var(--lime-tint)}
.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--ink-3)}
.book{font-weight:700}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:24px}
@media (max-width:520px){.two{grid-template-columns:minmax(0,1fr)}main{padding:16px}header{padding:16px}.card{padding:20px}.stat .v{font-size:30px}}
/* live feed: an inverted band */
.feed{background:var(--band);color:var(--on-band);border:2px solid var(--band);padding:28px}
.feed h2{margin:0;font-size:24px;line-height:1.15;font-weight:800;letter-spacing:-.02em}
.feed .sub{margin:4px 0 16px;color:var(--on-band-2);font-size:14px}
.feed ul{list-style:none;margin:0;padding:0;max-height:72vh;overflow-y:auto}
.feed li{display:grid;grid-template-columns:64px 1fr auto;gap:8px;align-items:center;padding:10px 0;
border-bottom:1px solid color-mix(in srgb,var(--on-band) 18%,transparent);font-size:14px}
.feed li.new{animation:flash 2.4s ease-out}
@keyframes flash{from{background:var(--lime);color:var(--on-lime)}to{background:transparent}}
.feed .t{color:var(--on-band-2)}
.feed .bk{display:block;color:var(--on-band-2);font-size:12px;font-weight:600}
.side{font-size:11px;font-weight:800;letter-spacing:.06em;padding:2px 6px;margin-right:6px;border:2px solid var(--on-band)}
.side.buy{background:var(--lime);color:var(--on-lime);border-color:var(--lime)}
.r{font-weight:800;font-variant-numeric:tabular-nums}
.feed .pos{color:#5cd68a}.feed .neg{color:#ff6b6b}
@media (prefers-color-scheme:dark){.feed .pos{color:#157347}.feed .neg{color:#c62828}}
footer{max-width:1360px;margin:0 auto;padding:0 32px 32px;color:var(--ink-3);font-size:12px;font-weight:600}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.win{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.win .label{margin-right:4px}
.win button{min-height:36px;padding:0 12px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:700 13px/1 inherit;font-family:inherit;cursor:pointer;border-radius:0}
.win button.on{background:var(--lime);color:var(--on-lime)}.win button:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.tabs{display:flex;gap:0;flex-wrap:wrap;background:var(--surface);border-bottom:2px solid var(--rule);padding:0 32px}
.tabs a{display:inline-flex;align-items:center;min-height:44px;padding:0 16px;font-size:14px;font-weight:700;color:var(--ink);
text-decoration:none;border-right:2px solid var(--rule)}.tabs a:first-child{border-left:2px solid var(--rule)}
.tabs a.on{background:var(--lime);color:var(--on-lime)}.tabs a:hover:not(.on){background:var(--sunken)}
@media (max-width:520px){.tabs{padding:0 16px}.tabs a{padding:0 10px;font-size:13px}}
a.lk{color:inherit;text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px}
a.lk:hover{text-decoration-thickness:2px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px;align-items:center}
.filters select,.filters input,.more button{min-height:44px;padding:0 12px;border:2px solid var(--rule);background:var(--surface);
color:var(--ink);font:600 14px/1 var(--sans);border-radius:0}
.filters input{min-width:180px}.more{margin:16px 0 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.more button{cursor:pointer;font-weight:700}.more button:hover{background:var(--sunken)}
.tag.warn{color:var(--warn);border-color:var(--warn)}
.tabs.sub{background:var(--sunken)}.tabs.sub a{min-height:40px;font-size:13px}
.progs{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}
@media (max-width:520px){.progs{grid-template-columns:minmax(0,1fr)}}
.prog h2{font-size:30px!important;line-height:1.05!important}.prog h2 a{color:inherit;text-decoration:none}.prog h2 a:hover{text-decoration:underline}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:12px 0 20px}
.kv div{border:2px solid var(--rule);padding:12px;display:flex;flex-direction:column;gap:4px;min-width:0}
.kv b{font-size:22px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.kv span{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}
.note{margin:0 0 16px;padding:12px 16px;border:2px solid var(--warn);color:var(--warn);font-size:14px;font-weight:600}
th.l,td.l{text-align:left}.stat .v.nm{font-size:26px;letter-spacing:-.02em}.tag.sm{font-size:11px;padding:2px 6px}
.fams{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}
.fams .card h2 a{color:inherit;text-decoration:none}.fams .card h2 a:hover{text-decoration:underline}
.mini td,.mini th{padding:7px 6px}
.chart svg{width:100%;height:260px;display:block}
.chart .ax{font:600 11px var(--sans);fill:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:10px 0 0;font-size:13px;font-weight:600}
.legend span{display:inline-flex;align-items:center;gap:6px}.legend svg{width:28px;height:8px}
</style></head><body>
<header>
<div class="lockup"><svg viewBox="0 0 100 100" aria-hidden="true"><rect class="mk-sq" width="100" height="100"/><polygon class="mk-z" points="24,26 76,26 76,39 38,61 76,61 76,74 24,74 24,61 62,39 24,39"/></svg>
<div><div class="word">MZFq</div><div class="desc">QUANT DESK</div></div></div>
<div class="status"><span id="conn" class="tag"><span class="dot"></span>Connecting</span><span id="health" class="tag">Bot</span>
<span id="built" class="label muted"></span></div>
</header>
<nav class="tabs" aria-label="Programs"><a href="#/" data-v="programs">MZFq</a><a href="#/supermeme" data-v="supermeme" id="tab-meme">Supermeme</a><a href="#/shorts" data-v="shorts">MZF Short Only</a></nav>
<nav class="tabs sub" id="subtabs" aria-label="Supermeme"><a href="#/supermeme" data-v="overview">Overview</a><a href="#/ledger" data-v="ledger">Trade ledger</a><a href="#/books" data-v="books">Books &amp; families</a><a href="#/market" data-v="market">Market</a></nav>
<main id="v-programs" hidden>
<div><p class="label eyebrow">Quant programs · paper trading</p><h1>MZFq</h1>
<p class="lead">Each program holds its strategies; Supermeme's are grouped into families of books. Everything here is paper trading.</p></div>
<div class="stats" id="p-stats"></div>
<div class="progs" id="progs"></div>
</main>
<main id="v-shorts" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/">MZFq</a> · program</p><h1>MZF Short Only</h1>
<p class="lead" id="sh-about">Short-only strategies on centralised-exchange perps.</p></div>
<div id="sh-note"></div>
<div class="stats" id="sh-stats"></div>
<section class="card"><p class="label eyebrow">Strategy</p><h2 id="sh-strat">Strategy 1: listing shorts</h2><p class="sub">Per group, against the research expectation per trade (after costs and funding). Judge after 30+ closed trades.</p><div class="scroll"><table id="sh-groups"></table></div></section>
<section class="card chart"><p class="label eyebrow">Equity</p><h2>Net asset value</h2><p class="sub">Hourly NAV (cash + open shorts marked to market, funding and fees included). Dashed: the starting capital.</p><div id="sh-chart"></div></section>
<section class="card"><p class="label eyebrow">Positions</p><h2>Open shorts</h2><div class="scroll"><table id="sh-open"></table></div></section>
<section class="card"><p class="label eyebrow">Positions</p><h2>Closed shorts</h2><div class="scroll"><table id="sh-closed"></table></div></section>
<section class="card"><p class="label eyebrow">Detector</p><h2>Recent listings</h2><p class="sub" id="sh-lcount"></p><div class="scroll"><table id="sh-listings"></table></div></section>
</main>
<main id="v-overview">
<div><p class="label eyebrow"><a class="lk" href="#/">MZFq</a> · Supermeme · paper books on Solana launches, simulated fills on live Jupiter quotes</p><h1>Supermeme</h1>
<p class="lead">Every book starts at $10,000. Trades and every table update live, within two seconds of a trade.</p></div>
<div class="stats">
<div class="stat lime"><span class="label">Leading book</span><span class="v nm" id="s-lead">-</span><span class="h" id="s-lead-h"></span></div>
<div class="stat"><span class="label">P&amp;L, all books</span><span class="v num" id="s-pnl">-</span><span class="h" id="s-pnl-h"></span></div>
<div class="stat"><span class="label">Closed trades</span><span class="v num" id="s-trades">-</span><span class="h" id="s-trades-h"></span></div>
<div class="stat"><span class="label">Trades, last hour</span><span class="v num" id="s-rate">0</span><span class="h">Closed in the past 60 minutes</span></div>
</div>
<div class="cols">
<section class="card"><p class="label eyebrow">Books</p><h2>Results by book</h2>
<p class="sub">Sorted by return. Click a header to sort. Open trades count at cost until sold; a trade with no route out counts as $0 until a later sale gets through. Rugged = closed at −80% or worse.</p>
<div class="scroll"><table id="books"></table></div></section>
<aside class="feed"><p class="label" style="color:var(--on-band-2);margin:0 0 6px">Live</p><h2>Trades</h2>
<p class="sub" id="feed-sub">Newest first</p><ul id="feed"></ul></aside>
</div>
<div class="win" id="win"><span class="label">Launch window</span><button data-w="3600">1 h</button><button data-w="21600" class="on">6 h</button><button data-w="86400">24 h</button></div>
<div class="two">
<section class="card"><p class="label eyebrow">Launch scorer · last <span id="lh">6</span> h</p><h2>Ideal trades</h2>
<p class="sub">Pool price at second ~6.5 in, minute 6 out, 6% cost. The yardstick the books are measured against.</p>
<div class="scroll"><table id="outcomes"></table></div></section>
<section class="card"><p class="label eyebrow">Execution · last <span id="lh2">6</span> h</p><h2>Real against ideal</h2>
<p class="sub">Each book's fills against the ideal trade on the same pools, and the signals it did not fill.</p>
<div class="scroll"><table id="gaps"></table></div></section>
</div>
</main>
<main id="v-market" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · meme market monitor · hand-made pumpswap pools</p><h1>Market</h1>
<p class="lead">How the launch market is behaving, hour by hour, against the past week: who is launching, how fast they pull, how much buying arrives, and whether our edge is holding. Ideal = the scorer's prices (second ~6.5 in, minute 6 out, 6% cost).</p></div>
<div id="mk-note"></div>
<div class="stats" id="mk-stats"></div>
<section class="card"><p class="label eyebrow">Shifts</p><h2>Moved away from the past week</h2><p class="sub" id="mk-shift-sub"></p><div class="scroll"><table id="mk-shifts"></table></div></section>
<section class="card"><p class="label eyebrow">Flags</p><h2>Saved flags</h2><p class="sub" id="mk-flag-sub"></p><div class="scroll"><table id="mk-flags"></table></div>
<h2 style="margin-top:24px">What the flags meant for sz</h2><p class="sub">Since sz's first trade: sz (size-weighted) and the market (ideal mean) in each measure's flagged hours against every other hour. A big negative gap means that kind of shift hurts sz.</p><div class="scroll"><table id="mk-flagsum"></table></div></section>
<div class="two" id="mk-charts"></div>
<section class="card"><p class="label eyebrow">Every measure</p><h2>Last hours against the past week</h2><p class="sub" id="mk-cmp-sub"></p><div class="scroll"><table id="mk-cmp"></table></div></section>
<section class="card"><p class="label eyebrow">Hourly</p><h2>Hour by hour</h2><p class="sub">Newest first, UTC. Liquid pools by liquidity at second 4 in sz's bands; sz = launchrw50_sz's real trades, size-weighted.</p><div class="scroll"><table id="mk-hours"></table></div></section>
</main>
<main id="v-ledger" hidden>
<div><p class="label eyebrow">Every trade of the running books</p><h1>Trade ledger</h1>
<p class="lead">Newest first. Open trades count at cost until sold; a sale with no route out counts as $0 until a later sale gets through.</p></div>
<section class="card"><div class="filters">
<select id="lf-fam" aria-label="Family"></select><select id="lf-book" aria-label="Book"></select>
<select id="lf-status" aria-label="Status"><option value="">Every status</option><option>Open</option><option>Exit blocked ($0)</option><option>Won</option><option>Lost</option></select>
<input id="lf-q" type="search" placeholder="Token or symbol" aria-label="Token or symbol"></div>
<div class="scroll"><table id="ledger"></table></div>
<p class="more"><button id="ledger-more">Show 200 more</button><span id="ledger-count" class="label muted"></span></p></section>
</main>
<main id="v-books" hidden>
<div><p class="label eyebrow">Paper books by family</p><h1>Books &amp; families</h1>
<p class="lead">A family is a group of books trading one idea with different settings. Open a family to compare its books, or a book for its own equity and trades.</p></div>
<div class="fams" id="fams"></div>
</main>
<main id="v-family" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · <a class="lk" href="#/books">Books &amp; families</a> · family</p><h1 id="f-name"></h1><p class="lead" id="f-about"></p></div>
<div class="stats" id="f-stats"></div>
<section class="card"><p class="label eyebrow">Books</p><h2>Books in this family</h2><p class="sub">Click a book to open it.</p><div class="scroll"><table id="f-books"></table></div></section>
<section class="card chart"><p class="label eyebrow">Equity</p><h2>Equity by book</h2><p class="sub">Each book's equity after every closed trade, from $10,000.</p><div id="f-chart"></div></section>
<section class="card"><p class="label eyebrow">Trades</p><h2>Family ledger</h2><div class="scroll"><table id="f-ledger"></table></div>
<p class="more"><button id="f-more">Show 200 more</button><span id="f-count" class="label muted"></span></p></section>
</main>
<main id="v-book" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · <a class="lk" id="b-fam" href="#/books"></a></p><h1 id="b-name"></h1><p class="lead" id="b-about"></p></div>
<div class="stats" id="b-stats"></div>
<section class="card chart"><p class="label eyebrow">Equity</p><h2>Equity curve</h2><p class="sub">Equity after every closed trade, from $10,000. Dashed: the starting $10,000.</p><div id="b-chart"></div></section>
<section class="card" id="b-gap-card"><p class="label eyebrow">Execution · last <span id="b-lh">6</span> h (launch window on the overview)</p><h2>Real against ideal</h2>
<p class="sub">This book's fills against the scorer's ideal trade on the same pools, and the flagged pools it did not buy.</p><div class="scroll"><table id="b-gap"></table></div></section>
<section class="card" id="b-swap-card" hidden><p class="label eyebrow">Real money · fill prices only</p><h2>Swap result against the twin</h2>
<p class="sub">At a cent a trade the net result is all fees and account rent, so the canary is judged on what the pool paid: sell price over buy price. The twin is the paper book it copies, on the same tokens.</p>
<div class="stats" id="b-swap-stats"></div><div class="scroll"><table id="b-swap"></table></div></section>
<section class="card"><p class="label eyebrow">Trades</p><h2>Book ledger</h2><div class="scroll"><table id="b-ledger"></table></div>
<p class="more"><button id="b-more">Show 200 more</button><span id="b-count" class="label muted"></span></p></section>
</main>
<footer>MZFq · paper trading only, no funds at risk · <span id="foot"></span></footer>
<script>
const $=id=>document.getElementById(id);
const fmt=(v,d=1,s="")=>v==null||Number.isNaN(v)?"-":(typeof v==="number"?v.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}):v)+s;
const sgn=(v,d=1,s="%")=>v==null||Number.isNaN(v)?"-":(v>0?"+":v<0?"−":"")+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d})+s;
const cls=v=>v==null?"":v>0?"pos":v<0?"neg":"";
const esc=t=>String(t??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
/* mean trade weighted by size: total P&L / total cost (a plain average counts a $25 trade like a $250 one) */
const wmean=tr=>{const c=tr.reduce((a,x)=>a+(x.cost||0),0);return c?100*tr.reduce((a,x)=>a+x.pnl,0)/c:null};
const median=a=>{if(!a.length)return null;const b=[...a].sort((x,y)=>x-y),m=b.length>>1;return b.length%2?b[m]:(b[m-1]+b[m])/2};
const S={market:null,marketEnabled:false,shorts:null,shortsEnabled:false,pos:new Map(),launch:new Map(),books:[],fams:[],cash:10000,stuck:0,pc:[],lc:[],skew:0,health:null,extra:null};
let sortKey="return_pct",sortDir=-1,win=21600,pending=false;
const R={v:"overview",arg:null},LIM={ledger:200,b:200,f:200};
const now=()=>Date.now()/1000-S.skew;
const obj=(cols,row)=>{const o={};cols.forEach((c,i)=>o[c]=row[i]);return o};
const bookLink=n=>`<a class="lk book" href="#/book/${encodeURIComponent(n)}">${esc(n)}</a>`;
const famLink=n=>`<a class="lk" href="#/family/${encodeURIComponent(n)}">${esc(n)}</a>`;
const famOf=n=>(S.books.find(b=>b.name===n)||{}).family||"";
const when=ts=>{const d=new Date(ts*1000);return d.toISOString().slice(5,10)+" "+d.toISOString().slice(11,19)};
const dur=s=>s==null?"-":s<90?Math.round(s)+"s":s<5400?Math.round(s/60)+"m":(s/3600).toFixed(1)+"h";
function merge(d){if(d.now)S.skew=Date.now()/1000-d.now;
(d.positions||[]).forEach(r=>{const o=obj(S.pc,r);S.pos.set(o.id,o)});
(d.launch||[]).forEach(r=>{const o=obj(S.lc,r);S.launch.set(o.pool,o)});
if("health" in d&&d.health)S.health=d.health}
/* one trade: status and result; stuck = no route out for S.stuck seconds, valued at $0 */
function trade(p,t){const cost=p.cost_usd||0,real=p.realized_usd||0;let st,pnl=null,end=null;
if(p.closed_at){end=p.closed_at;pnl=real-cost;st=pnl>0?"Won":"Lost"}
else if(p.blocked_since&&t-p.blocked_since>=S.stuck){end=p.blocked_since+S.stuck;pnl=real-cost;st="Exit blocked ($0)"}
else if(p.blocked_since){st="Exit blocked"}else{st="Open"}
return{id:p.id,book:p.strategy,sym:p.symbol||(p.token||"").slice(0,6),token:p.token,opened:p.opened_at,end,cost,pnl,
pct:pnl==null||!cost?null:100*pnl/cost,st,reason:p.exit_reason,held:(end||t)-p.opened_at,open:!p.closed_at&&st!=="Exit blocked ($0)",real,p}}
function trades(pred){const t=now(),out=[];for(const p of S.pos.values()){const x=trade(p,t);if(!pred||pred(x))out.push(x)}
return out.sort((a,b)=>b.opened-a.opened)}
function bookStats(b){const t=now(),c=b.cash||S.cash;let equity=c,open=0,started=null;const tr=[];
for(const p of S.pos.values()){if(p.strategy!==b.name)continue;started=started==null?p.opened_at:Math.min(started,p.opened_at);
const x=trade(p,t);
if(x.end!=null){tr.push({t:x.end,ret:x.cost?x.pnl/x.cost:0,pnl:x.pnl,cost:x.cost});equity+=x.pnl}
else{open++;equity+=x.real+(p.qty||0)*(p.entry_price||0)-x.cost}}
tr.sort((x,y)=>x.t-y.t);let eq=c,peak=c,dd=0;const pts=started==null?[]:[[started,c]];
for(const x of tr){eq+=x.pnl;peak=Math.max(peak,eq);dd=Math.max(dd,(peak-eq)/peak);pts.push([x.t,eq])}
const rets=tr.map(x=>x.ret*100),gain=tr.filter(x=>x.pnl>0).reduce((a,x)=>a+x.pnl,0),loss=-tr.filter(x=>x.pnl<0).reduce((a,x)=>a+x.pnl,0);
const rugs=tr.filter(x=>x.ret<=-0.8).length;
return{name:b.name,family:b.family,hold_min:b.hold_min,cash:c,started,equity,return_pct:(equity/c-1)*100,trades:tr.length,open,
win_rate:tr.length?tr.filter(x=>x.ret>0).length/tr.length:null,median_pct:median(rets),mean_pct:wmean(tr),eq_mean_pct:mean(rets),
profit_factor:loss>0?gain/loss:null,rug_pct:tr.length?100*rugs/tr.length:null,max_drawdown_pct:100*dd,
last_hour:tr.filter(x=>x.t>t-3600).length,tr,pts,gain,loss,rugs}}
const COLS=[["name","Book",0],["hold_min","Hold",0],["equity","Equity $",0],["return_pct","Return",2],["trades","Trades",0],["open","Open",0],["win_rate","Win",2],["median_pct","Median",1],["mean_pct","Mean",1],["profit_factor","PF",2],["rug_pct","Rugged",0],["max_drawdown_pct","Max DD",1]];
function booksTable(id,list){const rows=[...list].sort((a,b)=>{const x=a[sortKey],y=b[sortKey];return (x==null)-(y==null)||(x<y?-1:x>y?1:0)*sortDir});
const leader=list.filter(b=>b.trades>0).sort((a,b)=>b.return_pct-a.return_pct)[0];const exp=(S.extra&&S.extra.expected)||{};
let h="<tr>"+COLS.map(c=>`<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(sortDir<0?" ↓":" ↑"):""}</th>`).join("")+"</tr>";
for(const b of rows){h+=`<tr class="${leader&&b.name===leader.name?"lead-row":""}" title="${esc(exp[b.name]||"")}">`+COLS.map(([k,,d])=>{const v=b[k];
if(k==="name")return`<td>${bookLink(v)}</td>`;if(k==="hold_min")return`<td>${fmt(v,0)}m</td>`;if(k==="equity")return`<td>${fmt(v,0)}</td>`;
if(["return_pct","median_pct","mean_pct"].includes(k))return`<td class="${cls(v)}">${sgn(v,d)}</td>`;
if(k==="rug_pct"||k==="max_drawdown_pct")return`<td>${fmt(v,d,"%")}</td>`;return`<td>${fmt(v,d)}</td>`}).join("")+"</tr>"}
$(id).innerHTML=h;$(id).querySelectorAll("th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=sortKey===k?-sortDir:-1;sortKey=k;render()});
return leader}
function tiles(id,items){$(id).innerHTML=items.map(([l,v,h,lime,nm])=>`<div class="stat${lime?" lime":""}"><span class="label">${l}</span><span class="v num${nm?" nm":""}">${v}</span><span class="h">${h||""}</span></div>`).join("")}
function renderOverview(){const all=S.books.map(bookStats);const leader=booksTable("books",all);
if(leader){$("s-lead").innerHTML=bookLink(leader.name);$("s-lead-h").textContent=`${sgn(leader.return_pct,2)} on ${leader.trades} trades`}
const pnl=all.reduce((a,b)=>a+b.equity-b.cash,0);$("s-pnl").innerHTML=`<span class="${cls(pnl)}">${sgn(pnl,0,"")}</span>`;
$("s-pnl-h").textContent=`$ across ${all.length} books`;$("s-trades").textContent=fmt(all.reduce((a,b)=>a+b.trades,0),0);
$("s-trades-h").textContent=`${all.reduce((a,b)=>a+b.open,0)} open now`;$("s-rate").textContent=fmt(all.reduce((a,b)=>a+b.last_hour,0),0);renderLaunch()}
function group(rs){const r=rs.map(x=>x.ret*100);return{n:r.length,mean:mean(r),median:median(r),win:r.length?100*r.filter(x=>x>0).length/r.length:null,
rug:r.length?100*rs.filter(x=>x.ret<=-0.999).length/r.length:null,big:r.length?100*rs.filter(x=>x.ret>=0.5).length/r.length:null}}
const doneIn=()=>{const t=now();return[...S.launch.values()].filter(x=>x.decided_ts>=t-win&&x.liquid&&x.ret!=null)};
function gapRow(b,done){const t=now(),ideal=done.filter(x=>x.flags.includes(b.flag)),byTok=new Map(ideal.map(x=>[x.token,x]));
const mine=[...S.pos.values()].filter(p=>p.strategy===b.name&&p.closed_at&&p.opened_at>=t-win&&byTok.has(p.token));
const held=new Set([...S.pos.values()].filter(p=>p.strategy===b.name).map(p=>p.token));
const real=mine.map(p=>100*(p.realized_usd/p.cost_usd-1)),idl=mine.map(p=>100*byTok.get(p.token).ret),missed=ideal.filter(x=>!held.has(x.token)).map(x=>100*x.ret);
if(!mine.length&&!missed.length)return"";const gap=mine.length?mean(real)-mean(idl):null;
return`<tr><td>${bookLink(b.name)}</td><td>${mine.length}</td><td class="${cls(mean(real))}">${sgn(mean(real))}</td><td>${sgn(mean(idl))}</td><td class="${cls(gap)}">${sgn(gap,1," pts")}</td><td>${missed.length}</td><td>${sgn(mean(missed))}</td></tr>`}
const GAPHEAD="<tr><th>Book</th><th>Trades</th><th>Real</th><th>Ideal</th><th>Gap</th><th>Missed</th><th>Missed ideal</th></tr>";
function renderLaunch(){const done=doneIn();const label=win>=86400?"24":String(win/3600);$("lh").textContent=$("lh2").textContent=label;
const flags=[];const byFlag={};for(const b of S.books)if(b.flag){(byFlag[b.flag]=byFlag[b.flag]||[]).push(b.name);if(!flags.includes(b.flag))flags.push(b.flag)}
let h="<tr><th>Group</th><th>Trades</th><th>Mean</th><th>Median</th><th>Win</th><th>Rugged</th><th>+50%</th></tr>";
const line=(name,g)=>`<tr><td>${name}</td><td>${fmt(g.n,0)}</td><td class="${cls(g.mean)}">${sgn(g.mean)}</td><td>${sgn(g.median)}</td><td>${fmt(g.win,0,"%")}</td><td>${fmt(g.rug,0,"%")}</td><td>${fmt(g.big,0,"%")}</td></tr>`;
h+=line("Every liquid pool",group(done));
for(const f of flags)h+=line(byFlag[f].map(bookLink).join(", "),group(done.filter(x=>x.flags.includes(f))));
$("outcomes").innerHTML=h;$("gaps").innerHTML=GAPHEAD+S.books.filter(b=>b.flag).map(b=>gapRow(b,done)).join("")}
/* ledger */
const STAG={"Open":"lime","Exit blocked":"warn","Exit blocked ($0)":"neg","Won":"","Lost":""};
function ledgerTable(id,rows,limit,showBook){let h=`<tr><th>Bought (UTC)</th>${showBook?'<th class="l">Book</th>':""}<th class="l">Token</th><th class="l">Status</th><th>Result</th><th>Result $</th><th>Held</th><th>Cost $</th><th>Exit</th></tr>`;
for(const x of rows.slice(0,limit)){const tok=x.token?`<a class="lk" href="https://dexscreener.com/solana/${encodeURIComponent(x.token)}" target="_blank" rel="noopener" title="${esc(x.token)}">${esc(x.sym)}</a>`:esc(x.sym);
h+=`<tr><td class="mono">${when(x.opened)}</td>${showBook?`<td class="l">${bookLink(x.book)}</td>`:""}<td class="l">${tok}</td><td class="l"><span class="tag sm ${STAG[x.st]}">${esc(x.st)}</span></td>`+
`<td class="${cls(x.pct)}">${sgn(x.pct)}${x.p.swap_ret==null?"":`<br><span class="muted" title="fill prices only, fees and rent left out">swap ${sgn(100*x.p.swap_ret)}</span>`}</td><td class="${cls(x.pnl)}">${x.pnl==null?"-":sgn(x.pnl,2,"")}</td><td>${dur(x.held)}</td><td>${fmt(x.cost,2)}</td><td class="muted">${esc((x.reason||"").replace(/_/g," "))}</td></tr>`}
$(id).innerHTML=h;return rows.length}
const STATUS_OK=(x,st)=>!st||x.st===st;
function fillSelects(){const fs=$("lf-fam"),bs=$("lf-book"),fv=fs.value,bv=bs.value;
fs.innerHTML='<option value="">Every family</option>'+S.fams.map(f=>`<option ${f.name===fv?"selected":""}>${esc(f.name)}</option>`).join("");
bs.innerHTML='<option value="">Every book</option>'+S.books.filter(b=>!fs.value||b.family===fs.value).map(b=>`<option ${b.name===bv?"selected":""}>${esc(b.name)}</option>`).join("")}
function renderLedger(){const f=$("lf-fam").value,b=$("lf-book").value,st=$("lf-status").value,q=$("lf-q").value.trim().toLowerCase();
const rows=trades(x=>(!f||famOf(x.book)===f)&&(!b||x.book===b)&&STATUS_OK(x,st)&&(!q||(x.sym||"").toLowerCase().includes(q)||(x.token||"").toLowerCase().includes(q)));
const n=ledgerTable("ledger",rows,LIM.ledger,true);$("ledger-count").textContent=`${Math.min(n,LIM.ledger)} of ${n} trades`;$("ledger-more").hidden=n<=LIM.ledger}
/* equity chart: one line per series, inline SVG, ink tones and dash patterns (lime is never a line) */
const STY=[["var(--ink)",""],["var(--ink)","6 4"],["var(--ink-2)",""],["var(--ink-2)","6 4"],["var(--ink-3)",""],["var(--ink-3)","2 3"],["var(--ink-2)","2 3"],["var(--ink)","2 3"]];
function chart(id,series,base=S.cash,empty="No closed trades yet."){series=series.filter(s=>s.pts.length>1);if(!series.length){$(id).innerHTML=`<p class="sub">${esc(empty)}</p>`;return}
const W=800,H=260,L=72,Rr=12,T=12,B=28;let x0=Infinity,x1=-Infinity,y0=base,y1=base;
for(const s of series)for(const[x,y]of s.pts){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y)}
if(x1<=x0)x1=x0+1;const pad=(y1-y0)*0.08||50;y0-=pad;y1+=pad;
const X=x=>L+(x-x0)/(x1-x0)*(W-L-Rr),Y=y=>T+(y1-y)/(y1-y0)*(H-T-B);
let g=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Equity curve">`;
g+=`<line x1="${L}" x2="${W-Rr}" y1="${Y(base)}" y2="${Y(base)}" style="stroke:var(--rule-soft);stroke-width:1.5" stroke-dasharray="5 5" vector-effect="non-scaling-stroke"/>`;
g+=`<line x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" style="stroke:var(--rule);stroke-width:2" vector-effect="non-scaling-stroke"/><line x1="${L}" x2="${W-Rr}" y1="${H-B}" y2="${H-B}" style="stroke:var(--rule);stroke-width:2" vector-effect="non-scaling-stroke"/>`;
const ys=[];for(const v of[base,y1-pad,y0+pad]){if(ys.some(y=>Math.abs(y-Y(v))<14))continue;ys.push(Y(v));g+=`<text class="ax" x="${L-6}" y="${Y(v)+4}" text-anchor="end">${fmt(v,0)}</text>`}
g+=`<text class="ax" x="${L}" y="${H-8}">${when(x0).slice(0,11)}</text><text class="ax" x="${W-Rr}" y="${H-8}" text-anchor="end">${when(x1).slice(0,11)}</text>`;
series.forEach((s,i)=>{const[c,d]=STY[i%STY.length];g+=`<polyline fill="none" style="stroke:${c};stroke-width:2" ${d?`stroke-dasharray="${d}"`:""} vector-effect="non-scaling-stroke" points="${s.pts.map(([x,y])=>X(x).toFixed(1)+","+Y(y).toFixed(1)).join(" ")}"><title>${esc(s.name)}</title></polyline>`});
g+="</svg>";if(series.length>1)g+='<div class="legend">'+series.map((s,i)=>{const[c,d]=STY[i%STY.length];return`<span><svg viewBox="0 0 28 8"><line x1="0" x2="28" y1="4" y2="4" style="stroke:${c};stroke-width:2" ${d?`stroke-dasharray="${d}"`:""}/></svg>${esc(s.name)}${s.ret==null?"":" "+sgn(s.ret,1)}</span>`}).join("")+"</div>";
$(id).innerHTML=g}
/* families index, family view, book view */
function famStats(list){const tr=list.flatMap(b=>b.tr),rets=tr.map(x=>x.ret*100),pnl=list.reduce((a,b)=>a+b.equity-b.cash,0);
return{pnl,trades:tr.length,open:list.reduce((a,b)=>a+b.open,0),win:tr.length?100*tr.filter(x=>x.ret>0).length/tr.length:null,mean:wmean(tr),median:median(rets),
rug:tr.length?100*tr.filter(x=>x.ret<=-0.8).length/tr.length:null,best:[...list].filter(b=>b.trades).sort((a,b)=>b.return_pct-a.return_pct)[0]}}
function renderFams(){const all=S.books.map(bookStats);
$("fams").innerHTML=S.fams.map(f=>{const list=all.filter(b=>b.family===f.name);if(!list.length)return"";const s=famStats(list);
return`<section class="card"><p class="label eyebrow">Family · ${list.length} book${list.length>1?"s":""}</p><h2>${famLink(f.name)}</h2><p class="sub">${esc(f.about)}</p>`+
`<p class="sub"><b class="${cls(s.pnl)}">${sgn(s.pnl,0," $")}</b> · ${fmt(s.trades,0)} trades · mean ${sgn(s.mean)} · rugged ${fmt(s.rug,0,"%")}</p>`+
`<div class="scroll"><table class="mini"><tr><th>Book</th><th>Return</th><th>Trades</th><th>Mean</th></tr>`+
list.sort((a,b)=>b.return_pct-a.return_pct).map(b=>`<tr><td>${bookLink(b.name)}</td><td class="${cls(b.return_pct)}">${sgn(b.return_pct,2)}</td><td>${fmt(b.trades,0)}</td><td class="${cls(b.mean_pct)}">${sgn(b.mean_pct)}</td></tr>`).join("")+"</table></div></section>"}).join("")}
function renderFamily(name){const fam=S.fams.find(f=>f.name===name)||{name,about:""};const list=S.books.filter(b=>b.family===name).map(bookStats);
$("f-name").textContent=name;$("f-about").textContent=fam.about;const s=famStats(list);
tiles("f-stats",[["Leading book",s.best?bookLink(s.best.name):"-",s.best?`${sgn(s.best.return_pct,2)} on ${s.best.trades} trades`:"",true,true],
["P&amp;L, family",`<span class="${cls(s.pnl)}">${sgn(s.pnl,0,"")}</span>`,`$ across ${list.length} books`],
["Closed trades",fmt(s.trades,0),`${s.open} open now`],["Mean trade",`<span class="${cls(s.mean)}">${sgn(s.mean)}</span>`,`median ${sgn(s.median)} · win ${fmt(s.win,0,"%")}`],
["Rugged",fmt(s.rug,0,"%"),"closed at −80% or worse"]]);
booksTable("f-books",list);chart("f-chart",list.map(b=>({name:b.name,pts:b.pts,ret:b.return_pct})),list.length?list[0].cash:S.cash);
const names=new Set(list.map(b=>b.name)),rows=trades(x=>names.has(x.book));const n=ledgerTable("f-ledger",rows,LIM.f,true);
$("f-count").textContent=`${Math.min(n,LIM.f)} of ${n} trades`;$("f-more").hidden=n<=LIM.f}
function renderBook(name){const b=S.books.find(x=>x.name===name);if(!b){$("b-name").textContent=name;$("b-about").textContent="This book is not running.";return}
const s=bookStats(b),exp=(S.extra&&S.extra.expected)||{};$("b-name").textContent=name;$("b-fam").textContent=b.family;$("b-fam").href="#/family/"+encodeURIComponent(b.family);
$("b-about").textContent=exp[name]||`Hold ${fmt(b.hold_min,0)} min.`;
tiles("b-stats",[["Equity",fmt(s.equity,0),`return <b>${sgn(s.return_pct,2)}</b> · max drawdown ${fmt(s.max_drawdown_pct,1,"%")}`,true],
["Closed trades",fmt(s.trades,0),`${s.open} open now · ${s.last_hour} in the past hour`],
["Mean trade",`<span class="${cls(s.mean_pct)}">${sgn(s.mean_pct)}</span>`,`per $ traded · equal-weight ${sgn(s.eq_mean_pct)} · median ${sgn(s.median_pct)} · win ${fmt(s.win_rate==null?null:100*s.win_rate,0,"%")}`],
["Profit factor",fmt(s.profit_factor,2),`won ${fmt(s.gain,0)} $ · lost ${fmt(s.loss,0)} $`],["Rugged",fmt(s.rug_pct,0,"%"),`${s.rugs} trade${s.rugs===1?"":"s"} at −80% or worse`]]);
chart("b-chart",[{name,pts:s.pts,ret:s.return_pct}],s.cash);
if(b.flag){$("b-gap-card").hidden=false;$("b-lh").textContent=win>=86400?"24":String(win/3600);$("b-gap").innerHTML=GAPHEAD+(gapRow(b,doneIn())||'<tr><td class="muted">No finished trades in the window.</td></tr>')}else $("b-gap-card").hidden=true;
$("b-swap-card").hidden=!b.twin;if(b.twin)swapCard(b);
const rows=trades(x=>x.book===name);const n=ledgerTable("b-ledger",rows,LIM.b,false);$("b-count").textContent=`${Math.min(n,LIM.b)} of ${n} trades`;$("b-more").hidden=n<=LIM.b}
/* the canary judged on fill prices against its paper twin on the same tokens */
function swapCard(b){const t=now(),tw=new Map();
for(const p of S.pos.values())if(p.strategy===b.twin){const x=trade(p,t);if(x.pct!=null)tw.set(p.token,x)}
const rows=trades(x=>x.book===b.name&&x.p.swap_ret!=null);const sw=[],tv=[],gap=[];
let h=`<tr><th>Bought (UTC)</th><th class="l">Token</th><th>Canary swap</th><th>Twin result</th><th>Gap</th><th>Canary net</th><th>Net $</th></tr>`;
for(const x of rows){const s=100*x.p.swap_ret,w=tw.get(x.token),wp=w?w.pct:null;sw.push(s);if(wp!=null){tv.push(wp);gap.push(s-wp)}
if(sw.length>LIM.b)continue;
h+=`<tr><td class="mono">${when(x.opened)}</td><td class="l">${esc(x.sym)}</td><td class="${cls(s)}">${sgn(s)}</td><td class="${cls(wp)}">${sgn(wp)}</td><td class="${cls(wp==null?null:s-wp)}">${wp==null?"-":sgn(s-wp)}</td><td class="muted">${sgn(x.pct)}</td><td class="muted">${x.pnl==null?"-":sgn(x.pnl,2,"")}</td></tr>`}
const rug=sw.filter(v=>v<=-80).length;
tiles("b-swap-stats",[["Canary swap",`<span class="${cls(mean(sw))}">${sgn(mean(sw))}</span>`,`mean of ${sw.length} sold · median ${sgn(median(sw))}`],
["Twin, same tokens",`<span class="${cls(mean(tv))}">${sgn(mean(tv))}</span>`,`${tv.length} matched`],
["Gap",`<span class="${cls(mean(gap))}">${sgn(mean(gap))}</span>`,`canary swap minus twin · median ${sgn(median(gap))}`],
["Rugged (swap)",fmt(sw.length?100*rug/sw.length:null,0,"%"),`${rug} sold at −80% or worse`]]);
$("b-swap").innerHTML=rows.length?h:'<tr><td class="muted">No sold canary trades yet.</td></tr>'}
/* programs: Supermeme (all books) and MZF Short Only (the shortbook app's feed) */
function memeTotals(){const all=S.books.map(bookStats),tr=all.flatMap(b=>b.tr),pnl=all.reduce((a,b)=>a+b.equity-b.cash,0);
return{all,pnl,cap:all.reduce((a,b)=>a+b.cash,0),trades:tr.length,open:all.reduce((a,b)=>a+b.open,0),mean:wmean(tr),last_hour:all.reduce((a,b)=>a+b.last_hour,0)}}
function shortsState(){if(!S.shortsEnabled)return["off","Not connected: set TB_SHORTBOOK_FEED_URL on the Supermeme app."];
if(!S.shorts||S.shorts.nav_usd==null)return["wait",S.shorts&&S.shorts.error?`Short book unreachable: ${S.shorts.error}`:"Waiting for the short book's first update."];
if(S.shorts.stale)return["stale",`Showing the last update from ${when(S.shorts.fetched_ts||S.shorts.built_ts)} UTC: the short book is unreachable (${S.shorts.error||"no answer"}).`];
if(!S.shorts.healthy)return["stale","The short book's loop looks stalled (its heartbeat is old)."];return["ok",""]}
function shortStats(){const f=S.shorts;if(!f||f.nav_usd==null)return null;const cl=f.closed||[],rets=cl.map(p=>p.pnl_pct).filter(x=>x!=null);
return{nav:f.nav_usd,pnl:f.nav_usd-f.start_usd,ret:f.return_pct,open:(f.open||[]).length,exposure:(f.open||[]).reduce((a,p)=>a+p.size_usd,0),
closed:f.closed_total||0,mean:mean(rets),dd:f.max_drawdown_pct,pending:f.pending}}
function renderPrograms(){const m=memeTotals(),sh=shortStats(),[st,msg]=shortsState();
const comb=m.pnl+(sh?sh.pnl:0);
tiles("p-stats",[["P&amp;L, all programs",`<span class="${cls(comb)}">${sgn(comb,0,"")}</span>`,"$, paper",true],
["Supermeme",`<span class="${cls(m.pnl)}">${sgn(m.pnl,0,"")}</span>`,`$ on ${fmt(m.cap,0)} $ across ${m.all.length} books`],
["MZF Short Only",sh?`<span class="${cls(sh.pnl)}">${sgn(sh.pnl,0,"")}</span>`:"-",sh?`$ on ${fmt(S.shorts.start_usd,0)} $ · NAV ${fmt(sh.nav,0)}`:esc(msg)],
["Trades, last hour",fmt(m.last_hour,0),"Supermeme books (shorts hold 7 days)"]]);
const fams=S.fams.map(f=>{const list=m.all.filter(b=>b.family===f.name);if(!list.length)return"";const s=famStats(list);
return`<tr><td>${famLink(f.name)}</td><td>${list.length}</td><td class="${cls(s.pnl)}">${sgn(s.pnl,0,"")}</td><td>${fmt(s.trades,0)}</td><td class="${cls(s.mean)}">${sgn(s.mean)}</td></tr>`}).join("");
const meme=`<section class="card prog"><p class="label eyebrow">Program · Solana launch books</p><h2><a href="#/supermeme">Supermeme</a></h2>
<p class="sub">Paper books buying hand-made Solana pools seconds to minutes after launch, out within minutes. ${m.all.length} books in ${S.fams.filter(f=>m.all.some(b=>b.family===f.name)).length} families.</p>
<div class="kv"><div><span>P&amp;L $</span><b class="${cls(m.pnl)}">${sgn(m.pnl,0,"")}</b></div><div><span>Return on capital</span><b class="${cls(m.pnl)}">${sgn(m.cap?100*m.pnl/m.cap:null,2)}</b></div>
<div><span>Closed trades</span><b>${fmt(m.trades,0)}</b></div><div><span>Mean trade</span><b class="${cls(m.mean)}">${sgn(m.mean)}</b></div><div><span>Open now</span><b>${fmt(m.open,0)}</b></div></div>
<div class="scroll"><table class="mini"><tr><th>Family</th><th>Books</th><th>P&amp;L $</th><th>Trades</th><th>Mean</th></tr>${fams}</table></div>
<p class="more"><a class="lk" href="#/books">Books &amp; families</a> · <a class="lk" href="#/ledger">Trade ledger</a></p></section>`;
let shorts=`<section class="card prog"><p class="label eyebrow">Program · centralised-exchange shorts</p><h2><a href="#/shorts">MZF Short Only</a></h2>
<p class="sub">Short-only strategies on perps. Strategy 1 shorts new exchange listings at the listing hour and covers after 7 days.</p>`;
if(st!=="ok")shorts+=`<p class="note">${esc(msg)}</p>`;
if(sh){const strat=(S.shorts.strategies||[])[0]||{groups:{}};
shorts+=`<div class="kv"><div><span>NAV $</span><b>${fmt(sh.nav,0)}</b></div><div><span>Return</span><b class="${cls(sh.ret)}">${sgn(sh.ret,2)}</b></div>
<div><span>Open shorts</span><b>${fmt(sh.open,0)}</b></div><div><span>Closed shorts</span><b>${fmt(sh.closed,0)}</b></div><div><span>Max drawdown</span><b>${fmt(sh.dd==null?null:-sh.dd,1,"%")}</b></div></div>
<div class="scroll"><table class="mini"><tr><th>${esc(strat.name||"Strategy")}</th><th>Open</th><th>Closed</th><th>Mean</th><th>Research</th></tr>`+
Object.entries(strat.groups||{}).map(([g,v])=>`<tr><td>${esc(g)}</td><td>${v.open}</td><td>${v.closed}</td><td class="${cls(v.mean_pct)}">${sgn(v.mean_pct)}</td><td>${sgn(v.research_pct)}</td></tr>`).join("")+
`</table></div>`}
shorts+=`<p class="more"><a class="lk" href="#/shorts">Open MZF Short Only</a></p></section>`;
$("progs").innerHTML=meme+shorts}
const days=s=>s==null?"-":(s/86400).toFixed(1)+"d";
function renderShorts(){const[st,msg]=shortsState();$("sh-note").innerHTML=st==="ok"?"":`<p class="note">${esc(msg)}</p>`;const f=S.shorts,sh=shortStats();
if(!sh){["sh-stats","sh-groups","sh-chart","sh-open","sh-closed","sh-listings"].forEach(i=>$(i).innerHTML="");return}
const strat=(f.strategies||[])[0]||{groups:{}};$("sh-strat").textContent=strat.name||"Strategy";$("sh-about").textContent=strat.about||"";
tiles("sh-stats",[["Net asset value",fmt(sh.nav,0),`$ · started ${fmt(f.start_usd,0)} $`,true],["Return",`<span class="${cls(sh.ret)}">${sgn(sh.ret,2)}</span>`,`${sgn(sh.pnl,0," $")}`],
["Open shorts",fmt(sh.open,0),`${fmt(sh.exposure,0)} $ short · ${fmt(sh.pending,0)} listings pending`],["Closed shorts",fmt(sh.closed,0),`mean ${sgn(sh.mean)} per trade`],
["Max drawdown",fmt(sh.dd==null?null:-sh.dd,1,"%"),f.started_ts?`running since ${when(f.started_ts).slice(0,11)}`:""]]);
$("sh-groups").innerHTML="<tr><th>Group</th><th>Open</th><th>Closed</th><th>Mean</th><th>Research</th><th>Win</th><th>P&amp;L $</th></tr>"+
Object.entries(strat.groups||{}).map(([g,v])=>`<tr><td class="book">${esc(g)}</td><td>${v.open}</td><td>${v.closed}</td><td class="${cls(v.mean_pct)}">${sgn(v.mean_pct)}</td><td>${sgn(v.research_pct)}</td><td>${fmt(v.win_pct,0,"%")}</td><td class="${cls(v.pnl_usd)}">${sgn(v.pnl_usd,0,"")}</td></tr>`).join("");
chart("sh-chart",[{name:"NAV",pts:f.equity||[],ret:sh.ret}],f.start_usd);
const head="<tr><th>Opened (UTC)</th><th class=\"l\">Token</th><th class=\"l\">Group</th><th class=\"l\">Venue</th><th>Size $</th><th>Entry</th><th>Price</th><th>Move</th><th>Funding $</th><th>P&amp;L $</th><th>P&amp;L</th><th>",
row=(p,last)=>`<tr><td class="mono">${when(p.entry_ts)}</td><td class="l book">${esc(p.base)}</td><td class="l">${esc(p.group)}</td><td class="l">${esc(p.venue)}</td><td>${fmt(p.size_usd,0)}</td><td>${fmt(p.entry_px,4)}</td><td>${fmt(p.px,4)}</td><td class="${cls(p.price_pct)}">${sgn(p.price_pct)}</td><td class="${cls(p.funding_usd)}">${sgn(p.funding_usd,0,"")}</td><td class="${cls(p.pnl_usd)}">${sgn(p.pnl_usd,0,"")}</td><td class="${cls(p.pnl_pct)}">${sgn(p.pnl_pct)}</td><td>${last}</td></tr>`;
const t=now();$("sh-open").innerHTML=(f.open||[]).length?head+"Covers in</th></tr>"+f.open.map(p=>row(p,days(p.due_ts-t))).join(""):'<tr><td class="muted">No open shorts.</td></tr>';
$("sh-closed").innerHTML=(f.closed||[]).length?head+"Covered (UTC)</th></tr>"+f.closed.map(p=>row(p,`<span class="mono">${when(p.exit_ts)}</span>`)).join(""):'<tr><td class="muted">No closed shorts yet.</td></tr>';
$("sh-lcount").textContent=Object.entries(f.listing_counts||{}).map(([k,n])=>`${k} ${n}`).join(" · ");
$("sh-listings").innerHTML="<tr><th>Seen (UTC)</th><th class=\"l\">Venue</th><th class=\"l\">Token</th><th class=\"l\">Kind</th><th class=\"l\">Status</th><th class=\"l\">Reason</th></tr>"+
(f.listings||[]).map(r=>`<tr><td class="mono">${r.seen_ts?when(r.seen_ts):"-"}</td><td class="l">${esc(r.venue)}</td><td class="l book">${esc(r.base)}</td><td class="l">${esc(r.kind||"")}</td><td class="l"><span class="tag sm ${r.status==="entered"?"lime":r.status==="pending"?"warn":""}">${esc(r.status)}</span></td><td class="l muted">${esc(r.reason||"")}</td></tr>`).join("")}
/* meme market monitor: hourly rows + last 3 h against the past week, built by market_monitor.py */
const MK_RET=new Set(["market_mean","market_median","market_sized","rw_kept","rw_skipped","rw_gap","sz_picks_ideal","sz_real","sz_fill_gap","thin_mean","mid_mean","thick_mean","thin_sz_mean","sol_chg_pct","btc_chg_pct"]);
const MK_USD=new Set(["med_liq_usd","sol_usd","btc_usd"]);
const mkLabel=k=>((S.market&&S.market.metrics)||[]).find(m=>m[0]===k)?.[1]||k;
function mkVal(k,v){if(v==null)return"-";if(MK_RET.has(k))return`<span class="${cls(v)}">${sgn(v,1,k.endsWith("gap")?" pts":"%")}</span>`;
if(MK_USD.has(k))return fmt(v,0);if(k.endsWith("_pct"))return fmt(v,1,"%");return Math.abs(v)>=100?fmt(v,0):fmt(v,1)}
const zTag=z=>z==null?'<span class="muted">-</span>':`<span class="tag sm ${Math.abs(z)>=2?"warn":""}">${z>0?"+":"−"}${Math.abs(z).toFixed(1)}</span>`;
function marketState(){if(!S.marketEnabled)return["off","The market monitor is off: set TB_MARKET_MONITOR=1 on the Supermeme app."];
const m=S.market;if(!m||!m.rows.length)return["wait","Waiting for the monitor's first hours (it stores each hour about 15 minutes after it ends; the first run backfills)."];
if(now()-m.built_ts>600)return["stale",`The dashboard has not read the monitor's table since ${when(m.built_ts)} UTC (its read keeps failing; check the logs for "market monitor read failed"). The monitor itself may be fine.`];
if(now()-m.last_hour>3*3600+900){const s=m.status||{},hr=`The newest stored hour is ${when(m.last_hour).slice(0,11)}:00 UTC.`;
if(!s.ran)return["stale",`${hr} The monitor has no recorded run: its thread is not running (needs TB_MARKET_MONITOR=1 and the launch scorer, then a restart).`];
if(now()-s.ran>3*(m.poll_s||600))return["stale",`${hr} The monitor last ran at ${when(s.ran)} UTC and has not run since: its thread has stopped or is stuck. A restart brings it back.`];
if(s.error)return["stale",`${hr} The monitor runs but fails: ${s.error} (last success ${s.ok_ts?when(s.ok_ts)+" UTC":"never"}).`];
return["stale",`${hr} The monitor ran at ${when(s.ran)} UTC without error but stored nothing new: the launch scorer may have no new pools.`]}
return["ok",""]}
function renderMarket(){const[st,msg]=marketState();$("mk-note").innerHTML=st==="ok"?"":`<p class="note">${esc(msg)}</p>`;const m=S.market;
if(!m||!m.rows.length){["mk-stats","mk-shifts","mk-charts","mk-cmp","mk-hours","mk-flags","mk-flagsum"].forEach(i=>$(i).innerHTML="");return}
const c=m.compare||{},rh=m.recent_h,wk=Math.round(m.baseline_h/24);
const tile=(k,l)=>{const x=c[k]||{};return[l,mkVal(k,x.now),`week ${mkVal(k,x.baseline)} · z ${x.z==null?"-":(x.z>0?"+":"−")+Math.abs(x.z).toFixed(1)}`]};
const shifts=Object.entries(c).filter(([,x])=>x.z!=null&&Math.abs(x.z)>=2).sort((a,b)=>Math.abs(b[1].z)-Math.abs(a[1].z));
tiles("mk-stats",[["Shifts now",fmt(shifts.length,0),`measures 2+ standard deviations from the past ${wk} days`],
tile("liquid","Liquid pools per hour"),tile("thin_pct","Thin pools (under $15k)"),tile("rug_pct","Rugged by minute 6"),
tile("market_mean","Market, ideal mean"),tile("rw_gap","Rug-vs-win gap"),tile("sz_real","sz real result")]);
$("mk-shift-sub").textContent=`The last ${rh} hours against the ${wk} days before${m.baseline_since?` (from ${when(m.baseline_since).slice(0,11)} UTC at the earliest)`:""}; sz measures count only from its first trade. z = how many standard deviations of ${rh}-hour windows the last ${rh} hours sit from the weekly mean.`;
$("mk-shifts").innerHTML=shifts.length?"<tr><th class=\"l\">Measure</th><th class=\"l\">Group</th><th>Last "+rh+" h</th><th>Past week</th><th>z</th></tr>"+
shifts.map(([k,x])=>`<tr><td class="l book">${esc(mkLabel(k))}</td><td class="l muted">${esc(x.group)}</td><td>${mkVal(k,x.now)}</td><td>${mkVal(k,x.baseline)}</td><td>${zTag(x.z)}</td></tr>`).join("")
:'<tr><td class="muted">Nothing is 2+ standard deviations from the past week.</td></tr>';
const F=m.flags||{open:[],recent:[],by_metric:[]},fr=m.flag_rule||{},hr=h=>when(h).slice(0,8)+"h";
$("mk-flag-sub").textContent=`A flag opens when a measure stays ${fr.z||2}+ standard deviations from the past week for ${fr.hours||2}+ hours in a row, and closes below ${fr.close_z||1.5}. Open flags first, then those closed in the past 7 days.`;
const fl=(f,open)=>`<tr><td class="l"><span class="tag sm ${open?"warn":""}">${open?"Open":"Closed"}</span></td><td class="l book">${esc(f.label)}</td><td class="l">${f.direction==="up"?"↑ up":"↓ down"}</td><td class="mono">${hr(f.opened_hour)}</td><td class="mono">${open?"-":hr(f.closed_hour)}</td><td>${f.hours}</td><td>${zTag(f.peak_z)}</td><td>${mkVal(f.metric,f.open_value)} vs ${mkVal(f.metric,f.baseline)}</td><td class="${cls(f.sz_pct)}">${f.sz_pct==null?"-":sgn(f.sz_pct)}</td><td>${f.sz_trades}</td><td class="${cls(f.market_pct)}">${f.market_pct==null?"-":sgn(f.market_pct)}</td></tr>`;
$("mk-flags").innerHTML=(F.open.length||F.recent.length)?'<tr><th class="l">State</th><th class="l">Measure</th><th class="l">Direction</th><th>From</th><th>To</th><th>Hours</th><th>Peak z</th><th>Value vs week</th><th>sz</th><th>sz trades</th><th>Market</th></tr>'+F.open.map(f=>fl(f,true)).join("")+F.recent.map(f=>fl(f,false)).join(""):'<tr><td class="muted">No flags yet.</td></tr>';
$("mk-flagsum").innerHTML=F.by_metric.length?'<tr><th class="l">Measure</th><th class="l">Direction</th><th>Flags</th><th>Hours</th><th>sz in</th><th>sz other</th><th>Gap</th><th>Market in</th><th>Market other</th></tr>'+F.by_metric.map(g=>`<tr><td class="l book">${esc(g.label)}</td><td class="l">${g.direction==="up"?"↑ up":"↓ down"}</td><td>${g.flags}</td><td>${g.hours}</td><td class="${cls(g.in.sz_pct)}">${g.in.sz_pct==null?"-":sgn(g.in.sz_pct)}</td><td class="${cls(g.out.sz_pct)}">${g.out.sz_pct==null?"-":sgn(g.out.sz_pct)}</td><td class="${cls(g.sz_gap)}">${g.sz_gap==null?"-":sgn(g.sz_gap,1," pts")}</td><td class="${cls(g.in.market_pct)}">${g.in.market_pct==null?"-":sgn(g.in.market_pct)}</td><td class="${cls(g.out.market_pct)}">${g.out.market_pct==null?"-":sgn(g.out.market_pct)}</td></tr>`).join(""):'<tr><td class="muted">No flags yet.</td></tr>';
const rows=m.rows,pts=k=>rows.filter(r=>r[k]!=null).map(r=>[r.hour+1800,r[k]]);
const CH=[["Liquid pools per hour",[["liquid","pools"]],"liquid"],["Thin pools (under $15k), % of liquid",[["thin_pct","under $15k"]],"thin_pct"],
["Rugged by minute 6, %",[["rug_pct","rugged"]],"rug_pct"],["Median LP pull, seconds after creation",[["med_pull_s","pull second"]],"med_pull_s"],
["Ideal mean: every pool vs the sized market, %",[["market_mean","every pool"],["market_sized","sized like sz"]],"market_mean"],["sz real vs its picks at ideal, %",[["sz_real","sz real"],["sz_picks_ideal","sz picks, ideal"]],"sz_real"]];
$("mk-charts").innerHTML=CH.map((ch,i)=>`<section class="card chart"><p class="label eyebrow">Last ${Math.round((m.rows.length))} h</p><h2>${esc(ch[0])}</h2><p class="sub">Dashed: the past-week mean.</p><div id="mk-ch${i}"></div></section>`).join("");
CH.forEach((ch,i)=>{const base=(c[ch[2]]||{}).baseline;chart("mk-ch"+i,ch[1].map(([k,n])=>({name:n,pts:pts(k),ret:null})),base==null?0:base,"Not enough hours yet.")});
$("mk-cmp-sub").textContent=`Last ${rh} hours against the ${wk} days before; |z| of 2 or more is marked. Pooled measures (shares, means) weight each hour by its pools.`;
let h=`<tr><th class="l">Measure</th><th>Last ${rh} h</th><th>Past week</th><th>z</th></tr>`,g=null;
for(const[k,l,grp]of m.metrics){const x=c[k];if(!x||x.now==null)continue;if(grp!==g){g=grp;h+=`<tr><td class="l label muted" colspan="4">${esc(grp)}</td></tr>`}
h+=`<tr><td class="l">${esc(l)}</td><td>${mkVal(k,x.now)}</td><td>${mkVal(k,x.baseline)}</td><td>${zTag(x.z)}</td></tr>`}
$("mk-cmp").innerHTML=h;
const HC=[["pools","Pools"],["liquid","Liquid"],["thin_pct","<15k"],["mid_pct","15-25k"],["thick_pct","25k+"],["creators","Creators"],["top5_pct","Top 5"],["template_pct","Template"],["rug_pct","Rugged"],["med_pull_s","Pull s"],["big_win_pct","+50%"],["market_mean","Market"],["market_sized","Mkt sized"],["thin_mean","<15k mkt"],["rw_gap","RvW gap"],["sz_trades","sz n"],["sz_real","sz"],["btc_chg_pct","BTC"]];
$("mk-hours").innerHTML="<tr><th>Hour</th>"+HC.map(([,l])=>`<th>${l}</th>`).join("")+"</tr>"+
[...rows].reverse().slice(0,48).map(r=>`<tr><td class="mono">${when(r.hour).slice(0,8)}h</td>`+HC.map(([k])=>`<td>${mkVal(k,r[k])}</td>`).join("")+"</tr>").join("")}
/* router */
function route(){const h=decodeURIComponent(location.hash.replace(/^#\/?/,"")),[v,...rest]=h.split("/");const arg=rest.join("/");
R.v=({supermeme:"overview",ledger:"ledger",books:"books",book:"book",family:"family",shorts:"shorts",market:"market"})[v]||"programs";R.arg=arg||null;LIM.b=LIM.f=200;
for(const m of document.querySelectorAll("main"))m.hidden=m.id!=="v-"+R.v;
const meme=["overview","ledger","books","book","family","market"].includes(R.v);$("subtabs").hidden=!meme;
document.querySelectorAll(".tabs:not(.sub) a").forEach(a=>a.classList.toggle("on",a.dataset.v===(meme?"supermeme":R.v)));
document.querySelectorAll(".tabs.sub a").forEach(a=>a.classList.toggle("on",a.dataset.v===R.v||(a.dataset.v==="books"&&(R.v==="book"||R.v==="family"))));
alignSub();window.scrollTo(0,0);render()}
/* the Supermeme sub menu starts under the Supermeme tab (its left rule on the tab's left rule); on narrow screens it keeps the page gutter */
function alignSub(){const s=$("subtabs"),t=$("tab-meme");if(s.hidden)return;s.style.paddingLeft="";
const x=t.getBoundingClientRect().left-s.getBoundingClientRect().left-2,w=[...s.children].reduce((a,c)=>a+c.offsetWidth,0);
const cs=getComputedStyle(s);if(x>parseFloat(cs.paddingLeft)&&x+w<=s.clientWidth-parseFloat(cs.paddingRight))s.style.paddingLeft=x+"px"}
function renderHealth(){const ok=S.health?Object.values(S.health).every(x=>x.ok):null;
$("health").innerHTML=`<span class="dot"></span>${ok==null?"Bot":ok?"Bot healthy":"Bot stalled"}`;$("health").className="tag "+(ok==null?"":ok?"ink":"neg")}
function render(){renderHealth();if(!S.books.length)return;
if(R.v==="programs")renderPrograms();else if(R.v==="shorts")renderShorts();else if(R.v==="market")renderMarket();else if(R.v==="ledger")renderLedger();else if(R.v==="books")renderFams();else if(R.v==="book")renderBook(R.arg);else if(R.v==="family")renderFamily(R.arg);else renderOverview();
$("built").textContent="Updated "+new Date().toLocaleTimeString([],{hour12:false})}
function schedule(){if(!pending){pending=true;setTimeout(()=>{pending=false;render()},300)}}
function fill(f,isNew){const li=document.createElement("li");if(isNew)li.className="new";
const t=new Date(f.ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});
const side=f.side==="buy"?'<span class="side buy">BUY</span>':'<span class="side">SELL</span>';
const right=f.side==="sell"?`<span class="r ${cls(f.ret_pct)}">${f.ret_pct==null?"part":sgn(f.ret_pct)}</span>`:`<span class="r">$${fmt(f.usd,0)}</span>`;
const why=f.side==="sell"&&f.reason?` · ${esc(f.reason.replace(/_/g," "))}`:"";
li.innerHTML=`<span class="t mono">${t}</span><span>${side}<b>${esc(f.symbol)}</b><span class="bk"><a class="lk" href="#/book/${encodeURIComponent(f.book)}">${esc(f.book)}</a>${why}</span></span>${right}`;
const feed=$("feed");feed.prepend(li);while(feed.children.length>300)feed.lastChild.remove()}
document.querySelectorAll("#win button").forEach(bt=>bt.onclick=()=>{win=+bt.dataset.w;document.querySelectorAll("#win button").forEach(x=>x.classList.toggle("on",x===bt));render()});
$("lf-fam").onchange=()=>{fillSelects();LIM.ledger=200;renderLedger()};
for(const id of["lf-book","lf-status"])$(id).onchange=()=>{LIM.ledger=200;renderLedger()};
$("lf-q").oninput=()=>{LIM.ledger=200;renderLedger()};
$("ledger-more").onclick=()=>{LIM.ledger+=200;renderLedger()};$("b-more").onclick=()=>{LIM.b+=200;render()};$("f-more").onclick=()=>{LIM.f+=200;render()};
window.addEventListener("hashchange",route);window.addEventListener("resize",alignSub);if(document.fonts)document.fonts.ready.then(alignSub);
setInterval(render,30000);
function connect(){const es=new EventSource("api/stream");
es.onopen=()=>{$("conn").innerHTML='<span class="dot"></span>Live';$("conn").className="tag lime"};
es.onerror=()=>{$("conn").innerHTML='<span class="dot"></span>Reconnecting';$("conn").className="tag neg"};
es.addEventListener("init",e=>{const d=JSON.parse(e.data);S.books=d.books;S.fams=d.families||[];S.cash=d.cash;S.stuck=d.stuck_after_s;S.pc=d.pos_cols;S.lc=d.launch_cols;
S.shortsEnabled=!!d.shorts_enabled;S.marketEnabled=!!d.market_enabled;S.pos.clear();S.launch.clear();S.health=d.health;merge(d);fillSelects();render()});
es.addEventListener("shorts",e=>{S.shorts=JSON.parse(e.data);schedule()});
es.addEventListener("market",e=>{S.market=JSON.parse(e.data);schedule()});
es.addEventListener("delta",e=>{merge(JSON.parse(e.data));schedule()});
es.addEventListener("summary",e=>{S.extra=JSON.parse(e.data);const d=S.extra.downtime;$("foot").textContent=d?`${d.outages} outages, ${fmt(d.hours_offline,1)} h offline since start`:"";schedule()});
es.addEventListener("backlog",e=>{$("feed").innerHTML="";JSON.parse(e.data).forEach(f=>fill(f,false))});
es.addEventListener("fill",e=>fill(JSON.parse(e.data),true))}
route();connect();
</script></body></html>"""
