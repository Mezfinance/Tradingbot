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
        self._external = False
        threading.Thread(target=loop, name="dashboard", daemon=True).start()
        threading.Thread(target=live_loop, name="dashboard-live", daemon=True).start()
        return self

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
            if sent:
                quiet = time.time()
            elif time.time() - quiet > KEEPALIVE_S:
                write(b": keep-alive\n\n")
                quiet = time.time()
            stop.wait(POLL_S)


POS_COLS = ("id", "strategy", "symbol", "token", "cost_usd", "realized_usd", "qty", "entry_price", "opened_at",
            "closed_at", "exit_reason")
LAUNCH_COLS = ("pool", "token_mint", "create_ts", "decided_ts", "liquid", "ret", "status", "pulled_ts", "liq_usd",
               "top2", "top5", "top2h", "top5h", "low_h", "risk_pct", "rw_pct", "rw_low_pct")
LAUNCH_KEEP_S = 86400  # the scorer's pools sent to the page (it windows them 1/6/24 h)
STUCK_AFTER_S = 1800  # as engine.STUCK_AFTER_S: a position with no route out this long counts as $0


class Live:
    """Raw rows for the page, polled every POLL_S on read-only connections of their own (never the journal's
    lock, and sqlite releases the GIL while it reads). Each changed row gets a new version number, so every
    open page is sent only what changed since its last version."""

    def __init__(self, cfg: Config, journal: Journal | None = None, store=None):
        from tradingbot import model_book as mb
        self.cfg = cfg
        self.books = list(mb.active_books(cfg))
        self.flag_of = {b: mb.LAUNCH_CUT.get(b) for b in self.books}
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
            self._bump(self.pos, r["id"], row)

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
                       launch_live.rw_flags(r["low_h"], r["rw_pct"], r["rw_low_pct"])):
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
                "books": [{"name": b, "hold_min": self.holds[b], "flag": self.flag_of[b]} for b in self.books],
                "pos_cols": [*POS_COLS, "blocked_since"], "positions": [row for _, row in self.pos.values()],
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
<title>MZF Strategy Desk</title>
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
</style></head><body>
<header>
<div class="lockup"><svg viewBox="0 0 100 100" aria-hidden="true"><rect class="mk-sq" width="100" height="100"/><polygon class="mk-z" points="24,26 76,26 76,39 38,61 76,61 76,74 24,74 24,61 62,39 24,39"/></svg>
<div><div class="word">MZF</div><div class="desc">STRATEGY DESK</div></div></div>
<div class="status"><span id="conn" class="tag"><span class="dot"></span>Connecting</span><span id="health" class="tag">Bot</span>
<span id="built" class="label muted"></span></div>
</header>
<main>
<div><p class="label eyebrow">Paper books · simulated fills on live Jupiter quotes</p><h1>Launch strategies</h1>
<p class="lead">Every book starts at $10,000. Trades and every table update live, within two seconds of a trade.</p></div>
<div class="stats">
<div class="stat lime"><span class="label">Leading book</span><span class="v" id="s-lead">-</span><span class="h" id="s-lead-h"></span></div>
<div class="stat"><span class="label">P&amp;L, all books</span><span class="v num" id="s-pnl">-</span><span class="h" id="s-pnl-h"></span></div>
<div class="stat"><span class="label">Closed trades</span><span class="v num" id="s-trades">-</span><span class="h" id="s-trades-h"></span></div>
<div class="stat"><span class="label">Trades, last hour</span><span class="v num" id="s-rate">0</span><span class="h">Closed in the past 60 minutes</span></div>
</div>
<div class="cols">
<section class="card"><p class="label eyebrow">Books</p><h2>Results by book</h2>
<p class="sub">Sorted by return. Click a header to sort. Open trades count at cost until sold; a trade with no route out for 30 min counts as $0. Rugged = closed at −80% or worse.</p>
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
<footer>MZF Protocol · paper trading only, no funds at risk · <span id="foot"></span></footer>
<script>
const $=id=>document.getElementById(id);
const fmt=(v,d=1,s="")=>v==null||Number.isNaN(v)?"-":(typeof v==="number"?v.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}):v)+s;
const sgn=(v,d=1,s="%")=>v==null||Number.isNaN(v)?"-":(v>0?"+":v<0?"−":"")+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d})+s;
const cls=v=>v==null?"":v>0?"pos":v<0?"neg":"";
const esc=t=>String(t??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
const median=a=>{if(!a.length)return null;const b=[...a].sort((x,y)=>x-y),m=b.length>>1;return b.length%2?b[m]:(b[m-1]+b[m])/2};
const S={pos:new Map(),launch:new Map(),books:[],cash:10000,stuck:1800,pc:[],lc:[],skew:0,health:null,extra:null};
let sortKey="return_pct",sortDir=-1,win=21600,pending=false;
const now=()=>Date.now()/1000-S.skew;
const obj=(cols,row)=>{const o={};cols.forEach((c,i)=>o[c]=row[i]);return o};
function merge(d){if(d.now)S.skew=Date.now()/1000-d.now;
(d.positions||[]).forEach(r=>{const o=obj(S.pc,r);S.pos.set(o.id,o)});
(d.launch||[]).forEach(r=>{const o=obj(S.lc,r);S.launch.set(o.pool,o)});
if("health" in d&&d.health)S.health=d.health}
function bookStats(b){const t=now();let equity=S.cash,open=0,started=null;const tr=[];
for(const p of S.pos.values()){if(p.strategy!==b.name)continue;started=started==null?p.opened_at:Math.min(started,p.opened_at);
const cost=p.cost_usd||0,real=p.realized_usd||0;let pnl;
if(p.closed_at){pnl=real-cost;tr.push({t:p.closed_at,ret:cost?real/cost-1:0,pnl})}
else if(p.blocked_since&&t-p.blocked_since>=S.stuck){pnl=real-cost;tr.push({t:p.blocked_since+S.stuck,ret:cost?real/cost-1:-1,pnl})}
else{open++;pnl=real+(p.qty||0)*(p.entry_price||0)-cost}
equity+=pnl}
tr.sort((x,y)=>x.t-y.t);let eq=S.cash,peak=S.cash,dd=0;for(const x of tr){eq+=x.pnl;peak=Math.max(peak,eq);dd=Math.max(dd,(peak-eq)/peak)}
const rets=tr.map(x=>x.ret*100),gain=tr.filter(x=>x.pnl>0).reduce((a,x)=>a+x.pnl,0),loss=-tr.filter(x=>x.pnl<0).reduce((a,x)=>a+x.pnl,0);
const rugs=tr.filter(x=>x.ret<=-0.8).length;
return{name:b.name,hold_min:b.hold_min,started,equity,return_pct:(equity/S.cash-1)*100,trades:tr.length,open,
win_rate:tr.length?tr.filter(x=>x.ret>0).length/tr.length:null,median_pct:median(rets),mean_pct:mean(rets),
profit_factor:loss>0?gain/loss:(gain>0?null:null),rug_pct:tr.length?100*rugs/tr.length:null,max_drawdown_pct:100*dd,last_hour:tr.filter(x=>x.t>t-3600).length}}
const COLS=[["name","Book",0],["hold_min","Hold",0],["equity","Equity $",0],["return_pct","Return",2],["trades","Trades",0],["open","Open",0],["win_rate","Win",2],["median_pct","Median",1],["mean_pct","Mean",1],["profit_factor","PF",2],["rug_pct","Rugged",0],["max_drawdown_pct","Max DD",1]];
function renderBooks(){const all=S.books.map(bookStats);
const rows=[...all].sort((a,b)=>{const x=a[sortKey],y=b[sortKey];return (x==null)-(y==null)||(x<y?-1:x>y?1:0)*sortDir});
const leader=all.filter(b=>b.trades>0).sort((a,b)=>b.return_pct-a.return_pct)[0];const exp=(S.extra&&S.extra.expected)||{};
let h="<tr>"+COLS.map(c=>`<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(sortDir<0?" ↓":" ↑"):""}</th>`).join("")+"</tr>";
for(const b of rows){h+=`<tr class="${leader&&b.name===leader.name?"lead-row":""}" title="${esc(exp[b.name]||"")}">`+COLS.map(([k,,d])=>{const v=b[k];
if(k==="name")return`<td class="book">${esc(v)}</td>`;if(k==="hold_min")return`<td>${fmt(v,0)}m</td>`;if(k==="equity")return`<td>${fmt(v,0)}</td>`;
if(["return_pct","median_pct","mean_pct"].includes(k))return`<td class="${cls(v)}">${sgn(v,d)}</td>`;
if(k==="rug_pct"||k==="max_drawdown_pct")return`<td>${fmt(v,d,"%")}</td>`;return`<td>${fmt(v,d)}</td>`}).join("")+"</tr>"}
$("books").innerHTML=h;$("books").querySelectorAll("th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=sortKey===k?-sortDir:-1;sortKey=k;renderBooks()});
if(leader){$("s-lead").textContent=leader.name;$("s-lead-h").textContent=`${sgn(leader.return_pct,2)} on ${leader.trades} trades`}
const pnl=all.reduce((a,b)=>a+b.equity-S.cash,0);$("s-pnl").innerHTML=`<span class="${cls(pnl)}">${sgn(pnl,0,"")}</span>`;
$("s-pnl-h").textContent=`$ across ${all.length} books`;$("s-trades").textContent=fmt(all.reduce((a,b)=>a+b.trades,0),0);
$("s-trades-h").textContent=`${all.reduce((a,b)=>a+b.open,0)} open now`;$("s-rate").textContent=fmt(all.reduce((a,b)=>a+b.last_hour,0),0)}
function group(rs){const r=rs.map(x=>x.ret*100);return{n:r.length,mean:mean(r),median:median(r),win:r.length?100*r.filter(x=>x>0).length/r.length:null,
rug:r.length?100*rs.filter(x=>x.ret<=-0.999).length/r.length:null,big:r.length?100*rs.filter(x=>x.ret>=0.5).length/r.length:null}}
function renderLaunch(){const t=now(),done=[...S.launch.values()].filter(x=>x.decided_ts>=t-win&&x.liquid&&x.ret!=null);
const label=win>=86400?"24":String(win/3600);$("lh").textContent=$("lh2").textContent=label;
const flags=[];const byFlag={};for(const b of S.books)if(b.flag){(byFlag[b.flag]=byFlag[b.flag]||[]).push(b.name);if(!flags.includes(b.flag))flags.push(b.flag)}
let h="<tr><th>Group</th><th>Trades</th><th>Mean</th><th>Median</th><th>Win</th><th>Rugged</th><th>+50%</th></tr>";
const line=(name,g)=>`<tr><td>${name}</td><td>${fmt(g.n,0)}</td><td class="${cls(g.mean)}">${sgn(g.mean)}</td><td>${sgn(g.median)}</td><td>${fmt(g.win,0,"%")}</td><td>${fmt(g.rug,0,"%")}</td><td>${fmt(g.big,0,"%")}</td></tr>`;
h+=line("Every liquid pool",group(done));
for(const f of flags)h+=line(`<span class="book">${esc(byFlag[f].join(", "))}</span>`,group(done.filter(x=>x.flags.includes(f))));
$("outcomes").innerHTML=h;
h="<tr><th>Book</th><th>Trades</th><th>Real</th><th>Ideal</th><th>Gap</th><th>Missed</th><th>Missed ideal</th></tr>";
for(const b of S.books){if(!b.flag)continue;const ideal=done.filter(x=>x.flags.includes(b.flag));const byTok=new Map(ideal.map(x=>[x.token,x]));
const mine=[...S.pos.values()].filter(p=>p.strategy===b.name&&p.closed_at&&p.opened_at>=t-win&&byTok.has(p.token));
const held=new Set([...S.pos.values()].filter(p=>p.strategy===b.name).map(p=>p.token));
const real=mine.map(p=>100*(p.realized_usd/p.cost_usd-1)),idl=mine.map(p=>100*byTok.get(p.token).ret),missed=ideal.filter(x=>!held.has(x.token)).map(x=>100*x.ret);
if(!mine.length&&!missed.length)continue;const gap=mine.length?mean(real)-mean(idl):null;
h+=`<tr><td class="book">${esc(b.name)}</td><td>${mine.length}</td><td class="${cls(mean(real))}">${sgn(mean(real))}</td><td>${sgn(mean(idl))}</td><td class="${cls(gap)}">${sgn(gap,1," pts")}</td><td>${missed.length}</td><td>${sgn(mean(missed))}</td></tr>`}
$("gaps").innerHTML=h}
function renderHealth(){const ok=S.health?Object.values(S.health).every(x=>x.ok):null;
$("health").innerHTML=`<span class="dot"></span>${ok==null?"Bot":ok?"Bot healthy":"Bot stalled"}`;$("health").className="tag "+(ok==null?"":ok?"ink":"neg")}
function render(){renderBooks();renderLaunch();renderHealth();$("built").textContent="Updated "+new Date().toLocaleTimeString([],{hour12:false})}
function schedule(){if(!pending){pending=true;setTimeout(()=>{pending=false;render()},300)}}
function fill(f,isNew){const li=document.createElement("li");if(isNew)li.className="new";
const t=new Date(f.ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});
const side=f.side==="buy"?'<span class="side buy">BUY</span>':'<span class="side">SELL</span>';
const right=f.side==="sell"?`<span class="r ${cls(f.ret_pct)}">${f.ret_pct==null?"part":sgn(f.ret_pct)}</span>`:`<span class="r">$${fmt(f.usd,0)}</span>`;
const why=f.side==="sell"&&f.reason?` · ${esc(f.reason.replace(/_/g," "))}`:"";
li.innerHTML=`<span class="t mono">${t}</span><span>${side}<b>${esc(f.symbol)}</b><span class="bk">${esc(f.book)}${why}</span></span>${right}`;
const feed=$("feed");feed.prepend(li);while(feed.children.length>300)feed.lastChild.remove()}
document.querySelectorAll("#win button").forEach(bt=>bt.onclick=()=>{win=+bt.dataset.w;document.querySelectorAll("#win button").forEach(x=>x.classList.toggle("on",x===bt));renderLaunch()});
setInterval(render,30000);
function connect(){const es=new EventSource("api/stream");
es.onopen=()=>{$("conn").innerHTML='<span class="dot"></span>Live';$("conn").className="tag lime"};
es.onerror=()=>{$("conn").innerHTML='<span class="dot"></span>Reconnecting';$("conn").className="tag neg"};
es.addEventListener("init",e=>{const d=JSON.parse(e.data);S.books=d.books;S.cash=d.cash;S.stuck=d.stuck_after_s;S.pc=d.pos_cols;S.lc=d.launch_cols;
S.pos.clear();S.launch.clear();S.health=d.health;merge(d);render()});
es.addEventListener("delta",e=>{merge(JSON.parse(e.data));schedule()});
es.addEventListener("summary",e=>{S.extra=JSON.parse(e.data);const d=S.extra.downtime;$("foot").textContent=d?`${d.outages} outages, ${fmt(d.hours_offline,1)} h offline since start`:"";schedule()});
es.addEventListener("backlog",e=>{$("feed").innerHTML="";JSON.parse(e.data).forEach(f=>fill(f,false))});
es.addEventListener("fill",e=>fill(JSON.parse(e.data),true))}
connect();
</script></body></html>"""
