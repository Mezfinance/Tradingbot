"""Live dashboard (11-04): the paper books' trades streamed to a browser as they happen, plus the report tables.

Served by the health server (`health.start_health_server`) on `health_port`, behind a password: the Fly secret
`DASHBOARD_TOKEN` (no secret = no dashboard; /health stays open for Fly's check). Open
`https://<app>.fly.dev/?key=<token>` once: it sets a cookie, so the plain address works afterwards.

  /             the page (no build step, one HTML file)
  /api/stream   server-sent events: `fill` for every buy/sell (newest first on the page), `summary` when the
                tables are rebuilt (every SUMMARY_S), comments as keep-alives
  /api/summary  the latest tables as JSON

Fills are read from the journal by id every POLL_S, so a trade shows a second or two after it is booked. The
tables (report books + launch-scorer outcomes) are rebuilt by a low-priority child process
(`python -m tradingbot dashboard-summary PATH`) so neither a page view nor the bot's own threads wait on it."""
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
LAUNCH_HOURS = 6  # window of the launch scorer's ideal outcomes and real-vs-ideal gaps
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


def fills_after(journal: Journal, after_id: int, limit: int = 200) -> list[dict]:
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


def _book_row(b: dict) -> dict:
    s, r = b["book"], b["rugs"]
    return {"name": b["name"], "hold_min": b["hold_min"], "started": b["started"],
            "equity": s["equity"], "return_pct": s["return_pct"], "max_drawdown_pct": s["max_drawdown_pct"],
            "trades": s["closed_trades"], "open": len(s["open_positions"] or []), "win_rate": s["win_rate"],
            "median_pct": s["median_trade_return_pct"], "mean_pct": s["mean_trade_return_pct"],
            "profit_factor": s["profit_factor"], "rugs": r["rugs"], "rug_pct": r["rug_rate_pct"],
            "signals": b["signals"], "buys": b["buys"], "fees_usd": s["network_fees_usd"],
            "failed_txs": s["failed_txs"], "skips": s["top_skip_reasons"]}


def build_summary(journal: Journal, cfg: Config, store=None, now: float | None = None) -> dict:
    from tradingbot import report
    now = now if now is not None else time.time()
    rep = report.build_model(journal, cfg)
    out = {"built_ts": now, "books": [_book_row(b) for b in rep["books"]],
           "expected": {b["name"]: report.EXPECTED.get(b["name"]) for b in rep["books"]},
           "downtime": rep.get("downtime"), "last_backup": rep.get("last_backup")}
    from tradingbot import health
    out["health"] = health.status(journal, cfg)[1]
    if store is not None and cfg.launch_live:
        try:
            from tradingbot import launch_live
            st = launch_live.status(store, cfg, LAUNCH_HOURS, now=now, journal=journal)
            gaps = {}
            for book, g in (st.get("real vs ideal fills") or {}).items():
                gaps[book] = {"trades": g.get("trades"),
                              "real_mean_pct": (g.get("book (real fills)") or {}).get("mean_pct"),
                              "ideal_mean_pct": (g.get("scorer (ideal fills), same pools") or {}).get("mean_pct"),
                              "gap_pct": (g.get("gap per trade (real - ideal)") or {}).get("mean_pct"),
                              "missed_n": (g.get("signalled but not bought (ideal)") or {}).get("n"),
                              "missed_mean_pct": (g.get("signalled but not bought (ideal)") or {}).get("mean_pct")}
            ws = st.get("websocket") or {}
            out["launch"] = {"hours": LAUNCH_HOURS, "counts_today": st.get("counts_today"),
                             "credits_today": st.get("credits_today"), "websocket_connected": ws.get("connected"),
                             "outcomes": st.get("outcomes (fill at second ~6, out at minute 6, 6% cost)"),
                             "real_vs_ideal": gaps}
        except Exception as exc:  # noqa: BLE001 (the books still show without the scorer's view)
            log.warning("dashboard: launch status failed: %s", exc)
            out["launch"] = {"error": str(exc)}
    return out


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
        threading.Thread(target=loop, name="dashboard", daemon=True).start()
        return self

    def current(self) -> tuple[int, dict | None]:
        with self._lock:
            return self.version, self.summary

    def stream(self, write, stop: threading.Event | None = None) -> None:
        """Write server-sent events to `write(bytes)` until it raises (the page closed) or `stop` is set."""
        def send(event: str, data) -> None:
            write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())

        stop = stop or threading.Event()
        version, summary = self.current()
        if summary is not None:
            send("summary", summary)
        backlog = last_fills(self.journal)
        send("backlog", backlog)
        last_id = backlog[-1]["id"] if backlog else (self.journal.query("SELECT MAX(id) AS m FROM fills")[0]["m"] or 0)
        quiet = time.time()
        while not stop.is_set():
            new = fills_after(self.journal, last_id)
            for f in new:
                send("fill", f)
                last_id = f["id"]
            v, summary = self.current()
            if v != version and summary is not None:
                send("summary", summary)
                version = v
            if new or v != version:
                quiet = time.time()
            elif time.time() - quiet > KEEPALIVE_S:
                write(b": keep-alive\n\n")
                quiet = time.time()
            stop.wait(POLL_S)


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
</style></head><body>
<header>
<div class="lockup"><svg viewBox="0 0 100 100" aria-hidden="true"><rect class="mk-sq" width="100" height="100"/><polygon class="mk-z" points="24,26 76,26 76,39 38,61 76,61 76,74 24,74 24,61 62,39 24,39"/></svg>
<div><div class="word">MZF</div><div class="desc">STRATEGY DESK</div></div></div>
<div class="status"><span id="conn" class="tag"><span class="dot"></span>Connecting</span><span id="health" class="tag">Bot</span>
<span id="built" class="label muted"></span></div>
</header>
<main>
<div><p class="label eyebrow">Paper books · simulated fills on live Jupiter quotes</p><h1>Launch strategies</h1>
<p class="lead">Every book starts at $10,000. Trades stream as they are booked. Tables are rebuilt every two minutes.</p></div>
<div class="stats">
<div class="stat lime"><span class="label">Leading book</span><span class="v" id="s-lead">-</span><span class="h" id="s-lead-h"></span></div>
<div class="stat"><span class="label">P&amp;L, all books</span><span class="v num" id="s-pnl">-</span><span class="h" id="s-pnl-h"></span></div>
<div class="stat"><span class="label">Closed trades</span><span class="v num" id="s-trades">-</span><span class="h" id="s-trades-h"></span></div>
<div class="stat"><span class="label">Trades, last hour</span><span class="v num" id="s-rate">0</span><span class="h">Counted while this page is open</span></div>
</div>
<div class="cols">
<section class="card"><p class="label eyebrow">Books</p><h2>Results by book</h2>
<p class="sub">Sorted by return. Click a header to sort. The leading row is highlighted. Open launch trades count at cost until their pool-minute sale.</p>
<div class="scroll"><table id="books"></table></div></section>
<aside class="feed"><p class="label" style="color:var(--on-band-2);margin:0 0 6px">Live</p><h2>Trades</h2>
<p class="sub" id="feed-sub">Newest first</p><ul id="feed"></ul></aside>
</div>
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
const fmt=(v,d=1,s="")=>v==null?"-":(typeof v==="number"?v.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}):v)+s;
const sgn=(v,d=1,s="%")=>v==null?"-":(v>0?"+":v<0?"−":"")+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d})+s;
const cls=v=>v==null?"":v>0?"pos":v<0?"neg":"";
const ago=ts=>{const s=Math.max(0,Date.now()/1000-ts);return s<90?Math.round(s)+"s":s<5400?Math.round(s/60)+"m":s<172800?(s/3600).toFixed(1)+"h":(s/86400).toFixed(1)+"d"};
const esc=t=>String(t??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let sortKey="return_pct",sortDir=-1,last=null;
const COLS=[["name","Book",0],["hold_min","Hold",0],["equity","Equity $",0],["return_pct","Return",2],["trades","Trades",0],["open","Open",0],["win_rate","Win",2],["median_pct","Median",1],["mean_pct","Mean",1],["profit_factor","PF",2],["rug_pct","Rugged",0],["max_drawdown_pct","Max DD",1]];
function books(s){const rows=[...s.books].sort((a,b)=>{const x=a[sortKey],y=b[sortKey];return (x==null)-(y==null)||(x<y?-1:x>y?1:0)*sortDir});
const leader=[...s.books].filter(b=>b.trades>0).sort((a,b)=>(b.return_pct??-1e9)-(a.return_pct??-1e9))[0];
let h="<tr>"+COLS.map(c=>`<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(sortDir<0?" ↓":" ↑"):""}</th>`).join("")+"</tr>";
for(const b of rows){h+=`<tr class="${leader&&b.name===leader.name?"lead-row":""}" title="${esc(s.expected[b.name]||"")}">`+COLS.map(([k,,d])=>{const v=b[k];
if(k==="name")return`<td class="book">${esc(v)}</td>`;if(k==="started")return`<td class="muted">${v?ago(v):"-"}</td>`;
if(k==="hold_min")return`<td>${v}m</td>`;if(k==="equity")return`<td>${fmt(v,0)}</td>`;
if(["return_pct","median_pct","mean_pct"].includes(k))return`<td class="${cls(v)}">${sgn(v,d)}</td>`;
if(k==="rug_pct"||k==="max_drawdown_pct")return`<td>${fmt(v,d,"%")}</td>`;return`<td>${fmt(v,d)}</td>`}).join("")+"</tr>"}
$("books").innerHTML=h;$("books").querySelectorAll("th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=sortKey===k?-sortDir:-1;sortKey=k;books(last)});
if(leader){$("s-lead").textContent=leader.name;$("s-lead-h").textContent=`${sgn(leader.return_pct,2)} on ${leader.trades} trades`}
const pnl=s.books.reduce((a,b)=>a+(b.equity??0)-10000,0);$("s-pnl").innerHTML=`<span class="${cls(pnl)}">${sgn(pnl,0,"")}</span>`;
$("s-pnl-h").textContent=`$ across ${s.books.length} books`;
$("s-trades").textContent=fmt(s.books.reduce((a,b)=>a+(b.trades||0),0),0);
$("s-trades-h").textContent=`${s.books.reduce((a,b)=>a+(b.open||0),0)} open now`}
function launch(s){const L=s.launch;if(!L||L.error){$("outcomes").innerHTML=`<tr><td class="muted">${esc(L&&L.error?L.error:"Launch scorer off")}</td></tr>`;$("gaps").innerHTML="";return}
$("lh").textContent=$("lh2").textContent=L.hours;
let h="<tr><th>Group</th><th>Trades</th><th>Mean</th><th>Median</th><th>Win</th><th>Rugged</th><th>+50%</th></tr>";
for(const[k,g]of Object.entries(L.outcomes||{})){const big=g["returns (% of trades)"]?g["returns (% of trades)"]["+50% or more"]:null;
h+=`<tr><td>${esc(k)}</td><td>${fmt(g.n,0)}</td><td class="${cls(g.mean_pct)}">${sgn(g.mean_pct)}</td><td>${sgn(g.median_pct)}</td><td>${fmt(g.win_pct,0,"%")}</td><td>${fmt(g.rugged_pct,0,"%")}</td><td>${fmt(big,0,"%")}</td></tr>`}
$("outcomes").innerHTML=h;
h="<tr><th>Book</th><th>Trades</th><th>Real</th><th>Ideal</th><th>Gap</th><th>Missed</th><th>Missed ideal</th></tr>";
for(const[k,g]of Object.entries(L.real_vs_ideal||{})){if(!g.trades&&!g.missed_n)continue;
h+=`<tr><td class="book">${esc(k)}</td><td>${fmt(g.trades,0)}</td><td class="${cls(g.real_mean_pct)}">${sgn(g.real_mean_pct)}</td><td>${sgn(g.ideal_mean_pct)}</td><td class="${cls(g.gap_pct)}">${sgn(g.gap_pct,1," pts")}</td><td>${fmt(g.missed_n,0)}</td><td>${sgn(g.missed_mean_pct)}</td></tr>`}
$("gaps").innerHTML=h}
function summary(s){last=s;books(s);launch(s);$("built").textContent="Tables "+new Date(s.built_ts*1000).toLocaleTimeString();
const ok=Object.values(s.health||{}).every(x=>x.ok);$("health").innerHTML=`<span class="dot"></span>${ok?"Bot healthy":"Bot stalled"}`;$("health").className="tag "+(ok?"ink":"neg");
const d=s.downtime;$("foot").textContent=d?`${d.outages} outages, ${fmt(d.hours_offline,1)} h offline since start`:""}
const times=[];
function fill(f,isNew){const li=document.createElement("li");if(isNew){li.className="new";times.push(Date.now())}
const t=new Date(f.ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});
const side=f.side==="buy"?'<span class="side buy">BUY</span>':'<span class="side">SELL</span>';
const right=f.side==="sell"?`<span class="r ${cls(f.ret_pct)}">${f.ret_pct==null?"part":sgn(f.ret_pct)}</span>`:`<span class="r">$${fmt(f.usd,0)}</span>`;
const why=f.side==="sell"&&f.reason?` · ${esc(f.reason.replace(/_/g," "))}`:"";
li.innerHTML=`<span class="t mono">${t}</span><span>${side}<b>${esc(f.symbol)}</b><span class="bk">${esc(f.book)}${why}</span></span>${right}`;
const feed=$("feed");feed.prepend(li);while(feed.children.length>300)feed.lastChild.remove()}
function rate(){const cut=Date.now()-3600e3;while(times.length&&times[0]<cut)times.shift();$("s-rate").textContent=times.length}
setInterval(rate,10000);
function connect(){const es=new EventSource("api/stream");
es.onopen=()=>{$("conn").innerHTML='<span class="dot"></span>Live';$("conn").className="tag lime"};
es.onerror=()=>{$("conn").innerHTML='<span class="dot"></span>Reconnecting';$("conn").className="tag neg"};
es.addEventListener("summary",e=>summary(JSON.parse(e.data)));
es.addEventListener("backlog",e=>{$("feed").innerHTML="";JSON.parse(e.data).forEach(f=>fill(f,false))});
es.addEventListener("fill",e=>{fill(JSON.parse(e.data),true);rate()})}
connect();
</script></body></html>"""
