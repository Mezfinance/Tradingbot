"""python -m shortbook run | status | report [--json] | check

run     the paper book (detector + positions + equity) with a /health endpoint
status  heartbeat, NAV, open positions, pending listings
report  results by group vs the research expectation, closed trades, skip reasons
check   which exchange endpoints answer from this machine (run it once after the first deploy)"""
from __future__ import annotations
import argparse, collections, json, logging, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .book import Book
from .config import Config
from .http import Http
from .store import Store
from .venues import LISTING, PERPS, SpotAges, BinanceSpot

EXPECTED = {"B upbit": 13.3, "B binance": 16.7, "A upbit": 29.5, "B coinbase": 6.0, "all": 14.3}  # research, % per trade


def build(cfg: Config, http: Http | None = None) -> Book:
    http = http or Http()
    listing = {n: LISTING[n](http) for n in cfg.names("listing_venues") if n in LISTING}
    perps = {n: PERPS[n](http) for n in cfg.names("perp_venues") if n in PERPS}
    return Book(cfg, Store(cfg.db_path), listing, perps, ages=SpotAges(http), spot_age=BinanceSpot(http))


def _ts(t) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)) if t else "-"


def status(book: Book) -> dict:
    s, now = book.s, time.time()
    hb = s.get("heartbeat")
    return {
        "heartbeat_age_s": round(now - hb) if hb else None,
        "started": _ts(s.get("started")),
        "nav_usd": round(book.nav(), 2), "cash_usd": round(float(s.get("cash")), 2),
        "baselines": {r["k"][9:]: _ts(json.loads(r["v"])) for r in s.q("SELECT k, v FROM state WHERE k LIKE 'baseline:%'")},
        "markets_known": {r["venue"]: r["n"] for r in s.q("SELECT venue, COUNT(*) n FROM markets GROUP BY venue")},
        "pending": [dict(venue=r["venue"], base=r["base"], seen=_ts(r["seen_ts"])) for r in s.q("SELECT * FROM listings WHERE status='pending'")],
        "open": [dict(base=p["base"], group=p["group_name"], venue=p["venue"], size_usd=round(p["size_usd"]),
                      entry=p["entry_px"], last=p["last_px"], funding_usd=round(p["funding_usd"] or 0, 2),
                      unrealized_usd=round(book.unrealized(p), 2), opened=_ts(p["entry_ts"]), due=_ts(p["due_ts"]))
                 for p in book.open_positions()],
        "recent_errors": [dict(ts=_ts(r["ts"]), **json.loads(r["data"])) for r in s.q("SELECT * FROM events WHERE kind='loop_error' ORDER BY ts DESC LIMIT 5")],
    }


def report(book: Book) -> dict:
    s = book.s
    closed = s.q("SELECT * FROM positions WHERE status='closed' ORDER BY exit_ts")
    by = collections.defaultdict(list)
    for p in closed:
        r = p["pnl_usd"] / p["size_usd"] * 100
        by[p["group_name"]].append(r)
        by["all"].append(r)
    groups = {g: {"trades": len(v), "mean_pct": round(sum(v) / len(v), 2), "win_pct": round(100 * sum(x > 0 for x in v) / len(v)),
                  "research_pct": EXPECTED.get(g)} for g, v in by.items()}
    eq = s.q("SELECT nav FROM equity ORDER BY ts")
    peak, dd = 0.0, 0.0
    for r in eq:
        peak = max(peak, r["nav"]); dd = min(dd, r["nav"] / peak - 1 if peak else 0)
    start = book.cfg.cash_usd
    listings = collections.Counter((r["venue"], r["status"]) for r in s.q("SELECT venue, status FROM listings"))
    reasons = collections.Counter(r["reason"] for r in s.q("SELECT reason FROM listings WHERE status='skipped'"))
    return {
        "nav_usd": round(book.nav(), 2), "return_pct": round((book.nav() / start - 1) * 100, 2), "max_drawdown_pct": round(dd * 100, 2),
        "by_group": groups,
        "closed": [dict(base=p["base"], group=p["group_name"], venue=p["venue"], size_usd=round(p["size_usd"]),
                        price_pct=round((p["entry_px"] - p["exit_px"]) / p["entry_px"] * 100, 2),
                        funding_usd=round(p["funding_usd"] or 0, 2), fees_usd=round(p["fees_usd"], 2),
                        pnl_usd=round(p["pnl_usd"], 2), opened=_ts(p["entry_ts"]), closed=_ts(p["exit_ts"])) for p in closed[-30:]],
        "listings": {f"{v} {st}": n for (v, st), n in sorted(listings.items())},
        "skip_reasons": dict(reasons.most_common(15)),
        "note": "Research expectation per trade after costs and funding: B upbit +13.3%, B binance +16.7%, A upbit +29.5%, "
                "B coinbase +6.0%, all +14.3%. Judge after 30+ closed trades.",
    }


def check(cfg: Config) -> dict:
    """Probe each venue the way the book uses it."""
    out = {}
    fail = lambda h: "FAILED: " + "; ".join(f"{k}: {e[:120]}" for k, e in h.errors.items() if e)
    for n, cls in LISTING.items():
        http = Http(retries=1)
        m = cls(http).markets()
        out[f"listing {n}"] = f"ok, {len(m)} markets" if m else fail(http)
    for n, cls in PERPS.items():
        http = Http(retries=1)
        p = cls(http)
        cs = p.contracts()
        if not cs:
            out[f"perp {n}"] = fail(http)
            continue
        c = cs.get("BTC") or next(iter(cs.values()))
        q = p.quote(c)
        f = p.funding(c, time.time() - 86400)
        out[f"perp {n}"] = (f"ok, {len(cs)} perps; {c.symbol} price {q.price if q else None}, OI ${q.oi_usd or 0:,.0f}, "
                            f"{len(f)} funding records in 24 h")
    a = SpotAges(Http(retries=1)).ages()
    out["ages gate+okx"] = f"ok, {len(a)} tokens" if a else "FAILED"
    return out


class _Health(BaseHTTPRequestHandler):
    book: Book = None

    def do_GET(self):
        hb = self.book.s.get("heartbeat")
        ok = hb is not None and time.time() - hb < self.book.cfg.watchdog_s
        self.send_response(200 if ok else 503)
        self.end_headers()
        self.wfile.write(json.dumps({"ok": ok, "heartbeat_age_s": round(time.time() - hb) if hb else None}).encode())

    def log_message(self, *a):
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="shortbook")
    ap.add_argument("cmd", choices=["run", "status", "report", "check"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = Config.from_env()
    if a.cmd == "check":
        for k, v in check(cfg).items():
            print(f"{k:16} {v}")
        return 0
    book = build(cfg)
    if a.cmd == "run":
        if cfg.health_port:
            _Health.book = book
            srv = ThreadingHTTPServer(("0.0.0.0", cfg.health_port), _Health)
            threading.Thread(target=srv.serve_forever, daemon=True, name="health").start()
        logging.info("shortbook paper book: NAV $%s, groups %s", f"{book.nav():,.0f}", cfg.groups)
        book.run_forever()
        return 0
    out = status(book) if a.cmd == "status" else report(book)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
