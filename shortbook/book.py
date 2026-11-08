"""The paper short book: detect new spot listings, classify them, paper-short the perp, cover after 7 days.

Rules follow the research model (research/listings/s1_fund.py): each short = min(trade_pct of NAV, oi_cap_pct of the
venue's open interest), total open shorts <= gross_cap_pct of NAV, cost = fee_round_trip_pct + 2 x impact where
impact = impact_k x sqrt(size / 24h perp volume), funding from the venue's own funding history (a positive rate
means longs pay shorts), no stop. The venue with the largest open interest among those listing the perp is used."""
from __future__ import annotations
import json, logging, math, time
from .config import Config
from .store import Store
from . import leverage, market
from .venues import STABLE, Contract, Quote

log = logging.getLogger("shortbook")
CONTRACTS_TTL_S = 600
MIN_TRADE_USD = 100.0
DEFAULT_IMPACT = 0.005      # per side, when the venue reports no 24h volume
MAX_IMPACT = 0.05


class Book:
    def __init__(self, cfg: Config, store: Store, listing: dict, perps: dict, ages=None, spot_age=None, now=time.time):
        self.cfg, self.s, self.listing, self.perps = cfg, store, listing, perps
        self.ages, self.spot_age, self.now = ages, spot_age, now
        self._contracts: dict[str, dict[str, Contract]] = {}
        self._contracts_ts = 0.0
        leverage.ensure(self.s)
        if self.s.get("cash") is None:
            self.s.put("cash", cfg.cash_usd)
            self.s.put("started", self.now())

    # ------------------------------------------------------------ detection
    def detect(self) -> list[int]:
        """Record new markets; returns ids of new pending listings. The first read of a venue is the baseline."""
        new = []
        for name, v in self.listing.items():
            mk = v.markets()
            if not mk:
                continue
            known = {r["base"] for r in self.s.q("SELECT base FROM markets WHERE venue=?", (name,))}
            first = not self.s.get(f"baseline:{name}")
            t = self.now()
            for base, sym in mk.items():
                if base in known:
                    continue
                self.s.x("INSERT OR IGNORE INTO markets (venue, base, symbol, first_seen) VALUES (?, ?, ?, ?)", (name, base, sym, t))
                if first or base in STABLE:
                    continue
                lid = self.s.x("INSERT OR IGNORE INTO listings (venue, base, symbol, seen_ts, status) VALUES (?, ?, ?, ?, 'pending')",
                               (name, base, sym, t))
                if lid:
                    new.append(lid)
                    self.s.event("new_market", venue=name, base=base, symbol=sym)
                    log.info("new market %s %s (%s)", name, base, sym)
            if first:
                self.s.put(f"baseline:{name}", t)
                self.s.event("baseline", venue=name, markets=len(mk))
                log.info("baseline %s: %d markets", name, len(mk))
        return new

    def check_pending(self) -> None:
        for r in self.s.q("SELECT * FROM listings WHERE status='pending'"):
            if self.now() - r["seen_ts"] > self.cfg.pending_max_h * 3600:
                self._close_listing(r["id"], "expired", "no trades seen")
                continue
            v = self.listing.get(r["venue"])
            t = v.first_trade(r["symbol"]) if v else None
            if t:
                self.s.x("UPDATE listings SET trade_ts=? WHERE id=?", (t, r["id"]))
                self.evaluate(r["id"])

    # ------------------------------------------------------------ perps
    def contracts(self, force: bool = False) -> dict[str, dict[str, Contract]]:
        if force or not self._contracts or self.now() - self._contracts_ts > CONTRACTS_TTL_S:
            for name, p in self.perps.items():
                c = p.contracts()
                if c is not None:
                    self._contracts[name] = c
            self._contracts_ts = self.now()
        return self._contracts

    def _age_evidence(self, base: str, venue: str, trade_ts: float) -> dict[str, float]:
        """When the token first traded elsewhere: perp launches, Gate/OKX spot, Binance spot."""
        ev = {}
        for name, cs in self.contracts().items():
            c = cs.get(base)
            if c and c.launch_ts:
                ev[f"perp:{name}"] = c.launch_ts
        if self.ages:
            for name, t in self.ages.ages().get(base, {}).items():
                ev[f"spot:{name}"] = t
        if self.spot_age and venue != "binance":
            t = self.spot_age.first_trade_base(base)
            if t:
                ev["spot:binance"] = t
        return {k: t for k, t in ev.items() if t < trade_ts - 60}

    # ------------------------------------------------------------ entry
    def evaluate(self, lid: int) -> int | None:
        r = self.s.q("SELECT * FROM listings WHERE id=?", (lid,))[0]
        base, venue, trade_ts = r["base"], r["venue"], r["trade_ts"]
        data: dict = {}
        now = self.now()
        if now - trade_ts > self.cfg.max_entry_delay_h * 3600:
            return self._skip(lid, f"trading began {(now - trade_ts) / 3600:.1f} h before we saw it", data)
        ev = self._age_evidence(base, venue, trade_ts)
        first_ts = min(ev.values()) if ev else None
        age_d = (trade_ts - first_ts) / 86400 if first_ts else 0.0
        data.update(age_days=round(age_d, 2), age_evidence=ev)
        kind = "B" if age_d >= self.cfg.b_min_age_days else "A" if age_d <= self.cfg.a_max_age_days else None
        self.s.x("UPDATE listings SET kind=? WHERE id=?", (kind, lid))
        if kind is None:
            return self._skip(lid, f"between A and B (traded elsewhere {age_d:.1f} days)", data)
        if (kind, venue) not in self.cfg.group_set():
            return self._skip(lid, f"group {kind} {venue} not traded", data)
        if self.s.q("SELECT 1 FROM positions WHERE base=? AND status='open'", (base,)):
            return self._skip(lid, "already short this token", data)
        quotes: dict[str, tuple[Contract, Quote]] = {}
        for name, cs in self.contracts().items():
            c = cs.get(base)
            if c and (c.launch_ts is None or c.launch_ts <= now):
                q = self.perps[name].quote(c)
                if q:
                    quotes[name] = (c, q)
        data["perps"] = {n: {"symbol": c.symbol, "price": q.price, "oi_usd": q.oi_usd, "vol24_usd": q.vol24_usd,
                             "funding_rate": q.funding_rate} for n, (c, q) in quotes.items()}
        if not quotes:
            return self._skip(lid, "no perp on " + "/".join(self.perps), data)
        lv = self.listing.get(venue)
        lpx = lv.price_usd(r["symbol"]) if lv else None
        data["listing_price_usd"] = lpx
        if lpx:
            bad = [n for n, (c, q) in quotes.items() if max(lpx / q.price, q.price / lpx) > self.cfg.max_price_ratio]
            for n in bad:
                quotes.pop(n)
            if not quotes:
                return self._skip(lid, "perp price doesn't match the listing (different token with the same ticker?)", data)
        name, (c, q) = max(quotes.items(), key=lambda kv: kv[1][1].oi_usd or 0)
        if not q.oi_usd:
            return self._skip(lid, f"no open interest reported on {name}", data)
        nav = self.nav()
        open_usd = sum(p["size_usd"] for p in self.open_positions())
        want = self.cfg.trade_pct / 100 * nav
        cap = self.cfg.oi_cap_pct / 100 * q.oi_usd
        room = self.cfg.gross_cap_pct / 100 * nav - open_usd
        size = min(want, cap, room)
        data.update(nav=nav, want_usd=want, oi_cap_usd=cap, room_usd=room)
        if size < MIN_TRADE_USD:
            return self._skip(lid, f"no room (open shorts ${open_usd:,.0f} of NAV ${nav:,.0f})", data)
        impact = min(MAX_IMPACT, self.cfg.impact_k * math.sqrt(size / q.vol24_usd)) if q.vol24_usd else DEFAULT_IMPACT
        fees = size * (self.cfg.fee_round_trip_pct / 100 + 2 * impact)
        data["limits"] = self.perps[name].limits(c)
        group = f"{kind} {venue}"
        pid = self.s.x(
            "INSERT INTO positions (listing_id, group_name, base, venue, symbol, mult, size_usd, entry_px, entry_ts, due_ts,"
            " fees_usd, impact_pct, funding_usd, funding_last_ts, last_px, last_ts, status, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 'open', ?)",
            (lid, group, base, name, c.symbol, c.mult, size, q.price, now, now + self.cfg.hold_h * 3600, fees,
             impact * 100, now, q.price, now, json.dumps(data, default=str)))
        self._close_listing(lid, "entered", f"short ${size:,.0f} on {name} {c.symbol}", data)
        self.s.event("short_open", position_id=pid, base=base, group=group, venue=name, size_usd=size, price=q.price)
        log.info("SHORT %s (%s) $%.0f on %s @ %.6g", base, group, size, name, q.price)
        return pid

    def _skip(self, lid: int, reason: str, data: dict) -> None:
        self._close_listing(lid, "skipped", reason, data)
        r = self.s.q("SELECT venue, base FROM listings WHERE id=?", (lid,))[0]
        self.s.event("skip", venue=r["venue"], base=r["base"], reason=reason)
        log.info("skip %s %s: %s", r["venue"], r["base"], reason)
        return None

    def _close_listing(self, lid: int, status: str, reason: str, data: dict | None = None) -> None:
        self.s.x("UPDATE listings SET status=?, reason=?, data=? WHERE id=?",
                 (status, reason, json.dumps(data or {}, default=str), lid))

    # ------------------------------------------------------------ positions
    def open_positions(self):
        return self.s.q("SELECT * FROM positions WHERE status='open' ORDER BY id")

    @staticmethod
    def unrealized(p) -> float:
        return p["size_usd"] * (p["entry_px"] - p["last_px"]) / p["entry_px"] + (p["funding_usd"] or 0) - p["fees_usd"]

    def nav(self) -> float:
        return float(self.s.get("cash")) + sum(self.unrealized(p) for p in self.open_positions())

    def manage(self) -> None:
        cs = self.contracts()
        for p in self.open_positions():
            perp = self.perps.get(p["venue"])
            c = (cs.get(p["venue"]) or {}).get(p["base"]) or Contract(p["venue"], p["symbol"], p["base"], p["mult"], None)
            q = perp.quote(c) if perp else None
            now = self.now()
            last_px = q.price if q else p["last_px"]
            if q:
                self.s.x("UPDATE positions SET last_px=?, last_ts=? WHERE id=?", (q.price, now, p["id"]))
                leverage.track_high(self, p, q.price)
            # funding paid at each settlement since the last one we booked (positive rate: longs pay shorts)
            if perp:
                fund = p["funding_usd"] or 0.0
                last = p["funding_last_ts"] or p["entry_ts"]
                for ts, rate in perp.funding(c, last + 1):
                    if ts <= last or ts > min(now, p["due_ts"]):
                        continue
                    usd = rate * p["size_usd"] * last_px / p["entry_px"]
                    self.s.x("INSERT OR IGNORE INTO funding (position_id, ts, rate, usd) VALUES (?, ?, ?, ?)", (p["id"], ts, rate, usd))
                    fund += usd
                    last = ts
                self.s.x("UPDATE positions SET funding_usd=?, funding_last_ts=? WHERE id=?", (fund, last, p["id"]))
            if now >= p["due_ts"]:
                self.close(p["id"], last_px, "7 days" if q else "7 days (no live quote: last known price)")

    def close(self, pid: int, px: float, why: str) -> float:
        p = self.s.q("SELECT * FROM positions WHERE id=?", (pid,))[0]
        pnl = p["size_usd"] * (p["entry_px"] - px) / p["entry_px"] + (p["funding_usd"] or 0) - p["fees_usd"]
        self.s.x("UPDATE positions SET exit_px=?, exit_ts=?, last_px=?, pnl_usd=?, status='closed' WHERE id=?",
                 (px, self.now(), px, pnl, pid))
        self.s.put("cash", float(self.s.get("cash")) + pnl)
        self.s.event("short_close", position_id=pid, base=p["base"], price=px, pnl_usd=pnl, why=why)
        log.info("COVER %s @ %.6g: %+.0f USD (%s)", p["base"], px, pnl, why)
        return pnl

    def snapshot(self) -> None:
        ops = self.open_positions()
        unreal = sum(self.unrealized(p) for p in ops)
        cash = float(self.s.get("cash"))
        now = self.now()
        self.s.x("INSERT OR REPLACE INTO equity (ts, nav, cash, open_usd, unrealized) VALUES (?, ?, ?, ?, ?)",
                 (now, cash + unreal, cash, sum(p["size_usd"] for p in ops), unreal))
        leverage.snapshot(self, now)

    # ------------------------------------------------------------ loop
    def step(self, last: dict) -> None:
        """One pass of whatever is due. `last` keeps the time each job last ran."""
        now = self.now()
        jobs = (("detect", self.cfg.detect_s, lambda: (self.detect(), self.check_pending())),
                ("manage", self.cfg.manage_s, self.manage),
                ("equity", self.cfg.equity_s, self.snapshot),
                ("marks", self.cfg.manage_s, lambda: market.track(self)))
        for name, every, fn in jobs:
            if now - last.get(name, 0) >= every:
                try:
                    fn()
                except Exception as e:  # keep the loop alive; the error is journalled
                    log.exception("%s failed", name)
                    self.s.event("loop_error", job=name, error=f"{type(e).__name__}: {e}")
                last[name] = now
        self.s.put("heartbeat", now)

    def run_forever(self, stop=None) -> None:
        last: dict = {}
        while not (stop and stop.is_set()):
            self.step(last)
            time.sleep(1.0)
