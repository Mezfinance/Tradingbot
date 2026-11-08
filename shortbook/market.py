"""Market analysis for MZF Short Only: every new listing's price path, so the book's picks can be judged against
the listings it skipped, against shorting every listing and against buying every listing.

`track(book)` (a job in `Book.step`) records, for every listing that traded (status entered or skipped), the
listing venue's spot price at `T0_DELAY_S` after its first trade (the start, close to when the book enters) and at
`HORIZONS_H` after that, from historical 1-minute candles, so listings seen before this module existed are filled
in too. A price that can't be found is retried until `GIVE_UP_S` after it was due, then stored as NULL.

`analyse(store, cost_pct)` compares, at each horizon: the picks (shorted), the skipped listings (all and by reason),
shorting every listing and going long every listing, split by kind (A / B / between / not classified) and venue.
Short return = -(price_h / price_0 - 1) - cost, floored at -100%; long = price_h / price_0 - 1 - cost. Spot prices,
no funding: one yardstick for every listing. The picks' real book result (perp fills, fees, funding) is shown
beside it."""
from __future__ import annotations
import collections, random, statistics, time

T0_DELAY_S = 300
HORIZONS_H = (1, 24, 72, 168)
GIVE_UP_S = 86400
PER_STEP = 40
DRAWS = 1000

REASONS = (("trading began", "seen too late"), ("between A and B", "between A and B"),
           ("group ", "group not traded"), ("already short", "already short"), ("no perp", "no perp"),
           ("perp price doesn't match", "perp price mismatch"), ("no open interest", "no open interest"),
           ("no room", "no room"))


def reason_of(text: str | None) -> str:
    for prefix, name in REASONS:
        if (text or "").startswith(prefix):
            return name
    return "other"


def ensure(store) -> None:
    store.x("CREATE TABLE IF NOT EXISTS marks (listing_id INTEGER, h REAL, ts REAL, px REAL, PRIMARY KEY (listing_id, h))")


def track(book, limit: int = PER_STEP) -> int:
    """Fill due marks; returns how many price lookups were made."""
    s, now = book.s, book.now()
    ensure(s)
    have = collections.defaultdict(set)
    for r in s.q("SELECT listing_id, h FROM marks"):
        have[r["listing_id"]].add(r["h"])
    calls = 0
    rows = s.q("SELECT id, venue, symbol, trade_ts FROM listings WHERE trade_ts IS NOT NULL AND status IN ('entered', 'skipped')"
               " ORDER BY trade_ts DESC")
    for r in rows:
        v = book.listing.get(r["venue"])
        if v is None or not hasattr(v, "price_at"):
            continue
        t0 = r["trade_ts"] + T0_DELAY_S
        done = have[r["id"]]
        for h in (0, *HORIZONS_H):
            if h in done:
                continue
            due = t0 + h * 3600
            if due > now or (h and 0 not in done):
                break
            if calls >= limit:
                return calls
            calls += 1
            px = v.price_at(r["symbol"], due)
            if px or now - due > GIVE_UP_S:
                s.x("INSERT OR REPLACE INTO marks (listing_id, h, ts, px) VALUES (?, ?, ?, ?)", (r["id"], h, now, px or None))
                done.add(h)
                if not px and h == 0:
                    break
            else:
                break
    return calls


def _stats(rets: list[float], rng: random.Random) -> dict:
    if not rets:
        return {"n": 0}
    m = sum(rets) / len(rets)
    out = {"n": len(rets), "mean_pct": round(100 * m, 2), "median_pct": round(100 * statistics.median(rets), 2),
           "win_pct": round(100 * sum(x > 0 for x in rets) / len(rets)), "worst_pct": round(100 * min(rets), 1),
           "best_pct": round(100 * max(rets), 1)}
    if len(rets) >= 5:
        means = sorted(sum(rng.choices(rets, k=len(rets))) / len(rets) for _ in range(DRAWS))
        out["ci90"] = [round(100 * means[int(0.05 * DRAWS)], 2), round(100 * means[int(0.95 * DRAWS) - 1], 2)]
    return out


def _gap(a: list[float], b: list[float], rng: random.Random) -> dict | None:
    if len(a) < 3 or len(b) < 3:
        return None
    g = sum(a) / len(a) - sum(b) / len(b)
    draws = sorted(sum(rng.choices(a, k=len(a))) / len(a) - sum(rng.choices(b, k=len(b))) / len(b) for _ in range(DRAWS))
    return {"pts": round(100 * g, 2), "ci90": [round(100 * draws[int(0.05 * DRAWS)], 2), round(100 * draws[int(0.95 * DRAWS) - 1], 2)]}


def rows(store) -> list[dict]:
    ensure(store)
    marks = collections.defaultdict(dict)
    for m in store.q("SELECT listing_id, h, px FROM marks"):
        marks[m["listing_id"]][m["h"]] = m["px"]
    out = []
    for r in store.q("SELECT id, venue, base, kind, status, reason, trade_ts FROM listings"
                     " WHERE trade_ts IS NOT NULL AND status IN ('entered', 'skipped')"):
        mk = marks.get(r["id"], {})
        p0 = mk.get(0)
        moves = {h: mk[h] / p0 - 1 for h in HORIZONS_H if p0 and mk.get(h)}
        out.append({"id": r["id"], "venue": r["venue"], "base": r["base"], "kind": r["kind"] or "", "status": r["status"],
                    "reason": "" if r["status"] == "entered" else reason_of(r["reason"]), "trade_ts": r["trade_ts"],
                    "moves": moves})
    return out


def analyse(store, cost_pct: float = 0.2, seed: int = 7) -> dict:
    rng = random.Random(seed)
    data, cost = rows(store), cost_pct / 100
    short = lambda m: max(-1.0, -m - cost)
    long_ = lambda m: m - cost
    kind = lambda r: {"A": "A (new token)", "B": "B (traded elsewhere 7+ days)"}.get(r["kind"]) or (
        "between A and B" if r["reason"] == "between A and B" else "not classified")
    out = {"cost_pct": cost_pct, "t0_delay_s": T0_DELAY_S, "horizons_h": list(HORIZONS_H),
           "listings": len(data), "priced": sum(1 for r in data if r["moves"]),
           "since_ts": min((r["trade_ts"] for r in data), default=None), "horizons": {}}
    for h in HORIZONS_H:
        have = [r for r in data if h in r["moves"]]
        picks = [short(r["moves"][h]) for r in have if r["status"] == "entered"]
        skips = [short(r["moves"][h]) for r in have if r["status"] == "skipped"]
        every = [short(r["moves"][h]) for r in have]
        groups = {"picks (shorted)": _stats(picks, rng), "skipped": _stats(skips, rng),
                  "short every listing": _stats(every, rng),
                  "long every listing": _stats([long_(r["moves"][h]) for r in have], rng)}
        by_reason = collections.defaultdict(list)
        by_kind = collections.defaultdict(list)
        by_venue = collections.defaultdict(list)
        for r in have:
            x = short(r["moves"][h])
            if r["status"] == "skipped":
                by_reason[r["reason"]].append(x)
            by_kind[kind(r)].append(x)
            by_venue[r["venue"]].append(x)
        out["horizons"][str(h)] = {
            "groups": groups,
            "picks_vs_skipped": _gap(picks, skips, rng), "picks_vs_every": _gap(picks, every, rng),
            "skip_reasons": {k: _stats(v, rng) for k, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))},
            "kinds": {k: _stats(v, rng) for k, v in sorted(by_kind.items())},
            "venues": {k: _stats(v, rng) for k, v in sorted(by_venue.items())},
        }
    real = [p["pnl_usd"] / p["size_usd"] for p in store.q("SELECT pnl_usd, size_usd FROM positions WHERE status='closed'")
            if p["size_usd"]]
    out["picks_real"] = _stats(real, rng)
    return out


def _cell(st: dict) -> str:
    if not st or not st.get("n"):
        return f"{'-':>7} {'':>6} {'':>5} {'':>15}"
    ci = st.get("ci90")
    ci = f"{ci[0]:+.1f}..{ci[1]:+.1f}" if ci else ""
    return f"{st['mean_pct']:+7.1f} {st['median_pct']:+6.1f} {st['win_pct']:4d}% {ci:>15}"


def render(out: dict) -> str:
    since = time.strftime("%Y-%m-%d %H:%M", time.gmtime(out["since_ts"])) if out.get("since_ts") else "-"
    lines = [f"SHORT ONLY MARKET: {out['listings']} listings that traded since {since} UTC, {out['priced']} priced.",
             f"Spot price {out['t0_delay_s'] // 60} min after the first trade -> each horizon; {out['cost_pct']}% round trip, "
             "no funding; short floored at -100%.", ""]
    head = f"{'':30} {'n':>5} {'mean':>7} {'median':>6} {'win':>5} {'90% CI of mean':>15}"
    for h, hz in out["horizons"].items():
        lines += [f"== {h} h ==", head]
        for title, part in (("", hz["groups"]), ("skipped because", hz["skip_reasons"]), ("kind (short)", hz["kinds"]),
                            ("venue (short)", hz["venues"])):
            if title:
                lines.append(f"  {title}:")
            for k, st in part.items():
                lines.append(f"  {k[:28]:28} {st.get('n', 0):5d} {_cell(st)}")
        for k in ("picks_vs_skipped", "picks_vs_every"):
            g = hz.get(k)
            if g:
                lines.append(f"  {k.replace('_', ' ')}: {g['pts']:+.1f} pts (90% CI {g['ci90'][0]:+.1f}..{g['ci90'][1]:+.1f})")
        lines.append("")
    r = out.get("picks_real") or {}
    lines.append(f"Picks' real book result (perp fills, fees, funding, 7-day cover): {r.get('n', 0)} closed trades, "
                 + (f"mean {r['mean_pct']:+.1f}%, median {r['median_pct']:+.1f}%, win {r['win_pct']}%" if r.get("n") else "none yet"))
    return "\n".join(lines)
