"""Realistic NET of the live front-run fires, settled on Helius true reserves.

The gate fires from shreds (the real box decisions, in front-fires.jsonl). This settles each fire as a trade:
buy FRONT_USD at the pool's TRUE reserves at the entry slot and sell the tokens back at the exit slot, with our own
constant-product impact on both legs + the pumpswap fee + a fixed priority cost. Because our real buy and sell
execute on the TRUE on-chain pool (not the box's view), the NET is settled on Helius `sol_after`/`tok_after` --
that is what our money actually gets, so an edge here is real, not the box's belief. For the PRICE-FREE
fire-and-hold the box makes no price decision, so this core NET needs no box reserve stamp at all.

Two things are swept / assumed, both stated:
  * LANDING SLOT (latency): no send happens in paper, so we don't know if we'd land at N+1 / N+2 / N+3 against
    other snipers -- swept. Read the headline at the PESSIMISTIC landing; only real money pins it.
  * SIZE: constant-product impact grows fast on thin fresh pools, so $5 / $25 / $50 are swept.

When a fire carries the box's WS reserve stamp ({sol, tok, reserve_slot}, lamports / raw units), we also report
the SOL-reserve DRIFT (box view vs true at the stamp slot) -- how stale/wrong the box's view was. The full
price-level drift and the min_out revert cost need the token decimals (a per-pool lookup) and are a follow-up;
the SOL side is decimals-free (WSOL is always 9) and is the dominant price/liquidity driver, so it is the first
validity read. The min_out revert only refines the NET downward; it never creates the edge.

Research Helius key only (never the live key):
    python -m tradingbot frontrun-fill --fires ~/front-fires.jsonl --fetch --max-credits 20000 --rps 2
"""
import json
import statistics as st

from . import flurry_study as fl
from . import frontrun_verify as fv
from .data import helius as he

SIZES = (5.0, 25.0, 50.0)          # USD per trade, swept (impact grows with size)
OFFSETS = (1, 2, 3)                # landing slot N+k, swept (latency assumption)
DEFAULT_SOL_USD = 83.0
FEE_BPS = 25.0                     # pumpswap taker fee per leg (bps)
PRIORITY_USD = 0.035              # fixed priority + base fee, buy + sell (dilutes with size)
LAMPORTS = 1e9                    # WSOL decimals: box SOL stamp is lamports, true sol_after is SOL


def _reserve_points(rec: dict) -> list[tuple[int, float, float]]:
    """Sorted (slot, sol_reserve, tok_reserve) from a launch_record's successful txs (human uiAmount units).
    A reverted row (0/0 after) is skipped -- it does not change the pool state."""
    pts = []
    for r in rec.get("txs") or []:
        slot = r[fl._SLOT]
        if slot is None or (len(r) > fl._FAILED and bool(r[fl._FAILED])):
            continue
        sol, tok = r[fl._SOLA], r[fl._TOKA]
        if sol and tok:
            pts.append((int(slot), float(sol), float(tok)))
    pts.sort(key=lambda p: p[0])
    return pts


def _reserves_at(pts: list[tuple[int, float, float]], slot: int) -> tuple[float, float] | None:
    """Carry-forward reserves at the end of `slot`: the last trade at or before it. None if nothing yet."""
    out = None
    for s, sol, tok in pts:
        if s <= slot:
            out = (sol, tok)
        else:
            break
    return out


def _cp_net(rs1: float, rt1: float, rs2: float, rt2: float, sol_in: float,
            fee: float, priority_sol: float) -> float | None:
    """Constant-product round trip: buy sol_in at (rs1, rt1), sell the tokens at (rs2, rt2). Fee on each leg's
    output, fixed priority once. Returns the NET return on sol_in, capped [-1, RET_CAP]."""
    if rs1 <= 0 or rt1 <= 0 or rs2 <= 0 or rt2 <= 0 or sol_in <= 0:
        return None
    sol_eff = sol_in * (1 - fee)                       # fee eats into the SOL that actually buys
    tok = rt1 * sol_eff / (rs1 + sol_eff)              # tokens out (our buy impact)
    if tok <= 0:
        return None
    sol_out = rs2 * tok / (rt2 + tok)                  # sell all tokens back (our sell impact)
    proceeds = sol_out * (1 - fee)
    net = (proceeds - sol_in - priority_sol) / sol_in
    return max(-1.0, min(net, fl.RET_CAP))


def fill_one(fire: dict, gate_hold: int = 2, sizes=SIZES, offsets=OFFSETS, sol_usd: float = DEFAULT_SOL_USD,
             fee_bps: float = FEE_BPS, priority_usd: float = PRIORITY_USD, helius=None, rec: dict | None = None,
             span_s: float = 180.0, max_pages: int | None = None) -> dict:
    """Settle one fire. `rec` short-circuits the fetch (tests / a box-dumped series). Returns per (offset,size) NET
    plus the box SOL-reserve drift when the fire carries a stamp."""
    pool, slot = fire["pool"], int(fire["slot"])
    out = {"pool": pool, "slot": slot}
    if rec is None:
        if helius is None:
            out["status"] = "no helius"
            return out
        mint = fire.get("mint") or fv._resolve_mint(helius, pool)
        if not mint:
            out["status"] = "mint unresolved"
            return out
        txs, _reached, _err = helius.history_from_start(pool, span_s, max_pages or fl.MAX_PAGES)
        rec = he.launch_record(txs, pool, mint, window_s=span_s)
    pts = _reserve_points(rec)
    if len(pts) < 2:
        out["status"] = "no flow (pool had < 2 priced trades in history)"
        return out
    flow = fl.slot_flow(rec, "full")
    pull = (flow or {}).get("pull_slot")
    fee = fee_bps / 10000.0
    priority_sol = priority_usd / sol_usd
    cells = {}
    for k in offsets:
        entry_slot, exit_slot = slot + k, slot + k + gate_hold
        r1 = _reserves_at(pts, entry_slot)
        r2 = _reserves_at(pts, exit_slot)
        rugged = pull is not None and entry_slot < pull <= exit_slot
        for size in sizes:
            key = (k, size)
            if rugged:
                cells[key] = -1.0
            elif r1 and r2:
                cells[key] = _cp_net(r1[0], r1[1], r2[0], r2[1], size / sol_usd, fee, priority_sol)
            else:
                cells[key] = None
    out["status"] = "ok"
    out["cells"] = {f"{k}|{size}": v for (k, size), v in cells.items()}
    # box SOL-reserve drift (decimals-free): box stamp is lamports, true sol_after is SOL
    sol_box = fire.get("sol")
    if sol_box is not None and fire.get("reserve_slot") is not None:
        tr = _reserves_at(pts, int(fire["reserve_slot"]))
        if tr and tr[0] > 0:
            out["sol_drift"] = (float(sol_box) / LAMPORTS) / tr[0] - 1.0
    return out


def measure(fires: list[dict], gate_hold: int = 2, sizes=SIZES, offsets=OFFSETS, sol_usd: float = DEFAULT_SOL_USD,
            fee_bps: float = FEE_BPS, priority_usd: float = PRIORITY_USD, helius=None, have: dict | None = None,
            span_s: float = 180.0, max_pages: int | None = None) -> dict:
    """Settle every fire; aggregate NET per (offset, size)."""
    rows = [fill_one(f, gate_hold, sizes, offsets, sol_usd, fee_bps, priority_usd, helius=helius,
                     rec=(have or {}).get(f["pool"]), span_s=span_s, max_pages=max_pages) for f in fires]
    ok = [r for r in rows if r.get("status") == "ok"]
    agg = {}
    for k in offsets:
        for size in sizes:
            key = f"{k}|{size}"
            vals = [r["cells"].get(key) for r in ok if r["cells"].get(key) is not None]
            if vals:
                agg[key] = {"n": len(vals), "mean": st.mean(vals), "median": st.median(vals),
                            "win": sum(1 for v in vals if v > 0) / len(vals),
                            "rug": sum(1 for v in vals if v <= -0.999) / len(vals),
                            "per_fire_usd": st.mean(vals) * size}
    drifts = [r["sol_drift"] for r in ok if "sol_drift" in r]
    return {"n_fires": len(fires), "n_checked": len(ok), "gate_hold": gate_hold, "sizes": list(sizes),
            "offsets": list(offsets), "sol_usd": sol_usd, "agg": agg, "rows": rows,
            "n_stamped": len(drifts),
            "sol_drift": ({"n": len(drifts), "median_abs": st.median([abs(d) for d in drifts]),
                           "p90_abs": sorted(abs(d) for d in drifts)[int(0.9 * (len(drifts) - 1))]} if drifts else None)}


def render(out: dict) -> str:
    L = [f"FRONT-RUN NET  (settled on Helius true reserves; fires = real box decisions)",
         f"  fires {out['n_fires']}  checked {out['n_checked']}  hold {out['gate_hold']} slots  "
         f"SOL ${out['sol_usd']:.0f}", "",
         "  Each cell = NET return after our constant-product impact + fee + priority, -100% on a rug in the hold.",
         "  Landing slot (N+k) is ASSUMED (latency) -- read the PESSIMISTIC k; real money pins it.", ""]
    L.append(f"  {'land':>6}" + "".join(f"{'$'+str(int(s)):>22}" for s in out["sizes"]))
    for k in out["offsets"]:
        parts = [f"  N+{k:<4}"]
        for s in out["sizes"]:
            c = out["agg"].get(f"{k}|{s}")
            if c:
                parts.append(f"{c['mean']*100:>+7.1f}% md{c['median']*100:>+5.1f} w{c['win']*100:>3.0f} ${c['per_fire_usd']:>+6.3f}")
            else:
                parts.append(f"{'--':>22}")
        L.append("".join(parts))
    L += ["", "  (mean% / median / win% / $ per fire; rug share folded into the mean as -100%)"]
    if out.get("sol_drift"):
        d = out["sol_drift"]
        L += ["", f"  BOX SOL-RESERVE DRIFT (stamped fires {out['n_stamped']}): median |err| {d['median_abs']*100:.2f}%  "
                  f"p90 {d['p90_abs']*100:.2f}%  -- how far the box's WS SOL reserve was from truth at the fire."]
    else:
        L += ["", "  No box reserve stamps on these fires -> core NET only (price-free, settled on truth). Wire the",
              "  WS feed to add the SOL-drift validity read and the min_out revert cost."]
    best = max(out["agg"].items(), key=lambda kv: kv[1]["mean"], default=None)
    if best:
        (key, c) = best
        k, s = key.split("|")
        pay = c["mean"] > 0 and c["median"] > 0
        L += ["", f"  => BEST CELL land N+{k} size ${float(s):.0f}: mean {c['mean']*100:+.1f}% median {c['median']*100:+.1f}% "
                  f"win {c['win']*100:.0f}% ${c['per_fire_usd']:+.3f}/fire"
                  + ("  -- PAYS (mean AND median > 0)" if pay else "  -- TAIL only (median <= 0; the edge is the tail, needs real NET)")]
    return "\n".join(L)


def load_fires(path: str) -> list[dict]:
    text = open(path).read()
    return fv.parse_fires(text)
