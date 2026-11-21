"""Verify the live front-run box is reading signals correctly.

The gate scoring (precursor_gate) and the online SlotBank (frontrun_live) are already pinned to the study
field-for-field by tests/test_frontrun_live.py and tests/test_precursor_gate.py. So given a CORRECT per-slot
series the box fires exactly where the backtest did -- the one thing those tests can't cover is whether the box's
live ShredStream aggregation reproduces the chain's slots. This module closes that seam: it replays each pool the
box fired on through the study's OWN ground-truth path (Helius history_from_start -> launch_record -> slot_flow ->
frontrun_live.SlotBank + precursor_gate) and checks the gate fires on the box's fire slot, with the same
sym / m2 / eta the box logged.

The replay uses the VISIBILITY the gate was fit on (`gate["vis"]`, default 'full'): a 'direct' gate is replayed
through slot_flow(vis='direct') on vis-tagged recs, so the verifier reads the box's OWN view (top-level pumpswap
only). A gate fit on 'full' but scored against the box's direct view fires on the wrong slots -- fit and verify
on the same view (flurry-study --frontrun --gate-vis direct).

Why this isolates the feed, not the brain: the precursor gate is price- and magnitude-free (every predictor is a
symbol / motif / wallet-rank / sell-count), so a reserve-basis or vq mismatch CANNOT change which slots fire. Only
the per-slot wallet set (fee payer), the buy/sell classification, and emitting a record for every slot (dead slots
included, for the motifs / age) can -- and those are exactly what a mismatch here points at.

Run on Fly or the box with the RESEARCH Helius key (never the live key):
    python -m tradingbot frontrun-verify --fires ~/front-fires.jsonl --gate gate.json --tier 1 --fetch
    python -m tradingbot frontrun-verify --pairs "FULLPOOL:SLOT,FULLPOOL2:SLOT2" --gate gate.json --tier 1 --fetch

The box must log the FULL pool address (and ideally token_mint) on each fire; a truncated pool cannot be fetched.
"""
import json
import re

from . import flurry_study as fl
from . import frontrun_live as live
from . import precursor_gate as pg
from .data import helius as he
from .data import pumpswap

_FIRE_RE = re.compile(r"GATE FIRE\s+(\S+)\s+slot\s+(\d+)")


def parse_fires(text: str) -> list[dict]:
    """Accept a JSON list / JSONL of {pool, slot, token_mint?, sym?, m2?, eta?} OR a raw box log (grep the
    'GATE FIRE <pool> slot <N>' lines). Returns [{pool, slot, box}] where box holds whatever the box recorded."""
    text = text.strip()
    out: list[dict] = []
    if text.startswith("[") or text.lstrip().startswith("{"):
        rows = json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
        for r in rows:
            if r.get("pool") and r.get("slot") is not None:
                box = {k: r[k] for k in ("sym", "m2", "m3", "eta") if k in r}
                out.append({"pool": r["pool"], "slot": int(r["slot"]), "mint": r.get("token_mint"), "box": box})
        return out
    for m in _FIRE_RE.finditer(text):               # fall back to parsing a plain log blob
        out.append({"pool": m.group(1), "slot": int(m.group(2)), "mint": None, "box": {}})
    return out


def parse_pairs(spec: str) -> list[dict]:
    """"pool:slot[:mint],pool:slot" -> [{pool, slot, mint, box:{}}]. For feeding box-logged fires by hand."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        out.append({"pool": bits[0], "slot": int(bits[1]), "mint": (bits[2] if len(bits) > 2 else None), "box": {}})
    return out


def _resolve_mint(helius, pool: str) -> str | None:
    """The pool's token mint from its on-chain account (pumpswap layout). launch_record needs it to read vaults."""
    acc = helius.accounts([pool])
    if not acc or not acc.get(pool):
        return None
    parsed = pumpswap.parse_pool(acc[pool][1])
    return parsed.get("token_mint") if parsed else None


def verify_one(fire: dict, gate: dict, tier: str = "1", flurry_min: int = 3,
               helius=None, rec: dict | None = None, span_s: float = 180.0) -> dict:
    """Replay one fired pool through the study's ground-truth path and score the box's gate on every slot.
    `rec` (a launch_record dict) short-circuits the fetch -- used by tests and when the box dumped its own series."""
    pool, slot, box = fire["pool"], int(fire["slot"]), fire.get("box") or {}
    gate_vis = gate.get("vis", "full")              # replay through the VIEW the gate was fit on = the box's own view
    res = {"pool": pool, "slot": slot, "box": box, "vis": gate_vis}
    if rec is None:
        if helius is None:
            res["status"] = "no helius"
            return res
        mint = fire.get("mint") or _resolve_mint(helius, pool)
        if not mint:
            res["status"] = "mint unresolved"
            return res
        txs, _more, _err = helius.history_from_start(pool, span_s, fl.MAX_PAGES)
        rec = he.launch_record(txs, pool, mint, window_s=span_s, with_vis=(gate_vis != "full"))
    saved = fl.USE_VQ
    try:
        fl.USE_VQ = False                           # the gate is price-free, so the basis cannot change a fire
        flow = fl.slot_flow(rec, gate_vis)          # direct = only top-level pumpswap (what the box decodes)
    finally:
        fl.USE_VQ = saved
    if not flow:
        res["status"] = "no flow (pool had < 3 priced slots in history)"
        return res
    bank = live.SlotBank(flurry_min)
    thr = (gate.get("threshold") or {}).get(str(tier))
    hold = int(gate.get("hold", 2))
    fired, at = [], None
    for s in flow["slots"]:
        r = bank.push(s)
        e = pg.eta(r, gate)
        hit = thr is not None and e >= float(thr)
        if hit:
            fired.append(s["slot"])
        if s["slot"] == slot:
            # abs_move AT the signal slot = the backrun discriminator: ~0 => a genuine precursor fire (the move
            # is ahead, in N+1..N+hold); large => the fire overlaps a move already landed (backrun contamination).
            entry = s.get("price") or fl._price_at(flow["slots"], slot)
            exitp = fl._price_at(flow["slots"], slot + hold)
            fwd = (exitp / entry - 1.0) if (entry and exitp) else None
            at = {"sym": r["sym"], "m2": r["m2"], "m3": r["m3"], "eta": round(e, 4), "fires": hit,
                  "abs_move": round(float(r["f"].get("abs_move", 0.0)), 4),
                  "fwd_move": (round(fwd, 4) if fwd is not None else None)}
    res["status"] = "ok"
    res["n_slots"] = len(flow["slots"])
    res["in_flow"] = at is not None
    res["offline"] = at
    res["fired_slots"] = fired
    res["reproduced"] = bool(at and at["fires"])    # the box fire reproduces on chain ground truth
    if at and box:                                  # field-by-field agreement where the box recorded its record
        res["match"] = {k: (box.get(k) == at.get(k)) for k in ("sym", "m2", "m3") if k in box}
        if "eta" in box:
            res["match"]["eta"] = abs(float(box["eta"]) - at["eta"]) < 0.02
    return res


def verify(fires: list[dict], gate: dict, helius=None, have: dict | None = None,
           tier: str = "1", flurry_min: int = 3, span_s: float = 180.0) -> dict:
    """Verify every fire. `have` (pool -> launch_record dict) injects ground-truth recs without Helius (tests)."""
    if "_terms" not in gate:
        pg.compile_gate(gate)
    rows = [verify_one(f, gate, tier, flurry_min, helius=helius,
                       rec=(have or {}).get(f["pool"]), span_s=span_s) for f in fires]
    ok = [r for r in rows if r.get("status") == "ok"]
    repro = sum(1 for r in ok if r["reproduced"])
    return {"tier": tier, "vis": gate.get("vis", "full"), "n_fires": len(fires), "n_checked": len(ok),
            "n_reproduced": repro, "n_diverged": len(ok) - repro, "rows": rows}


def render(out: dict) -> str:
    L = [f"FRONT-RUN FEED VERIFY  (tier {out['tier']}: top {out['tier']}%,  gate vis '{out.get('vis', 'full')}'"
         f"{' = the box view' if out.get('vis') == 'direct' else ''})",
         f"  fires {out['n_fires']}  checked {out['n_checked']}  "
         f"reproduced {out['n_reproduced']}  DIVERGED {out['n_diverged']}", ""]
    L.append(f"  {'pool':<20}{'slot':>12}{'in?':>5}{'fires?':>7}{'sym':>5}{'m2':>5}{'eta':>8}"
             f"{'absmv@N':>9}{'fwd':>8}  kind  match")
    for r in out["rows"]:
        if r.get("status") != "ok":
            L.append(f"  {r['pool'][:20]:<20}{r['slot']:>12}  -- {r.get('status')}")
            continue
        o = r["offline"] or {}
        mm = r.get("match")
        ms = "" if mm is None else ("ok" if all(mm.values()) else "MISMATCH " + ",".join(k for k, v in mm.items() if not v))
        flag = "  <-- box fired but ground truth does NOT" if not r["reproduced"] else ""
        am = o.get("abs_move")
        fw = o.get("fwd_move")
        # abs_move at N near 0 = clean precursor (front-run); large = overlaps a landed move (backrun territory)
        kind = "?" if am is None else ("precursor" if am < 0.01 else "backrun?")
        L.append(f"  {r['pool'][:20]:<20}{r['slot']:>12}{('y' if r['in_flow'] else 'n'):>5}"
                 f"{('y' if o.get('fires') else 'n'):>7}{str(o.get('sym','')):>5}{str(o.get('m2','')):>5}"
                 f"{(o.get('eta') if o.get('eta') is not None else 0):>8.3f}"
                 f"{(am if am is not None else 0):>9.4f}{(fw if fw is not None else 0):>+8.3f}  {kind:<10}{ms}{flag}")
    if out["n_diverged"]:
        L += ["", "  DIVERGED fires = the box's live slot aggregation did not reproduce on chain ground truth.",
              "  Check, in order: the per-slot buyer wallet (fee payer = keys[0]); buy vs sell classification;",
              "  and whether the box emits a record for EVERY slot incl. dead ones (motifs / age depend on it)."]
    elif out["n_checked"]:
        L += ["", "  Every checked box fire reproduces on Helius ground truth -> the live feed reads slots correctly."]
    return "\n".join(L)
