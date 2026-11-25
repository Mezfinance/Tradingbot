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
            sl = r.get("slot", r.get("signal_slot"))     # the live box writes 'signal_slot'; verify uses 'slot'
            if r.get("pool") and sl is not None:
                box = {k: r[k] for k in ("sym", "m2", "m3", "eta") if k in r}
                out.append({"pool": r["pool"], "slot": int(sl), "mint": r.get("token_mint"), "box": box})
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


def _is_truncated(fired_slot: int, flow: dict, reached: bool) -> bool:
    """True when the fetch was page-capped (reached=False) BEFORE the fired slot, so the fired slot is missing
    from the offline flow for a fetch reason, not a feed/box bug -- raise --pages to include it. A fired slot
    at or below the last fetched slot but still not an active slot IS a real divergence (box fired on a dead
    slot). An injected rec (tests / a box-dumped series) is complete, so reached is True and this is never hit."""
    if reached:
        return False
    slots = flow.get("slots") or []
    last = slots[-1]["slot"] if slots else -1
    return fired_slot > last


def verify_one(fire: dict, gate: dict, tier: str = "1", flurry_min: int = 3, helius=None,
               rec: dict | None = None, span_s: float = 180.0, max_pages: int | None = None,
               box_exact: bool = False, hybrid: bool = False, ws_only: bool = False) -> dict:
    """Replay one fired pool through the study's ground-truth path and score the box's gate on every slot.
    `rec` (a launch_record dict) short-circuits the fetch -- used by tests and when the box dumped its own series.
    `box_exact` builds the ground truth with the box's LUT drop replicated (launch_record vis_box_exact), so a
    `direct` gate is verified against EXACTLY what the box can see -- a MISMATCH that survives box_exact is a
    deeper gap than LUT (shred loss); one that vanishes is the drop_lut completeness gap.
    `hybrid` builds the ground truth as flurry_study._hybrid_bank (the box's live architecture: the CURRENT slot
    scored from the DIRECT shred view, the prior slots rebased to the FULL Helius view) instead of a single-vis
    SlotBank -- so a hybrid-gate box fire is verified against the exact offline hybrid the gate was fit on."""
    pool, slot, box = fire["pool"], int(fire["slot"]), fire.get("box") or {}
    is_cap = isinstance(rec, dict) and ("confirmed" in rec or "shred" in rec)   # a box capture, not a launch_record
    ws = ws_only or is_cap                          # single complete feed, no rebase (the box's + study's own path)
    hybrid = hybrid and not ws                      # ws-only never rebases, whatever the gate stamped
    gate_vis = "ws-only" if ws else gate.get("vis", "full")   # replay through the view the gate was fit on
    res = {"pool": pool, "slot": slot, "box": box, "vis": gate_vis}
    reached = True                                  # an injected rec/capture is complete; only a real fetch truncates
    if rec is None:
        if helius is None:
            res["status"] = "no helius"
            return res
        mint = fire.get("mint") or _resolve_mint(helius, pool)
        if not mint:
            res["status"] = "mint unresolved"
            return res
        txs, reached, _err = helius.history_from_start(pool, span_s, max_pages or fl.MAX_PAGES)
        rec = he.launch_record(txs, pool, mint, window_s=span_s, with_vis=(hybrid or (not ws and gate_vis != "full")),
                               vis_box_exact=box_exact)
    saved_vq, saved_c, saved_da = fl.USE_VQ, fl.CAUSAL_WALLET, fl.DROP_ARTIFACTS
    try:
        fl.USE_VQ = False                           # the gate is price-free, so the basis cannot change a fire
        if ws:
            fl.CAUSAL_WALLET = True                 # causal (online) wallet ranks, as the gate was fit
            fl.DROP_ARTIFACTS = True                # score the capture EXACTLY as --frontrun fit it (reserve-collapse drop)
            flow = (fl._flow_from_series(rec.get("confirmed") or [], priced=True) if is_cap
                    else fl.slot_flow(rec, "full"))  # the single complete+priced feed the box serves -- no rebase
            recs = None
        elif hybrid:
            fl.CAUSAL_WALLET = True                 # the frontrun gate is fit on causal (online) wallet ranks
            flow = fl.slot_flow(rec, "direct")      # fires land on direct (shred) slots; prices feed the fwd move
            recs = fl._hybrid_bank(rec, 0.0, fl._tgt(flurry_min=flurry_min))  # direct current + full rebased prior
        else:
            flow = fl.slot_flow(rec, gate_vis)      # direct = only top-level pumpswap (what the box decodes)
            recs = None
    finally:
        fl.USE_VQ, fl.CAUSAL_WALLET, fl.DROP_ARTIFACTS = saved_vq, saved_c, saved_da
    if not flow or (hybrid and not recs):
        res["status"] = "no flow (pool had < 3 priced slots in history)"
        return res
    slots = flow["slots"]
    bank = None if hybrid else live.SlotBank(flurry_min)
    thr = (gate.get("threshold") or {}).get(str(tier))
    hold = int(gate.get("hold", 2))
    fired, at = [], None
    for i, s in enumerate(slots):
        r = recs[i] if hybrid else bank.push(s)     # hybrid recs align to the direct-flow slots (same slot_flow)
        e = pg.eta(r, gate)
        hit = thr is not None and e >= float(thr)
        if hit:
            fired.append(s["slot"])
        if s["slot"] == slot:
            # abs_move AT the signal slot = the backrun discriminator: ~0 => a genuine precursor fire (the move
            # is ahead, in N+1..N+hold); large => the fire overlaps a move already landed (backrun contamination).
            # In hybrid the current-slot abs_move is zeroed (shreds carry no price), so it is always ~0 by design.
            entry = s.get("price") or fl._price_at(slots, slot)
            exitp = fl._price_at(slots, slot + hold)
            fwd = (exitp / entry - 1.0) if (entry and exitp) else None
            at = {"sym": r["sym"], "m2": r["m2"], "m3": r["m3"], "eta": round(e, 4), "fires": hit,
                  "abs_move": round(float(r["f"].get("abs_move", 0.0)), 4),
                  "fwd_move": (round(fwd, 4) if fwd is not None else None)}
    if at is None and _is_truncated(slot, flow, reached):
        res["status"] = "truncated"
        res["note"] = "fetch capped at %d pages before the fired slot (raise --pages; not a box error)" % (max_pages or fl.MAX_PAGES)
        return res
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
           tier: str = "1", flurry_min: int = 3, span_s: float = 180.0, max_pages: int | None = None,
           box_exact: bool = False, hybrid: bool = False, ws_only: bool = False) -> dict:
    """Verify every fire. `have` (pool -> launch_record OR box-capture dict) injects ground truth without Helius.
    ws_only: score the capture's single complete+priced feed (confirmed series) through SlotBank with NO rebase --
    the box's and study's own path, so a train=serve gate reproduces by construction (frontrun-verify is then an
    integration check: did the live fires come from the captured feed, not a hybrid reconstruction)."""
    hybrid = (hybrid or bool(gate.get("hybrid"))) and not ws_only   # ws-only never rebases, whatever the gate stamped
    if "_terms" not in gate:
        pg.compile_gate(gate)
    rows = [verify_one(f, gate, tier, flurry_min, helius=helius,
                       rec=(have or {}).get(f["pool"]), span_s=span_s, max_pages=max_pages,
                       box_exact=box_exact, hybrid=hybrid, ws_only=ws_only) for f in fires]
    ok = [r for r in rows if r.get("status") == "ok"]
    repro = sum(1 for r in ok if r["reproduced"])
    trunc = sum(1 for r in rows if r.get("status") == "truncated")
    return {"tier": tier, "vis": gate.get("vis", "full"), "box_exact": box_exact, "hybrid": hybrid,
            "n_fires": len(fires), "n_checked": len(ok), "n_reproduced": repro, "n_diverged": len(ok) - repro,
            "n_truncated": trunc, "dist": _distance(ok), "rows": rows}


def _distance(ok: list[dict]) -> dict:
    """How far off the live box is from the offline ground truth, quantified. repro_rate is the headline: the
    share of box fires the offline hybrid also fires on. The misses split by CAUSE: a box fire on a slot the
    offline gate would NOT fire (present_nofire -> the features crossed threshold live but not offline: usually a
    rebased prior symbol differing, i.e. timing/decode) vs a box fire on a slot the offline flow does not even
    hold (absent -> the live aggregated a slot chain history does not show). Symbol-agreement rates and the mean
    |eta gap| say how close the scoring is even when it does fire."""
    n = len(ok)
    repro = sum(1 for r in ok if r["reproduced"])
    present_nofire = sum(1 for r in ok if r["in_flow"] and not r["reproduced"])
    absent = sum(1 for r in ok if not r["in_flow"])
    sym_ck = sym_ok = m2_ck = m2_ok = 0
    deta: list[float] = []
    for r in ok:
        mm = r.get("match") or {}
        if "sym" in mm:
            sym_ck += 1; sym_ok += bool(mm["sym"])
        if "m2" in mm:
            m2_ck += 1; m2_ok += bool(mm["m2"])
        o, b = r.get("offline") or {}, r.get("box") or {}
        if o.get("eta") is not None and "eta" in b:
            deta.append(abs(float(b["eta"]) - float(o["eta"])))
    return {"checked": n, "reproduced": repro, "repro_rate": (repro / n if n else None),
            "present_but_offline_nofire": present_nofire, "slot_absent_offline": absent,
            "sym_checked": sym_ck, "sym_match": sym_ok, "sym_rate": (sym_ok / sym_ck if sym_ck else None),
            "m2_checked": m2_ck, "m2_match": m2_ok, "m2_rate": (m2_ok / m2_ck if m2_ck else None),
            "mean_abs_eta_gap": (sum(deta) / len(deta) if deta else None)}


def render(out: dict) -> str:
    _mode = ("HYBRID: direct current slot + Helius-rebased prior" if out.get("hybrid")
             else "gate vis '%s'%s" % (out.get("vis", "full"), " = the box view" if out.get("vis") == "direct" else ""))
    L = [f"FRONT-RUN FEED VERIFY  (tier {out['tier']}: top {out['tier']}%,  {_mode}"
         f"{',  BOX-EXACT: ground truth drops LUT swaps too' if out.get('box_exact') else ''})",
         f"  fires {out['n_fires']}  checked {out['n_checked']}  "
         f"reproduced {out['n_reproduced']}  DIVERGED {out['n_diverged']}"
         + (f"  truncated {out['n_truncated']}" if out.get('n_truncated') else ""), ""]
    d = out.get("dist") or {}
    if d.get("checked"):
        pct = lambda x: "n/a" if x is None else f"{x * 100:.0f}%"
        L += [f"  HOW FAR OFF  (checked {d['checked']})",
              f"    reproduced          {d['reproduced']}/{d['checked']}  ({pct(d.get('repro_rate'))})"
              "   <- share of box fires the offline hybrid also fires on",
              f"    of the misses:      {d['present_but_offline_nofire']} fired where offline would NOT "
              "(features/timing: prior symbol differs)",
              f"                        {d['slot_absent_offline']} fired on a slot absent from offline "
              "(shred decode / feed)",
              f"    current sym agree   {d['sym_match']}/{d['sym_checked']}  ({pct(d.get('sym_rate'))})"
              "   <- shred decode fidelity",
              f"    prior sym (m2) agree {d['m2_match']}/{d['m2_checked']}  ({pct(d.get('m2_rate'))})"
              "  <- Helius rebase fidelity",
              f"    mean |eta gap|      {'n/a' if d.get('mean_abs_eta_gap') is None else format(d['mean_abs_eta_gap'], '.3f')}"
              "   <- scoring closeness when it does fire", ""]
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
    if out.get("n_truncated"):
        L += ["", f"  {out['n_truncated']} fires were FETCH-TRUNCATED (busy pool, history capped before the slot) --",
              "  not divergences; raise --pages to fetch them. They are excluded from reproduced/DIVERGED."]
    if out["n_diverged"]:
        if out.get("hybrid"):
            L += ["", "  DIVERGED under HYBRID = the box's live hybrid (shred current slot + Helius-rebased prior)",
                  "  did not reproduce the offline _hybrid_bank on the same pool history. A MISMATCH on the PRIOR",
                  "  symbol (m2/m3) points at the rebase landing late (the full slot arrived after the direct fire",
                  "  -> prev '.'; raise --shred-delay-s or buffer); a MISMATCH on the current sym points at the shred",
                  "  decode (per-slot buyer keys[0] / buy-sell classification). box fired but ground truth does NOT",
                  "  = the live rebase surfaced a precursor the offline hybrid did not."]
        elif out.get("box_exact"):
            L += ["", "  DIVERGED under BOX-EXACT = the gap SURVIVES replicating the box's LUT drop, so it is NOT",
                  "  drop_lut completeness -- it is deeper (shreds lost upstream, invisible to drop_parse). No",
                  "  LUT resolver on the current feed fixes this; the complete-tx decode (LocalShred) is the answer."]
        else:
            L += ["", "  DIVERGED fires = the box's live slot aggregation did not reproduce on chain ground truth.",
                  "  Re-run with --box-exact: if the mismatches VANISH, the cause is the box's drop_lut (LUT-pooled",
                  "  direct swaps it can't resolve) and a LUT resolver / LocalShred fixes it; if they SURVIVE, the",
                  "  gap is deeper than LUT. Otherwise check the per-slot buyer (keys[0]) and buy/sell classification."]
    elif out["n_checked"]:
        tail = " under BOX-EXACT -> the ONLY gap was the box's LUT drop (drop_lut); resolve LUTs and the box is correct." \
            if out.get("box_exact") else " -> the live feed reads slots correctly."
        L += ["", "  Every checked box fire reproduces on Helius ground truth" + tail]
    return "\n".join(L)
