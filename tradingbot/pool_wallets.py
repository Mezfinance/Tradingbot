"""Wallet-pattern fingerprint of ruggers vs survivors (11-13, read only; Helius research key only with --fetch).

Owner's idea: can the *set and behaviour of the wallets* trading a pool -- repeated wallets, wash wallets that
both buy and sell, a fixed-size / metronomic bot, top-wallet concentration -- tell a rugger apart from a survivor,
ideally BEFORE our entry so it becomes a *selection* filter (canary-slots settled that selection, not exit speed,
is the lever)?

For every probed pool (`pool_tracker.feint_pools` + the scorer's `watch_launch_live` row for create_ts /
token_mint / pulled_ts / ret) it fetches the full transaction stream from pool creation
(`Helius.history_from_start`, parsed per-wallet by `helius.trades`: each record is [sec_after_creation, wallet,
side +1 buy / -1 sell, token size]; the pool's own vault owner is excluded), builds wallet-pattern features over
the first `window_s` of trading, and tests whether they separate RUGGERS (LP pulled by minute 6, or ret <=
-99.9%) from SURVIVORS.  Features are computed at `entry_s` (the ~66 s confirmation = what a live selection filter
would see) and over the full window (descriptive: does the fingerprint exist at all, even if only after entry).

Per feature: AUC rugger-vs-survivor (= P(a random rugger scores higher than a random survivor)) with a
pool-bootstrap 90% CI, top/bottom-feature-quintile rug rate.  Plus a time-split selection test: fit a combined
z-score (feature signs from the earlier half) and skip the riskiest k% on the later half, reading the rug rate
kept vs skipped and survivor retention.  Stream cached in watch.db `watch_pool_wallets`; read only.

CLI: `pool-wallets [--hours N] [--window-s 120] [--entry-s 66] [--fetch] [--max-credits 20000] [--rps 2] [--json]`.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter
from random import Random

from tradingbot import pool_tracker

MINUTE6_S = 360.0               # LP pulled at/before this pool age = a rugger
MIN_TRADES = 8                  # fewer trades than this in the window: too sparse to characterise
MAX_PAGES = 6                   # history_from_start pages (100 txs each) per pool
HIST_CREDITS_PER_PAGE = 10
DEFAULT_WINDOW_S = 120.0        # fetch + full-window features over the first N s of trading
DEFAULT_ENTRY_S = 66.0         # ~confirmation (pool_tracker.CONF_AFTER_S + a margin): the selection-usable cut
BOOT = 500                     # pool-bootstrap draws for the AUC CI
SELECT_K = (20, 30, 50)        # skip the riskiest k% in the selection test

SCHEMA = ("CREATE TABLE IF NOT EXISTS watch_pool_wallets(pool TEXT PRIMARY KEY, data TEXT, ts REAL)")

# Feature names in a stable order; the study drops any a pool is too sparse to compute.
FEATURE_NAMES = (
    "n_trades", "n_wallets", "trades_per_wallet", "wash_frac", "top1_share", "top3_share",
    "repeat_frac", "buy_frac", "size_repeat_frac", "cadence_cv", "creator_frac",
)
FEATURE_HELP = {
    "n_trades": "trades in the window", "n_wallets": "distinct trader wallets",
    "trades_per_wallet": "trades / wallet (bot churn)", "wash_frac": "wallets that both bought and sold",
    "top1_share": "busiest wallet's share of trades", "top3_share": "top-3 wallets' share of trades",
    "repeat_frac": "trades from wallets with >=2 trades", "buy_frac": "buys / all trades",
    "size_repeat_frac": "trades on a size shared by >=3 (fixed-size bot)",
    "cadence_cv": "inter-trade gap variability (low = metronomic)", "creator_frac": "trades by the creator wallet",
}


# --- fetch -----------------------------------------------------------------------------------------

def _parse_stream(txs, done, ok, pool, token_mint, create_ts) -> dict:
    """Parse a from-creation tx list into per-wallet trade records (shared by the batch study and the live gate).
    Each record is [sec_after_creation, wallet, +1 buy / -1 sell, abs token size]; the pool's own vault leg is dropped."""
    from tradingbot.data import helius as H
    recs: list = []
    creator = None
    if txs:
        creator = (H.pool_creation(txs[0], pool) or {}).get("creator")
        ct = create_ts or txs[0].get("blockTime") or 0
        for t in H.trades(txs, token_mint):
            bt, _slot, owner, delta = t[0], t[1], t[2], t[3]
            if owner == pool or not bt or not delta:   # skip the pool's own vault leg
                continue
            recs.append([round(bt - ct, 1), owner, 1 if delta > 0 else -1, abs(float(delta))])
    return {"recs": recs, "creator": creator, "complete": bool(done and ok)}


def fetch_streams(store, helius, metas: list[dict], window_s: float, max_credits: int,
                  progress=None) -> tuple[dict, int]:
    """Per pool, the first `window_s` of trading from creation -> per-wallet records, cached in
    `watch_pool_wallets`. With `helius` it fetches pools not yet cached complete, in a stable hash order so a
    credit cap leaves an even sample; without it, reads only the cache. Returns ({pool: data}, credits spent)."""
    store.execute(SCHEMA)
    have: dict = {}
    for r in store.query("SELECT pool, data FROM watch_pool_wallets", ()):
        try:
            have[r["pool"]] = json.loads(r["data"])
        except Exception:  # noqa: BLE001 (a malformed cache row)
            pass
    credits = 0
    order = sorted(metas, key=lambda m: hashlib.sha1(("pw:" + m["pool"]).encode()).hexdigest())
    for i, m in enumerate(order):
        pool = m["pool"]
        cur = have.get(pool)
        if cur and cur.get("complete"):
            continue
        if helius is None or credits >= max_credits:
            continue
        txs, done, ok = helius.history_from_start(pool, window_s, max_pages=MAX_PAGES)
        credits += max(1, math.ceil(len(txs) / 100)) * HIST_CREDITS_PER_PAGE
        data = _parse_stream(txs, done, ok, pool, m["token_mint"], m.get("create_ts"))
        store.execute("INSERT OR REPLACE INTO watch_pool_wallets(pool, data, ts) VALUES(?,?,?)",
                      (pool, json.dumps(data), time.time()))
        have[pool] = data
        if progress:
            progress(i + 1, len(order), credits)
    return have, credits


# --- features --------------------------------------------------------------------------------------

def _sig3(x: float) -> float:
    """Round a token size to 3 significant figures (so near-identical bot clips bucket together)."""
    if x <= 0:
        return 0.0
    d = math.floor(math.log10(x))
    return round(x, -(int(d) - 2))


def wallet_features(data: dict, upto: float) -> dict | None:
    """Wallet-pattern features over the trades with sec <= `upto`. None when too few trades to characterise."""
    recs = [r for r in data.get("recs", []) if r[0] <= upto]
    n = len(recs)
    if n < MIN_TRADES:
        return None
    wallets: dict = {}
    for _sec, w, side, tok in recs:
        e = wallets.setdefault(w, {"buys": 0, "sells": 0, "n": 0})
        e["n"] += 1
        e["buys" if side > 0 else "sells"] += 1
    nw = len(wallets)
    buys = sum(e["buys"] for e in wallets.values())
    counts = sorted((e["n"] for e in wallets.values()), reverse=True)
    wash = sum(1 for e in wallets.values() if e["buys"] > 0 and e["sells"] > 0)
    repeat_trades = sum(e["n"] for e in wallets.values() if e["n"] >= 2)
    sc = Counter(_sig3(r[3]) for r in recs)
    size_repeat = sum(c for c in sc.values() if c >= 3) / n
    secs = sorted(r[0] for r in recs)
    gaps = [b - a for a, b in zip(secs, secs[1:]) if b > a]
    cadence_cv = (statistics.pstdev(gaps) / (sum(gaps) / len(gaps))) if len(gaps) >= 2 and sum(gaps) > 0 else None
    creator = data.get("creator")
    creator_frac = (wallets.get(creator, {}).get("n", 0) / n) if creator else 0.0
    return {
        "n_trades": float(n), "n_wallets": float(nw), "trades_per_wallet": n / nw,
        "wash_frac": wash / nw, "top1_share": counts[0] / n, "top3_share": sum(counts[:3]) / n,
        "repeat_frac": repeat_trades / n, "buy_frac": buys / n, "size_repeat_frac": size_repeat,
        "cadence_cv": cadence_cv, "creator_frac": creator_frac,
    }


# --- live selection gate (wired into the two-stage entry) ------------------------------------------

# The tells `pool-wallets` found clearly away from AUC 0.50 and mechanistic: the creator trading its own pool (rug), and
# a thin, shallow book -- few wallets / trades, low churn, no fixed-size bot -- vs a deep organic crowd (survivor).
# sign = +1 when a HIGHER value is more rug, -1 when a LOWER value is more rug.
GATE_FEATURES = {"creator_frac": 1.0, "n_wallets": -1.0, "n_trades": -1.0,
                 "trades_per_wallet": -1.0, "size_repeat_frac": -1.0}
GATE_MIN_REF = 20   # fewer recent probed pools than this: too little to rank -> the gate fails open


def fetch_live(helius, pool, token_mint, create_ts, window_s: float = DEFAULT_ENTRY_S + 15.0) -> dict:
    """One `history_from_start` read of a pool's first `window_s` of trading -> per-wallet records (the live gate's
    input). Same parse as the batch study."""
    txs, done, ok = helius.history_from_start(pool, window_s, max_pages=MAX_PAGES)
    return _parse_stream(txs, done, ok, pool, token_mint, create_ts)


def reference_features(store, entry_s: float = DEFAULT_ENTRY_S, limit: int = 400) -> list[dict]:
    """Entry-cut wallet features of the most recent cached probed pools -- the rolling reference the live gate ranks
    against (self-calibrating, no retrain)."""
    store.execute(SCHEMA)
    out = []
    for r in store.query("SELECT data FROM watch_pool_wallets ORDER BY ts DESC LIMIT ?", (limit,)):
        try:
            fe = wallet_features(json.loads(r["data"]), entry_s)
        except Exception:  # noqa: BLE001 (a malformed cache row)
            fe = None
        if fe:
            out.append(fe)
    return out


def _gate_combined(fe: dict, stats: dict) -> float:
    return sum(sign * ((fe[name] - mu) / sd) for name, (mu, sd, sign) in stats.items())


def gate_score(fe: dict | None, reference: list[dict], features: dict = GATE_FEATURES) -> tuple:
    """(score, rank_pct, reason). rank_pct = fraction of recent probed pools this pool scores riskier than (high =
    riskier, toward rug). reason 'too_sparse' (no usable stream yet) or 'insufficient_history' (< GATE_MIN_REF reference
    pools) -> the caller fails open. The score z-scores each gate feature against the reference and signs it toward rug."""
    if not fe:
        return (None, None, "too_sparse")
    ref = [r for r in reference if all(r.get(n) is not None for n in features)]
    if len(ref) < GATE_MIN_REF:
        return (None, None, "insufficient_history")
    stats = {}
    for name, sign in features.items():
        vals = [r[name] for r in ref]
        stats[name] = (statistics.mean(vals), statistics.pstdev(vals) or 1.0, sign)
    score = _gate_combined(fe, stats)
    ref_scores = [_gate_combined(r, stats) for r in ref]
    rank = sum(1 for s in ref_scores if s < score) / len(ref_scores)
    return (score, rank, "ok")


# --- separation ------------------------------------------------------------------------------------

def _auc(pos: list[float], neg: list[float]) -> float:
    """P(a random rugger value > a random survivor value), ties 0.5. 0.5 = no separation; >0.5 = higher feature
    is more rug; <0.5 = lower is more rug."""
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def _auc_ci(rows: list[tuple], name: str, rng: Random) -> list[float]:
    """Pool-bootstrap 90% CI for the AUC (resample the pools, recompute)."""
    idx = list(range(len(rows)))
    aucs = []
    for _ in range(BOOT):
        pos, neg = [], []
        for _ in idx:
            m, f = rows[rng.randrange(len(rows))]
            v = f.get(name)
            if v is None:
                continue
            (pos if m["rugger"] else neg).append(v)
        if pos and neg:
            aucs.append(_auc(pos, neg))
    if not aucs:
        return [0.5, 0.5]
    aucs.sort()
    return [aucs[int(0.05 * len(aucs))], aucs[int(0.95 * len(aucs)) - 1]]


def _quintile_rug(vals: list[tuple]) -> tuple[float | None, float | None]:
    """Rug rate in the top vs bottom feature quintile (vals = [(value, rugger_bool)])."""
    vals = sorted(vals, key=lambda t: t[0])
    k = max(1, len(vals) // 5)
    bot = [r for _v, r in vals[:k]]
    top = [r for _v, r in vals[-k:]]
    return (sum(top) / len(top) if top else None, sum(bot) / len(bot) if bot else None)


def _feature_study(rows: list[tuple], seed: int = 7) -> dict:
    rng = Random(seed)
    out = []
    for name in FEATURE_NAMES:
        pos = [f[name] for (m, f) in rows if m["rugger"] and f.get(name) is not None]
        neg = [f[name] for (m, f) in rows if not m["rugger"] and f.get(name) is not None]
        if len(pos) < 3 or len(neg) < 3:
            continue
        top_q, bot_q = _quintile_rug([(f[name], m["rugger"]) for (m, f) in rows if f.get(name) is not None])
        out.append({
            "feature": name, "auc": _auc(pos, neg), "ci": _auc_ci(rows, name, rng),
            "rugger_mean": statistics.mean(pos), "surv_mean": statistics.mean(neg),
            "n_rug": len(pos), "n_surv": len(neg), "top_q_rug": top_q, "bot_q_rug": bot_q,
        })
    out.sort(key=lambda d: abs(d["auc"] - 0.5), reverse=True)
    return {"n_pools": len(rows), "n_rug": sum(1 for m, _f in rows if m["rugger"]),
            "n_surv": sum(1 for m, _f in rows if not m["rugger"]), "features": out,
            "selection": _selection(rows, out)}


def _selection(rows: list[tuple], feats: list[dict]) -> dict | None:
    """Time-split selection test: on the earlier half fit each feature's mean/std and its rug direction (sign of
    AUC-0.5), build a combined z-score, then on the later half skip the riskiest k% and read the rug rate kept vs
    skipped and survivor retention. Honest (out-of-sample within the probe set), but tiny n -> directional only."""
    use = [d for d in feats if abs(d["auc"] - 0.5) >= 0.05]
    if len(use) < 2 or len(rows) < 12:
        return None
    ordered = sorted(rows, key=lambda t: t[0].get("create_ts") or 0)
    half = len(ordered) // 2
    train, test = ordered[:half], ordered[half:]
    norm = {}
    for d in use:
        vals = [f[d["feature"]] for (m, f) in train if f.get(d["feature"]) is not None]
        if len(vals) < 3:
            continue
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) or 1.0
        norm[d["feature"]] = (mu, sd, 1.0 if d["auc"] >= 0.5 else -1.0)
    if len(norm) < 2:
        return None

    def score(f):
        return sum(sign * ((f[name] - mu) / sd) for name, (mu, sd, sign) in norm.items() if f.get(name) is not None)

    scored = sorted(((score(f), m["rugger"]) for (m, f) in test), key=lambda t: t[0], reverse=True)  # riskiest first
    n = len(scored)
    base = sum(r for _s, r in scored) / n if n else None
    out = []
    for k in SELECT_K:
        cut = max(1, round(n * k / 100))
        skipped = scored[:cut]
        kept = scored[cut:]
        out.append({
            "skip_pct": k, "kept_n": len(kept),
            "kept_rug": (sum(r for _s, r in kept) / len(kept)) if kept else None,
            "skipped_rug": (sum(r for _s, r in skipped) / len(skipped)) if skipped else None,
            "surv_kept_pct": (sum(1 for _s, r in kept if not r) / max(1, sum(1 for _s, r in scored if not r))),
        })
    return {"test_n": n, "base_rug": base, "by_skip": out}


# --- study -----------------------------------------------------------------------------------------

def study(journal, store, cfg=None, helius=None, hours: float = 720.0, window_s: float = DEFAULT_WINDOW_S,
          entry_s: float = DEFAULT_ENTRY_S, max_credits: int = 20_000, progress=None, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    since = now - hours * 3600
    feints = pool_tracker.feint_pools(journal.query, since)
    scorer = pool_tracker.scorer_rows(store.query, since)
    metas = []
    for pool in feints:
        s = scorer.get(pool) or {}
        create, mint = s.get("create_ts"), s.get("token_mint")
        if not create or not mint:
            continue
        ret, pulled, status = s.get("ret"), s.get("pulled_ts"), s.get("status")
        if ret is None and pulled is None and status not in ("done", "thin_done"):
            continue  # no settled outcome yet
        rugger = (pulled is not None and pulled <= create + MINUTE6_S) or (ret is not None and ret <= -0.999)
        metas.append({"pool": pool, "token_mint": mint, "create_ts": create, "rugger": bool(rugger), "ret": ret})
    have, credits = fetch_streams(store, helius, metas, window_s, max_credits, progress)
    rows_entry, rows_full = [], []
    for m in metas:
        data = have.get(m["pool"])
        if not data or not data.get("recs"):
            continue
        fe = wallet_features(data, entry_s)
        ff = wallet_features(data, window_s)
        if fe:
            rows_entry.append((m, fe))
        if ff:
            rows_full.append((m, ff))
    return {
        "hours": hours, "window_s": window_s, "entry_s": entry_s, "credits": credits,
        "n_pools": len(metas), "n_ruggers": sum(1 for m in metas if m["rugger"]),
        "n_with_stream": sum(1 for m in metas if (have.get(m["pool"]) or {}).get("recs")),
        "entry": _feature_study(rows_entry), "full": _feature_study(rows_full),
    }


# --- render ----------------------------------------------------------------------------------------

def _f(v, dp=2):
    return "—" if v is None else f"{v:.{dp}f}"


def _pct(v, dp=0):
    return "—" if v is None else f"{v * 100:.{dp}f}%"


def _render_features(fs: dict, title: str) -> list[str]:
    o = [f"\n{title} ({fs['n_pools']} pools: {fs['n_rug']} ruggers, {fs['n_surv']} survivors)"]
    o.append(f"  (AUC = P(rugger's value > survivor's); 0.50 = no tell, away from 0.50 either way = a tell)")
    o.append(f"  {'feature':<18} {'AUC':>5} {'90% CI':>13} {'rug mean':>9} {'surv mean':>9} "
             f"{'topQ rug':>8} {'botQ rug':>8}")
    for d in fs["features"]:
        o.append(f"  {d['feature']:<18} {d['auc']:>5.2f} {('[' + _f(d['ci'][0]) + ',' + _f(d['ci'][1]) + ']'):>13} "
                 f"{_f(d['rugger_mean']):>9} {_f(d['surv_mean']):>9} {_pct(d['top_q_rug']):>8} {_pct(d['bot_q_rug']):>8}")
    sel = fs.get("selection")
    if sel:
        o.append(f"  selection (time-split, combined z of the tells; test n={sel['test_n']}, "
                 f"base rug {_pct(sel['base_rug'])}):")
        o.append(f"    {'skip riskiest':>13} {'kept n':>7} {'kept rug':>9} {'skipped rug':>12} {'survivors kept':>15}")
        for r in sel["by_skip"]:
            o.append(f"    {str(r['skip_pct']) + '%':>13} {r['kept_n']:>7} {_pct(r['kept_rug']):>9} "
                     f"{_pct(r['skipped_rug']):>12} {_pct(r['surv_kept_pct']):>15}")
    return o


def render(r: dict) -> str:
    o = [f"POOL WALLETS — can the wallet pattern tell a rugger from a survivor? (last {r['hours']:.0f} h)",
         f"{r['n_pools']} probed pools with a settled outcome ({r['n_ruggers']} ruggers); "
         f"{r['n_with_stream']} have a fetched tx stream; {r['credits']} research credits",
         f"window {r['window_s']:.0f}s, entry cut {r['entry_s']:.0f}s",
         "",
         "feature key: " + "; ".join(f"{k}={v}" for k, v in FEATURE_HELP.items())]
    o += _render_features(r["entry"], f"AT ENTRY (<= {r['entry_s']:.0f}s — a usable SELECTION filter)")
    o += _render_features(r["full"], f"FULL WINDOW (<= {r['window_s']:.0f}s — descriptive: does the fingerprint exist at all)")
    return "\n".join(o)
