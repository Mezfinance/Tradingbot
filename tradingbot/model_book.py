"""The 5-minute pump model and its paper book (`model5`).

The watcher's research (watch_analysis) found that a logistic model on what a pool looks like 5 minutes
after it appears (buyers, buy pressure, volume, liquidity, pool type, time of day, price vs first seen)
ranks pumps well, and that buying the top ~2% made money in a walk-forward test before execution costs
were measured for real. This module turns that into a paper strategy:

- `train()` builds the same snapshot-only rows the research used (no enrichment: live scoring has to be
  cheap and immediate), fits the model on the last `model_train_days` finished days, and picks the score
  cut-off that selects the top `model_book_top_pct` of recent pools. It runs once a day in the report's
  child process (`python -m tradingbot watch-model-train`) and writes `<db dir>/model/model5.json`.
- `ModelBook.consider()` is called by the watcher for every 5-minute snapshot. Pools above the cut-off are
  queued; the engine's `model` thread buys them through the quote-based simulator, so the book measures
  the real slippage and failed transactions the research could only assume (6% round trip).

Exits for this book are its own (`exit_config`): +100% take profit, -50% stop, out after `model_book_hold_min`
(10 minutes since 10-28: most rugs come after the 15-minute mark).
No Claude anywhere.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from tradingbot import watch_analysis as wa
from tradingbot.config import Config
from tradingbot.watch_features import load_snaps

log = logging.getLogger(__name__)

NAME = "model5"  # 24h hold
FAST = "model5_10m"  # 10-minute hold, same signals, own cash and ledger
CLIMB = "model5_climb"  # FAST's rules + out after 3 min unless the price climbed 5%
# Only pools the rug catcher calls `repeat operator`, no pool-age exit; the name carries the hold in minutes
OPS9, OPS10 = "model5_ops9", "model5_ops10"
OPS_HOLD_MIN = {OPS9: 9.0, OPS10: 10.0}
# Early books (10-31): one hand-made-pool model scored at first sight (~pool minute 2), sold by pool minute 5/6/7
EARLY5, EARLY6, EARLY7 = "early5", "early6", "early7"
EARLY_EXIT_MIN = {EARLY5: 5.0, EARLY6: 6.0, EARLY7: 7.0}
EARLY_BOOKS = tuple(EARLY_EXIT_MIN)
EARLY_MAX_SEEN_MIN = 4.0  # first sight later than this leaves no time to trade (and was left out of training)
EARLY_GAIN = 0.10  # label: the 5-minute snapshot price >= first-sight price x 1.10 after costs, pool intact
BOOKS = (NAME, FAST, CLIMB, OPS9, OPS10) + EARLY_BOOKS
SHORT = (FAST, CLIMB, OPS9, OPS10)  # the short-hold books (entry-age guard, rug catcher report)
AGE_EXIT = (FAST, CLIMB)  # books with the pool-age exit for hand-made pools
AGE_MIN = 5
FEATURES = sorted(set(wa._static_features({"name": "x", "dex": "x", "created_ts": 0}))
                  | set(wa._snap_features({})) | {"log_price_vs_first"})
RELOAD_CHECK_S = 300


def model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / f"{NAME}.json"


def early_model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / "early.json"


def features(pool: dict, snap: dict) -> dict:
    """The model's inputs for one pool at its 5-minute snapshot (`pool` needs dex, name, created_ts, first)."""
    out = wa._static_features(pool)
    out.update(wa._snap_features(snap))
    first = json.loads(pool["first"]) if pool.get("first") else {}
    entry, fp = wa._f(snap.get("price")), wa._f(first.get("price"))
    out["log_price_vs_first"] = math.log(entry / fp) if entry and fp else None
    return out


def is_handmade(dex: str, mint: str) -> bool:
    """A pumpswap pool of a mint pump.fun didn't create: made by hand, its creator holds the LP and can pull it.
    On the 10-26 backup 87% of these were pulled within 24h (27% by a pool age of 17 min)."""
    return dex == "pumpswap" and not mint.endswith("pump")


def exit_deadline(cfg: Config, name: str, sig: dict) -> float | None:
    """Pool-age exit: the early books sell every position by their pool minute; the 10-min and climb books sell
    hand-made pools by created + `model_book_handmade_exit_age_min`."""
    if name in EARLY_EXIT_MIN:
        return sig["created_ts"] + EARLY_EXIT_MIN[name] * 60 if sig.get("created_ts") else None
    age = cfg.model_book_handmade_exit_age_min
    if name not in AGE_EXIT or age <= 0 or not sig.get("created_ts") or not is_handmade(sig.get("dex") or "", sig["mint"]):
        return None
    return sig["created_ts"] + age * 60


def active_books(cfg: Config) -> list[str]:
    return ([NAME] if cfg.model_book_slow else []) + ([FAST] if cfg.model_book_fast else []) \
        + ([CLIMB] if cfg.model_book_climb else []) + ([OPS9, OPS10] if cfg.model_book_ops else []) \
        + (list(EARLY_BOOKS) if cfg.early_book else [])


def verdict_skip(name: str, sig: dict) -> str | None:
    """The ops books trade only pools whose creator launched a pool before (10-28 backtest with exact on-chain pull
    times: the 60 repeat-operator picks +8.1%/trade at a 9-min hold, +7.6% at 10, +3.6% at 12, vs +0.6% for all
    86 picks at 10)."""
    if name not in OPS_HOLD_MIN:
        return None
    verdict = (sig.get("rug_check") or {}).get("verdict", "unknown")
    return None if verdict == "repeat operator" else f"rug catcher: {verdict}"


def climb_check(cfg: Config, name: str, bought_ts: float, market_price: float) -> tuple[float, float] | None:
    """model5_climb only: (check time, price the position must have reached by then)."""
    if name != CLIMB or not market_price:
        return None
    return bought_ts + cfg.model_book_climb_after_min * 60, market_price * (1 + cfg.model_book_climb_min_gain_pct / 100)


def hold_min(cfg: Config, name: str = NAME) -> float:
    if name in OPS_HOLD_MIN:
        return OPS_HOLD_MIN[name]
    if name in EARLY_EXIT_MIN:  # from the buy; the pool-minute deadline normally sells first
        return EARLY_EXIT_MIN[name]
    return cfg.model_book_fast_hold_min if name in SHORT else cfg.model_book_hold_min


def exit_config(cfg: Config, name: str = NAME) -> Config:
    """A model book's exits: one take-profit at +100%, a -50% stop, out after its hold time."""
    return replace(cfg, stop_loss_pct=cfg.model_book_sl_pct, take_profit_pcts=(cfg.model_book_tp_pct,),
                   take_profit_fractions=(1.0,), max_hold_hours=hold_min(cfg, name) / 60)


class Model:
    def __init__(self, names, mu, sd, med, w, b, threshold, meta):
        self.names, self.mu, self.sd, self.med = names, np.asarray(mu), np.asarray(sd), med
        self.w, self.b, self.threshold, self.meta = np.asarray(w), b, threshold, meta

    def score(self, feat: dict) -> float:
        vals = []
        for n in self.names:  # a missing or non-finite input takes the training median, as in training
            v = feat.get(n)
            vals.append(float(v) if v is not None and math.isfinite(float(v)) else self.med[n])
        return float(((np.array(vals) - self.mu) / self.sd) @ self.w + self.b)

    def to_json(self) -> dict:
        return {"names": self.names, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "med": self.med,
                "w": self.w.tolist(), "b": self.b, "threshold": self.threshold, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        return cls(d["names"], d["mu"], d["sd"], d["med"], d["w"], d["b"], d["threshold"], d.get("meta", {}))


def training_rows(store, now: float | None = None, days: float = 4) -> list[dict]:
    """Snapshot-only rows at 5 minutes with their 24h labels, for pools finished in the last `days`."""
    now = now if now is not None else wa.data_now(store)
    cutoff = now - wa.DAY_S - AGE_MIN * 60 - 1800
    since = cutoff - days * wa.DAY_S
    snaps = load_snaps(store, since)
    paths = wa._future_paths(store, snaps, since)
    rows = []
    for p in store.query("SELECT pool, mint, dex, name, created_ts, first FROM watch_pools "
                         "WHERE created_ts BETWEEN ? AND ? ORDER BY created_ts", (since, cutoff)):
        ss = snaps.get(p["pool"], [])
        at = next(((ts, d) for ts, age, d in ss if age == AGE_MIN), None)
        if at is None:
            continue
        ts, d = at
        entry = wa._f(d.get("price"))
        out = wa.label(paths.get(p["mint"], []), ts, entry or 0, [(t, x) for t, _, x in ss], wa._f(d.get("liq")),
                       any(age > AGE_MIN for _, age, _ in ss))
        if out is None:
            continue
        row = {"pool": p["pool"], "created_ts": p["created_ts"]}
        row.update({k: v for k, v in features(dict(p), d).items() if v is not None})
        row.update(out)
        rows.append(row)
    return rows


def train(store, cfg: Config, now: float | None = None) -> Model | None:
    rows = training_rows(store, now, cfg.model_train_days)
    if len(rows) < wa.MIN_ROWS or not any(r["pump2x"] for r in rows):
        log.warning("model5: not enough rows to train (%d)", len(rows))
        return None
    names = [n for n in FEATURES if any(r.get(n) is not None for r in rows)]
    x, med = wa._matrix(rows, names)
    y = np.array([float(r["pump2x"]) for r in rows])
    mu, sd = x.mean(0), x.std(0) + 1e-9
    w, b = wa.logistic((x - mu) / sd, y)
    scores = ((x - mu) / sd) @ w + b
    recent = scores[[r["created_ts"] >= rows[-1]["created_ts"] - 2 * wa.DAY_S for r in rows]]
    threshold = float(np.quantile(recent, 1 - cfg.model_book_top_pct / 100))
    top = scores >= threshold
    meta = {"trained_ts": time.time(), "rows": len(rows), "pump_rate": float(y.mean()),
            "top_pump_rate_in_sample": float(y[top].mean()) if top.any() else None,
            "top_trade_ret_in_sample": float(np.mean([r["trade_ret"] for r, t in zip(rows, top) if t])) if top.any() else None,
            "days": cfg.model_train_days, "top_pct": cfg.model_book_top_pct}
    return Model(names, mu.tolist(), sd.tolist(), {n: float(med[n]) for n in names}, w.tolist(), float(b),
                 threshold, meta)


def train_and_save(store, cfg: Config) -> Path | None:
    model = train(store, cfg)
    if model is None:
        return None
    path = model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model.to_json()))
    os.replace(tmp, path)
    log.info("model5 trained on %d rows (pump rate %.1f%%), cut-off %.3f", model.meta["rows"],
             model.meta["pump_rate"] * 100, model.threshold)
    return path


# --- rug logging -------------------------------------------------------------------------------
# The research found ~11.5% of top-2% picks rug (the -50% stop gaps to <=0.26x) and that holder data
# fetched at 15 minutes added nothing. These record, for the book's own trades, the holder audit at the
# moment of buying and why each losing trade lost, so the live rug rate and its cost can be compared.
RUG_RETURN = -0.80  # a trade that lost >=80%: the stop did not protect it
LIQ_PULLED_SHARE = 0.25  # liquidity now below 25% of entry liquidity = pulled, not dumped into


def early_features(pool: dict, first: dict, seen_ts: float) -> dict:
    """The early model's inputs: what GeckoTerminal showed when the watcher first saw the pool."""
    out = wa._static_features(pool)
    out.update(wa._snap_features(first))
    out["seen_age_min"] = (seen_ts - pool["created_ts"]) / 60
    return out


def early_training_rows(store, now: float | None = None, days: float = 4) -> list[dict]:
    """Hand-made pumpswap pools first seen within EARLY_MAX_SEEN_MIN, labelled by their 5-minute snapshot
    (~pool minute 6): `up` = price >= first-sight x (1 + EARLY_GAIN) after the 6% round trip, pool not emptied."""
    now = now if now is not None else wa.data_now(store)
    rows = []
    for p in store.query(
            "SELECT w.pool, w.mint, w.dex, w.name, w.created_ts, w.seen_ts, w.first, s.data AS snap5 FROM watch_pools w "
            "JOIN watch_snaps s ON s.pool = w.pool AND s.age_min = 5 WHERE w.dex = 'pumpswap' AND w.created_ts >= ? "
            "AND w.first IS NOT NULL AND s.data IS NOT NULL AND s.data != '{}' ORDER BY w.created_ts",
            (now - days * wa.DAY_S,)):
        if not is_handmade(p["dex"], p["mint"]) or not p["created_ts"] or not p["seen_ts"]:
            continue
        if not 0 <= p["seen_ts"] - p["created_ts"] <= EARLY_MAX_SEEN_MIN * 60:
            continue
        first, s5 = json.loads(p["first"]), json.loads(p["snap5"])
        pf, lf, p5, l5 = (wa._f(first.get("price")), wa._f(first.get("liq")) or 0.0, wa._f(s5.get("price")),
                          wa._f(s5.get("liq")) or 0.0)
        if not pf or not p5:
            continue
        pulled = lf > 0 and l5 < wa.RUG_LIQ_SHARE * lf
        ret = -1.0 if pulled else min(p5 / pf - 1 - wa.ROUND_TRIP_COST, 3.0)  # cap: unit glitches print 500x
        row = {"pool": p["pool"], "created_ts": p["created_ts"], "up": float(ret >= EARLY_GAIN), "ret": ret}
        row.update({k: v for k, v in early_features(dict(p), first, p["seen_ts"]).items() if v is not None})
        rows.append(row)
    return rows


def train_early(store, cfg: Config, now: float | None = None) -> Model | None:
    rows = early_training_rows(store, now, cfg.model_train_days)
    if len(rows) < wa.MIN_ROWS or not any(r["up"] for r in rows):
        log.warning("early model: not enough rows to train (%d)", len(rows))
        return None
    names = sorted({k for r in rows for k in r} - {"pool", "created_ts", "up", "ret"})
    x, med = wa._matrix(rows, names)
    y = np.array([r["up"] for r in rows])
    mu, sd = x.mean(0), x.std(0) + 1e-9
    w, b = wa.logistic((x - mu) / sd, y)
    scores = ((x - mu) / sd) @ w + b
    recent = scores[[r["created_ts"] >= rows[-1]["created_ts"] - 2 * wa.DAY_S for r in rows]]
    threshold = float(np.quantile(recent, 1 - cfg.early_top_pct / 100))
    top = scores >= threshold
    meta = {"trained_ts": time.time(), "rows": len(rows), "pump_rate": float(y.mean()),
            "top_pump_rate_in_sample": float(y[top].mean()) if top.any() else None,
            "top_trade_ret_in_sample": float(np.mean([r["ret"] for r, t in zip(rows, top) if t])) if top.any() else None,
            "days": cfg.model_train_days, "top_pct": cfg.early_top_pct}
    return Model(names, mu.tolist(), sd.tolist(), {n: float(med[n]) for n in names}, w.tolist(), float(b),
                 threshold, meta)


def train_early_and_save(store, cfg: Config) -> Path | None:
    model = train_early(store, cfg)
    if model is None:
        return None
    path = early_model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model.to_json()))
    os.replace(tmp, path)
    log.info("early model trained on %d rows (up rate %.1f%%), cut-off %.3f", model.meta["rows"],
             model.meta["pump_rate"] * 100, model.threshold)
    return path


def early_status(store, cfg: Config, hours: float = 8, now: float | None = None, journal=None) -> dict:
    """What the early books' funnel looked like over the last `hours`: pools seen, hand-made ones, how soon after
    creation the watcher saw them, liquidity, and how their first-sight scores compare with the model's cut-off."""
    now = now or time.time()
    path = early_model_path(cfg)
    model = Model.from_json(json.loads(path.read_text())) if path.exists() else None
    pct = lambda xs, q: round(float(np.percentile(xs, q)), 3) if xs else None
    seen_all, handmade, ages, early, liq_ok, scores = 0, 0, [], 0, 0, []
    for p in store.query("SELECT pool, mint, dex, name, created_ts, seen_ts, first FROM watch_pools WHERE seen_ts >= ?",
                         (now - hours * 3600,)):
        seen_all += 1
        if not is_handmade(p["dex"] or "", p["mint"] or "") or not p["created_ts"] or not p["seen_ts"]:
            continue
        handmade += 1
        age = (p["seen_ts"] - p["created_ts"]) / 60
        ages.append(age)
        if not 0 <= age <= EARLY_MAX_SEEN_MIN or not p["first"]:
            continue
        early += 1
        first = json.loads(p["first"])
        if (wa._f(first.get("liq")) or 0.0) < cfg.model_book_min_liq_usd or not wa._f(first.get("price")):
            continue
        liq_ok += 1
        if model is not None:
            scores.append(model.score(early_features(dict(p), first, p["seen_ts"])))
    out = {"hours": hours, "pools_seen": seen_all, "hand_made": handmade,
           "hand_made_seen_age_min": {"p10": pct(ages, 10), "median": pct(ages, 50), "p90": pct(ages, 90)},
           f"hand_made_seen_within_{EARLY_MAX_SEEN_MIN:g}_min": early, "with_enough_liquidity": liq_ok,
           "model": None if model is None else {"trained_rows": model.meta.get("rows"), "cut_off": round(model.threshold, 3),
                                                "top_pct": model.meta.get("top_pct")}}
    if model is not None:
        out["scores"] = {"p50": pct(scores, 50), "p90": pct(scores, 90), "p95": pct(scores, 95), "max": pct(scores, 100),
                         "at_or_above_cut_off": sum(s >= model.threshold for s in scores),
                         f"cut_off_for_top_{cfg.early_top_pct:g}pct_of_these": pct(scores, 100 - cfg.early_top_pct)}
    if journal is not None:
        since = now - hours * 3600
        out["books"] = {b: {"buys": journal.query("SELECT COUNT(*) FROM events WHERE kind='model_buy' AND strategy=? "
                                                 "AND ts >= ?", (b, since))[0][0],
                            "skips": [tuple(r) for r in journal.query(
                                "SELECT detail, COUNT(*) FROM events WHERE kind='entry_skipped' AND strategy=? AND ts >= ? "
                                "GROUP BY detail", (b, since))]} for b in EARLY_BOOKS}
    return out


ROLLING_WINDOW_S = 24 * 3600  # the rolling cut-off ranks a pool against the eligible pools of the past day
ROLLING_MIN_SCORES = 50  # fewer eligible pools than this in the window: no rolling cut-off yet


def early_candidates(store, cfg: Config, model: Model, since: float, until: float) -> list[dict]:
    """Hand-made pools first seen in [since, until) within EARLY_MAX_SEEN_MIN, with enough liquidity, scored by
    `model`, oldest first; with their 5-minute snapshot outcome when it exists (as the model was trained)."""
    out = []
    for p in store.query(
            "SELECT w.pool, w.mint, w.dex, w.name, w.created_ts, w.seen_ts, w.first, s.data AS snap5 FROM watch_pools w "
            "LEFT JOIN watch_snaps s ON s.pool = w.pool AND s.age_min = 5 WHERE w.seen_ts >= ? AND w.seen_ts < ? "
            "AND w.dex = 'pumpswap' AND w.first IS NOT NULL ORDER BY w.seen_ts", (since, until)):
        if not is_handmade(p["dex"], p["mint"]) or not p["created_ts"]:
            continue
        if not 0 <= p["seen_ts"] - p["created_ts"] <= EARLY_MAX_SEEN_MIN * 60:
            continue
        first = json.loads(p["first"])
        pf, lf = wa._f(first.get("price")), wa._f(first.get("liq")) or 0.0
        if not pf or lf < cfg.model_book_min_liq_usd:
            continue
        ret = None
        s5 = json.loads(p["snap5"]) if p["snap5"] and p["snap5"] != "{}" else None
        if s5 and wa._f(s5.get("price")):
            pulled = (wa._f(s5.get("liq")) or 0.0) < wa.RUG_LIQ_SHARE * lf
            ret = -1.0 if pulled else min(wa._f(s5["price"]) / pf - 1 - wa.ROUND_TRIP_COST, 3.0)
        out.append({"pool": p["pool"], "name": (p["name"] or "").split(" / ")[0], "seen_ts": p["seen_ts"],
                    "score": model.score(early_features(dict(p), first, p["seen_ts"])), "ret": ret})
    return out


def rolling_cut_off(history: list[tuple[float, float]], ts: float, top_pct: float) -> float | None:
    """The (100 - top_pct) percentile of eligible scores seen in the ROLLING_WINDOW_S before `ts`."""
    window = [s for t, s in history if ts - ROLLING_WINDOW_S <= t < ts]
    return float(np.percentile(window, 100 - top_pct)) if len(window) >= ROLLING_MIN_SCORES else None


def early_backtest(store, cfg: Config, hours: float = 8, now: float | None = None) -> dict:
    """How the last `hours` of eligible pools would have done under the fixed cut-off (the model's, set at training)
    and a rolling one (top `early_top_pct` of the previous day's eligible pools), scored on each pool's 5-minute
    snapshot (~pool minute 6; emptied pool = -100%; 6% round trip)."""
    now = now or time.time()
    path = early_model_path(cfg)
    if not path.exists():
        return {"error": "no early model yet"}
    model = Model.from_json(json.loads(path.read_text()))
    cands = early_candidates(store, cfg, model, now - hours * 3600 - ROLLING_WINDOW_S, now)
    history = [(c["seen_ts"], c["score"]) for c in cands]
    window = [c for c in cands if c["seen_ts"] >= now - hours * 3600]

    def summary(picks):
        done = [c["ret"] for c in picks if c["ret"] is not None]
        a = np.array(done)
        return {"picks": len(picks), "with_outcome": len(done),
                "mean_pct": round(float(a.mean()) * 100, 1) if done else None,
                "median_pct": round(float(np.median(a)) * 100, 1) if done else None,
                "win_pct": round(float((a > 0).mean()) * 100) if done else None,
                "emptied_pct": round(float((a <= -0.99).mean()) * 100) if done else None,
                "trades": [[time.strftime("%H:%M", time.gmtime(c["seen_ts"])), c["name"][:14], round(c["score"], 2),
                            None if c["ret"] is None else round(c["ret"] * 100, 1)] for c in picks]}
    fixed = [c for c in window if c["score"] >= model.threshold]
    rolling = []
    for c in window:
        cut = rolling_cut_off(history, c["seen_ts"], cfg.early_top_pct)
        if cut is not None and c["score"] >= cut:
            rolling.append(c)
    everything = summary(window)
    everything.pop("trades")
    return {"hours": hours, "eligible_pools": len(window), "fixed_cut_off": round(model.threshold, 3),
            "all_eligible": everything, "fixed": summary(fixed), "rolling": summary(rolling),
            "note": "outcome = 5-minute snapshot (~pool minute 6) vs first-sight price, 6% round trip; emptied = -100%"}


def entry_audit(extras, mint: str) -> dict:
    """Jupiter's holder audit right now (free, one call)."""
    t = extras.jupiter_token(mint) if extras is not None else None
    if not t:
        return {}
    audit = t.get("audit") or {}
    return {"holders": t.get("holderCount"), "top_holders_pct": audit.get("topHoldersPercentage"),
            "dev_balance_pct": audit.get("devBalancePercentage"), "dev_mints": audit.get("devMints"),
            "dev_migrations": audit.get("devMigrations"), "organic_score": t.get("organicScore"),
            "liquidity": t.get("liquidity"), "is_sus": audit.get("isSus"),
            "mint_authority_off": audit.get("mintAuthorityDisabled"),
            "freeze_authority_off": audit.get("freezeAuthorityDisabled")}


def exit_cause(extras, mint: str, entry_liq: float | None, ret: float, blocked: bool) -> dict:
    """Why a losing trade lost: no_route (could not sell at all), liquidity_pulled, dumped (pool intact,
    price crashed past the stop) or stopped (an ordinary -50..-80% stop)."""
    now = entry_audit(extras, mint)
    liq = now.get("liquidity")
    if blocked:
        cause = "no_route"
    elif liq is not None and entry_liq and liq < LIQ_PULLED_SHARE * entry_liq:
        cause = "liquidity_pulled"
    elif ret <= RUG_RETURN:
        cause = "dumped"
    else:
        cause = "stopped"
    return {"cause": cause, "return_pct": round(ret * 100, 1), "entry_liq": entry_liq, "liq_now": liq,
            "holders_now": now.get("holders"), "top_holders_pct_now": now.get("top_holders_pct")}


def rug_summary(journal, stuck: set[int] | None = None, name: str = NAME) -> dict:
    """Live rug rate and causes for a model book's finished trades, and entry audits of rugs vs the rest."""
    stuck = stuck or set()
    audits = {r["token"]: json.loads(r["detail"]) for r in journal.query(
        "SELECT token, detail FROM events WHERE strategy=? AND kind='model_entry_audit' ORDER BY ts", (name,))}
    causes: dict[str, int] = {}
    for r in journal.query("SELECT detail FROM events WHERE strategy=? AND kind='model_exit_cause'", (name,)):
        c = json.loads(r["detail"]).get("cause", "?")
        causes[c] = causes.get(c, 0) + 1
    trades = []
    for p in journal.query("SELECT id, token, cost_usd, realized_usd, closed_at FROM positions WHERE strategy=?",
                           (name,)):
        if p["closed_at"] is None and p["id"] not in stuck:
            continue
        ret = (p["realized_usd"] or 0) / p["cost_usd"] - 1 if p["cost_usd"] else 0.0
        trades.append((ret, audits.get(p["token"])))
    rugs = [a for r, a in trades if r <= RUG_RETURN]
    rest = [a for r, a in trades if r > RUG_RETURN]

    def medians(rows):
        rows = [a for a in rows if a]
        out = {}
        for k in ("holders", "top_holders_pct", "dev_balance_pct", "dev_mints", "organic_score", "liquidity"):
            vals = sorted(float(a[k]) for a in rows if isinstance(a.get(k), (int, float)) and not isinstance(a[k], bool))
            out[k] = round(vals[len(vals) // 2], 2) if vals else None
        sus = [a["is_sus"] for a in rows if a.get("is_sus") is not None]
        return {"n_with_audit": len(rows), "median": out,
                "flagged_suspicious_pct": round(100 * sum(sus) / len(sus), 1) if sus else None}

    return {"finished_trades": len(trades), "rugs": len(rugs),
            "rug_rate_pct": round(100 * len(rugs) / len(trades), 1) if trades else None,
            "research_rug_rate_pct": {FAST: 16.0, CLIMB: 7.0, OPS9: 13.0, OPS10: 15.0, EARLY5: 10.0, EARLY6: 12.0, EARLY7: 14.0}.get(name, 76.0),  # 10-28 backtests
            "losing_exit_causes": causes,
            "entry_audit_rugs": medians(rugs), "entry_audit_others": medians(rest)}


def operator_verdict(store, pool: dict, helius=None) -> dict:
    """Rug catcher (blind test): who created this pumpswap pool and whether that wallet launched a pool before
    (10-28 backtest: pools of brand-new wallets were pulled during the hold 32% of the time, -26%/trade; repeat
    operators 17%, -2.5%). Only pools recorded before this one count."""
    if pool.get("dex") == "pumpswap" and (pool.get("mint") or "").endswith("pump"):
        return {"verdict": "graduation"}  # pump.fun's migration makes each of these pools from a fresh wallet
    if store is None:
        return {"verdict": "unknown"}
    if pool.get("dex") and pool["dex"] != "pumpswap":
        return {"verdict": "unknown", "reason": f"not a pumpswap pool ({pool['dex']})"}
    row = store.query("SELECT creator FROM watch_creators WHERE pool=?", (pool["pool"],))
    creator = row[0]["creator"] if row else None
    if not row and helius is not None:
        # The enricher reads creators between slow tasks, often after the 5-minute signal (10-29: every live
        # verdict came back unknown), so a signal reads its own pool's creator: one account read.
        try:
            found = helius.pool_creators([pool["pool"]])
        except Exception as exc:  # noqa: BLE001 (the verdict must never stop a signal)
            log.warning("rug catcher creator read failed: %s", exc)
            found = {}
        if pool["pool"] in found:
            creator = found[pool["pool"]]
            store.execute("INSERT OR REPLACE INTO watch_creators(pool, creator, ts) VALUES(?,?,?)",
                          (pool["pool"], creator, time.time()))
            row = [{"creator": creator}]
    if not creator:
        return {"verdict": "unknown", "reason": "not a pumpswap pool" if row else "creator not read yet"}
    prior = store.query(
        "SELECT COUNT(*) AS n FROM watch_creators c JOIN watch_pools p ON p.pool = c.pool "
        "WHERE c.creator=? AND c.pool != ? AND p.created_ts < ?",
        (creator, pool["pool"], (pool.get("created_ts") or time.time()) - 30))[0]["n"]
    return {"verdict": "repeat operator" if prior else "new wallet", "creator": creator, "prior_pools": prior}


def rug_catcher_summary(journal, books=SHORT, now: float | None = None) -> dict:
    """Outcome of the books' trades grouped by the verdict logged before each signal was traded.
    A position still open 40+ minutes after its buy counts as rugged (the short books hold 10-15 minutes)."""
    now = now or time.time()
    preds = journal.query("SELECT ts, token, detail FROM events WHERE kind='rug_prediction' ORDER BY ts")
    out = {}
    for name in books:
        groups: dict[str, dict] = {}
        for p in preds:
            d = json.loads(p["detail"])
            verdict = d.get("verdict", "unknown")
            if verdict == "unknown" and d.get("reason"):
                verdict = "unknown: " + ("not read" if d["reason"].startswith("creator") else "other dex")
            g = groups.setdefault(verdict, {"signals": 0, "trades": 0, "pnl_usd": 0.0, "rugged": 0})
            g["signals"] += 1
            pos = journal.query("SELECT realized_usd, cost_usd, closed_at, opened_at FROM positions WHERE strategy=? "
                                "AND token=? AND opened_at BETWEEN ? AND ?", (name, p["token"], p["ts"] - 60, p["ts"] + 600))
            for r in pos:
                if r["closed_at"] is None and now - r["opened_at"] < 2400:
                    continue  # still being held normally
                pnl = (r["realized_usd"] - r["cost_usd"]) if r["closed_at"] is not None else -r["cost_usd"]
                g["trades"] += 1
                g["pnl_usd"] += pnl
                g["rugged"] += pnl <= -0.8 * r["cost_usd"]
        for g in groups.values():
            g["pnl_usd"] = round(g["pnl_usd"], 2)
            g["per_trade_usd"] = round(g["pnl_usd"] / g["trades"], 2) if g["trades"] else None
            g["rug_rate_pct"] = round(100 * g["rugged"] / g["trades"], 1) if g["trades"] else None
        out[name] = groups
    return out


class ModelBook:
    """Scores 5-minute snapshots and queues the top ones for the engine's `model` thread to buy."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Model | None = None
        self._mtimes = {"model": 0.0, "early": 0.0}
        self._checked = 0.0
        self.signals: queue.Queue = queue.Queue(maxsize=500)
        self.store = None  # the watcher's store, for the rug catcher's creator lookup (set by the engine)
        self.helius = None  # reads a signal's pool creator when the enricher hasn't yet (set by the engine)
        self.early: Model | None = None
        self.reload(force=True)

    def reload(self, force: bool = False) -> None:
        if not force and time.time() - self._checked < RELOAD_CHECK_S:
            return
        self._checked = time.time()
        for path, attr, label in ((model_path(self.cfg), "model", "model5"),
                                  (early_model_path(self.cfg), "early", "early model")):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime != self._mtimes[attr]:
                m = Model.from_json(json.loads(path.read_text()))
                setattr(self, attr, m)
                self._mtimes[attr] = mtime
                log.info("%s loaded (%s rows, cut-off %.3f)", label, m.meta.get("rows"), m.threshold)

    def consider_first(self, pool: dict, first: dict) -> float | None:
        """Called by the watcher when it first sees a pool (`pool` has created_ts and seen_ts). Hand-made pools
        that score in the early model's top `early_top_pct` are queued for the early books."""
        if not self.cfg.early_book or not first or not is_handmade(pool.get("dex") or "", pool.get("mint") or ""):
            return None
        self.reload()
        if self.early is None or not pool.get("created_ts") or not pool.get("seen_ts"):
            return None
        if not 0 <= pool["seen_ts"] - pool["created_ts"] <= EARLY_MAX_SEEN_MIN * 60:
            return None
        score = self.early.score(early_features(pool, first, pool["seen_ts"]))
        liq = wa._f(first.get("liq")) or 0.0
        if score < self.early.threshold or liq < self.cfg.model_book_min_liq_usd or not wa._f(first.get("price")):
            return None
        try:
            self.signals.put_nowait({"pool": pool["pool"], "mint": pool["mint"], "name": pool.get("name") or "",
                                     "score": score, "price": wa._f(first["price"]), "liq": liq, "ts": time.time(),
                                     "created_ts": pool["created_ts"], "dex": pool.get("dex") or "", "kind": "early"})
        except queue.Full:
            return None
        return score

    def consider(self, pool: dict, age: int, snap: dict) -> float | None:
        """Called by the watcher for each new snapshot. Returns the score when the pool is queued."""
        if age != AGE_MIN or not snap:
            return None
        self.reload()
        if self.model is None:
            return None
        score = self.model.score(features(pool, snap))
        if score < self.model.threshold:
            return None
        liq = wa._f(snap.get("liq")) or 0.0
        if liq < self.cfg.model_book_min_liq_usd or not wa._f(snap.get("price")):
            return None
        try:
            self.signals.put_nowait({"pool": pool["pool"], "mint": pool["mint"], "name": pool.get("name") or "",
                                     "score": score, "price": wa._f(snap["price"]), "liq": liq, "ts": time.time(),
                                     "created_ts": pool.get("created_ts"), "dex": pool.get("dex") or "",
                                     "rug_check": operator_verdict(self.store, pool, self.helius)})
        except queue.Full:
            return None
        return score
