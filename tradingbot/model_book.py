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
BOOKS = (NAME, FAST)
AGE_MIN = 5
FEATURES = sorted(set(wa._static_features({"name": "x", "dex": "x", "created_ts": 0}))
                  | set(wa._snap_features({})) | {"log_price_vs_first"})
RELOAD_CHECK_S = 300


def model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / f"{NAME}.json"


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
    """Pool-age exit for model5_10m's hand-made pools: sell by created + `model_book_handmade_exit_age_min`."""
    age = cfg.model_book_handmade_exit_age_min
    if name != FAST or age <= 0 or not sig.get("created_ts") or not is_handmade(sig.get("dex") or "", sig["mint"]):
        return None
    return sig["created_ts"] + age * 60


def hold_min(cfg: Config, name: str = NAME) -> float:
    return cfg.model_book_fast_hold_min if name == FAST else cfg.model_book_hold_min


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
            "research_rug_rate_pct": 16.0 if name == FAST else 76.0,  # 10-28 backtests: 10-min hold vs 24h
            "losing_exit_causes": causes,
            "entry_audit_rugs": medians(rugs), "entry_audit_others": medians(rest)}


class ModelBook:
    """Scores 5-minute snapshots and queues the top ones for the engine's `model` thread to buy."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Model | None = None
        self._mtime = 0.0
        self._checked = 0.0
        self.signals: queue.Queue = queue.Queue(maxsize=500)
        self.reload(force=True)

    def reload(self, force: bool = False) -> None:
        if not force and time.time() - self._checked < RELOAD_CHECK_S:
            return
        self._checked = time.time()
        path = model_path(self.cfg)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            self.model = Model.from_json(json.loads(path.read_text()))
            self._mtime = mtime
            log.info("model5 loaded (%s rows, cut-off %.3f)", self.model.meta.get("rows"), self.model.threshold)

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
                                     "created_ts": pool.get("created_ts"), "dex": pool.get("dex") or ""})
        except queue.Full:
            return None
        return score
