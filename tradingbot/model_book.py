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

Exits for this book are its own (`exit_config`): +100% take profit, -50% stop, 24h, as in the research.
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

NAME = "model5"
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


def exit_config(cfg: Config) -> Config:
    """This book's exits, as in the research: one take-profit at +100%, a -50% stop, out after 24h."""
    return replace(cfg, stop_loss_pct=cfg.model_book_sl_pct, take_profit_pcts=(cfg.model_book_tp_pct,),
                   take_profit_fractions=(1.0,), max_hold_hours=cfg.model_book_hold_h)


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
        out = wa._outcome(paths.get(p["mint"], []), ts, entry or 0)
        if out is None and entry and any(age > AGE_MIN for _, age, _ in ss):
            out = wa.FLAT_OUTCOME
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
                                     "score": score, "price": wa._f(snap["price"]), "liq": liq, "ts": time.time()})
        except queue.Full:
            return None
        return score
