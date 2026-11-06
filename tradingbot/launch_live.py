"""Second-4 launch scorer. It records every hand-made pool's prediction and pool-level outcome, and sends its
top-2% / top-5% picks to the launch4_2 / launch4_5 paper books (engine model-book path: Jupiter quotes, simulated
landing latency and reverts, sold at pool minute 6). Jupiter routes these pools ~0.1-2.4 s after creation
(measured 11-02 on 6 live pools, mostly via "Pump.fun Amm").

The timing backtest (`launch_analysis.timing`) found that a model on a hand-made pumpswap pool's first 4 seconds
of on-chain trading picks winners (top 2% / 5%, filled at seconds 4-7, out at pool minute ~6). This runs that
rule live, without money, to test it out of sample and to measure what a live entry would face:

1. A websocket on 82-byte Token-2022 accounts (pumpswap LP mints; their authority is the pool) sees every new
   pool within ~1 s of creation for ~120 Helius credits a day (pool accounts change on every swap; LP mints
   only on create / deposit / withdraw, so a pull shows as supply 0).
2. The pool account is read (1 credit): hand-made (no coin creator), SOL pair, token mint not ending in `pump`
   (the backtest's universe).
3. `launch_live_decide_delay_s` after the notification, Helius history (10 credits per 100 txs) is parsed by the
   backtest's own `launch_record` + `launch_features`, so the features match the training rows exactly. The
   token's name is unknown this early, so the model is trained without name features (`LIVE_DROP`).
4. Liquid pools (2 x pool SOL x SOL price >= `model_book_min_liq_usd`, not pulled) are scored by `launch4.json`;
   top 2% / top 5% use the training scores' quantiles. The prediction is stored before any outcome is known.
5. The pool's vaults are read at second FILL_S (would-be fill) and at EXIT_S (pool minute 6): return =
   exit / fill - 1 - 6%, or -100% if the pool's SOL fell below 10% of its peak (or the LP supply went to 0).

Rows live in watch.db table `watch_launch_live`; CLI `launch-status`."""
from __future__ import annotations

import asyncio
import collections
import hashlib
import heapq
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tradingbot import launch_analysis as la
from tradingbot import watch_analysis as wa
from tradingbot.config import Config
from tradingbot.data import pumpswap as ps
from tradingbot.data.helius import Helius, launch_record

log = logging.getLogger(__name__)

DECIDE_S = 4  # features from the chain up to this second after creation
FILL_S = 6  # the would-be fill is read at this second (the backtest's edge held for fills at 4-7)
EXIT_S = 360  # pool minute 6
TOO_LATE_S = 20  # a decision later than this after creation is not an entry at second 5
HISTORY_PAGES = 2
EMPTY_RETRIES = 2  # Helius may not have indexed a seconds-old pool yet: ask again this many times
EMPTY_RETRY_S = 1.5
MIN_TRAIN = 200
MODEL_RELOAD_S = 300
SEEN_MAX = 200_000
STATE_SAVE_S = 60
WS_HELIUS = "wss://mainnet.helius-rpc.com/?api-key={key}"
WS_PUBLIC = "wss://api.mainnet-beta.solana.com"
TOP2, TOP5 = 0.02, 0.05
# History-trained model (11-03): 7-day walk-forward on every hand-made pool, no hour/weekend features, top 5% of
# each day: +8.8%/trade (90% CI +2.8..+14.7), 23% rugged vs -5.2% / 31% for every pool; with them +4.6%.
CALENDAR = ("hour_sin", "hour_cos", "weekend")
HIST_DROP = tuple(la.LIVE_DROP) + CALENDAR
ROLLING_S = 86400  # its cut-offs: the top 2% / 5% of the past day's scores of liquid pools
ROLLING_MIN = 50  # fewer scores than this: fall back to the training scores' quantiles

SCHEMA = """CREATE TABLE IF NOT EXISTS watch_launch_live(
    pool TEXT PRIMARY KEY, lp_mint TEXT, token_mint TEXT, creator TEXT, notified_ts REAL, create_ts REAL,
    decided_ts REAL, indexed_s REAL, liquid INTEGER, score REAL, top2 INTEGER, top5 INTEGER, feat TEXT,
    sol_at_decide REAL, peak_sol REAL, sol_vault TEXT, token_vault TEXT, fill_ts REAL, fill_px REAL,
    fill_sol REAL, exit_ts REAL, exit_px REAL, exit_sol REAL, pulled_ts REAL, ret REAL, status TEXT,
    reason TEXT, credits INTEGER DEFAULT 0, liq_usd REAL)"""
# Backtest edge per trade after its 6% cost (decide 4 / fill 6, no names, 11-02), for the size estimate in status.
BACKTEST_EDGE = {"top 2%": 0.21, "top 5%": 0.06}  # on 1-minute trade candles (snapshot prices said 0.33 / 0.27)


def ensure_schema(store) -> None:
    import sqlite3
    store.execute(SCHEMA)
    for col in ("liq_usd REAL",  # added 11-02: pool liquidity at second 4 in USD
                "score_h REAL", "top2h INTEGER", "top5h INTEGER",  # 11-03: the history-trained model's verdict
                # 11-04: history model's bottom 20% (rolling 24 h), early-pull risk and its rank among those pools
                "low_h INTEGER", "risk REAL", "risk_pct REAL",
                # 11-04: rug-vs-win score and its rank among the past day's liquid / bottom-20% pools
                "rw REAL", "rw_pct REAL", "rw_low_pct REAL",
                # 11-05: the history model's bottom 20% of the past 6 h (the rug-vs-win books' D1-D2, as the study)
                "rw_low6 INTEGER",
                # 11-05: the earlier live rule's score and ranks, kept for the rw24 twin books
                "rw24 REAL", "rw24_pct REAL", "rw24_low_pct REAL"):
        try:
            store.execute(f"ALTER TABLE watch_launch_live ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass


def model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / "launch4.json"


def hist_model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / "launch4h.json"


def pull_model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / "pull.json"


def rw_model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / "rw.json"


def rw24_model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / "rw24.json"


# --- training -------------------------------------------------------------------------------------

def training_rows(store, cfg: Config, now: float | None = None) -> list[dict]:
    now = now if now is not None else time.time()
    data = la.rows(store, cfg, now - cfg.launch_train_days * wa.DAY_S)
    return [r for r in data if r.get("sample") and DECIDE_S in r.get("feat", {}) and FILL_S in r.get("ret", {})
            and r["liquid"].get(DECIDE_S)]


def train(store, cfg: Config, now: float | None = None):
    """Logistic on second-4 features (no name), label: the second-6 fill made money; cut-offs = the training
    scores' top 2% / 5% (as in the timing backtest)."""
    from tradingbot.model_book import Model
    rs = training_rows(store, cfg, now)
    if len(rs) < MIN_TRAIN:
        log.warning("launch4 model: not enough rows to train (%d)", len(rs))
        return None
    names = sorted({k for r in rs for k in r["feat"][DECIDE_S]} - set(la.LIVE_DROP))
    x, med = wa._matrix([r["feat"][DECIDE_S] for r in rs], names)
    rets = np.array([r["ret"][FILL_S] for r in rs])
    y = (rets > 0).astype(float)
    if y.min() == y.max():
        return None
    mu, sd = x.mean(0), x.std(0) + 1e-9
    w, b = wa.logistic((x - mu) / sd, y)
    scores = ((x - mu) / sd) @ w + b
    cut2, cut5 = float(np.quantile(scores, 1 - TOP2)), float(np.quantile(scores, 1 - TOP5))
    top = lambda c: scores >= c - 1e-9
    meta = {"trained_ts": time.time(), "rows": len(rs), "win_rate": float(y.mean()),
            "score_quantiles": {str(q): float(np.quantile(scores, q / 100)) for q in (10, 50, 90, 95, 98)},
            "mean_ret_all": float(rets.mean()), "cut2": cut2, "cut5": cut5,
            "top2_in_sample": {"n": int(top(cut2).sum()), "mean_ret": float(rets[top(cut2)].mean())},
            "top5_in_sample": {"n": int(top(cut5).sum()), "mean_ret": float(rets[top(cut5)].mean())},
            "decide_s": DECIDE_S, "fill_s": FILL_S, "days": cfg.launch_train_days}
    return Model(names, mu.tolist(), sd.tolist(), {n: float(med[n]) for n in names}, w.tolist(), float(b),
                 cut5, meta)


def hist_training_rows(store, cfg: Config, now: float | None = None) -> list[dict]:
    """Every liquid hand-made pool with a known minute-6 outcome: the chain-only history (`launch_hist`, all pools,
    listed or not) plus the scorer's own finished rows (vault prices), the newer of the two kept per pool."""
    from tradingbot import launch_hist
    now = now if now is not None else time.time()
    since = now - cfg.launch_train_days * wa.DAY_S
    rows = {}
    try:
        hist, _ = launch_hist.rows(store, cfg)
    except Exception:  # no history table yet
        hist = []
    for r in hist:
        if r["liquid"] and r["created_ts"] >= since and DECIDE_S in r["feat"] and FILL_S in r["ret"]:
            rows[r["pool"]] = {"created_ts": r["created_ts"], "feat": r["feat"][DECIDE_S], "ret": r["ret"][FILL_S]}
    ensure_schema(store)
    for r in store.query("SELECT pool, create_ts, feat, ret FROM watch_launch_live WHERE status='done' AND liquid=1 "
                         "AND feat IS NOT NULL AND ret IS NOT NULL AND create_ts >= ?", (since,)):
        rows.setdefault(r["pool"], {"created_ts": r["create_ts"], "feat": json.loads(r["feat"]), "ret": r["ret"]})
    return sorted(rows.values(), key=lambda r: r["created_ts"])


def train_hist(store, cfg: Config, now: float | None = None):
    """Logistic on second-4 features without names or hour/weekend, standardised values capped at
    `la.CLIP_Z` (as in the walk-forward), label: the second-6 fill made money."""
    from tradingbot.model_book import Model
    rs = hist_training_rows(store, cfg, now)
    if len(rs) < MIN_TRAIN:
        log.warning("launch4h model: not enough rows to train (%d)", len(rs))
        return None
    names = sorted({k for r in rs for k in r["feat"]} - set(HIST_DROP))
    x, med = wa._matrix([r["feat"] for r in rs], names)
    rets = np.array([r["ret"] for r in rs])
    y = (rets > 0).astype(float)
    if y.min() == y.max():
        return None
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.clip((x - mu) / sd, -la.CLIP_Z, la.CLIP_Z)
    w, b = wa.logistic(z, y)
    scores = z @ w + b
    cut2, cut5 = float(np.quantile(scores, 1 - TOP2)), float(np.quantile(scores, 1 - TOP5))
    top = lambda c: scores >= c - 1e-9
    meta = {"trained_ts": time.time(), "rows": len(rs), "win_rate": float(y.mean()), "clip_z": la.CLIP_Z,
            "score_quantiles": {str(q): float(np.quantile(scores, q / 100)) for q in (10, 20, 50, 90, 95, 98)},
            "mean_ret_all": float(rets.mean()), "cut2": cut2, "cut5": cut5,
            "top2_in_sample": {"n": int(top(cut2).sum()), "mean_ret": float(rets[top(cut2)].mean())},
            "top5_in_sample": {"n": int(top(cut5).sum()), "mean_ret": float(rets[top(cut5)].mean())},
            "decide_s": DECIDE_S, "fill_s": FILL_S, "days": cfg.launch_train_days, "dropped": list(CALENDAR)}
    return Model(names, mu.tolist(), sd.tolist(), {n: float(med[n]) for n in names}, w.tolist(), float(b),
                 cut5, meta)


def _save(model, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model.to_json()))
    os.replace(tmp, path)
    return path


def train_hist_and_save(store, cfg: Config) -> Path | None:
    model = train_hist(store, cfg)
    if model is None:
        return None
    log.info("launch4h model trained on %d rows", model.meta["rows"])
    return _save(model, hist_model_path(cfg))


LOW_SHARE = 0.2  # the history model's bottom 20% of the past day's scores = its deciles D1-D2
RISK_LEVELS = tuple(range(0, 101, 5))  # percentiles of training risk kept in the pull model's meta


def low_flags(low_h, risk_pct) -> dict:
    """Signal flags of the low-decile books: `d12` = the history model's bottom 20% (D1-D2); `d12s<k>` = also not
    in the riskiest k% by early-pull risk (needs a pull model)."""
    out = {"d12": bool(low_h)}
    for k in PULL_SKIPS:
        out[f"d12s{k}"] = bool(low_h) and risk_pct is not None and risk_pct < 100 - k
    return out


def train_pull(store, cfg: Config, now: float | None = None):
    """Early-pull model (11-04, `launch-early-pulls`): logistic on the second-4 features + creator record, label
    LP pulled < EARLY_RUG_S, trained on the scorer's finished liquid pools of the last `launch_train_days`. Meta
    keeps the risk percentiles of the training pools in the history model's bottom 20% (fallback ranking until
    the live scorer has seen ROLLING_MIN of them)."""
    from tradingbot.model_book import Model
    ensure_schema(store)
    now = now if now is not None else time.time()
    rows = _pull_rows(store, now - cfg.launch_train_days * 86400)
    if len(rows) < PULL_MIN_TRAIN:
        log.warning("pull model: not enough rows to train (%d)", len(rows))
        return None
    y = np.array([r["y"] for r in rows])
    if y.min() == y.max():
        return None
    names = sorted({k for r in rows for k in r["x"]})
    x, med = wa._matrix([r["x"] for r in rows], names)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.clip((x - mu) / sd, -la.CLIP_Z, la.CLIP_Z)
    w, b = wa.logistic(z, y)
    risk = z @ w + b
    sh = [r["score_h"] for r in rows if r["score_h"] is not None]
    cut = float(np.quantile(sh, LOW_SHARE)) if len(sh) >= ROLLING_MIN else None
    low = np.array([cut is not None and r["score_h"] is not None and r["score_h"] <= cut for r in rows])
    base = risk[low] if low.sum() >= ROLLING_MIN else risk
    meta = {"trained_ts": time.time(), "rows": len(rows), "early_pull_rate": float(y.mean()),
            "low_rows": int(low.sum()), "auc_in_sample": wa.auc(risk, y), "clip_z": la.CLIP_Z,
            "risk_quantiles": [float(np.quantile(base, q / 100)) for q in RISK_LEVELS],
            "early_rug_s": EARLY_RUG_S, "days": cfg.launch_train_days}
    return Model(names, mu.tolist(), sd.tolist(), {n: float(med[n]) for n in names}, w.tolist(), float(b), 0.0, meta)


# Rug-vs-win filter (11-04, `launch-filter-study` option 4): P(rugged before the minute-6 exit) - P(+50% or more).
# Walk-forward on 2,041 pools (8 blocks of 6 h): skipping the riskiest 50% kept +7.6%/trade over every liquid pool
# (keep all -1.6%, every random skip worse) and +13.1% over the history model's bottom 20% (keep all +4.1%, 97%).
RW_SKIP = 50  # % of pools skipped as riskiest (rug odds over win odds)
RW_BIG = 0.5
# 11-05: the live books now follow the study's rule as closely as a live system can (user: the 24 h ranking and
# nightly 14-day training were changes from the backtest). The study retrained before every 6 h block on every
# earlier finished pool and skipped the riskiest half within each block (D1-D2 also ranked within the block);
# live can't see the rest of a block, so each pool is ranked against the past RW_BLOCK_S instead, and the model
# is retrained every RW_BLOCK_S on all finished pools (`LaunchLive._rw_retrain_loop`).
RW_BLOCK_S = 6 * 3600
RW_RETRAIN_TIMEOUT_S = 900
RW_RETRY_S = 600  # check the model's age (and retry a failed retrain) this often
# The earlier live rule (11-04 to 11-05), kept as twin books at the user's request so a good spell under either
# rule isn't missed: trained once a day on the last `launch_train_days`, ranked against the past 24 h, bottom 20%
# = `low_h` (the history model's past 24 h).
RW24_WINDOW_S = 86400
RW24_RETRAIN_S = 86400


def rw_low(r) -> bool:
    """The rug-vs-win books' bottom-20% flag: the 6 h one (`rw_low6`), or `low_h` on rows scored before 11-05."""
    v = r["rw_low6"] if "rw_low6" in r.keys() else None
    return bool(r["low_h"] if v is None else v)


def rw24_flags(low_h, rw24_pct, rw24_low_pct) -> dict:
    """Signal flags of the earlier-rule twin books: `rw50_24` / `d12rw50_24` (see RW24_WINDOW_S)."""
    keep = 100 - RW_SKIP
    return {"rw50_24": rw24_pct is not None and rw24_pct < keep,
            "d12rw50_24": bool(low_h) and rw24_low_pct is not None and rw24_low_pct < keep}


def rw_flags(low, rw_pct, rw_low_pct) -> dict:
    """Signal flags of the rug-vs-win books: `rw50` = in the less risky half of the past RW_BLOCK_S's liquid
    pools; `d12rw50` = in the history model's bottom 20% of the past RW_BLOCK_S (`rw_low6`) and the less risky
    half of those."""
    keep = 100 - RW_SKIP
    return {"rw50": rw_pct is not None and rw_pct < keep,
            "d12rw50": bool(low) and rw_low_pct is not None and rw_low_pct < keep}


class RugWinModel:
    """Two logistics on the early-pull inputs (second-4 features + creator record): rugged before the exit, and
    +50% or more. Score = P(rug) - P(win); higher = riskier."""

    def __init__(self, rug, big, meta):
        self.rug, self.big, self.meta = rug, big, meta

    def score(self, x: dict) -> float:
        sig = lambda v: 1 / (1 + math.exp(-max(-50.0, min(50.0, v))))
        return sig(self.rug.score(x)) - sig(self.big.score(x))

    def to_json(self) -> dict:
        return {"rug": self.rug.to_json(), "big": self.big.to_json(), "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "RugWinModel":
        from tradingbot.model_book import Model
        return cls(Model.from_json(d["rug"]), Model.from_json(d["big"]), d.get("meta", {}))


def _fit_capped(rows: list[dict], y: np.ndarray):
    from tradingbot.model_book import Model
    names = sorted({k for r in rows for k in r["x"]})
    x, med = wa._matrix([r["x"] for r in rows], names)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    w, b = wa.logistic(np.clip((x - mu) / sd, -la.CLIP_Z, la.CLIP_Z), y)
    return Model(names, mu.tolist(), sd.tolist(), {n: float(med[n]) for n in names}, w.tolist(), float(b), 0.0,
                 {"clip_z": la.CLIP_Z})


def train_rw(store, cfg: Config, now: float | None = None, days: float | None = None):
    """The rug-vs-win model on the scorer's finished liquid pools of the last `launch_train_days`. Meta keeps the
    score percentiles of every training pool and of those in the history model's bottom 20% (fallback ranking
    until the live scorer has seen ROLLING_MIN pools of each)."""
    ensure_schema(store)
    now = now if now is not None else time.time()
    # every finished pool, as the study (11-05); `days` = the earlier rule's window (the rw24 twin books)
    rows = _pull_rows(store, now - days * 86400 if days else 0.0)
    if len(rows) < PULL_MIN_TRAIN:
        log.warning("rug-vs-win model: not enough rows to train (%d)", len(rows))
        return None
    rug = np.array([float(r["ret"] <= -0.999) for r in rows])
    big = np.array([float(r["ret"] >= RW_BIG) for r in rows])
    if rug.min() == rug.max() or big.min() == big.max():
        return None
    model = RugWinModel(_fit_capped(rows, rug), _fit_capped(rows, big), {})
    scores = np.array([model.score(r["x"]) for r in rows])
    sh = [r["score_h"] for r in rows if r["score_h"] is not None]
    cut = float(np.quantile(sh, LOW_SHARE)) if len(sh) >= ROLLING_MIN else None
    low = np.array([cut is not None and r["score_h"] is not None and r["score_h"] <= cut for r in rows])
    model.meta = {"trained_ts": time.time(), "rows": len(rows), "rug_rate": float(rug.mean()),
                  "big_win_rate": float(big.mean()), "low_rows": int(low.sum()),
                  "quantiles": [float(np.quantile(scores, q / 100)) for q in RISK_LEVELS],
                  "low_quantiles": [float(np.quantile(scores[low] if low.sum() >= ROLLING_MIN else scores, q / 100))
                                    for q in RISK_LEVELS],
                  "days": round((now - rows[0]["create_ts"]) / 86400, 1)}
    return model


def train_rw24_and_save(store, cfg: Config, now: float | None = None) -> Path | None:
    """The earlier rule's model (rw24.json): the last `launch_train_days` only."""
    model = train_rw(store, cfg, now=now, days=cfg.launch_train_days)
    if model is None:
        return None
    log.info("rug-vs-win (24 h rule) model trained on %d rows", model.meta["rows"])
    return _save(model, rw24_model_path(cfg))


def train_rw_and_save(store, cfg: Config) -> Path | None:
    model = train_rw(store, cfg)
    if model is None:
        return None
    log.info("rug-vs-win model trained on %d rows", model.meta["rows"])
    return _save(model, rw_model_path(cfg))


def _rank_pct(past: list, value: float, quantiles) -> float | None:
    """Percentile of `value` among the past day's scores, or the training quantiles while there are too few."""
    if len(past) >= ROLLING_MIN:
        return 100.0 * float(np.mean(np.array(past) < value))
    return float(np.interp(value, quantiles, RISK_LEVELS)) if quantiles else None


def train_pull_and_save(store, cfg: Config) -> Path | None:
    model = train_pull(store, cfg)
    if model is None:
        return None
    log.info("pull model trained on %d rows (early-pull rate %.1f%%)", model.meta["rows"],
             100 * model.meta["early_pull_rate"])
    return _save(model, pull_model_path(cfg))


def train_and_save(store, cfg: Config) -> Path | None:
    model = train(store, cfg)
    if model is None:
        return None
    path = model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model.to_json()))
    os.replace(tmp, path)
    log.info("launch4 model trained on %d rows, cut-offs top2 %.3f top5 %.3f", model.meta["rows"],
             model.meta["cut2"], model.meta["cut5"])
    return path


def live_features(rec: dict, create_ts: float, prior_pools: int) -> dict:
    """What `launch_analysis.rows` builds at second DECIDE_S (static features without a name + chain features)."""
    static = wa._static_features({"name": "", "dex": "pumpswap", "created_ts": create_ts})
    return {**static, **la.launch_features(rec["txs"], DECIDE_S, rec.get("creator"), prior_pools,
                                            rec.get("seed_sol"))}


# --- live scorer ----------------------------------------------------------------------------------

class LaunchLive:
    def __init__(self, cfg: Config, store, helius: Helius | None = None, clock=time.time, workers: int = 4,
                 signals=None):
        from tradingbot.enrich import Budget
        self.cfg, self.store, self.clock = cfg, store, clock
        self.signals = signals  # the engine's model-signal queue: top-2% / top-5% picks go to the launch books
        self.helius = helius or Helius()
        self.budget = Budget(store, "launch_live", cfg.launch_live_daily_credits)
        ensure_schema(store)
        self.lock = threading.Lock()
        self.jobs: list = []  # heap of (due, seq, kind, pool)
        self.seq = 0
        self.seen: collections.OrderedDict = collections.OrderedDict()  # LP mints handled
        self.lp_pool: dict[str, str] = {}  # LP mint -> pool, for pools waiting on their outcome
        self.counts: collections.Counter = collections.Counter()
        self.ws = {"connected": False, "messages": 0, "bytes": 0, "reconnects": 0, "last_error": None}
        self.history_s = collections.deque(maxlen=50)  # seconds each decision's history call took
        self.start_lag_s = collections.deque(maxlen=50)  # seconds each first decide attempt started after its due time
        self.model, self.model_mtime, self.model_checked = None, None, 0.0
        self.hist_model, self.hist_mtime, self.hist_checked = None, None, 0.0
        self.hist_scores: collections.deque = collections.deque()  # (decided_ts, score_h) of liquid pools, past day
        self.pull_model, self.pull_mtime, self.pull_checked = None, None, 0.0
        self.low_risks: collections.deque = collections.deque()  # (decided_ts, risk) of bottom-20% pools, past day
        self.rw_model, self.rw_mtime, self.rw_checked = None, None, 0.0
        self.rw_all: collections.deque = collections.deque()  # (decided_ts, rw) of liquid pools, past RW_BLOCK_S
        self.rw_low: collections.deque = collections.deque()  # (decided_ts, rw) of bottom-20% pools, past RW_BLOCK_S
        # the earlier live rule, kept as twin books (11-05): nightly 14-day model, 24 h ranking, low_h
        self.rw24_model, self.rw24_mtime, self.rw24_checked = None, None, 0.0
        self.rw24_all: collections.deque = collections.deque()
        self.rw24_low: collections.deque = collections.deque()
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="launch-live") if workers else None
        self.saved_at = 0.0
        self.tries: dict[str, int] = {}
        self._resume()

    # bookkeeping ------------------------------------------------------------------------------------
    def _count(self, what: str) -> None:
        with self.lock:
            self.counts[what] += 1

    def _schedule(self, due: float, kind: str, pool: str) -> None:
        with self.lock:
            self.seq += 1
            heapq.heappush(self.jobs, (due, self.seq, kind, pool))

    def _update(self, pool: str, **cols) -> None:
        keys = ", ".join(f"{k}=?" for k in cols)
        self.store.execute(f"UPDATE watch_launch_live SET {keys} WHERE pool=?", (*cols.values(), pool))

    def _row(self, pool: str):
        r = self.store.query("SELECT * FROM watch_launch_live WHERE pool=?", (pool,))
        return r[0] if r else None

    def _spend(self, before: int, pool: str | None = None) -> None:
        used = self.helius.credits_used - before
        if used:
            self.budget.add(used)
            if pool:
                self.store.execute("UPDATE watch_launch_live SET credits=credits+? WHERE pool=?", (used, pool))

    def _resume(self) -> None:
        """After a restart: re-arm outcome reads that are still due, mark the rest lost."""
        now = self.clock()
        for r in self.store.query("SELECT * FROM watch_launch_live WHERE notified_ts > ?", (now - 2 * wa.DAY_S,)):
            self.seen[r["lp_mint"]] = True
            if r["status"] == "pending":
                self._update(r["pool"], status="lost", reason="restart before the decision")
            elif r["status"] == "scored":
                if now < (r["create_ts"] or 0) + EXIT_S:
                    self.lp_pool[r["lp_mint"]] = r["pool"]
                    if r["fill_ts"] is None:
                        self._schedule(now, "fill", r["pool"])
                    self._schedule(r["create_ts"] + EXIT_S, "exit", r["pool"])
                else:
                    self._update(r["pool"], status="lost", reason="restart: exit read missed")

    def save_state(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self.saved_at < STATE_SAVE_S:
            return
        self.saved_at = now
        day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        with self.lock:
            counts, self.counts = self.counts, collections.Counter()
            ws = dict(self.ws, saved_ts=now)
            calls, lags = sorted(self.history_s), sorted(self.start_lag_s)
        from tradingbot.data import helius as hl
        ws["helius_process"] = {k: round(v, 1) for k, v in hl.STATS.items()}  # whole bot process, since start
        if calls:  # how long the decision's history call took (recent ones), pace queue and 429 back-off included
            ws["history_call_s"] = {"n": len(calls), "median": round(calls[len(calls) // 2], 2),
                                    "max": round(calls[-1], 2)}
        if lags:  # a large lag means the worker threads were busy with other jobs
            ws["decide_start_lag_s"] = {"n": len(lags), "median": round(lags[len(lags) // 2], 2),
                                        "max": round(lags[-1], 2)}
        key = f"launch_live:counts:{day}"
        old = collections.Counter(json.loads(self.store.get_state(key) or "{}"))
        old.update(counts)
        self.store.set_state(key, json.dumps(old))
        self.store.set_state("launch_live:ws", json.dumps(ws))

    def _load_model(self):
        now = self.clock()
        if self.model is not None and now - self.model_checked < MODEL_RELOAD_S:
            return self.model
        self.model_checked = now
        path = model_path(self.cfg)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return self.model
        if mtime != self.model_mtime:
            from tradingbot.model_book import Model
            try:
                self.model = Model.from_json(json.loads(path.read_text()))
                self.model_mtime = mtime
            except (OSError, ValueError, KeyError) as exc:
                log.warning("launch4 model unreadable: %s", exc)
        return self.model

    def _load_hist_model(self):
        """The history-trained model (launch4h.json). When a new one is loaded, the past day's liquid pools are
        re-scored from their stored features, so the rolling cut-offs compare like with like."""
        now = self.clock()
        if self.hist_model is not None and now - self.hist_checked < MODEL_RELOAD_S:
            return self.hist_model
        self.hist_checked = now
        path = hist_model_path(self.cfg)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return self.hist_model
        if mtime != self.hist_mtime:
            from tradingbot.model_book import Model
            try:
                model = Model.from_json(json.loads(path.read_text()))
            except (OSError, ValueError, KeyError) as exc:
                log.warning("launch4h model unreadable: %s", exc)
                return self.hist_model
            past = [(r["decided_ts"], model.score(json.loads(r["feat"]))) for r in self.store.query(
                "SELECT decided_ts, feat FROM watch_launch_live WHERE liquid=1 AND feat IS NOT NULL "
                "AND decided_ts >= ? ORDER BY decided_ts", (now - ROLLING_S,))]
            with self.lock:
                self.hist_model, self.hist_mtime = model, mtime
                self.hist_scores = collections.deque(past)
        return self.hist_model

    def _op_live(self, creator: str | None, create_ts: float) -> dict:
        """The creator's record from this table, as `_creator_history` builds it for training."""
        if not creator:
            return _op_features([], create_ts)
        prior = self.store.query("SELECT create_ts, pulled_ts FROM watch_launch_live WHERE creator=? AND liquid=1 "
                                 "AND status IN ('scored', 'done') AND create_ts < ?", (creator, create_ts))
        return _op_features([dict(p) for p in prior], create_ts)

    def _load_pull_model(self):
        """The early-pull model (pull.json). A new one re-scores the past day's bottom-20% pools."""
        now = self.clock()
        if self.pull_model is not None and now - self.pull_checked < MODEL_RELOAD_S:
            return self.pull_model
        self.pull_checked = now
        path = pull_model_path(self.cfg)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return self.pull_model
        if mtime != self.pull_mtime:
            from tradingbot.model_book import Model
            try:
                model = Model.from_json(json.loads(path.read_text()))
            except (OSError, ValueError, KeyError) as exc:
                log.warning("pull model unreadable: %s", exc)
                return self.pull_model
            past = [(r["decided_ts"], model.score(pull_inputs(json.loads(r["feat"]),
                                                               self._op_live(r["creator"], r["create_ts"]))))
                    for r in self.store.query(
                        "SELECT creator, create_ts, decided_ts, feat FROM watch_launch_live WHERE low_h=1 "
                        "AND feat IS NOT NULL AND decided_ts >= ? ORDER BY decided_ts", (now - ROLLING_S,))]
            with self.lock:
                self.pull_model, self.pull_mtime = model, mtime
                self.low_risks = collections.deque(past)
        return self.pull_model

    def _risk_verdict(self, feat: dict, creator: str | None, create_ts: float, now: float) -> dict:
        """Early-pull risk of a bottom-20% pool and its percentile among the past day's bottom-20% pools (the
        training pools' percentiles until ROLLING_MIN of them have been seen)."""
        model = self._load_pull_model()
        if model is None:
            return {}
        risk = model.score(pull_inputs(feat, self._op_live(creator, create_ts)))
        with self.lock:
            while self.low_risks and self.low_risks[0][0] < now - ROLLING_S:
                self.low_risks.popleft()
            past = [r for _, r in self.low_risks]
            self.low_risks.append((now, risk))
        if len(past) >= ROLLING_MIN:
            pct = 100.0 * float(np.mean(np.array(past) < risk))
        else:
            qs = model.meta.get("risk_quantiles")
            pct = float(np.interp(risk, qs, RISK_LEVELS)) if qs else None
        return {"risk": risk, "risk_pct": pct}

    def _rw_track(self, track: str):
        """(model path, ranking window, bottom-20% test) of a rug-vs-win rule: `rw` = the study's rule since 11-05
        (6 h), `rw24` = the earlier live rule kept as twin books (24 h ranking, nightly 14-day model, low_h)."""
        if track == "rw24":
            return rw24_model_path(self.cfg), RW24_WINDOW_S, lambda r: bool(r["low_h"])
        return rw_model_path(self.cfg), RW_BLOCK_S, rw_low

    def _load_rw_model(self, track: str = "rw"):
        """A rug-vs-win model (rw.json / rw24.json). A new one re-scores its ranking window's liquid pools."""
        path, window, is_low = self._rw_track(track)
        now = self.clock()
        model = getattr(self, f"{track}_model")
        if model is not None and now - getattr(self, f"{track}_checked") < MODEL_RELOAD_S:
            return model
        setattr(self, f"{track}_checked", now)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return model
        if mtime != getattr(self, f"{track}_mtime"):
            try:
                new = RugWinModel.from_json(json.loads(path.read_text()))
            except (OSError, ValueError, KeyError) as exc:
                log.warning("rug-vs-win model %s unreadable: %s", path.name, exc)
                return model
            every, low = [], []
            for r in self.store.query("SELECT creator, create_ts, decided_ts, feat, low_h, rw_low6 "
                                      "FROM watch_launch_live WHERE liquid=1 AND feat IS NOT NULL AND decided_ts >= ? "
                                      "ORDER BY decided_ts", (now - window,)):
                sc = new.score(pull_inputs(json.loads(r["feat"]), self._op_live(r["creator"], r["create_ts"])))
                every.append((r["decided_ts"], sc))
                if is_low(r):
                    low.append((r["decided_ts"], sc))
            with self.lock:
                setattr(self, f"{track}_model", new)
                setattr(self, f"{track}_mtime", mtime)
                setattr(self, f"{track}_all", collections.deque(every))
                setattr(self, f"{track}_low", collections.deque(low))
            model = new
        return model

    def _rw_verdict(self, feat: dict, creator: str | None, create_ts: float, now: float, low: bool,
                    track: str = "rw") -> dict:
        """Rug-vs-win score of a liquid pool and its percentile among the ranking window's liquid pools (and, for
        a pool in that rule's bottom 20%, among those). Columns `<track>`, `<track>_pct`, `<track>_low_pct`."""
        model = self._load_rw_model(track)
        if model is None:
            return {}
        _, window, _ = self._rw_track(track)
        rw = model.score(pull_inputs(feat, self._op_live(creator, create_ts)))
        with self.lock:
            every, lows = getattr(self, f"{track}_all"), getattr(self, f"{track}_low")
            for dq in (every, lows):
                while dq and dq[0][0] < now - window:
                    dq.popleft()
            past_all, past_low = [v for _, v in every], [v for _, v in lows]
            every.append((now, rw))
            if low:
                lows.append((now, rw))
        out = {track: rw, f"{track}_pct": _rank_pct(past_all, rw, model.meta.get("quantiles"))}
        if low:
            out[f"{track}_low_pct"] = _rank_pct(past_low, rw, model.meta.get("low_quantiles"))
        return out

    def _hist_verdict(self, feat: dict, now: float) -> dict:
        """score_h and the rolling top-2% / top-5% flags for a liquid pool (the score joins the history after)."""
        model = self._load_hist_model()
        if model is None:
            return {}
        score = model.score(feat)
        with self.lock:
            while self.hist_scores and self.hist_scores[0][0] < now - ROLLING_S:
                self.hist_scores.popleft()
            past = [s for _, s in self.hist_scores]
            recent = [s for t, s in self.hist_scores if t >= now - RW_BLOCK_S]
            self.hist_scores.append((now, score))
        fallback = (model.meta.get("score_quantiles") or {}).get("20")
        if len(past) >= ROLLING_MIN:
            cut2, cut5 = float(np.quantile(past, 1 - TOP2)), float(np.quantile(past, 1 - TOP5))
            low = float(np.quantile(past, LOW_SHARE))
        else:
            cut2, cut5 = model.meta.get("cut2"), model.meta.get("cut5")
            low = fallback
        low6 = float(np.quantile(recent, LOW_SHARE)) if len(recent) >= ROLLING_MIN else low
        pick = lambda c: int(c is not None and score >= c - 1e-9)
        under = lambda c: int(c is not None and score <= c + 1e-9)
        return {"score_h": score, "top2h": pick(cut2), "top5h": pick(cut5), "low_h": under(low),
                "rw_low6": under(low6)}

    # websocket input ------------------------------------------------------------------------------
    def on_message(self, msg: str) -> None:
        """One websocket message (programNotification on Token-2022, dataSize 82)."""
        with self.lock:
            self.ws["messages"] += 1
            self.ws["bytes"] += len(msg)
        try:
            body = json.loads(msg)
            value = body["params"]["result"]["value"]
            import base64
            raw = base64.b64decode(value["account"]["data"][0])
        except (ValueError, KeyError, TypeError):
            return
        self.on_lp_mint(value["pubkey"], raw)

    def on_lp_mint(self, lp_mint: str, raw: bytes) -> None:
        m = ps.parse_lp_mint(raw)
        if m is None or not m["authority"]:
            return
        now = self.clock()
        with self.lock:
            pool = self.lp_pool.get(lp_mint)
        if pool is not None:
            if m["supply"] == 0:  # liquidity withdrawn while we wait for the exit read
                self.store.execute("UPDATE watch_launch_live SET pulled_ts=COALESCE(pulled_ts, ?) WHERE pool=?",
                                   (now, pool))
            return
        with self.lock:
            if lp_mint in self.seen or m["supply"] == 0:
                return
            self.seen[lp_mint] = True
            if len(self.seen) > SEEN_MAX:
                self.seen.popitem(last=False)
        self._count("new LP mints")
        self._schedule(now, "new", json.dumps({"lp": lp_mint, "pool": m["authority"], "t": now}))

    # jobs -----------------------------------------------------------------------------------------
    def next_due(self) -> float | None:
        with self.lock:
            return self.jobs[0][0] if self.jobs else None

    def step(self) -> int:
        """Run every job that is due; returns how many were started."""
        now = self.clock()
        due = []
        with self.lock:
            while self.jobs and self.jobs[0][0] <= now:
                due.append(heapq.heappop(self.jobs))
        for _, _, kind, arg in due:
            fn = {"new": self._new, "decide": self._decide, "fill": self._fill, "exit": self._exit}[kind]
            if self.pool is not None:
                self.pool.submit(self._safe, fn, arg)
            else:
                self._safe(fn, arg)
        self.save_state()
        return len(due)

    def _safe(self, fn, arg) -> None:
        try:
            fn(arg)
        except Exception:  # one pool's failure must not stop the scorer
            log.exception("launch-live %s failed", fn.__name__)
            self._count(f"error in {fn.__name__}")

    def _new(self, arg: str) -> None:
        job = json.loads(arg)
        pool, lp, notified = job["pool"], job["lp"], job["t"]
        if not self.budget.allows(1):
            self._count("skipped: out of budget")
            return
        before = self.helius.credits_used
        accts = self.helius.accounts([pool])
        self._spend(before)
        acct = (accts or {}).get(pool)
        if accts is None:
            self._count("skipped: pool read failed")
            return
        if not acct or acct[0] != ps.PROGRAM:
            self._count("skipped: not a pumpswap pool")
            return
        p = ps.parse_pool(acct[1])
        if p is None:
            self._count("skipped: not a SOL pair")
        elif p["lp_mint"] != lp:
            self._count("skipped: LP mint doesn't match the pool")
        elif not p["handmade"]:
            self._count("skipped: graduation")
        elif p["token_mint"].endswith("pump"):
            self._count("skipped: pump mint")
        else:
            self._count("hand-made pools")
            self.store.execute(
                "INSERT OR IGNORE INTO watch_launch_live(pool, lp_mint, token_mint, creator, notified_ts, sol_vault, "
                "token_vault, status, credits) VALUES(?,?,?,?,?,?,?,?,?)",
                (pool, lp, p["token_mint"], p["creator"], notified, p["sol_vault"], p["token_vault"], "pending",
                 self.helius.credits_used - before))
            self._schedule(notified + self.cfg.launch_live_decide_delay_s, "decide", pool)

    def _decide(self, pool: str) -> None:
        r = self._row(pool)
        if r is None or r["status"] != "pending":
            return
        if not self.budget.allows(10 * HISTORY_PAGES):
            self._update(pool, status="skipped", reason="out of budget")
            return
        if pool not in self.tries and r["notified_ts"]:
            with self.lock:
                self.start_lag_s.append(self.clock() - r["notified_ts"] - self.cfg.launch_live_decide_delay_s)
        before = self.helius.credits_used
        started = time.monotonic()
        txs, done, ok = self.helius.history_until(pool, self.clock(), HISTORY_PAGES)
        with self.lock:
            self.history_s.append(time.monotonic() - started)
        self._spend(before, pool)
        now = self.clock()
        if ok and not txs and self.tries.get(pool, 0) < EMPTY_RETRIES:
            self.tries[pool] = self.tries.get(pool, 0) + 1
            self._count("history empty, retried")
            self._schedule(now + EMPTY_RETRY_S, "decide", pool)
            return
        self.tries.pop(pool, None)
        if not ok or not txs:
            self._update(pool, status="skipped", reason="history call failed" if not ok else "no transactions yet",
                         decided_ts=now)
            return
        rec = launch_record(txs, pool, r["token_mint"], window_s=60, exhausted=done)
        create = rec["create_ts"]
        rows = rec["txs"]
        indexed = rows[-1][0] if rows else 0.0
        cols = {"create_ts": create, "decided_ts": now, "indexed_s": indexed, "creator": rec.get("creator")}
        upto = [t for t in rows if t[0] <= DECIDE_S]
        if now - create > TOO_LATE_S:
            self._update(pool, status="skipped", reason="too late (pool older than 20 s)", **cols)
            return
        if not upto:
            self._update(pool, status="skipped", reason="nothing on chain by second 4", **cols)
            return
        if not done and indexed < DECIDE_S:
            self._update(pool, status="skipped", reason="history pages ended before second 4", **cols)
            return
        sol4 = upto[-1][6]
        sol_usd = la._sol_usd(self.store, now)
        liquid = (sol_usd is None or 2 * sol4 * sol_usd >= self.cfg.model_book_min_liq_usd) \
            and not la.pulled_by(rows, DECIDE_S)
        feat = live_features(rec, create, la._prior_pools(self.store, rec.get("creator"), create))
        model = self._load_model()
        score = model.score(feat) if model is not None else None
        meta = model.meta if model is not None else {}
        pick = lambda c: int(bool(liquid and score is not None and c is not None and score >= c - 1e-9))
        cols.update(liquid=int(liquid), score=score, top2=pick(meta.get("cut2")), top5=pick(meta.get("cut5")),
                    feat=json.dumps(feat), sol_at_decide=sol4, peak_sol=max(t[6] for t in upto),
                    liq_usd=2 * sol4 * sol_usd if sol_usd else None)
        if not liquid:
            self._update(pool, status="skipped", reason="not liquid at second 4", **cols)
            return
        cols.update(self._hist_verdict(feat, now))
        if cols.get("low_h"):
            cols.update(self._risk_verdict(feat, rec.get("creator"), create, now))
        cols.update(self._rw_verdict(feat, rec.get("creator"), create, now, bool(cols.get("rw_low6"))))
        if self.cfg.launch_rw24_book:
            cols.update(self._rw_verdict(feat, rec.get("creator"), create, now, bool(cols.get("low_h")), track="rw24"))
        flags = {**low_flags(cols.get("low_h"), cols.get("risk_pct")),
                 **rw_flags(cols.get("rw_low6"), cols.get("rw_pct"), cols.get("rw_low_pct")),
                 **rw24_flags(cols.get("low_h"), cols.get("rw24_pct"), cols.get("rw24_low_pct"))}
        self._update(pool, status="scored", reason=None if model else "no launch4 model yet", **cols)
        if self.signals is not None and (cols["top2"] or cols["top5"] or cols.get("top2h") or cols.get("top5h")
                                         or flags["d12"] or flags["rw50"] or flags["d12rw50"]
                                         or flags["rw50_24"] or flags["d12rw50_24"]):
            px = la.chain_price(rows, DECIDE_S)  # SOL per whole token (vault UI amounts)
            if sol_usd and px:
                self.signals.put_nowait({
                    "pool": pool, "mint": r["token_mint"], "name": "", "score": score, "price": px * sol_usd,
                    "liq": 2 * sol4 * sol_usd, "ts": now, "decided_ts": now, "created_ts": create,
                    "dex": "pumpswap", "kind": "launch", "top2": bool(cols["top2"]), "top5": bool(cols["top5"]),
                    "top2h": bool(cols.get("top2h")), "top5h": bool(cols.get("top5h")), **flags})
                self._count("signals sent to the launch books")
            else:
                self._count("pick not sent: no SOL price")
        with self.lock:
            self.lp_pool[r["lp_mint"]] = pool
        self._schedule(max(now, create + FILL_S + 0.5), "fill", pool)
        self._schedule(create + EXIT_S, "exit", pool)

    def _vaults(self, r) -> tuple[float, float] | None:
        """(SOL in the pool, raw token units in the pool) from its two vaults (1 credit)."""
        before = self.helius.credits_used
        accts = self.helius.accounts([r["sol_vault"], r["token_vault"]], commitment="processed")
        self._spend(before, r["pool"])
        if not accts:
            return None
        s, t = accts.get(r["sol_vault"]), accts.get(r["token_vault"])
        sol = ps.token_amount(s[1]) if s else 0
        tok = ps.token_amount(t[1]) if t else 0
        return (sol or 0) / 1e9, float(tok or 0)

    def _fill(self, pool: str) -> None:
        r = self._row(pool)
        if r is None or r["status"] != "scored" or r["fill_ts"] is not None:
            return
        v = self._vaults(r)
        now = self.clock()
        if v is None:
            self._update(pool, reason="fill read failed")
            return
        sol, tok = v
        self._update(pool, fill_ts=now, fill_sol=sol, fill_px=(sol / tok) if sol and tok else None,
                     peak_sol=max(r["peak_sol"] or 0.0, sol))

    def _exit(self, pool: str) -> None:
        r = self._row(pool)
        if r is None or r["status"] != "scored":
            return
        with self.lock:
            self.lp_pool.pop(r["lp_mint"], None)
        v = self._vaults(r)
        now = self.clock()
        if v is None:
            self._update(pool, status="lost", reason="exit read failed", exit_ts=now)
            return
        sol, tok = v
        px = (sol / tok) if sol and tok else None
        peak = max(r["peak_sol"] or 0.0, r["fill_sol"] or 0.0)
        if r["fill_px"] is None:
            self._update(pool, status="lost", reason="no fill price", exit_ts=now, exit_sol=sol, exit_px=px)
            return
        pulled = r["pulled_ts"] is not None or (peak > 0 and sol < la.PULLED_SHARE * peak) or not px
        ret = -1.0 if pulled else min(px / r["fill_px"] - 1 - la.COST, 3.0)
        self._update(pool, status="done", exit_ts=now, exit_sol=sol, exit_px=px, ret=ret)

    # websocket ------------------------------------------------------------------------------------
    def ws_url(self) -> str:
        return WS_HELIUS.format(key=self.helius.api_key) if self.helius.enabled else WS_PUBLIC

    async def _ws_loop(self, stop: threading.Event) -> None:
        sub = {"jsonrpc": "2.0", "id": 1, "method": "programSubscribe",
               "params": [ps.TOKEN_2022, {"encoding": "base64", "commitment": "confirmed",
                                          "filters": [{"dataSize": ps.LP_MINT_SIZE}]}]}
        backoff = 1.0
        while not stop.is_set():
            try:
                import websockets  # inside the try: a missing package shows in last_error, not a dead thread
                async with websockets.connect(self.ws_url(), max_size=2 ** 22, ping_interval=20,
                                              ping_timeout=20, open_timeout=20) as ws:
                    await ws.send(json.dumps(sub))
                    with self.lock:
                        self.ws["connected"] = True
                    backoff = 1.0
                    async for msg in ws:
                        if stop.is_set():
                            return
                        self.on_message(msg)
            except Exception as exc:  # network drops are normal: reconnect with back-off
                with self.lock:
                    self.ws.update(connected=False, last_error=f"{type(exc).__name__}: {str(exc)[:120]}")
                    self.ws["reconnects"] += 1
                log.warning("launch-live websocket: %s; reconnecting in %.0f s", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(60.0, backoff * 2)

    def start(self) -> threading.Event:
        """Websocket thread + scheduler thread (daemons). Returns the stop event."""
        stop = threading.Event()
        threading.Thread(target=lambda: asyncio.run(self._ws_loop(stop)), name="launch-ws", daemon=True).start()

        def run():
            while not stop.is_set():
                try:
                    self.step()
                except Exception:
                    log.exception("launch-live step failed")
                due = self.next_due()
                wait = 0.5 if due is None else min(0.5, max(0.02, due - self.clock()))
                stop.wait(wait)
        threading.Thread(target=run, name="launch-live", daemon=True).start()
        if (self.cfg.launch_rw_book or self.cfg.launch_rw24_book) and self.cfg.db_path != ":memory:":
            threading.Thread(target=self._rw_retrain_loop, args=(stop,), name="rw-retrain", daemon=True).start()
        return stop

    def _rw_retrain_loop(self, stop: threading.Event) -> None:
        """Retrain the rug-vs-win model every RW_BLOCK_S on every finished pool, as the study did before each
        6 h block (it used to be trained once a night on 14 days). A low-priority child process
        (`launch-rw-train`), so the scorer's threads never wait on it; the scorer reloads the new file within
        MODEL_RELOAD_S and re-ranks the past RW_BLOCK_S."""
        import os
        import subprocess
        import sys

        def low_priority():
            try:
                os.nice(19)
                with open("/proc/self/oom_score_adj", "w") as fh:
                    fh.write("1000")
            except OSError:
                pass
        jobs = ([(rw_model_path, RW_BLOCK_S, "launch-rw-train")] if self.cfg.launch_rw_book else []) + \
            ([(rw24_model_path, RW24_RETRAIN_S, "launch-rw24-train")] if self.cfg.launch_rw24_book else [])
        while not stop.is_set():
            wait = RW_RETRY_S
            for path_fn, every, command in jobs:
                try:
                    age = time.time() - path_fn(self.cfg).stat().st_mtime
                except OSError:
                    age = float("inf")
                if age < every:
                    wait = min(wait, every - age + 5)
                    continue
                try:
                    res = subprocess.run([sys.executable, "-m", "tradingbot", command],
                                         env={**os.environ, "TB_DB_PATH": self.cfg.db_path},
                                         preexec_fn=low_priority, timeout=RW_RETRAIN_TIMEOUT_S,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    if res.returncode != 0:
                        log.error("%s: child exited %s: %s", command, res.returncode,
                                  res.stderr.decode(errors="replace")[-300:])
                    else:
                        log.info("rug-vs-win model retrained (%s, every %d h)", command, every // 3600)
                except Exception as exc:  # noqa: BLE001 (the old model keeps scoring)
                    log.error("%s could not run: %s: %s", command, type(exc).__name__, exc)
            stop.wait(wait)  # a failed run (no new file) is retried after RW_RETRY_S


# --- status ---------------------------------------------------------------------------------------

def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 2) if vals else None


RETURN_BANDS = (("rugged (-100%)", -9.0, -0.999), ("-50% to -99%", -0.999, -0.5), ("0 to -50%", -0.5, 0.0),
                ("0 to +50%", 0.0, 0.5), ("+50% or more", 0.5, 99.0))


def _group(rows, rng) -> dict:
    rets = [r["ret"] for r in rows if r["ret"] is not None]
    if not rets:
        return {"n": 0}
    n = len(rets)
    out = {"n": n, "mean_pct": round(100 * float(np.mean(rets)), 1),
           "median_pct": round(100 * float(np.median(rets)), 1),
           "win_pct": round(100 * sum(x > 0 for x in rets) / n),
           "rugged_pct": round(100 * sum(x <= -0.999 for x in rets) / n),
           "returns (% of trades)": {name: round(100 * sum(lo <= x < hi for x in rets) / n)
                                     for name, lo, hi in RETURN_BANDS}}
    if len(rets) >= 5:
        boots = [np.mean(rng.choice(rets, len(rets))) for _ in range(2000)]
        out["ci90"] = [round(100 * float(np.quantile(boots, q)), 1) for q in (0.05, 0.95)]
    return out


COMPARE_TOPS = (2.0, 5.0, 8.0)  # equal-share cuts: the old model's live top 2% has drifted to ~8% of pools
EARLY_RUG_S = 90  # pulled before GeckoTerminal would list the pool (~70 s): pools the old model never trained on


def _rug_split(rows) -> dict:
    """Rugs by when the LP was pulled: early (< EARLY_RUG_S after creation), later, or drained with no pull seen."""
    rugs = [r for r in rows if r["ret"] is not None and r["ret"] <= -0.999]
    n = len([r for r in rows if r["ret"] is not None]) or 1
    early = sum(1 for r in rugs if r["pulled_ts"] and r["pulled_ts"] - r["create_ts"] < EARLY_RUG_S)
    late = sum(1 for r in rugs if r["pulled_ts"] and r["pulled_ts"] - r["create_ts"] >= EARLY_RUG_S)
    return {"early_rug_pct": round(100 * early / n, 1), "later_rug_pct": round(100 * late / n, 1),
            "drained_no_pull_pct": round(100 * (len(rugs) - early - late) / n, 1)}


def compare(store, hours: float | None = None, now: float | None = None, seed: int = 7) -> dict:
    """Old (launch4.json) vs history (launch4h.json) model on the same pools: every finished liquid pool both
    scored, ranked by each model within the window (top 2/5/8%), plus the live flags. Outcomes are the scorer's
    ideal trades (vault price at second ~6.5 in, minute 6 out, 6% cost); rugs split by pull time."""
    ensure_schema(store)
    now = now if now is not None else time.time()
    since = now - hours * 3600 if hours else 0
    rows = store.query("SELECT pool, create_ts, score, score_h, top2, top5, top2h, top5h, ret, pulled_ts "
                       "FROM watch_launch_live WHERE status='done' AND liquid=1 AND ret IS NOT NULL "
                       "AND score IS NOT NULL AND score_h IS NOT NULL AND create_ts >= ? ORDER BY create_ts",
                       (since,))
    if not rows:
        return {"pools": 0, "note": "no finished pools scored by both models yet"}
    rng = np.random.default_rng(seed)
    group = lambda rs: {**_group(rs, rng), **_rug_split(rs)}
    out = {"pools": len(rows),
           "from": datetime.fromtimestamp(rows[0]["create_ts"], timezone.utc).strftime("%m-%d %H:%M"),
           "to": datetime.fromtimestamp(rows[-1]["create_ts"], timezone.utc).strftime("%m-%d %H:%M"),
           "every liquid pool": group(rows)}
    ranked = {}
    for t in COMPARE_TOPS:
        k = max(1, round(len(rows) * t / 100))
        old = sorted(rows, key=lambda r: -r["score"])[:k]
        hist = sorted(rows, key=lambda r: -r["score_h"])[:k]
        both = {r["pool"] for r in old} & {r["pool"] for r in hist}
        ranked[f"top {t:g}%"] = {"picks_each": k, "old model": group(old), "history model": group(hist),
                                 "picked by both": len(both),
                                 "overlap_pct": round(100 * len(both) / k)}
    out["ranked in the window (same share of pools for both)"] = ranked
    out["live flags (as the books saw them)"] = {
        "old top2": group([r for r in rows if r["top2"]]), "old top5": group([r for r in rows if r["top5"]]),
        "history top2h": group([r for r in rows if r["top2h"]]),
        "history top5h": group([r for r in rows if r["top5h"]])}
    from tradingbot.model_book import decile_index, decile_label, BIG_WIN
    deciles = {}
    for name, key in (("old model", "score"), ("history model", "score_h")):
        idx = decile_index([r[key] for r in rows])
        deciles[name] = {}
        for i in range(10):
            b = [r for r, j in zip(rows, idx) if j == i]
            deciles[name][decile_label(i)] = {
                **_group(b, rng), **_rug_split(b),
                "big_win_pct": round(100 * sum(r["ret"] >= BIG_WIN for r in b) / len(b)) if b else None}
    out["by score decile (ranked within the window)"] = deciles
    out["note"] = (f"big win = +{BIG_WIN:.0%} or more after costs; ideal pool-level trades; early rug = LP pulled < {EARLY_RUG_S} s after creation, before "
                   "GeckoTerminal lists a pool (the old model's training pools all survived that); "
                   "drained = pool SOL fell below 10% of its peak or no price, no LP pull seen")
    return out


PATH_SCHEMA = """CREATE TABLE IF NOT EXISTS watch_launch_path(
    pool TEXT, sec INTEGER, sol REAL, tok REAL, ts REAL, PRIMARY KEY(pool, sec))"""
EXIT_MINUTES = (1, 2, 3, 4, 5)  # earlier exits tried against the books' minute 6
PULL_BUCKETS = (30, 60, 90, 120, 180, 240, 300, 360)  # seconds after creation
STUDY_BASELINE = 60  # random other liquid pools, for comparison


def _raw_vault(tx: dict, pool: str, mint: str) -> float | None:
    """Raw token units (or SOL for wSOL) the pool holds after `tx`, the units the scorer's fill price uses."""
    for b in (tx.get("meta") or {}).get("postTokenBalances") or []:
        if b.get("owner") == pool and b.get("mint") == mint:
            amt = b.get("uiTokenAmount") or {}
            return float(amt.get("uiAmount") or 0) if mint == ps.WSOL else float(amt.get("amount") or 0)
    return None


def _pull_timing(rows) -> dict:
    """When each rug's LP was pulled (seconds after creation), as a share of the whole group."""
    n = len(rows) or 1
    out, lo = {}, 0
    for hi in PULL_BUCKETS:
        k = sum(1 for r in rows if r["pulled_ts"] and lo <= r["pulled_ts"] - r["create_ts"] < hi)
        out[f"{lo}-{hi}s"] = f"{k} ({100 * k / n:.0f}%)"
        lo = hi
    late = sum(1 for r in rows if r["pulled_ts"] and r["pulled_ts"] - r["create_ts"] >= lo)
    drained = sum(1 for r in rows if r["ret"] is not None and r["ret"] <= -0.999 and not r["pulled_ts"])
    out["after 360s (counted as no rug at minute 6)"] = late
    out["drained, no pull seen"] = drained
    return out


def exit_study(store, helius, hours: float | None = None, max_credits: int = 10_000, now: float | None = None,
               seed: int = 7) -> dict:
    """When do the picks rug, and would an earlier exit have paid? Picks = each model's top 8% of the same pools
    (ranked in the window, as `compare`) plus the history model's live top5h/top2h flags, and a random baseline of
    other liquid pools. For each, the pool's state at minutes 1-5 (balances after its last tx by then; one
    Helius call each, saved in `watch_launch_path`) gives the return of selling then: vault price vs the scorer's
    fill price, 6% cost, -100% once the LP was pulled or the pool's SOL fell below 10% of its peak."""
    ensure_schema(store)
    store.execute(PATH_SCHEMA)
    now = now if now is not None else time.time()
    since = now - hours * 3600 if hours else 0
    rows = store.query("SELECT pool, token_mint, create_ts, score, score_h, top2h, top5h, fill_px, fill_sol, "
                       "peak_sol, ret, pulled_ts FROM watch_launch_live WHERE status='done' AND liquid=1 "
                       "AND ret IS NOT NULL AND score IS NOT NULL AND score_h IS NOT NULL AND fill_px IS NOT NULL "
                       "AND create_ts >= ? ORDER BY create_ts", (since,))
    if not rows:
        return {"pools": 0}
    k = max(1, round(len(rows) * 0.08))
    groups = {"old model top 8%": sorted(rows, key=lambda r: -r["score"])[:k],
              "history model top 8%": sorted(rows, key=lambda r: -r["score_h"])[:k],
              "history live top5h (launch4h_5's picks)": [r for r in rows if r["top5h"]],
              "history live top2h (launch4h_2's picks)": [r for r in rows if r["top2h"]]}
    picked = {r["pool"] for g in groups.values() for r in g}
    rest = sorted((r for r in rows if r["pool"] not in picked),
                  key=lambda r: hashlib.sha1(f"study:{r['pool']}".encode()).hexdigest())
    groups["random other liquid pools"] = rest[:STUDY_BASELINE]
    # fetch the missing minute states, cheapest first (nothing is fetched twice)
    have = {(r["pool"], r["sec"]): r for r in store.query("SELECT pool, sec, sol, tok FROM watch_launch_path")}
    need = {r["pool"]: r for g in groups.values() for r in g}
    stats = collections.Counter()
    start_credits = helius.credits_used
    for r in need.values():
        for m in EXIT_MINUTES:
            sec = 60 * m
            if (r["pool"], sec) in have:
                continue
            if r["pulled_ts"] and r["pulled_ts"] <= r["create_ts"] + sec:
                continue  # pulled by then: -100%, no call needed
            if helius.credits_used - start_credits + 10 > max_credits:
                stats["left for a rerun (credit cap)"] += 1
                continue
            tx = helius.last_before(r["pool"], r["create_ts"] + sec)
            if tx is False:
                stats["call failed"] += 1
                continue
            sol = _raw_vault(tx, r["pool"], ps.WSOL) if tx else None
            tok = _raw_vault(tx, r["pool"], r["token_mint"]) if tx else None
            store.execute("INSERT OR REPLACE INTO watch_launch_path(pool, sec, sol, tok, ts) VALUES (?,?,?,?,?)",
                          (r["pool"], sec, sol, tok, time.time()))
            have[(r["pool"], sec)] = {"sol": sol, "tok": tok}
            stats["fetched"] += 1
    stats["credits_used"] = helius.credits_used - start_credits
    rng = np.random.default_rng(seed)

    def state_at(r, sec, peak):
        """(return if sold then, pool SOL / SOL at the fill, gone: 'pulled' | 'drained' | None, new peak)."""
        if r["pulled_ts"] and r["pulled_ts"] <= r["create_ts"] + sec:
            return -1.0, 0.0, "pulled", peak
        st = have.get((r["pool"], sec))
        if st is None:
            return None, None, None, peak
        sol, tok = st["sol"], st["tok"]
        if not sol or not tok:
            return -1.0, 0.0, "drained", peak
        peak = max(peak, sol)
        share = sol / r["fill_sol"] if r["fill_sol"] else None
        if peak > 0 and sol < la.PULLED_SHARE * peak:
            return -1.0, share, "drained", peak
        return min((sol / tok) / r["fill_px"] - 1 - la.COST, 3.0), share, None, peak

    out = {"pools": len(rows), "fetch": dict(stats), "groups": {}}
    for name, g in groups.items():
        by_min = {}
        rets = {m: [] for m in EXIT_MINUTES}
        liq = {m: [] for m in EXIT_MINUTES}
        for r in g:
            peak = max(r["peak_sol"] or 0.0, r["fill_sol"] or 0.0)
            for m in EXIT_MINUTES:
                x, share, gone, peak = state_at(r, 60 * m, peak)
                if x is not None:
                    rets[m].append({"ret": x})
                    liq[m].append((share, gone))
        for m in EXIT_MINUTES:
            by_min[f"sell at minute {m}"] = _group(rets[m], rng)
        by_min["sell at minute 6 (the books)"] = _group(g, rng)
        liquidity = {}
        for m in EXIT_MINUTES:
            xs = liq[m]
            if not xs:
                continue
            n = len(xs)
            shares = [s for s, gone in xs if s is not None and not gone]
            liquidity[f"minute {m}"] = {
                "LP pulled by now %": round(100 * sum(1 for _, gone in xs if gone == "pulled") / n),
                "drained without a pull seen %": round(100 * sum(1 for _, gone in xs if gone == "drained") / n),
                "pool SOL below half the fill's (still live) %": round(
                    100 * sum(1 for s in shares if s < 0.5) / n),
                "median pool SOL vs fill (still live)": round(float(np.median(shares)), 2) if shares else None}
        out["groups"][name] = {"n": len(g), "rug timing (LP pulls)": _pull_timing(g),
                               "liquidity by minute": liquidity, "returns by exit": by_min}
    out["note"] = ("ideal pool-level trades from the scorer's fill price; minute-m price = pool balances after its "
                   "last tx by then; rugs before the exit = -100%; pools whose minute state wasn't fetched are left "
                   "out of that minute only")
    return out


LATE_ENTRY_S = 90  # after the early pulls (EARLY_RUG_S)
LATE_EXITS_S = (180, 360, 600, 1200)
LATE_BASELINE = 60


def late_entry_study(store, helius, hours: float | None = None, max_credits: int = 12_000,
                     entry_s: int = LATE_ENTRY_S, exits: tuple = LATE_EXITS_S, now: float | None = None,
                     seed: int = 7) -> dict:
    """11-04 user idea: the history model's lowest deciles (D1-D2) hold the most big wins and the most early rugs.
    Enter them at `entry_s` (only pools not pulled or drained by then) and sell at each of `exits`. Prices = pool
    balances after its last tx by that second (one Helius call each, cached in `watch_launch_path`); 6% cost,
    capped +300%; -100% once the LP was pulled or the pool's SOL fell below 10% of the highest seen since entry
    (or of the fill's). Compared with a random sample of the other liquid pools (D3-D10) under the same rule."""
    ensure_schema(store)
    store.execute(PATH_SCHEMA)
    now = now if now is not None else time.time()
    since = now - hours * 3600 if hours else 0
    rows = store.query("SELECT pool, token_mint, create_ts, score_h, fill_px, fill_sol, ret, pulled_ts "
                       "FROM watch_launch_live WHERE status='done' AND liquid=1 AND ret IS NOT NULL "
                       "AND score_h IS NOT NULL AND fill_px IS NOT NULL AND fill_sol IS NOT NULL "
                       "AND create_ts >= ? ORDER BY create_ts", (since,))
    if not rows:
        return {"pools": 0}
    from tradingbot.model_book import decile_index
    idx = dict(zip((r["pool"] for r in rows), decile_index([r["score_h"] for r in rows])))
    low = [r for r in rows if idx[r["pool"]] <= 1]
    rest = sorted((r for r in rows if idx[r["pool"]] > 1),
                  key=lambda r: hashlib.sha1(f"late:{r['pool']}".encode()).hexdigest())[:LATE_BASELINE]
    groups = {"history D1 (lowest 10%)": [r for r in low if idx[r["pool"]] == 0],
              "history D2": [r for r in low if idx[r["pool"]] == 1],
              "history D1-D2": low, f"random {LATE_BASELINE} of D3-D10": rest}
    pulled_by = lambda r, sec: bool(r["pulled_ts"]) and r["pulled_ts"] <= r["create_ts"] + sec
    have = {(r["pool"], r["sec"]): r for r in store.query("SELECT pool, sec, sol, tok FROM watch_launch_path")}
    stats = collections.Counter()
    start = helius.credits_used

    def fetch(r, sec):
        if (r["pool"], sec) in have:
            return have[(r["pool"], sec)]
        if helius.credits_used - start + 10 > max_credits:
            stats["left for a rerun (credit cap)"] += 1
            return None
        tx = helius.last_before(r["pool"], r["create_ts"] + sec)
        if tx is False:
            stats["call failed"] += 1
            return None
        st = {"sol": _raw_vault(tx, r["pool"], ps.WSOL) if tx else None,
              "tok": _raw_vault(tx, r["pool"], r["token_mint"]) if tx else None}
        store.execute("INSERT OR REPLACE INTO watch_launch_path(pool, sec, sol, tok, ts) VALUES (?,?,?,?,?)",
                      (r["pool"], sec, st["sol"], st["tok"], time.time()))
        have[(r["pool"], sec)] = st
        stats["fetched"] += 1
        return st

    def trade(r):
        """None = not fetched; {'skipped': why} = no entry; else entry move and the return at each exit."""
        if pulled_by(r, entry_s):
            return {"skipped": "pulled before entry"}
        st = fetch(r, entry_s)
        if st is None:
            return None
        if not st["sol"] or not st["tok"] or st["sol"] < la.PULLED_SHARE * r["fill_sol"]:
            return {"skipped": "drained before entry"}
        entry, peak, out, dead = st["sol"] / st["tok"], st["sol"], {}, False
        for sec in exits:
            if dead or pulled_by(r, sec):
                out[sec], dead = -1.0, True
                continue
            s = fetch(r, sec)
            if s is None:
                break
            if not s["sol"] or not s["tok"] or s["sol"] < la.PULLED_SHARE * peak:
                out[sec], dead = -1.0, True
                continue
            peak = max(peak, s["sol"])
            out[sec] = min((s["sol"] / s["tok"]) / entry - 1 - la.COST, 3.0)
        return {"before": entry / r["fill_px"] - 1, "rets": out}

    trades = {}
    for r in low + rest:  # the low deciles first, so a credit cap cuts the baseline
        trades[r["pool"]] = trade(r)
    stats["credits_used"] = helius.credits_used - start
    rng = np.random.default_rng(seed)

    def summary(rets):
        g = _group([{"ret": x} for x in rets], rng)
        if rets:
            g["big_win_pct"] = round(100 * sum(x >= 0.5 for x in rets) / len(rets))
        return g

    out = {"pools": len(rows), "entry_s": entry_s, "fetch": dict(stats), "groups": {}}
    for name, g in groups.items():
        done = [(r, trades[r["pool"]]) for r in g if trades.get(r["pool"]) is not None]
        entered = [(r, t) for r, t in done if "rets" in t]
        skips = collections.Counter(t["skipped"] for _, t in done if "skipped" in t)
        big = [r for r, _ in entered if r["ret"] >= 0.5]
        out["groups"][name] = {
            "n": len(g), "not fetched (credit cap)": len(g) - len(done), "entered": len(entered), **skips,
            "from second 6 to minute 6 (all pools, the scorer's trade)": summary([r["ret"] for r, _ in done]),
            "from second 6 to minute 6 (entered pools only)": summary([r["ret"] for r, _ in entered]),
            f"big winners kept (second 6 -> minute 6 >= +50%, entered at {entry_s} s)": len(big),
            f"median move second 6 -> {entry_s} s (entered)": round(100 * float(np.median(
                [t["before"] for _, t in entered])), 1) if entered else None,
            **{f"enter {entry_s} s, sell at {sec // 60} min" if sec % 60 == 0 else f"sell at {sec} s":
               summary([t["rets"][sec] for _, t in entered if sec in t["rets"]]) for sec in exits}}
    out["note"] = (f"history-model deciles ranked within the window; entry at second {entry_s} at the pool's price "
                   "then (no latency or impact), 6% round trip, capped +300%; -100% once pulled or drained "
                   "(SOL < 10% of the highest seen since entry); pulls after minute 6 are only seen as drains")
    return out


PULL_BLOCK_H = 6  # walk-forward test blocks
PULL_MIN_TRAIN = 300
PULL_SKIPS = (10, 20, 30, 50)  # % of pools skipped as most likely to be pulled early
PULL_RANDOM_DRAWS = 500


def _op_features(prior: list, create_ts: float) -> dict:
    """The creator's record by the decision (create + DECIDE_S) from its earlier pools in this table (dicts with
    create_ts, pulled_ts): how many, how many pulled within EARLY_RUG_S (only pulls seen by then), how recent."""
    now = create_ts + DECIDE_S
    prior = [p for p in prior if p["create_ts"] < create_ts]
    pulled = [p for p in prior if p["pulled_ts"] and p["pulled_ts"] <= now]
    early = [p for p in pulled if p["pulled_ts"] - p["create_ts"] < EARLY_RUG_S]
    last = max((p["create_ts"] for p in prior), default=None)
    return {"op_prior": math.log1p(len(prior)), "op_early_pulls": math.log1p(len(early)),
            "op_early_share": len(early) / len(prior) if prior else 0.0,
            "op_pull_share": len(pulled) / len(prior) if prior else 0.0,
            "op_min_since_last": math.log1p((create_ts - last) / 60) if last else 0.0}


def _creator_history(rows: list[dict]) -> dict:
    """`_op_features` for every row, from the other rows of the same creator."""
    by_creator = collections.defaultdict(list)
    for r in rows:
        if r["creator"]:
            by_creator[r["creator"]].append(r)
    return {r["pool"]: _op_features(by_creator.get(r["creator"], []), r["create_ts"]) for r in rows}


PULL_DROP = tuple(la.LIVE_DROP) + CALENDAR


def pull_inputs(feat: dict, op: dict) -> dict:
    """The early-pull model's inputs: the scorer's second-4 features (no names, no hour/weekend) + creator record."""
    return {**{k: v for k, v in feat.items() if k not in PULL_DROP}, **op}


def _pull_rows(store, since: float = 0.0) -> list[dict]:
    """Finished liquid scorer rows with the early-pull inputs (`x`) and label (`y`: LP pulled < EARLY_RUG_S)."""
    rows = [dict(r) for r in store.query(
        "SELECT pool, creator, create_ts, feat, score_h, ret, pulled_ts FROM watch_launch_live "
        "WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND feat IS NOT NULL AND create_ts >= ? "
        "ORDER BY create_ts", (since,))]
    hist = _creator_history(rows)
    for r in rows:
        r["x"] = pull_inputs(json.loads(r["feat"]), hist[r["pool"]])
        r["y"] = float(bool(r["pulled_ts"]) and r["pulled_ts"] - r["create_ts"] < EARLY_RUG_S)
    return rows


def early_pull_study(store, hours: float | None = None, now: float | None = None, seed: int = 7) -> dict:
    """11-04: entering the history model's D1-D2 at 90 s missed their big moves, but the D1-D2 pools that
    survived 90 s made +23.5% from second 6 (hindsight). Can second-4 data spot the early pullers? A walk-forward
    logistic (expanding window, PULL_BLOCK_H test blocks) on the scorer's own features + the creator's history in
    this table, label = LP pulled < EARLY_RUG_S after creation. Reports the AUC, and the scorer's second-6 trade
    with the riskiest k% skipped vs the same share skipped at random (the check an 11-01 filter failed)."""
    ensure_schema(store)
    now = now if now is not None else time.time()
    rows = _pull_rows(store, now - hours * 3600 if hours else 0)
    if len(rows) < PULL_MIN_TRAIN + 50:
        return {"pools": len(rows), "note": f"needs {PULL_MIN_TRAIN + 50}+ finished liquid pools"}
    rng = np.random.default_rng(seed)
    start = rows[0]["create_ts"]
    block = PULL_BLOCK_H * 3600
    tested, blocks = [], []
    t = start + block
    while t < rows[-1]["create_ts"]:
        train = [r for r in rows if r["create_ts"] < t]
        test = [r for r in rows if t <= r["create_ts"] < t + block]
        t += block
        if len(train) < PULL_MIN_TRAIN or not test:
            continue
        y = np.array([r["y"] for r in train])
        if y.min() == y.max():
            continue
        names = sorted({k for r in train for k in r["x"]})
        xs, med = wa._matrix([r["x"] for r in train], names)
        mu, sd = xs.mean(0), xs.std(0) + 1e-9
        z = lambda m: np.clip((m - mu) / sd, -la.CLIP_Z, la.CLIP_Z)
        w, b = wa.logistic(z(xs), y)
        p = z(wa._matrix([r["x"] for r in test], names, med)[0]) @ w + b
        # history-model deciles within the block (the late-entry study ranked within its window)
        dec = None
        if all(r["score_h"] is not None for r in test):
            from tradingbot.model_book import decile_index
            dec = decile_index([r["score_h"] for r in test])
        for i, (r, s) in enumerate(zip(test, p)):
            tested.append({"risk": float(s), "y": r["y"], "ret": r["ret"],
                           "low": dec is not None and dec[i] <= 1, "block": len(blocks)})
        blocks.append({"from": datetime.fromtimestamp(t - block, timezone.utc).strftime("%m-%d %H:%M"),
                       "train": len(train), "test": len(test),
                       "auc": wa.auc(np.array(p), np.array([r["y"] for r in test]))})
        top = sorted(zip(w, names), key=lambda x: -abs(x[0]))[:8]
        blocks[-1]["top_weights"] = {n: round(float(v), 2) for v, n in top}
    if not tested:
        return {"pools": len(rows), "note": "no block had enough training rows"}

    def ev(group: list[dict]) -> dict:
        if not group:
            return {"n": 0}
        rets = np.array([g["ret"] for g in group])
        ys = np.array([g["y"] for g in group])
        out = {"n": len(group), "early_pull_pct": round(100 * ys.mean(), 1),
               "auc": wa.auc(np.array([g["risk"] for g in group]), ys),
               "keep all": {"mean_pct": round(100 * rets.mean(), 1), "rugged_pct": round(100 * (rets <= -0.999).mean()),
                            "big_win_pct": round(100 * (rets >= 0.5).mean())}}
        # skip within each block, so a block whose scores run high isn't skipped wholesale
        by_block = collections.defaultdict(list)
        for i, g in enumerate(group):
            by_block[g["block"]].append(i)
        for k in PULL_SKIPS:
            keep_idx = []
            for idx in by_block.values():
                idx = sorted(idx, key=lambda i: group[i]["risk"])
                keep_idx += idx[:len(idx) - round(len(idx) * k / 100)]
            kept = rets[keep_idx]
            skipped = np.setdiff1d(np.arange(len(group)), keep_idx)
            rand = []
            for _ in range(PULL_RANDOM_DRAWS):
                pick = []
                for idx in by_block.values():
                    pick += list(rng.choice(idx, len(idx) - round(len(idx) * k / 100), replace=False))
                rand.append(rets[pick].mean())
            rand = np.array(rand)
            out[f"skip riskiest {k}%"] = {
                "kept": len(kept), "mean_pct": round(100 * kept.mean(), 1),
                "rugged_pct": round(100 * (kept <= -0.999).mean()),
                "early_pulls_left_pct": round(100 * ys[keep_idx].mean(), 1),
                "big_win_pct": round(100 * (kept >= 0.5).mean()),
                "skipped_that_were_early_pulls_pct": round(100 * ys[skipped].mean(), 1) if len(skipped) else None,
                "skipped_big_winners": int((rets[skipped] >= 0.5).sum()),
                "random_skip_mean_pct": round(100 * rand.mean(), 1),
                "random_skip_90pct_range": [round(100 * float(np.percentile(rand, 5)), 1),
                                            round(100 * float(np.percentile(rand, 95)), 1)],
                "beats_random_share_pct": round(100 * (rand < kept.mean()).mean())}
        return out

    return {"pools": len(rows), "tested": len(tested), "blocks": blocks,
            "every liquid pool (test blocks)": ev(tested),
            "history D1-D2 (test blocks)": ev([g for g in tested if g["low"]]),
            "note": (f"label = LP pulled < {EARLY_RUG_S} s after creation; walk-forward, expanding window, "
                     f"{PULL_BLOCK_H} h test blocks, min {PULL_MIN_TRAIN} training pools; features = the live "
                     "scorer's second-4 features (no names, no hour/weekend) + the creator's earlier pools in this "
                     "table (early pulls seen by the decision); trade = the scorer's second-6 to minute-6 return, "
                     "6% cost; skips made within each block; beats_random_share = share of random skips of the "
                     "same size that did worse")}


TOKEN_AGE_SCHEMA = "CREATE TABLE IF NOT EXISTS watch_token_age(mint TEXT PRIMARY KEY, first_ts REAL, ts REAL)"
FRESH_TOKEN_S = 600  # minted less than 10 minutes before its pool = a new token launched with this pool
AGE_BUCKETS = ((FRESH_TOKEN_S, "fresh (minted < 10 min before the pool)"), (3600, "10 min - 1 h old"),
               (86400, "1 h - 1 day old"), (float("inf"), "over 1 day old"))
AGE_BASELINE = 150


def _age_bucket(age_s: float | None) -> str:
    if age_s is None:
        return "unknown"
    return next(name for hi, name in AGE_BUCKETS if age_s < hi)


def token_age_study(store, helius, hours: float | None = None, max_credits: int = 8_000, now: float | None = None,
                    seed: int = 7) -> dict:
    """11-04: some of the low-decile books' buys were tokens with ~1,050 holders at pool second 6, i.e. new
    hand-made pools of tokens that already trade elsewhere (Jupiter then routes through their older pools, and the
    backtest's vault prices measure a different trade). How many of the history model's D1-D2 pools are these, and
    how did they do vs genuinely new tokens? Token age = pool creation minus the mint's oldest transaction (one
    Helius call per mint, cached in `watch_token_age`); D1-D2 fetched first, then a random AGE_BASELINE of D3-D10."""
    ensure_schema(store)
    store.execute(TOKEN_AGE_SCHEMA)
    now = now if now is not None else time.time()
    since = now - hours * 3600 if hours else 0
    rows = [dict(r) for r in store.query(
        "SELECT pool, token_mint, create_ts, score_h, ret, pulled_ts FROM watch_launch_live WHERE status='done' "
        "AND liquid=1 AND ret IS NOT NULL AND score_h IS NOT NULL AND token_mint IS NOT NULL AND create_ts >= ? "
        "ORDER BY create_ts", (since,))]
    if not rows:
        return {"pools": 0}
    from tradingbot.model_book import BIG_WIN, decile_index
    idx = dict(zip((r["pool"] for r in rows), decile_index([r["score_h"] for r in rows])))
    low = [r for r in rows if idx[r["pool"]] <= 1]
    rest = sorted((r for r in rows if idx[r["pool"]] > 1),
                  key=lambda r: hashlib.sha1(f"age:{r['pool']}".encode()).hexdigest())[:AGE_BASELINE]
    have = {r["mint"]: r["first_ts"] for r in store.query("SELECT mint, first_ts FROM watch_token_age")}
    stats = collections.Counter()
    start = helius.credits_used
    for r in low + rest:
        m = r["token_mint"]
        if m in have:
            continue
        if helius.credits_used - start + 10 > max_credits:
            stats["left for a rerun (credit cap)"] += 1
            continue
        t = helius.first_tx_time(m)
        if t is False:
            stats["call failed"] += 1
            continue
        store.execute("INSERT OR REPLACE INTO watch_token_age(mint, first_ts, ts) VALUES (?,?,?)", (m, t, time.time()))
        have[m] = t
        stats["fetched"] += 1
    stats["credits_used"] = helius.credits_used - start
    rng = np.random.default_rng(seed)
    for r in low + rest:
        first = have.get(r["token_mint"], "missing")
        r["age_s"] = None if first in (None, "missing") else max(0.0, r["create_ts"] - first)
        r["bucket"] = "not fetched" if first == "missing" else _age_bucket(r["age_s"])

    def group(rs):
        if not rs:
            return {"n": 0}
        return {**_group(rs, rng), **_rug_split(rs),
                "big_win_pct": round(100 * sum(r["ret"] >= BIG_WIN for r in rs) / len(rs))}

    def split(rs):
        buckets = [name for _, name in AGE_BUCKETS] + ["unknown", "not fetched"]
        n = len(rs) or 1
        return {"all": group(rs),
                "share by token age (%)": {b: round(100 * sum(r["bucket"] == b for r in rs) / n, 1) for b in buckets
                                           if any(r["bucket"] == b for r in rs)},
                **{b: group([r for r in rs if r["bucket"] == b]) for b in buckets
                   if any(r["bucket"] == b for r in rs)},
                "median token age at pool creation (h)": round(float(np.median(
                    [r["age_s"] for r in rs if r["age_s"] is not None])) / 3600, 2)
                if any(r["age_s"] is not None for r in rs) else None}

    return {"pools": len(rows), "fetch": dict(stats),
            "history D1-D2": split(low), f"random {AGE_BASELINE} of D3-D10": split(rest),
            "note": (f"token age = pool creation - the mint's oldest transaction; fresh = under {FRESH_TOKEN_S} s; "
                     "returns are the scorer's ideal second-6 to minute-6 trades on the new pool's vault prices, 6% "
                     "cost; deciles ranked within the window")}


def low_decile_trades(store, hours: float | None = None, now: float | None = None) -> dict:
    """Every finished trade of the history model's bottom 20% (D1-D2, ranked within the window as the studies
    do), one row each: the scorer's ideal second-6 to minute-6 return (6% cost), when the LP was pulled, and the
    early-pull risk percentile where the live scorer recorded one (pools decided after the 11-04 deploy)."""
    ensure_schema(store)
    now = now if now is not None else time.time()
    since = now - hours * 3600 if hours else 0
    rows = [dict(r) for r in store.query(
        "SELECT pool, token_mint, create_ts, score_h, ret, pulled_ts, risk_pct FROM watch_launch_live "
        "WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND score_h IS NOT NULL AND create_ts >= ? "
        "ORDER BY create_ts", (since,))]
    if not rows:
        return {"pools": 0, "trades": []}
    from tradingbot.model_book import decile_index
    idx = decile_index([r["score_h"] for r in rows])
    low = [r for r, i in zip(rows, idx) if i <= 1]
    trades = [{"created (UTC)": datetime.fromtimestamp(r["create_ts"], timezone.utc).strftime("%m-%d %H:%M:%S"),
               "token": r["token_mint"], "return_pct": round(100 * r["ret"], 1),
               "pulled_after_s": round(r["pulled_ts"] - r["create_ts"]) if r["pulled_ts"] else None,
               "risk_pct": round(r["risk_pct"]) if r["risk_pct"] is not None else None} for r in low]
    return {"pools": len(rows), "d1_d2_trades": len(trades),
            "summary": _group(low, np.random.default_rng(0)), "trades": trades}


def parity(store, rows) -> dict:
    """Live features vs the ones the backtest builds from the enricher's later launch row (same pool)."""
    diffs = []
    for r in rows:
        e = store.query("SELECT data FROM watch_enrich WHERE pool=? AND kind='launch'", (r["pool"],))
        if not e or not r["feat"]:
            continue
        d = json.loads(e[0]["data"])
        if not d.get("txs"):
            continue
        ref = la.launch_features(d["txs"], DECIDE_S, d.get("creator"), 0, d.get("seed_sol"))
        live = json.loads(r["feat"])
        keys = [k for k in ref if k in live and not k.endswith("prior_pools") and k != "repeat_creator"]
        diffs.append(max((abs(ref[k] - live[k]) for k in keys), default=0.0))
    if not diffs:
        return {"compared": 0}
    return {"compared": len(diffs), "identical": sum(d < 1e-6 for d in diffs),
            "median_max_diff": round(float(np.median(diffs)), 4), "worst_max_diff": round(float(max(diffs)), 4)}


def fill_gap(store, journal, book: str, since: float, rng) -> dict:
    """A book's real (Jupiter-simulated) trades vs the scorer's ideal trade on the same pools: vault price at
    second ~6.5 in, vault price at minute 6 out, flat 6% cost. The gap is what a real fill costs live."""
    pairs = []
    for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ? AND closed_at IS NOT NULL", (book, since)):
        r = store.query("SELECT create_ts, ret FROM watch_launch_live WHERE token_mint=? AND status='done' "
                        "ORDER BY notified_ts DESC LIMIT 1", (p["token"],))
        if not r or r[0]["ret"] is None or not p["cost_usd"]:
            continue
        pairs.append({"real": p["realized_usd"] / p["cost_usd"] - 1, "ideal": r[0]["ret"],
                      "in_s": p["opened_at"] - r[0]["create_ts"], "out_s": p["closed_at"] - r[0]["create_ts"]})
    # pools the book was signalled on but didn't buy (declined, failed, skipped): their ideal trades
    from tradingbot import model_book as mb
    flag = mb.LAUNCH_CUT.get(book)
    bought = {p["token"] for p in journal.query("SELECT token FROM positions WHERE strategy=? AND opened_at >= ?",
                                                (book, since - 600))}
    missed = []
    if flag:
        for r in store.query("SELECT token_mint, ret, top2, top5, top2h, top5h, low_h, risk_pct, rw_pct, rw_low_pct, rw_low6, rw24_pct, rw24_low_pct "
                             "FROM watch_launch_live "
                             "WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND create_ts >= ?", (since,)):
            hit = r[flag] if flag in ("top2", "top5", "top2h", "top5h") else {
                **low_flags(r["low_h"], r["risk_pct"]), **rw_flags(rw_low(r), r["rw_pct"], r["rw_low_pct"]),
                **rw24_flags(r["low_h"], r["rw24_pct"], r["rw24_low_pct"])}[flag]
            if hit and r["token_mint"] not in bought:
                missed.append({"ret": r["ret"]})
    if not pairs:
        return {"trades": 0, "signalled but not bought (ideal)": _group(missed, rng)}
    gaps = [x["real"] - x["ideal"] for x in pairs]
    return {"trades": len(pairs),
            "signalled but not bought (ideal)": _group(missed, rng),
            "book (real fills)": _group([{"ret": x["real"]} for x in pairs], rng),
            "scorer (ideal fills), same pools": _group([{"ret": x["ideal"]} for x in pairs], rng),
            "gap per trade (real - ideal)": {"mean_pct": round(100 * float(np.mean(gaps)), 1),
                                             "median_pct": round(100 * float(np.median(gaps)), 1)},
            "filled at pool second (median)": _median([x["in_s"] for x in pairs]),
            "sold at pool second (median)": _median([x["out_s"] for x in pairs])}


def liquidity(rows, hours: float) -> dict:
    """Pool liquidity at second 4 (2 x pool SOL x SOL price) of every liquid scored pool and of each pick group,
    and what it allows: buying $s into a pool with $L of liquidity moves the price ~2s/L each way, so profit per
    trade s x (edge - 4s/L) peaks at s = edge x L / 8 (profit edge^2 x L / 16). Shown at the backtest edge and half
    of it, for the median pick pool, with the picks per day seen in the window."""
    groups = {"every liquid pool": rows, "top 2%": [r for r in rows if r["top2"]],
              "top 5%": [r for r in rows if r["top5"]]}
    out = {}
    for g, rs in groups.items():
        liq = [r["liq_usd"] for r in rs if r["liq_usd"]]
        if not liq:
            out[g] = {"n": 0}
            continue
        q = lambda p: round(float(np.percentile(liq, p)))
        out[g] = {"n": len(liq), "p10": q(10), "median": q(50), "p90": q(90)}
        if g in BACKTEST_EDGE:
            med, per_day = float(np.median(liq)), len(rs) * 24 / hours
            out[g]["picks_per_day"] = round(per_day, 1)
            for label, e in (("backtest edge", BACKTEST_EDGE[g]), ("half edge", BACKTEST_EDGE[g] / 2)):
                profit = e * e * med / 16
                out[g][label] = {"edge_pct": round(100 * e, 1), "best_size_usd": round(e * med / 8),
                                 "profit_per_trade_usd": round(profit), "per_day_usd": round(profit * per_day)}
    return out


def drift(rows, cfg: Config, top: int = 10) -> dict:
    """Is the live scorer seeing what the model was trained on? Share of live liquid pools above each cut-off
    (should be ~2% / ~5%), live score quantiles next to the training ones, and the features whose live mean is
    furthest from the training mean, in training standard deviations (z), with how often each is present."""
    try:
        m = json.loads(model_path(cfg).read_text())
    except (OSError, ValueError):
        return {"model": None}
    scored = [r for r in rows if r["score"] is not None and r["feat"]]
    if not scored:
        return {"pools": 0}
    scores = np.array([r["score"] for r in scored])
    meta = m.get("meta", {})
    out = {"pools": len(scored),
           "share_above_cut2_pct": round(100 * float(np.mean(scores >= meta.get("cut2", np.inf) - 1e-9)), 1),
           "share_above_cut5_pct": round(100 * float(np.mean(scores >= meta.get("cut5", np.inf) - 1e-9)), 1),
           "live_score_quantiles": {str(q): round(float(np.quantile(scores, q / 100)), 3) for q in (10, 50, 90, 95, 98)},
           "training_score_quantiles": {k: round(v, 3) for k, v in (meta.get("score_quantiles") or {}).items()}}
    feats = [json.loads(r["feat"]) for r in scored]
    zs = []
    for name, mu, sd in zip(m["names"], m["mu"], m["sd"]):
        vals = [f[name] for f in feats if f.get(name) is not None]
        if not vals:
            zs.append((name, None, 0))
            continue
        zs.append((name, (float(np.mean(vals)) - mu) / sd, len(vals)))
    missing = [n for n, z, k in zs if z is None]
    ranked = sorted([x for x in zs if x[1] is not None], key=lambda x: -abs(x[1]))[:top]
    out["most_shifted_features"] = [{"feature": n, "z": round(z, 2), "present_pct": round(100 * k / len(feats))}
                                    for n, z, k in ranked]
    out["features_never_present_live"] = missing
    return out


def status(store, cfg: Config, hours: float = 24, now: float | None = None, journal=None) -> dict:
    now = now if now is not None else time.time()
    ensure_schema(store)
    rows = store.query("SELECT * FROM watch_launch_live WHERE notified_ts >= ?", (now - hours * 3600,))
    day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
    out: dict = {"hours": hours,
                 "websocket": json.loads(store.get_state("launch_live:ws") or "{}"),
                 "counts_today": json.loads(store.get_state(f"launch_live:counts:{day}") or "{}"),
                 "credits_today": float(store.get_state(f"spend:launch_live:{day}") or 0),
                 "pools": len(rows),
                 "status": dict(collections.Counter(r["status"] for r in rows)),
                 "skip_reasons": dict(collections.Counter(r["reason"] for r in rows if r["status"] in ("skipped", "lost"))
                                      .most_common(10))}
    decided = [r for r in rows if r["decided_ts"] and r["create_ts"]]
    out["timing_s"] = {"notified_after_creation": _median([r["notified_ts"] - r["create_ts"] for r in decided]),
                       "decided_after_creation": _median([r["decided_ts"] - r["create_ts"] for r in decided]),
                       "history_indexed_to": _median([r["indexed_s"] for r in decided]),
                       "fill_after_creation": _median([r["fill_ts"] - r["create_ts"] for r in decided if r["fill_ts"]])}
    scored = [r for r in rows if r["score"] is not None and r["liquid"]]
    out["scored_liquid"] = len(scored)
    out["note"] = ("outcomes below are pool-level (vault price at second ~6 vs minute 6, 6% cost); the books' own "
                   "Jupiter-simulated trades are in `report`")
    out["picks"] = {"top 2%": sum(r["top2"] or 0 for r in scored), "top 5%": sum(r["top5"] or 0 for r in scored),
                    "top 2% (history model)": sum(r["top2h"] or 0 for r in scored),
                    "top 5% (history model)": sum(r["top5h"] or 0 for r in scored)}
    out["liquidity at second 4 (USD) and trade size"] = liquidity(scored, hours)
    rng = np.random.default_rng(0)
    done = [r for r in rows if r["status"] == "done"]
    out["outcomes (fill at second ~6, out at minute 6, 6% cost)"] = {
        "every liquid pool": _group(done, rng),
        "top 2%": _group([r for r in done if r["top2"]], rng),
        "top 5%": _group([r for r in done if r["top5"]], rng),
        "top 2% (history model, rolling cut)": _group([r for r in done if r["top2h"]], rng),
        "top 5% (history model, rolling cut)": _group([r for r in done if r["top5h"]], rng),
        **{f"history bottom 20% ({'all' if k == 'd12' else 'skip riskiest ' + k[4:] + '%'})": _group(
            [r for r in done if low_flags(r["low_h"], r["risk_pct"])[k]], rng)
           for k in ("d12",) + tuple(f"d12s{s}" for s in PULL_SKIPS)},
        f"rug-vs-win: every liquid pool minus the riskiest {RW_SKIP}%": _group(
            [r for r in done if rw_flags(rw_low(r), r["rw_pct"], r["rw_low_pct"])["rw50"]], rng),
        f"rug-vs-win: history bottom 20% minus the riskiest {RW_SKIP}%": _group(
            [r for r in done if rw_flags(rw_low(r), r["rw_pct"], r["rw_low_pct"])["d12rw50"]], rng),
        f"rug-vs-win 24 h rule: every liquid pool minus the riskiest {RW_SKIP}%": _group(
            [r for r in done if rw24_flags(r["low_h"], r["rw24_pct"], r["rw24_low_pct"])["rw50_24"]], rng),
        f"rug-vs-win 24 h rule: history bottom 20% minus the riskiest {RW_SKIP}%": _group(
            [r for r in done if rw24_flags(r["low_h"], r["rw24_pct"], r["rw24_low_pct"])["d12rw50_24"]], rng)}
    out["feature parity with the backtest"] = parity(store, scored)
    out["live vs training (drift)"] = drift(scored, cfg)
    if journal is not None:  # the launch books' own buys: where the seconds from creation to the fill go
        from tradingbot import model_book as mb
        out["book_timing_s"] = mb.early_timing(journal, now - hours * 3600, tuple(mb.LAUNCH_CUT), mb.LAUNCH_STAGES)
        out["book_skips"] = {b: dict(collections.Counter(r["detail"] for r in journal.query(
            "SELECT detail FROM events WHERE kind='entry_skipped' AND strategy=? AND ts >= ?",
            (b, now - hours * 3600))).most_common(8)) for b in mb.LAUNCH_CUT}
        out["real vs ideal fills"] = {b: fill_gap(store, journal, b, now - hours * 3600, rng) for b in mb.LAUNCH_CUT}
    try:
        meta = json.loads(model_path(cfg).read_text()).get("meta", {})
        out["model"] = {k: meta.get(k) for k in ("trained_ts", "rows", "win_rate", "cut2", "cut5",
                                                 "top2_in_sample", "top5_in_sample")}
    except (OSError, ValueError):
        out["model"] = None
    return out


RW_CHECK_DRAWS = 2000
RW_CHECK_BLOCK_H = 6


def _rw_rules():
    """(name, score column, all-pools keep test, bottom-20% keep test) for both rug-vs-win rules; a test returns
    True = kept, False = skipped, None = not in that universe (unscored)."""
    keep = 100 - RW_SKIP

    def side(pct):
        return None if pct is None else pct < keep
    return (
        ("study rule (6 h)", "rw",
         lambda r: side(r["rw_pct"]),
         lambda r: side(r["rw_low_pct"]) if rw_low(r) else None,
         ("launchrw50", "launchd12rw50")),
        ("24 h rule", "rw24",
         lambda r: side(r["rw24_pct"]),
         lambda r: side(r["rw24_low_pct"]) if r["low_h"] else None,
         ("launchrw50_24h", "launchd12rw50_24h")))


def _keep_test(rows, kept_fn, score_col, rng) -> dict:
    """Is a rule's kept half better than chance? Kept vs skipped, kept vs every pool, kept vs random keeps of the
    same size from the same pools, per-block gaps, and how its score sorts rugs and big wins (AUC)."""
    from tradingbot.watch_analysis import auc
    pool = [r for r in rows if kept_fn(r) is not None]
    kept = [r for r in pool if kept_fn(r)]
    skipped = [r for r in pool if not kept_fn(r)]
    if len(kept) < 5 or len(skipped) < 5:
        return {"pools": len(pool), "note": "too few pools"}
    k = np.array([r["ret"] for r in kept])
    s = np.array([r["ret"] for r in skipped])
    a = np.array([r["ret"] for r in pool])
    gaps = [rng.choice(k, len(k)).mean() - rng.choice(s, len(s)).mean() for _ in range(RW_CHECK_DRAWS)]
    rand = np.array([rng.choice(a, len(k), replace=False).mean() for _ in range(RW_CHECK_DRAWS)])
    out = {"pools": len(pool), "every pool": _group(pool, rng), "kept": _group(kept, rng),
           "skipped": _group(skipped, rng),
           "kept minus skipped (pts)": {"mean": round(100 * float(k.mean() - s.mean()), 1),
                                        "ci90": [round(100 * float(np.quantile(gaps, q)), 1) for q in (0.05, 0.95)],
                                        "share of bootstrap draws > 0": round(float(np.mean(np.array(gaps) > 0)), 3)},
           "kept minus every pool (pts)": round(100 * float(k.mean() - a.mean()), 1),
           "random keeps of the same size": {"mean_pct": round(100 * float(rand.mean()), 1),
                                             "p90_pct": round(100 * float(np.quantile(rand, 0.9)), 1),
                                             "share worse than kept": round(float(np.mean(rand < k.mean())), 3)}}
    scored = [r for r in pool if r[score_col] is not None]
    if len(scored) >= 20:
        sc = np.array([r[score_col] for r in scored], dtype=float)
        rets = np.array([r["ret"] for r in scored])
        # score = P(rug) - P(big win): a working rule ranks rugs high (AUC > 0.5) and big wins low (< 0.5)
        out["AUC (score = riskiness)"] = {
            "rugs (want > 0.5)": _r3(auc(sc, (rets <= -0.999).astype(int))),
            "big wins +50% (want < 0.5)": _r3(auc(sc, (rets >= RW_BIG).astype(int))),
            "losers (want > 0.5)": _r3(auc(sc, (rets <= 0).astype(int)))}
    blocks: dict = {}
    for r in pool:
        b = int(r["create_ts"] // (RW_CHECK_BLOCK_H * 3600)) * RW_CHECK_BLOCK_H * 3600
        blocks.setdefault(b, []).append(r)
    rows_b = []
    for b in sorted(blocks):
        kb = [r["ret"] for r in blocks[b] if kept_fn(r)]
        sb = [r["ret"] for r in blocks[b] if not kept_fn(r)]
        rows_b.append({"block (UTC)": datetime.fromtimestamp(b, timezone.utc).strftime("%m-%d %H:00"),
                       "kept n": len(kb), "kept mean": _pct(kb), "skipped n": len(sb), "skipped mean": _pct(sb),
                       "gap": round(_pct(kb) - _pct(sb), 1) if kb and sb else None})
    out["by 6 h block"] = rows_b
    gaps_b = [x["gap"] for x in rows_b if x["gap"] is not None]
    out["blocks where kept beat skipped"] = f"{sum(g > 0 for g in gaps_b)} of {len(gaps_b)}"
    return out


def _pct(xs) -> float | None:
    return round(100 * float(np.mean(xs)), 1) if xs else None


def _r3(x):
    return None if x is None else round(x, 3)


def rw_check(store, journal=None, hours: float | None = None, now: float | None = None) -> dict:
    """Is the rug-vs-win books' lead real? For each rule (study 6 h, earlier 24 h) and universe (every liquid pool,
    history bottom 20%), on the scorer's finished liquid pools: kept vs skipped, vs every pool and vs random keeps
    of the same size, per 6 h block, AUCs; plus the books' real fills vs ideal and launch4_2's top 2% for scale.
    `hours` = the recent window (e.g. overnight); the whole life of the scores is shown too."""
    now = now if now is not None else time.time()
    ensure_schema(store)
    rows = store.query("SELECT * FROM watch_launch_live WHERE status='done' AND liquid=1 AND ret IS NOT NULL "
                       "ORDER BY create_ts")
    windows = {"whole life": rows}
    if hours:
        windows[f"last {hours:g} h"] = [r for r in rows if r["create_ts"] >= now - hours * 3600]
    out: dict = {}
    for wname, wrows in windows.items():
        rng = np.random.default_rng(0)
        w: dict = {"pools": len(wrows),
                   "from": datetime.fromtimestamp(wrows[0]["create_ts"], timezone.utc).isoformat() if wrows else None,
                   "every liquid pool": _group(wrows, rng),
                   "launch4_2 top 2% (scale)": _group([r for r in wrows if r["top2"]], rng)}
        for name, col, all_fn, low_fn, books in _rw_rules():
            w[name] = {"every liquid pool minus the riskiest half": _keep_test(wrows, all_fn, col, rng),
                       "history bottom 20% minus the riskiest half": _keep_test(wrows, low_fn, col, rng)}
            if journal is not None and wrows:
                w[name]["real fills"] = {b: fill_gap(store, journal, b, wrows[0]["create_ts"], rng) for b in books}
        out[wname] = w
    return out


EARLY_STUCK_S = 1800  # an early-book position still open this long after its buy can't have sold: counts as -100%


def _rw_lookup(store) -> tuple[dict, dict]:
    """The scorer's rug-vs-win scores by pool and by token mint (mint -> rows sorted by creation)."""
    by_pool, by_mint = {}, {}
    for r in store.query("SELECT pool, token_mint, create_ts, rw, rw_pct, rw24, rw24_pct FROM watch_launch_live "
                         "WHERE rw_pct IS NOT NULL OR rw24_pct IS NOT NULL ORDER BY create_ts"):
        d = dict(r)
        by_pool[d["pool"]] = d
        if d["token_mint"]:
            by_mint.setdefault(d["token_mint"], []).append(d)
    return by_pool, by_mint


def early_rw_check(store, journal, cfg, hours: float | None = None, now: float | None = None) -> dict:
    """Would the rug-vs-win filter help the early books? Every pool the early books could buy (hand-made, seen
    within 4 min, liquid at first sight; outcome = 5-minute snapshot vs first sight, 6% round trip, emptied = -100%)
    and the early books' real trades, each matched to the launch scorer's second-4 rug-vs-win score for the same
    pool (trades by token), then split kept / skipped by each rule (riskiest half skipped) as rw-check does."""
    from tradingbot import model_book as mb
    now = now if now is not None else time.time()
    ensure_schema(store)
    by_pool, by_mint = _rw_lookup(store)
    since = now - hours * 3600 if hours else 0.0
    path = mb.early_model_path(cfg)
    model = mb.Model.from_json(json.loads(path.read_text())) if path.exists() else None
    cands = [c for c in mb.early_candidates(store, cfg, model, since, now) if c["ret"] is not None]
    pools = []
    for c in cands:
        s = by_pool.get(c["pool"])
        pools.append({"ret": c["ret"], "create_ts": c["seen_ts"], "score": c["score"],
                      **({k: s[k] for k in ("rw", "rw_pct", "rw24", "rw24_pct")} if s else
                         {"rw": None, "rw_pct": None, "rw24": None, "rw24_pct": None})})
    picks = [p for p in pools if model is not None and p["score"] is not None and p["score"] >= model.threshold]

    trades: dict = {}
    for book in mb.EARLY_BOOKS:
        rows = []
        for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                               "WHERE strategy=? AND opened_at >= ? ORDER BY opened_at", (book, since)):
            if p["closed_at"] and p["cost_usd"]:
                ret = p["realized_usd"] / p["cost_usd"] - 1
            elif not p["closed_at"] and now - p["opened_at"] >= EARLY_STUCK_S:
                ret = -1.0  # never sold: worth $0
            else:
                continue
            match = [s for s in by_mint.get(p["token"], []) if s["create_ts"] <= p["opened_at"]]
            s = match[-1] if match else None
            rows.append({"ret": ret, "create_ts": p["opened_at"],
                         **({k: s[k] for k in ("rw", "rw_pct", "rw24", "rw24_pct")} if s else
                            {"rw": None, "rw_pct": None, "rw24": None, "rw24_pct": None})})
        if rows:
            trades[book] = rows

    keep = 100 - RW_SKIP
    rules = (("study rule (6 h)", "rw", lambda r: None if r["rw_pct"] is None else r["rw_pct"] < keep),
             ("24 h rule", "rw24", lambda r: None if r["rw24_pct"] is None else r["rw24_pct"] < keep))

    def section(rows):
        rng = np.random.default_rng(0)
        scored = [r for r in rows if r["rw_pct"] is not None or r["rw24_pct"] is not None]
        out = {"n": len(rows), "with a rug-vs-win score": len(scored), "all": _group(rows, rng),
               "unscored (not liquid at second 4, or missed by the scorer)":
                   _group([r for r in rows if r["rw_pct"] is None and r["rw24_pct"] is None], rng)}
        for name, col, fn in rules:
            out[name] = _keep_test(rows, fn, col, rng)
        return out

    return {"window": f"last {hours:g} h" if hours else "whole life",
            "every eligible early pool (snapshot yardstick)": section(pools),
            "early model picks (fixed cut-off, snapshot yardstick)": section(picks),
            "early books' real trades": {b: section(r) for b, r in trades.items()},
            "note": f"kept = rug-vs-win percentile < {keep} (the riskiest half skipped); pool outcomes are the "
                    "5-minute snapshot (~pool minute 6) vs first sight; trades still open "
                    f"{EARLY_STUCK_S // 60} min after the buy count as -100%"}


# --- fill-gap study (11-06): where do real fills fall short of the scorer's ideal trade? -------------------------
GAP_BOOKS = ("launchrw50", "launchrw50_24h")
GAP_NEVER_SOLD_S = 1800  # still open this long after the buy: never sold, counts as -100%
GAP_WORST = 25


def _gap_bucket(rows, key, edges) -> list:
    """Trades grouped by `key` into `edges` bands: n, mean / median gap, and each band's share of the total gap."""
    total = sum(r["gap"] for r in rows) or 1e-9
    out = []
    for name, lo, hi in edges:
        g = [r for r in rows if r[key] is not None and lo <= r[key] < hi]
        if not g:
            continue
        gaps = [r["gap"] for r in g]
        out.append({"band": name, "n": len(g), "mean_gap_pct": _pct(gaps),
                    "median_gap_pct": round(100 * float(np.median(gaps)), 1),
                    "share_of_total_gap_pct": round(100 * sum(gaps) / total)})
    return out


def fill_gap_study(store, journal, books=GAP_BOOKS, hours: float | None = None, now: float | None = None) -> dict:
    """Each launch-book trade (closed, or still open GAP_NEVER_SOLD_S after the buy = -100%) against the scorer's
    ideal trade on the same pool (vault price at second ~6.5 in, minute 6 out, flat 6% cost, capped +300%, rug =
    -100%). Splits the gap by what the ideal trade did, when the book filled and sold, pool liquidity, exit reason,
    swap-only return vs fees, and lists the worst trades."""
    now = now if now is not None else time.time()
    ensure_schema(store)
    since = now - hours * 3600 if hours else 0.0
    out: dict = {"window": f"last {hours:g} h" if hours else "whole life"}
    for book in books:
        rows = []
        for p in journal.query("SELECT id, token, symbol, opened_at, closed_at, cost_usd, realized_usd, exit_reason "
                               "FROM positions WHERE strategy=? AND opened_at >= ?", (book, since)):
            if not p["cost_usd"]:
                continue
            if p["closed_at"]:
                real = p["realized_usd"] / p["cost_usd"] - 1
            elif now - p["opened_at"] >= GAP_NEVER_SOLD_S:
                real = -1.0
            else:
                continue
            s = store.query("SELECT create_ts, ret, liq_usd, fill_ts, exit_ts, pulled_ts FROM watch_launch_live "
                            "WHERE token_mint=? AND status='done' AND create_ts <= ? ORDER BY create_ts DESC LIMIT 1",
                            (p["token"], p["opened_at"]))
            if not s or s[0]["ret"] is None:
                continue
            s = s[0]
            fills = journal.query("SELECT side, price, usd, fee_usd FROM fills WHERE position_id=? ORDER BY ts",
                                  (p["id"],))
            buys = [f for f in fills if f["side"] == "buy"]
            sells = [f for f in fills if f["side"] == "sell" and (f["usd"] or 0) > 0]
            sold = sum(f["usd"] for f in sells)
            swap = (sum(f["price"] * f["usd"] for f in sells) / sold / buys[0]["price"] - 1
                    if buys and buys[0]["price"] and sold > 0 else None)
            ideal = s["ret"]
            capped, rugged = ideal >= 2.999, ideal <= -0.999
            rows.append({
                "token": p["token"], "symbol": p["symbol"], "real": real, "ideal": ideal, "gap": real - ideal,
                "in_s": p["opened_at"] - s["create_ts"],
                "out_s": (p["closed_at"] - s["create_ts"]) if p["closed_at"] else None,
                "liq_usd": s["liq_usd"], "exit": p["exit_reason"] if p["closed_at"] else "never sold",
                "fee_pct": sum((f["fee_usd"] or 0) for f in fills) / p["cost_usd"],
                "swap": swap, "capped": capped, "rugged": rugged,
                # the scorer's gross move (its 6% cost added back) vs the book's swap-only move: price, not fees
                "swap_gap": (swap - (ideal + 0.06)) if swap is not None and not capped and not rugged else None,
                "pulled_before_sale": bool(s["pulled_ts"] and p["closed_at"] and s["pulled_ts"] < p["closed_at"]),
            })
        if not rows:
            out[book] = {"trades": 0}
            continue
        gaps = np.array([r["gap"] for r in rows])
        sg = [r["swap_gap"] for r in rows if r["swap_gap"] is not None]
        ideal_band = lambda r: ("ideal rugged (-100%)" if r["rugged"] else "ideal capped (+300%)" if r["capped"]
                                else "ideal < 0" if r["ideal"] < 0 else "ideal 0..+50%" if r["ideal"] < 0.5
                                else "ideal +50%..+300%")
        for r in rows:
            r["ideal_band"] = ideal_band(r)
        by_ideal = []
        total = float(gaps.sum()) or 1e-9
        for band in ("ideal rugged (-100%)", "ideal < 0", "ideal 0..+50%", "ideal +50%..+300%", "ideal capped (+300%)"):
            g = [r["gap"] for r in rows if r["ideal_band"] == band]
            if g:
                by_ideal.append({"band": band, "n": len(g), "mean_gap_pct": _pct(g),
                                 "median_gap_pct": round(100 * float(np.median(g)), 1),
                                 "share_of_total_gap_pct": round(100 * sum(g) / total)})
        by_exit: dict = {}
        for r in rows:
            by_exit.setdefault(r["exit"] or "unknown", []).append(r["gap"])
        out[book] = {
            "trades": len(rows),
            "real mean / median %": [_pct(list(r["real"] for r in rows)),
                                      round(100 * float(np.median([r["real"] for r in rows])), 1)],
            "ideal mean / median %": [_pct(list(r["ideal"] for r in rows)),
                                       round(100 * float(np.median([r["ideal"] for r in rows])), 1)],
            "gap mean / median (pts)": [round(100 * float(gaps.mean()), 1), round(100 * float(np.median(gaps)), 1)],
            "gap mean without the worst 2% of trades": round(100 * float(np.sort(gaps)[max(1, len(gaps) // 50):].mean()), 1),
            "fees mean % of cost": _pct([r["fee_pct"] for r in rows]),
            "price-only gap (swap move vs scorer's gross move), mean / median": (
                [round(100 * float(np.mean(sg)), 1), round(100 * float(np.median(sg)), 1), len(sg)] if sg else None),
            "sold after the LP was pulled": sum(r["pulled_before_sale"] for r in rows),
            "by what the ideal trade did": by_ideal,
            "by fill second (pool age at buy)": _gap_bucket(rows, "in_s", (
                ("< 8 s", 0, 8), ("8-10 s", 8, 10), ("10-15 s", 10, 15), ("15-30 s", 15, 30), ("30 s+", 30, 1e9))),
            "by sale second (pool age at sale)": _gap_bucket(rows, "out_s", (
                ("< 355 s", 0, 355), ("355-375 s", 355, 375), ("375-420 s", 375, 420), ("420 s+", 420, 1e12))),
            "by liquidity at second 4 (USD)": _gap_bucket(rows, "liq_usd", (
                ("< 10k", 0, 1e4), ("10-15k", 1e4, 1.5e4), ("15-25k", 1.5e4, 2.5e4), ("25-50k", 2.5e4, 5e4),
                ("50k+", 5e4, 1e12))),
            "by exit reason": [{"exit": k, "n": len(v), "mean_gap_pct": _pct(v),
                                "share_of_total_gap_pct": round(100 * sum(v) / total)}
                               for k, v in sorted(by_exit.items(), key=lambda kv: sum(kv[1]))],
            "worst trades": [{"symbol": r["symbol"], "token": r["token"][:8], "real_pct": round(100 * r["real"], 1),
                              "ideal_pct": round(100 * r["ideal"], 1), "gap": round(100 * r["gap"], 1),
                              "swap_pct": None if r["swap"] is None else round(100 * r["swap"], 1),
                              "in_s": round(r["in_s"], 1), "out_s": None if r["out_s"] is None else round(r["out_s"]),
                              "liq_usd": None if r["liq_usd"] is None else round(r["liq_usd"]), "exit": r["exit"],
                              "pulled_before_sale": r["pulled_before_sale"]}
                             for r in sorted(rows, key=lambda r: r["gap"])[:GAP_WORST]],
        }
    out["note"] = ("gap = real return - scorer's ideal return on the same pool; ideal = vault price at second ~6.5 "
                   "in, minute 6 out, flat 6% cost, capped +300%, rug = -100%; price-only gap leaves out rugged and "
                   "capped pools and both sides' costs (real fees, the scorer's 6%)")
    return out


# --- size study (11-06): would sizing by pool liquidity earn more, without giving up the big winners? ------------
SIZE_FIXED = (25, 50, 100, 250, 500, 1000)
SIZE_SHARES = (0.0025, 0.005, 0.01, 0.02)  # trade size as a share of the pool's liquidity at second 4
SIZE_MIN, SIZE_MAX = 20.0, 2000.0
SIZE_BIG = 0.5  # ideal +50% or more = a big winner
SIZE_LIQ_BANDS = (("< 15k", 0, 1.5e4), ("15-25k", 1.5e4, 2.5e4), ("25-50k", 2.5e4, 5e4), ("50k+", 5e4, 1e12))


def _amm_keep(size: float, r_in: float, gross: float, r_out: float) -> float:
    """Share of spot value kept on a constant-product round trip: buy `size` USD into a pool holding `r_in` USD of
    SOL, the price moves by `gross`, then sell into `r_out` USD of SOL. Pool fees and timing are in the real trade."""
    buy = 1 / (1 + size / r_in) if r_in > 0 else 1.0
    value = size * (1 + gross) * buy
    sell = 1 / (1 + value / r_out) if r_out > 0 else 1.0
    return buy * sell


def size_study(store, journal, book: str = "launchrw50", hours: float | None = None, now: float | None = None) -> dict:
    """Each real trade of `book` re-sized: its real result at its real size, with only the price impact changed
    (constant-product pool, reserves at the fill and at minute 6 from the scorer's vault reads). Rugs stay -100%.
    Compares fixed sizes with sizes scaled to liquidity: total $, $ from big winners and rugs, $ by liquidity band."""
    now = now if now is not None else time.time()
    ensure_schema(store)
    since = now - hours * 3600 if hours else 0.0
    trades = []
    for p in journal.query("SELECT id, token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ? AND closed_at IS NOT NULL", (book, since)):
        s = store.query("SELECT ret, liq_usd, sol_at_decide, fill_sol, exit_sol, fill_px, exit_px FROM watch_launch_live "
                        "WHERE token_mint=? AND status='done' AND create_ts <= ? ORDER BY create_ts DESC LIMIT 1",
                        (p["token"], p["opened_at"]))
        if not s or not p["cost_usd"] or not s[0]["liq_usd"] or not s[0]["sol_at_decide"] or not s[0]["fill_sol"]:
            continue
        s = s[0]
        sol_usd = s["liq_usd"] / (2 * s["sol_at_decide"])
        fees = sum((f["fee_usd"] or 0) for f in journal.query("SELECT fee_usd FROM fills WHERE position_id=?", (p["id"],)))
        gross = (s["exit_px"] / s["fill_px"] - 1) if s["exit_px"] and s["fill_px"] else 0.0
        trades.append({"s0": p["cost_usd"], "proceeds0": p["realized_usd"] + fees, "fees": fees,
                       "liq": s["liq_usd"], "r_in": s["fill_sol"] * sol_usd,
                       "r_out": (s["exit_sol"] or 0) * sol_usd, "gross": gross,
                       "ideal": s["ret"], "rug": p["realized_usd"] <= 0.01 * p["cost_usd"]})
    if not trades:
        return {"book": book, "trades": 0}

    def result(t, size):
        if t["rug"]:
            return -size  # nothing comes back
        keep0 = _amm_keep(t["s0"], t["r_in"], t["gross"], t["r_out"])
        keep = _amm_keep(size, t["r_in"], t["gross"], t["r_out"])
        proceeds = size * (t["proceeds0"] / t["s0"]) * keep / keep0
        return proceeds - t["fees"] - size

    def policy(sizer):
        rows = [(t, sizer(t)) for t in trades]
        pnl = [result(t, s) for t, s in rows]
        big = [x for (t, _), x in zip(rows, pnl) if (t["ideal"] or 0) >= SIZE_BIG]
        rug = [x for (t, _), x in zip(rows, pnl) if t["rug"]]
        sizes = [s for _, s in rows]
        out = {"total_usd": round(sum(pnl)), "avg_size_usd": round(float(np.mean(sizes))),
               "per_100_deployed_usd": round(100 * sum(pnl) / sum(sizes), 2),
               "big_winners_usd": round(sum(big)), "rugs_usd": round(sum(rug)),
               "worst_trade_usd": round(min(pnl))}
        out["by liquidity"] = {name: round(sum(x for (t, _), x in zip(rows, pnl) if lo <= t["liq"] < hi))
                               for name, lo, hi in SIZE_LIQ_BANDS}
        return out

    clip = lambda x: min(SIZE_MAX, max(SIZE_MIN, x))
    mean_liq = float(np.mean([t["liq"] for t in trades]))
    policies = {f"fixed ${a}": policy(lambda t, a=a: float(a)) for a in SIZE_FIXED}
    for f in SIZE_SHARES:
        policies[f"{100 * f:g}% of liquidity"] = policy(lambda t, f=f: clip(f * t["liq"]))
    policies["liquidity-scaled, same average size as $100"] = policy(lambda t: clip(100 * t["liq"] / mean_liq))
    # calibration: the model's own impact at $100 by band vs the books' measured real-minus-ideal gap
    calib = []
    for name, lo, hi in SIZE_LIQ_BANDS:
        g = [t for t in trades if lo <= t["liq"] < hi and not t["rug"]]
        if g:
            calib.append({"band": name, "n": len(g),
                          "model round-trip impact at $100 (%)": round(100 * float(np.median(
                              [1 - _amm_keep(100, t["r_in"], t["gross"], t["r_out"]) for t in g])), 2),
                          "measured gap real - ideal (median pts)": round(100 * float(np.median(
                              [t["proceeds0"] / t["s0"] - 1 - t["fees"] / t["s0"] - (t["ideal"] or 0) for t in g])), 1),
                          "big winners %": round(100 * sum((t["ideal"] or 0) >= SIZE_BIG for t in g) / len(g))})
    return {"book": book, "trades": len(trades), "rugs": sum(t["rug"] for t in trades),
            "big winners (ideal +50% or more)": sum((t["ideal"] or 0) >= SIZE_BIG for t in trades),
            "mean liquidity at second 4 (USD)": round(mean_liq),
            "calibration by liquidity": calib, "policies": policies,
            "note": ("each trade keeps its real result at its real size; only the constant-product price impact "
                     "changes with size (reserves at the scorer's fill and minute-6 reads); rugs lose the whole "
                     "stake at any size; bigger sizes also move the pool for other traders and may be harder to fill, "
                     "which this ignores, so the larger sizes are optimistic")}
