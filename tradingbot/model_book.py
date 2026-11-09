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

import calendar
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
# Rolling twins (10-31): same model and exits, but buy the top `early_top_pct` of the past day's eligible pools
EARLY5R, EARLY6R, EARLY7R = "early5r", "early6r", "early7r"
EARLY_FIXED = (EARLY5, EARLY6, EARLY7)
EARLY_ROLLING = (EARLY5R, EARLY6R, EARLY7R)
# Wide twin (11-03): early6's model and exits, but a lower fixed cut-off: the top `early_wide_top_pct` (16%) of the
# recent training scores (`meta.cut_wide`). Mean + 1 SD was tried first: the scores are skewed, so it sat at ~top 5%.
EARLY6W = "early6w"
# Tight twin (11-03): early6's model and exits, top `early_tight_top_pct` (2%) of the recent training scores
# (`meta.cut_tight`). Walk-forward 10-16..11-03: +1.1%/trade (n=202, CI -7.5..+9.9) vs top 5% -1.9%, every pool
# -3.4%; but -8.3% on 10-16..10-29 and +15.2% on 10-30..11-03.
EARLY6T = "early6t"
EARLY_EXIT_MIN = {EARLY5: 5.0, EARLY6: 6.0, EARLY7: 7.0, EARLY5R: 5.0, EARLY6R: 6.0, EARLY7R: 7.0, EARLY6W: 6.0, EARLY6T: 6.0}
EARLY_BOOKS = tuple(EARLY_EXIT_MIN)
# Launch books (11-02): picks of the second-4 launch scorer (launch_live), top 2% / top 5%, bought through Jupiter
# (it routes hand-made pools within ~0.1-2.4 s of creation) and sold at pool minute 6.
LAUNCH2, LAUNCH5 = "launch4_2", "launch4_5"
# Their twins (11-03): a model trained on the chain-only history of every hand-made pool (launch_hist) plus the
# scorer's own outcomes, no hour/weekend features, picking the top 2% / 5% of the past 24 h's scores (rolling).
LAUNCH2H, LAUNCH5H = "launch4h_2", "launch4h_5"
LAUNCH_HIST_CUT = {LAUNCH2H: "top2h", LAUNCH5H: "top5h"}
# Minute-5 twin (11-04): launch4_2's signals sold at pool minute 5. launch-exit-study on 72 old-model picks:
# minute 5 +13.0% (CI -1.8..+28.6) vs minute 6 +6.2%; pulls cluster at 60-120 s, two more land in minute 5-6.
LAUNCH2M5 = "launch4_2m5"
# Its top-5% twin (11-05): launch4_5's picks sold at pool minute 5 (the model retrained 11-05 and its live "top 2%"
# fell from ~8-10% of pools to ~2%, so launch4_5 is now the nearest to the wider group launch4_2m5 traded at first).
LAUNCH5M5 = "launch4_5m5"
LAUNCH_M5 = (LAUNCH2M5, LAUNCH5M5)
# Low-decile books (11-04): the history model's bottom 20% (rolling 24 h; D1-D2 held the most big wins and the most
# early pulls), with the riskiest 0/10/20/30/50% by the early-pull model skipped. launch-early-pulls walk-forward on
# 188 D1-D2 pools: keep all +2.9%, skip 10% +6.6%, 20% +5.5%, 30% +12.0% (98% of random skips worse), 50% +12.8%.
LAUNCH_LOW_CUT = {"launchd12": "d12", "launchd12_s10": "d12s10", "launchd12_s20": "d12s20",
                  "launchd12_s30": "d12s30", "launchd12_s50": "d12s50"}
# Rug-vs-win books (11-04, launch-filter-study option 4): skip the riskiest half by P(rug before the exit) -
# P(+50% win). Walk-forward on 2,041 pools: every liquid pool +7.6%/trade (keep all -1.6%), the history model's
# bottom 20% +13.1% (keep all +4.1%); both beat random skips of the same size.
LAUNCH_RW_CUT = {"launchrw50": "rw50", "launchd12rw50": "d12rw50"}
# Twins on the earlier live rule (11-05; the books above moved to the study's 6 h rule): nightly 14-day model,
# ranked against the past 24 h, bottom 20% = low_h. Run side by side so a good spell under either isn't missed.
LAUNCH_RW24_CUT = {"launchrw50_24h": "rw50_24", "launchd12rw50_24h": "d12rw50_24"}
# Size-by-band twin (11-06, size-study on 1,039 launchrw50 trades): launchrw50's signals, sized by the pool's
# liquidity at second 4. The edge sat in thin pools (< $15k: 48% big winners, best at $250), $15-25k lost at every
# size (so it gets a token $25 to keep measuring it), deeper pools stay at $100. Chosen after looking: this is the test.
LAUNCH_SIZED = "launchrw50_sz"
LAUNCH_SIZED_CUT = {LAUNCH_SIZED: "rw50"}
SIZE_BANDS = ((15_000.0, 250.0), (25_000.0, 25.0), (float("inf"), 100.0))  # (liquidity below, trade size $)
# By-band twins (11-08, `band-rw-study`): the live rug-vs-win model ranked within each liquidity band (flag
# `rwband`, launch_live.BAND_RULES): $5-15k skip the riskiest 20%, $15-25k skip 70%, $25-40k the live rule, nothing
# above $40k. Same rule in both books; only the thin band's size differs ($100 vs $250), to measure price impact.
LAUNCH_BAND = "launchrwband"
LAUNCH_BAND_SZ = "launchrwband_sz"
LAUNCH_BAND_CUT = {LAUNCH_BAND: "rwband", LAUNCH_BAND_SZ: "rwband"}
BAND_SIZE_BANDS = {LAUNCH_BAND: ((15_000.0, 100.0), (float("inf"), 100.0)),
                   LAUNCH_BAND_SZ: ((15_000.0, 250.0), (float("inf"), 100.0))}  # (liquidity below, trade size $)
# Longer holds in $25-40k pools (11-08, `hold-study --kept-only`, 336 kept pools): the live rug-vs-win kept half
# (flag `rw50big`, launch_live.HOLD_BAND) held to pool minute 10 made +7.9% (+5.5 pts over minute 6, 90% CI +2.0..+8.8),
# to minute 15 +11.0% (+8.7 pts, CI +2.9..+14.2); ideal fills, hold times chosen after looking. Twins: out at minute 10
# vs 15, $250 a trade (near the impact-limited best size for these pools), capped at 2% of the pool as usual.
LAUNCH_HOLD10 = "launchrw50_25k10"
LAUNCH_HOLD15 = "launchrw50_25k15"
# 11-09: the minute-15 book at the size the user would run it: the engine's 2%-of-pool cap with a $500 ceiling (every
# $25-40k pool's 2% is $500-800, so $500 a trade); compared with launchrw50_25k15 on the same tokens it measures the
# impact of the bigger size
LAUNCH_HOLD15_MAX = "launchrw50_25k15max"
LAUNCH_HOLD_CUT = {LAUNCH_HOLD10: "rw50big", LAUNCH_HOLD15: "rw50big", LAUNCH_HOLD15_MAX: "rw50big"}
BAND_SIZE_BANDS.update({LAUNCH_HOLD10: ((float("inf"), 250.0),), LAUNCH_HOLD15: ((float("inf"), 250.0),),
                        LAUNCH_HOLD15_MAX: ((float("inf"), 500.0),)})
# +50% filter in $40-80k pools with a conditional hold (11-08, `band-prize`, 571 pools, ideal fills, rule chosen
# after looking): the +50% model (launch_live.BIG_BAND, flag `big30`) keeps the best 30% of the band's past day;
# at pool minute 6 and 10 the position stays only while a sell quote would return at least the cost and (at 10)
# no less than at 6, else it sells there; the rest sell at pool minute 15. Backtest +6.6%/trade vs +2.6% selling
# at minute 6 (every $40-80k pool -15%); $250 a trade (2% of a $40k pool is $800).
LAUNCH_BIG = "launchbig40k"
LAUNCH_BIG_CUT = {LAUNCH_BIG: "big30"}
BAND_SIZE_BANDS[LAUNCH_BIG] = ((float("inf"), 250.0),)
HOLD_CHECK_MIN = {LAUNCH_BIG: (6.0, 10.0)}  # pool minutes of the conditional hold checks
HOLD_CHECK_GAIN = 0.0  # stay only while the sale would return at least the cost ...
HOLD_CHECK_RISING = True  # ... and no less than at the previous check
# Real-money canary (11-06; was canary_rw50_24h, never switched on): copies the size-by-band book (`CANARY_TWIN`),
# same signals and minute-6 exit, filled by the live Jupiter executor; every pool (fly.toml sets
# `launch_canary_max_liq_usd` 0), sized like the twin's bands in the same order: $250 / $25 / $100 -> $5 / $0.50 / $2 (1/50).
CANARY4 = "canary_rw50_sz"
CANARY_TWIN = LAUNCH_SIZED
CANARY_BOOKS = (CANARY4,)
CANARY_SIZE_BANDS = ((15_000.0, 5.0), (25_000.0, 0.5), (float("inf"), 2.0))  # (liquidity below, $)
LAUNCH_CUT = {LAUNCH2: "top2", LAUNCH5: "top5", LAUNCH2M5: "top2", LAUNCH5M5: "top5", **LAUNCH_HIST_CUT, **LAUNCH_LOW_CUT,
              **LAUNCH_RW_CUT, **LAUNCH_RW24_CUT, **LAUNCH_SIZED_CUT, **LAUNCH_BAND_CUT, **LAUNCH_HOLD_CUT, **LAUNCH_BIG_CUT}
LAUNCH_CUT[CANARY4] = LAUNCH_CUT[CANARY_TWIN]  # the canary takes its twin's signals
LAUNCH_EXIT_MIN = {b: 5.0 if b in LAUNCH_M5 else 10.0 if b == LAUNCH_HOLD10 else 15.0 if b in (LAUNCH_HOLD15, LAUNCH_HOLD15_MAX, LAUNCH_BIG) else 6.0
                   for b in LAUNCH_CUT}
POOL_MINUTE_EXIT = {**EARLY_EXIT_MIN, **LAUNCH_EXIT_MIN}  # books that sell every position by a pool minute
EARLY_MAX_SEEN_MIN = 4.0  # first sight later than this leaves no time to trade (and was left out of training)
EARLY_GAIN = 0.10  # label: the 5-minute snapshot price >= first-sight price x 1.10 after costs, pool intact
BOOKS = (NAME, FAST, CLIMB, OPS9, OPS10) + EARLY_BOOKS + tuple(LAUNCH_CUT)
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
    if name in POOL_MINUTE_EXIT:
        return sig["created_ts"] + POOL_MINUTE_EXIT[name] * 60 if sig.get("created_ts") else None
    age = cfg.model_book_handmade_exit_age_min
    if name not in AGE_EXIT or age <= 0 or not sig.get("created_ts") or not is_handmade(sig.get("dex") or "", sig["mint"]):
        return None
    return sig["created_ts"] + age * 60


def band_usd(name: str, liq: float | None) -> float | None:
    """Trade size of a size-by-band book (the sized twin, or the canary at its small sizes) for a pool with this
    liquidity; None for every other book."""
    bands = SIZE_BANDS if name in LAUNCH_SIZED_CUT else CANARY_SIZE_BANDS if name in CANARY_BOOKS \
        else BAND_SIZE_BANDS.get(name)
    if bands is None or liq is None:
        return None
    return next(usd for below, usd in bands if liq < below)


def active_books(cfg: Config) -> list[str]:
    off = {b.strip() for b in (cfg.disabled_books or "").split(",") if b.strip()}
    return [b for b in _family_books(cfg) if b not in off]


def _family_books(cfg: Config) -> list[str]:
    return ([NAME] if cfg.model_book_slow else []) + ([FAST] if cfg.model_book_fast else []) \
        + ([CLIMB] if cfg.model_book_climb else []) + ([OPS9, OPS10] if cfg.model_book_ops else []) \
        + (list(EARLY_FIXED) if cfg.early_book else []) + (list(EARLY_ROLLING) if cfg.early_rolling else []) \
        + ([EARLY6W] if cfg.early_wide else []) + ([EARLY6T] if cfg.early_tight else []) \
        + ([LAUNCH2, LAUNCH5] if cfg.launch_book and cfg.launch_live else []) \
        + (list(LAUNCH_M5) if cfg.launch_m5_book and cfg.launch_live else []) \
        + (list(LAUNCH_HIST_CUT) if cfg.launch_hist_book and cfg.launch_live else []) \
        + (list(LAUNCH_LOW_CUT) if cfg.launch_low_book and cfg.launch_live else []) \
        + (list(LAUNCH_RW_CUT) if cfg.launch_rw_book and cfg.launch_live else [])\
        + (list(LAUNCH_RW24_CUT) if cfg.launch_rw24_book and cfg.launch_live else []) \
        + (list(LAUNCH_SIZED_CUT) if cfg.launch_sized_book and cfg.launch_live else []) \
        + (list(LAUNCH_BAND_CUT) if cfg.launch_band_book and cfg.launch_live else []) \
        + (list(LAUNCH_HOLD_CUT) if cfg.launch_hold_book and cfg.launch_live else []) \
        + (list(LAUNCH_BIG_CUT) if cfg.launch_big_book and cfg.launch_live else []) \
        + ([CANARY4] if cfg.launch_canary and cfg.launch_live else [])


def early_wants(name: str, sig: dict) -> bool:
    """Which early books take an early signal: the fixed books its fixed cut-off pass, the rolling ones its rolling,
    the wide book its wide (top 16%) cut-off pass."""
    if name == EARLY6W:
        return bool(sig.get("wide"))
    if name == EARLY6T:
        return bool(sig.get("tight"))
    return bool(sig.get("rolling")) if name in EARLY_ROLLING else bool(sig.get("fixed"))


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


def hold_checks(name: str, sig: dict) -> list[float] | None:
    """Times of a book's conditional hold checks (pool minutes in HOLD_CHECK_MIN after the pool's creation)."""
    mins = HOLD_CHECK_MIN.get(name)
    if not mins or not sig.get("created_ts"):
        return None
    return [sig["created_ts"] + m * 60 for m in mins]


def hold_check_fails(ret: float, last: float | None) -> bool:
    """band_prize.hold_result's rule on a live sell quote: sell unless up HOLD_CHECK_GAIN and (with
    HOLD_CHECK_RISING) no lower than at the previous check."""
    return ret < HOLD_CHECK_GAIN or (HOLD_CHECK_RISING and last is not None and ret < last)


def signal_kind(name: str) -> str | None:
    """Which signals a book takes: early (first sight), launch (second-4 scorer) or model5 (None)."""
    return "early" if name in EARLY_BOOKS else "launch" if name in LAUNCH_CUT else None


def wants(name: str, sig: dict) -> bool:
    if name in EARLY_BOOKS:
        return early_wants(name, sig)
    if name in LAUNCH_CUT:
        return bool(sig.get(LAUNCH_CUT[name]))
    return True


def hold_min(cfg: Config, name: str = NAME) -> float:
    if name in LAUNCH_EXIT_MIN:  # from the buy (~pool second 6); the pool-minute deadline normally sells first
        return LAUNCH_EXIT_MIN[name]
    if name in OPS_HOLD_MIN:
        return OPS_HOLD_MIN[name]
    if name in EARLY_EXIT_MIN:  # from the buy; the pool-minute deadline normally sells first
        return EARLY_EXIT_MIN[name]
    return cfg.model_book_fast_hold_min if name in SHORT else cfg.model_book_hold_min


def exit_config(cfg: Config, name: str = NAME) -> Config:
    """A model book's exits: one take-profit at +100%, a -50% stop, out after its hold time. The launch books
    hold to pool minute 6 with no stop or take-profit, as in their backtest."""
    if name in LAUNCH_CUT:
        return replace(cfg, stop_loss_pct=99.0, take_profit_pcts=(10_000.0,), take_profit_fractions=(1.0,),
                       max_hold_hours=hold_min(cfg, name) / 60)
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
        z = (np.array(vals) - self.mu) / self.sd
        if self.meta.get("clip_z"):  # models trained on capped inputs (launch4h) score capped inputs too
            z = np.clip(z, -self.meta["clip_z"], self.meta["clip_z"])
        return float(z @ self.w + self.b)

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
        row = {"pool": p["pool"], "created_ts": p["created_ts"], "up": float(ret >= EARLY_GAIN), "ret": ret,
               "first_liq": lf}
        row.update({k: v for k, v in early_features(dict(p), first, p["seen_ts"]).items() if v is not None})
        rows.append(row)
    return rows


EARLY_NOT_FEATURES = {"pool", "created_ts", "up", "ret", "first_liq"}


def train_early(store, cfg: Config, now: float | None = None) -> Model | None:
    return fit_early(early_training_rows(store, now, cfg.model_train_days), cfg)


def fit_early(rows: list[dict], cfg: Config) -> Model | None:
    """The early model on `rows` (oldest first): logistic on `up`, fixed cut-off = top `early_top_pct` of the last
    2 days' scores, `cut_wide` = top `early_wide_top_pct` of the same scores."""
    if len(rows) < wa.MIN_ROWS or not any(r["up"] for r in rows):
        log.warning("early model: not enough rows to train (%d)", len(rows))
        return None
    names = sorted({k for r in rows for k in r} - EARLY_NOT_FEATURES)
    x, med = wa._matrix(rows, names)
    y = np.array([r["up"] for r in rows])
    mu, sd = x.mean(0), x.std(0) + 1e-9
    w, b = wa.logistic((x - mu) / sd, y)
    scores = ((x - mu) / sd) @ w + b
    recent = scores[[r["created_ts"] >= rows[-1]["created_ts"] - 2 * wa.DAY_S for r in rows]]
    threshold = float(np.quantile(recent, 1 - cfg.early_top_pct / 100))
    top = scores >= threshold
    cut_wide = float(np.quantile(recent, 1 - cfg.early_wide_top_pct / 100))  # early6w: top early_wide_top_pct
    meta = {"trained_ts": time.time(), "rows": len(rows), "pump_rate": float(y.mean()),
            "cut_tight": float(np.quantile(recent, 1 - cfg.early_tight_top_pct / 100)),
            "cut_wide": cut_wide, "wide_share_recent": float(np.mean(recent >= cut_wide)),
            "cuts": {f"{t:g}": float(np.quantile(recent, 1 - t / 100)) for t in WALK_TOPS},
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
        out["timing_s"] = early_timing(journal, since)
    return out


LAUNCH_STAGES = (("created", "decided"), ("decided", "picked"), ("picked", "started"), ("started", "filled"),
                 ("created", "filled"))


def early_timing(journal, since: float, books=EARLY_BOOKS, stages=None) -> dict:
    """Median seconds per stage of the early books' buys (events `early_timing`): GeckoTerminal listing the pool
    (created -> seen = the watcher's step start), the new-pools call and scoring (seen -> found), the queue
    (found -> picked), the book's own checks (picked -> started) and quotes plus the simulated landing
    (started -> filled)."""
    stages = stages or (("created", "seen"), ("seen", "found"), ("found", "picked"), ("picked", "started"),
                        ("started", "filled"), ("created", "filled"))
    out = {}
    for b in books:
        rows = [json.loads(r[0]) for r in journal.query(
            "SELECT detail FROM events WHERE kind='early_timing' AND strategy=? AND ts >= ?", (b, since))]
        if rows:
            out[b] = {"buys": len(rows), **{f"{a}->{z}": round(float(np.median([r[z] - r[a] for r in rows
                                                                                 if a in r and z in r])), 1)
                                           for a, z in stages if any(a in r and z in r for r in rows)}}
    return out


ROLLING_WINDOW_S = 24 * 3600  # the rolling cut-off ranks a pool against the eligible pools of the past day
ROLLING_MIN_SCORES = 50  # fewer eligible pools than this in the window: no rolling cut-off yet


def early_candidates(store, cfg: Config, model: Model | None, since: float, until: float) -> list[dict]:
    """Hand-made pools first seen in [since, until) within EARLY_MAX_SEEN_MIN, with enough liquidity, scored by
    `model` (None: score None), oldest first; with their 5-minute snapshot outcome when it exists (as the model
    was trained)."""
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
                    "score": model.score(early_features(dict(p), first, p["seen_ts"])) if model else None,
                    "ret": ret})
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


# early-walk-forward compares these top-% cut-offs (early6 = 5, early6t = 2, early6w = 16; 11-04: the super-premium
# 1 / 0.5 / 0.25 / 0.1% too). A cut-off from 2 days of training scores (~1,500 pools) at 0.1% rests on 1-2 scores.
WALK_TOPS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 16.0)


def early_walk_forward(store, cfg: Config, now: float | None = None, seed: int = 7) -> dict:
    """Walk-forward test of the early model's cut-offs on every day the store holds: each UTC day is scored by an
    early model trained only on the `model_train_days` before it (as the nightly retrain does), and its eligible
    pools (liquid at first sight) are picked by each top-% cut-off in WALK_TOPS, set on the last 2 training days'
    scores like the live one. Outcome = the 5-minute snapshot (~pool minute 6) vs first sight, 6% round trip,
    emptied = -100%."""
    rows = early_training_rows(store, now, days=10_000)
    if not rows:
        return {"error": "no rows"}
    rng = np.random.default_rng(seed)
    day = lambda ts: time.strftime("%Y-%m-%d", time.gmtime(ts))
    days = sorted({day(r["created_ts"]) for r in rows})
    label = lambda t: f"top {t:g}%"
    groups = {"every eligible pool": [], **{label(t): [] for t in WALK_TOPS}}
    by_day, cuts, day_scores = [], [], []
    for d in days:
        start = calendar.timegm(time.strptime(d, "%Y-%m-%d"))  # UTC midnight
        train = [r for r in rows if start - cfg.model_train_days * wa.DAY_S <= r["created_ts"] < start]
        test = [r for r in rows if day(r["created_ts"]) == d and r["first_liq"] >= cfg.model_book_min_liq_usd]
        if len(train) < wa.MIN_ROWS or not test:
            continue
        model = fit_early(train, cfg)
        if model is None:
            continue
        scored = [(model.score({k: v for k, v in r.items() if k not in EARLY_NOT_FEATURES}), r["ret"]) for r in test]
        day_scores.append(scored)
        picks = {"every eligible pool": [x for _, x in scored]}
        for t in WALK_TOPS:
            cut = model.meta["cuts"][f"{t:g}"]
            picks[label(t)] = [x for s, x in scored if s >= cut]
        for k, v in picks.items():
            groups[k] += v
        by_day.append({"day": d, "train_rows": len(train), "eligible": len(test),
                       "all_mean_pct": _ret_summary(picks["every eligible pool"], rng, ci=False)["mean_pct"],
                       **{label(t): _ret_summary(picks[label(t)], rng, ci=False) for t in WALK_TOPS}})
        cuts.append({"day": d, **{f"share_above_{label(t)}_cut_pct":
                                  round(100 * len(picks[label(t)]) / len(scored), 1) for t in WALK_TOPS}})
    return {"days_tested": len(by_day), "first_day": days[0], "last_day": days[-1],
            "overall": {k: _ret_summary(v, rng) for k, v in groups.items()},
            "by score decile (ranked within each day)": decile_table(day_scores, rng),
            "by_day": by_day, "cut_offs": cuts,
            "note": "walk-forward by UTC day, trained on the previous model_train_days; outcome = 5-minute snapshot "
                    "(~pool minute 6) vs first-sight price, 6% round trip, emptied = -100%; the last day may be "
                    f"partial; big win = +{BIG_WIN:.0%} or more after costs"}


BIG_WIN = 0.5  # a trade that made +50% or more after costs counts as a pump in the decile tables


def decile_index(scores) -> list[int]:
    """Score bucket 0 (lowest 10%) .. 9 (top 10%) of each score within its own list (ties keep input order)."""
    n = len(scores)
    ranks = np.argsort(np.argsort(np.asarray(scores, dtype=float), kind="stable"), kind="stable")
    return [min(9, int(10 * r / n)) for r in ranks]


def decile_label(i: int) -> str:
    return f"D{i + 1}" + (" (lowest scores)" if i == 0 else " (highest scores)" if i == 9 else "")


def decile_table(groups: list[list[tuple[float, float]]], rng) -> dict:
    """11-04 two-score plan, step 1: do pump rate and rug rate rise together with the score? Each group (a test
    day, or the whole window) is split into ten buckets by its own scores; buckets are pooled across groups."""
    buckets: list[list[float]] = [[] for _ in range(10)]
    for g in groups:
        if not g:
            continue
        for i, (_, ret) in zip(decile_index([s for s, _ in g]), g):
            buckets[i].append(ret)
    return {decile_label(i): {**_ret_summary(b, rng),
                              "big_win_pct": round(100 * sum(x >= BIG_WIN for x in b) / len(b)) if b else None}
            for i, b in enumerate(buckets)}


def _ret_summary(rets: list[float], rng, ci: bool = True) -> dict:
    if not rets:
        return {"n": 0}
    a = np.array(rets)
    out = {"n": len(a), "mean_pct": round(float(a.mean()) * 100, 1), "median_pct": round(float(np.median(a)) * 100, 1),
           "win_pct": round(float((a > 0).mean()) * 100), "emptied_pct": round(float((a <= -0.99).mean()) * 100)}
    if ci and len(a) > 1:
        means = rng.choice(a, (2000, len(a))).mean(1)
        out["ci90"] = [round(float(np.percentile(means, 5)) * 100, 1), round(float(np.percentile(means, 95)) * 100, 1)]
    return out


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
            "research_rug_rate_pct": {FAST: 16.0, CLIMB: 7.0, OPS9: 13.0, OPS10: 15.0, EARLY5: 25.0, EARLY6: 25.0, EARLY7: 25.0,
                                      EARLY5R: 26.0, EARLY6R: 26.0, EARLY7R: 26.0, EARLY6W: 25.0, EARLY6T: 27.0,
                                      "launchd12": 30.0, "launchd12_s10": 27.0, "launchd12_s20": 24.0,
                                      "launchd12_s30": 18.0, "launchd12_s50": 11.0,
                                      "launchrw50": 21.0, "launchd12rw50": 17.0, LAUNCH_SIZED: 23.0,
                                      LAUNCH_BAND: 25.0, LAUNCH_BAND_SZ: 25.0, LAUNCH_HOLD10: 27.0,
                                      LAUNCH_HOLD15: 34.0, LAUNCH_HOLD15_MAX: 34.0, LAUNCH_BIG: 30.0}.get(name, 76.0),  # backtests
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
        self._history: list[tuple[float, float]] | None = None  # (seen_ts, score) of eligible pools, past day
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
                if attr == "early":
                    self._history = None  # a new model scores differently: rebuild the rolling window
                log.info("%s loaded (%s rows, cut-off %.3f)", label, m.meta.get("rows"), m.threshold)

    def consider_first(self, pool: dict, first: dict) -> float | None:
        """Called by the watcher when it first sees a pool (`pool` has created_ts and seen_ts). Hand-made pools
        at or above the early model's fixed cut-off (fixed books) or in the top `early_top_pct` of the past day's
        eligible pools (rolling books) are queued."""
        if not (self.cfg.early_book or self.cfg.early_rolling or self.cfg.early_wide
                or self.cfg.early_tight) or not first or \
                not is_handmade(pool.get("dex") or "", pool.get("mint") or ""):
            return None
        self.reload()
        if self.early is None or not pool.get("created_ts") or not pool.get("seen_ts"):
            return None
        if not 0 <= pool["seen_ts"] - pool["created_ts"] <= EARLY_MAX_SEEN_MIN * 60:
            return None
        liq = wa._f(first.get("liq")) or 0.0
        if liq < self.cfg.model_book_min_liq_usd or not wa._f(first.get("price")):
            return None
        score = self.early.score(early_features(pool, first, pool["seen_ts"]))
        fixed = self.cfg.early_book and score >= self.early.threshold
        rolling = False
        if self.cfg.early_rolling:
            history = self._rolling_history(pool["seen_ts"])
            cut = rolling_cut_off(history, pool["seen_ts"], self.cfg.early_top_pct)
            rolling = cut is not None and score >= cut
            history.append((pool["seen_ts"], score))
        cut_wide = self.early.meta.get("cut_wide")  # None on a model trained before 11-03: no wide signals
        wide = bool(self.cfg.early_wide and cut_wide is not None and score >= cut_wide)
        cut_tight = self.early.meta.get("cut_tight")  # None on a model trained before: no tight signals
        tight = bool(self.cfg.early_tight and cut_tight is not None and score >= cut_tight)
        if not (fixed or rolling or wide or tight):
            return None
        try:
            self.signals.put_nowait({"pool": pool["pool"], "mint": pool["mint"], "name": pool.get("name") or "",
                                     "score": score, "price": wa._f(first["price"]), "liq": liq, "ts": time.time(),
                                     "created_ts": pool["created_ts"], "seen_ts": pool["seen_ts"],
                                     "dex": pool.get("dex") or "", "kind": "early",
                                     "fixed": fixed, "rolling": rolling, "wide": wide, "tight": tight})
        except queue.Full:
            return None
        if self.store is not None:  # the enricher records this pool's launch data whatever the random sample
            try:
                self.store.set_state(f"early_signal:{pool['pool']}", repr(time.time()))
            except Exception:  # noqa: BLE001 (research bookkeeping must never stop a signal)
                log.warning("could not mark early signal %s", pool["pool"], exc_info=True)
        return score

    def _rolling_history(self, now: float) -> list[tuple[float, float]]:
        """The past day's eligible scores; rebuilt from the watcher's store at start-up and after a retrain."""
        if self._history is None:
            self._history = []
            if self.store is not None and self.early is not None:
                self._history = [(c["seen_ts"], c["score"]) for c in
                                 early_candidates(self.store, self.cfg, self.early, now - ROLLING_WINDOW_S, now)]
        self._history = [h for h in self._history if h[0] >= now - ROLLING_WINDOW_S]
        return self._history

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
