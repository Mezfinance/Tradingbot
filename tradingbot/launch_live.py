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
import heapq
import json
import logging
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

SCHEMA = """CREATE TABLE IF NOT EXISTS watch_launch_live(
    pool TEXT PRIMARY KEY, lp_mint TEXT, token_mint TEXT, creator TEXT, notified_ts REAL, create_ts REAL,
    decided_ts REAL, indexed_s REAL, liquid INTEGER, score REAL, top2 INTEGER, top5 INTEGER, feat TEXT,
    sol_at_decide REAL, peak_sol REAL, sol_vault TEXT, token_vault TEXT, fill_ts REAL, fill_px REAL,
    fill_sol REAL, exit_ts REAL, exit_px REAL, exit_sol REAL, pulled_ts REAL, ret REAL, status TEXT,
    reason TEXT, credits INTEGER DEFAULT 0, liq_usd REAL)"""
# Backtest edge per trade after its 6% cost (decide 4 / fill 6, no names, 11-02), for the size estimate in status.
BACKTEST_EDGE = {"top 2%": 0.33, "top 5%": 0.27}


def ensure_schema(store) -> None:
    import sqlite3
    store.execute(SCHEMA)
    try:  # added 11-02: pool liquidity at second 4 in USD
        store.execute("ALTER TABLE watch_launch_live ADD COLUMN liq_usd REAL")
    except sqlite3.OperationalError:
        pass


def model_path(cfg: Config) -> Path:
    return Path(cfg.db_path).resolve().parent / "model" / "launch4.json"


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
            "mean_ret_all": float(rets.mean()), "cut2": cut2, "cut5": cut5,
            "top2_in_sample": {"n": int(top(cut2).sum()), "mean_ret": float(rets[top(cut2)].mean())},
            "top5_in_sample": {"n": int(top(cut5).sum()), "mean_ret": float(rets[top(cut5)].mean())},
            "decide_s": DECIDE_S, "fill_s": FILL_S, "days": cfg.launch_train_days}
    return Model(names, mu.tolist(), sd.tolist(), {n: float(med[n]) for n in names}, w.tolist(), float(b),
                 cut5, meta)


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
        self.model, self.model_mtime, self.model_checked = None, None, 0.0
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
        before = self.helius.credits_used
        txs, done, ok = self.helius.history_until(pool, self.clock(), HISTORY_PAGES)
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
        self._update(pool, status="scored", reason=None if model else "no launch4 model yet", **cols)
        if self.signals is not None and (cols["top2"] or cols["top5"]):
            px = la.chain_price(rows, DECIDE_S)  # SOL per whole token (vault UI amounts)
            if sol_usd and px:
                self.signals.put_nowait({
                    "pool": pool, "mint": r["token_mint"], "name": "", "score": score, "price": px * sol_usd,
                    "liq": 2 * sol4 * sol_usd, "ts": now, "decided_ts": now, "created_ts": create,
                    "dex": "pumpswap", "kind": "launch", "top2": bool(cols["top2"]), "top5": bool(cols["top5"])})
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
        import websockets
        sub = {"jsonrpc": "2.0", "id": 1, "method": "programSubscribe",
               "params": [ps.TOKEN_2022, {"encoding": "base64", "commitment": "confirmed",
                                          "filters": [{"dataSize": ps.LP_MINT_SIZE}]}]}
        backoff = 1.0
        while not stop.is_set():
            try:
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
        return stop


# --- status ---------------------------------------------------------------------------------------

def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 2) if vals else None


def _group(rows, rng) -> dict:
    rets = [r["ret"] for r in rows if r["ret"] is not None]
    if not rets:
        return {"n": 0}
    out = {"n": len(rets), "mean_pct": round(100 * float(np.mean(rets)), 1),
           "median_pct": round(100 * float(np.median(rets)), 1),
           "rugged_pct": round(100 * sum(x <= -0.999 for x in rets) / len(rets))}
    if len(rets) >= 5:
        boots = [np.mean(rng.choice(rets, len(rets))) for _ in range(2000)]
        out["ci90"] = [round(100 * float(np.quantile(boots, q)), 1) for q in (0.05, 0.95)]
    return out


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
    if not pairs:
        return {"trades": 0}
    gaps = [x["real"] - x["ideal"] for x in pairs]
    return {"trades": len(pairs),
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
    out["picks"] = {"top 2%": sum(r["top2"] or 0 for r in scored), "top 5%": sum(r["top5"] or 0 for r in scored)}
    out["liquidity at second 4 (USD) and trade size"] = liquidity(scored, hours)
    rng = np.random.default_rng(0)
    done = [r for r in rows if r["status"] == "done"]
    out["outcomes (fill at second ~6, out at minute 6, 6% cost)"] = {
        "every liquid pool": _group(done, rng),
        "top 2%": _group([r for r in done if r["top2"]], rng),
        "top 5%": _group([r for r in done if r["top5"]], rng)}
    out["feature parity with the backtest"] = parity(store, scored)
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
