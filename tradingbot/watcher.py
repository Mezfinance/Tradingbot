"""Pump watcher: records every new Solana pool and its early numbers, for research.

To learn what predicts a pump, the data must include the tokens that did NOT pump,
measured the same way and at the same ages. Documenting only the winners would credit
them with features thousands of dead tokens also had. So the watcher:

  1. discovers new pools from GeckoTerminal (bonding-curve launches are sampled at
     `watch_launch_sample_pct`, chosen by address hash so restarts keep the same sample;
     pools on regular exchanges, which include every graduation, are all kept);
  2. snapshots each pool's numbers at fixed ages (`watch_ages_min`): price, liquidity,
     buys/sells, unique buyers/sellers and volume per window, price changes;
  3. fetches token info once (`watch_info_age_min`): socials, developer wallet and holding,
     bonding-curve progress, mint/freeze authority, holders;
  4. after 24h, fetches 5-minute candles for pools that traded (`watch_ohlcv_min_volume_usd`),
     so the analysis can measure the best price reached after any snapshot.

No Claude calls and no trading. Data lives in its own SQLite file (watch.db), backed up
daily to the bucket under `watch/`. The analysis (watch_analysis.py) reads it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from tradingbot.config import Config
from tradingbot.data.geckoterminal import NOT_MEMECOINS, GeckoTerminal, base_token, dex_id

log = logging.getLogger(__name__)

# Bonding-curve launchpads: tens of thousands of launches a day, so these are sampled.
LAUNCHPAD_DEXES = {"pump-fun", "meteora-dbc", "raydium-launchlab", "letsbonk", "moonshot", "moonit", "boop-fun",
                   "bags-fm", "stonkfun"}
REPORT_MAX_TRIES = 3  # attempts per UTC day (a crash would otherwise retry forever)
REPORT_RETRY_S = 3600
REPORT_TIMEOUT_S = 4 * 3600
REPORT_KEEP_LOCAL = 3  # dated report folders kept on the volume (all are in the bucket)
WINDOWS = ("m5", "m15", "m30", "h1", "h6", "h24")
MAX_PAGES = 3  # new_pools pages per poll; ~20 pools a page
SNAP_CALLS_PER_STEP = 4  # 30 pools per call
INFO_CALLS_PER_STEP = 4
INFO_USEFUL_S = 45 * 60  # info arriving later than this misses the 60-minute snapshot: not worth a call
INFO_MAX_TRIES = 3
INFO_RETRY_S = 600
OHLCV_CALLS_PER_STEP = 2
OHLCV_MAX_TRIES = 3
OHLCV_WINDOW_S = 300 * 5 * 60  # 300 five-minute candles: the pool's first 25 hours
DAY_S = 86400

SCHEMA = """
CREATE TABLE IF NOT EXISTS watch_pools (
    pool TEXT PRIMARY KEY,
    mint TEXT NOT NULL,
    dex TEXT,
    name TEXT,
    created_ts REAL NOT NULL,
    seen_ts REAL NOT NULL,
    first TEXT,                 -- compact numbers when first seen (JSON)
    info TEXT, info_ts REAL,    -- token info (JSON), '{}' when unavailable
    ohlcv TEXT, ohlcv_ts REAL,  -- 5-min candles (JSON) or '[]'
    closed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS watch_pools_created ON watch_pools(closed, created_ts);
CREATE INDEX IF NOT EXISTS watch_pools_mint ON watch_pools(mint);
CREATE TABLE IF NOT EXISTS watch_snaps (
    pool TEXT NOT NULL,
    age_min INTEGER NOT NULL,
    ts REAL NOT NULL,
    data TEXT,                  -- compact numbers (JSON); NULL = missed (too late), '{}' = pool not returned
    PRIMARY KEY (pool, age_min)
);
CREATE TABLE IF NOT EXISTS watch_state (key TEXT PRIMARY KEY, value TEXT);
-- enrichment (enrich.py): one row per pool and source, data JSON ('{}' = tried, nothing usable)
CREATE TABLE IF NOT EXISTS watch_enrich (
    pool TEXT NOT NULL, kind TEXT NOT NULL, ts REAL NOT NULL, data TEXT,
    PRIMARY KEY (pool, kind)
);
CREATE INDEX IF NOT EXISTS watch_enrich_ts ON watch_enrich(ts);
-- attention lists over time (DexScreener boosts/profiles, GeckoTerminal trending)
CREATE TABLE IF NOT EXISTS watch_lists (ts REAL NOT NULL, list TEXT NOT NULL, rank INTEGER, mint TEXT, amount REAL);
CREATE INDEX IF NOT EXISTS watch_lists_mint ON watch_lists(mint, ts);
-- contract addresses posted in public Telegram channels
CREATE TABLE IF NOT EXISTS watch_mentions (
    source TEXT NOT NULL, channel TEXT, post TEXT NOT NULL, mint TEXT NOT NULL, post_ts REAL, seen_ts REAL,
    PRIMARY KEY (source, post, mint)
);
CREATE INDEX IF NOT EXISTS watch_mentions_mint ON watch_mentions(mint);
-- market conditions: SOL price and how many new pools appeared (before sampling)
CREATE TABLE IF NOT EXISTS watch_market (ts REAL NOT NULL, sol_usd REAL, new_pools INTEGER, new_launchpad INTEGER);
-- market-wide numbers over time (long format: key, value)
CREATE TABLE IF NOT EXISTS watch_macro (ts REAL NOT NULL, key TEXT NOT NULL, value REAL);
CREATE INDEX IF NOT EXISTS watch_macro_key ON watch_macro(key, ts);
-- latest posts of watched mega-accounts (X), for theme matching
CREATE TABLE IF NOT EXISTS watch_xfeed (ts REAL NOT NULL, account TEXT, post_ts REAL, text TEXT);
-- wallet origins (Helius), cached across tokens
CREATE TABLE IF NOT EXISTS watch_wallets (wallet TEXT PRIMARY KEY, first_ts REAL, funder TEXT, checked_ts REAL);
"""


def watch_db_path(cfg: Config) -> str:
    if cfg.watch_db_path:
        return cfg.watch_db_path
    return str(Path(cfg.db_path).resolve().parent / "watch.db")


def sampled(address: str, pct: float) -> bool:
    """Stable pseudo-random choice by address, so a restart keeps the same sample."""
    return int(hashlib.sha1(address.encode()).hexdigest()[:8], 16) % 10_000 < pct * 100


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compact(attrs: dict) -> dict:
    """The numbers worth keeping from a GeckoTerminal pool, flattened (e.g. buyers_m5, vol_h1)."""
    out: dict = {
        "price": _num(attrs.get("base_token_price_usd")),
        "fdv": _num(attrs.get("fdv_usd")),
        "mcap": _num(attrs.get("market_cap_usd")),
        "liq": _num(attrs.get("reserve_in_usd")),
    }
    tx = attrs.get("transactions") or {}
    vol = attrs.get("volume_usd") or {}
    chg = attrs.get("price_change_percentage") or {}
    for w in WINDOWS:
        t = tx.get(w) or {}
        for k in ("buys", "sells", "buyers", "sellers"):
            out[f"{k}_{w}"] = t.get(k)
        out[f"vol_{w}"] = _num(vol.get(w))
        out[f"chg_{w}"] = _num(chg.get(w))
    return out


def compact_info(info: dict | None) -> dict:
    if not info:
        return {}
    holders = info.get("holders") or {}
    launch = info.get("launchpad_details") or {}
    image = info.get("image") or {}
    return {
        "websites": len(info.get("websites") or []),
        "website_domains": [_domain(w) for w in (info.get("websites") or [])][:3],
        "twitter_handle": (info.get("twitter_handle") or "")[:120],
        "telegram_handle": (info.get("telegram_handle") or "")[:80],
        "description": (info.get("description") or "")[:400],
        "image_url": image.get("thumb") or info.get("image_url") or "",
        "twitter": bool(info.get("twitter_handle")),
        "telegram": bool(info.get("telegram_handle")),
        "discord": bool(info.get("discord_url")),
        "description_len": len(info.get("description") or ""),
        "gt_score": _num(info.get("gt_score")),
        "mint_authority": info.get("mint_authority"),
        "freeze_authority": info.get("freeze_authority"),
        "is_honeypot": info.get("is_honeypot"),
        "developer": info.get("developer_address"),
        "dev_holding_pct": _num(info.get("developer_holding_percentage")),
        "holders": holders.get("count"),
        "holder_distribution": holders.get("distribution_percentage"),
        "graduation_pct": _num(launch.get("graduation_percentage")),
        "graduated": launch.get("completed"),
        "categories": info.get("categories") or [],
        "coingecko_id": info.get("coingecko_coin_id") or "",
    }


def _domain(url: str) -> str:
    url = (url or "").lower().split("://", 1)[-1]
    return url.split("/", 1)[0].removeprefix("www.")[:80]


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class WatchStore:
    """Thread-safe SQLite store for the watcher (separate file from the trading journal)."""

    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            if path != ":memory:":
                # WAL: readers in other processes (the report, `fly machine exec` commands) never block the
                # watcher's writes, and its writes never block them.
                self.db.execute("PRAGMA journal_mode=WAL")
                self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript(SCHEMA)
            cols = {r[1] for r in self.db.execute("PRAGMA table_info(watch_pools)")}
            if "ohlcv_end" not in cols:  # end of the candle window asked for (NULL: old 'latest 300' fetch)
                self.db.execute("ALTER TABLE watch_pools ADD COLUMN ohlcv_end REAL")
            self.db.commit()

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.db.execute(sql, params)
            self.db.commit()
            return cur

    def executemany(self, sql: str, rows) -> None:
        with self.lock:
            self.db.executemany(sql, rows)
            self.db.commit()

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, params).fetchall()

    def get_state(self, key: str) -> str | None:
        rows = self.query("SELECT value FROM watch_state WHERE key=?", (key,))
        return rows[0]["value"] if rows else None

    def set_state(self, key: str, value: str) -> None:
        self.execute("INSERT INTO watch_state(key, value) VALUES(?,?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def backup(self, dest: str) -> None:
        with self.lock:
            target = sqlite3.connect(dest)
            try:
                self.db.backup(target)
            finally:
                target.close()


class Watcher:
    def __init__(self, cfg: Config, gecko: GeckoTerminal, store: WatchStore, s3_factory=None,
                 report_in_subprocess: bool = False):
        self.cfg, self.gecko, self.store = cfg, gecko, store
        self.ages = sorted(int(a) for a in cfg.watch_ages_min.split(",") if a.strip())
        self.s3_factory = s3_factory
        self.report_in_subprocess = report_in_subprocess
        self._seen: dict[str, bool] = {}  # every new pool seen recently -> is a launchpad pool (before sampling)
        self.seen_counts = [0, 0]  # new pools, of which launchpad, since the market row was last written

    @classmethod
    def from_config(cls, cfg: Config) -> "Watcher":
        from tradingbot.backups import s3_client
        return cls(cfg, GeckoTerminal(), WatchStore(watch_db_path(cfg)), s3_factory=s3_client,
                   report_in_subprocess=True)

    # --- one pass ----------------------------------------------------------------
    def step(self, now: float | None = None) -> dict:
        now = now or time.time()
        done = {"new": self.discover(now), "snaps": self.snapshot(now), "info": self.fetch_info(now),
                "ohlcv": self.fetch_ohlcv(now)}
        self.backup_if_due()
        self.report_if_due()
        self.store.set_state("heartbeat", repr(time.time()))
        return done

    def discover(self, now: float) -> int:
        added = 0
        for page in range(1, MAX_PAGES + 1):
            pools = self.gecko.new_pools(page)
            if not pools:
                break
            rows, known = [], 0
            for p in pools:
                attrs = p.get("attributes") or {}
                addr, mint, dex = attrs.get("address"), base_token(p), dex_id(p)
                if not addr or not mint or mint in NOT_MEMECOINS:
                    continue
                if addr not in self._seen:
                    self._seen[addr] = dex in LAUNCHPAD_DEXES
                    self.seen_counts[0] += 1
                    self.seen_counts[1] += dex in LAUNCHPAD_DEXES
                    if len(self._seen) > 20_000:
                        for old in list(self._seen)[:10_000]:
                            del self._seen[old]
                if self.store.query("SELECT 1 FROM watch_pools WHERE pool=?", (addr,)):
                    known += 1
                    continue
                if dex in LAUNCHPAD_DEXES and not sampled(addr, self.cfg.watch_launch_sample_pct):
                    continue
                created = _ts(attrs.get("pool_created_at")) or now
                rows.append((addr, mint, dex, attrs.get("name"), created, now, json.dumps(compact(attrs))))
            self.store.executemany(
                "INSERT OR IGNORE INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts, first) "
                "VALUES(?,?,?,?,?,?,?)", rows)
            added += len(rows)
            if known:  # caught up with what we already have
                break
        return added

    def snapshot(self, now: float) -> int:
        """Take due snapshots, youngest ages first (early numbers are the predictive ones)."""
        budget = SNAP_CALLS_PER_STEP * 30
        taken = 0
        for age in self.ages:
            if budget <= 0:
                break
            due = self.store.query(
                """SELECT pool, created_ts FROM watch_pools p WHERE closed=0 AND created_ts <= ?
                   AND NOT EXISTS (SELECT 1 FROM watch_snaps s WHERE s.pool=p.pool AND s.age_min=?)
                   ORDER BY created_ts LIMIT ?""", (now - age * 60, age, budget))
            if not due:
                continue
            late_limit = max(600.0, age * 60 * 0.5)  # a snapshot this late would mislabel its age
            missed = [(r["pool"], age, now, None) for r in due if now - (r["created_ts"] + age * 60) > late_limit]
            fresh = [r["pool"] for r in due if now - (r["created_ts"] + age * 60) <= late_limit]
            data = self.gecko.pools(fresh) if fresh else {}
            rows = [(p, age, now, json.dumps(compact(data[p])) if p in data else "{}") for p in fresh]
            self.store.executemany("INSERT OR IGNORE INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
                                   rows + missed)
            taken += len(rows)
            budget -= len(due)
        return taken

    def fetch_info(self, now: float) -> int:
        """Token info once per pool, newest pools first: info only feeds a factor if it was fetched before the
        snapshot being scored, so a pool still waiting INFO_USEFUL_S after it became due is given up on (its
        15/30/60-minute snapshots are taken by then). A failed fetch (usually GeckoTerminal's rate limit) is
        retried up to INFO_MAX_TRIES times, INFO_RETRY_S apart."""
        due_after = self.cfg.watch_info_age_min * 60
        self.store.execute("UPDATE watch_pools SET info='{}', info_ts=? WHERE info_ts IS NULL AND created_ts < ?",
                           (now, now - due_after - INFO_USEFUL_S))
        due = self.store.query(
            """SELECT pool, mint, created_ts FROM watch_pools WHERE info_ts IS NULL AND created_ts <= ?
               ORDER BY created_ts DESC LIMIT ?""", (now - due_after, INFO_CALLS_PER_STEP * 20))
        tries = getattr(self, "_info_tries", None)
        if tries is None:
            tries = self._info_tries = {}
        calls = 0
        for r in due:
            n, next_ts = tries.get(r["pool"], (0, 0.0))
            if n >= INFO_MAX_TRIES:
                self.store.execute("UPDATE watch_pools SET info='{}', info_ts=? WHERE pool=?", (now, r["pool"]))
                tries.pop(r["pool"], None)
                continue
            if now < next_ts:
                continue
            if calls >= INFO_CALLS_PER_STEP:
                break
            info = self.gecko.token_info(r["mint"])
            calls += 1
            if info is None:  # request failed: try again later instead of recording "no info"
                tries[r["pool"]] = (n + 1, now + INFO_RETRY_S)
                continue
            tries.pop(r["pool"], None)
            self.store.execute("UPDATE watch_pools SET info=?, info_ts=? WHERE pool=?",
                               (json.dumps(compact_info(info)), now, r["pool"]))
        return calls

    def fetch_ohlcv(self, now: float) -> int:
        """After 24h: 5-min candles of the pool's first OHLCV_WINDOW_S (asked for by end time, so a late fetch
        still gets the right window) for pools that traded; quiet pools are closed without them. Spare calls
        re-fetch pools whose candles were taken as 'the latest 300' before this fix (`ohlcv_end` NULL)."""
        last_age = self.ages[-1] if self.ages else 1440
        due = self.store.query(
            """SELECT p.pool, p.created_ts, s.data FROM watch_pools p
               LEFT JOIN watch_snaps s ON s.pool=p.pool AND s.age_min=?
               WHERE p.closed=0 AND p.created_ts <= ? ORDER BY p.created_ts LIMIT 50""",
            (last_age, now - max(last_age * 60, DAY_S) - 1800))
        tries = getattr(self, "_ohlcv_tries", None)
        if tries is None:
            tries = self._ohlcv_tries = {}
        calls = 0
        for r in due:
            snap = json.loads(r["data"]) if r["data"] else {}
            vol = snap.get("vol_h24") or 0
            if vol < self.cfg.watch_ohlcv_min_volume_usd:
                self.store.execute("UPDATE watch_pools SET ohlcv='[]', ohlcv_ts=?, closed=1 WHERE pool=?",
                                   (now, r["pool"]))
                continue
            if calls >= OHLCV_CALLS_PER_STEP:
                break
            calls += 1
            if self._store_candles(r["pool"], r["created_ts"], now):
                tries.pop(r["pool"], None)
            elif tries.get(r["pool"], 0) + 1 >= OHLCV_MAX_TRIES:  # keeps failing: close without candles
                self.store.execute("UPDATE watch_pools SET ohlcv='[]', ohlcv_ts=?, closed=1 WHERE pool=?",
                                   (now, r["pool"]))
                tries.pop(r["pool"], None)
            else:
                tries[r["pool"]] = tries.get(r["pool"], 0) + 1
        while calls < OHLCV_CALLS_PER_STEP and self._refetch_queue():
            pool, created = self._refetch.pop()
            calls += 1
            self._store_candles(pool, created, now)
        return calls

    def _store_candles(self, pool: str, created: float, now: float) -> bool:
        end = created + OHLCV_WINDOW_S
        candles = self.gecko.ohlcv(pool, before=int(end))
        if candles is None:
            return False
        self.store.execute("UPDATE watch_pools SET ohlcv=?, ohlcv_ts=?, ohlcv_end=?, closed=1 WHERE pool=?",
                           (json.dumps([c for c in candles if c[0] >= created - 300]), now, end, pool))
        return True

    def _refetch_queue(self) -> list:
        """Pools whose candles were fetched late as 'the latest 300' (missing the start), built once per run."""
        if getattr(self, "_refetch", None) is None:
            # 'Latest 300' counts only 5-min periods with trades, so most late fetches still start at creation;
            # only busy tokens (the interesting ones) lost their start. Read once per run.
            self._refetch = [(r["pool"], r["created_ts"]) for r in self.store.query(
                """SELECT pool, created_ts, ohlcv FROM watch_pools WHERE ohlcv_end IS NULL AND closed=1
                   AND ohlcv_ts > created_ts + ? AND ohlcv != '[]' ORDER BY created_ts""",
                (OHLCV_WINDOW_S,)) if json.loads(r["ohlcv"])[0][0] > r["created_ts"] + 3600][::-1]
            # (reversed so pop() takes the oldest first: its history is the first to age out)
        return self._refetch

    # --- backup ------------------------------------------------------------------
    def backup_if_due(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        date = now.strftime("%Y-%m-%d")
        if self.store.path == ":memory:" or self.store.get_state("last_backup") == date:
            return
        try:
            out_dir = Path(self.store.path).resolve().parent / "backups"
            out_dir.mkdir(parents=True, exist_ok=True)
            raw, final = out_dir / f"watch-{date}.db", out_dir / f"watch-{date}.db.gz"
            self.store.backup(str(raw))
            try:
                with open(raw, "rb") as src, gzip.open(final, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
            finally:
                raw.unlink(missing_ok=True)
            for old in sorted(out_dir.glob("watch-*.db.gz"))[:-self.cfg.watch_backup_keep_local]:
                old.unlink()
            client = self.s3_factory() if self.s3_factory else None
            if client is not None:
                client.upload_file(str(final), os.environ["BUCKET_NAME"], f"watch/{final.name}")
            log.info("watch backup %s: %.1f MB%s", date, final.stat().st_size / 1e6,
                     " -> bucket" if client is not None else "")
        except Exception as exc:  # noqa: BLE001 (a failed backup must not stop collection)
            log.error("watch backup failed: %s: %s", type(exc).__name__, exc)
        self.store.set_state("last_backup", date)

    # --- daily report --------------------------------------------------------------
    def report_if_due(self, now: datetime | None = None, wait: bool = False) -> threading.Thread | None:
        """Once a UTC day, write the analysis to <db dir>/reports/watch-DATE/ and the bucket.

        Runs in its own thread so a slow analysis never holds up data collection, and (on the server) in its
        own process, so running out of memory kills the report and never the bot. A failed attempt is retried
        up to REPORT_MAX_TRIES times, REPORT_RETRY_S apart; `last_report` is set only when a report is written.
        """
        now = now or datetime.now(timezone.utc)
        date = now.strftime("%Y-%m-%d")
        if (self.store.path == ":memory:" or self.store.get_state("last_report") == date
                or getattr(self, "_report_thread", None) is not None and self._report_thread.is_alive()):
            return None
        tries, last = (self.store.get_state(f"report_try:{date}") or "0,0").split(",")
        if int(tries) >= REPORT_MAX_TRIES or now.timestamp() - float(last) < REPORT_RETRY_S:
            return None
        self.store.set_state(f"report_try:{date}", f"{int(tries) + 1},{now.timestamp()}")
        target = self._report_in_child if self.report_in_subprocess else self.write_report
        t = threading.Thread(target=target, args=(date,), name="watch-report", daemon=True)
        self._report_thread = t
        t.start()
        if wait:
            t.join()
        return t

    def _report_in_child(self, date: str) -> None:
        """`python -m tradingbot watch-report-now DATE` in a child process that the kernel kills first when
        memory runs out (oom_score_adj 1000), so the bot keeps running whatever the report needs."""
        def first_to_die():
            try:
                with open("/proc/self/oom_score_adj", "w") as fh:
                    fh.write("1000")
            except OSError:
                pass
        try:
            res = subprocess.run([sys.executable, "-m", "tradingbot", "watch-report-now", date],
                                 preexec_fn=first_to_die, timeout=REPORT_TIMEOUT_S)
            if res.returncode != 0:
                log.error("watch report %s: child exited %s%s", date, res.returncode,
                          " (killed: out of memory?)" if res.returncode in (-9, 137) else "")
        except Exception as exc:  # noqa: BLE001 (a failed report must not stop collection)
            log.error("watch report %s could not run: %s: %s", date, type(exc).__name__, exc)

    def write_report(self, date: str) -> Path | None:
        from tradingbot import watch_analysis as wa
        try:
            root = Path(self.store.path).resolve().parent / "reports"
            out = root / f"watch-{date}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "status.json").write_text(json.dumps(self.status(), indent=2))
            labels, subsets = ("pump2x", "clean2x", "pump5x"), (None, "enriched")
            extra: dict = {}

            def per_age(age: int, rows: list[dict]) -> None:
                if age == 15:  # is Helius / X worth paying for (15-min features)
                    extra["paid_value"] = wa.paid_value(rows, "pump2x")
                raw = out / f"rows-{age}m.csv"
                if wa.export_csv(self.store, str(raw), age, rows=rows):
                    with open(raw, "rb") as src, gzip.open(f"{raw}.gz", "wb", compresslevel=6) as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
                raw.unlink(missing_ok=True)

            res = wa.report_by_age(self.store, labels, subsets, detail_age=15, on_rows=per_age)
            full: dict = {"paid_value_15m": extra.get("paid_value")}
            text = [wa.render_paid_value(extra["paid_value"]), ""] if extra.get("paid_value") else []
            for label in labels:
                for subset in subsets:
                    summ, detail = res["summary"][(label, subset)], res["detail"][(label, subset)]
                    full[f"{label}:{subset or 'all'}"] = {"summary": summ, "detail_15m": detail}
                    text += [f"===== {label}, {subset or 'all'} tokens =====", wa.render(detail, summ), ""]
            (out / "report.json").write_text(json.dumps(full, indent=2, default=str))
            (out / "report.txt").write_text("\n".join(text))
            for old in sorted(root.glob("watch-*"))[:-REPORT_KEEP_LOCAL]:
                shutil.rmtree(old, ignore_errors=True)
            client = self.s3_factory() if self.s3_factory else None
            if client is not None:
                for f in sorted(out.iterdir()):
                    client.upload_file(str(f), os.environ["BUCKET_NAME"], f"watch/reports/{out.name}/{f.name}")
            self.store.set_state("last_report", date)
            log.info("watch report %s written%s", date, " and uploaded" if client is not None else "")
            return out
        except Exception as exc:  # noqa: BLE001 (a failed report must not stop collection)
            log.error("watch report failed: %s: %s", type(exc).__name__, exc)
            return None

    # --- status ------------------------------------------------------------------
    def status(self) -> dict:
        q = self.store.query
        day_ago = time.time() - DAY_S
        return {
            "pools": q("SELECT COUNT(*) AS n FROM watch_pools")[0]["n"],
            "pools_last_24h": q("SELECT COUNT(*) AS n FROM watch_pools WHERE seen_ts > ?", (day_ago,))[0]["n"],
            "by_dex": {r["dex"]: r["n"] for r in q("SELECT dex, COUNT(*) AS n FROM watch_pools GROUP BY dex "
                                                   "ORDER BY n DESC LIMIT 8")},
            "snapshots": {r["age_min"]: r["n"] for r in q(
                "SELECT age_min, COUNT(*) AS n FROM watch_snaps WHERE data IS NOT NULL GROUP BY age_min")},
            "missed_snapshots": q("SELECT COUNT(*) AS n FROM watch_snaps WHERE data IS NULL")[0]["n"],
            "with_info": q("SELECT COUNT(*) AS n FROM watch_pools WHERE info IS NOT NULL AND info != '{}'")[0]["n"],
            "with_candles": q("SELECT COUNT(*) AS n FROM watch_pools WHERE ohlcv IS NOT NULL AND ohlcv != '[]'")[0]["n"],
            "closed": q("SELECT COUNT(*) AS n FROM watch_pools WHERE closed=1")[0]["n"],
            "last_backup": self.store.get_state("last_backup"),
            "db_mb": round(os.path.getsize(self.store.path) / 1e6, 1) if self.store.path != ":memory:" else 0,
        }


LINK_EXPIRY_S = 7 * DAY_S  # the longest an S3 presigned link may live


def download_links(client, bucket: str, expires_s: int = LINK_EXPIRY_S) -> dict:
    """Presigned download links for the newest watch report and database backup in the bucket."""
    keys: list[str] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": "watch/"}
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    reports = sorted({k.split("/")[2] for k in keys if k.startswith("watch/reports/") and k.count("/") >= 3})
    backups = sorted(k for k in keys if k.startswith("watch/watch-") and k.endswith(".db.gz"))

    def link(key: str) -> str:
        return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                             ExpiresIn=expires_s)

    latest = reports[-1] if reports else None
    return {
        "latest_report": latest,
        "report_files": {k.rsplit("/", 1)[1]: link(k) for k in keys if latest and k.startswith(f"watch/reports/{latest}/")},
        "latest_backup": {backups[-1].rsplit("/", 1)[1]: link(backups[-1])} if backups else {},
        "older_reports": reports[:-1],
        "links_valid_days": round(expires_s / DAY_S, 1),
    }
