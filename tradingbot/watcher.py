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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from tradingbot.config import Config
from tradingbot.data.geckoterminal import NOT_MEMECOINS, GeckoTerminal, base_token, dex_id

log = logging.getLogger(__name__)

# Bonding-curve launchpads: tens of thousands of launches a day, so these are sampled.
LAUNCHPAD_DEXES = {"pump-fun", "meteora-dbc", "raydium-launchlab", "letsbonk", "moonshot", "boop-fun", "bags-fm"}
WINDOWS = ("m5", "m15", "m30", "h1", "h6", "h24")
MAX_PAGES = 3  # new_pools pages per poll; ~20 pools a page
SNAP_CALLS_PER_STEP = 4  # 30 pools per call
INFO_CALLS_PER_STEP = 3
OHLCV_CALLS_PER_STEP = 1
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
    return {
        "websites": len(info.get("websites") or []),
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
    }


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
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)
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
    def __init__(self, cfg: Config, gecko: GeckoTerminal, store: WatchStore, s3_factory=None):
        self.cfg, self.gecko, self.store = cfg, gecko, store
        self.ages = sorted(int(a) for a in cfg.watch_ages_min.split(",") if a.strip())
        self.s3_factory = s3_factory

    @classmethod
    def from_config(cls, cfg: Config) -> "Watcher":
        from tradingbot.backups import s3_client
        return cls(cfg, GeckoTerminal(), WatchStore(watch_db_path(cfg)), s3_factory=s3_client)

    # --- one pass ----------------------------------------------------------------
    def step(self, now: float | None = None) -> dict:
        now = now or time.time()
        done = {"new": self.discover(now), "snaps": self.snapshot(now), "info": self.fetch_info(now),
                "ohlcv": self.fetch_ohlcv(now)}
        self.backup_if_due()
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
        due = self.store.query(
            """SELECT pool, mint, created_ts FROM watch_pools WHERE info_ts IS NULL AND created_ts <= ?
               ORDER BY created_ts LIMIT ?""", (now - self.cfg.watch_info_age_min * 60, INFO_CALLS_PER_STEP * 4))
        calls = 0
        for r in due:
            if now - r["created_ts"] > self.cfg.watch_info_age_min * 60 + 7200:  # too late to mean "at 15 min"
                self.store.execute("UPDATE watch_pools SET info='{}', info_ts=? WHERE pool=?", (now, r["pool"]))
                continue
            if calls >= INFO_CALLS_PER_STEP:
                break
            info = self.gecko.token_info(r["mint"])
            calls += 1
            self.store.execute("UPDATE watch_pools SET info=?, info_ts=? WHERE pool=?",
                               (json.dumps(compact_info(info)), now, r["pool"]))
        return calls

    def fetch_ohlcv(self, now: float) -> int:
        """After 24h: candles for pools that traded; quiet pools are closed without them."""
        last_age = self.ages[-1] if self.ages else 1440
        due = self.store.query(
            """SELECT p.pool, s.data FROM watch_pools p
               LEFT JOIN watch_snaps s ON s.pool=p.pool AND s.age_min=?
               WHERE p.closed=0 AND p.created_ts <= ? ORDER BY p.created_ts LIMIT 50""",
            (last_age, now - max(last_age * 60, DAY_S) - 1800))
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
            candles = self.gecko.ohlcv(r["pool"])
            calls += 1
            self.store.execute("UPDATE watch_pools SET ohlcv=?, ohlcv_ts=?, closed=1 WHERE pool=?",
                               (json.dumps(candles or []), now, r["pool"]))
        return calls

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
