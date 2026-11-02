"""SQLite journal for the paper short book. One writer (the book's loop); reads from the CLI are safe (WAL)."""
from __future__ import annotations
import json, sqlite3, threading, time

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (venue TEXT, base TEXT, symbol TEXT, first_seen REAL, PRIMARY KEY (venue, base));
CREATE TABLE IF NOT EXISTS listings (
  id INTEGER PRIMARY KEY, venue TEXT, base TEXT, symbol TEXT, seen_ts REAL, trade_ts REAL,
  kind TEXT, status TEXT, reason TEXT, data TEXT, UNIQUE (venue, base));
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY, listing_id INTEGER, group_name TEXT, base TEXT, venue TEXT, symbol TEXT, mult REAL,
  size_usd REAL, entry_px REAL, entry_ts REAL, due_ts REAL, fees_usd REAL, impact_pct REAL,
  funding_usd REAL DEFAULT 0, funding_last_ts REAL, last_px REAL, last_ts REAL,
  exit_px REAL, exit_ts REAL, pnl_usd REAL, status TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS funding (position_id INTEGER, ts REAL, rate REAL, usd REAL, PRIMARY KEY (position_id, ts));
CREATE TABLE IF NOT EXISTS equity (ts REAL PRIMARY KEY, nav REAL, cash REAL, open_usd REAL, unrealized REAL);
CREATE TABLE IF NOT EXISTS events (ts REAL, kind TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS state (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    def q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def x(self, sql: str, args=()) -> int:
        with self.lock:
            return self.db.execute(sql, args).lastrowid

    def get(self, k: str, default=None):
        r = self.q("SELECT v FROM state WHERE k=?", (k,))
        return json.loads(r[0]["v"]) if r else default

    def put(self, k: str, v) -> None:
        self.x("INSERT OR REPLACE INTO state (k, v) VALUES (?, ?)", (k, json.dumps(v)))

    def event(self, kind: str, **data) -> None:
        self.x("INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)", (time.time(), kind, json.dumps(data, default=str)))
