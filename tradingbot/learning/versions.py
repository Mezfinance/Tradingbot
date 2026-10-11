"""Versioned settings chosen by the optimiser, applied live and on every start-up.

Each version stores the full set of tunable values, so rolling back is just
re-applying the parent. Version history lives in `config_versions`.
"""

from __future__ import annotations

import json
import logging
import time

from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.learning.params import TP_KEYS, TUNABLES, current_values

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS config_versions (
    id INTEGER PRIMARY KEY,
    ts REAL, parent_id INTEGER,
    settings TEXT,          -- full tunable values (JSON)
    changes TEXT,           -- only what changed vs the parent (JSON)
    reason TEXT, evidence TEXT,
    status TEXT,            -- active | superseded | rolled_back
    verdict TEXT            -- result of the post-change review, if any
);
CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY,
    ts REAL, source TEXT, title TEXT, problem TEXT, evidence TEXT, idea TEXT,
    priority INTEGER DEFAULT 3,
    status TEXT DEFAULT 'open',   -- open | building | done | rejected
    result TEXT
);
"""


def ensure(journal: Journal) -> None:
    journal.db.executescript(SCHEMA_SQL)


def apply_to(cfg: Config, settings: dict) -> None:
    """Set tunable values on the live Config object (shared by every component)."""
    tps = list(cfg.take_profit_pcts)
    for name, value in settings.items():
        if name in TP_KEYS:
            tps[TP_KEYS.index(name)] = float(value)
        elif name in TUNABLES:
            setattr(cfg, name, type(getattr(cfg, name))(value))
    cfg.take_profit_pcts = tps


def active(journal: Journal) -> dict | None:
    ensure(journal)
    rows = journal.query("SELECT * FROM config_versions WHERE status='active' ORDER BY id DESC LIMIT 1")
    return dict(rows[0]) if rows else None


def apply_active(journal: Journal, cfg: Config) -> int | None:
    """On start-up: re-apply the optimiser's active settings over the env-var defaults."""
    v = active(journal)
    if v is None:
        return None
    apply_to(cfg, json.loads(v["settings"]))
    log.info("applied learned settings version %d: %s", v["id"], v["changes"])
    return v["id"]


def ensure_baseline(journal: Journal, cfg: Config) -> int:
    """Version 1 is whatever the bot started with, so there is always something to roll back to."""
    v = active(journal)
    if v is not None:
        return v["id"]
    return journal.execute(
        """INSERT INTO config_versions(ts, parent_id, settings, changes, reason, evidence, status)
           VALUES(?, NULL, ?, '{}', 'baseline', '{}', 'active')""",
        (time.time(), json.dumps(current_values(cfg))))


def activate(journal: Journal, cfg: Config, changes: dict, reason: str, evidence: dict) -> int:
    parent = active(journal)
    parent_id = parent["id"] if parent else ensure_baseline(journal, cfg)
    settings = {**current_values(cfg), **changes}
    journal.execute("UPDATE config_versions SET status='superseded' WHERE status='active'")
    vid = journal.execute(
        """INSERT INTO config_versions(ts, parent_id, settings, changes, reason, evidence, status)
           VALUES(?,?,?,?,?,?, 'active')""",
        (time.time(), parent_id, json.dumps(settings), json.dumps(changes), reason, json.dumps(evidence)))
    apply_to(cfg, settings)
    log.warning("learned settings v%d now live: %s (%s)", vid, changes, reason)
    return vid


def rollback(journal: Journal, cfg: Config, why: str) -> int | None:
    v = active(journal)
    if v is None or v["parent_id"] is None:
        return None
    journal.execute("UPDATE config_versions SET status='rolled_back', verdict=? WHERE id=?", (why, v["id"]))
    journal.execute("UPDATE config_versions SET status='active' WHERE id=?", (v["parent_id"],))
    parent = journal.query("SELECT settings FROM config_versions WHERE id=?", (v["parent_id"],))[0]
    apply_to(cfg, json.loads(parent["settings"]))
    log.warning("rolled back settings v%d -> v%d: %s", v["id"], v["parent_id"], why)
    return v["parent_id"]


def history(journal: Journal) -> list[dict]:
    ensure(journal)
    return [dict(r) for r in journal.query("SELECT * FROM config_versions ORDER BY id DESC")]
