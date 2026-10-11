"""The playbook: lessons distilled from post-mortems, fed back into every new analysis.

A lesson is a tagged, testable statement. Each post-mortem can create one, and can
count as evidence for or against existing ones. Status follows the evidence:

  candidate  not enough evidence yet (not shown to the analyst)
  active     support >= lesson_min_support and support >= 2 x contradict
  retired    contradicted at least as often as supported (and at least twice)

The analyst sees active lessons, except on a random `playbook_holdout` share of
analyses (the control arm). Comparing the two arms shows whether the playbook helps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time

from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.learning.mission import MISSION
from tradingbot.llm import OK, structured_call

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS lessons (
    tag TEXT PRIMARY KEY,
    statement TEXT, applies_when TEXT,
    support INTEGER DEFAULT 0, contradict INTEGER DEFAULT 0,
    first_ts REAL, last_ts REAL,
    status TEXT DEFAULT 'candidate'
);
"""
MAX_ACTIVE = 12
MIN_LESSON_CONFIDENCE = 0.3
CONSOLIDATE_EVERY_S = 24 * 3600
CONSOLIDATE_MIN_LESSONS = 15

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {"merges": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "keep_tag": {"type": "string"},
            "merge_tags": {"type": "array", "items": {"type": "string"}},
            "statement": {"type": "string"},
            "applies_when": {"type": "string"},
        },
        "required": ["keep_tag", "merge_tags", "statement", "applies_when"],
        "additionalProperties": False,
    }}},
    "required": ["merges"],
    "additionalProperties": False,
}

CONSOLIDATE_SYSTEM = """You maintain the lesson playbook of a memecoin trading bot. Lessons were \
written one at a time by trade reviews, so several may say the same thing under different tags. \
Find groups that state the same lesson (not merely related ones) and merge each group into one \
clear, testable statement. Keep a lesson separate when its condition or action differs. Return \
an empty list if nothing should merge."""
CONSOLIDATE_SYSTEM = f"{MISSION}\n\n{CONSOLIDATE_SYSTEM}"


def clean_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (tag or "").lower()).strip("_")[:60]


class Playbook:
    def __init__(self, journal: Journal, cfg: Config, rng: random.Random | None = None):
        self.journal = journal
        self.cfg = cfg
        self.rng = rng or random.Random()
        journal.ensure_schema(SCHEMA_SQL)

    # --- evidence -------------------------------------------------------------
    def record(self, analysis: dict) -> None:
        """Fold one post-mortem into the lesson counts."""
        now = time.time()
        lesson = analysis.get("lesson") or {}
        own = clean_tag(lesson.get("tag", ""))
        if own and lesson.get("statement") and lesson.get("confidence", 0) >= MIN_LESSON_CONFIDENCE:
            self.journal.execute(
                """INSERT INTO lessons(tag, statement, applies_when, support, first_ts, last_ts)
                   VALUES(?,?,?,1,?,?)
                   ON CONFLICT(tag) DO UPDATE SET support = support + 1, last_ts = excluded.last_ts""",
                (own, lesson["statement"], lesson.get("applies_when", ""), now, now))
        for tag in {clean_tag(t) for t in analysis.get("supports_lessons", [])} - {own, ""}:
            self.journal.execute("UPDATE lessons SET support = support + 1, last_ts = ? WHERE tag = ?", (now, tag))
        for tag in {clean_tag(t) for t in analysis.get("contradicts_lessons", [])} - {own, ""}:
            self.journal.execute("UPDATE lessons SET contradict = contradict + 1, last_ts = ? WHERE tag = ?",
                                 (now, tag))
        self.refresh_status()

    def refresh_status(self) -> None:
        n = self.cfg.lesson_min_support
        self.journal.execute(
            """UPDATE lessons SET status = CASE
                 WHEN contradict >= 2 AND contradict >= support THEN 'retired'
                 WHEN support >= ? AND support >= 2 * contradict THEN 'active'
                 ELSE 'candidate' END""", (n,))

    # --- what the analyst sees ------------------------------------------------
    def active(self) -> list[dict]:
        rows = self.journal.query(
            "SELECT * FROM lessons WHERE status='active' ORDER BY support - contradict DESC, last_ts DESC LIMIT ?",
            (MAX_ACTIVE,))
        return [dict(r) for r in rows]

    def render(self) -> tuple[str, str]:
        """(text for the analyst, short version hash). Empty text when there are no active lessons."""
        lessons = self.active()
        if not lessons:
            return "", ""
        lines = [f"- [{l['tag']}] {l['statement']} (applies when: {l['applies_when'] or 'always'}; "
                 f"evidence {l['support']} for / {l['contradict']} against)" for l in lessons]
        text = ("<playbook>\nLessons from reviews of this desk's own past trades. They are advisory: weigh "
                "them against the current evidence, and say in your thesis when one changed your "
                "decision.\n" + "\n".join(lines) + "\n</playbook>")
        return text, hashlib.sha1(text.encode()).hexdigest()[:8]

    def choose_arm(self) -> tuple[str, str]:
        """('playbook' | 'control' | 'none', text to include)."""
        text, version = self.render()
        if not text:
            return "none", ""
        if self.rng.random() < self.cfg.playbook_holdout:
            return f"control:{version}", ""
        return f"playbook:{version}", text

    # --- upkeep -----------------------------------------------------------------
    def consolidate_if_due(self, client) -> float:
        """Merge duplicate lessons with Claude about once a day. Returns the cost."""
        last = float(self.journal.get_state("playbook_consolidated") or 0)
        lessons = [dict(r) for r in self.journal.query("SELECT * FROM lessons WHERE status != 'retired'")]
        if time.time() - last < CONSOLIDATE_EVERY_S or len(lessons) < CONSOLIDATE_MIN_LESSONS:
            return 0.0
        self.journal.set_state("playbook_consolidated", repr(time.time()))
        listing = "\n".join(f"- {l['tag']}: {l['statement']} (when: {l['applies_when']}; "
                            f"{l['support']} for / {l['contradict']} against)" for l in lessons)
        res = structured_call(client, self.cfg, system=CONSOLIDATE_SYSTEM, user=listing,
                              schema=CONSOLIDATE_SCHEMA, effort="low")
        if res.outcome == OK:
            self.apply_merges(res.data["merges"])
        return res.cost_usd

    def apply_merges(self, merges: list[dict]) -> None:
        for m in merges:
            keep = clean_tag(m["keep_tag"])
            group = [t for t in {clean_tag(t) for t in m["merge_tags"]} if t and t != keep]
            rows = self.journal.query(
                f"SELECT * FROM lessons WHERE tag IN ({','.join('?' * (len(group) + 1))})", (keep, *group))
            if not rows:
                continue
            support = sum(r["support"] for r in rows)
            contradict = sum(r["contradict"] for r in rows)
            first = min(r["first_ts"] for r in rows)
            last = max(r["last_ts"] for r in rows)
            for t in group:
                self.journal.execute("DELETE FROM lessons WHERE tag = ?", (t,))
            self.journal.execute(
                """INSERT INTO lessons(tag, statement, applies_when, support, contradict, first_ts, last_ts)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(tag) DO UPDATE SET statement=excluded.statement, applies_when=excluded.applies_when,
                     support=excluded.support, contradict=excluded.contradict,
                     first_ts=excluded.first_ts, last_ts=excluded.last_ts""",
                (keep, m["statement"], m["applies_when"], support, contradict, first, last))
            log.info("playbook: merged %s into %s", group, keep)
        self.refresh_status()


def arm_report(journal: Journal) -> dict:
    """Claude-book trade results and buy forward returns, split by playbook arm."""
    out = {}
    for arm in ("playbook", "control"):
        trades = journal.query(
            """SELECT p.realized_usd, p.cost_usd FROM positions p JOIN decisions d ON d.id = p.decision_id
               WHERE p.strategy='claude' AND p.closed_at IS NOT NULL AND d.playbook_arm LIKE ?""", (f"{arm}:%",))
        rets = sorted((r["realized_usd"] / r["cost_usd"] - 1) * 100 for r in trades if r["cost_usd"])
        out[arm] = {
            "closed_trades": len(rets),
            "median_trade_pct": rets[len(rets) // 2] if rets else None,
            "total_pnl_usd": round(sum(r["realized_usd"] - r["cost_usd"] for r in trades), 2),
        }
    return out


def dump(journal: Journal) -> list[dict]:
    try:
        rows = journal.query("SELECT * FROM lessons ORDER BY status='active' DESC, support - contradict DESC")
    except Exception:  # noqa: BLE001 (no lessons table yet)
        return []
    return [dict(r) for r in rows]


def as_json(journal: Journal) -> str:
    return json.dumps(dump(journal), indent=2)
