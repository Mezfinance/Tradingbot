"""Background learning loop. Runs on its own thread so trading never waits for it.

Each step, within a daily spending cap:
  1. post-mortems for Claude-book trades closed at least `postmortem_delay_h` ago
     (the delay lets the review see what happened after the exit),
  2. post-mortems for tokens Claude passed on that later rose `missed_winner_multiple`x,
  3. playbook upkeep, the settings optimiser and the code engineer (later parts).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.learning import dossier, postmortem
from tradingbot.learning.playbook import Playbook
from tradingbot.llm import OK, UNAVAILABLE, LLMResult

log = logging.getLogger(__name__)

LEARNING_COST_KINDS = ("postmortem", "playbook", "optimizer", "engineer")
UNAVAILABLE_PAUSE_S = 600


class Learner:
    def __init__(self, journal: Journal, cfg: Config, client=None):
        self.journal = journal
        self.cfg = cfg
        self._client = client
        self.paused_until = 0.0
        self.playbook = Playbook(journal, cfg)

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # --- budget -------------------------------------------------------------
    def spent_today(self) -> float:
        day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        marks = ",".join("?" * len(LEARNING_COST_KINDS))
        return self.journal.query(
            f"SELECT COALESCE(SUM(usd), 0) AS s FROM costs WHERE ts >= ? AND kind IN ({marks})",
            (day, *LEARNING_COST_KINDS))[0]["s"]

    def can_spend(self) -> bool:
        return time.time() >= self.paused_until and self.spent_today() < self.cfg.learning_budget_usd_per_day

    def record(self, kind: str, res: LLMResult, token: str | None = None) -> None:
        if res.cost_usd:
            self.journal.log_cost(kind, res.cost_usd, token)
        if res.outcome == UNAVAILABLE:
            self.paused_until = time.time() + UNAVAILABLE_PAUSE_S
            log.error("learning paused for %d min: Claude API unavailable (%s)", UNAVAILABLE_PAUSE_S // 60, res.error)

    # --- work queue -----------------------------------------------------------
    def pending_trades(self) -> list:
        return self.journal.query(
            """SELECT p.id, p.symbol, p.token, p.decision_id FROM positions p
               WHERE p.strategy='claude' AND p.closed_at IS NOT NULL AND p.closed_at < ?
                 AND NOT EXISTS (SELECT 1 FROM postmortems m WHERE m.kind='trade' AND m.position_id=p.id)
               ORDER BY p.closed_at""",
            (time.time() - self.cfg.postmortem_delay_h * 3600,))

    def pending_missed(self) -> list:
        return self.journal.query(
            """SELECT t.decision_id, t.token, d.symbol FROM tracking t JOIN decisions d ON d.id=t.decision_id
               WHERE t.done=1 AND d.stage='analysed' AND d.action='pass' AND t.start_price > 0
                 AND t.max_price >= t.start_price * ?
                 AND NOT EXISTS (SELECT 1 FROM postmortems m WHERE m.kind='missed' AND m.decision_id=t.decision_id)
               ORDER BY t.start_ts""",
            (self.cfg.missed_winner_multiple,))

    def missed_today(self) -> int:
        day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        return self.journal.query("SELECT COUNT(*) AS n FROM postmortems WHERE kind='missed' AND ts >= ?",
                                  (day,))[0]["n"]

    # --- steps ----------------------------------------------------------------
    def step(self) -> int:
        """Do up to `learning_max_per_cycle` reviews. Returns how many were completed."""
        done = 0
        for row in self.pending_trades():
            if done >= self.cfg.learning_max_per_cycle or not self.can_spend():
                return done
            done += self._review_trade(row)
        for row in self.pending_missed():
            if (done >= self.cfg.learning_max_per_cycle or not self.can_spend()
                    or self.missed_today() >= self.cfg.missed_reviews_per_day):
                break
            done += self._review_missed(row)
        if self.can_spend():
            cost = self.playbook.consolidate_if_due(self.client)
            if cost:
                self.journal.log_cost("playbook", cost)
        return done

    def lessons(self) -> list[dict]:
        """Existing playbook lessons for context (filled in by the playbook, part 2)."""
        try:
            rows = self.journal.query("SELECT tag, statement FROM lessons WHERE status != 'retired' "
                                      "ORDER BY support - contradict DESC LIMIT 40")
        except Exception:  # noqa: BLE001 (table not created yet)
            return []
        return [dict(r) for r in rows]

    def _review_trade(self, row) -> int:
        d = dossier.trade_dossier(self.journal, row["id"])
        if d is None:
            return 0
        res = postmortem.review(self.client, self.cfg, d, self.lessons())
        self.record("postmortem", res, row["token"])
        return self._store(res, "trade", row["id"], row["decision_id"], row["token"], row["symbol"],
                           d["result"]["return_pct"])

    def _review_missed(self, row) -> int:
        d = dossier.missed_dossier(self.journal, row["decision_id"])
        if d is None:
            return 0
        res = postmortem.review(self.client, self.cfg, d, self.lessons())
        self.record("postmortem", res, row["token"])
        return self._store(res, "missed", None, row["decision_id"], row["token"], row["symbol"],
                           d["outcome_after_pass"]["best_pct"])

    def _store(self, res: LLMResult, kind: str, position_id, decision_id, token, symbol, ret) -> int:
        if res.outcome != OK:
            if res.outcome != UNAVAILABLE:
                log.warning("post-mortem for %s failed: %s", symbol, res.error)
            return 0
        a = res.data
        self.journal.execute(
            """INSERT OR IGNORE INTO postmortems(ts, kind, position_id, decision_id, token, symbol, return_pct,
               category, analysis, cost_usd) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), kind, position_id, decision_id, token, symbol, ret, a["category"], json.dumps(a),
             res.cost_usd))
        lesson = a["lesson"]["statement"] or "no lesson"
        log.info("post-mortem %s %s (%s): %s -> %s", kind, symbol, _fmt(ret), a["category"], lesson)
        self.after_postmortem(a)
        return 1

    def after_postmortem(self, analysis: dict) -> None:
        self.playbook.record(analysis)


def _fmt(ret) -> str:
    return "n/a" if ret is None else f"{ret:+.1f}%"
