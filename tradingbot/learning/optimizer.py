"""Settings optimiser: Claude proposes, recorded data decides.

Every `optimizer_every_trades` closed Claude trades (or weekly with a few new ones),
Claude reviews the evidence (stats, post-mortem categories, parameter hints,
lessons) and may call `test_settings` as often as it likes to try ideas against
recorded candidate price paths. Its final proposal is then validated by code, not
by Claude:

  * every change within the hard bounds in params.py (sizing is not replay-testable),
  * enough history: `optimizer_min_trades` candidates, `optimizer_min_holdout` in the holdout,
  * on the most recent `optimizer_holdout_frac` of candidates (never used to justify the
    idea in the train summary), P&L improves by at least `optimizer_min_gain_pct`% of
    trade size per candidate,
  * the earlier train period is not made worse,
  * at least half as many holdout trades as now (no "improving" by not trading).

Passing changes go live at once as a new settings version. After
`rollback_min_trades` further trades, the version is re-checked against its parent
on the trades taken since; if the parent would have done better, it rolls back.
Code-level ideas become proposals for the engineer (part 4).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import Counter

from tradingbot import replay
from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.learning import params, versions
from tradingbot.learning.mission import MISSION
from tradingbot.llm import OK, LLMResult, tool_loop

log = logging.getLogger(__name__)

REPLAY_SIZE_USD = 50.0


# --- evaluation on recorded data ------------------------------------------------

def _snapshot(row) -> dict:
    try:
        return json.loads(row["snapshot"]) if row["snapshot"] else {}
    except (TypeError, ValueError):
        return {}


def passes(row, cfg: Config) -> bool:
    """Would this past candidate still be bought under `cfg`? Unknown fields count as passing."""
    if row["action"] != "buy" or (row["conviction"] or 0) < cfg.min_conviction:
        return False
    if (row["factor_score"] or 0) < cfg.min_factor_score:
        return False
    snap = _snapshot(row)
    liq = snap.get("liquidity_usd", row["liquidity"])
    fdv = snap.get("fdv_usd", row["fdv"])
    if liq is not None and liq < cfg.min_liquidity_usd:
        return False
    if fdv is not None and fdv > cfg.max_fdv_usd:
        return False
    vol = (snap.get("volume") or {}).get("h1")
    if vol is not None and vol < cfg.min_volume_h1_usd:
        return False
    created, fetched = snap.get("created_at_ms"), snap.get("fetched_at")
    if created and fetched:
        age_min = (fetched * 1000 - created) / 60_000
        if age_min < cfg.min_age_minutes or age_min > cfg.max_age_hours * 60:
            return False
    return True


def history(j: Journal) -> list:
    """Every analysed token Claude wanted to buy, with its recorded price path start, oldest first."""
    return j.query(
        """SELECT d.id AS decision_id, d.action, d.conviction, d.factor_score, d.liquidity, d.fdv, d.snapshot,
                  t.token, t.start_ts, t.start_price
           FROM decisions d JOIN tracking t ON t.decision_id = d.id
           WHERE d.stage='analysed' AND d.action='buy' AND t.start_price > 0
           ORDER BY t.start_ts""")


def _path(j: Journal, token: str, start: float) -> list[tuple[float, float]]:
    return [(r["ts"], r["price"]) for r in j.query(
        "SELECT ts, price FROM prices WHERE token=? AND ts>? AND ts<=? ORDER BY ts",
        (token, start, start + 24 * 3600 + 120))]


def simulate_set(j: Journal, rows: list, cfg: Config, costs: replay.CostModel) -> dict:
    trades = []
    for r in rows:
        if not passes(r, cfg):
            continue
        path = _path(j, r["token"], r["start_ts"])
        if path:
            trades.append(replay.simulate(r, path, cfg, costs, REPLAY_SIZE_USD))
    s = replay.summarise(trades)
    s["candidates"] = len(rows)
    s.setdefault("total_usd", 0.0)
    return s


def with_changes(cfg: Config, changes: dict) -> Config:
    tuned = copy.deepcopy(cfg)
    versions.apply_to(tuned, changes)
    return tuned


def evaluate(j: Journal, cfg: Config, changes: dict, holdout_frac: float) -> dict:
    rows = history(j)
    cut = int(len(rows) * (1 - holdout_frac))
    train, hold = rows[:cut], rows[cut:]
    costs = replay.cost_model(j)
    tuned = with_changes(cfg, changes)
    return {
        "train": {"current": simulate_set(j, train, cfg, costs), "proposed": simulate_set(j, train, tuned, costs)},
        "holdout": {"current": simulate_set(j, hold, cfg, costs), "proposed": simulate_set(j, hold, tuned, costs)},
        "cost_model": costs.source,
    }


def acceptance(ev: dict, cfg: Config) -> tuple[bool, str]:
    tr, ho = ev["train"], ev["holdout"]
    total = tr["current"]["candidates"] + ho["current"]["candidates"]
    if total < cfg.optimizer_min_trades:
        return False, f"only {total} candidates with price paths (need {cfg.optimizer_min_trades})"
    if ho["current"]["candidates"] < cfg.optimizer_min_holdout:
        return False, f"holdout too small ({ho['current']['candidates']})"
    need = cfg.optimizer_min_gain_pct / 100 * REPLAY_SIZE_USD * ho["current"]["candidates"]
    gain = ho["proposed"]["total_usd"] - ho["current"]["total_usd"]
    if gain < need:
        return False, f"holdout gain ${gain:+.2f} < required ${need:.2f}"
    if tr["proposed"]["total_usd"] < tr["current"]["total_usd"]:
        return False, "worse on the earlier (train) period"
    if ho["proposed"].get("n", 0) < 0.5 * ho["current"].get("n", 0):
        return False, "achieves the gain mainly by trading far less"
    return True, f"holdout gain ${gain:+.2f} (required ${need:.2f}), train not worse"


# --- the Claude-driven search ------------------------------------------------------

SYSTEM = """You tune the settings of a memecoin trading bot using its own recorded history. You \
get statistics, the categories and parameter hints from per-trade post-mortems, and the current \
playbook. Use the test_settings tool to try ideas: it replays every past candidate the analyst \
wanted to buy under the settings you give, on recorded price paths, and returns results for an \
earlier "train" period and a later "holdout" period.

Be a disciplined quant:
- Form hypotheses from the post-mortem evidence first, then test them. Do not grid-search blindly.
- Prefer few, well-supported changes. Small samples make big moves look good by chance.
- The holdout decides. A change that only helps on train is overfitted.
- Only settings in the allowed list can change, within their bounds. Filter thresholds can only \
get stricter than the defaults. Sizing settings are not replay-testable and cannot be changed here.
- It is fine to propose no change.

Also list code-level improvements that settings cannot achieve (missing data, a flawed rule, a \
signal the post-mortems keep asking for), with the evidence. These go to an automated engineer, \
so make each one specific, self-contained and testable."""
SYSTEM = f"{MISSION}\n\n{SYSTEM}"

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {"type": "array", "items": {
            "type": "object",
            "properties": {"parameter": {"type": "string", "enum": sorted(params.TUNABLES)},
                           "value": {"type": "number"}},
            "required": ["parameter", "value"], "additionalProperties": False}},
        "rationale": {"type": "string"},
        "code_proposals": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "problem": {"type": "string"},
                           "evidence": {"type": "string"}, "idea": {"type": "string"},
                           "priority": {"type": "integer"}},
            "required": ["title", "problem", "evidence", "idea", "priority"], "additionalProperties": False}},
    },
    "required": ["changes", "rationale", "code_proposals"],
    "additionalProperties": False,
}

TEST_TOOL = {
    "name": "test_settings",
    "description": ("Replay all past buy candidates under the given setting changes (others stay as now). "
                    "Returns train and holdout results for current vs proposed settings."),
    "input_schema": {
        "type": "object",
        "properties": {"changes": {"type": "object", "additionalProperties": {"type": "number"},
                                   "description": "parameter -> new value"}},
        "required": ["changes"],
    },
}


def closed_trades(j: Journal) -> int:
    return j.query("SELECT COUNT(*) AS n FROM positions WHERE strategy='claude' AND closed_at IS NOT NULL")[0]["n"]


class Optimizer:
    def __init__(self, journal: Journal, cfg: Config):
        self.journal = journal
        self.cfg = cfg
        versions.ensure(journal)

    # --- scheduling ---------------------------------------------------------------
    def due(self) -> bool:
        last_n = int(self.journal.get_state("optimizer_trades_at_last_run") or 0)
        last_ts = float(self.journal.get_state("optimizer_last_run") or 0)
        new = closed_trades(self.journal) - last_n
        if new >= self.cfg.optimizer_every_trades:
            return True
        return new >= 5 and time.time() - last_ts >= 7 * 86400

    def evidence(self) -> str:
        from tradingbot import report
        from tradingbot.learning import playbook

        rep = report.build(self.journal, self.cfg)
        hints: Counter = Counter()
        cats: Counter = Counter()
        for r in self.journal.query("SELECT category, analysis FROM postmortems"):
            cats[r["category"]] += 1
            for h in json.loads(r["analysis"]).get("parameter_hints", []):
                hints[f"{h['parameter']} {h['direction']}"] += 1
        bounds = {n: {"min": t.lo, "max": t.hi, "kind": t.kind, "meaning": t.why}
                  for n, t in params.TUNABLES.items() if t.kind != "sizing"}
        return json.dumps({
            "current_settings": params.current_values(self.cfg),
            "allowed_changes": bounds,
            "strategies": [{k: s[k] for k in ("strategy", "return_pct", "closed_trades", "win_rate",
                                              "median_trade_return_pct", "profit_factor", "exit_reasons")}
                           for s in rep["strategies"]],
            "candidate_outcomes": rep["claude_selection"],
            "postmortem_categories": dict(cats),
            "parameter_hints": dict(hints.most_common(20)),
            "active_lessons": [l["statement"] for l in playbook.dump(self.journal) if l["status"] == "active"],
            "previous_versions": [{k: v[k] for k in ("id", "changes", "reason", "status", "verdict")}
                                  for v in versions.history(self.journal)[:5]],
        }, default=str)

    # --- main entry points -------------------------------------------------------
    def run(self, client) -> LLMResult:
        """One optimisation round. Returns the Claude result (cost is on it)."""
        self.journal.set_state("optimizer_last_run", repr(time.time()))
        self.journal.set_state("optimizer_trades_at_last_run", str(closed_trades(self.journal)))
        versions.ensure_baseline(self.journal, self.cfg)

        def test_settings(inp: dict) -> str:
            ok, problems = params.validate(inp.get("changes", {}), self.cfg)
            ok = {k: v for k, v in ok.items() if params.TUNABLES[k].kind != "sizing"}
            ev = evaluate(self.journal, self.cfg, ok, self.cfg.optimizer_holdout_frac)
            accepted, why = acceptance(ev, self.cfg)
            return json.dumps({"tested_changes": ok, "rejected_inputs": problems, "results": ev,
                               "would_be_accepted": accepted, "acceptance_note": why}, default=str)

        res = tool_loop(client, self.cfg, system=SYSTEM, user=self.evidence(), tools=[TEST_TOOL],
                        handlers={"test_settings": test_settings}, schema=FINAL_SCHEMA, effort="high",
                        max_turns=25, budget_usd=self.cfg.optimizer_budget_usd)
        if res.outcome != OK:
            log.warning("optimiser round ended without a proposal: %s", res.error)
            return res
        self._store_proposals(res.data["code_proposals"])
        self._decide({c["parameter"]: c["value"] for c in res.data["changes"]}, res.data["rationale"])
        return res

    def _decide(self, changes: dict, rationale: str) -> int | None:
        ok, problems = params.validate(changes, self.cfg)
        ok = {k: v for k, v in ok.items() if params.TUNABLES[k].kind != "sizing"}
        current = params.current_values(self.cfg)
        ok = {k: v for k, v in ok.items() if v != current.get(k)}
        if not ok:
            log.info("optimiser: no settings change (%s)", "; ".join(problems) or rationale[:200])
            return None
        ev = evaluate(self.journal, self.cfg, ok, self.cfg.optimizer_holdout_frac)
        accepted, why = acceptance(ev, self.cfg)
        if not accepted:
            log.info("optimiser: proposal %s rejected by validation: %s", ok, why)
            self.journal.log_event("optimizer_rejected", None, f"{ok}: {why}")
            return None
        return versions.activate(self.journal, self.cfg, ok, f"{why}. {rationale}"[:2000], ev)

    def review_active(self) -> str | None:
        """Roll back the active version if its parent would have done better on trades since."""
        v = versions.active(self.journal)
        if v is None or v["parent_id"] is None or v["verdict"]:
            return None
        since = [r for r in history(self.journal) if r["start_ts"] >= v["ts"]]
        traded = self.journal.query(
            "SELECT COUNT(*) AS n FROM positions WHERE strategy='claude' AND closed_at IS NOT NULL AND opened_at >= ?",
            (v["ts"],))[0]["n"]
        if traded < self.cfg.rollback_min_trades:
            return None
        parent_settings = json.loads(self.journal.query(
            "SELECT settings FROM config_versions WHERE id=?", (v["parent_id"],))[0]["settings"])
        costs = replay.cost_model(self.journal)
        now = simulate_set(self.journal, since, self.cfg, costs)
        before = simulate_set(self.journal, since, with_changes(self.cfg, parent_settings), costs)
        margin = self.cfg.optimizer_min_gain_pct / 100 * REPLAY_SIZE_USD * max(1, len(since))
        if before["total_usd"] - now["total_usd"] >= margin:
            why = (f"since v{v['id']} went live, v{v['parent_id']} would have made "
                   f"${before['total_usd']:+.2f} vs ${now['total_usd']:+.2f}")
            versions.rollback(self.journal, self.cfg, why)
            return why
        verdict = f"kept after {traded} trades: ${now['total_usd']:+.2f} vs parent ${before['total_usd']:+.2f}"
        self.journal.execute("UPDATE config_versions SET verdict=? WHERE id=?", (verdict, v["id"]))
        return verdict

    def _store_proposals(self, proposals: list[dict]) -> None:
        open_titles = {r["title"].lower() for r in self.journal.query(
            "SELECT title FROM proposals WHERE status IN ('open','building')")}
        for p in proposals:
            if p["title"].lower() in open_titles:
                continue
            self.journal.execute(
                "INSERT INTO proposals(ts, source, title, problem, evidence, idea, priority) VALUES(?,?,?,?,?,?,?)",
                (time.time(), "optimizer", p["title"], p["problem"], p["evidence"], p["idea"],
                 max(1, min(5, p["priority"]))))
