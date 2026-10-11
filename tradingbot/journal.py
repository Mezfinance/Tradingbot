"""SQLite journal: every screening decision, every Claude verdict, every fill.

The point is measurement. `outcome_price` is back-filled later for *all* analysed
tokens (bought or passed), so we can check whether Claude's picks actually beat
the ones it rejected before risking real money.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict

from tradingbot.models import FactorResult, PairSnapshot, Position, Verdict

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    token TEXT NOT NULL,
    symbol TEXT,
    stage TEXT NOT NULL,              -- screened_out | analysed
    price REAL,
    liquidity REAL,
    fdv REAL,
    factor_score REAL,
    components TEXT,
    rejections TEXT,
    verdict TEXT,
    action TEXT,                      -- buy | pass | NULL
    conviction REAL,
    outcome_price REAL,
    outcome_ts REAL
);
CREATE INDEX IF NOT EXISTS decisions_token ON decisions(token, ts);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL,
    symbol TEXT,
    entry_price REAL, qty REAL, initial_qty REAL, cost_usd REAL,
    opened_at REAL, stop_price REAL, peak_price REAL,
    tp_hits INTEGER DEFAULT 0, realized_usd REAL DEFAULT 0,
    closed_at REAL, exit_reason TEXT, decision_id INTEGER
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY,
    ts REAL, position_id INTEGER, side TEXT, qty REAL, price REAL,
    usd REAL, fee_usd REAL, reason TEXT, tx TEXT
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


class Journal:
    def __init__(self, path: str = "tradingbot.db"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # --- state ------------------------------------------------------------
    def get_cash(self, default: float) -> float:
        row = self.db.execute("SELECT value FROM state WHERE key='cash'").fetchone()
        if row is None:
            self.set_cash(default)
            return default
        return float(row["value"])

    def set_cash(self, cash: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO state(key, value) VALUES('cash', ?)", (str(cash),))
        self.db.commit()

    # --- decisions --------------------------------------------------------
    def recently_seen(self, token: str, within_s: float) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM decisions WHERE token=? AND ts>? AND stage='analysed' LIMIT 1",
            (token, time.time() - within_s),
        ).fetchone()
        return row is not None

    def log_decision(
        self,
        p: PairSnapshot,
        factors: FactorResult,
        verdict: Verdict | None = None,
        analysed: bool = False,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO decisions(ts, token, symbol, stage, price, liquidity, fdv, factor_score,
               components, rejections, verdict, action, conviction)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), p.token_address, p.symbol,
                "analysed" if analysed else "screened_out",
                p.price_usd, p.liquidity_usd, p.fdv_usd, factors.score,
                json.dumps(factors.components), json.dumps(factors.rejections),
                verdict.model_dump_json() if verdict else None,
                verdict.action if verdict else None,
                verdict.conviction if verdict else None,
            ),
        )
        self.db.commit()
        return cur.lastrowid

    def pending_outcomes(self, older_than_s: float) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT id, token, price FROM decisions WHERE stage='analysed' AND outcome_price IS NULL AND ts<?",
            (time.time() - older_than_s,),
        ).fetchall()

    def set_outcome(self, decision_id: int, price: float | None) -> None:
        self.db.execute(
            "UPDATE decisions SET outcome_price=?, outcome_ts=? WHERE id=?",
            (price if price is not None else 0.0, time.time(), decision_id),
        )
        self.db.commit()

    # --- positions --------------------------------------------------------
    def open_positions(self) -> list[Position]:
        rows = self.db.execute("SELECT * FROM positions WHERE closed_at IS NULL").fetchall()
        return [_row_to_position(r) for r in rows]

    def save_position(self, pos: Position, decision_id: int | None = None) -> Position:
        d = asdict(pos)
        if pos.id is None:
            d.pop("id")
            d["decision_id"] = decision_id
            d["token"] = d.pop("token_address")
            cols = ",".join(d)
            cur = self.db.execute(
                f"INSERT INTO positions({cols}) VALUES({','.join('?' * len(d))})", tuple(d.values())
            )
            pos.id = cur.lastrowid
        else:
            self.db.execute(
                """UPDATE positions SET qty=?, stop_price=?, peak_price=?, tp_hits=?, realized_usd=?,
                   closed_at=?, exit_reason=? WHERE id=?""",
                (pos.qty, pos.stop_price, pos.peak_price, pos.tp_hits, pos.realized_usd,
                 pos.closed_at, pos.exit_reason, pos.id),
            )
        self.db.commit()
        return pos

    def log_fill(self, position_id: int, side: str, qty: float, price: float, usd: float,
                 fee_usd: float, reason: str, tx: str | None) -> None:
        self.db.execute(
            "INSERT INTO fills(ts, position_id, side, qty, price, usd, fee_usd, reason, tx) VALUES(?,?,?,?,?,?,?,?,?)",
            (time.time(), position_id, side, qty, price, usd, fee_usd, reason, tx),
        )
        self.db.commit()

    def realized_since(self, since_ts: float) -> float:
        """Net realised P&L from sells since a timestamp (sell proceeds minus their cost basis)."""
        row = self.db.execute(
            """SELECT COALESCE(SUM(f.usd - f.qty * p.cost_usd / p.initial_qty), 0) AS pnl
               FROM fills f JOIN positions p ON p.id = f.position_id
               WHERE f.side='sell' AND f.ts>=?""",
            (since_ts,),
        ).fetchone()
        return float(row["pnl"])

    # --- reporting --------------------------------------------------------
    def stats(self) -> dict:
        q = self.db.execute
        closed = q("SELECT realized_usd - cost_usd AS pnl FROM positions WHERE closed_at IS NOT NULL").fetchall()
        wins = [r["pnl"] for r in closed if r["pnl"] > 0]
        losses = [r["pnl"] for r in closed if r["pnl"] <= 0]

        def fwd(action: str) -> tuple[int, float | None]:
            rows = q(
                """SELECT price, outcome_price FROM decisions
                   WHERE stage='analysed' AND action=? AND outcome_price IS NOT NULL AND price>0""",
                (action,),
            ).fetchall()
            if not rows:
                return 0, None
            rets = [(r["outcome_price"] / r["price"] - 1) * 100 for r in rows]
            return len(rets), sum(rets) / len(rets)

        n_buy, ret_buy = fwd("buy")
        n_pass, ret_pass = fwd("pass")
        return {
            "screened_out": q("SELECT COUNT(*) FROM decisions WHERE stage='screened_out'").fetchone()[0],
            "analysed": q("SELECT COUNT(*) FROM decisions WHERE stage='analysed'").fetchone()[0],
            "closed_trades": len(closed),
            "open_trades": q("SELECT COUNT(*) FROM positions WHERE closed_at IS NULL").fetchone()[0],
            "win_rate": len(wins) / len(closed) if closed else None,
            "net_realised_usd": sum(r["pnl"] for r in closed),
            "avg_win_usd": sum(wins) / len(wins) if wins else None,
            "avg_loss_usd": sum(losses) / len(losses) if losses else None,
            "claude_buy_fwd_return_pct": ret_buy, "claude_buy_samples": n_buy,
            "claude_pass_fwd_return_pct": ret_pass, "claude_pass_samples": n_pass,
        }


def _row_to_position(r: sqlite3.Row) -> Position:
    return Position(
        id=r["id"], token_address=r["token"], symbol=r["symbol"], entry_price=r["entry_price"],
        qty=r["qty"], initial_qty=r["initial_qty"], cost_usd=r["cost_usd"], opened_at=r["opened_at"],
        stop_price=r["stop_price"], peak_price=r["peak_price"], tp_hits=r["tp_hits"],
        realized_usd=r["realized_usd"], closed_at=r["closed_at"], exit_reason=r["exit_reason"],
    )
