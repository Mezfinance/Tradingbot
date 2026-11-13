"""The real-money launch canary (canary_rw50_sz): status, calibration against its simulated twin, smoke test.

The canary buys launchrw50_sz's thin-pool picks (`mb.CANARY_TWIN`) for real and sells them at pool minute 6. Each of its
trades is paired with the twin's simulated trade on the same token, so the gap between the two is the simulator's error:
entry price, exit price, fees, and how many picks the real book missed or couldn't sell.
"""

from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone

from tradingbot import model_book as mb
from tradingbot.data.jupiter import USDC_MINT

SIM_TWIN = mb.CANARY_TWIN
SMOKE_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"  # BONK: deep pools, for a tiny buy/sell test

# --- two-stage entry: confirmation-window volatility gate (11-13) ---------------------------------
# At a probe SURVIVOR, read the pool's reserves across the confirmation window via LIVE Helius (the paid key) and
# require the peak-to-trough price range to clear a floor. A dead-flat pool is a dud (it won't run); a pool that
# dipped and recovered is fine (a down-move before entry is to our benefit) -- so this is a RANGE, not a net up-move.
CONF_AFTER_S = 60.0                        # confirmation ≈ probe age + the ~48s watch + margin (probe_exits.CONF_AFTER_S)
CONF_OFFSETS = (-58, -44, -30, -16, 0)     # relative to conf_ts (mirrors probe_exits.CONF_FETCH_OFFSETS)


def confirmation_range(helius, pool: str, token_mint: str | None, conf_ts: float,
                       offsets=CONF_OFFSETS) -> tuple[float | None, int]:
    """Peak-to-trough price range over the confirmation window via live Helius (one `last_before` per offset, ~10
    credits each). Returns (conf_range, readable_samples); conf_range is None when fewer than 2 samples read."""
    from tradingbot import launch_live as ll
    from tradingbot.data import pumpswap as ps
    prices = []
    for off in offsets:
        tx = helius.last_before(pool, conf_ts + off)
        if not isinstance(tx, dict):
            continue
        sol = ll._raw_vault(tx, pool, ps.WSOL)
        tok = ll._raw_vault(tx, pool, token_mint) if token_mint else None
        if sol and tok:
            prices.append(sol / tok)
    if len(prices) < 2:
        return None, len(prices)
    lo, hi = min(prices), max(prices)
    return (((hi - lo) / lo) if lo else None), len(prices)


def volatility_ok(helius, pool: str, token_mint: str | None, conf_ts: float, min_pct: float,
                  offsets=CONF_OFFSETS) -> tuple[bool, float | None, int]:
    """True if the confirmation-window range >= `min_pct`. Fails CLOSED (False) when it can't be measured (no
    helius / fewer than 2 readable samples), so a pool we can't confirm is skipped rather than entered blind."""
    rng, n = confirmation_range(helius, pool, token_mint, conf_ts, offsets) if helius is not None else (None, 0)
    if rng is None:
        return False, rng, n
    return rng >= (min_pct / 100.0), rng, n


def _trades(journal, book: str, since: float) -> dict[str, dict]:
    """Closed or open positions of `book` since `since`, keyed by token (first position per token)."""
    out: dict[str, dict] = {}
    for p in journal.query("SELECT * FROM positions WHERE strategy=? AND opened_at >= ? ORDER BY opened_at",
                           (book, since)):
        fills = journal.query("SELECT side, price, usd, fee_usd, ts FROM fills WHERE position_id=? ORDER BY ts",
                              (p["id"],))
        buys = [f for f in fills if f["side"] == "buy"]
        sells = [f for f in fills if f["side"] == "sell"]
        sold_usd = sum(f["usd"] for f in sells)
        fees = sum(f["fee_usd"] or 0 for f in fills)
        row = {"token": p["token"], "symbol": p["symbol"], "opened": p["opened_at"], "closed": p["closed_at"],
               "cost": p["cost_usd"], "ret": (p["realized_usd"] / p["cost_usd"] - 1) * 100
               if p["closed_at"] and p["cost_usd"] else None,
               "buy_px": buys[0]["price"] if buys else None,
               "sell_px": (sum(f["price"] * f["usd"] for f in sells) / sold_usd) if sells and sold_usd > 0 else None,
               "fees": fees, "fee_pct": fees / p["cost_usd"] * 100 if p["cost_usd"] else None,
               "exit": p["exit_reason"]}
        # swap only: what the price moves gave, before network fees and rent
        if row["buy_px"] and row["sell_px"]:
            row["swap_ret"] = (row["sell_px"] / row["buy_px"] - 1) * 100
        out.setdefault(p["token"], row)
    return out


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 2) if xs else None


def calibration(journal, hours: float | None = None) -> dict:
    since = time.time() - hours * 3600 if hours else 0.0
    real = _trades(journal, mb.CANARY4, since)
    if real:
        since = max(since, min(r["opened"] for r in real.values()) - 60)
    sim = _trades(journal, SIM_TWIN, since)
    pairs = []
    for tok, r in real.items():
        s = sim.get(tok)
        if s is None or r["ret"] is None or s["ret"] is None:
            continue
        pairs.append({
            "symbol": r["symbol"], "token": tok, "real_ret": round(r["ret"], 2), "sim_ret": round(s["ret"], 2),
            "gap": round(r["ret"] - s["ret"], 2),
            "swap_gap": round(r["swap_ret"] - s["swap_ret"], 2) if "swap_ret" in r and "swap_ret" in s else None,
            "entry_vs_sim_pct": round((r["buy_px"] / s["buy_px"] - 1) * 100, 2) if r["buy_px"] and s["buy_px"] else None,
            "exit_vs_sim_pct": round((r["sell_px"] / s["sell_px"] - 1) * 100, 2) if r["sell_px"] and s["sell_px"] else None,
            "real_fee_pct": round(r["fee_pct"], 2) if r["fee_pct"] is not None else None,
            "sim_fee_pct": round(s["fee_pct"], 2) if s["fee_pct"] is not None else None,
            "fill_delay_s": round(r["opened"] - s["opened"], 1), "real_exit": r["exit"],
        })
    sim_after = {t: s for t, s in sim.items() if not real or s["opened"] >= min(r["opened"] for r in real.values())}
    missed = [s["symbol"] for t, s in sim_after.items() if t not in real]
    return {
        "real_trades": len(real), "closed": sum(1 for r in real.values() if r["ret"] is not None),
        "open": sum(1 for r in real.values() if r["closed"] is None), "paired": len(pairs),
        "real_mean_ret": _mean([p["real_ret"] for p in pairs]), "sim_mean_ret": _mean([p["sim_ret"] for p in pairs]),
        "mean_gap": _mean([p["gap"] for p in pairs]), "mean_swap_gap": _mean([p["swap_gap"] for p in pairs]),
        "mean_entry_vs_sim_pct": _mean([p["entry_vs_sim_pct"] for p in pairs]),
        "mean_exit_vs_sim_pct": _mean([p["exit_vs_sim_pct"] for p in pairs]),
        "real_fee_pct": _mean([p["real_fee_pct"] for p in pairs]), "sim_fee_pct": _mean([p["sim_fee_pct"] for p in pairs]),
        "median_fill_delay_s": round(statistics.median([p["fill_delay_s"] for p in pairs]), 1) if pairs else None,
        "sim_picks_not_bought": len(missed), "pairs": pairs,
    }


def status(journal, cfg) -> dict:
    from tradingbot.engine import CANARY_HALT
    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    skips = journal.query("SELECT detail, COUNT(*) n FROM events WHERE strategy=? AND kind='entry_skipped' AND ts>=? "
                          "GROUP BY detail ORDER BY n DESC LIMIT 10", (mb.CANARY4, day_start))
    failed = journal.query("SELECT COUNT(*) n, COALESCE(SUM(usd),0) usd FROM events WHERE strategy=? "
                           "AND kind='tx_failed' AND ts>=?", (mb.CANARY4, day_start))[0]
    return {
        "enabled": bool(cfg.launch_canary), "halted": bool(journal.get_state(CANARY_HALT)),
        "cash": round(journal.get_cash(cfg.launch_canary_budget_usd, mb.CANARY4), 2),
        "open_positions": len(journal.open_positions(mb.CANARY4)),
        "realized_today": round(journal.realized_since(day_start, mb.CANARY4), 2),
        "limits": {"usd": cfg.launch_canary_usd, "budget": cfg.launch_canary_budget_usd,
                   "daily_loss": cfg.launch_canary_daily_loss_usd, "max_open": cfg.launch_canary_max_open,
                   "every_n": cfg.launch_canary_every_n, "max_liq_usd": cfg.launch_canary_max_liq_usd,
                   "twin": SIM_TWIN},
        "failed_tx_today": {"n": failed["n"], "usd": round(failed["usd"], 4)},
        "skips_today": {r["detail"]: r["n"] for r in skips},
    }


def smoke(executor, mint: str = SMOKE_MINT, usd: float = 2.0) -> dict:
    """Buy `usd` of a deep token for real, then sell it all back. Proves the key, RPC, Jupiter build, fill
    parsing and account closing work before the canary trades launches."""
    out: dict = {"wallet": executor.owner, "usdc_before": executor.token_balance(USDC_MINT) / 1e6, "sol_before": executor.sol_balance()}
    t0 = time.time()
    buy = executor.buy(mint, usd, 0.0)
    out["buy"] = None if buy is None else {**buy.__dict__, "seconds": round(time.time() - t0, 1)}
    if buy is None:
        out["buy_note"] = executor.last_note
        return out
    if not buy.ok:
        return out
    t1 = time.time()
    sell = executor.sell(mint, buy.qty, 0.0)
    out["sell"] = None if sell is None else {**sell.__dict__, "seconds": round(time.time() - t1, 1)}
    if sell is not None and sell.ok:
        out["round_trip_usd"] = round(sell.usd - buy.usd, 4)
        out["round_trip_pct"] = round((sell.usd / buy.usd - 1) * 100, 2)
    out["usdc_after"] = executor.token_balance(USDC_MINT) / 1e6
    out["sol_after"] = executor.sol_balance()
    return out


def _sol_balance(executor):
    """SOL of the direct executor, or (under rotation) of the first wallet; best-effort for the smoke summary."""
    if hasattr(executor, "sol_balance"):
        return executor.sol_balance()
    subs = getattr(executor, "subs", None)
    return subs[0].sol_balance() if subs and hasattr(subs[0], "sol_balance") else None


def feint_smoke(executor, mint: str, usd: float, pool: str | None = None) -> dict:
    """Run one real feint (atomic $usd buy + sell back, re-read the pool) and report the verdict. On a deep, healthy
    pool this should return 'alive' with sol_after ~ sol_before -- it proves the atomic buy+sell tx lands on-chain,
    the pool re-read works and the verdict logic fires, before a real-money feint measurement run is armed. commit is
    False (no follow-on buy), so under rotation it also frees the wallet."""
    if pool and hasattr(executor, "remember_pool"):
        executor.remember_pool(mint, pool)
    out: dict = {"mint": mint, "usd": usd, "sol_before": _sol_balance(executor)}
    t0 = time.time()
    out["verdict"] = executor.feint(mint, usd, commit=False)
    out["seconds"] = round(time.time() - t0, 1)
    out["last_feint"] = getattr(executor, "last_feint", {})
    out["note"] = getattr(executor, "last_note", "")
    out["sol_after"] = _sol_balance(executor)
    return out


def probe_smoke(executor, mint: str, usd: float, pool: str | None = None) -> dict:
    """Run one real 1-slot probe (buy + next-slot sell) at a low `usd` on a deep, healthy pool and report the result.
    On a healthy pool it should return verdict 'ok' with recovered ~ spent and a slot_gap of 1-3 -- it proves the two
    transactions land, the sell fires in the next slot(s) and the stake comes back, before a real-money probe run is
    armed. Opens no position; under rotation it frees the wallet (a clean probe holds nothing)."""
    if pool and hasattr(executor, "remember_pool"):
        executor.remember_pool(mint, pool)
    out: dict = {"mint": mint, "usd": usd, "sol_before": _sol_balance(executor)}
    t0 = time.time()
    out["result"] = executor.probe_1slot(mint, usd)
    out["seconds"] = round(time.time() - t0, 1)
    out["note"] = getattr(executor, "last_note", "")
    out["sol_after"] = _sol_balance(executor)
    return out
