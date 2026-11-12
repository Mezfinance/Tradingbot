import base64
import time

from tradingbot import pool_tracker as pt


def _acct(amount: int) -> dict:
    raw = bytearray(165)
    raw[64:72] = int(amount).to_bytes(8, "little")
    return {"data": [base64.b64encode(bytes(raw)).decode(), "base64"]}


def test_feint_pools_keeps_the_earliest_probe_per_pool():
    import json
    rows = [
        {"ts": 100.0, "token": "MintA", "kind": "canary_feint", "detail": json.dumps({"pool": "P1", "verdict": "alive", "sol_before": 600})},
        {"ts": 150.0, "token": "MintA", "kind": "canary_feint", "detail": json.dumps({"pool": "P1", "verdict": "pulled", "sol_before": 10})},
        {"ts": 120.0, "token": "MintB", "kind": "canary_feint", "detail": json.dumps({"pool": "P2", "verdict": "inconclusive", "note": "x"})},
        {"ts": 130.0, "token": "MintC", "kind": "canary_feint", "detail": "{}"},  # no pool: dropped
    ]
    q = lambda sql, p: [] if "canary_probe_verdict" in sql else [r for r in rows if r["ts"] >= p[0]]
    out = pt.feint_pools(q, 0.0)
    assert set(out) == {"P1", "P2"}
    assert out["P1"]["verdict"] == "alive" and out["P1"]["feint_ts"] == 100.0  # earliest kept
    assert out["P1"]["kind"] == "feint"


def test_feint_pools_includes_1slot_probes_and_the_survival_verdict():
    import json
    rows = [
        {"ts": 100.0, "token": "MintA", "kind": "canary_probe",
         "detail": json.dumps({"pool": "PA", "verdict": "ok", "sol_before": 10, "slot_gap": 1})},
        {"ts": 110.0, "token": "MintB", "kind": "canary_feint", "detail": json.dumps({"pool": "PB", "verdict": "alive"})},
    ]
    verds = [{"detail": json.dumps({"pool": "PA", "verdict": "survivor", "sol_after": 11})}]
    q = lambda sql, p: (verds if "canary_probe_verdict" in sql else [r for r in rows if r["ts"] >= p[0]])
    out = pt.feint_pools(q, 0.0)
    assert out["PA"]["kind"] == "probe" and out["PA"]["verdict"] == "ok"  # the probe's own result
    assert out["PA"]["watch_verdict"] == "survivor" and out["PA"]["sol_after"] == 11  # the ~48s watch read, overlaid
    assert out["PB"]["kind"] == "feint" and out["PB"]["watch_verdict"] is None


def test_read_vaults_parses_token_amounts_from_base64():
    calls = {}

    def rpc(method, params):
        calls["m"] = method
        chunk = params[0]
        amt = {"sv": 50_000_000_000, "tv": 1_000_000_000_000}  # 50 SOL, 1e12 raw tokens
        return {"value": [_acct(amt[a]) for a in chunk]}

    v = pt.read_vaults(rpc, [{"pool": "P1", "sol_vault": "sv", "token_vault": "tv"}])
    assert calls["m"] == "getMultipleAccounts"
    sol, tok = v["P1"]
    assert round(sol, 3) == 50.0 and tok == 1e12


def test_update_path_tracks_peak_detects_pull_and_finds_the_best_entry():
    now = 1000.0
    created, feint = now - 300, now - 294
    path = None
    # price climbs with liquidity, then the LP is pulled to near zero
    for i, (sol, tok) in enumerate([(50, 1e12), (90, 1e12), (120, 1e12), (2, 1e12)]):
        path = pt.update_path(path, created + 6 + i * 60, created, feint, sol, tok, 100.0, None)
    assert round(path["peak_liq"]) == 24000          # 2 * 120 * 100
    assert path["pulled_ts"] is not None             # last read is < 10% of peak
    assert path["opt_age"] == 6                       # cheapest price was the first sample (pool age 6s)
    assert path["live_liq"] == 2 * 2 * 100


def test_assemble_joins_feint_scorer_and_path():
    now = 1000.0
    feints = {"P1": {"pool": "P1", "mint": "MintA", "verdict": "alive", "note": "", "feint_ts": now - 300,
                     "sol_before": 70, "sol_after": 72}}
    scorer = {"P1": {"pool": "P1", "token_mint": "MintA", "create_ts": now - 306, "decided_ts": now - 302,
                     "liq_usd": 14000, "sol_at_decide": 70, "peak_sol": 140, "sol_vault": "sv", "token_vault": "tv",
                     "fill_px": 1e-7, "exit_px": 1.3e-7, "exit_sol": 95, "pulled_ts": None, "ret": 0.22,
                     "status": "done"}}
    rows = pt.assemble(feints, scorer, 100.0, {}, now)
    r = rows[0]
    assert round(r["probe_pool_age_s"]) == 6
    assert r["liq_probe"] == 14000
    assert round(r["peak_liq"]) == 28000            # liq_usd * peak_sol/sol_at_decide = 14000 * 2
    assert round(r["pnl25"], 1) == 5.5              # $25 * 0.22
    assert r["pulled"] is False and r["status"] == "done"


def test_assemble_falls_back_to_the_feint_reserve_when_there_is_no_scorer_row():
    now = 1000.0
    feints = {"P9": {"pool": "P9", "mint": "M", "verdict": "pulled", "note": "", "feint_ts": now - 10,
                     "sol_before": 80, "sol_after": 1}}
    rows = pt.assemble(feints, {}, 100.0, {}, now)
    assert rows[0]["liq_probe"] == 2 * 80 * 100     # no scorer row: 2 * sol_before * sol_usd
    assert rows[0]["status"] == "pending"


def test_summary_counts_verdicts_and_pnl():
    rows = [{"verdict": "alive", "pulled": False, "ret_m6": 0.2, "pnl25": 5.0},
            {"verdict": "pulled", "pulled": True, "ret_m6": -1.0, "pnl25": -25.0},
            {"verdict": "inconclusive", "pulled": False, "ret_m6": None, "pnl25": None}]
    s = pt.summary(rows)
    assert s["pools"] == 3 and s["alive"] == 1 and s["pulled_at_probe"] == 1 and s["inconclusive"] == 1
    assert s["pulled_since"] == 1 and s["with_outcome"] == 2 and s["would_pnl25"] == -20.0
