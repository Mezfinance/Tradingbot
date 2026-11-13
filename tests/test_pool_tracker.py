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


def test_honeypot_probe_never_shows_as_a_survivor_and_flags_no_money_sent():
    # A honeypot was caught by the free pre-buy sim: nothing broadcast, so a "survivor" survival read on it is
    # meaningless and dangerous (survivor = a pool to enter for real). It must never carry a survivor/rugger verdict,
    # and the row must say money was not committed. Covers historical rows where the watch wrongly ran.
    import json
    rows = [{"ts": 100.0, "token": "MintH", "kind": "canary_probe",
             "detail": json.dumps({"pool": "PH", "verdict": "honeypot", "note": "probe: honeypot (sim: ...)"})}]
    verds = [{"detail": json.dumps({"pool": "PH", "verdict": "survivor", "sol_after": 7})}]  # a stale/bogus read
    q = lambda sql, p: (verds if "canary_probe_verdict" in sql else [r for r in rows if r["ts"] >= p[0]])
    out = pt.feint_pools(q, 0.0)
    assert out["PH"]["watch_verdict"] is None  # the survivor read is ignored for a non-committed probe
    row = pt.assemble(out, {}, 100.0, {}, 200.0)[0]
    assert row["verdict"] == "honeypot" and row["watch_verdict"] is None  # shown as honeypot, not survivor
    assert row["sent"] is False and row["committed"] is False  # $0 ever left the wallet


def test_rows_carry_the_executing_wallet_from_probe_and_feint_events():
    # The dashboard shows which rotation wallet ran each probe; probes store `wallet_addr`, feints `wallet`.
    import json
    rows = [
        {"ts": 100.0, "token": "MintP", "kind": "canary_probe",
         "detail": json.dumps({"pool": "PP", "verdict": "ok", "slot_gap": 1, "wallet_addr": "Wa11etPubkeyAAAA"})},
        {"ts": 110.0, "token": "MintF", "kind": "canary_feint",
         "detail": json.dumps({"pool": "PF", "verdict": "alive", "wallet": "Wa11etPubkeyBBBB"})},
    ]
    q = lambda sql, p: [] if "canary_probe_verdict" in sql else [r for r in rows if r["ts"] >= p[0]]
    out = pt.feint_pools(q, 0.0)
    assert out["PP"]["wallet"] == "Wa11etPubkeyAAAA" and out["PF"]["wallet"] == "Wa11etPubkeyBBBB"
    by_pool = {r["pool"]: r for r in pt.assemble(out, {}, 100.0, {}, 200.0)}
    assert by_pool["PP"]["wallet"] == "Wa11etPubkeyAAAA" and by_pool["PF"]["wallet"] == "Wa11etPubkeyBBBB"


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


def test_update_path_measures_the_confirmation_entry_vs_the_best_entry():
    created = 1000.0
    feint = created + 6  # probe at pool age 6s; survival confirmation ~ 6 + 60 = 66s
    path = None
    # a survivor that climbs and is never pulled: samples at pool age 6 / 70 / 360 (minute 6 = exit)
    for age, sol in [(6, 50.0), (70, 100.0), (360, 120.0)]:
        path = pt.update_path(path, created + age, created, feint, sol, 1e12, 100.0, None)
    assert path["pulled_ts"] is None
    assert path["opt_age"] == 6 and round(path["opt_ret"], 2) == 1.34   # hindsight best = the 6s low
    assert path["conf_age"] == 70                                       # first price at/after confirmation (66s)
    assert round(path["conf_ret"], 2) == 0.14                           # entering at ~60s makes far less than best


def test_update_path_builds_the_hold_curve_to_catch_ripsnorters():
    created = 1000.0
    feint = created + 6  # confirmation ~66s
    path = None
    # a survivor that is flat at minute 6 but rips to 3x by minute 15, never pulled
    for age, sol in [(6, 50.0), (70, 60.0), (900, 180.0)]:  # 900s = minute 15
        path = pt.update_path(path, created + age, created, feint, sol, 1e12, 100.0, None)
    assert path["pulled_ts"] is None
    cb = path["conf_by"]
    assert round(cb[6], 2) == -0.06                 # flat at minute 6 (just the 6% cost)
    assert round(cb[15], 2) == 1.94                 # +194% held to minute 15 -- a ripsnorter the minute-6 exit misses
    assert cb[15] > cb[6]


def test_hold_curve_aggregates_survivor_returns_by_horizon():
    surv = [
        {"conf_by": {6: 0.1, 10: 0.3, 15: 1.5, 30: -1.0}},  # rips to +150% at 15, then rugs by 30
        {"conf_by": {6: 0.0, 10: 0.1, 15: 0.2}},            # no 30-min sample yet
    ]
    c = pt.hold_curve(surv)
    assert c["15"]["n"] == 2 and c["15"]["big"] == 0.5      # one of two hit +100% at 15m
    assert c["30"]["n"] == 1 and c["30"]["mean"] == -1.0    # the only 30m value rugged
    assert "60" not in c                                     # no survivor has a 60m value


def test_summary_reports_survivor_entry_vs_best():
    rows = [
        {"watch_verdict": "survivor", "conf_ret": 0.14, "opt_ret": 1.34, "ret_m6": None, "pnl25": None, "verdict": "survivor", "pulled": False},
        {"watch_verdict": "survivor", "conf_ret": -1.0, "opt_ret": 0.5, "ret_m6": None, "pnl25": None, "verdict": "survivor", "pulled": True},
        {"watch_verdict": "rugger", "conf_ret": None, "opt_ret": None, "ret_m6": None, "pnl25": None, "verdict": "rugger", "pulled": True},
    ]
    s = pt.summary(rows)
    assert s["survivor"] == 2 and s["rugger"] == 1 and s["survivor_entry_n"] == 2
    assert round(s["survivor_entry_mean"], 3) == round((0.14 - 1.0) / 2, 3)  # a later-rugged survivor counts as -100%
    assert round(s["survivor_best_mean"], 3) == 0.92


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
    assert r["ret_m6"] == 0.22                       # the scorer's second-6 ideal, kept for context
    assert r["pnl25"] is None                        # $25 @ m6 is the confirmation entry: no reserve path -> no number
    assert r["pulled"] is False and r["status"] == "done"


def test_assemble_scores_a_scorer_pull_as_minus_100_even_when_the_path_froze_pre_rug():
    # 9KaQvM4Z: a survivor whose dashboard path froze at a pre-rug sample (so conf_ret looked like a small -8% loss),
    # but the scorer flagged a full LP pull at minute 2. The hold-to-m6 entry must score -100%, not the stale loss --
    # otherwise a full rug inflates survivor_entry_mean (the go/no-go number for committing real money).
    now = 1000.0
    created = now - 300
    feints = {"P": {"pool": "P", "mint": "M", "kind": "probe", "verdict": "ok", "watch_verdict": "survivor",
                    "note": "", "feint_ts": created + 6, "sol_before": 65}}
    scorer = {"P": {"pool": "P", "token_mint": "M", "create_ts": created, "liq_usd": 32600, "sol_at_decide": 100,
                    "peak_sol": 108, "pulled_ts": created + 120, "ret": None, "status": "done"}}  # pull at minute 2
    # the frozen path: the dashboard never saw the drain (path pulled_ts None), so conf_ret/opt_ret look like small moves
    paths = {"P": {"pts": [], "peak_liq": 35100, "live_liq": 34800, "pulled_ts": None,
                   "conf_age": 60.0, "conf_ret": -0.08, "opt_age": 41.0, "opt_ret": 0.07,
                   "conf_by": {6: -0.08, 10: -0.08, 15: -0.08}}}
    row = pt.assemble(feints, scorer, 100.0, paths, now)[0]
    assert row["pulled"] is True and round(row["pull_pool_age_s"]) == 120
    assert row["conf_ret"] == -1.0 and row["opt_ret"] == -1.0           # the full rug, not the stale -8%
    assert row["pnl25"] == -25.0                                        # $25 @ m6 = the confirmation entry = -$25
    assert row["conf_by"][6] == -1.0 and row["conf_by"][15] == -1.0     # every horizon past the 2m pull is -100%
    s = pt.summary([row])
    assert s["survivor_entry_n"] == 1 and s["survivor_entry_mean"] == -1.0  # the go/no-go number reflects the rug


def test_assemble_keeps_a_post_m6_pull_from_zeroing_the_hold_to_m6_entry():
    # a pool pulled AFTER minute 6: a hold-to-m6 entry exits before the pull, so conf_ret/opt_ret stay as measured.
    now = 1000.0
    created = now - 600
    feints = {"P": {"pool": "P", "mint": "M", "kind": "probe", "verdict": "ok", "watch_verdict": "survivor",
                    "note": "", "feint_ts": created + 6, "sol_before": 65}}
    scorer = {"P": {"pool": "P", "token_mint": "M", "create_ts": created, "liq_usd": 20000, "sol_at_decide": 100,
                    "peak_sol": 120, "pulled_ts": created + 480, "ret": 0.3, "status": "done"}}  # pull at minute 8
    paths = {"P": {"pts": [], "peak_liq": 24000, "live_liq": 23000, "pulled_ts": None,
                   "conf_age": 60.0, "conf_ret": 0.25, "opt_age": 40.0, "opt_ret": 0.6,
                   "conf_by": {6: 0.25, 10: -1.0, 15: -1.0}}}
    row = pt.assemble(feints, scorer, 100.0, paths, now)[0]
    assert row["conf_ret"] == 0.25 and row["opt_ret"] == 0.6           # the m6 exit beat the minute-8 pull
    assert row["pnl25"] == 25.0 * 0.25                                 # $25 @ m6 = the confirmation entry (+$6.25)
    assert row["conf_by"][6] == 0.25 and row["conf_by"][10] == -1.0    # holding to 10m past the 8m pull is -100%


def test_assemble_falls_back_to_the_feint_reserve_when_there_is_no_scorer_row():
    now = 1000.0
    feints = {"P9": {"pool": "P9", "mint": "M", "verdict": "pulled", "note": "", "feint_ts": now - 10,
                     "sol_before": 80, "sol_after": 1}}
    rows = pt.assemble(feints, {}, 100.0, {}, now)
    assert rows[0]["liq_probe"] == 2 * 80 * 100     # no scorer row: 2 * sol_before * sol_usd
    assert rows[0]["status"] == "pending"


def test_scorer_rows_by_pools_fetches_only_the_given_pools():
    from tradingbot.watcher import WatchStore
    from tradingbot import launch_live
    store = WatchStore(":memory:")
    launch_live.ensure_schema(store)
    for pool, mint, ts in [("PoolA", "MintA", 100.0), ("PoolB", "MintB", 200.0), ("PoolC", "MintC", 300.0)]:
        store.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, decided_ts, liquid, status) "
                      "VALUES(?,?,?,?,1,'scored')", (pool, mint, ts, ts + 4))
    # by pools: only the asked-for pools, regardless of create_ts (a probed pool from long ago still resolves)
    got = pt.scorer_rows(store.query, 0.0, pools=["PoolA", "PoolC"])
    assert set(got) == {"PoolA", "PoolC"}
    assert pt.scorer_rows(store.query, 0.0, pools=[]) == {}
    # the time-window form still works for the other callers
    assert set(pt.scorer_rows(store.query, 250.0)) == {"PoolC"}


def test_load_paths_reads_the_persisted_file_and_int_keys_conf_by(tmp_path):
    import json
    db = str(tmp_path / "bot.db")
    (tmp_path / "pool_paths.json").write_text(json.dumps(
        {"P1": {"pts": [], "peak_liq": 34800.0, "conf_ret": 0.12, "conf_by": {"6": 0.12, "15": 0.3}},
         "bad": "not a dict"}))
    got = pt.load_paths(db)
    assert got["P1"]["peak_liq"] == 34800.0 and got["P1"]["conf_by"][15] == 0.3  # int key, not "15"
    assert "bad" not in got
    assert pt.load_paths(":memory:") == {} and pt.load_paths(str(tmp_path / "empty" / "bot.db")) == {}  # no file


def test_summary_counts_verdicts_and_pnl():
    rows = [{"verdict": "alive", "pulled": False, "ret_m6": 0.2, "pnl25": 5.0},
            {"verdict": "pulled", "pulled": True, "ret_m6": -1.0, "pnl25": -25.0},
            {"verdict": "inconclusive", "pulled": False, "ret_m6": None, "pnl25": None}]
    s = pt.summary(rows)
    assert s["pools"] == 3 and s["alive"] == 1 and s["pulled_at_probe"] == 1 and s["inconclusive"] == 1
    assert s["pulled_since"] == 1 and s["with_outcome"] == 2 and s["would_pnl25"] == -20.0
