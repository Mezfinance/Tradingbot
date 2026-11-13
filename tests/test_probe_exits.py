import json

from tradingbot import probe_exits as pe


def _e(series, pull=None):
    peak = max(m for _, m in series)
    return {"series": series, "pull_after_conf": pull, "peak_mult": peak, "m6_ret": None}


def test_take_profit_counts_climbers_and_probes_per_climber():
    climber = _e([(0, 1.0), (60, 1.2), (120, 1.3), (300, 1.25)])          # peak +30%
    modest = _e([(0, 1.0), (60, 1.1), (120, 1.22), (360, 1.0)])           # peak +22%
    flat_rug = _e([(0, 1.0), (60, 1.0)], pull=120)                        # flat then pulled
    survs = [climber, modest, flat_rug]
    out = {t["tp_pct"]: t for t in pe.take_profit(survs, committed=30)}
    # +20% TP: climber and modest both reach it, flat_rug does not
    assert out[0.20]["climbers"] == 2
    assert out[0.20]["probes_per_climber"] == 30 / 2
    # +25% TP: only the +30% climber reaches it
    assert out[0.25]["climbers"] == 1
    assert out[0.50]["climbers"] == 0 and out[0.50]["probes_per_climber"] is None


def test_tp_ret_banks_the_target_and_zeroes_a_pull_before_it():
    climber = _e([(0, 1.0), (60, 1.3)])
    assert abs(pe._tp_ret(climber, 0.25) - (0.25 - pe.COST)) < 1e-9       # hit TP -> +25% minus cost
    flat_rug = _e([(0, 1.0), (60, 1.0)], pull=90)
    assert pe._tp_ret(flat_rug, 0.25) == -1.0                            # never hit TP, then pulled


def test_trailing_fires_on_a_fader_but_not_against_a_one_slot_rug():
    fader = _e([(0, 1.0), (60, 1.5), (120, 1.1)])                        # -27% off the 1.5 peak
    r = pe._trail_ret(fader, 0.20)
    assert abs(r - (1.1 - 1.0 - pe.COST)) < 1e-9                         # exits at 1.1
    rug = _e([(0, 1.0), (60, 1.05)], pull=90)                            # pulled before any trail fires
    assert pe._trail_ret(rug, 0.20) == -1.0


def test_fixed_hold_zeroes_a_pull_within_the_window():
    e = _e([(0, 1.0), (60, 1.2), (180, 1.4)], pull=120)
    assert pe._hold_ret(e, 60) == pe._clamp(1.2 - 1.0 - pe.COST)         # out at 60s, before the pull
    assert pe._hold_ret(e, 180) == -1.0                                  # held past the 120s pull


def test_climb_gate_bails_a_flat_pool_before_its_later_rug():
    # v8AykTHP: sat flat ~2 min then pulled at minute 3 (pull_after_conf ~120s).
    flat_rug = _e([(0, 1.0), (60, 1.0)], pull=120)
    r = pe._climb_ret(flat_rug, check=60, gain=0.05)                     # not up 5% at 60s -> bail at the check
    assert abs(r - (1.0 - 1.0 - pe.COST)) < 1e-9                         # ~ -cost, NOT -100%
    climber = _e([(0, 1.0), (60, 1.2), (360, 1.5)])
    assert pe._climb_ret(climber, check=60, gain=0.05) == pe._hold_ret(climber, pe.CLIMB_HOLD_S)  # rides it
    pulled_early = _e([(0, 1.0)], pull=30)
    assert pe._climb_ret(pulled_early, check=60, gain=0.05) == -1.0      # pulled before the check


def test_pull_distribution_finds_the_safe_window():
    survs = [{"pull_after_conf": None}, {"pull_after_conf": 200.0}, {"pull_after_conf": 400.0}]
    d = pe.pull_distribution(survs)
    assert d["n_survivors"] == 3 and d["n_pulled"] == 2
    assert d["earliest_pull_s"] == 200.0
    assert d["safe_window_s"] == 180                                     # no pull within 180s; first pull at 200
    by = {b["t_s"]: b for b in d["buckets"]}
    assert by[180]["pulled_by"] == 0 and by[240]["pulled_by"] == 1


def test_study_end_to_end_without_helius():
    conf = pe.CONF_AFTER_S
    events = [{"ts": 1000.0, "token": "MintS", "kind": "canary_probe",
               "detail": json.dumps({"pool": "PS", "verdict": "ok", "slot_gap": 1})},
              {"ts": 1000.0, "token": "MintR", "kind": "canary_probe",
               "detail": json.dumps({"pool": "PR", "verdict": "ok", "slot_gap": 1})}]
    verds = [{"detail": json.dumps({"pool": "PS", "verdict": "survivor"})},
             {"detail": json.dumps({"pool": "PR", "verdict": "survivor"})}]

    class J:
        def query(self, sql, params=()):
            if "canary_probe_verdict" in sql:
                return verds
            return [r for r in events if r["ts"] >= params[0]]

    from tradingbot import pool_tracker as pt
    created = 1000.0 - 7  # probe at pool age 7s
    base = {c: None for c in pt.SCORER_COLS}
    scorer = {
        "PS": {**base, "pool": "PS", "token_mint": "MintS", "create_ts": created, "liq_usd": 15000,
               "pulled_ts": None, "ret": 0.25},
        "PR": {**base, "pool": "PR", "token_mint": "MintR", "create_ts": created, "liq_usd": 14000,
               "pulled_ts": 1000.0 + conf + 120, "ret": -1.0},
    }

    class S:
        def query(self, sql, params=()):
            return [r for r in scorer.values() if r["create_ts"] >= params[0]]

    # confirmation age = (1000 + conf) - created
    conf_age = (1000.0 + conf) - created
    paths = {
        "PS": {"pts": [(conf_age, 15000, 1.0), (conf_age + 60, 16000, 1.2), (conf_age + 300, 15000, 1.3)],
               "peak_liq": 16000, "pulled_ts": None},
        "PR": {"pts": [(conf_age, 14000, 1.0), (conf_age + 60, 14000, 1.0)],
               "peak_liq": 14000, "pulled_ts": 1000.0 + conf + 120},
    }
    out = pe.study(J(), S(), cfg=None, hours=24.0, helius=None, paths=paths, now=1000.0 + conf + 400)
    assert out["survivors"] == 2 and out["survivors_with_path"] == 2
    assert out["probes_committed"] == 2
    # PR pulled 120s after confirmation -> it is in the pull distribution
    assert out["pull_distribution"]["n_pulled"] == 1
    # the +30% survivor is a climber for a +25% TP; the flat-rug is not
    tp25 = next(t for t in out["take_profit"] if t["tp_pct"] == 0.25)
    assert tp25["climbers"] == 1
    # the paths here have <2 samples inside the confirmation window, so no conf_range
    assert "volatility" in out and out["volatility"]["with_range"] == 0
    assert "kept_climb" in out and out["kept_vol_pct"] == pe.DEFAULT_VOL
    assert pe.render(out).startswith("PROBE EXITS")


def test_conf_range_measures_the_window_and_the_filter_splits_movers_from_the_flat():
    created = 1000.0
    feint_ts = 1007.0                                   # probe at pool age 7
    conf_age = feint_ts + pe.CONF_AFTER_S - created     # 67
    f = {"pool": "P", "feint_ts": feint_ts, "kind": "probe", "verdict": "ok", "watch_verdict": "survivor"}
    s = {"create_ts": created, "pulled_ts": None, "ret": 0.1}
    # mover: dips to 0.95 and recovers inside the conf window, then runs to 1.2 after
    mover = pe._entry_series(f, s, {"peak_liq": 20000, "pts": [
        (12, 20000, 1.0), (30, 19000, 0.95), (50, 20000, 1.0),
        (conf_age, 20000, 1.0), (conf_age + 60, 20000, 1.2)]}, None)
    assert mover is not None
    assert abs(mover["conf_range"] - (1.0 - 0.95) / 0.95) < 1e-9   # peak-to-trough; dip-and-recover counts
    # dead-flat: no movement in the window
    flat = pe._entry_series(f, s, {"peak_liq": 20000, "pts": [
        (12, 20000, 1.0), (30, 20000, 1.0), (conf_age, 20000, 1.0), (conf_age + 60, 20000, 1.0)]}, None)
    assert flat["conf_range"] < 0.01
    vf = pe.volatility_filter([mover, flat], committed=10)
    assert vf["with_range"] == 2 and vf["no_range"] == 0
    b2 = next(b for b in vf["by_threshold"] if b["vol_pct"] == 0.02)
    assert b2["kept"]["n"] == 1 and b2["skipped"]["n"] == 1        # mover kept at 2%, flat skipped
    assert b2["kept"]["win"] == 1.0 and b2["skipped"]["win"] == 0.0  # mover profits, flat bleeds -cost
    b5 = next(b for b in vf["by_threshold"] if b["vol_pct"] == 0.05)
    assert b5["kept"]["n"] == 1                                    # the ~5.3% range still clears 5%


def test_conf_range_from_a_fetched_series_splits_the_window_at_sec_zero():
    created = 1000.0
    feint_ts = 1007.0
    f = {"pool": "P", "feint_ts": feint_ts, "kind": "probe", "verdict": "ok", "watch_verdict": "survivor"}
    s = {"create_ts": created, "pulled_ts": None, "ret": 0.5}
    # (sec, sol, tok): sec <= 0 = conf window, sec > 0 = after conf; price = sol / tok
    fetched = [(-58, 1.0, 1.0), (-30, 1.1, 1.0), (0, 1.0, 1.0), (60, 1.5, 1.0)]
    e = pe._entry_series(f, s, None, fetched)
    assert abs(e["conf_range"] - 0.10) < 1e-9     # 1.0 -> 1.1 across the conf-window samples
    assert e["series"][0] == (60.0, 1.5)          # entry = price at sec 0 (1.0); post sample 60 -> 1.5x
