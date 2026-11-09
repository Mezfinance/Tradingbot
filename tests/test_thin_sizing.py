"""Thin-band books and the thin-band sizing study (11-09)."""
import random

from tradingbot import launch_live as ll
from tradingbot import model_book as mb
from tradingbot import thin_sizing as ts
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

T0 = 1_780_000_000.0


def test_thin_flags_and_books():
    assert ll.thin_flags(10_000, 30.0, 50.0) == {"thin": True, "thin_rw50": True, "thin_band": True}
    assert ll.thin_flags(10_000, 70.0, 90.0) == {"thin": True, "thin_rw50": False, "thin_band": False}
    assert not any(ll.thin_flags(20_000, 10.0, 10.0).values())          # not thin
    assert not any(ll.thin_flags(10_000, 10.0, 10.0, liquid=False).values())
    assert len(mb.LAUNCH_THIN_CUT) == 9 and mb.LAUNCH_THIN_CUT["thin_band_250"] == "thin_band"
    assert mb.band_usd("thin_all_100", 9_000) == 100.0 and mb.band_usd("thin_rw_250", 9_000) == 250.0
    assert mb.band_usd("thin_band_2p", 9_000) == float("inf")           # the engine's 2%-of-pool cap sizes it
    sig = {"thin": True, "thin_rw50": False, "thin_band": True, "created_ts": 100.0, "mint": "M"}
    assert mb.wants("thin_all_2p", sig) and mb.wants("thin_band_100", sig) and not mb.wants("thin_rw_100", sig)
    from tradingbot.config import Config
    from tradingbot import dashboard
    cfg = Config(launch_live=True, launch_thin_book=True)
    assert set(mb.LAUNCH_THIN_CUT) <= set(mb.active_books(cfg))
    assert mb.exit_deadline(cfg, "thin_all_100", sig) == 100.0 + 360
    assert dashboard.family_of("thin_rw_2p") == "Thin band"


def _rows(st, n, rng):
    ll.ensure_schema(st)
    for i in range(n):
        rug = rng.random() < 0.25
        liq = rng.uniform(5_000, 14_900)
        sol_at = 50.0
        fill_sol = liq / (2 * 100.0)          # SOL price $100
        move = 0.0 if rug else rng.choice([0.1, 0.2, 0.6])
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, decided_ts, liquid, status, ret, liq_usd, "
                   "sol_at_decide, fill_sol, exit_sol, fill_px, exit_px, rw_pct, rwb_pct) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (f"P{i}", f"M{i}", T0 + i * 600, T0 + i * 600 + 4, 1, "done",
                    -1.0 if rug else move - ts.COST, liq, liq / 200.0, fill_sol,
                    0.0 if rug else fill_sol * (1 + move) ** 0.5, 1e-6, 1e-6 * (1 + move), rng.uniform(0, 100),
                    rng.uniform(0, 100)))


def test_bigger_trades_lose_more_to_their_own_impact_and_rugs_cost_the_whole_stake():
    r = {"ret": 0.14, "gross": 0.20, "r_in": 5_000.0, "r_out": 5_500.0, "liq_usd": 10_000.0}
    small, big = ts.sized_ret(r, 50, 0.0), ts.sized_ret(r, 1_000, 0.0)
    assert abs(small - 0.14) < 0.03 and big < small - 0.2
    assert ts.sized_ret(dict(r, ret=-1.0), 100, 0.0) == -1.0
    assert ts.size_of(r, 500, 0.02) == 200.0 and ts.size_of(r, ts.INF, 0.03) == 300.0


def test_study_scores_every_sizing_with_a_walk_forward_and_a_bootstrap():
    st = WatchStore(":memory:")
    _rows(st, 400, random.Random(4))
    out = ts.study(st, Journal(":memory:"), now=T0 + 400 * 600)
    assert out["pools"] == 400 and out["gap"]["gap"] == ts.DEFAULT_GAP
    f = out["filters"]["every thin pool"]
    assert f["pools"] == 400 and set(f["sizings"]) == {s[0] for s in ts.SIZINGS}
    a, b = f["sizings"]["$50"], f["sizings"]["5% of pool"]
    assert b["mean_size_usd"] > a["mean_size_usd"] and b["sized_pct"] < a["sized_pct"]   # impact grows with size
    assert a["bootstrap"]["median_year_usd"] is not None and f["walk_forward"]["blocks"] > 1
    assert 0 < out["filters"]["sz picks (rw50)"]["pools"] < 400
    assert "THIN-BAND SIZING" in ts.render(out) and "walk-forward" in ts.render(out)


def test_thin_sizing_runs_through_the_cli(cfg, tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    cli.main(["thin-sizing"])
    assert "no finished thin pools" in capsys.readouterr().out


def test_bootstrap_drawdown_runs_along_each_path_not_across_paths():
    import numpy as np
    # a steady winner with small down days never draws down 30% of $10k in a year
    daily = np.array([1000.0, 900.0, -200.0, 1100.0, 800.0])
    b = ts._bootstrap(daily, 10_000, np.random.default_rng(1))
    assert b["p_dd_over_30pct"] == 0
    assert b["dd_p90_usd"] >= -2_000
    # a big losing day shows up as a drawdown
    b = ts._bootstrap(np.array([500.0, -4_000.0]), 10_000, np.random.default_rng(1))
    assert b["p_dd_over_30pct"] == 100
