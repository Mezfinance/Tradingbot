"""hold-study: returns by hold time per liquidity band, paired against the minute-6 exit."""
from tradingbot import hold_study as hs
from tradingbot import launch_live as ll
from tradingbot.watcher import WatchStore
from tests.test_launch_live import _PathHelius

T0 = 1_800_000_000.0


def _store(n=20):
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    states = {}
    for i in range(n):
        for kind, liq in (("S", 30_000.0), ("B", 100_000.0)):
            pool = f"{kind}{i}"
            st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, liq_usd, fill_px, fill_sol, "
                       "peak_sol, ret, rw_pct, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (pool, "M" + pool, T0, 1, liq, 100.0 / 1_000_000, 100.0, 160.0, 0.54, 30.0 if i % 2 else 70.0,
                        "done"))
            if kind == "S":   # small pools climb 10% a minute, then are emptied at minute 8
                states[pool] = lambda ts: ((100.0 * (1 + 0.1 * (ts - T0) / 60)) if ts - T0 < 480 else 0.5, 1_000_000)
            else:             # big pools keep climbing
                states[pool] = lambda ts: (100.0 * (1 + 0.1 * (ts - T0) / 60), 1_000_000)
    return st, _PathHelius(states)


def test_holds_split_by_band_and_compare_with_minute_six():
    st, h = _store()
    out = hs.study(st, h, now=T0 + 7200, max_credits=10**6)
    small, big = out["bands"]["$25-40k"]["every pool"], out["bands"]["$80-150k"]["every pool"]
    assert small["n"] == 20 and big["n"] == 20
    assert small["holds"]["1 min"]["mean_pct"] == 4.0                     # +10% - 6% cost
    assert small["holds"]["10 min"]["rugged_pct"] == 100 and small["holds"]["10 min"]["vs_minute6_pts"] < -100
    assert big["holds"]["15 min"]["mean_pct"] > big["holds"]["6 min (books)"]["mean_pct"]
    assert big["holds"]["15 min"]["vs_minute6_ci90"][0] > 0
    assert out["bands"]["$80-150k"]["rug-vs-win kept (rw_pct < 50)"]["n"] == 10
    calls = h.calls
    hs.study(st, h, now=T0 + 7200, max_credits=10**6)
    assert h.calls == calls                                                  # cached states are never fetched twice
    assert "HOLD TIMES BY POOL SIZE" in hs.render(out)


def test_credit_cap_fills_bands_evenly_and_young_pools_wait():
    st, h = _store()
    out = hs.study(st, h, now=T0 + 7200, max_credits=400)                   # 40 calls
    assert "stopped at the credit cap" in " ".join(out["fetch"])
    got = {b: g["every pool"]["holds"]["1 min"]["n"] for b, g in out["bands"].items()}
    assert abs(got["$25-40k"] - got["$80-150k"]) <= 1
    assert hs.study(st, h, now=T0 + 600)["pools"] == 0                       # nothing an hour old yet


def test_small_pools_are_left_out_unless_asked():
    st, h = _store(4)
    st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, liq_usd, fill_px, fill_sol, peak_sol, "
               "ret, status) VALUES ('T','MT',?,1,8000,0.0001,100,100,0.1,'done')", (T0,))
    h.states["T"] = lambda ts: (100.0, 1_000_000)
    assert "$5-15k" not in hs.study(st, h, now=T0 + 7200, max_credits=10**6)["bands"]
    assert "$5-15k" in hs.study(st, h, now=T0 + 7200, max_credits=10**6, min_liq=0)["bands"]
