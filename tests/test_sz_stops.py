"""sz-stops: stop losses, trailing stops and climb checks replayed on a book's real trades (11-09)."""
import json

from tradingbot import launch_live as ll
from tradingbot import sz_stops as ss
from tradingbot.data import pumpswap as ps
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

D = 86400.0
T0 = 1_780_000_000.0 // D * D


def _tx(t, sol, tok, pool, mint):
    bal = lambda s, k: [{"owner": pool, "mint": ps.WSOL, "uiTokenAmount": {"uiAmount": s}},  # noqa: E731
                        {"owner": pool, "mint": mint, "uiTokenAmount": {"amount": str(k)}}]
    return {"blockTime": t, "meta": {"err": None, "preTokenBalances": bal(sol, tok), "postTokenBalances": bal(sol, tok)}}


def _setup(pools, complete=True, liq=None):
    """pools: name -> (price function of seconds after creation, real return, pulled second or None)."""
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    st.execute(ss.SCHEMA)
    for i, (name, (fn, ret, pulled)) in enumerate(pools.items()):
        c = T0 + i * 1000
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, status, pulled_ts, liq_usd, "
                   "sol_at_decide) VALUES(?,?,?,?,?,?,?)",
                   (f"P{name}", f"M{name}", c, "done", c + pulled if pulled else None, liq, 50.0 if liq else None))
        j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) "
                  "VALUES(?,?,?,?,?,?)", (f"M{name}", 100.0, c + 10, c + 360, 100.0 * (1 + ret), ss.BOOK))
        # pool tokens fixed at 1e6, so price = SOL / 1e6; SOL = 100 x price multiple
        states = [(fn(s) if isinstance(fn(s), tuple) else (100 * fn(s), 1_000_000)) for s in range(5, 371)]
        rows = ss.path_rows([_tx(c + s, sol, tok, f"P{name}", f"M{name}") for s, (sol, tok) in zip(range(5, 371), states)],
                            f"P{name}", f"M{name}")
        st.execute("INSERT INTO watch_trade_path(pool, data, ts) VALUES(?,?,?)",
                   (f"P{name}", json.dumps({"rows": rows, "complete": complete}), T0))
    return st, j


def _rules(out):
    return {r["rule"]: r for r in out["rules"]}


def test_a_stop_sells_a_slow_fall_but_not_a_one_transaction_pull():
    slow = lambda s: 1 - 0.002 * max(0, s - 10)          # noqa: E731  -1% every 5 s from our buy: -70% by the sale
    pull = lambda s: 1.0 if s < 200 else 0.0              # noqa: E731
    st, j = _setup({"slow": (slow, -0.76, None), "pull": (pull, -1.0, 200)})
    out = ss.study(st, j, now=T0 + 3 * D)
    assert out["coverage"]["usable"] == 2
    r = _rules(out)
    assert r["no rule"]["usd"] == -176
    s20 = r["stop -20%"]
    assert s20["fired"] == 2 and s20["fired_on_rugs"] == 1        # it fires on the pull, but sells at nothing
    assert 45 < s20["delta_usd"] < 52                              # slow fall sold near -21% instead of -70%
    assert s20["saved_usd"] == s20["delta_usd"] and s20["cut_winners_usd"] == 0
    assert "STOPS AND CLIMB CHECKS" in ss.render(out)


def test_climb_sells_a_pool_that_never_rose_and_keeps_one_that_did():
    flat_then_up = lambda s: 1.0 if s < 200 else 1.6       # noqa: E731  flat at minute 3, +60% later
    early_rise = lambda s: 1.0 + (0.08 if 60 < s < 120 else 0.0) - (0.3 if s > 300 else 0.0)  # noqa: E731
    st, j = _setup({"late": (flat_then_up, 0.50, None), "early": (early_rise, -0.36, None)})
    out = ss.study(st, j, now=T0 + 3 * D)
    r = _rules(out)
    c = r["climb 3m +5%"]
    assert c["fired"] == 1                                        # 'late' sold at minute 3; 'early' passed at +8%
    assert -61 < c["delta_usd"] < -59                             # gave up its +60% move
    assert c["cut_winners_usd"] == c["delta_usd"]
    assert r["climb 3m +10%"]["fired"] == 2                       # +8% isn't enough for a +10% bar
    assert r["climb 1m +0%"]["fired"] == 1 and r["climb 2m +0%"]["fired"] == 1   # 'late' never rose above the buy


def test_drawdown_limits_combine_with_exit_rules_and_early_sales_count_early():
    slow = lambda s: 1 - 0.002 * max(0, s - 10)           # noqa: E731  -76% each at minute 6, -21% if stopped
    up = lambda s: 1.0 + 0.001 * max(0, s - 10)           # noqa: E731
    pools = {f"l{i}": (slow, -0.76, None) for i in range(3)}
    pools.update({f"w{i}": (up, 0.30, None) for i in range(3)})
    st, j = _setup(pools)                                  # one buy every 1,000 s, all in one UTC day
    out = ss.study(st, j, now=T0 + 3 * D)
    g = {(c["rule"], c["limit"]): c for c in out["combos"]}
    # $100 limit: the first two losers are sold (-$152) before the third buy, so the rest of the day is skipped
    assert g[("no rule", "day loss 1%")]["trades"] == 2
    assert g[("no rule", "day loss 10%")]["trades"] == 6               # $1,000 never reached
    # with a -20% stop each loser costs ~$21 and is sold ~100 s after its buy: -$63 never reaches $100
    assert g[("stop -20%", "day loss 1%")]["trades"] == 6
    assert len(out["limits"]) == len(ss.limits()) - 1 and all(c["delta_ci"] is not None for c in out["limits"])
    text = ss.render(out)
    assert "DRAWDOWN LIMITS ALONE" in text and "EXIT RULE x DRAWDOWN LIMIT" in text and "BEST 10 COMBINATIONS" in text


def test_paths_cut_by_the_page_cap_are_not_judged():
    st, j = _setup({"a": (lambda s: 0.5, -0.56, None)}, complete=False)
    out = ss.study(st, j, now=T0 + 3 * D)
    assert out["coverage"]["usable"] == 0 and out["coverage"]["cut_by_page_cap"] == 1
    assert all(r["delta_usd"] == 0 for r in out["rules"])


def test_own_sale_impact_comes_from_the_reserves_at_each_sale():
    assert ss._keep((100.0, 1_000_000.0), None) == 1.0
    assert ss._keep((50.0, 2_000_000.0), 10_000.0) > ss._keep((50.0, 500_000.0), 10_000.0)   # deeper = smaller impact
    # liquidity withdrawn at second 60 (both reserves down, price halves), then a further slide at 200 on thin reserves
    pool = lambda s: (100.0, 1e6) if s < 60 else (25.0, 5e5) if s < 200 else (5.0, 2.5e5)   # noqa: E731
    with_impact = ss.study(*_setup({"f": (pool, -0.86, None)}, liq=2_000.0), now=T0 + 3 * D)
    without = ss.study(*_setup({"f": (pool, -0.86, None)}), now=T0 + 3 * D)
    assert with_impact["coverage"]["impact_modelled"] == 1 and without["coverage"]["impact_modelled"] == 0
    a, b = _rules(with_impact)["stop -20%"], _rules(without)["stop -20%"]
    assert a["fired"] == b["fired"] == 1
    # SOL $20 (liq $2k / 2 / 50 SOL), so $100 = 50k tokens at the buy price. Stop sells at price 0.5 into a 500k token
    # reserve (keep 0.909 -> 0.455); the real sale was at 0.2 into 250k (keep 0.833 -> 0.167): +28.8 pts vs +30.0
    assert a["delta_usd"] == 29 and b["delta_usd"] == 30


def test_fetch_reads_each_hold_once_within_the_credit_cap():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    for i in range(3):
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, status) VALUES(?,?,?,?)",
                   (f"P{i}", f"M{i}", T0 + i * 1000, "done"))
        j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) "
                  "VALUES(?,?,?,?,?,?)", (f"M{i}", 100.0, T0 + i * 1000 + 10, T0 + i * 1000 + 360, 110.0, ss.BOOK))

    class FakeHelius:
        credits_used = 0
        calls = []

        def _rpc(self, method, params, credits):
            self.credits_used += credits
            self.calls.append(params)
            pool, lo = params[0], params[1]["filters"]["blockTime"]["gte"]
            return {"data": [_tx(lo + s, 50, 100, pool, "M" + pool[1:]) for s in range(0, 360, 30)]}

    h = FakeHelius()
    tr = ss.trades(j, st, ss.BOOK, 0, T0 + 9000)
    got = ss.fetch_paths(st, h, tr, max_credits=95)
    assert got["fetched"] == 2 and got["stopped_by_cap"]          # room for a full 80-credit hold is kept back
    assert ss.fetch_paths(st, h, tr, max_credits=95)["fetched"] == 1
    assert len(ss.load_paths(st)) == 3
    assert {c[1]["filters"]["blockTime"]["gte"] for c in h.calls} == {int(t["opened"]) - ss.PRE_S for t in tr}


def test_cli_runs_without_data(cfg, tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    cli.main(["sz-stops"])
    assert "STOPS AND CLIMB CHECKS" in capsys.readouterr().out
