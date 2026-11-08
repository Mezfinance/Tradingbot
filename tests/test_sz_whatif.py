"""sz-whatif: launchrw50_sz rebuilt with another book's trades in some bands, at sz's sizes."""
import json

import pytest

from tradingbot import launch_live as ll
from tradingbot import sz_whatif as sw
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

T0 = 1_700_006_400.0


def _pool(st, i, liq):
    st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, status, ret, liq_usd, feat) "
               "VALUES (?,?,?,?,?,?,?,?)", (f"P{i}", f"M{i}", T0 + i * 60, 1, "done", 0.0, liq, json.dumps({})))


def _buy(j, book, i, cost, back, held_min=6):
    ts = T0 + i * 60 + 10
    j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
              "VALUES (?,?,?,?,0,?,?,?)",
              (f"M{i}", f"M{i}", cost, back, ts, None if back is None else ts + held_min * 60, book))


def test_whatif_swaps_bands_at_sz_sizes():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    for i, liq in enumerate((10_000, 30_000, 30_000, 60_000, 60_000, 100_000), start=1):
        _pool(st, i, liq)
    _buy(j, "launchrw50_sz", 1, 250, 300)          # thin: +20%, kept in every variant
    _buy(j, "launchrw50_sz", 2, 100, 90)           # $25-40k: -10%
    _buy(j, "launchrw50_sz", 4, 100, 0)            # $40-80k: rug
    _buy(j, "launchrw50_sz", 6, 100, 105)          # $80k+: +5%, kept
    _buy(j, "launchrw50_25k15", 2, 250, 300, 15)   # same pool held to 15: +20%
    _buy(j, "launchrw50_25k15", 3, 250, 275, 15)   # a pool sz didn't hold: +10%
    _buy(j, "launchbig40k", 5, 250, 375, 15)       # the +50% filter's pick: +50%
    out = sw.whatif(st, j, now=T0 + 6 * 3600)
    v = out["variants"]
    assert "no trades yet" in v["$25-40k held to 10"]["note"]
    hold = v["$25-40k held to 15"]
    assert hold["sz"]["n"] == 4 and hold["sz"]["pnl_usd"] == pytest.approx(50 - 10 - 100 + 5)
    # swap at sz's $100 for $25-40k: +20 + 10 replace -10
    assert hold["variant"]["pnl_usd"] == pytest.approx(50 + 20 + 10 - 100 + 5)
    assert hold["variant minus sz"]["pnl_usd"] == pytest.approx(40)
    assert hold["in the swapped bands"]["variant"]["n"] == 2
    # by default the window starts at the swap books' first buy: the $40-80k book bought last, so the combined
    # variant only sees the trades from a minute before that buy
    late = v["$25-40k held to 15 + $40-80k +50% filter, held while rising"]
    assert late["sz"]["n"] == 2 and late["variant"]["pnl_usd"] == pytest.approx(50 + 5)
    out = sw.whatif(st, j, hours=8, now=T0 + 6 * 3600)
    v = out["variants"]
    both = v["$25-40k held to 15 + $40-80k +50% filter, held while rising"]
    assert both["variant"]["pnl_usd"] == pytest.approx(50 + 20 + 10 + 50 + 5)
    assert both["variant by band"]["$40-80k"]["n"] == 1 and both["sz by band"]["$40-80k"]["n"] == 1
    # money held longer earns per hour: 15 min at $100 costs 2.5x the money-minutes of 6 min
    assert both["variant"]["money_hours"] > both["sz"]["money_hours"]
    assert "SZ WHAT-IF" in sw.render(out)
