import json

import pytest

from tradingbot import watch_analysis as wa
from tradingbot.watch_features import Context, words
from tradingbot.watcher import WatchStore

T0 = 1_800_000_000.0
SNAP = T0 + 15 * 60  # the 15-minute snapshot time used below


def store_with_pool(pool="P", mint="M", name="Doge Rocket / SOL", created=T0, first_price=1.0):
    st = WatchStore(":memory:")
    add(st, pool, mint, name, created, first_price)
    return st


def add(st, pool, mint, name, created, first_price=1.0, snaps=None):
    st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts, first) VALUES(?,?,?,?,?,?,?)",
               (pool, mint, "pump-fun", name, created, created, json.dumps({"price": first_price})))
    for ts, price in (snaps or []):
        st.execute("INSERT INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
                   (pool, int((ts - created) / 60), ts, json.dumps({"price": price})))


def enrich(st, pool, kind, ts, data):
    st.execute("INSERT INTO watch_enrich(pool, kind, ts, data) VALUES(?,?,?,?)", (pool, kind, ts, json.dumps(data)))


def feats(st, pool="P", ts=SNAP, info_ok=False):
    p = dict(st.query("SELECT * FROM watch_pools WHERE pool=?", (pool,))[0])
    return Context(st).features(pool, p, ts, info_ok)


def test_enrichment_fetched_after_the_snapshot_is_ignored():
    st = store_with_pool()
    enrich(st, "P", "pumpfun15", SNAP + 1, {"is_live": True, "creator": "D"})
    assert "pf_live" not in feats(st)
    enrich(st, "P", "pumpfun5", SNAP - 60, {"is_live": True, "creator": "D", "creator_launches": 1424,
                                            "creator_recent_prior": 10, "creator_prior_graduated": 1,
                                            "creator_prior_names": ["DOGE ROCKET"]})
    f = feats(st)
    assert f["pf_live"] == 1.0 and f["creator_grad_rate"] == 0.1 and f["creator_reused_name"] == 1.0


def test_helius_cuts_trades_at_the_snapshot_and_excludes_the_pool():
    st = store_with_pool()
    trades = [[T0 + 1, 100, "DEV", 50, "DEV"], [T0 + 1, 100, "BUNDLER", 100, "BUNDLER"],
              [T0 + 5, 102, "SNIPER", 30, "SNIPER"], [T0 + 60, 140, "LATE", 10, "LATE"],
              [T0 + 1, 100, "CURVE", -180, "BUNDLER"],
              [SNAP + 30, 999, "DEV", -50, "DEV"]]  # the developer sells AFTER the snapshot
    enrich(st, "P", "helius", SNAP - 30, {
        "trades": trades, "supply": 1000.0, "creator": "DEV", "pool_accounts": ["CURVE", "P"], "n_txs": 6,
        "holders": [["CURVE", 800.0], ["BUNDLER", 100.0], ["DEV", 50.0]],
        "wallets": {"BUNDLER": {"first_ts": T0 - 600, "funder": "DEV"}, "SNIPER": {"first_ts": T0 - 600,
                                                                                   "funder": "DEV"}}})
    f = feats(st)
    assert f["hel_bundle_share"] == pytest.approx(0.10)  # launch slot + 1: the bundler, not the slot-102 sniper
    assert f["hel_sniper_share_10s"] == pytest.approx(0.13)  # first 10 seconds: bundler + sniper
    assert f["hel_dev_initial_share"] == pytest.approx(0.05)
    assert f["hel_dev_sold"] == 0.0  # the later sell is invisible at the snapshot
    assert f["hel_top1_share"] == pytest.approx(0.10)  # the curve's 80% is not a holder
    assert f["hel_fresh_wallet_share"] == 1.0 and f["hel_same_funder_share"] == 1.0
    assert f["hel_funded_by_creator"] == 1.0


def test_smart_money_needs_an_earlier_pump_that_already_happened():
    st = WatchStore(":memory:")
    # An earlier token that WHALE bought early and that hit 2x at T0 - 3600 (before our snapshot).
    add(st, "OLD", "OLDM", "Old / SOL", T0 - 7200, 1.0, snaps=[(T0 - 3600, 2.5)])
    enrich(st, "OLD", "helius", T0 - 7000, {"trades": [[T0 - 7190, 1, "WHALE", 10, "WHALE"]], "supply": 100.0,
                                            "pool_accounts": [], "creator": "X"})
    # An earlier token LATEBIRD bought, which only pumps AFTER our snapshot.
    add(st, "OLD2", "OLD2M", "Other / SOL", T0 - 7200, 1.0, snaps=[(SNAP + 600, 3.0)])
    enrich(st, "OLD2", "helius", T0 - 7000, {"trades": [[T0 - 7190, 1, "LATEBIRD", 10, "LATEBIRD"]],
                                             "supply": 100.0, "pool_accounts": [], "creator": "X"})
    add(st, "P", "M", "New / SOL", T0)
    enrich(st, "P", "helius", SNAP - 10, {"trades": [[T0 + 2, 5, "WHALE", 5, "WHALE"],
                                                     [T0 + 3, 6, "LATEBIRD", 5, "LATEBIRD"]],
                                          "supply": 100.0, "pool_accounts": [], "creator": "DEV"})
    f = feats(st)
    assert f["hel_smart_wallets"] == 1.0  # WHALE counts, LATEBIRD's pump hadn't happened yet


def test_x_posts_after_the_snapshot_do_not_count():
    st = store_with_pool()
    enrich(st, "P", "x", SNAP - 5, {"count_7d": 3, "posts": [
        {"ts": SNAP - 100, "author": "a", "followers": 80_000, "likes": 3, "reposts": 1},
        {"ts": SNAP + 100, "author": "b", "followers": 1_000_000}]})
    f = feats(st)
    assert f["x_kol"] == 1.0 and f["x_log_max_followers"] == pytest.approx(__import__("math").log1p(80_000))


def test_lists_telegram_and_paid_orders_respect_time():
    st = store_with_pool()
    st.execute("INSERT INTO watch_lists(ts, list, rank, mint, amount) VALUES(?,?,?,?,?)",
               (SNAP - 60, "gecko:trending_5m", 4, "M", 0))
    st.execute("INSERT INTO watch_lists(ts, list, rank, mint, amount) VALUES(?,?,?,?,?)",
               (SNAP + 60, "dex:token-boosts:top", 1, "M", 500))
    st.execute("INSERT INTO watch_mentions(source, channel, post, mint, post_ts, seen_ts) VALUES(?,?,?,?,?,?)",
               ("tg", "calls", "calls/1", "M", SNAP - 120, SNAP))
    st.execute("INSERT INTO watch_mentions(source, channel, post, mint, post_ts, seen_ts) VALUES(?,?,?,?,?,?)",
               ("tg", "calls2", "calls2/9", "M", SNAP + 120, SNAP + 130))
    enrich(st, "P", "orders", SNAP - 1, {"orders": [{"type": "tokenProfile", "paid_ts": T0 + 300},
                                                    {"type": "tokenAd", "paid_ts": SNAP + 50}]})
    f = feats(st)
    assert f["list_gt5m"] == 1.0 and f["list_gt5m_inv_rank"] == 0.25 and f["list_dexboost_top"] == 0.0
    assert f["tg_channels"] == 1.0 and f["tg_first_min_after_launch"] == pytest.approx(13.0)
    assert f["paid_profile"] == 1.0 and f["paid_ads"] == 0.0 and f["paid_min_after_launch"] == 5.0


def test_theme_heat_and_copied_images():
    st = WatchStore(":memory:")
    add(st, "A", "AM", "Rocket Frog / SOL", T0 - 1800, 1.0, snaps=[(T0 - 600, 2.2)])  # pumped before
    add(st, "B", "BM", "Frog King / SOL", T0 - 900)
    add(st, "P", "M", "Frog Moon / SOL", T0)
    for pool, created in (("A", T0 - 1800), ("P", T0)):
        enrich(st, pool, "image", created + 60, {"dhash": "abc"})
    f = feats(st)
    assert words("Frog Moon / SOL") == {"frog", "moon"}
    assert f["theme_launches_1h"] == pytest.approx(__import__("math").log1p(2))  # A and B, not itself
    assert f["theme_pumped_6h"] == pytest.approx(__import__("math").log1p(1))
    assert f["image_ever_reused"] == 1.0


def test_market_conditions_use_only_past_rows():
    st = store_with_pool()
    for ts, sol, n in ((SNAP - 3700, 100.0, 50), (SNAP - 1800, 101.0, 60), (SNAP - 60, 110.0, 40),
                       (SNAP + 600, 200.0, 999)):
        st.execute("INSERT INTO watch_market(ts, sol_usd, new_pools, new_launchpad) VALUES(?,?,?,?)",
                   (ts, sol, n, n // 2))
    f = feats(st)
    assert f["mkt_sol_chg_1h"] == pytest.approx(0.10)
    assert f["mkt_new_pools_1h"] == pytest.approx(__import__("math").log1p(100))


def test_enriched_subset_analysis_runs_on_rows_with_wallet_data():
    rows = [{"pool": f"P{i}", "mint": f"M{i}", "dex": "pump-fun", "created_ts": T0 + i, "age": 15,
             "hel_early_buyers": float(i % 7), "x": float(i % 3), "pump2x": i % 4 == 0, "pump5x": False,
             "clean2x": False, "trade_ret": 0.0, "best_mult": 1.0, "best_close_mult": 1.0, "worst_mult": 1.0}
            for i in range(300)] + [{"pool": "Z", "mint": "Z", "dex": "pump-fun", "created_ts": T0, "age": 15,
                                     "pump2x": True, "pump5x": False, "clean2x": False, "trade_ret": 0.0,
                                     "best_mult": 1.0, "best_close_mult": 1.0, "worst_mult": 1.0}]
    a = wa.analyse(WatchStore(":memory:"), 15, "pump2x", subset="enriched", rows=rows)
    assert a["rows"] == 300 and a["subset"] == "enriched"
