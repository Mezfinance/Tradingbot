import json
import math

import pytest

from tradingbot.config import Config
from tradingbot.watch_features import Context
from tradingbot.watch_features_more import More, moon_phase
from tradingbot.watcher import WatchStore
from tests.test_watch_features import SNAP, T0, add, enrich


@pytest.fixture(autouse=True)
def strict():
    More.strict = True
    yield
    More.strict = False


def rich_store():
    st = WatchStore(":memory:")
    add(st, "P", "M", "FROG / SOL", T0, 1.0, snaps=[(T0 + 300, 1.5), (SNAP, 1.2)])
    add(st, "OLD", "OM", "FROG / SOL", T0 - 3 * 86400, 1.0, snaps=[(T0 - 3 * 86400 + 600, 3.0)])
    trades = [[T0 + 1, 10, "DEV", 50, "DEV", 5000, 0], [T0 + 2, 10, "B1", 100, "B1", 15000, 100000],
              [T0 + 20, 12, "B2", 30, "B2", 6000, 0], [T0 + 30, 13, "B1", -20, "B1", 5000, 0],
              [T0 + 40, 14, "B3", 1, "DEV", 5000, 0], [SNAP + 60, 99, "B4", 500, "B4", 5000, 0]]
    enrich(st, "P", "helius", SNAP - 30, {"trades": trades, "supply": 1000.0, "creator": "DEV", "pool_accounts": ["P"],
                                          "holders": [["B1", 80.0], ["B2", 30.0], ["DEV", 50.0]], "n_txs": 6,
                                          "wallets": {"B1": {"first_ts": T0 - 100, "funder": "CEX1"}},
                                          "sol_balances": {"B1": 3.0, "B2": 0.5, "DEV": 10.0}})
    enrich(st, "P", "pumpfun5", T0 + 310, {"creator": "DEV"})
    enrich(st, "P", "jupiter", SNAP - 20, {"usd_price": 1.3, "organic_score": 40, "verified": False,
                                          "token_program": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
                                          "decimals": 6, "total_supply": 1e9, "dev_mints": 3,
                                          "stats5m": {"buyVolume": 100, "buyOrganicVolume": 40, "numNetBuyers": 5},
                                          "stats1h": {"liquidityChange": 12}})
    enrich(st, "P", "rugcheck", SNAP - 20, {"score": 500, "risks": ["Mutable metadata", "Top 10 insiders"],
                                           "lp_locked_pct": 100})
    enrich(st, "P", "jupquote", SNAP - 20, {"routable": True, "round_trip_loss_pct_100": 8.0, "impact_pct_500": 3.0})
    enrich(st, "P", "dev", SNAP - 20, {"profile": {"username": True, "followers": 20000, "x_username": "dev"},
                                      "prior_best_ath_usd": 250000, "origin": {"first_ts": T0 - 10 * 86400, "funder": "CEX1"}})
    enrich(st, "P", "links", SNAP - 20, {"telegram_members": 300, "website": {"reachable": True, "bytes": 5000,
                                                                            "links": 4, "template_host": True},
                                        "domain_registered_ts": T0 - 2 * 86400})
    enrich(st, "P", "crosschain", SNAP - 20, {"other_chain_pairs": 2, "other_chain_max_vol24": 50000,
                                             "solana_older_max_vol24": 1000})
    enrich(st, "P", "x", SNAP - 20, {"count_7d": 5, "cashtag_count_7d": 40, "posts": [
        {"ts": SNAP - 100, "username": "paidguy", "followers": 10}],
        "linked_post": {"likes": 900, "views": 100000, "quotes": 20, "author_followers": 2_000_000}})
    enrich(st, "P", "sentiment", SNAP - 10, {"sentiment": 0.7, "shill_share": 0.4, "rug_warnings": False,
                                            "organic_discussion": True})
    enrich(st, "P", "narrative", SNAP - 10, {"category": "animal", "memeability": 0.8, "has_lore": True,
                                            "description_tone": "lore", "humor": "absurd", "image_has_animal": True})
    enrich(st, "P", "image", SNAP - 10, {"dhash": "x", "mean_rgb": [200, 40, 40], "animated": True})
    enrich(st, "P", "candles15", SNAP - 5, {"candles": [[T0 + 60 * i, 1.0, 1.2, 0.9, 1.0 + 0.05 * i, 10]
                                                         for i in range(20)]})
    enrich(st, "P", "orders", SNAP - 5, {"orders": [{"type": "trendingBarAd", "paid_ts": T0 + 400}]})
    for ts, key, val in ((SNAP - 3600, "btc_usd", 80000), (SNAP - 60, "btc_usd", 84000), (SNAP + 60, "btc_usd", 1),
                         (SNAP - 60, "fear_greed", 70), (SNAP - 86400, "meme_wif_usd", 1.0),
                         (SNAP - 60, "meme_wif_usd", 0.8)):
        st.execute("INSERT INTO watch_macro(ts, key, value) VALUES(?,?,?)", (ts, key, val))
    st.execute("INSERT INTO watch_xfeed(ts, account, post_ts, text) VALUES(?,?,?,?)",
               (SNAP, "elonmusk", SNAP - 500, "I love this frog"))
    st.execute("INSERT INTO watch_xfeed(ts, account, post_ts, text) VALUES(?,?,?,?)",
               (SNAP + 900, "elonmusk", SNAP + 800, "frog frog frog"))
    st.execute("INSERT INTO watch_mentions(source, channel, post, mint, post_ts, seen_ts) VALUES(?,?,?,?,?,?)",
               ("4chan", "biz", "biz/1", "M", SNAP - 50, SNAP))
    st.set_state("perp_symbols", json.dumps(["FROG", "SOL"]))
    return st


def features():
    st = rich_store()
    cfg = Config(db_path=":memory:", cex_wallets="CEX1", x_paid_callers="@paidguy", kol_wallets="B2")
    ctx = Context(st, cfg)
    p = dict(st.query("SELECT * FROM watch_pools WHERE pool='P'")[0])
    snap = {"price": 1.2, "fdv": 35000, "chg_m5": 10, "chg_h1": 40, "vol_m5": 500, "vol_h1": 2000}
    return ctx.features("P", p, SNAP, False, snap)


def test_every_factor_group_computes_and_respects_time():
    f = features()
    assert f["pp_drawdown_from_high"] == pytest.approx(1.2 / 1.5 - 1)
    assert f["pp_mcap_levels_crossed"] == 2.0 and f["pp_jup_vs_snapshot_price"] == pytest.approx(1.3 / 1.2 - 1)
    assert f["c1_green_run"] >= 1 and f["c1_above_vwap"] in (0.0, 1.0)
    assert f["fl_largest_buy_share"] == pytest.approx(0.1) and f["fl_early_seller_share"] == pytest.approx(1 / 4)  # DEV, B1, B2, B3 bought; B1 sold
    assert f["jup_stats5m_organic_buy_share"] == pytest.approx(0.4)
    assert f["lq_impact_500"] == 3.0 and f["lq_quote_sol"] == 1.0 and f["lq_jup_verified"] == 0.0
    assert f["ho_insiders_flag"] == 1.0 and f["ho_cex_share"] == 0.0 and f["ho_airdropped_share"] > 0
    assert f["wl_jito_tipped_share"] == pytest.approx(1 / 4) and f["wl_kol_buyers"] == 1.0
    assert f["wl_cex_funded_share"] == 1.0
    assert f["dv_public_figure"] == 1.0 and f["dv_wallet_age_days"] == pytest.approx(10.0)
    assert f["sc_mutable_metadata"] == 1.0 and f["sc_token_2022"] == 1.0 and f["sc_sell_ok"] == 1.0
    assert f["id_ticker_reused"] == pytest.approx(math.log1p(1)) and f["id_ticker_prior_pumped"] == 1.0
    assert f["nr_has_lore"] == 1.0 and f["nr_humor_absurd"] == 1.0 and f["nr_other_chain_twins"] > 0
    assert f["sx_paid_callers"] == 1.0 and f["sx_linked_post_views"] > 0 and f["sx_sentiment"] == 0.7
    assert f["sx_theme_in_mega_posts"] == 1.0 and f["sx_mega_posts_24h"] == 1.0  # the later post is invisible
    assert f["sl_site_template"] == 1.0 and f["sl_domain_age_days"] == pytest.approx(2.0)
    assert f["sl_biz_mentions"] == 1.0
    assert f["at_trending_bar_ad"] == 1.0
    assert f["mk_btc_chg_1h"] == pytest.approx(0.05)  # the value after the snapshot is ignored
    assert f["mk_meme_index_chg_24h"] == pytest.approx(-0.2)
    assert f["lt_perp_listed"] == 1.0
    assert f["mn_wash_share"] == 0.0 and "cal_moon_sin" in f
    assert "tg_mentions" not in f or f.get("tg_channels", 0) == 0.0  # 4chan is not Telegram


def test_candles_after_the_snapshot_are_not_used():
    f = features()
    # 20 one-minute candles from launch; only those closed by the 15-minute snapshot count
    assert f["c1_green_run"] <= 15


def test_moon_phase_known_dates():
    from datetime import datetime, timezone
    full = datetime(2026, 10, 26, 4, 12, tzinfo=timezone.utc).timestamp()  # a full moon (almanac)
    assert abs(moon_phase(full) - 0.5) < 0.03
