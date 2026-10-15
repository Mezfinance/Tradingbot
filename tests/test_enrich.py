import io
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx

from tradingbot.config import Config
from tradingbot.data.helius import Helius, funder_of, trades
from tradingbot.data.pumpfun import compact_coin, compact_creator
from tradingbot.data.telegram import parse_page
from tradingbot.data.xapi import XClient
from tradingbot.enrich import Budget, Enricher, dhash
from tradingbot.watcher import WatchStore

T0 = 1_800_000_000.0
MINT = "MintAAAApump"


def png(color):
    from PIL import Image
    img = Image.new("RGB", (64, 64), color)
    for x in range(32):
        img.putpixel((x, x), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_dhash_survives_resizing_and_spots_different_pictures():
    from PIL import Image
    a = png((10, 10, 10))
    big = io.BytesIO()
    Image.open(io.BytesIO(a)).resize((256, 256)).save(big, "JPEG")
    assert dhash(a) == dhash(big.getvalue())
    assert dhash(b"not an image") is None


def test_budget_is_spread_over_the_day():
    b = Budget(WatchStore(":memory:"), "x", daily=24.0)
    one_am = datetime(2026, 10, 16, 1, tzinfo=timezone.utc)
    assert b.allows(2.0, one_am) and not b.allows(2.5, one_am)  # 2/24 of the day by 01:59
    b.add(2.0, one_am)
    assert not b.allows(0.1, one_am)
    assert b.allows(20.0, datetime(2026, 10, 16, 23, tzinfo=timezone.utc))


def test_telegram_page_parsing():
    page = ('<div data-post="calls/41"><time datetime="2026-10-15T01:02:03+00:00"></time>'
            '<div class="text">ape <b>7jNwX4Me9zGEjmWY7zARDG2LYjVs1WXD6DZ1JvSfpump</b> now</div></div>'
            '<div data-post="calls/42"><time datetime="2026-10-15T01:05:00+00:00"></time>gm frens</div>')
    [post] = parse_page(page)
    assert post["post"] == "calls/41" and post["mints"] == ["7jNwX4Me9zGEjmWY7zARDG2LYjVs1WXD6DZ1JvSfpump"]
    assert post["ts"] == datetime(2026, 10, 15, 1, 2, 3, tzinfo=timezone.utc).timestamp()


def tx(block_time, slot, payer, changes, err=None, sol=None):
    """changes: {owner: (pre, post)} token balances of MINT."""
    pre = [{"mint": MINT, "owner": o, "uiTokenAmount": {"uiAmount": a}} for o, (a, _) in changes.items()]
    post = [{"mint": MINT, "owner": o, "uiTokenAmount": {"uiAmount": b}} for o, (_, b) in changes.items()]
    keys = [payer] + [k for k in (sol or {}) if k != payer]
    meta = {"err": err, "preTokenBalances": pre, "postTokenBalances": post}
    if sol:
        meta["preBalances"] = [sol[k][0] for k in keys]
        meta["postBalances"] = [sol[k][1] for k in keys]
    return {"blockTime": block_time, "slot": slot, "meta": meta,
            "transaction": {"message": {"accountKeys": [{"pubkey": k} for k in keys]}}}


def test_helius_trades_and_funder():
    txs = [tx(100, 5, "DEV", {"DEV": (0, 50), "CURVE": (1000, 950)}),
           tx(101, 6, "BOT", {"BOT": (0, 30), "CURVE": (950, 920)}),
           tx(102, 7, "X", {"X": (0, 1)}, err="failed")]
    rows = trades(txs, MINT)
    assert [100, 5, "DEV", 50, "DEV"] in [r[:5] for r in rows] and [101, 6, "CURVE", -30, "BOT"] in [r[:5] for r in rows]
    assert all(r[2] != "X" for r in rows)  # failed transactions are ignored
    funding = tx(90, 1, "CEX", {}, sol={"CEX": (10_000, 4_000), "NEWBIE": (0, 5_000)})
    assert funder_of(funding, "NEWBIE") == "CEX"


def test_pumpfun_compacting():
    c = compact_coin({"creator": "C", "is_currently_live": True, "ath_market_cap": 9000, "usd_market_cap": 3000,
                      "security_verdict": {"verdict": "allow"}, "twitter": "https://x.com/a/status/1"})
    assert c["is_live"] and c["ath_market_cap_usd"] == 9000 and c["security_verdict"] == "allow"
    hist = compact_creator({"count": 1424, "coins": [
        {"symbol": "A", "created_timestamp": 1_000, "complete": True},
        {"symbol": "B", "created_timestamp": 5_000, "complete": False},
        {"symbol": "LATER", "created_timestamp": 99_000}, {"symbol": "NOTS"}]}, before_ms=10_000)
    assert hist["creator_launches"] == 1424 and hist["creator_recent_prior"] == 2
    assert hist["creator_prior_graduated"] == 1 and hist["creator_prior_names"] == ["A", "B"]


def test_x_client_costs_and_parses():
    def handler(req):
        if req.url.path.endswith("/counts/recent"):
            return httpx.Response(200, json={"meta": {"total_tweet_count": 7}})
        return httpx.Response(200, json={
            "data": [{"id": "1", "author_id": "u1", "created_at": "2026-10-15T00:00:00Z",
                      "public_metrics": {"like_count": 5, "retweet_count": 2, "reply_count": 1}}],
            "includes": {"users": [{"id": "u1", "created_at": "2020-01-01T00:00:00Z", "verified": True,
                                    "public_metrics": {"followers_count": 120_000}}]}})
    x = XClient("tok", httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x"))
    assert x.count("q") == 7
    [p] = x.search("q")
    assert p["followers"] == 120_000 and p["verified"] and abs(x.cost_usd - 0.01) < 1e-9
    assert not XClient("").enabled


# --- the enricher with fake providers ------------------------------------------------
class FakeHelius:
    enabled = True

    def __init__(self):
        self.credits_used = 0

    def history(self, address, limit=100):
        self.credits_used += 10
        return [tx(T0 + 1, 10, "DEV", {"DEV": (0, 50), "CURVE": (1000, 950)}),
                tx(T0 + 2, 10, "SNIPER", {"SNIPER": (0, 200), "CURVE": (950, 750)})]

    def largest_accounts(self, mint):
        self.credits_used += 1
        return [{"address": "acct1", "uiAmountString": "700"}, {"address": "acct2", "uiAmountString": "200"}]

    def owners(self, accts):
        self.credits_used += 1
        return {"acct1": "CURVE", "acct2": "SNIPER"}

    def token_supply(self, mint):
        self.credits_used += 1
        return 1000.0

    def wallet_origin(self, w):
        self.credits_used += 10
        return {"first_ts": T0 - 60, "funder": "CEX"}

    def balances(self, wallets):
        self.credits_used += 1
        return {w: 2.5 for w in wallets}


class FakeExtras:
    """Offline stand-in for every free source."""

    def jupiter_token(self, mint):
        return {"holderCount": 120, "organicScore": 40.0, "tags": ["verified"], "audit": {"topHoldersPercentage": 12},
                "stats5m": {"buyOrganicVolume": 50, "buyVolume": 100}, "stats1h": {}}

    def jupiter_trending(self, interval="1h", limit=100):
        return [MINT]

    def rugcheck(self, mint):
        return {"score_normalised": 3, "risks": [{"name": "Mutable metadata"}], "lpLockedPct": 100}

    def dex_search(self, q):
        return [{"chainId": "base", "baseToken": {"symbol": q, "address": "0xabc"}, "volume": {"h24": 9000},
                 "priceChange": {"h24": 300}}]

    def telegram_members(self, h):
        return 512

    def discord_members(self, u):
        return 99

    def website(self, u):
        return {"reachable": True, "bytes": 1000, "links": 3, "scripts": 1, "images": 2, "template_host": False}

    def domain_registered(self, d):
        return T0 - 86400

    def coinbase_spot(self, pair):
        return 80000.0 if pair.startswith("BTC") else 2400.0

    def fear_greed(self):
        return 55.0

    def hyperliquid(self):
        return {"sol_funding": 0.0001, "sol_oi": 5e6, "perp_symbols": ["SOL", "WIF"]}

    def solana_stables(self):
        return 16e9, 15e9

    def solana_load(self):
        return {"tps": 4000.0, "priority_fee_median": 1000}

    def biz_catalog(self):
        return [{"post": "biz/1", "ts": T0, "mints": [MINT]}]


def store_with(pools):
    st = WatchStore(":memory:")
    for pool, mint, dex, buyers in pools:
        st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts) VALUES(?,?,?,?,?,?)",
                   (pool, mint, dex, f"{pool} / SOL", T0, T0))
        st.execute("INSERT INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
                   (pool, 5, T0 + 300, json.dumps({"buyers_m5": buyers})))
    return st


def make_enricher(st, **kw):
    pump = SimpleNamespace(coin=lambda m: {"creator": "DEV", "bonding_curve": "CURVE", "image_uri": "",
                                           "telegram": "https://t.me/grp", "website": "https://frog.xyz"},
                           creator_coins=lambda c, limit=50: {"count": 3, "coins": []},
                           user=lambda a: {"username": "dev", "followers": "1200", "x_username": "devx"})
    dex = SimpleNamespace(lists=lambda: {"token-boosts:top": [(1, MINT, 500.0)]}, orders=lambda m: [])
    gecko = SimpleNamespace(trending_ranked=lambda d: [MINT], sol_price=lambda: 104.0,
                            token_prices=lambda mints: {m: 1.0 for m in mints})
    quotes = SimpleNamespace(round_trip=lambda m, usd: (7.5, 1.2),
                             quote=lambda a, b, raw: SimpleNamespace(price_impact_pct=4.0, route=["Raydium"]))
    cfg = Config(db_path=":memory:", **kw)
    return Enricher(cfg, st, None, gecko=gecko, dex=dex, pump=pump, helius=FakeHelius(),
                    x=XClient(""), llm_client=None, http=httpx.Client(transport=httpx.MockTransport(
                        lambda r: httpx.Response(404))), tg_fetch=lambda ch: [], extras=FakeExtras(), jupiter=quotes)


def test_paid_lookups_only_for_active_launches_and_graduations():
    st = store_with([("BUSY", MINT, "pump-fun", 40), ("QUIET", "Q" * 10, "pump-fun", 3),
                     ("GRAD", "G" * 10, "pumpswap", 0)])
    e = make_enricher(st)
    e.helius_budget.allows = lambda cost, now=None: True  # budget open regardless of the test's clock
    for _ in range(4):
        e.run_tasks(T0 + 20 * 60)
    got = {(r["pool"], r["kind"]): json.loads(r["data"]) for r in st.query("SELECT * FROM watch_enrich")}
    assert got[("QUIET", "helius")] == {"skipped": "not active enough"}
    busy = got[("BUSY", "helius")]
    assert busy["creator"] == "DEV" and "SNIPER" in busy["wallets"] and busy["supply"] == 1000.0
    assert ["SNIPER", 200.0] in busy["holders"] and busy["pool_accounts"] == ["BUSY", "CURVE"]
    assert "helius" in {k for p, k in got if p == "GRAD"}
    assert got[("BUSY", "x")] == {"skipped": "no X_BEARER_TOKEN"}
    assert st.query("SELECT funder FROM watch_wallets WHERE wallet='SNIPER'")[0]["funder"] == "CEX"


def test_out_of_budget_waits_instead_of_marking_done():
    st = store_with([("BUSY", MINT, "pump-fun", 40)])
    e = make_enricher(st, helius_daily_credits=0)
    e.run_tasks(T0 + 20 * 60)
    assert not st.query("SELECT 1 FROM watch_enrich WHERE kind='helius'")


def test_lists_and_market_are_recorded():
    st = WatchStore(":memory:")
    e = make_enricher(st)
    e.watcher = SimpleNamespace(seen_counts=[30, 25])
    assert e.poll_lists(T0) == 3  # one DexScreener row + gecko 5m + gecko 1h
    e.poll_market(T0)
    m = dict(st.query("SELECT * FROM watch_market")[0])
    assert m["sol_usd"] == 104.0 and m["new_pools"] == 30 and e.watcher.seen_counts == [0, 0]


def test_free_per_token_sources_are_recorded():
    st = store_with([("BUSY", MINT, "pump-fun", 40)])
    e = make_enricher(st)
    e.helius_budget.allows = lambda cost, now=None: True
    for _ in range(3):
        e.run_tasks(T0 + 20 * 60)
    got = {r["kind"]: json.loads(r["data"]) for r in st.query("SELECT kind, data FROM watch_enrich WHERE pool='BUSY'")}
    assert got["jupiter"]["holders"] == 120 and got["jupiter"]["verified"]
    assert got["rugcheck"]["risks"] == ["Mutable metadata"]
    assert got["jupquote"]["round_trip_loss_pct_100"] == 7.5 and got["jupquote"]["impact_pct_500"] == 4.0
    assert got["links"]["telegram_members"] == 512 and got["links"]["domain_registered_ts"] == T0 - 86400
    assert got["crosschain"]["other_chain_pairs"] == 1
    assert got["dev"]["profile"]["followers"] == 1200
    assert got["helius"]["sol_balances"]["SNIPER"] == 2.5


def test_market_wide_sources_are_recorded():
    st = WatchStore(":memory:")
    e = make_enricher(st)
    assert e.poll_macro(T0) > 5
    keys = {r["key"] for r in st.query("SELECT key FROM watch_macro")}
    assert {"btc_usd", "fear_greed", "sol_funding", "sol_stables_7d_chg", "tps", "meme_wif_usd"} <= keys
    assert json.loads(st.get_state("perp_symbols")) == ["SOL", "WIF"]
    assert e.poll_biz(T0) == 1
    assert st.query("SELECT source FROM watch_mentions")[0]["source"] == "4chan"
