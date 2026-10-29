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


def test_graduation_reuses_the_launch_pools_paid_rows():
    st = store_with([("LAUNCH", MINT, "pump-fun", 40), ("GRADP", MINT, "pumpswap", 40)])
    e = make_enricher(st)
    e.helius_budget.allows = lambda cost, now=None: True
    calls = []
    real = e.helius.history
    e.helius.history = lambda a, limit=100: calls.append(a) or real(a, limit)
    for _ in range(4):
        e.run_tasks(T0 + 20 * 60)
    rows = {r["pool"]: (r["ts"], json.loads(r["data"])) for r in st.query("SELECT * FROM watch_enrich WHERE kind='helius'")}
    assert len(calls) == 1  # paid once for the token
    [copy] = [p for p, (_, d) in rows.items() if "copied_from" in d]
    src = rows[copy][1]["copied_from"]
    assert rows[copy][0] == rows[src][0]  # keeps the fetch time: no peeking
    assert copy in rows[copy][1]["pool_accounts"] and rows[copy][1]["trades"] == rows[src][1]["trades"]


def test_only_genuine_graduations_skip_the_activity_check():
    st = store_with([("ORCA", "O" * 10, "orca", 3), ("BUSYORCA", "B" * 10, "orca", 40),
                     ("CURVE1", "C" * 10, "meteora-dbc", 3), ("DAMM", "C" * 10, "meteora-damm-v2", 0),
                     ("PUMPGRAD", "Zpump", "meteora-damm-v2", 0)])
    e = make_enricher(st)
    pools = {r["pool"]: dict(r) for r in e.store.query(
        "SELECT p.*, s.data AS snap5 FROM watch_pools p JOIN watch_snaps s ON s.pool=p.pool")}
    assert not e.eligible(pools["ORCA"]) and e.eligible(pools["BUSYORCA"])
    assert e.eligible(pools["DAMM"]) and e.eligible(pools["PUMPGRAD"])  # launchpad sibling / pump mint
    assert not e.eligible(pools["CURVE1"])


def seed_scores(st, kind, now, scores):
    for i, s in enumerate(scores):
        st.execute("INSERT INTO watch_enrich(pool, kind, ts, data) VALUES(?,?,?,?)",
                   (f"old{i}", kind, now - 5 * 3600, json.dumps({"score": s})))


def pool_row(name, buyers):
    return {"pool": name, "mint": name, "dex": "pumpswap", "snap5": json.dumps({"buyers_m5": buyers})}


def test_rationing_spends_on_the_best_scores_plus_a_random_control_slice():
    now = T0 + 6 * 3600
    st = WatchStore(":memory:")
    e = make_enricher(st, helius_daily_credits=3600)  # 5 lookups/hour at 30 credits
    assert e.select(pool_row("early", 0), "helius", now)["selection"] == "all"  # no history yet: everyone
    seed_scores(st, "helius", now, [i / 10 for i in range(100)])  # 20 eligible/hour -> coverage 0.25
    now += 1  # next pass (history is read once per pass)
    hot = e.select(pool_row("hot", 10_000), "helius", now)
    assert hot["pay"] and hot["selection"] in ("ranked", "control")
    picks = [e.select(pool_row(f"cold{i}", 0), "helius", now) for i in range(400)]
    control = [p for p in picks if p["pay"]]
    assert all(p["selection"] == "control" for p in control)
    assert 0.02 < len(control) / 400 < 0.09  # 20% of 0.25 coverage
    skipped = [p for p in picks if not p["pay"]][0]
    assert skipped["coverage"] == 0.25 and skipped["threshold"] > 7
    e2 = make_enricher(st, helius_daily_credits=1_000_000)
    assert e2.select(pool_row("any", 0), "helius", now)["selection"] == "all"  # budget covers everyone


def test_rationed_lookups_record_their_selection_and_cost():
    st = store_with([("BUSY", MINT, "pump-fun", 40), ("MEH", "M" * 10, "pumpswap", 0)])
    now = T0 + 20 * 60
    seed_scores(st, "helius", now, [4.0] * 40)
    e = make_enricher(st, helius_daily_credits=24 * 30 * 4)  # covers half of the 8 eligible per hour
    e.helius_budget.allows = lambda cost, now=None: True
    for _ in range(3):
        e.run_tasks(now)
    got = {r["pool"]: json.loads(r["data"]) for r in st.query("SELECT * FROM watch_enrich WHERE kind='helius' "
                                                               "AND pool IN ('BUSY','MEH')")}
    assert got["BUSY"]["selection"] in ("ranked", "control") and got["BUSY"]["score"] > 4
    meh = got["MEH"]
    assert meh.get("selection") == "control" or meh["skipped"] == "below rank threshold"
    assert float(st.get_state("avgcost:helius")) > 0


def test_status_shows_how_paid_lookups_were_chosen():
    st = store_with([("LAUNCH", MINT, "pump-fun", 40), ("GRADP", MINT, "pumpswap", 40), ("QUIET", "Q" * 10, "orca", 1)])
    e = make_enricher(st)
    e.helius_budget.allows = lambda cost, now=None: True
    for _ in range(3):
        e.run_tasks(T0 + 20 * 60)
    sel = e.status()["paid_selection_24h"]["helius"]
    assert sel == {"all": 1, "copied": 1, "not active enough": 1}


def _optx(ts, payer, logs, pool=None, sol=(None, None)):
    bal = lambda amt: [{"owner": pool, "mint": "So11111111111111111111111111111111111111112",
                        "uiTokenAmount": {"uiAmount": amt}}] if amt is not None else []
    return {"blockTime": ts, "transaction": {"message": {"accountKeys": [{"pubkey": payer}]}},
            "meta": {"err": None, "logMessages": logs, "preTokenBalances": bal(sol[0]),
                     "postTokenBalances": bal(sol[1])}}


class OperatorHelius(FakeHelius):
    """NEW was created by OP; OP's earlier pool OLD1 was pulled 10 minutes after launch, OLD2 not (yet)."""

    def history(self, address, limit=100):
        self.credits_used += 10
        if address == "NEW":
            return [_optx(T0, "OP", ["Program log: Instruction: CreatePool"], "NEW", (None, 85.0))]
        return super().history(address, limit)

    def recent(self, address, limit=50):
        self.credits_used += 10
        if address == "OLD1":
            return [_optx(T0 - 3000 + 30, "X", ["Program log: Instruction: Buy"]),
                    _optx(T0 - 3000, "OP", ["Program log: Instruction: Withdraw", "Program log: Instruction: Burn"],
                          "OLD1", (110.0, 0.0))]
        return [_optx(T0 - 100, "Y", ["Program log: Instruction: Sell", "Program log: Instruction: WithdrawFees"])]


def test_operator_lookup_records_the_creator_and_how_fast_it_pulled_before():
    from tradingbot.data.helius import find_withdraw
    st = store_with([("NEW", "NewMint", "pumpswap", 50), ("LOW", "LowMint", "pumpswap", 50)])
    st.execute("UPDATE watch_snaps SET data=? WHERE pool='NEW'", (json.dumps({"buyers_m5": 50, "liq": 30000}),))
    st.execute("UPDATE watch_snaps SET data=? WHERE pool='LOW'", (json.dumps({"buyers_m5": 50, "liq": 900}),))
    for p, created in (("OLD1", T0 - 3600), ("OLD2", T0 - 1800)):
        st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts) VALUES(?,?,?,?,?,?)",
                   (p, p + "Mint", "pumpswap", "SCAM / SOL", created, created))
        st.execute("INSERT INTO watch_enrich(pool, kind, ts, data) VALUES(?,?,?,?)",
                   (p, "operator", created + 330, json.dumps({"creator": "OP", "funder": "F"})))
    e = make_enricher(st)
    e.helius = OperatorHelius()
    e.operator_budget.allows = lambda cost, now=None: True
    e.run_tasks(T0 + 6 * 60)
    got = {r["pool"]: json.loads(r["data"]) for r in st.query("SELECT * FROM watch_enrich WHERE kind='operator'")}
    assert got["LOW"] == {"skipped": "below liquidity"}
    new = got["NEW"]
    assert new["creator"] == "OP" and new["seed_sol"] == 85.0 and new["create_ts"] == T0
    assert new["wallet_first_ts"] == T0 - 60 and new["funder"] == "CEX"
    assert new["prior_pools"] == 2 and new["prior_pulled"] == 1 and new["pull_age_min"] == 10.0
    assert new["history"][1][:2] == [T0 - 3600, T0 - 3000] and new["history"][0][1] is None
    assert tuple(st.query("SELECT by, sol_out FROM watch_pulls WHERE pool='OLD1'")[0]) == ("OP", 110.0)
    assert e.status()["pulls_found"] == 1
    # a fee withdrawal is not a liquidity pull
    assert find_withdraw([_optx(1, "Y", ["Program log: Instruction: WithdrawFees"], "P", (5.0, 4.9))], "P") is None


def test_creators_of_pumpswap_pools_are_recorded_and_judge_new_wallets():
    from tradingbot.model_book import operator_verdict
    st = WatchStore(":memory:")
    for p, created in (("OLD", T0 - 3600), ("NEW", T0), ("SOLO", T0), ("CURVE", T0)):
        st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts) VALUES(?,?,?,?,?,?)",
                   (p, p + "Mint", "pump-fun" if p == "CURVE" else "pumpswap", p, created, created))
    e = make_enricher(st)
    asked = []
    e.helius.pool_creators = lambda pools: asked.extend(pools) or {"OLD": "OP", "NEW": "OP", "SOLO": "FRESH"}
    assert e.poll_creators(T0 + 60) == 3
    assert sorted(asked) == ["NEW", "OLD", "SOLO"]  # only pumpswap pools
    assert e.poll_creators(T0 + 80) == 0  # nothing left to read
    pool = lambda p: dict(st.query("SELECT * FROM watch_pools WHERE pool=?", (p,))[0])
    assert operator_verdict(st, pool("NEW")) == {"verdict": "repeat operator", "creator": "OP", "prior_pools": 1}
    assert operator_verdict(st, pool("OLD"))["verdict"] == "new wallet"  # its later pool doesn't count
    assert operator_verdict(st, pool("SOLO"))["verdict"] == "new wallet"
    assert operator_verdict(st, pool("CURVE")) == {"verdict": "unknown", "reason": "creator not read yet"}
    assert operator_verdict(None, pool("NEW")) == {"verdict": "unknown"}
    assert e.status()["creators_known"] == 3
