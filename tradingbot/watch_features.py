"""Pump factors from the enrichment data (enrich.py), each as of a snapshot time `ts`.

The rule for every feature here: only information that existed at `ts`. Enrichment fetched
after `ts` is ignored; trades, posts, list placements and Telegram mentions after `ts` are
cut off; another token's pump counts (smart money, theme heat, market heat) only once it
had already happened by `ts`. "Pumped by then" means a snapshot price at or above 2x the
pool's first-seen price, which the watcher recorded live.
"""

from __future__ import annotations

import bisect
import json
import math
import re
from collections import defaultdict

from tradingbot.watcher import WatchStore

HOUR = 3600.0
KOL_FOLLOWERS = 50_000
NARRATIVE_CATEGORIES = (
    "animal", "ai_tech", "political", "celebrity_person", "news_event", "tweet_moment", "meme_reference",
    "brand_impersonation", "crypto_meta", "culture_entertainment", "joke_nsfw", "other")
STOPWORDS = {"sol", "the", "coin", "token", "official", "and", "for", "inu", "pump", "fun", "usd", "usdc"}
WORD_RE = re.compile(r"[a-z0-9]+")


def _log1p(x) -> float | None:
    try:
        return math.log1p(max(float(x), 0.0))
    except (TypeError, ValueError):
        return None


def words(name: str) -> set[str]:
    base = (name or "").split(" / ")[0].lower()
    return {w for w in WORD_RE.findall(base) if len(w) >= 3 and w not in STOPWORDS}


def _gini(values: list[float]) -> float | None:
    v = sorted(x for x in values if x > 0)
    if len(v) < 2:
        return None
    n, cum = len(v), 0.0
    for i, x in enumerate(v, 1):
        cum += i * x
    return (2 * cum) / (n * sum(v)) - (n + 1) / n


class Context:
    def __init__(self, store: WatchStore, cfg=None):
        pools = [dict(r) for r in store.query("SELECT pool, mint, name, created_ts, first FROM watch_pools")]
        self.created = {p["pool"]: p["created_ts"] for p in pools}
        self.enrich: dict[str, dict[str, tuple[float, dict]]] = defaultdict(dict)
        for r in store.query("SELECT pool, kind, ts, data FROM watch_enrich"):
            self.enrich[r["pool"]][r["kind"]] = (r["ts"], json.loads(r["data"]) if r["data"] else {})

        # When each token first traded at 2x its first-seen price, from live snapshots.
        first_price = {}
        for p in pools:
            try:
                first_price[p["pool"]] = float(json.loads(p["first"] or "{}").get("price") or 0)
            except (TypeError, ValueError):
                first_price[p["pool"]] = 0.0
        mint_of = {p["pool"]: p["mint"] for p in pools}
        self.t2x: dict[str, float] = {}
        for r in store.query("SELECT pool, ts, data FROM watch_snaps WHERE data IS NOT NULL AND data != '{}' "
                             "ORDER BY ts"):
            fp = first_price.get(r["pool"]) or 0
            price = json.loads(r["data"]).get("price") or 0
            m = mint_of.get(r["pool"])
            if fp > 0 and price >= 2 * fp and m and m not in self.t2x:
                self.t2x[m] = r["ts"]
        self.t2x_sorted = sorted(self.t2x.values())
        self.created_sorted = sorted(self.created.values())

        # Theme heat: launches and pumps among tokens sharing a name word.
        self._words_of_mint: dict[str, set[str]] = defaultdict(set)
        for p in pools:
            self._words_of_mint[p["mint"]] |= words(p["name"])
        self.word_created: dict[str, list[float]] = defaultdict(list)
        self.word_pumps: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for p in pools:
            for w in words(p["name"]):
                self.word_created[w].append(p["created_ts"])
                if p["mint"] in self.t2x:
                    self.word_pumps[w].append((self.t2x[p["mint"]], p["mint"]))
        for v in self.word_created.values():
            v.sort()
        for v in self.word_pumps.values():
            v.sort()
        self.word_pump_ts = {w: [t for t, _ in v] for w, v in self.word_pumps.items()}
        self.t2x_by_time = sorted((t, m) for m, t in self.t2x.items())

        # Attention lists and Telegram mentions per token, in time order.
        self.lists: dict[str, list[tuple[float, str, int, float]]] = defaultdict(list)
        for r in store.query("SELECT ts, list, rank, mint, amount FROM watch_lists ORDER BY ts"):
            self.lists[r["mint"]].append((r["ts"], r["list"], r["rank"], r["amount"] or 0.0))
        self.mentions: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for r in store.query("SELECT mint, channel, post_ts FROM watch_mentions WHERE post_ts IS NOT NULL"):
            self.mentions[r["mint"]].append((r["post_ts"], r["channel"]))
        self.market = [(r["ts"], r["sol_usd"], r["new_pools"] or 0, r["new_launchpad"] or 0)
                       for r in store.query("SELECT * FROM watch_market ORDER BY ts")]
        self.market_ts = [m[0] for m in self.market]

        # Images seen on other tokens, and which wallets bought early into which tokens.
        self.image_seen: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self.early_wallets: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for pool, kinds in self.enrich.items():
            mint, created = mint_of.get(pool), self.created.get(pool, 0)
            img = kinds.get("image", (0, {}))[1].get("dhash")
            if img:
                self.image_seen[img].append((created, mint))
            hel = kinds.get("helius", (0, {}))[1]
            if hel.get("copied_from"):
                continue  # the same token's launch pool already counted these wallets
            accounts = set(hel.get("pool_accounts") or [])
            for w in {t[2] for t in hel.get("trades") or [] if t[3] > 0 and t[2] not in accounts}:
                self.early_wallets[w].append((created, mint))
        for v in self.early_wallets.values():
            v.sort()
        from tradingbot.watch_features_more import More  # the long tail of factors (FACTORS.md)
        self.more = More(self, store, cfg)

    # --- helpers -------------------------------------------------------------------
    def _known(self, pool: str, kind: str, ts: float) -> dict | None:
        got = self.enrich.get(pool, {}).get(kind)
        if not got or got[0] > ts or not got[1] or "skipped" in got[1]:
            return None
        return got[1]

    def selection(self, pool: str, kind: str, ts: float) -> str:
        """How a paid lookup was chosen, for judging the paid data on an unbiased slice:
        all / ranked / control / legacy (before rationing) / below_rank / skipped / later / none."""
        got = self.enrich.get(pool, {}).get(kind)
        if not got:
            return "none"
        if got[0] > ts:
            return "later"
        data = got[1]
        if "skipped" in data:
            return "below_rank" if data["skipped"] == "below rank threshold" else "skipped"
        return data.get("selection") or ("legacy" if data else "none")

    def pumps_between(self, word: str, lo: float, hi: float, exclude_mint: str) -> int:
        """Tokens sharing `word` that first hit 2x in (lo, hi], not counting `exclude_mint`."""
        n = self._count_between(self.word_pump_ts.get(word, []), lo, hi)
        own = self.t2x.get(exclude_mint)
        return n - (own is not None and lo < own <= hi and word in self._words_of_mint.get(exclude_mint, ()))

    def _count_between(self, sorted_vals: list[float], lo: float, hi: float) -> int:
        return bisect.bisect_right(sorted_vals, hi) - bisect.bisect_right(sorted_vals, lo)

    # --- all features for one pool at time ts ----------------------------------------------
    def features(self, pool: str, p: dict, ts: float, info_ok: bool, snap: dict | None = None) -> dict:
        out: dict = {}
        created, mint = p["created_ts"], p["mint"]
        out.update(self._pumpfun(pool, p, ts, info_ok))
        out.update(self._orders(pool, created, ts))
        out.update(self._narrative(pool, ts))
        out.update(self._image(pool, created, mint, ts))
        out.update(self._helius(pool, created, mint, ts))
        out.update(self._x(pool, created, ts))
        out.update(self._lists(mint, created, ts))
        out.update(self._telegram(mint, created, ts))
        out.update(self._market(ts))
        out.update(self._themes(p["name"], mint, created, ts))
        info = json.loads(p["info"]) if info_ok and p.get("info") else {}
        out.update(self.more.features(pool, p, ts, snap or {}, info))
        return out

    def _pumpfun(self, pool, p, ts, info_ok) -> dict:
        pf = self._known(pool, "pumpfun15", ts) or self._known(pool, "pumpfun5", ts)
        info = json.loads(p["info"]) if info_ok and p.get("info") else {}
        twitter = (pf or {}).get("twitter") or info.get("twitter_handle") or ""
        out = {
            "tw_post_link": float("/status/" in twitter), "tw_account": float(bool(twitter) and "/status/" not in twitter),
            "tw_none": float(not twitter),
            "website_domain_known": float(bool((pf or {}).get("website") or info.get("website_domains"))),
        }
        if not pf:
            return out
        mcap, ath = pf.get("usd_market_cap"), pf.get("ath_market_cap_usd")
        out.update({
            "pf_live": float(pf.get("is_live", False)), "pf_boosted": float((pf.get("boost_mode") or "NONE") != "NONE"),
            "pf_complete": float(pf.get("complete", False)),
            "pf_verified": float(pf.get("verified", False)), "pf_nsfw": float(pf.get("nsfw", False)),
            "pf_transfer_fee": float((pf.get("transfer_fee_bps") or 0) > 0),
            "pf_transfer_hook": float(pf.get("transfer_hook", False)),
            "pf_security_allow": float(pf.get("security_verdict") == "allow"),
            "pf_cashback": float(pf.get("is_cashback", False)), "pf_agent": float(pf.get("tokenized_agent", False)),
            "pf_mcap_vs_ath": (mcap / ath) if mcap and ath else None,
            "pf_log_vol_1h": _log1p(pf.get("volume_1h_usd")),
        })
        hist = self._known(pool, "pumpfun5", ts) or {}
        if "creator_launches" in hist:
            prior = hist.get("creator_recent_prior") or 0
            name = (p["name"] or "").split(" / ")[0].upper()
            out.update({
                "creator_log_launches": _log1p(hist.get("creator_launches")),
                "creator_grad_rate": (hist.get("creator_prior_graduated") or 0) / prior if prior else None,
                "creator_launches_24h": _log1p(hist.get("creator_launches_24h")),
                "creator_log_gap_min": _log1p(hist.get("creator_last_launch_gap_min")),
                "creator_first_launch": float((hist.get("creator_launches") or 0) <= 1),
                "creator_reused_name": float(any(n.upper() == name for n in hist.get("creator_prior_names") or [])),
            })
        return out

    def _orders(self, pool, created, ts) -> dict:
        od = self._known(pool, "orders", ts)
        if od is None:
            return {}
        paid = [o for o in od.get("orders") or [] if 0 < (o.get("paid_ts") or 0) <= ts]
        types = [o.get("type") for o in paid]
        first = min((o["paid_ts"] for o in paid), default=None)
        return {
            "paid_profile": float("tokenProfile" in types), "paid_boosts": float(types.count("boost")),
            "paid_ads": float(any(t in ("tokenAd", "trendingBarAd") for t in types)),
            "paid_cto": float("communityTakeover" in types),
            "paid_min_after_launch": (first - created) / 60 if first else None,
        }

    def _narrative(self, pool, ts) -> dict:
        n = self._known(pool, "narrative", ts)
        if n is None:
            return {}
        out = {f"nar_{c}": float(n.get("category") == c) for c in NARRATIVE_CATEGORIES}
        out.update({"nar_copycat": float(bool(n.get("copycat_of_existing"))),
                    "nar_post_launch": float(bool(n.get("references_specific_post"))),
                    "nar_real_person": float(bool(n.get("names_real_person"))),
                    "nar_originality": n.get("originality")})
        return out

    def _image(self, pool, created, mint, ts) -> dict:
        img = self._known(pool, "image", ts)
        if not img or not img.get("dhash"):
            return {}
        others = {m for c, m in self.image_seen[img["dhash"]] if c < created and m != mint}
        return {"image_reused": _log1p(len(others)), "image_ever_reused": float(bool(others))}

    def _helius(self, pool, created, mint, ts) -> dict:
        h = self._known(pool, "helius", ts)
        if h is None:
            return {}
        accounts = set(h.get("pool_accounts") or [])
        creator = h.get("creator")
        supply = h.get("supply") or 0
        tr = [t for t in h.get("trades") or [] if t[0] is not None and t[0] <= ts]
        out: dict = {"hel_n_txs": _log1p(h.get("n_txs"))}
        if tr and supply > 0:
            first_slot = min(t[1] for t in tr)
            first_ts = min(t[0] for t in tr)
            buys = [t for t in tr if t[3] > 0 and t[2] not in accounts]
            non_dev = [t for t in buys if t[2] != creator]
            by_owner: dict[str, float] = defaultdict(float)
            for t in non_dev:
                by_owner[t[2]] += t[3]
            sizes = [round(t[3], 0) for t in non_dev]
            out.update({
                "hel_bundle_share": sum(t[3] for t in non_dev if t[1] <= first_slot + 1) / supply,
                "hel_dev_initial_share": sum(t[3] for t in buys if t[2] == creator and t[1] == first_slot) / supply,
                "hel_sniper_share_10s": sum(t[3] for t in non_dev if t[0] - first_ts <= 10) / supply,
                "hel_early_buyers": _log1p(len(by_owner)),
                "hel_top_buyer_share": max(by_owner.values()) / supply if by_owner else None,
                "hel_buy_gini": _gini(list(by_owner.values())),
                "hel_equal_size_buys": (1 - len(set(sizes)) / len(sizes)) if len(sizes) >= 5 else None,
                "hel_dev_sold": float(any(t[2] == creator and t[3] < 0 for t in tr)),
                "hel_dev_sold_share": -sum(t[3] for t in tr if t[2] == creator and t[3] < 0) / supply,
            })
            # Smart money: early buyers here who were early in an EARLIER token that had pumped by ts.
            # Serial snipers: early in many earlier tokens.
            smart = snipers = 0
            for w in by_owner:
                hist = [(c, m) for c, m in self.early_wallets.get(w, []) if c < created and m != mint]
                smart += any(self.t2x.get(m, math.inf) <= ts for _, m in hist)
                snipers += len(hist) >= 5
            out.update({"hel_smart_wallets": float(smart), "hel_serial_snipers": float(snipers),
                        "hel_smart_share": smart / len(by_owner) if by_owner else None})
        holders = [(o, amt) for o, amt in h.get("holders") or [] if o not in accounts]
        if holders and supply > 0:
            amounts = sorted((a for _, a in holders), reverse=True)
            out.update({"hel_top1_share": amounts[0] / supply, "hel_top10_share": sum(amounts[:10]) / supply,
                        "hel_creator_holding": sum(a for o, a in holders if o == creator) / supply})
        wallets = [v for v in (h.get("wallets") or {}).values() if v and v.get("first_ts")]
        if wallets:
            ages = [max(created - w["first_ts"], 0) for w in wallets]
            funders = [w.get("funder") for w in wallets if w.get("funder")]
            top_funder = max((funders.count(f) for f in set(funders)), default=0)
            out.update({
                "hel_fresh_wallet_share": sum(a < 86400 for a in ages) / len(ages),
                "hel_log_median_wallet_age_h": _log1p(sorted(ages)[len(ages) // 2] / HOUR),
                "hel_same_funder_share": top_funder / len(wallets) if len(wallets) >= 2 else None,
                "hel_funded_by_creator": (sum(f == creator for f in funders) / len(wallets)) if creator else None,
            })
        return out

    def _x(self, pool, created, ts) -> dict:
        x = self._known(pool, "x", ts)
        if x is None:
            return {}
        posts = [q for q in x.get("posts") or [] if q.get("ts") and q["ts"] <= ts]
        followers = [q.get("followers") or 0 for q in posts]
        return {
            "x_count_7d": _log1p(x.get("count_7d")), "x_posts": _log1p(len(posts)),
            "x_authors": _log1p(len({q.get("author") for q in posts})),
            "x_log_max_followers": _log1p(max(followers, default=0)),
            "x_kol": float(any(f >= KOL_FOLLOWERS for f in followers)),
            "x_verified": float(any(q.get("verified") for q in posts)),
            "x_log_engagement": _log1p(sum((q.get("likes") or 0) + (q.get("reposts") or 0) for q in posts)),
            "x_new_account_share": (sum(1 for q in posts if q.get("author_created")
                                        and created - q["author_created"] < 30 * 86400) / len(posts)) if posts else None,
        }

    def _lists(self, mint, created, ts) -> dict:
        seen = [e for e in self.lists.get(mint, []) if e[0] <= ts]
        out = {}
        for key, prefix in (("gecko:trending_5m", "gt5m"), ("gecko:trending_1h", "gt1h"),
                            ("dex:token-boosts:top", "dexboost_top"), ("dex:token-boosts:latest", "dexboost_new"),
                            ("dex:token-profiles:latest", "dexprofile")):
            hits = [e for e in seen if e[1] == key]
            out[f"list_{prefix}"] = float(bool(hits))
            out[f"list_{prefix}_inv_rank"] = max((1 / e[2] for e in hits if e[2]), default=0.0)
        boosts = [e[3] for e in seen if e[1].startswith("dex:token-boosts")]
        out["list_boost_amount"] = _log1p(max(boosts, default=0))
        out["list_first_min_after_launch"] = ((seen[0][0] - created) / 60) if seen else None
        return out

    def _telegram(self, mint, created, ts) -> dict:
        m = [e for e in self.mentions.get(mint, []) if e[0] <= ts and e[1] != "biz"]  # biz = 4chan, counted apart
        if not self.mentions and not m:
            return {}
        return {"tg_mentions": _log1p(len(m)), "tg_channels": float(len({c for _, c in m})),
                "tg_first_min_after_launch": ((min(e[0] for e in m) - created) / 60) if m else None}

    def _market(self, ts) -> dict:
        if not self.market:
            return {}
        i = bisect.bisect_right(self.market_ts, ts) - 1
        if i < 0:
            return {}
        j = bisect.bisect_right(self.market_ts, ts - HOUR) - 1
        k = bisect.bisect_right(self.market_ts, ts - 24 * HOUR) - 1
        sol_now = self.market[i][1]
        sol_1h = self.market[j][1] if j >= 0 else None
        sol_24h = self.market[k][1] if k >= 0 else None
        hour_rows = self.market[j + 1:i + 1]
        launched_6h = self._count_between(self.created_sorted, ts - 6 * HOUR, ts)
        pumped_6h = self._count_between(self.t2x_sorted, ts - 6 * HOUR, ts)
        return {
            "mkt_sol_chg_1h": (sol_now / sol_1h - 1) if sol_now and sol_1h else None,
            "mkt_sol_chg_24h": (sol_now / sol_24h - 1) if sol_now and sol_24h else None,
            "mkt_new_pools_1h": _log1p(sum(r[2] for r in hour_rows)),
            "mkt_launchpad_share_1h": (sum(r[3] for r in hour_rows) / sum(r[2] for r in hour_rows))
            if sum(r[2] for r in hour_rows) else None,
            "mkt_recent_pump_rate_6h": pumped_6h / launched_6h if launched_6h else None,
        }

    def _themes(self, name, mint, created, ts) -> dict:
        ws = words(name)
        if not ws:
            return {"theme_launches_1h": 0.0, "theme_pumped_6h": 0.0}
        launches = max(self._count_between(self.word_created[w], ts - HOUR, ts) for w in ws)
        launches -= created > ts - HOUR  # not counting this token itself
        pumped = max(self.pumps_between(w, ts - 6 * HOUR, ts, mint) for w in ws)
        return {"theme_launches_1h": _log1p(max(launches, 0)), "theme_pumped_6h": _log1p(pumped)}
