"""The long tail of pump factors (FACTORS.md numbers in comments), as of snapshot time `ts`.

Same rule as watch_features.py: only data stamped at or before `ts`. Built on the data a
`Context` already loaded, plus snapshots, market-wide series and the watched X feed.
"""

from __future__ import annotations

import bisect
import json
import math
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timezone
from statistics import median

from tradingbot.watch_features import HOUR, _gini, _log1p, words

DAY = 24 * HOUR
TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
MCAP_LEVELS = (10_000, 30_000, 69_000, 100_000, 250_000, 1_000_000)
AMM_DEXES = {"pumpswap", "raydium", "meteora-damm-v2", "meteora", "orca", "raydium-cp", "raydium-clmm"}
# Holidays that move retail flow (fixed list; extend yearly).
US_HOLIDAYS = {date(2026, 11, 26), date(2026, 12, 25), date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
               date(2027, 5, 31), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24)}
ASIA_HOLIDAYS = {date(2026, 10, d) for d in range(1, 8)} | {date(2027, 2, d) for d in range(6, 13)}
# Mercury retrograde windows (approximate, from public almanacs): a sentiment marker, almost surely noise.
MERCURY_RETROGRADE = [(date(2026, 10, 24), date(2026, 11, 13)), (date(2027, 2, 9), date(2027, 3, 3)),
                      (date(2027, 6, 10), date(2027, 7, 4)), (date(2027, 10, 7), date(2027, 10, 28))]
KNOWN_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc).timestamp()
SYNODIC_MONTH = 29.530588 * DAY


def moon_phase(ts: float) -> float:
    """0 = new moon, 0.5 = full moon."""
    return ((ts - KNOWN_NEW_MOON) % SYNODIC_MONTH) / SYNODIC_MONTH


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _symbol(name: str) -> str:
    return ((name or "").split(" / ")[0].split() or [""])[0]


def _quote(name: str) -> str:
    return (name or "").split(" / ")[-1].strip().upper() if " / " in (name or "") else ""


class More:
    strict = False  # tests set this so a failing factor group raises instead of being skipped

    def __init__(self, ctx, store, cfg=None):
        self.ctx, self.store = ctx, store
        self.cfg = cfg
        lists = lambda v: {x.strip() for x in (v or "").split(",") if x.strip()}  # noqa: E731
        self.kol = lists(getattr(cfg, "kol_wallets", ""))
        self.cex = lists(getattr(cfg, "cex_wallets", ""))
        self.paid_callers = {x.lower().lstrip("@") for x in lists(getattr(cfg, "x_paid_callers", ""))}

        rows = [dict(r) for r in store.query("SELECT pool, mint, dex, name, created_ts, first FROM watch_pools")]
        self.pool = {r["pool"]: r for r in rows}
        self.pools_by_mint: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            self.pools_by_mint[r["mint"]].append(r)
        self.first_price = {r["pool"]: _f(json.loads(r["first"] or "{}").get("price")) for r in rows}
        self.snaps: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
        for r in store.query("SELECT pool, age_min, ts, data FROM watch_snaps WHERE data IS NOT NULL AND data != '{}' "
                             "ORDER BY ts"):
            self.snaps[r["pool"]].append((r["ts"], r["age_min"], json.loads(r["data"])))

        # When each token first fell to 10% of its first price (a rug) and first reached 5x.
        self.trug, self.t5x = {}, {}
        for pool, ss in self.snaps.items():
            fp, mint = self.first_price.get(pool), self.pool[pool]["mint"]
            for ts, _a, d in ss:
                p = _f(d.get("price")) or 0
                if fp and p and p <= 0.1 * fp and mint not in self.trug:
                    self.trug[mint] = ts
                if fp and p >= 5 * fp and mint not in self.t5x:
                    self.t5x[mint] = ts
        self.t5x_sorted = sorted(self.t5x.values())

        # Price change since first seen at each age, for "vs the typical new token at the same age".
        self.by_age: dict[int, list[tuple[float, float, float]]] = defaultdict(list)  # age -> (ts, log chg, log vol)
        for pool, ss in self.snaps.items():
            fp = self.first_price.get(pool)
            for ts, age, d in ss:
                p = _f(d.get("price"))
                if fp and p:
                    self.by_age[age].append((ts, math.log(p / fp), _log1p(d.get("vol_m5")) or 0.0))
        for v in self.by_age.values():
            v.sort()
        self.by_age_ts = {a: [x[0] for x in v] for a, v in self.by_age.items()}

        # AMM (post-graduation) pool creation times per mint, and in time order overall.
        self.amm_created = sorted(r["created_ts"] for r in rows if r["dex"] in AMM_DEXES)
        # Same ticker: creation times of distinct tokens, and when any of them first hit 2x.
        sym_mints: dict[str, dict[str, float]] = defaultdict(dict)
        for r in rows:
            m = sym_mints[_symbol(r["name"]).upper()]
            m[r["mint"]] = min(m.get(r["mint"], math.inf), r["created_ts"])
        self.symbol_created = {s: sorted(v.values()) for s, v in sym_mints.items()}
        self.symbol_pumps = {s: sorted(ctx.t2x[m] for m in v if m in ctx.t2x) for s, v in sym_mints.items()}
        self.word_rug_ts: dict[str, list[float]] = defaultdict(list)
        seen_rug: set[tuple[str, str]] = set()
        for r in rows:
            if r["mint"] in self.trug:
                for w in words(r["name"]):
                    if (w, r["mint"]) not in seen_rug:
                        seen_rug.add((w, r["mint"]))
                        self.word_rug_ts[w].append(self.trug[r["mint"]])
        for v in self.word_rug_ts.values():
            v.sort()
        # Narrative categories over time: creation times per category, and pump times per category.
        self.cat_created: dict[str, list[float]] = defaultdict(list)
        self.cat_pumps: dict[str, list[float]] = defaultdict(list)
        self.all_cat_created: list[float] = []

        # Market-wide series.
        self.macro: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in store.query("SELECT ts, key, value FROM watch_macro ORDER BY ts"):
            self.macro[r["key"]].append((r["ts"], r["value"]))
        self.macro_ts = {k: [x[0] for x in v] for k, v in self.macro.items()}
        self.xfeed = [(r["post_ts"] or r["ts"], (r["text"] or "").lower())
                      for r in store.query("SELECT ts, post_ts, text FROM watch_xfeed")]
        self.perps = {s.upper() for s in json.loads(store.get_state("perp_symbols") or "[]")}

        # Narrative category per pool (for category heat) and calling-channel history (for track records).
        self.category = {p: k["narrative"][1].get("category") for p, k in ctx.enrich.items()
                         if "narrative" in k and k["narrative"][1].get("category") and p in self.pool}
        for q, cat in self.category.items():
            created, mint = self.pool[q]["created_ts"], self.pool[q]["mint"]
            self.cat_created[cat].append(created)
            self.all_cat_created.append(created)
            if mint in ctx.t2x:
                self.cat_pumps[cat].append(ctx.t2x[mint])
        for v in list(self.cat_created.values()) + list(self.cat_pumps.values()) + [self.all_cat_created]:
            v.sort()
        self.channel_calls: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for r in store.query("SELECT channel, mint, post_ts FROM watch_mentions WHERE post_ts IS NOT NULL"):
            self.channel_calls[r["channel"]].append((r["post_ts"], r["mint"]))
        self.creator_of = {p: k["pumpfun5"][1].get("creator") for p, k in ctx.enrich.items()
                           if "pumpfun5" in k and k["pumpfun5"][1].get("creator")}
        self.pools_by_creator: dict[str, list[str]] = defaultdict(list)
        for p, c in self.creator_of.items():
            self.pools_by_creator[c].append(p)

    # --- helpers -------------------------------------------------------------------------
    def _known(self, pool, kind, ts):
        return self.ctx._known(pool, kind, ts)

    def _series_at(self, key: str, ts: float, back: float = 0.0) -> float | None:
        xs = self.macro_ts.get(key)
        if not xs:
            return None
        i = bisect.bisect_right(xs, ts - back) - 1
        return self.macro[key][i][1] if i >= 0 else None

    def _chg(self, key: str, ts: float, back: float) -> float | None:
        now, then = self._series_at(key, ts), self._series_at(key, ts, back)
        return (now / then - 1) if now and then else None

    # --- everything --------------------------------------------------------------------------
    def features(self, pool: str, p: dict, ts: float, snap: dict, info: dict) -> dict:
        out: dict = {}
        for fn in (self._price_path, self._candles, self._flow, self._liquidity, self._curve, self._holders,
                   self._wallets, self._developer, self._security, self._identity, self._narrative,
                   self._social, self._attention, self._market, self._cross, self._calendar,
                   self._later, self._manipulation):
            try:
                out.update(fn(pool, p, ts, snap, info))
            except Exception:  # noqa: BLE001 (one malformed record must not drop the whole row)
                if self.strict:
                    raise
                continue
        return out

    # A. price path (5, 6, 10, 13-17, 19-25)
    def _price_path(self, pool, p, ts, snap, info):
        fp = self.first_price.get(pool)
        path = [(p["created_ts"], fp)] + [(t, _f(d.get("price"))) for t, _a, d in self.snaps.get(pool, []) if t <= ts]
        path = [(t, x) for t, x in path if x]
        if len(path) < 2 or not fp:
            return {}
        prices = [x for _, x in path]
        now = prices[-1]
        hi_i = max(range(len(prices)), key=prices.__getitem__)
        lo = min(prices)
        ups = sum(b > a for a, b in zip(prices, prices[1:]))
        doubles = sum(b >= 2 * a for a, b in zip(prices, prices[1:]))
        halves = sum(b <= a / 2 for a, b in zip(prices, prices[1:]))
        fdv = _f(snap.get("fdv")) or 0
        nxt = next((lv for lv in MCAP_LEVELS if lv > fdv), None)
        mint = p["mint"]
        out = {
            "pp_drawdown_from_high": now / max(prices) - 1,                                    # 5
            "pp_up_steps_share": ups / (len(prices) - 1),                                     # 6
            "pp_min_since_high": (ts - path[hi_i][0]) / 60,                                   # 10
            "pp_mcap_levels_crossed": float(sum(fdv >= lv for lv in MCAP_LEVELS)),            # 13
            "pp_to_next_level": (nxt / fdv - 1) if nxt and fdv else None,
            "pp_min_to_2x": ((self.ctx.t2x[mint] - p["created_ts"]) / 60
                             if self.ctx.t2x.get(mint, math.inf) <= ts else None),            # 15
            "pp_doubles": float(doubles), "pp_halves": float(halves),                         # 16
            "pp_recovery_from_low": now / lo - 1 if lo else None,                             # 17
            "pp_failed_pump": float(max(prices) >= 2 * fp and now <= 0.5 * max(prices)),      # 25
        }
        chg_m5, chg_h1 = _f(snap.get("chg_m5")), _f(snap.get("chg_h1"))
        out["pp_m5_vs_h1"] = (chg_m5 / chg_h1) if chg_m5 is not None and chg_h1 else None   # 22
        vol_m5, vol_h1 = _f(snap.get("vol_m5")), _f(snap.get("vol_h1"))
        out["pp_late_surge"] = (vol_m5 / vol_h1) if vol_m5 is not None and vol_h1 else None  # 23, 43
        jup = self._known(pool, "jupiter", ts)
        if jup and jup.get("usd_price") and now:
            out["pp_jup_vs_snapshot_price"] = jup["usd_price"] / now - 1                     # 19
        # 20, 44: vs the typical new token at the same age in the last hour
        age = round((ts - p["created_ts"]) / 60)
        near = min(self.by_age, key=lambda a: abs(a - age)) if self.by_age else None
        if near is not None:
            xs = self.by_age_ts[near]
            window = self.by_age[near][bisect.bisect_left(xs, ts - HOUR):bisect.bisect_right(xs, ts)]
            if len(window) >= 5:
                out["pp_vs_median_token"] = math.log(now / fp) - median(w[1] for w in window)
                out["pp_vol_vs_median_token"] = (_log1p(vol_m5) or 0) - median(w[2] for w in window)
        sol_1h = self.ctx._market(ts).get("mkt_sol_chg_1h")
        out["pp_vs_sol"] = (math.log(now / fp) - math.log(1 + sol_1h)) if sol_1h is not None else None  # 21
        # 24: the graduated pool's first minutes
        for q in self.pools_by_mint.get(mint, []):
            if q["dex"] in AMM_DEXES and q["pool"] != pool:
                ss = [(t, _f(d.get("price"))) for t, _a, d in self.snaps.get(q["pool"], []) if t <= ts]
                qfp = self.first_price.get(q["pool"])
                if ss and qfp and ss[0][1]:
                    out["pp_grad_pool_first_chg"] = ss[0][1] / qfp - 1
        return out

    # A. 7-9, 11, 18 from 1-minute candles fetched at ~15 minutes, only candles closed by ts
    def _candles(self, pool, p, ts, snap, info):
        c = self._known(pool, "candles15", ts)
        rows = [r for r in (c or {}).get("candles") or [] if r[0] + 60 <= ts]
        if len(rows) < 3:
            return {}
        closes = [r[4] for r in rows]
        rets = [b / a - 1 for a, b in zip(closes, closes[1:]) if a]
        run = best = 0
        for r in rows:
            run = run + 1 if r[4] >= r[1] else 0
            best = max(best, run)
        vol = sum(r[5] for r in rows)
        vwap = sum(r[4] * r[5] for r in rows) / vol if vol else None
        wicks = [(r[2] - max(r[1], r[4])) / (r[2] - r[3]) for r in rows if r[2] > r[3]]
        return {
            "c1_volatility": (sum(x * x for x in rets) / len(rets)) ** 0.5 if rets else None,   # 7
            "c1_green_run": float(best),                                                       # 8
            "c1_worst_minute": min(rets) if rets else None,                                     # 9
            "c1_above_vwap": float(closes[-1] > vwap) if vwap else None,                        # 11
            "c1_upper_wick": sum(wicks) / len(wicks) if wicks else None,                         # 18
        }

    def _trades(self, pool, ts):
        h = self._known(pool, "helius", ts)
        if not h:
            return None, [], set(), 0, None
        acc = set(h.get("pool_accounts") or [])
        tr = [t for t in h.get("trades") or [] if t[0] is not None and t[0] <= ts]
        return h, tr, acc, h.get("supply") or 0, h.get("creator")

    # B. 35-42, 45 + Jupiter's organic flow (241, 298)
    def _flow(self, pool, p, ts, snap, info):
        out = {}
        jup = self._known(pool, "jupiter", ts)
        if jup:
            for w in ("stats5m", "stats1h"):
                s = jup.get(w) or {}
                bv, ov = _f(s.get("buyVolume")), _f(s.get("buyOrganicVolume"))
                out[f"jup_{w}_organic_buy_share"] = (ov / bv) if bv and ov is not None else None
                out[f"jup_{w}_net_buyers"] = _f(s.get("numNetBuyers"))
                out[f"jup_{w}_organic_buyers"] = _log1p(s.get("numOrganicBuyers"))
                out[f"jup_{w}_holder_change"] = _f(s.get("holderChange"))                       # 81
            out["jup_organic_score"] = _f(jup.get("organic_score"))
        h, tr, acc, supply, creator = self._trades(pool, ts)
        if not tr or supply <= 0:
            return out
        buys = [t for t in tr if t[3] > 0 and t[2] not in acc]
        sells = [t for t in tr if t[3] < 0 and t[2] not in acc]
        sizes = sorted(t[3] for t in buys)
        by_owner = defaultdict(float)
        for t in buys:
            by_owner[t[2]] += t[3]
        top5 = sum(sorted(by_owner.values(), reverse=True)[:5])
        first = min(t[0] for t in tr)
        span = max(t[0] for t in tr) - first
        half = first + span / 2
        times = sorted(t[0] for t in tr)
        gaps = [b - a for a, b in zip(times, times[1:])]
        buy_count = defaultdict(int)
        for t in buys:
            buy_count[t[2]] += 1
        sellers = {t[2] for t in sells}
        out.update({
            "fl_median_buy_share": (sizes[len(sizes) // 2] / supply) if sizes else None,         # 35
            "fl_largest_buy_share": (sizes[-1] / supply) if sizes else None,                     # 36
            "fl_top5_buy_share": top5 / sum(by_owner.values()) if by_owner else None,            # 37
            "fl_net_buy_share": (sum(t[3] for t in buys) + sum(t[3] for t in sells)) / supply,   # 38
            "fl_buy_size_skew": (sizes[int(len(sizes) * 0.9)] / sizes[len(sizes) // 2])
            if len(sizes) >= 5 and sizes[len(sizes) // 2] else None,                             # 39
            "fl_buyer_trend": ((len({t[2] for t in buys if t[0] > half}) + 1)
                               / (len({t[2] for t in buys if t[0] <= half}) + 1)) if span else None,  # 40
            "fl_repeat_buyer_share": (sum(n > 1 for n in buy_count.values()) / len(buy_count)) if buy_count else None,  # 41
            "fl_early_seller_share": (len(sellers & set(by_owner)) / len(by_owner)) if by_owner else None,  # 42
            "fl_median_gap_s": median(gaps) if gaps else None,                                    # 45
        })
        return out

    # C. 50-57, 60
    def _liquidity(self, pool, p, ts, snap, info):
        mint, quote = p["mint"], _quote(p["name"])
        siblings = [q for q in self.pools_by_mint.get(mint, []) if q["created_ts"] <= ts]
        out = {"lq_pools": float(len(siblings)),                                                 # 50
               "lq_quote_sol": float(quote == "SOL"), "lq_quote_usd": float(quote in ("USDC", "USDT")),
               "lq_quote_other": float(quote not in ("SOL", "USDC", "USDT", ""))}               # 51, 271
        rc = self._known(pool, "rugcheck", ts)
        if rc:
            out["lq_lp_locked_pct"] = _f(rc.get("lp_locked_pct"))                               # 52
        jup = self._known(pool, "jupiter", ts)
        if jup:
            out["lq_liquidity_change_1h"] = _f((jup.get("stats1h") or {}).get("liquidityChange"))  # 53
            out["lq_jup_verified"] = float(bool(jup.get("verified")))                           # 57
        q = self._known(pool, "jupquote", ts)
        if q:
            out.update({"lq_impact_500": _f(q.get("impact_pct_500")),                          # 54
                        "lq_round_trip_loss_100": _f(q.get("round_trip_loss_pct_100")),         # 55
                        "lq_routable": float(bool(q.get("routable")))})                        # 56
        for s in siblings:                                                                       # 60
            if s["dex"] in AMM_DEXES:
                hit = next((t for t, _a, d in self.snaps.get(s["pool"], []) if t <= ts and (_f(d.get("vol_h24")) or 0) >= 10_000), None)
                if hit:
                    out["lq_min_grad_to_10k"] = (hit - s["created_ts"]) / 60
        return out

    # D. 63, 64
    def _curve(self, pool, p, ts, snap, info):
        out = {}
        amm = [q for q in self.pools_by_mint.get(p["mint"], []) if q["dex"] in AMM_DEXES and q["created_ts"] <= ts]
        launch = [q for q in self.pools_by_mint.get(p["mint"], []) if q["dex"] not in AMM_DEXES]
        if amm and launch:
            out["cv_min_to_graduate"] = (min(q["created_ts"] for q in amm) - min(q["created_ts"] for q in launch)) / 60
        gp = _f(info.get("graduation_pct")) if info else None
        age_min = (ts - p["created_ts"]) / 60
        if gp is not None and age_min > 0:
            out["cv_progress_per_min"] = gp / age_min                                            # 64
            out["cv_remaining"] = max(0.0, 100 - gp) / 100                                       # 14
        return out

    # E. 82-93
    def _holders(self, pool, p, ts, snap, info):
        h, tr, acc, supply, creator = self._trades(pool, ts)
        out = {}
        if info and isinstance(info.get("holder_distribution"), dict):                          # 91
            for k, v in info["holder_distribution"].items():
                fv = _f(str(v).rstrip("%"))
                if fv is not None:
                    out[f"ho_gt_{k}"[:40]] = fv
        rc = self._known(pool, "rugcheck", ts)
        if rc:
            out["ho_insiders_flag"] = float(any("insider" in (r or "").lower() for r in rc.get("risks") or []))  # 92
        if not h or supply <= 0:
            return out
        price = _f(snap.get("price")) or 0
        holders = [(o, a) for o, a in h.get("holders") or [] if o and o not in acc]
        amounts = [a for _, a in holders]
        out["ho_top20_gini"] = _gini(amounts)                                                     # 83
        out["ho_over_1k_usd"] = float(sum(a * price >= 1000 for a in amounts))                    # 82
        net = defaultdict(float)
        for t in tr:
            if t[2] not in acc:
                net[t[2]] += t[3]
        bought = {t[2] for t in tr if t[3] > 0 and t[2] not in acc}
        if bought:
            out["ho_still_holding_share"] = sum(net[o] > 0 for o in bought) / len(bought)        # 84
            out["ho_fully_exited_share"] = sum(net[o] <= 0 for o in bought) / len(bought)        # 93
        wallets = h.get("wallets") or {}
        aged = [(o, a) for o, a in holders if (wallets.get(o) or {}).get("first_ts")]
        if aged:
            fresh = sum(a for o, a in aged if p["created_ts"] - wallets[o]["first_ts"] < DAY)
            out["ho_fresh_wallet_holding"] = fresh / supply                                       # 85
        if self.cex:
            out["ho_cex_share"] = sum(a for o, a in holders if o in self.cex) / supply            # 86
        sol = h.get("sol_balances") or {}
        top_sol = [sol[o] for o, _ in holders if o in sol]
        if top_sol:
            out["ho_log_median_sol"] = _log1p(sorted(top_sol)[len(top_sol) // 2])                # 88
        funders = [(wallets.get(o) or {}).get("funder") for o, _ in holders]
        funders = [f for f in funders if f]
        if len(funders) >= 2:
            out["ho_clustered_share"] = max(funders.count(f) for f in set(funders)) / len(funders)  # 89
        sent = [t for t in tr if t[3] > 0 and t[2] not in acc and t[4] and t[4] != t[2]]
        out["ho_airdropped_share"] = len({t[2] for t in sent}) / max(len(bought), 1)              # 90, 300
        # 87: early buyers who were also early in the most recent token that pumped
        i = bisect.bisect_right(self.ctx.t2x_by_time, (ts, "\uffff")) - 1
        while i >= 0 and self.ctx.t2x_by_time[i][1] == p["mint"]:
            i -= 1
        if i >= 0:
            last = self.ctx.t2x_by_time[i][1]
            overlap = sum(any(m == last for _, m in self.ctx.early_wallets.get(o, [])) for o in bought)
            out["ho_overlap_last_winner"] = float(overlap)
        return out

    # F. 106, 108-110, 112-115
    def _wallets(self, pool, p, ts, snap, info):
        h, tr, acc, supply, creator = self._trades(pool, ts)
        if not tr:
            return {}
        buys = [t for t in tr if t[3] > 0 and t[2] not in acc]
        buyers = {t[2] for t in buys}
        wins = 0
        for w in buyers:
            wins += sum(1 for c, m in self.ctx.early_wallets.get(w, []) if c < p["created_ts"] and m != p["mint"]
                        and self.ctx.t2x.get(m, math.inf) <= ts)
        first_buy = {}
        for t in sorted(buys):
            first_buy.setdefault(t[2], t[0])
        dumped = {t[2] for t in tr if t[3] < 0 and t[2] in first_buy and t[0] - first_buy[t[2]] <= 300}
        tipped = [t for t in buys if len(t) > 6 and (t[6] or 0) > 0]
        fees = [(t[5] or 0) - 5000 for t in buys if len(t) > 5 and t[5]]
        slots = defaultdict(lambda: [0, 0])
        for t in tr:
            if t[2] not in acc:
                slots[(t[1], t[2])][0 if t[3] > 0 else 1] += 1
        wallets = (h or {}).get("wallets") or {}
        out = {
            "wl_smart_wins": _log1p(wins),                                                      # 106
            "wl_dump_within_5m_share": len(dumped) / len(buyers) if buyers else None,           # 110
            "wl_jito_tipped_share": len(tipped) / len(buys) if buys else None,                  # 112
            "wl_log_median_tip_lamports": _log1p(median(t[6] for t in tipped)) if tipped else 0.0,
            "wl_log_median_priority_fee": _log1p(median(fees)) if fees else None,               # 113
            "wl_sandwich_wallets": float(sum(1 for (_s, _w), (b, s) in slots.items() if b and s)),  # 114
        }
        if self.kol:
            out["wl_kol_buyers"] = float(len(buyers & self.kol))                               # 108, 109
        if self.cex:
            out["wl_cex_funded_share"] = (sum(1 for w in wallets.values() if (w or {}).get("funder") in self.cex)
                                          / len(wallets)) if wallets else None                  # 115
        return out

    # G. 124, 126-130, 132, 133
    def _developer(self, pool, p, ts, snap, info):
        out = {}
        dev = self._known(pool, "dev", ts)
        creator = self.creator_of.get(pool)
        if dev:
            prof = dev.get("profile") or {}
            out.update({
                "dv_prior_best_ath": _log1p(dev.get("prior_best_ath_usd")),                     # 124
                "dv_has_username": float(bool(prof.get("username"))),                           # 129
                "dv_log_followers": _log1p(prof.get("followers")),                              # 130
                "dv_x_linked": float(bool(prof.get("x_username"))),
                "dv_public_figure": float((prof.get("followers") or 0) >= 10_000 and bool(prof.get("x_username"))),  # 132
                "dv_badges": float(prof.get("badges") or 0),
            })
            origin = dev.get("origin") or {}
            if origin.get("first_ts"):
                out["dv_wallet_age_days"] = max(p["created_ts"] - origin["first_ts"], 0) / DAY     # 126
            if self.cex and origin.get("funder"):
                out["dv_cex_funded"] = float(origin["funder"] in self.cex)
        if creator:                                                                              # 127, 133
            earlier = [q for q in self.pools_by_creator.get(creator, []) if self.pool[q]["created_ts"] < p["created_ts"]]
            out["dv_prior_rugs_seen"] = float(sum(self.trug.get(self.pool[q]["mint"], math.inf) <= ts for q in earlier))
            mine = words(p["name"])
            out["dv_same_theme_launches"] = float(sum(bool(words(self.pool[q]["name"]) & mine) for q in earlier
                                                      if p["created_ts"] - self.pool[q]["created_ts"] <= DAY))
        h, tr, acc, supply, cr = self._trades(pool, ts)
        if tr and cr and supply:                                                                 # 128
            out_slots = {t[1] for t in tr if t[2] == cr and t[3] < 0}
            pool_slots = {t[1] for t in tr if t[2] in acc}
            out["dv_transfers_out"] = float(len(out_slots - pool_slots))
        return out

    # H. 139-144
    def _security(self, pool, p, ts, snap, info):
        out = {}
        rc = self._known(pool, "rugcheck", ts)
        if rc:
            risks = " ".join((r or "").lower() for r in rc.get("risks") or [])
            out.update({"sc_rugcheck_score": _log1p(rc.get("score")),                         # 139
                        "sc_mutable_metadata": float("mutable" in risks),                       # 141
                        "sc_update_authority": float("authority" in risks and "mint" not in risks),  # 142
                        "sc_risk_count": float(len(rc.get("risks") or []))})
        jup = self._known(pool, "jupiter", ts)
        if jup:
            out["sc_token_2022"] = float(jup.get("token_program") == TOKEN_2022)               # 140
            out["sc_odd_decimals"] = float((jup.get("decimals") or 6) != 6)                     # 144
            ts_supply = _f(jup.get("total_supply"))
            out["sc_odd_supply"] = float(bool(ts_supply) and abs(ts_supply - 1e9) / 1e9 > 0.1)
            out["sc_dev_mints"] = _log1p(jup.get("dev_mints"))
        q = self._known(pool, "jupquote", ts)
        if q:
            loss = _f(q.get("round_trip_loss_pct_100"))
            out["sc_sell_ok"] = float(bool(q.get("routable")) and loss is not None and loss < 50)  # 143
        return out

    # I. 151-161
    def _identity(self, pool, p, ts, snap, info):
        name = (p["name"] or "").split(" / ")[0]
        sym = _symbol(p["name"])
        out = {
            "id_ticker_len": float(len(sym)),                                                    # 151
            "id_emoji": float(any(unicodedata.category(ch) == "So" for ch in name)),            # 155
            "id_non_latin": float(any(ord(ch) > 0x2E7F and unicodedata.category(ch).startswith("L") for ch in name)),  # 156
        }
        key = sym.upper()
        older = bisect.bisect_left(self.symbol_created.get(key, []), p["created_ts"])            # tokens created before
        out["id_ticker_reused"] = _log1p(older)                                                   # 153, 272
        own = self.ctx.t2x.get(p["mint"])
        pumped = bisect.bisect_right(self.symbol_pumps.get(key, []), ts) - (own is not None and own <= ts)
        out["id_ticker_prior_pumped"] = float(pumped > 0)
        cc = self._known(pool, "crosschain", ts)
        if cc:
            out["id_older_same_ticker_vol"] = _log1p(cc.get("solana_older_max_vol24"))
        img = self._known(pool, "image", ts)
        if img:
            rgb = img.get("mean_rgb")
            if rgb:
                r, g, b = (x / 255 for x in rgb)
                mx, mn = max(r, g, b), min(r, g, b)
                out["id_img_brightness"] = (mx + mn) / 2                                         # 160
                out["id_img_saturation"] = (mx - mn)
            out["id_img_animated"] = float(bool(img.get("animated")))                           # 161
        return out

    # J. 152, 157-159, 163, 171-180 (Haiku fields + our own heat measures)
    def _narrative(self, pool, p, ts, snap, info):
        out = {}
        n = self._known(pool, "narrative", ts)
        if n:
            for k in ("ticker_is_common_word", "has_lore", "tied_to_scheduled_event", "tied_to_breaking_news",
                      "references_viral_video", "community_takeover", "claims_team_doxxed", "claims_buyback_or_burn",
                      "claims_product_or_game", "image_ai_generated", "image_has_face", "image_has_animal",
                      "image_has_text"):
                if k in n:
                    out[f"nr_{k}"] = float(bool(n[k]))
            out["nr_memeability"] = _f(n.get("memeability"))
            for tone in ("lore", "shill", "meme", "informative"):
                out[f"nr_tone_{tone}"] = float(n.get("description_tone") == tone)
            for hum in ("absurd", "wholesome", "edgy"):
                out[f"nr_humor_{hum}"] = float(n.get("humor") == hum)
            cat = n.get("category")                                                               # 171, 268
            if cat:
                cb = self.ctx._count_between
                launched = cb(self.cat_created[cat], ts - 7 * DAY, ts)
                out["nr_category_pump_rate_7d"] = (cb(self.cat_pumps[cat], ts - 7 * DAY, ts) / launched
                                                   if launched else None)
                hour_all = cb(self.all_cat_created, ts - HOUR, ts)
                out["nr_category_share_1h"] = (cb(self.cat_created[cat], ts - HOUR, ts) / hour_all
                                               if hour_all else None)
        mine = words(p["name"])                                                                   # 172, 264
        if mine:
            out["nr_first_of_theme_7d"] = float(not any(
                self.ctx._count_between(self.ctx.word_created[w], ts - 7 * DAY, p["created_ts"] - 1) for w in mine))
        cc = self._known(pool, "crosschain", ts)
        if cc:                                                                                     # 180, 265
            out["nr_other_chain_twins"] = _log1p(cc.get("other_chain_pairs"))
            out["nr_other_chain_vol"] = _log1p(cc.get("other_chain_max_vol24"))
        return out

    # K + L. 190-197, 199, 203-205, 210, 212-214
    def _social(self, pool, p, ts, snap, info):
        out = {}
        x = self._known(pool, "x", ts)
        if x:
            posts = [q for q in x.get("posts") or [] if q.get("ts") and q["ts"] <= ts]
            out["sx_posts_last_10m"] = float(sum(ts - q["ts"] <= 600 for q in posts))             # 190
            out["sx_cashtag_count"] = _log1p(x.get("cashtag_count_7d"))                           # 191
            acct = x.get("token_account") or {}
            if acct.get("created"):
                out["sx_account_age_days"] = max(p["created_ts"] - acct["created"], 0) / DAY     # 192
                out["sx_account_followers"] = _log1p(acct.get("followers"))
            lp = x.get("linked_post") or {}
            if lp:
                out["sx_linked_post_likes"] = _log1p(lp.get("likes"))                             # 193
                out["sx_linked_post_views"] = _log1p(lp.get("views"))
                out["sx_linked_post_quotes"] = _log1p(lp.get("quotes"))                           # 194
                out["sx_linked_author_followers"] = _log1p(lp.get("author_followers"))
            if self.paid_callers:
                out["sx_paid_callers"] = float(sum((q.get("username") or "") in self.paid_callers for q in posts))  # 197
        s = self._known(pool, "sentiment", ts)
        if s:
            out.update({"sx_sentiment": _f(s.get("sentiment")), "sx_shill_share": _f(s.get("shill_share")),  # 196, 199
                        "sx_rug_warnings": float(bool(s.get("rug_warnings"))),
                        "sx_organic_discussion": float(bool(s.get("organic_discussion")))})
        mine = words(p["name"])                                                                    # 195, 154, 287
        if mine and self.xfeed:
            day = [t for pt, t in self.xfeed if ts - DAY <= (pt or 0) <= ts]
            out["sx_theme_in_mega_posts"] = float(any(w in t for t in day for w in mine))
            out["sx_mega_posts_24h"] = float(len(day))
        links = self._known(pool, "links", ts)
        if links:
            out["sl_telegram_members"] = _log1p(links.get("telegram_members"))                    # 203
            out["sl_discord_members"] = _log1p(links.get("discord_members"))                      # 205
            web = links.get("website") or {}
            if web:
                out.update({"sl_site_reachable": float(bool(web.get("reachable"))),               # 212
                            "sl_site_log_bytes": _log1p(web.get("bytes")), "sl_site_links": _log1p(web.get("links")),
                            "sl_site_template": float(bool(web.get("template_host")))})          # 214
            if links.get("domain_registered_ts"):
                out["sl_domain_age_days"] = max(p["created_ts"] - links["domain_registered_ts"], 0) / DAY  # 213
        mentions = [e for e in self.ctx.mentions.get(p["mint"], []) if e[0] <= ts]
        out["sl_biz_mentions"] = float(sum(1 for _, ch in mentions if ch == "biz"))               # 210
        callers = {ch for _, ch in mentions if ch != "biz"}                                        # 204
        if callers:
            hits = total = 0
            for ch in callers:
                for pt, m in self.channel_calls.get(ch, []):
                    if pt < p["created_ts"] and m != p["mint"]:
                        total += 1
                        hits += self.ctx.t2x.get(m, math.inf) <= min(ts, pt + DAY)
            out["sl_caller_hit_rate"] = hits / total if total else None
        return out

    # M + N. 225, 226, 237, 238
    def _attention(self, pool, p, ts, snap, info):
        seen = [e for e in self.ctx.lists.get(p["mint"], []) if e[0] <= ts]
        jup = [e for e in seen if e[1] == "jup:trending_1h"]
        out = {"at_jup_trending": float(bool(jup)),                                                # 225
               "at_jup_trending_inv_rank": max((1 / e[2] for e in jup if e[2]), default=0.0),
               "at_coingecko_listed": float(bool((info or {}).get("coingecko_id")))}             # 226
        amounts = [e[3] for e in seen if e[1].startswith("dex:token-boosts")]
        out["at_boost_steps"] = float(sum(b > a for a, b in zip(amounts, amounts[1:])))          # 238
        od = self._known(pool, "orders", ts)
        if od:
            types = [o.get("type") for o in od.get("orders") or [] if 0 < (o.get("paid_ts") or 0) <= ts]
            out["at_trending_bar_ad"] = float("trendingBarAd" in types)                          # 237
        return out

    # O. 248, 249, 251-255, 257-259, 288
    def _market(self, pool, p, ts, snap, info):
        meme = [self._chg(k, ts, DAY) for k in ("meme_wif_usd", "meme_bonk_usd", "meme_popcat_usd")]
        meme = [m for m in meme if m is not None]
        out = {
            "mk_btc_chg_1h": self._chg("btc_usd", ts, HOUR), "mk_btc_chg_24h": self._chg("btc_usd", ts, DAY),  # 248
            "mk_eth_chg_24h": self._chg("eth_usd", ts, DAY),                                      # 249
            "mk_graduations_1h": float(self.ctx._count_between(self.amm_created, ts - HOUR, ts)),  # 251
            "mk_tps": self._series_at("tps", ts), "mk_priority_fee": _log1p(self._series_at("priority_fee_median", ts)),  # 252
            "mk_sol_funding": self._series_at("sol_funding", ts),                                 # 253
            "mk_sol_oi_chg_24h": self._chg("sol_open_interest", ts, DAY),
            "mk_stables_7d_chg": self._series_at("sol_stables_7d_chg", ts),                       # 254
            "mk_fear_greed": self._series_at("fear_greed", ts),                                   # 255
            "mk_meme_index_chg_24h": sum(meme) / len(meme) if meme else None,                     # 257
            "mk_5x_last_6h": float(self.ctx._count_between(self.t5x_sorted, ts - 6 * HOUR, ts)),  # 258
        }
        # 288: hours since the memecoin index last fell 10% within a day
        series = self.macro.get("meme_wif_usd", [])
        last_crash = None
        for i, (t, v) in enumerate(series):
            if t > ts:
                break
            j = bisect.bisect_left(self.macro_ts["meme_wif_usd"], t - DAY)
            if j < i and series[j][1] and v / series[j][1] - 1 <= -0.10:
                last_crash = t
        out["mk_hours_since_meme_crash"] = (ts - last_crash) / HOUR if last_crash else None
        return out

    # P. 261-263, 266, 267, 269, 272, 274
    def _cross(self, pool, p, ts, snap, info):
        mine = words(p["name"])
        out = {}
        if mine:
            out["cx_theme_pumps_1h"] = _log1p(max(self.ctx.pumps_between(w, ts - HOUR, ts, p["mint"])
                                                   for w in mine))                                     # 261
            own_rug = self.trug.get(p["mint"])
            out["cx_theme_rugs_6h"] = _log1p(max(
                self.ctx._count_between(self.word_rug_ts.get(w, []), ts - 6 * HOUR, ts)
                - (own_rug is not None and ts - 6 * HOUR < own_rug <= ts) for w in mine))            # 262
            out["cx_theme_copies_10m"] = _log1p(max(self.ctx._count_between(self.ctx.word_created[w], ts - 600, ts)
                                                     for w in mine) - (p["created_ts"] > ts - 600))  # 263
        h, tr, acc, supply, creator = self._trades(pool, ts)
        if tr:
            buyers = {t[2] for t in tr if t[3] > 0 and t[2] not in acc}
            from_winner = from_rug = 0
            for w in buyers:
                hist = [m for c, m in self.ctx.early_wallets.get(w, []) if c < p["created_ts"] and m != p["mint"]]
                from_winner += any(ts - DAY <= self.ctx.t2x.get(m, -math.inf) <= ts for m in hist)   # 266
                from_rug += any(ts - DAY <= self.trug.get(m, -math.inf) <= ts for m in hist)          # 267
            out["cx_buyers_from_winner"] = float(from_winner)
            out["cx_buyers_from_rug"] = float(from_rug)
        out["cx_launches_same_minute"] = float(self.ctx._count_between(self.ctx.created_sorted, p["created_ts"] - 60,
                                                                      p["created_ts"]) - 1)        # 269
        out["cx_graduations_10m"] = float(self.ctx._count_between(self.amm_created, ts - 600, ts))  # 274
        return out

    # Q. 277-283, 287 (session/day) - the literal stars included
    def _calendar(self, pool, p, ts, snap, info):
        dt = datetime.fromtimestamp(ts, timezone.utc)
        d = dt.date()
        last_day = (date(d.year + (d.month == 12), d.month % 12 + 1, 1) - d).days == 1
        phase = moon_phase(ts)
        out = {f"cal_dow_{i}": float(dt.weekday() == i) for i in range(7)}                       # 277
        out.update({
            "cal_asia_session": float(0 <= dt.hour < 8), "cal_europe_session": float(7 <= dt.hour < 16),  # 278
            "cal_us_session": float(13 <= dt.hour < 21),
            "cal_us_market_open": float(dt.weekday() < 5 and (13, 30) <= (dt.hour, dt.minute) < (20, 0)
                                        and d not in US_HOLIDAYS),                                    # 259
            "cal_us_holiday": float(d in US_HOLIDAYS),                                             # 279
            "cal_asia_holiday": float(d in ASIA_HOLIDAYS),                                         # 280
            "cal_month_edge": float(d.day <= 3 or last_day or d.day >= 28),                        # 281
            "cal_moon_sin": math.sin(2 * math.pi * phase), "cal_moon_cos": math.cos(2 * math.pi * phase),  # 282
            "cal_full_moon": float(abs(phase - 0.5) < 0.04),
            "cal_mercury_retrograde": float(any(a <= d <= b for a, b in MERCURY_RETROGRADE)),     # 283
        })
        return out

    # R. 290, 292, 294-296
    def _later(self, pool, p, ts, snap, info):
        out = {"lt_perp_listed": float(_symbol(p["name"]).upper() in self.perps) if self.perps else None}  # 290
        out["lt_coingecko_page"] = float(bool((info or {}).get("coingecko_id")))                   # 292
        return out

    # S. 297-299 (300 is ho_airdropped_share)
    def _manipulation(self, pool, p, ts, snap, info):
        h, tr, acc, supply, creator = self._trades(pool, ts)
        if not tr:
            return {}
        counts = defaultdict(lambda: [0, 0])
        for t in tr:
            if t[2] not in acc:
                counts[t[2]][0 if t[3] > 0 else 1] += 1
        traders = len(counts) or 1
        first_slot = min(t[1] for t in tr)
        per_slot = defaultdict(set)
        for t in tr:
            if t[3] > 0 and t[2] not in acc and t[1] > first_slot + 1:
                per_slot[t[1]].add(t[2])
        tiny = [t for t in tr if t[2] not in acc and supply and abs(t[3]) / supply < 1e-5]
        return {
            "mn_wash_share": sum(b >= 2 and s >= 2 for b, s in counts.values()) / traders,          # 297
            "mn_tiny_trade_share": len(tiny) / len(tr),                                              # 298
            "mn_max_buyers_one_slot": float(max((len(v) for v in per_slot.values()), default=0)),   # 299
        }
