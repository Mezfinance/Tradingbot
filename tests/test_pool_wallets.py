import json

from tradingbot import pool_wallets as pw
from tradingbot import pool_tracker as pt


def test_auc_separates_planted_values():
    assert pw._auc([3, 4, 5], [0, 1, 2]) == 1.0       # ruggers all higher
    assert pw._auc([1, 1], [1, 1]) == 0.5             # identical = no tell
    assert pw._auc([0, 1], [3, 4]) == 0.0             # ruggers all lower


def _bot_stream():
    # one metronomic wash bot (buys+sells, fixed size, 2 s cadence) dominating, + a few one-off wallets
    recs = []
    for i, t in enumerate(range(0, 60, 2)):
        recs.append([float(t), "BOT", 1 if i % 2 == 0 else -1, 100.0])
    for j, w in enumerate(("a", "b", "c")):
        recs.append([float(5 + j), w, 1, 10.0 + j])
    return {"recs": recs, "creator": "DEV", "complete": True}


def _diverse_stream(seed):
    # many distinct wallets, one trade each, varied sizes and irregular gaps: no bot fingerprint
    recs = []
    t = 0.0
    for k in range(30):
        t += 0.3 + (k % 7)
        recs.append([round(t, 1), f"w{seed}_{k}", 1 if k % 3 else -1, float(5 + (k * 37) % 50)])
    return {"recs": recs, "creator": "DEV2", "complete": True}


def _rug_stream():
    # the creator trading its own pool, few wallets, few trades, no fixed-size bot: the rug fingerprint
    recs = [[1.0, "DEV", 1, 100.0], [3.0, "DEV", 1, 55.0], [8.0, "DEV", -1, 30.0], [12.0, "DEV", 1, 70.0],
            [5.0, "x", 1, 41.0], [6.0, "y", 1, 22.0], [9.0, "x", -1, 13.0], [14.0, "z", 1, 33.0], [18.0, "y", -1, 7.0]]
    return {"recs": recs, "creator": "DEV", "complete": True}


class _MemStore:
    def __init__(self):
        import sqlite3
        self.c = sqlite3.connect(":memory:")
        self.c.row_factory = sqlite3.Row

    def execute(self, sql, p=()):
        self.c.execute(sql, p)
        self.c.commit()

    def query(self, sql, p=()):
        return [dict(r) for r in self.c.execute(sql, p).fetchall()]


def test_gate_score_ranks_the_rug_higher_and_fails_open():
    ref = [f for f in (pw.wallet_features(_diverse_stream(i), 66) for i in range(25)) if f]
    s_r, rank_r, why_r = pw.gate_score(pw.wallet_features(_rug_stream(), 66), ref)
    s_c, rank_c, why_c = pw.gate_score(pw.wallet_features(_diverse_stream(99), 66), ref)
    assert why_r == "ok" and why_c == "ok"
    assert rank_r > rank_c and rank_r >= 0.5          # the rug pattern ranks in the riskiest half
    assert pw.gate_score(pw.wallet_features(_rug_stream(), 66), ref[:5])[2] == "insufficient_history"
    assert pw.gate_score(None, ref)[2] == "too_sparse"  # no usable stream -> caller fails open


def test_reference_features_reads_the_cache():
    store = _MemStore()
    store.execute(pw.SCHEMA)
    for i in range(3):
        store.execute("INSERT OR REPLACE INTO watch_pool_wallets(pool, data, ts) VALUES(?,?,?)",
                      (f"P{i}", json.dumps(_diverse_stream(i)), 1000.0 + i))
    ref = pw.reference_features(store, 66)
    assert len(ref) == 3 and all("n_wallets" in f for f in ref)


def test_wallet_features_flag_the_bot_pattern():
    rug = pw.wallet_features(_bot_stream(), 66)
    surv = pw.wallet_features(_diverse_stream(1), 66)
    assert rug and surv
    assert rug["top1_share"] > surv["top1_share"]          # the bot dominates the rugger's trades
    assert rug["trades_per_wallet"] > surv["trades_per_wallet"]
    assert rug["wash_frac"] > surv["wash_frac"]            # the bot both buys and sells
    assert surv["cadence_cv"] is not None


class _J:
    def __init__(self, metas):
        self.metas = metas

    def query(self, sql, p=()):
        if "canary_feint" in sql or "canary_probe" in sql and "verdict" not in sql:
            return [{"ts": m["create_ts"] + 60, "token": m["mint"], "kind": "canary_probe",
                     "detail": json.dumps({"pool": m["pool"], "verdict": "ok"})} for m in self.metas]
        return []          # no verdict rows


class _S:
    def __init__(self, metas, cache):
        self.metas = metas
        self.cache = cache

    def query(self, sql, p=()):
        if "watch_market" in sql:
            return [{"sol_usd": 200.0}]
        if "watch_pool_wallets" in sql:
            return [{"pool": pool, "data": json.dumps(d)} for pool, d in self.cache.items()]
        if "watch_launch_live" in sql:
            base = {c: None for c in pt.SCORER_COLS}
            return [{**base, "pool": m["pool"], "token_mint": m["mint"], "create_ts": m["create_ts"],
                     "pulled_ts": (m["create_ts"] + 40) if m["rug"] else None,
                     "ret": -1.0 if m["rug"] else 0.3, "status": "done"} for m in self.metas]
        return []

    def execute(self, *a):
        pass


def test_study_separates_ruggers_by_wallet_pattern():
    metas, cache = [], {}
    for i in range(10):
        p = f"RUG{i}"
        metas.append({"pool": p, "mint": f"m{p}", "create_ts": 1000.0 + i, "rug": True})
        cache[p] = _bot_stream()
    for i in range(10):
        p = f"SURV{i}"
        metas.append({"pool": p, "mint": f"m{p}", "create_ts": 2000.0 + i, "rug": False})
        cache[p] = _diverse_stream(i)
    out = pw.study(_J(metas), _S(metas, cache), helius=None, now=3000.0)
    assert out["n_pools"] == 20 and out["n_ruggers"] == 10
    feats = {d["feature"]: d for d in out["entry"]["features"]}
    # the bot fingerprint separates ruggers from survivors at entry
    assert feats["top1_share"]["auc"] > 0.8               # ruggers score much higher
    assert feats["top1_share"]["top_q_rug"] == 1.0        # the top feature quintile is all ruggers
    txt = pw.render(out)
    assert "POOL WALLETS" in txt and "AT ENTRY" in txt
