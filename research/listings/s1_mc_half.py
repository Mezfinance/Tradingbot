import json, numpy as np
src = open("scripts/s1_fund.py").read().split("res = {")[0]
exec(src)
rng = np.random.default_rng(2); span = T[-1]["ts"] - T[0]["ts"]; rate = len(T) / span
out = {}
for nav0 in SIZES:
    r = {1: [], 2: [], 3: []}
    for sim in range(200):
        n = rng.poisson(rate * 3 * 365 * 86400)
        ts = np.sort(rng.uniform(0, 3 * 365 * 86400, n)) + 1.9e9
        tr = [dict(T[i], ts=float(x)) for i, x in zip(rng.integers(0, len(T), n), ts)]
        c, _ = run(tr, nav0, haircut=0.5, start=1.9e9); navs = np.array([v for d, v in c])
        for y in (1, 2, 3): r[y].append(navs[min(len(navs) - 1, y * 365)] / nav0 - 1)
    out[nav0] = {y: [float(np.percentile(r[y], p)) for p in (10, 50, 90)] for y in r}
    out[nav0]["p_loss_1y"] = float(np.mean(np.array(r[1]) < 0))
    print(nav0, {y: [round(x * 100) for x in out[nav0][y]] for y in (1, 2, 3)}, out[nav0]["p_loss_1y"], flush=True)
json.dump(out, open("cex/s1_mc_half.json", "w"))
