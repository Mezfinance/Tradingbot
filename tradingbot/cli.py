"""Command line entry point.

  python -m tradingbot scan     # screen live tokens, no Claude calls, no trades
  python -m tradingbot once     # one full tick (manage, research, enter)
  python -m tradingbot run      # loop forever
  python -m tradingbot report   # journal statistics
  python -m tradingbot replay --group bought --set stop_loss_pct=20 --split 2026-10-18
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from tradingbot.config import LIVE_CONFIRM_PHRASE, Config


def canary_executor(cfg: Config, jupiter):
    """The launch canary's live executor: Jupiter routes, swaps straight on the pumpswap pool
    (`launch_canary_route` = 'direct'), or enters as a liquidity provider ('lp', 11-10). With `launch_canary_lp_entry`
    on, the route stays 'direct' (so the 1-slot probes are direct round-trips) but the per-wallet sub is the LP
    executor, so only the two-stage entry's commit (`buy`) and its exit (`sell`) are LP deposit/withdraw; `probe_1slot`
    is the inherited direct method, so probes are unchanged."""
    if cfg.launch_canary_route == "lp" or (cfg.launch_canary_route == "direct" and cfg.launch_canary_lp_entry):
        from tradingbot.execution.lp_pumpswap import LpPumpswapExecutor as cls
    elif cfg.launch_canary_route == "direct":
        from tradingbot.execution.pumpswap_direct import DirectPumpswapExecutor as cls
    else:
        from tradingbot.execution.jupiter import JupiterExecutor as cls

    def make(keypair=None):
        return cls(cfg, jupiter, keypair=keypair, priority_fee_sol=cfg.launch_canary_priority_fee_sol,
                   sell_slippage_bps=cfg.launch_canary_sell_slippage_bps)

    if cfg.launch_canary_rotate and cfg.launch_canary_route == "direct":
        from tradingbot.execution.rotating import RotatingExecutor, read_canary_wallets

        wallets = read_canary_wallets(cfg)
        if len(wallets) >= 1:  # rotation runs with a single wallet too: it reuses that one wallet for every trade
            subs = [make(k) for k in wallets]
            for s in subs:  # the rotator owns pool persistence; stop the per-wallet executors sharing one pool file
                s._pool_file = None
            db = getattr(cfg, "db_path", ":memory:")
            state = None if db == ":memory:" else os.path.join(os.path.dirname(db) or ".", "canary_rotation.json")
            return RotatingExecutor(subs, state_path=state)
    return make()


def build_engine(cfg: Config, with_analyst: bool = True):
    from tradingbot.analyst import Analyst
    from tradingbot.data.dexscreener import DexScreener
    from tradingbot.data.jupiter import Jupiter
    from tradingbot.data.rugcheck import RugCheck
    from tradingbot.engine import Engine
    from tradingbot.journal import Journal

    from tradingbot import canary_settings
    canary_settings.apply_overlay(cfg)  # dashboard-saved canary settings persist across restarts (defensive, no-op if none)

    jupiter = Jupiter(cfg.slippage_bps)
    if cfg.mode == "live":
        from tradingbot.execution.jupiter import JupiterExecutor
        executor = JupiterExecutor(cfg, jupiter)
    else:
        from tradingbot.execution.sim import SimExecutor
        executor = SimExecutor(cfg, jupiter)
    from dataclasses import replace

    from tradingbot.execution.sim import SimExecutor
    from tradingbot.shadows import parse_shadows

    # Baselines are always simulated, even when the main book trades live.
    shadow_executors = {name: SimExecutor(cfg, jupiter) for name in parse_shadows(cfg.shadows)}
    canary = None
    if cfg.canary_every_n > 0:
        from tradingbot.execution.jupiter import JupiterExecutor
        canary = JupiterExecutor(cfg, jupiter)
    launch_canary = None
    if (cfg.launch_canary or cfg.launch_fast_canary) and cfg.model_book:
        # the fast canary buys seconds-old pools, which only the direct pumpswap route can swap (Jupiter can't route them)
        route = "direct" if cfg.launch_fast_canary and not cfg.launch_canary else cfg.launch_canary_route
        launch_canary = canary_executor(
            replace(cfg, max_price_impact_pct=cfg.model_book_max_impact_pct, launch_canary_route=route), jupiter)
    analyst = Analyst(cfg) if with_analyst else None
    triage = None
    if analyst is not None and cfg.triage:
        from tradingbot.triage import Triage
        triage = Triage(cfg, analyst.client)
    from tradingbot.data.geckoterminal import GeckoTerminal

    # model5 is always simulated, with its own impact cap (the Claude book's 3% suits older, deeper pools).
    model_executor = (SimExecutor(replace(cfg, max_price_impact_pct=cfg.model_book_max_impact_pct), jupiter)
                      if cfg.model_book else None)
    from tradingbot.data.extras import Extras
    return Engine(cfg, DexScreener(), RugCheck(), analyst, executor, Journal(cfg.db_path), jupiter,
                  shadow_executors=shadow_executors, canary_executor=canary, gecko=GeckoTerminal(),
                  triage=triage, model_executor=model_executor, extras=Extras() if cfg.model_book else None,
                  launch_canary_executor=launch_canary)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "keyvault":  # free-form sub-args, handled outside the main parser
        from tradingbot import keyvault

        keyvault.main(argv[1:])
        return
    if argv and argv[0] == "provision":  # re-mint + fund the canary wallets; its own argparse
        from tradingbot import provision

        provision.main(argv[1:])
        return
    parser = argparse.ArgumentParser(prog="tradingbot")
    parser.add_argument("command", choices=["scan", "once", "run", "report", "replay", "health", "backup",
                                            "backup-now", "backup-list", "learn-now", "postmortems", "playbook", "optimize-now", "settings",
                                            "settings-rollback", "code-versions", "code-diff", "code-rollback",
                                            "propose", "proposals", "watch-status", "watch-report",
                                            "watch-export", "watch-links", "watch-report-now",
                                            "watch-model-train", "model-status", "early-status",
                                            "early-backtest", "early-walk-forward", "launch-backtest", "launch-backfill", "launch-status", "launch-compare",
                                            "launch-hist-backfill", "launch-hist-test", "launch-exit-study",
                                            "launch-late-entry", "launch-early-pulls",
                                            "launch-pull-train", "launch-token-age", "launch-d12-trades",
                                            "launch-filter-study", "launch-rw-train", "launch-rw24-train", "launch-big-train", "dashboard-summary",
                                            "canary-status", "canary-stop", "canary-start", "canary-one", "canary-smoke", "canary-feint-smoke", "canary-probe-smoke", "rw-check", "early-rw-check", "early-rw-study", "fill-gap-study", "size-study", "canary-rug-check", "canary-sweep", "block-compare", "market-monitor", "rw-bands", "band-rw-study", "band-compare", "risk-model", "history-backfill", "hold-study", "band-prize", "sz-whatif", "safeguards", "thin-sizing", "missed-signals", "flag-tracker", "dd-limits", "sz-stops", "canary-package", "canary-trades", "pool-tracker", "canary-slots", "probe-exits", "pool-wallets", "flurry-study", "frontrun-verify", "frontrun-fill", "localshred-merge", "capture-fresh", "shred-completeness", "symbol-flip", "router-direction"])
    parser.add_argument("path", nargs="?",
                        help="backup: destination file; code-diff/code-rollback: version (vN, or 'image')")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--all", action="store_true", help="report: every book, not just model5")
    parser.add_argument("--rps", type=float, default=2.0,
                        help="launch-hist-backfill: Helius calls per second (the live scorer shares the plan's limit)")
    parser.add_argument("--json", action="store_true", help="report: print raw JSON")
    parser.add_argument("--no-live", action="store_true", help="pool-tracker: skip the live reserve read (scorer data only)")
    parser.add_argument("--workers", type=int, default=8, help="history-backfill: parallel Helius calls")
    parser.add_argument("--estimate", action="store_true", help="history-backfill: list the pools and the credits only")
    parser.add_argument("--relist", action="store_true",
                        help="history-backfill: list the pools again instead of reusing the saved list (< 7 days old)")
    parser.add_argument("--pulls-only", action="store_true",
                        help="history-backfill: only add LP pull times to pools saved before they were recorded")
    parser.add_argument("--history", action="store_true",
                        help="risk-model: replay the books over hist.db + the scorer table instead of their real trades")
    parser.add_argument("--retrain-h", type=float, default=24.0, help="risk-model --history: model retrain interval")
    parser.add_argument("--fresh", action="store_true",
                        help="safeguards: replay again instead of using the replay saved by the last risk-model --history")
    parser.add_argument("--hold15", action="store_true",
                        help="risk-model --history: add sz with the minute-15 hold at $500 in $25-40k pools (fetches "
                             "those pools' minute-15 states first, ~10 credits each, live key; --max-credits, --rps)")
    parser.add_argument("--per-band", type=int, default=150, help="hold-study: pools sampled per liquidity band")
    parser.add_argument("--min-liq", type=float, default=25_000.0, help="hold-study: smallest pool (USD at second 4)")
    parser.add_argument("--max-liq", type=float, default=0.0, help="hold-study: largest pool (USD at second 4; 0 = none)")
    parser.add_argument("--kept-only", action="store_true", help="hold-study: only the rug-vs-win kept half")
    parser.add_argument("--fetch", action="store_true",
                        help="band-prize: fetch minute-10/15/30 pool states from Helius first (~10 credits each)")
    parser.add_argument("--books", default="", help="risk-model: comma list of books (default: every running book)")
    parser.add_argument("--year", action="store_true", help="dd-limits: also replay the limits over the year (hist.db)")
    parser.add_argument("--book", default="launchrw50_sz", help="missed-signals / flag-tracker: the launch book to look at")
    parser.add_argument("--group", default="bought", help="replay: bought|passed|survivors|filters|all")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="replay: override a setting, e.g. stop_loss_pct=20 (repeatable)")
    parser.add_argument("--split", help="replay: YYYY-MM-DD; tune before it, check after it")
    parser.add_argument("--size", type=float, default=50.0, help="replay: USD per trade")
    parser.add_argument("--age", type=int, default=15, help="watch-report/watch-export: feature age in minutes")
    parser.add_argument("--label", default="pump2x", help="watch-report: pump2x | pump5x | clean2x")
    parser.add_argument("--enriched", action="store_true", help="watch-report: only tokens with Helius/X data")
    parser.add_argument("--control", action="store_true",
                        help="watch-report: only tokens whose Helius lookup ignored the free score (control slice)")
    parser.add_argument("--paid-value", action="store_true",
                        help="watch-report: is Helius / X data worth paying for (control slice, 15-min features)")
    parser.add_argument("--hours", type=float, default=None, help="early-status / early-backtest (default 8) / launch-backtest (default all): look back this many hours")
    parser.add_argument("--days", type=float, default=7.0, help="launch-backfill: pools created in the last N days")
    parser.add_argument("--test-from", default=None,
                        help="launch-backtest: score only days from this UTC date (YYYY-MM-DD); earlier days only train")
    parser.add_argument("--no-name", action="store_true",
                        help="launch-backtest: timing table without name features (the live scorer can't see the name)")
    parser.add_argument("--fetch-candles", type=int, default=0, metavar="N",
                        help="launch-backtest: fetch historical 1-minute candles for the decide-4/fill-6 picks plus N "
                             "random liquid pools (GeckoTerminal, saved for reruns) for the exit timing check")
    parser.add_argument("--export-exit", type=int, default=0, metavar="N",
                        help="launch-backtest: print the decide-4/fill-6 picks + N random liquid pools as JSON "
                             "(times, snapshot return, chain link) for the exit timing check run elsewhere")
    parser.add_argument("--timing", action="store_true",
                        help="launch-backtest: only the decide-at-d / fill-at-f timing table (faster)")
    parser.add_argument("--max-pools", type=int, default=2000, help="launch-backfill: at most this many pools")
    parser.add_argument("--max-credits", type=int, default=None,
                        help="Helius credit cap (defaults: backfills 60000, launch-exit-study 10000, "
                             "launch-late-entry 12000)")
    parser.add_argument("--window-s", type=float, default=120.0, help="pool-wallets: fetch + full-window features over the first N s of trading")
    parser.add_argument("--entry-s", type=float, default=66.0, help="pool-wallets: the selection cut (~confirmation); features here are what a live filter would see")
    parser.add_argument("--span-s", type=float, default=180.0, help="flurry-study: fetch + reconstruct the first N s of each pool's trading")
    parser.add_argument("--cost-pct", type=float, default=1.0, help="flurry-study: round-trip cost (pool fee + tip) as a %% of notional")
    parser.add_argument("--probed-only", action="store_true", help="flurry-study: restrict to pools we have probed/feinted")
    parser.add_argument("--flurry-min", type=int, default=3, help="flurry-study: a slot is a flurry at >= this many buys")
    parser.add_argument("--target", choices=["price", "flurry"], default="price", help="flurry-study: what discover/characterize/motif predict -- a price up-move (default) or a buy-count flurry")
    parser.add_argument("--move-pct", type=float, default=1.0, help="flurry-study: price target -- a rise of this many %% counts as a hit (default = round-trip cost)")
    parser.add_argument("--horizon", type=int, default=2, help="flurry-study: price target -- the up-move may land within the next this-many slots")
    parser.add_argument("--scalp", action="store_true", help="flurry-study: run the after-the-flurry continuation table instead of the lead-lag characterization")
    parser.add_argument("--discover", action="store_true", help="flurry-study: data-driven discovery (auto rule bank + motifs, holdout-confirmed) instead of the named-event characterization")
    parser.add_argument("--hybrid", action="store_true", help="flurry-study --discover / --frontrun: the box's live architecture -- score the CURRENT slot from the DIRECT (shred) view (its own abs_move/move zeroed, not live-available), rebase the motif history + lagged prev_* (incl. prev_abs_move) to the FULL (Helius) view. Tests whether a rebased prior-slot abs_move predictor holds once history is Helius-accurate but the current slot is shred-fresh.")
    parser.add_argument("--combos", action="store_true", help="flurry-study --discover: also beam-search CONJUNCTIONS of filters (AND them together), kept only where the stacked rule clears a stricter family-wise z guard AND is sharper out of sample than any of its single filters -- does stacking filters surface a cleaner edge")
    parser.add_argument("--tail", action="store_true", help="flurry-study: tail-concentration breakdown of the discovered survivors (is the edge a broad tail or a few pools)")
    parser.add_argument("--discover-econ", action="store_true", help="flurry-study: economics-ranked discovery -- SELECT rules by the scalp P&L they'd have made (significant vs the control on mean OR median, holdout-confirmed, family-wise guarded), not by lift. Surfaces rules whose lift was unremarkable but whose median/mean%% pays, labelled broad vs tail with move direction; runs both sides (momentum + reversion) so the up-side fat tail can be distrusted while the down-side tail is read as the fills-favoured bounce.")
    parser.add_argument("--creation", action="store_true", help="flurry-study: reactive first-buy backrun -- see the pool's FIRST BUY in slot N (ShredStream), enter front of N+1 ('straight away'), hold k slots AFTER entry; reports the return per hold, where the first LP pull lands relative to OUR entry (does the 2-slot dodge hold at creation), and the raw move the first buy leads to. Free off the cached flows.")
    parser.add_argument("--slot0", action="store_true", help="flurry-study: slot-0 landing -- if we land in the creation slot, sweep the hold length and report the smallest hold that is EV+ (if any), with the raw opening drift and the cumulative rug-by-hold, i.e. how long we must hold and whether EV+ means holding past the ~2-slot safe window. Free off the cached flows.")
    parser.add_argument("--multibuy", action="store_true", help="flurry-study: band each flurry slot by its buy count (0-15, 15-25, ... 95+), enter the slot after (front of N+1) and sweep the hold 1-10 slots, reporting the net scalp per band x hold so the hold where the mean peaks is the ideal exit for that band -- the dynamics change once we are in, so the ideal exit is a few slots after entry. Free off the cached flows.")
    parser.add_argument("--pools", type=str, default=None, help="flurry-study: run the literal study on a live-captured pool universe (JSON/JSONL of {pool, token_mint, create_ts} from the box absmove.py MODE=capture) instead of the scorer's pools -- route #1, exact study on ShredStream-surfaced pools")
    parser.add_argument("--flows", type=str, default=None, help="flurry-study: run the literal walk-forward (discover/scalp/tail) on slot flows reconstructed LIVE by the box surf.py (JSONL of {pool, token_mint, create_ts, rec}); create_ts = first-seen wall-clock, so the time split is a prospective live walk-forward. No Helius.")
    parser.add_argument("--max-hold", type=int, default=None, help="flurry-study: report the scalp at EVERY slot hold 1..N as its own column (flurry/motif/discovered-rule tables), to see continuation slot by slot. A hold >2 re-exposes to the reactive LP pull, so rugged%% rises -- read it as the hold/pull tradeoff, not a safe scalp.")
    parser.add_argument("--frontrun", action="store_true", help="flurry-study: front-run study -- can an indicator predict a >= --move-pct (default 30%%) up-move early enough to enter <=3 slots before it and exit 1-2 after? Perfect-foresight ceiling + lead-swept z-guarded indicator discovery + enter-on-signal net vs control. Forces --vq (effective reserve basis).")
    parser.add_argument("--vq", action="store_true", help="flurry-study: price on the effective (virtual_quote-aware) reserve basis, matching the live WS stream, instead of the raw vault ratio (which inflates early-life moves into basis artifacts). Needs --fetch to populate vq. Off by default so other modes stay reproducible; --frontrun forces it on.")
    parser.add_argument("--export-gate", metavar="PATH", help="flurry-study --frontrun: write the fitted precursor front-run gate (joint-logit intercept + non-move predictor coefficients + top-k%% eta thresholds) to a JSON the live box runner loads via tradingbot.precursor_gate.")
    parser.add_argument("--no-vq", action="store_true", help="flurry-study --frontrun: force the RAW reserve-ratio basis instead of the effective (vq-aware) one. Overrides --frontrun's automatic vq, so the header reads 'OFF'. NOTE: on a localshred-merge capture vq is already inert (rec['vq'] is never populated), so this only makes the basis explicit -- the numbers are unchanged.")
    parser.add_argument("--no-throttle", action="store_true", help="flurry-study --frontrun: run the discovery + logit gate UNTHROTTLED -- drop the family-wise z correction (z guard 1.96, no 3.0 floor) so every individually-significant rule survives into the logit gate, move features included. Lets it discover everything naturally; decide the exclusions from the evidence.")
    parser.add_argument("--ws-only", action="store_true", help="flurry-study --frontrun --flows: study a capture using ONLY its Helius-WS ('confirmed') per-slot series as a single complete+priced feed -- no shred, no hybrid rebase, no frontier. The processed WS is same-slot + complete, so this is the one-feed architecture (drops ShredStream). Works on the existing two-feed captures (uses just the confirmed half) and on WS-only captures.")
    parser.add_argument("--mid-slot", action="store_true", help="flurry-study --frontrun: the re-fit for front-of-N+1. A complete slot is only knowable once the NEXT slot starts, so a complete-slot gate tops out at front-of-N+2 (feed-lag 2-3). Fit the gate on the PARTIAL current slot at its FIRST tx (the earliest DURING-slot decision) + completed prior slots, predicting N+1 -- served by frontrun_live.StreamingRunner (first-tx-only), so train == serve. Read the holdout AUC + FRONT-RUN GATE net at the mid-slot decision point. Plain single full-feed path (not --ws-only/--hybrid).")
    parser.add_argument("--min-liq-usd", type=float, default=0.0, help="flurry-study --frontrun (and the other flow modes): keep only capture pools whose scorer-recorded liquidity at second 4 (watch_launch_live.liq_usd) is >= this, matching the LIVE bot's model_book_min_liq_usd. A WS-only/LocalShred capture has no liquidity floor, so sub-$5k pools (where the raw reserve ratio is most inflated and genuine moves largest) otherwise swamp the ceiling. 0 = off. Pools with no scorer liq record are dropped. Use 5000 for the honest 'what we can bank' read.")
    parser.add_argument("--keep-artifacts", action="store_true", help="flurry-study --frontrun/--discover/--discover-econ: KEEP reserve-collapse artifact slots (a near-full buy empties the token side for one tx -> the reserve ratio/price explodes). Default drops them entirely (token reserve < 15%% of the pool median, or price-only: price > 1/0.15 x the pool median) so the discover/logit TARGET counts only genuine, tradeable up-moves -- a cap bounds an artifact's effect on the mean but still lets it count as an up-move, so only dropping it stops discovery honing on artifacts. Pass this to restore the old (capped-but-kept) behaviour for comparison.")
    parser.add_argument("--gate-hold", type=int, default=2, help="flurry-study --export-gate: slots to hold after entering front of N+1 (default 2, the paying cell).")
    parser.add_argument("--capture", metavar="PATH", help="localshred-merge: the --capture feature JSONL from run_localshred.py --capture (our decode's per-slot features of each fresh pool)")
    parser.add_argument("--out", metavar="PATH", help="localshred-merge: write the merged --flows JSONL (our features + the Helius price path) here, for flurry-study --frontrun --flows")
    parser.add_argument("--fresh-slots", type=int, default=450, help="capture-fresh: keep a captured pool only if its first captured slot is within this many slots (~450 = ~180 s, the live pool_fresh window) of its creation slot -- i.e. born during the session, caught fresh")
    parser.add_argument("--min-cap-slots", type=int, default=6, help="localshred-merge: skip captured pools with fewer slots than this before the Helius fetch (thin pools rarely yield >=3 Helius-priced slots, so they'd be fetched then dropped -- wasting credits)")
    parser.add_argument("--min-cap-swaps", type=int, default=12, help="localshred-merge: skip captured pools with fewer total buy+sell events than this")
    parser.add_argument("--include-failed", action="store_true", help="flurry-study --frontrun: failed-inclusive refit A/B -- fit/eval the gate on decision features that COUNT reverted txs (what the live shred feed sees; no status in shreds) vs the success-only training view (vault deltas), on the SAME pools. Tests whether the front-run edge survives and whether re-fitting train=serve fixes the live over-firing. Refetches a superset.")
    parser.add_argument("--gate-vis", choices=["full", "direct", "buyer_plus"], default="direct", help="flurry-study --frontrun: the ShredStream visibility to FIT the gate on. The live box decodes only TOP-LEVEL pumpswap swaps, so 'direct' (default) makes the exported gate fire on the box's own view (the vis-experiment showed direct carries ~89%% of the full edge); 'full' reproduces the historical full-flow study. frontrun-verify reads the gate's stamped vis to replay through the same view.")
    parser.add_argument("--vis-experiment", action="store_true", help="flurry-study: refit the front-run gate on each box-reproducible VISIBILITY (full / direct-only / buyer-plus) with the target always the real full-flow move, and compare the precursor gate's holdout edge -- does the box's ShredStream view (top-level pumpswap only; aggregator CPIs invisible) carry the study's edge? Forces --vq; needs --fetch to visibility-tag the rows.")
    parser.add_argument("--entry-window", type=int, nargs="?", const=10, default=None, metavar="W", help="flurry-study: patience study -- keep the safe 1-2 slot hold, but sweep WHICH entry slot (N+1..N+W, default W=10) to take it after a slot-N signal. A pull during the wait = no entry (no loss); reports per-offset EV, best-of-window ceiling, and a causal up-tick-ride rule vs a random-active-slot control, out of sample.")
    parser.add_argument("--samples", type=str, default=None, help="router-direction: the box's router-samples JSONL ({router,sig,pool,payer,data(hex),accounts,keys})")
    parser.add_argument("--routers", type=str, default=None, help="router-direction: fetch labeled samples from Helius for these router program ids (comma-separated, or 'pins' for the built-in top-20). One getTransactionsForAddress per router gives the top-level router ix + the inner pumpswap CPI (true direction) -- no box capture needed")
    parser.add_argument("--routers-file", type=str, default=None, help="router-direction: a JSON {program_id: routed-tx count} (the box's full cpi_routers dump) -- fetch+fit EVERY router (highest volume first, budget-capped) and report volume-weighted coverage of the routed flow")
    parser.add_argument("--per-router", type=int, default=2000, help="router-direction per-router fetch: recent (desc) transactions to read per router, paged in 1000s (~10 credits per 100). Deep enough to see buy AND sell")
    parser.add_argument("--top", type=int, default=0, help="router-direction: only fetch/decode the top-N routers by --routers-file volume (0 = all). The owner's 'top 20 routers'")
    parser.add_argument("--route-rules", type=str, default=None, help="shred-completeness: router_rules.json -- also score how the box's router rules read each successful routed swap on the same pools/slots, and print the share of real swaps decoded with them")
    parser.add_argument("--raw-out", type=str, default=None, help="router-direction fetch: also write EVERY fetched tx (label-free compact record, gzip JSONL) here, for offline decoding")
    parser.add_argument("--diagnose", type=str, default=None, help="router-direction: comma-separated router ids; read ONE page (100 txs, ~10 credits) each and print why its txs were kept/dropped as samples")
    parser.add_argument("--skip-sampled", nargs="?", const="__default__", default=None, help="router-direction --routers-file: skip routers we already hold enough samples for (>=4 buy AND >=4 sell). Default list = the summary shipped in tradingbot/router_data/router_sampled.json; or pass a summary json / samples JSONL path. Only routers that still need samples are fetched")
    parser.add_argument("--decode", action="store_true", help="router-direction: print a per-router decode worksheet (buy vs sell discriminators + the side-encoding rule) as well as the coverage table")
    parser.add_argument("--dump", action="store_true", help="router-direction: print RAW per-router buy/sell example transactions for hand decoding (no rule-fitting), and save all labeled samples to --samples-out")
    parser.add_argument("--samples-out", type=str, default="/data/router-samples.jsonl", help="router-direction --dump: path to write the raw labeled samples JSONL (download and feed in chunks)")
    parser.add_argument("--from-pumpswap", action="store_true", help="router-direction: sample from the pumpswap side (one desc sweep of the pumpswap program, bucketed by top-level router) instead of one fetch per router -- recent, balanced. Spreads thin across routers; prefer --routers-file --top N for deep per-router decode")
    parser.add_argument("--fires", type=str, default=None, help="frontrun-verify: JSON/JSONL of box fires {pool, slot, token_mint?, sym?, m2?, eta?} or a raw box log to grep for 'GATE FIRE' lines")
    parser.add_argument("--pairs", type=str, default=None, help="frontrun-verify: box fires by hand as 'pool:slot[:mint],pool:slot'")
    parser.add_argument("--gate", type=str, default=None, help="frontrun-verify: the exported gate JSON (flurry-study --export-gate)")
    parser.add_argument("--tier", type=str, default="1", help="frontrun-verify: gate tier to check (1 = top 1%%, 2 = top 2%%)")
    parser.add_argument("--pages", type=int, default=None, help="frontrun-verify: history pages to fetch per pool (100 txs each; raise for busy pools so a late fired slot is fetched rather than falsely truncated; default 6)")
    parser.add_argument("--sol-usd", type=float, default=83.0, help="frontrun-fill: SOL price in USD for sizing the trade (default 83)")
    parser.add_argument("--hold", type=int, default=2, help="frontrun-fill: slots to hold (sell at N+k+hold); default 2 = exit N+2, dodges the >=2-slot reactive pull")
    parser.add_argument("--box-exact", action="store_true", help="frontrun-verify: build the ground truth with the box's LUT drop replicated (a direct swap whose pool account is LUT-loaded is dropped, as the box does). If MISMATCH rows vanish under this, the cause is the box's drop_lut (resolve LUTs / LocalShred); if they survive, the gap is deeper than LUT.")
    # frontrun-verify --hybrid reuses the --hybrid flag defined above (flurry-study); both are store_true -> args.hybrid.
    parser.add_argument("--usd", type=float, default=2.0, help="canary-smoke: USD to buy and sell back")
    parser.add_argument("--pool", help="canary-smoke on the direct route: the token's pumpswap pool (else looked up)")
    parser.add_argument("--apply", action="store_true", help="canary-sweep: close the empty accounts (default: list)")
    parser.add_argument("--idea", help="propose: what to build")
    parser.add_argument("--problem", default="", help="propose: what it should fix or improve")
    parser.add_argument("--priority", type=int, default=1, help="propose: 1 (first) to 5 (last)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = Config.from_env()
    logging.getLogger(__name__).info("mode=%s model=%s", cfg.mode, cfg.model)

    if args.command == "dashboard-summary":  # run by the dashboard in a low-priority child process
        import os
        from tradingbot import dashboard
        from tradingbot.journal import Journal
        store = None
        if cfg.launch_live:
            from tradingbot.watcher import WatchStore, watch_db_path
            store = WatchStore(watch_db_path(cfg))
        out = dashboard.build_summary(Journal(cfg.db_path), cfg, store)
        tmp = args.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, default=str)
        os.replace(tmp, args.path)
        return

    if args.command == "report":
        from tradingbot import report
        from tradingbot.journal import Journal
        j = Journal(cfg.db_path)
        if args.all:  # every book, Claude's selection, learning, triage (Claude trading is off since 10-27)
            rep = report.build(j, cfg)
            print(json.dumps(rep, indent=2, default=str) if args.json else report.render(rep))
        else:
            rep = report.build_model(j, cfg)
            print(json.dumps(rep, indent=2, default=str) if args.json else report.render_model(rep))
        return

    if args.command == "health":
        from tradingbot import health
        from tradingbot.journal import Journal
        ok, detail = health.status(Journal(cfg.db_path), cfg)
        print(json.dumps({"ok": ok, **detail}, indent=2))
        raise SystemExit(0 if ok else 1)

    if args.command == "backup":
        from tradingbot.journal import Journal
        if not args.path:
            parser.error("backup needs a destination path")
        Journal(cfg.db_path).backup(args.path)
        print(f"backed up {cfg.db_path} -> {args.path}")
        return

    if args.command == "learn-now":
        from tradingbot.journal import Journal
        from tradingbot.learning.learner import Learner
        print(f"completed {Learner(Journal(cfg.db_path), cfg).step()} review(s)")
        return

    if args.command in ("optimize-now", "settings", "settings-rollback"):
        from tradingbot.journal import Journal
        from tradingbot.learning import versions
        from tradingbot.learning.learner import Learner
        j = Journal(cfg.db_path)
        versions.apply_active(j, cfg)
        if args.command == "optimize-now":
            learner = Learner(j, cfg)
            res = learner.optimizer.run(learner.client)
            learner.record("optimizer", res)
            print(json.dumps({"outcome": res.outcome, "cost_usd": round(res.cost_usd, 3), "result": res.data,
                              "error": res.error}, indent=2, default=str))
        elif args.command == "settings-rollback":
            print(versions.rollback(j, cfg, "manual rollback") or "nothing to roll back")
        else:
            for v in versions.history(j):
                print(f"v{v['id']} [{v['status']}] {v['changes']} :: {v['reason'][:160]}"
                      + (f" :: {v['verdict']}" if v["verdict"] else ""))
        return

    if args.command in ("propose", "proposals"):
        import time as _time

        from tradingbot.journal import Journal
        from tradingbot.learning import versions
        j = Journal(cfg.db_path)
        versions.ensure(j)
        if args.command == "propose":
            if not args.path or not args.idea:
                parser.error('propose needs a title and --idea, e.g. propose "Mention velocity" --idea "..."')
            pid = j.execute(
                "INSERT INTO proposals(ts, source, title, problem, evidence, idea, priority) VALUES(?,?,?,?,?,?,?)",
                (_time.time(), "user", args.path, args.problem or args.idea, "requested by the operator",
                 args.idea, max(1, min(5, args.priority))))
            print(f"queued proposal #{pid} for the engineer (it picks the highest priority first, "
                  f"at most one attempt per {cfg.engineer_interval_h:g}h)")
            return
        for r in j.query("SELECT * FROM proposals ORDER BY status='open' DESC, priority, ts"):
            print(f"#{r['id']} [{r['status']}] p{r['priority']} ({r['source']}) {r['title']}")
            print(f"    idea: {r['idea'][:200]}")
            if r["result"]:
                print(f"    result: {r['result'][:200]}")
        return

    if args.command in ("code-versions", "code-diff", "code-rollback"):
        from tradingbot.learning import codebase
        if args.command == "code-versions":
            running = codebase.running_version(cfg)
            print(f"running now: {running}; active on next start: {codebase.active_version(cfg) or 'image'}")
            if codebase.crashed_versions(cfg):
                print(f"fell back from (crash loop or bad checksum): {codebase.crashed_versions(cfg)}")
            for v in codebase.list_versions(cfg):
                flags = ("ACTIVE " if v["active"] else "") + ("" if v["verified"] else "CHECKSUM-FAILED ")
                print(f"{v['version']} <- {v['parent'] or 'image'} {flags}:: {v.get('title', '')}")
                print(f"    {v.get('summary', '')[:200]}  files: {v.get('files_changed', [])}")
            return
        if not args.path:
            parser.error(f"{args.command} needs a version, e.g. v3 (or 'image' for code-rollback)")
        if args.command == "code-diff":
            print((codebase.code_root(cfg) / "versions" / args.path / "DIFF.patch").read_text())
            return
        target = codebase.switch_to(cfg, args.path)
        print(f"next start will run {target}. Restart to apply, e.g. `fly apps restart <app>`.")
        return

    if args.command == "watch-status":
        from tradingbot.enrich import Enricher
        from tradingbot.watcher import Watcher
        w = Watcher.from_config(cfg)
        print(json.dumps({**w.status(), "enrichment": Enricher.from_config(cfg, w.store, w).status()}, indent=2))
        return

    if args.command == "watch-links":
        import os
        from tradingbot.backups import s3_client
        from tradingbot.watcher import download_links
        client = s3_client()
        if client is None:
            print("no bucket configured (fly storage create); reports are only on the volume under /data/reports")
            return
        print(json.dumps(download_links(client, os.environ["BUCKET_NAME"]), indent=2))
        return

    if args.command == "watch-model-train":
        from tradingbot import model_book
        from tradingbot.watcher import WatchStore, watch_db_path
        store = WatchStore(watch_db_path(cfg))
        out = model_book.train_and_save(store, cfg)
        print(f"model written to {out}" if out else "not enough finished data to train yet")
        if cfg.early_book:
            out = model_book.train_early_and_save(store, cfg)
            print(f"early model written to {out}" if out else "early model: not enough data to train yet")
        if cfg.launch_live:
            from tradingbot import launch_live
            out = launch_live.train_and_save(store, cfg)
            print(f"launch4 model written to {out}" if out else "launch4 model: not enough data to train yet")
            out = launch_live.train_hist_and_save(store, cfg)
            print(f"launch4h model written to {out}" if out else "launch4h model: not enough data to train yet")
            out = launch_live.train_pull_and_save(store, cfg)
            print(f"pull model written to {out}" if out else "pull model: not enough data to train yet")
            out = launch_live.train_rw_and_save(store, cfg)
            print(f"rug-vs-win model written to {out}" if out else "rug-vs-win model: not enough data to train yet")
            if cfg.launch_big_book or cfg.launch_big_mid_book:
                out = launch_live.train_big_and_save(store, cfg)
                print(f"+50% model written to {out}" if out else "+50% model: not enough data to train yet")
        return

    if args.command == "launch-big-train":  # only the +50% model of launchbig40k (retrained every 6 h by the bot)
        from tradingbot import launch_live
        from tradingbot.watcher import WatchStore, watch_db_path
        out = launch_live.train_big_and_save(WatchStore(watch_db_path(cfg)), cfg)
        print(f"+50% model written to {out}" if out else "+50% model: not enough data to train yet")
        return

    if args.command == "launch-rw-train":  # only the rug-vs-win model (light; watch-model-train does all)
        from tradingbot import launch_live
        from tradingbot.watcher import WatchStore, watch_db_path
        out = launch_live.train_rw_and_save(WatchStore(watch_db_path(cfg)), cfg)
        print(f"rug-vs-win model written to {out}" if out else "rug-vs-win model: not enough data to train yet")
        return

    if args.command in ("canary-status", "canary-stop", "canary-start", "canary-one"):  # the real-money launch canary
        from tradingbot import launch_canary
        from tradingbot.engine import CANARY_ALLOW, CANARY_HALT, CANARY_ABORT_ALL
        from tradingbot.journal import Journal
        import time as _time  # a later `import time` in main() makes the bare name local here
        journal = Journal(cfg.db_path)
        if args.command == "canary-stop":
            journal.set_state(CANARY_HALT, repr(_time.time()))
            print("launch canary halted: no new buys (open positions still sell at their pool minute)")
        elif args.command == "canary-start":
            journal.execute("DELETE FROM state WHERE key IN (?, ?, ?)", (CANARY_HALT, CANARY_ALLOW, CANARY_ABORT_ALL))
            print("launch canary kill switch cleared (no trade-at-a-time limit)")
        elif args.command == "canary-one":  # exactly one more buy, then it waits for the next canary-one
            journal.set_state(CANARY_ALLOW, "1")
            journal.execute("DELETE FROM state WHERE key=?", (CANARY_HALT,))
            print("launch canary: one buy allowed, then it waits for canary-one again")
        out = {"status": launch_canary.status(journal, cfg), "calibration": launch_canary.calibration(journal, args.hours)}
        print(json.dumps(out, indent=2, default=str))
        return

    if args.command == "canary-smoke":  # one tiny real buy + sell of a deep token
        if cfg.live_confirm != LIVE_CONFIRM_PHRASE:
            raise SystemExit(f"canary-smoke spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        from tradingbot import launch_canary
        from tradingbot.data.jupiter import Jupiter
        ex = canary_executor(cfg, Jupiter(cfg.slippage_bps))
        if cfg.launch_canary_route == "direct" and not args.path:
            raise SystemExit("canary-smoke on the direct route needs a token with a pumpswap pool: canary-smoke MINT")
        if args.pool and hasattr(ex, "remember_pool"):
            ex.remember_pool(args.path, args.pool)
        print(json.dumps(launch_canary.smoke(ex, args.path or launch_canary.SMOKE_MINT, args.usd), indent=2, default=str))
        return

    if args.command == "canary-feint-smoke":  # one real feint (atomic buy+sell + re-read) on a live pool
        if cfg.live_confirm != LIVE_CONFIRM_PHRASE:
            raise SystemExit(f"canary-feint-smoke spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        if cfg.launch_canary_route != "direct":
            raise SystemExit("the feint is built for the direct route; set TB_LAUNCH_CANARY_ROUTE=direct")
        if not args.path:
            raise SystemExit("canary-feint-smoke needs a token with a pumpswap pool: canary-feint-smoke MINT [--pool POOL]")
        from tradingbot import launch_canary
        from tradingbot.data.jupiter import Jupiter
        ex = canary_executor(cfg, Jupiter(cfg.slippage_bps))
        usd = args.usd if args.usd != 2.0 else cfg.launch_canary_feint_usd  # default to the configured feint size
        print(json.dumps(launch_canary.feint_smoke(ex, args.path, usd, args.pool), indent=2, default=str))
        return

    if args.command == "canary-probe-smoke":  # one real 1-slot probe (buy + next-slot sell) on a live pool, low cost
        if cfg.live_confirm != LIVE_CONFIRM_PHRASE:
            raise SystemExit(f"canary-probe-smoke spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        if cfg.launch_canary_route != "direct":
            raise SystemExit("the 1-slot probe is built for the direct route; set TB_LAUNCH_CANARY_ROUTE=direct")
        if not args.path:
            raise SystemExit("canary-probe-smoke needs a token with a pumpswap pool: canary-probe-smoke MINT [--pool POOL]")
        from tradingbot import launch_canary
        from tradingbot.data.jupiter import Jupiter
        ex = canary_executor(cfg, Jupiter(cfg.slippage_bps))
        usd = args.usd if args.usd != 2.0 else 1.0  # default a low-cost smoke (prove the mechanism before a $25 run)
        print(json.dumps(launch_canary.probe_smoke(ex, args.path, usd, args.pool), indent=2, default=str))
        return

    if args.command == "canary-sweep":  # reclaim rent + dust from the canary's wallets (every rotation wallet on the direct route)
        from tradingbot import model_book as mb
        from tradingbot.data.jupiter import Jupiter
        from tradingbot.journal import Journal
        if args.apply and cfg.live_confirm != LIVE_CONFIRM_PHRASE:
            raise SystemExit(f"canary-sweep --apply signs transactions; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        journal = Journal(cfg.db_path)
        keep = {p.token_address for b in mb.CANARY_BOOKS for p in journal.open_positions(b)}
        ex = canary_executor(cfg, Jupiter(cfg.slippage_bps))  # the rotator (if on) sweeps every wallet; direct sells dust
        print(json.dumps(ex.sweep(keep, apply=args.apply), indent=2, default=str))
        return

    if args.command == "launch-rw24-train":  # the earlier rule's rug-vs-win model (rw24 twin books)
        from tradingbot import launch_live
        from tradingbot.watcher import WatchStore, watch_db_path
        out = launch_live.train_rw24_and_save(WatchStore(watch_db_path(cfg)), cfg)
        print(f"rug-vs-win (24 h rule) model written to {out}" if out else "rug-vs-win (24 h rule) model: not enough data")
        return

    if args.command == "launch-pull-train":  # only the early-pull model (light; watch-model-train does all)
        from tradingbot import launch_live
        from tradingbot.watcher import WatchStore, watch_db_path
        out = launch_live.train_pull_and_save(WatchStore(watch_db_path(cfg)), cfg)
        print(f"pull model written to {out}" if out else "pull model: not enough data to train yet")
        return

    if args.command == "launch-hist-backfill":
        import httpx
        from tradingbot import launch_hist
        from tradingbot.data.helius import Helius
        from tradingbot.watcher import WatchStore, watch_db_path
        helius = Helius(client=httpx.Client(timeout=60))
        helius.min_interval_s = 1 / max(args.rps, 0.1)  # its own process: pace it below what the live scorer needs
        out = launch_hist.backfill(WatchStore(watch_db_path(cfg)), helius, args.days,
                                   args.max_credits or 60_000, progress=lambda i, n, c: print(
                                       f"pools {i}/{n}, credits {c}", file=sys.stderr, flush=True))
        print(json.dumps(out, indent=2))
        return

    if args.command == "history-backfill":  # a year (or --days) of hand-made pumpswap pools on the second Helius key
        import httpx
        from tradingbot import launch_hist
        from tradingbot.data.helius import Helius
        from tradingbot.watcher import WatchStore, hist_db_path
        import os as _os
        import time as _time  # a later `import time` in main() makes the bare name local here
        key = _os.environ.get("HELIUS_BACKFILL_API_KEY", "")
        if not key:
            print("HELIUS_BACKFILL_API_KEY is not set: fly secrets set HELIUS_BACKFILL_API_KEY=<key> -a supermeme "
                  "(the history backfill never uses the live scorer's key)", file=sys.stderr)
            sys.exit(2)
        days = args.days if args.days != 7.0 else 365.0
        store = WatchStore(hist_db_path(cfg))
        added = launch_hist.fill_sol_prices(store, _time.time() - days * 86400 - 86400, _time.time())
        print(f"SOL prices: {added} hours added", file=sys.stderr, flush=True)
        helius = Helius(api_key=key, client=httpx.Client(timeout=60))
        helius.min_interval_s = 1 / max(args.rps if args.rps != 2.0 else 10.0, 0.1)  # its own account: own rate
        if args.pulls_only:
            print(json.dumps(launch_hist.pulls(store, helius, workers=args.workers, progress=lambda i, n, c: print(
                f"pools {i}/{n}, credits {c}", file=sys.stderr, flush=True)), indent=2))
            return
        out = launch_hist.backfill(store, helius, days, args.max_credits or 15_000_000, workers=args.workers,
                                   estimate=args.estimate, seed=7, relist=args.relist, progress=lambda i, n, c: print(
                                       f"pools {i}/{n}, credits {c}", file=sys.stderr, flush=True))
        print(json.dumps(out, indent=2))
        return

    if args.command == "hold-study":
        import httpx as _httpx
        from tradingbot import hold_study
        from tradingbot.data.helius import Helius as _Helius
        from tradingbot.watcher import WatchStore as _WS, watch_db_path as _wdp
        helius = _Helius(client=_httpx.Client(timeout=60))
        helius.min_interval_s = 1 / max(args.rps, 0.1)  # leave the live scorer its share of the plan's rate
        out = hold_study.study(_WS(_wdp(cfg)), helius, args.hours, args.per_band, int(args.max_credits or 40_000),
                               min_liq=args.min_liq, max_liq=args.max_liq or float("inf"),
                               kept_only=args.kept_only)
        print(json.dumps(out, indent=2) if args.json else hold_study.render(out))
        return

    if args.command == "launch-exit-study":
        import httpx
        from tradingbot import launch_live
        from tradingbot.data.helius import Helius
        from tradingbot.watcher import WatchStore, watch_db_path
        helius = Helius(client=httpx.Client(timeout=60))
        helius.min_interval_s = 1 / max(args.rps, 0.1)  # leave the live scorer its share of the plan's rate
        print(json.dumps(launch_live.exit_study(WatchStore(watch_db_path(cfg)), helius, args.hours,
                                                int(args.max_credits or 10_000)), indent=2))
        return

    if args.command == "launch-filter-study":
        import httpx
        from tradingbot import launch_filter
        from tradingbot.data.helius import Helius
        from tradingbot.watcher import WatchStore, watch_db_path
        helius = Helius(client=httpx.Client(timeout=60))
        helius.min_interval_s = 1 / max(args.rps, 0.1)  # leave the live scorer its share of the plan's rate
        print(json.dumps(launch_filter.study(WatchStore(watch_db_path(cfg)), helius, args.hours,
                                             int(args.max_credits or 6_000)), indent=2))
        return

    if args.command == "launch-d12-trades":
        from tradingbot import launch_live
        from tradingbot.watcher import WatchStore, watch_db_path
        out = launch_live.low_decile_trades(WatchStore(watch_db_path(cfg)), args.hours)
        if args.json:
            print(json.dumps(out, indent=2))
            return
        print(f"history-model D1-D2 trades: {out.get('d1_d2_trades', 0)} of {out['pools']} finished liquid pools")
        print(json.dumps(out.get("summary", {}), indent=2))
        print(f"{'created (UTC)':16} {'token':45} {'return %':>9} {'pulled after s':>15} {'risk pct':>9}")
        for t in out["trades"]:
            pulled = "" if t["pulled_after_s"] is None else str(t["pulled_after_s"])
            risk = "" if t["risk_pct"] is None else str(t["risk_pct"])
            print(f"{t['created (UTC)']:16} {t['token']:45} {t['return_pct']:>9} {pulled:>15} {risk:>9}")
        return

    if args.command == "launch-token-age":
        import httpx
        from tradingbot import launch_live
        from tradingbot.data.helius import Helius
        from tradingbot.watcher import WatchStore, watch_db_path
        helius = Helius(client=httpx.Client(timeout=60))
        helius.min_interval_s = 1 / max(args.rps, 0.1)  # leave the live scorer its share of the plan's rate
        print(json.dumps(launch_live.token_age_study(WatchStore(watch_db_path(cfg)), helius, args.hours,
                                                     int(args.max_credits or 8_000)), indent=2))
        return

    if args.command == "launch-early-pulls":
        from tradingbot import launch_live
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(launch_live.early_pull_study(WatchStore(watch_db_path(cfg)), args.hours), indent=2))
        return

    if args.command == "launch-late-entry":
        import httpx
        from tradingbot import launch_live
        from tradingbot.data.helius import Helius
        from tradingbot.watcher import WatchStore, watch_db_path
        helius = Helius(client=httpx.Client(timeout=60))
        helius.min_interval_s = 1 / max(args.rps, 0.1)  # leave the live scorer its share of the plan's rate
        print(json.dumps(launch_live.late_entry_study(WatchStore(watch_db_path(cfg)), helius, args.hours,
                                                      int(args.max_credits or 12_000)), indent=2))
        return

    if args.command == "launch-hist-test":
        from tradingbot import launch_hist
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(launch_hist.test(WatchStore(watch_db_path(cfg)), cfg), indent=2))
        return

    if args.command == "launch-compare":
        from tradingbot import launch_live
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(launch_live.compare(WatchStore(watch_db_path(cfg)), args.hours), indent=2))
        return

    if args.command == "rw-check":  # is the rug-vs-win books' lead real?
        from tradingbot import launch_live
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(launch_live.rw_check(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path),
                                              args.hours or 12), indent=2, default=str))
        return

    if args.command == "market-monitor":  # hourly meme-market measures vs the trailing week
        from tradingbot import market_monitor as mm
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        store = WatchStore(watch_db_path(cfg))
        mm.MarketMonitor(store, Journal(cfg.db_path), cfg).step()  # fill any missing finished hours first
        out = mm.status(store, hours=args.hours or 24, since=cfg.market_monitor_since)
        print(json.dumps(out, indent=2, default=str) if args.json else mm.render(out))
        return
    if args.command == "block-compare":  # is the sized book ahead in every kind of market?
        from tradingbot import launch_live
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        out = launch_live.block_compare(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path), cfg, hours=args.hours)
        print(json.dumps(out, indent=2, default=str) if args.json else launch_live.render_block_compare(out))
        return

    if args.command == "safeguards":  # 11-09: pause / stop rules for bad markets, tested on the year replay
        from tradingbot import safeguards
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path, hist_db_path
        import os as _os
        if not _os.path.exists(hist_db_path(cfg)):
            print("no hist.db: run history-backfill first")
            return
        out = safeguards.study(WatchStore(hist_db_path(cfg)), WatchStore(watch_db_path(cfg)), Journal(cfg.db_path), cfg,
                               fresh=args.fresh)
        print(json.dumps(out, indent=2, default=str) if args.json else safeguards.render(out))
        return

    if args.command == "risk-model":  # risk-adjusted results per book + the best trade size (default whole life)
        from tradingbot import risk_model
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        if args.history:
            from tradingbot import risk_history
            from tradingbot.watcher import hist_db_path
            import os as _os
            hist = WatchStore(hist_db_path(cfg)) if _os.path.exists(hist_db_path(cfg)) else None
            helius = None
            if args.hold15:   # 11-09: minute-15 states for sz's new $25-40k rule, on the live key (backup when out)
                from tradingbot.data.helius import Helius as _H
                helius = _H()
                helius.min_interval_s = 1 / max(args.rps, 0.1)   # leave the live scorer its share of the plan's rate
            out = risk_history.model(hist, WatchStore(watch_db_path(cfg)), Journal(cfg.db_path), cfg,
                                     retrain_h=args.retrain_h, hold15=args.hold15, helius=helius,
                                     max_credits=int(args.max_credits or 80_000),
                                     progress=lambda i, n, c: print(f"minute-15 states {i}/{n}, {c} credits", flush=True))
            print(json.dumps(out, indent=2, default=str) if args.json else risk_history.render(out))
            return
        books = [b.strip() for b in args.books.split(",") if b.strip()] or None
        out = risk_model.model(Journal(cfg.db_path), WatchStore(watch_db_path(cfg)), cfg, books=books, hours=args.hours)
        print(json.dumps(out, indent=2, default=str) if args.json else risk_model.render(out))
        return

    if args.command == "band-compare":  # by-band rug-vs-win books vs the sized book (default since the band books began)
        from tradingbot import band_compare
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        out = band_compare.compare(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path), hours=args.hours)
        print(json.dumps(out, indent=2, default=str) if args.json else band_compare.render(out))
        return

    if args.command == "sz-whatif":  # sz with another book's rule in some bands (real trades at sz's sizes)
        from tradingbot import sz_whatif
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        out = sz_whatif.whatif(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path), hours=args.hours)
        print(json.dumps(out, indent=2, default=str) if args.json else sz_whatif.render(out))
        return

    if args.command == "thin-sizing":  # which trade size suits pools under $15k (rebuilt from the scorer's reads)
        from tradingbot import thin_sizing
        from tradingbot.journal import Journal as _TsJournal
        from tradingbot.watcher import WatchStore as _TsWS, watch_db_path as _ts_wdp
        out = thin_sizing.study(_TsWS(_ts_wdp(cfg)), _TsJournal(cfg.db_path), hours=args.hours)
        print(json.dumps(out, indent=2, default=str) if args.json else thin_sizing.render(out))
        return

    if args.command == "sz-stops":  # stop losses and climb checks on a book's real trades (pool paths from Helius)
        from tradingbot import sz_stops
        from tradingbot.journal import Journal as _SsJournal
        from tradingbot.watcher import WatchStore as _SsWS, watch_db_path as _ss_wdp
        ss_helius = None
        if args.fetch:
            import httpx as _ss_httpx
            from tradingbot.data.helius import Helius as _SsHelius
            ss_helius = _SsHelius(client=_ss_httpx.Client(timeout=60))
            ss_helius.min_interval_s = 1 / max(args.rps, 0.1)  # leave the live scorer its share of the plan's rate
        out = sz_stops.study(_SsWS(_ss_wdp(cfg)), _SsJournal(cfg.db_path), book=args.book, hours=args.hours,
                             helius=ss_helius, max_credits=int(args.max_credits or 15_000))
        print(json.dumps(out, indent=2, default=str) if args.json else sz_stops.render(out))
        return

    if args.command == "pool-tracker":  # live state of every pool the canary has feinted (11-12)
        from tradingbot import dashboard as _pt_dash
        from tradingbot import pool_tracker as _pt
        from tradingbot.journal import Journal as _PtJournal
        from tradingbot.watcher import WatchStore as _PtWS, watch_db_path as _pt_wdp
        _pt_rpc = None if args.no_live else (lambda m, p: _pt_dash._solana_rpc(_pt_dash.WALLETS_RPC, m, p))
        rows, su = _pt.rows_for(_PtJournal(cfg.db_path).query, _PtWS(_pt_wdp(cfg)).query,
                                hours=args.hours or (_pt.TRACK_WINDOW_S / 3600), rpc=_pt_rpc,
                                paths=_pt.load_paths(cfg.db_path))  # the dashboard's persisted peak/conf numbers
        if args.json:
            print(json.dumps({"rows": rows, "summary": _pt.summary(rows), "sol_usd": su}, indent=2, default=str))
        else:
            print(_pt.render(rows, su))
        return

    if args.command == "probe-exits":  # exit rules for the 1-slot probe's survivors (11-12)
        from tradingbot import probe_exits as _pe
        from tradingbot.journal import Journal as _PeJournal
        from tradingbot.watcher import WatchStore as _PeWS, watch_db_path as _pe_wdp
        pe_helius = None
        if args.fetch:
            import httpx as _pe_httpx
            from tradingbot.data.helius import Helius as _PeHelius
            pe_helius = _PeHelius(client=_pe_httpx.Client(timeout=60))  # research (backup) key, never the live key
            pe_helius.min_interval_s = 1 / max(args.rps, 0.1)
        if args.pool:  # reconstruct one pool's reserve path (climb-peak timing / stall before a pull)
            out = _pe.dump_pool(_PeJournal(cfg.db_path), _PeWS(_pe_wdp(cfg)), cfg, args.pool,
                                helius=pe_helius, max_credits=int(args.max_credits or 4_000))
            print(json.dumps(out, indent=2, default=str) if args.json else _pe.render_dump(out))
            return
        out = _pe.study(_PeJournal(cfg.db_path), _PeWS(_pe_wdp(cfg)), cfg=cfg, hours=args.hours or 24.0,
                        helius=pe_helius, max_credits=int(args.max_credits or 8_000))
        print(json.dumps(out, indent=2, default=str) if args.json else _pe.render(out))
        return

    if args.command == "pool-wallets":  # wallet-pattern fingerprint of ruggers vs survivors (11-13)
        from tradingbot import pool_wallets as _pw
        from tradingbot.journal import Journal as _PwJournal
        from tradingbot.watcher import WatchStore as _PwWS, watch_db_path as _pw_wdp
        pw_helius = None
        if args.fetch:
            import httpx as _pw_httpx
            from tradingbot.data.helius import Helius as _PwHelius
            pw_helius = _PwHelius(client=_pw_httpx.Client(timeout=60))  # research (backup) key, never the live key
            pw_helius.min_interval_s = 1 / max(args.rps, 0.1)
        out = _pw.study(_PwJournal(cfg.db_path), _PwWS(_pw_wdp(cfg)), cfg=cfg, hours=args.hours or 720.0,
                        window_s=args.window_s, entry_s=args.entry_s, helius=pw_helius,
                        max_credits=int(args.max_credits or 20_000),
                        progress=lambda i, n, c: print(f"streams {i}/{n}, {c} credits", flush=True))
        print(json.dumps(out, indent=2, default=str) if args.json else _pw.render(out))
        return

    if args.command == "frontrun-verify":  # does the live box fire on the slots the study scored? (feed check)
        from tradingbot import frontrun_verify as _fv
        if not args.gate:
            print("frontrun-verify needs --gate <gate.json>", file=sys.stderr)
            return
        with open(args.gate) as _gh:
            _gate = json.load(_gh)
        if args.fires:
            with open(args.fires) as _fh:
                _fires = _fv.parse_fires(_fh.read())
        elif args.pairs:
            _fires = _fv.parse_pairs(args.pairs)
        else:
            print("frontrun-verify needs --fires <path> or --pairs 'pool:slot,...'", file=sys.stderr)
            return
        _have = None
        if getattr(args, "flows", None):            # ws-only train=serve: ground truth is the box's OWN capture
            from tradingbot import flurry_study as _fv_fs
            _, _have = _fv_fs.flows_from_file(args.flows)   # pool -> {confirmed,...}; verify scores it single-feed, no rebase
        _fvh = None
        if args.fetch:
            import httpx as _fv_httpx
            from tradingbot.data.helius import Helius as _FvHelius
            _fvh = _FvHelius(client=_fv_httpx.Client(timeout=60))  # research (backup) key, never the live key
            _fvh.min_interval_s = 1 / max(args.rps, 0.1)
        _out = _fv.verify(_fires, _gate, helius=_fvh, have=_have, tier=args.tier, flurry_min=args.flurry_min, span_s=args.span_s, max_pages=args.pages, box_exact=args.box_exact, hybrid=args.hybrid, ws_only=getattr(args, "ws_only", False))
        print(json.dumps(_out, indent=2, default=str) if args.json else _fv.render(_out))
        return

    if args.command == "frontrun-fill":  # realistic NET of the live fires, settled on Helius true reserves
        from tradingbot import frontrun_fill as _ff
        if not args.fires:
            print("frontrun-fill needs --fires <path>", file=sys.stderr)
            return
        _ffires = _ff.load_fires(args.fires)
        _hold = args.hold
        if args.gate:
            with open(args.gate) as _gh:
                _hold = int(json.load(_gh).get("hold", args.hold))
        _ffh = None
        if args.fetch:
            import httpx as _ff_httpx
            from tradingbot.data.helius import Helius as _FfHelius
            _ffh = _FfHelius(client=_ff_httpx.Client(timeout=60))  # research (backup) key, never the live key
            _ffh.min_interval_s = 1 / max(args.rps, 0.1)
        _out = _ff.measure(_ffires, gate_hold=_hold, sol_usd=args.sol_usd, helius=_ffh, span_s=args.span_s, max_pages=args.pages)
        print(json.dumps(_out, indent=2, default=str) if args.json else _ff.render(_out))
        return

    if args.command == "localshred-merge":  # our --capture features + Helius price -> flurry-study --flows recs
        from tradingbot import flurry_study as _lm
        import httpx as _lm_httpx
        from tradingbot.data.helius import Helius as _LmHelius
        if not args.capture or not args.out:
            print("localshred-merge needs --capture <in.jsonl> --out <flows.jsonl>", file=sys.stderr)
            return
        _lm_helius = _LmHelius(client=_lm_httpx.Client(timeout=60))   # research (backup) key, never the live key
        _lm_helius.min_interval_s = 1 / max(args.rps, 0.1)
        with open(args.capture) as _cf:
            _cap_rows = [json.loads(x) for x in _cf if x.strip()]
        _lm_prog = lambda i, n, c: print(f"merge {i}/{n}, {c} credits", flush=True)
        _rows, _cr = _lm.merge_capture_flows(_cap_rows, _lm_helius, span_s=args.span_s,
                                             max_credits=int(args.max_credits or 20_000), progress=_lm_prog,
                                             min_cap_slots=args.min_cap_slots, min_cap_swaps=args.min_cap_swaps)
        with open(args.out, "w") as _of:
            for _r in _rows:
                _of.write(json.dumps(_r) + "\n")
        print(f"merged {len(_rows)} of {len(_cap_rows)} captured pools -> {args.out} ({_cr} research credits)")
        return

    if args.command == "capture-fresh":  # keep only pools caught fresh (born during the capture) + stamp vq
        from tradingbot import fresh_filter as _ff
        import httpx as _ff_httpx
        from tradingbot.data.helius import Helius as _FfHelius
        if not args.capture or not args.out:
            print("capture-fresh needs --capture <wscap_pertx.jsonl> --out <wscap_fresh.jsonl>", file=sys.stderr)
            sys.exit(2)
        with open(args.capture) as _ffc:
            _ff_rows = [json.loads(x) for x in _ffc if x.strip()]
        _ff_helius = _FfHelius(client=_ff_httpx.Client(timeout=60))   # research (backup) key, never the live key
        _ff_helius.min_interval_s = 1 / max(args.rps, 0.1)
        _ff_prog = lambda i, n, s: print(f"fresh {s['fresh']}/{i} of {n}, {s['credits']} credits",
                                         file=sys.stderr, flush=True)
        _ff_kept, _ff_stats = _ff.fresh_filter(_ff_rows, _ff_helius, fresh_slots=args.fresh_slots,
                                               max_credits=int(args.max_credits or 0), progress=_ff_prog)
        with open(args.out, "w") as _ffo:
            for _r in _ff_kept:
                _ffo.write(json.dumps(_r) + "\n")
        print(f"kept {len(_ff_kept)} fresh of {_ff_stats['total']} pools -> {args.out} "
              f"(stale {_ff_stats['stale']}, unreadable {_ff_stats['unreadable']}, "
              f"{_ff_stats['credits']} research credits{', capped' if _ff_stats['capped'] else ''})")
        return

    if args.command == "shred-completeness":  # the box's LocalShred capture vs Helius truth: what we have vs miss
        from tradingbot import shred_completeness as _sc
        import httpx as _sc_httpx
        from tradingbot.data.helius import Helius as _ScHelius
        _cap_path = args.capture or "/root/ls/ourfeed.jsonl"
        try:
            _sc_rows = _sc.load_capture(_cap_path)
        except FileNotFoundError:
            print(f"shred-completeness: no capture at {_cap_path} "
                  f"(pass --capture <file>; the box writes it to /root/ls/ourfeed.jsonl)", file=sys.stderr)
            sys.exit(2)
        if not _sc_rows:
            print(f"shred-completeness: {_cap_path} has no capture records", file=sys.stderr)
            sys.exit(2)
        _sc_helius = _ScHelius(client=_sc_httpx.Client(timeout=60))   # research (backup) key, never the live key
        _sc_helius.min_interval_s = 1 / max(args.rps, 0.1)
        _sc_prog = lambda i, n, c: print(f"compare {i}/{n}, {c} credits", file=sys.stderr, flush=True)
        _sc_cls = None
        if args.route_rules:
            from tradingbot import route_rules as _sc_rr
            _sc_cls, _sc_n = _sc_rr.make_classifier(args.route_rules)
            print(f"shred-completeness: {_sc_n} router rules from {args.route_rules}", file=sys.stderr)
        _sc_out = _sc.study(_sc_rows, _sc_helius, max_credits=int(args.max_credits or 40_000),
                            span_s=args.span_s, max_pools=args.max_pools, progress=_sc_prog,
                            route_classify=_sc_cls)
        print(json.dumps(_sc_out, indent=2) if args.json else _sc.render(_sc_out))
        return

    if args.command == "symbol-flip":  # the gate-level number: does the pre-LUT box fire where the direct view does? (11-23)
        from tradingbot import symbol_flip as _sf
        from tradingbot import precursor_gate as _sf_pg
        import httpx as _sf_httpx
        from tradingbot.data.helius import Helius as _SfHelius
        if not args.gate:
            print("symbol-flip: pass --gate <gate.json> (the hybrid gate, flurry-study --export-gate)", file=sys.stderr)
            sys.exit(2)
        _sf_gate = _sf_pg.load(args.gate)
        _cap_path = args.capture or "/root/ls/ourfeed.jsonl"
        try:
            _sf_rows = _sf.load_capture(_cap_path)
        except FileNotFoundError:
            print(f"symbol-flip: no capture at {_cap_path} (pass --capture <file>)", file=sys.stderr)
            sys.exit(2)
        if not _sf_rows:
            print(f"symbol-flip: {_cap_path} has no capture records", file=sys.stderr)
            sys.exit(2)
        _sf_helius = _SfHelius(client=_sf_httpx.Client(timeout=60))   # research (backup) key, never the live key
        _sf_helius.min_interval_s = 1 / max(args.rps, 0.1)
        _sf_prog = lambda i, n, c: print(f"flip {i}/{n}, {c} credits", file=sys.stderr, flush=True)
        _sf_out = _sf.study(_sf_rows, _sf_helius, _sf_gate, tier=str(args.tier),
                            max_credits=int(args.max_credits or 40_000), span_s=args.span_s,
                            max_pools=args.max_pools, progress=_sf_prog)
        print(json.dumps(_sf_out, indent=2) if args.json else _sf.render(_sf_out))
        return

    if args.command == "router-direction":  # per-router buy/sell rule from Helius-truth labels (11-27)
        from tradingbot import router_direction as _rd
        import httpx as _rd_httpx
        from tradingbot.data.helius import Helius as _RdHelius
        if args.diagnose:                                         # why does a router yield no samples? ~10 credits each
            _h = _RdHelius(client=_rd_httpx.Client(timeout=60))   # research (backup) key, never the live key
            _h.min_interval_s = 1 / max(args.rps, 0.1)
            for _prog in [p.strip() for p in args.diagnose.split(",") if p.strip()]:
                print(json.dumps(_rd.diagnose_router(_h, _prog)), flush=True)
            print(f"credits used: {_h.credits_used}", file=sys.stderr)
            return
        if args.from_pumpswap or args.routers or args.routers_file:   # fetch mode: labeled samples from Helius
            _counts = None
            _raw_f = None
            if args.routers_file:                                 # the box's routed-volume tally = coverage denominator
                with open(args.routers_file) as f:
                    _counts = {str(k): int(v) for k, v in json.load(f).items()}
            _h = _RdHelius(client=_rd_httpx.Client(timeout=60))   # research (backup) key, never the live key
            _h.min_interval_s = 1 / max(args.rps, 0.1)
            _cap = int(args.max_credits or 50_000)
            if args.from_pumpswap:                                # recent, balanced, volume-weighted: one pumpswap sweep
                _pages = max(1, _cap // 100)                      # ~100 credits per 1000-tx page
                _rd_samples = _rd.fetch_pumpswap_routed(_h, max_pages=_pages, per_page=1000)
                print(f"pumpswap-side sweep: {len(_rd_samples)} routed samples over "
                      f"{len({s['router'] for s in _rd_samples})} routers, {_h.credits_used} credits",
                      file=sys.stderr, flush=True)
            else:
                if args.routers_file and args.skip_sampled:      # top-up: only routers still missing samples
                    _sp = _rd.SAMPLED_PATH if args.skip_sampled == "__default__" else args.skip_sampled
                    _have = _rd.load_sampled(_sp)
                    _progs = _rd.routers_to_fetch(_counts, _have)
                    print(f"skip-sampled: {len(_have)} routers already held; {len(_progs)} still need samples",
                          file=sys.stderr, flush=True)
                elif args.routers_file:
                    _progs = sorted(_counts, key=lambda p: -_counts[p])  # highest routed volume first
                elif args.routers == "pins":
                    _progs = sorted(_rd.PIN_TARGETS)
                else:
                    _progs = [p.strip() for p in args.routers.split(",") if p.strip()]
                if args.top > 0:
                    _progs = _progs[:args.top]                    # the owner's "top N routers"
                _pp = min(1000, max(1, int(args.per_router)))     # per-page follows --per-router (budget spread)
                _pages = max(1, (int(args.per_router) + _pp - 1) // _pp)
                _rd_samples, _fetched = [], 0
                if args.raw_out:
                    import gzip as _rd_gz
                    _raw_f = _rd_gz.open(args.raw_out, "at")
                for _prog in _progs:
                    if _h.credits_used >= _cap:
                        print(f"credit cap {_cap} reached after {_fetched}/{len(_progs)} routers", file=sys.stderr)
                        break
                    _sink = (lambda r: _raw_f.write(json.dumps(r, separators=(",", ":")) + "\n")) if _raw_f else None
                    _rd_samples += _rd.fetch_router_samples(_h, _prog, pages=_pages, per_page=_pp, raw_sink=_sink)
                    if _raw_f:
                        _raw_f.flush()
                    _fetched += 1
                    if _fetched % 10 == 0:
                        print(f"fetched {_fetched}/{len(_progs)} routers, {_h.credits_used} credits",
                              file=sys.stderr, flush=True)
            if args.raw_out and _raw_f:
                _raw_f.close()
                print(f"wrote raw txs to {args.raw_out}", file=sys.stderr)
            if args.dump:                                         # raw examples for hand decoding + full JSONL
                _seen_sigs: set = set()
                _uniq = []
                for s in _rd_samples:                             # a tx routed through 2 fetched routers: keep once
                    if s.get("sig") and (s["router"], s["sig"]) in _seen_sigs:
                        continue
                    _seen_sigs.add((s["router"], s.get("sig")))
                    _uniq.append(s)
                _rd_samples = _uniq
                with open(args.samples_out, "w") as f:
                    for s in _rd_samples:
                        f.write(json.dumps({"router": s["router"], "sig": s.get("sig"), "pool": s.get("pool"),
                                            "payer": s.get("payer"), "token": s.get("token"),
                                            "data": s["data"].hex(), "accounts": s["accounts"],
                                            "keys": s["keys"], "dir": s["dir"]}) + "\n")
                print(f"wrote {len(_rd_samples)} labeled samples to {args.samples_out}", file=sys.stderr)
                print(_rd.dump_examples(_rd_samples, counts=_counts))
                return
            if args.decode:
                print(_rd.decode_report(_rd_samples, counts=_counts))
            _rd_out = _rd.study(_rd_samples, counts=_counts, seed=0)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(_rd_out["accepted"], f, indent=2)
                print(f"wrote {_rd_out['n_accepted']} validated rules to {args.out}", file=sys.stderr)
            print(json.dumps(_rd_out, indent=2, default=str) if args.json else _rd.render(_rd_out))
            return
        _samp_path = args.samples or "/root/ls/router-samples.jsonl"
        try:
            _rd_samples = _rd.load_samples(_samp_path)
        except FileNotFoundError:
            print(f"router-direction: no samples at {_samp_path} (pass --samples <file>; the box writes router-samples.jsonl)", file=sys.stderr)
            sys.exit(2)
        if not _rd_samples:
            print(f"router-direction: {_samp_path} has no samples", file=sys.stderr)
            sys.exit(2)
        _cache_path = _samp_path + ".labels.json"   # {sig: "buy"/"sell"} so reruns are free
        try:
            with open(_cache_path) as f:
                _labels = json.load(f)
        except (OSError, ValueError):
            _labels = {}
        if args.fetch:
            _h = _RdHelius(client=_rd_httpx.Client(timeout=60))   # research (backup) key, never the live key
            _h.min_interval_s = 1 / max(args.rps, 0.1)
            _cap = int(args.max_credits or 20_000)
            _spent = _n = 0
            for s in _rd_samples:
                sig = s.get("sig")
                if not sig:
                    continue
                if sig in _labels:
                    s["dir"] = _labels[sig]
                    continue
                if _spent + _rd.GETTX_CREDITS > _cap:
                    break
                d = _rd.fetch_label(_h, sig, s.get("pool"))
                _spent += _rd.GETTX_CREDITS
                _n += 1
                if d:
                    _labels[sig] = d
                    s["dir"] = d
                if _n % 50 == 0:
                    print(f"labeled {_n}, {_spent} credits", file=sys.stderr, flush=True)
            try:
                with open(_cache_path, "w") as f:
                    json.dump(_labels, f)
            except OSError:
                pass
        else:
            for s in _rd_samples:
                if s.get("sig") in _labels:
                    s["dir"] = _labels[s["sig"]]
        _rd_out = _rd.study(_rd_samples, seed=0)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(_rd_out["accepted"], f, indent=2)
            print(f"wrote {_rd_out['n_accepted']} validated rules to {args.out}", file=sys.stderr)
        print(json.dumps(_rd_out, indent=2, default=str) if args.json else _rd.render(_rd_out))
        return

    if args.command == "flurry-study":  # slot-level buy-flurry continuation for the reactive-backrun scalp (11-17)
        from tradingbot import flurry_study as _fl
        _fl.HYBRID = bool(getattr(args, "hybrid", False))   # box live arch: shred current slot + Helius-rebased history
        _fl.WS_ONLY = bool(getattr(args, "ws_only", False))   # WS-only capture: the confirmed series is the single feed
        _fl.MID_SLOT = bool(getattr(args, "mid_slot", False))  # re-fit: decide DURING N on the first-tx partial (front of N+1)
        if (args.vq or args.frontrun or args.vis_experiment) and not getattr(args, "no_vq", False):   # effective (vq-aware) reserve basis; frontrun/experiment need it (--no-vq forces raw)
            _fl.USE_VQ = True
        # drop reserve-collapse artifact slots entirely (not just cap their return) so discover/logit fit on genuine
        # moves; on for the discovery modes by default, --keep-artifacts restores the old behaviour for comparison.
        _fl.DROP_ARTIFACTS = bool(args.frontrun or args.discover or args.discover_econ) and not args.keep_artifacts
        from tradingbot.journal import Journal as _FlJournal
        from tradingbot.watcher import WatchStore as _FlWS, watch_db_path as _fl_wdp
        fl_helius = None
        if args.fetch:
            import httpx as _fl_httpx
            from tradingbot.data.helius import Helius as _FlHelius
            fl_helius = _FlHelius(client=_fl_httpx.Client(timeout=60))  # research (backup) key, never the live key
            fl_helius.min_interval_s = 1 / max(args.rps, 0.1)
        prog = lambda i, n, c: print(f"flows {i}/{n}, {c} credits", flush=True)
        if args.max_hold and args.max_hold >= 1:   # scalp at every slot hold 1..N (its own column in each table)
            _fl.HOLDS = _fl.MOTIF_HOLDS = tuple(range(1, args.max_hold + 1))
        _tp = dict(target=args.target, move_pct=args.move_pct, horizon=args.horizon)
        # route #1: run the literal study on the pool universe ShredStream surfaced live (box MODE=capture),
        # or on slot flows reconstructed live by the box surf.py (prospective walk-forward, no Helius).
        _pools_meta = _fl.metas_from_pools(args.pools) if args.pools else None
        _flows_have = None
        if args.flows:
            _pools_meta, _flows_have = _fl.flows_from_file(args.flows)
            print(f"live flows from {args.flows}: {len(_pools_meta)} pools (prospective walk-forward by first-seen)", flush=True)
        elif _pools_meta is not None:
            print(f"pool universe from {args.pools}: {len(_pools_meta)} pools", flush=True)
        _mp = dict(metas=_pools_meta, have=_flows_have)
        if args.entry_window is not None:  # patience study: sweep the entry slot in N+1..N+W, keep the 1-2 slot hold
            import time as _fl_time
            from tradingbot import pool_tracker as _fl_pt
            _ws = _FlWS(_fl_wdp(cfg))
            if _pools_meta is not None:
                _metas = _pools_meta
            else:
                _since = _fl_time.time() - (args.hours or 720.0) * 3600
                _scorer = _fl_pt.scorer_rows(_ws.query, _since)
                _metas = [{"pool": p, "token_mint": (_scorer[p] or {}).get("token_mint"),
                           "create_ts": (_scorer[p] or {}).get("create_ts")}
                          for p in _scorer if (_scorer[p] or {}).get("token_mint") and (_scorer[p] or {}).get("create_ts")]
            _have = _flows_have if _flows_have is not None else _fl.fetch_flows(_ws, fl_helius, _metas, args.span_s, int(args.max_credits or 0), prog)[0]
            _tgt = _fl._tgt(args.target, args.flurry_min, args.horizon, args.move_pct)
            if args.discover:   # discover version: auto rule bank on train, z-guarded on the holdout
                out = _fl.entry_window_discover(_have, _metas, _tgt, args.cost_pct / 100.0, window=args.entry_window)
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_entry_window_discover(out))
            else:
                out = _fl.entry_window(_have, _metas, _tgt, args.cost_pct / 100.0, window=args.entry_window)
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_entry_window(out))
            return
        if args.tail or args.discover_econ or args.creation or args.slot0 or args.multibuy or args.frontrun or args.vis_experiment:  # all read flows the same way
            import time as _fl_time
            from tradingbot import pool_tracker as _fl_pt
            _ws = _FlWS(_fl_wdp(cfg))
            if _pools_meta is not None:
                _metas = _pools_meta
            else:
                _now = _fl_time.time()
                _since = _now - (args.hours or 720.0) * 3600
                _scorer = _fl_pt.scorer_rows(_ws.query, _since)
                _metas = [{"pool": p, "token_mint": (_scorer[p] or {}).get("token_mint"),
                           "create_ts": (_scorer[p] or {}).get("create_ts")}
                          for p in _scorer if (_scorer[p] or {}).get("token_mint") and (_scorer[p] or {}).get("create_ts")]
            if _flows_have is not None:
                _have = _flows_have
            else:
                # a restricted-visibility gate fit (frontrun --gate-vis direct/buyer_plus) needs the per-trade
                # direct/aggregator tag, same as the vis-experiment -- else slot_flow(vis=) reads an empty view.
                _incl_failed = bool(args.frontrun and args.include_failed)
                _with_vis = args.vis_experiment or (args.frontrun and args.gate_vis != "full") or _incl_failed
                _have, _cr = _fl.fetch_flows(_ws, fl_helius, _metas, args.span_s,
                                             int(args.max_credits or 0), prog, with_vis=_with_vis,
                                             include_failed=_incl_failed)
            if args.min_liq_usd and args.min_liq_usd > 0:   # match the live bot's $5k floor (peek5 carries no liq field)
                _extra = None
                if fl_helius is not None:                   # --fetch: backfill decision-time liq for pools the scorer missed
                    _scmap = _fl.scorer_liq(_ws.query, [m["pool"] for m in _metas])
                    _unknown = [m["pool"] for m in _metas if m["pool"] not in _scmap]
                    _mints = {m["pool"]: m.get("token_mint") for m in _metas}
                    print(f"liquidity floor: {len(_unknown)} pools lack a scorer record -- reading seed liq from Helius "
                          f"(research key, ~10 credits/pool, cached)...", flush=True)
                    _extra = _fl.backfill_liq(fl_helius, _ws, _unknown, _mints, args.sol_usd,
                                              int(args.max_credits or 0), prog)
                _metas, _have, _lqs = _fl.filter_by_liq(_ws.query, _metas, _have, args.min_liq_usd, extra_liq=_extra)
                if _lqs.get("error"):
                    print(f"liquidity floor ${args.min_liq_usd:,.0f}: no watch_launch_live join ({_lqs['error']}) -- NOT filtered", flush=True)
                else:
                    print(f"liquidity floor ${_lqs['floor']:,.0f}: kept {_lqs['kept']}/{_lqs['total']} pools "
                          f"(liq from scorer {_lqs['from_scorer']} + fetch {_lqs['from_fetch']}; "
                          f"{_lqs['dropped_below']} below floor, {_lqs['dropped_unknown']} still no liq record)", flush=True)
            _tgt = _fl._tgt(args.target, args.flurry_min, args.horizon, args.move_pct)
            if args.multibuy:  # band flurries by buy count, enter the slot after, sweep the hold 1-10
                out = _fl.multibuy_study(_have, _metas, args.cost_pct / 100.0)
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_multibuy(out))
            elif args.slot0:  # land in the creation slot, sweep the hold: how long to EV+?
                out = _fl.slot0_study(_have, _metas, args.cost_pct / 100.0)
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_slot0(out))
            elif args.creation:  # reactive first-buy backrun: enter front of first-buy+1, hold k slots
                out = _fl.creation_study(_have, _metas, args.cost_pct / 100.0)
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_creation(out))
            elif args.discover_econ:  # rules SELECTED by scalp economics (mean OR median), not lift
                out = _fl.discover_econ(_have, _metas, _tgt, args.cost_pct / 100.0)
                out["target"] = _tgt
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_econ(out))
            elif args.vis_experiment:   # does the box's ShredStream visibility carry the front-run edge?
                _mv = args.move_pct / 100.0 if abs(args.move_pct - 1.0) > 1e-9 else _fl.FRONT_MOVE
                out = _fl.vis_experiment(_have, _metas, move=_mv, cost=args.cost_pct / 100.0,
                                         flurry_min=args.flurry_min, hold=args.gate_hold)
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_vis_experiment(out))
            elif args.frontrun:   # front-run a >= move up-move: ceiling + lead-swept discovery + enter-on-signal net
                _mv = args.move_pct / 100.0 if abs(args.move_pct - 1.0) > 1e-9 else _fl.FRONT_MOVE
                if args.include_failed:   # failed-inclusive refit A/B: success-only vs failed-inclusive (same pools)
                    _o_s = _fl.frontrun(_have, _metas, move=_mv, cost=args.cost_pct / 100.0,
                                        flurry_min=args.flurry_min, vis=args.gate_vis, include_failed=False,
                                        no_throttle=args.no_throttle, tail=args.tail)
                    _o_f = _fl.frontrun(_have, _metas, move=_mv, cost=args.cost_pct / 100.0,
                                        flurry_min=args.flurry_min, vis=args.gate_vis, include_failed=True,
                                        no_throttle=args.no_throttle, tail=args.tail)
                    print(json.dumps({"success_only": _o_s, "failed_inclusive": _o_f}, indent=2, default=str)
                          if args.json else _fl.render_frontrun_failed_compare(_o_s, _o_f))
                    return
                out = _fl.frontrun(_have, _metas, move=_mv, cost=args.cost_pct / 100.0,
                                   flurry_min=args.flurry_min, vis=args.gate_vis, no_throttle=args.no_throttle,
                                   tail=args.tail)
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_frontrun(out))
                if args.export_gate:   # write the fitted precursor gate for the live box runner (precursor_gate.load)
                    _gate = _fl.export_gate(out, hold=args.gate_hold)
                    if _gate is None:
                        print("  (no precursor predictor survived on the %s view -- nothing to export)" % args.gate_vis)
                    else:
                        with open(args.export_gate, "w") as _gh:
                            json.dump(_gate, _gh, indent=2)
                        print("  gate -> %s  (vis %s, %d predictors, thresholds %s, hold %d)"
                              % (args.export_gate, _gate["vis"], len(_gate["predictors"]),
                                 ",".join(sorted(_gate["threshold"])), _gate["hold"]))
            else:
                out = _fl.tail_report(_have, _metas, _tgt, args.cost_pct / 100.0, hold=args.horizon)
                print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_tail(out))
            return
        if args.scalp:  # the after-the-flurry continuation scalp table (secondary)
            out = _fl.study(_FlJournal(cfg.db_path), _FlWS(_fl_wdp(cfg)), cfg=cfg, hours=args.hours or 720.0,
                            span_s=args.span_s, cost_pct=args.cost_pct, probed_only=args.probed_only,
                            helius=fl_helius, max_credits=int(args.max_credits or 20_000), progress=prog, **_tp, **_mp)
            print(json.dumps(out, indent=2, default=str) if args.json else _fl.render(out))
            return
        if args.discover:  # data-driven: let the auto rule bank + motifs name the precursor, holdout-confirmed
            out = _fl.discover(_FlJournal(cfg.db_path), _FlWS(_fl_wdp(cfg)), cfg=cfg, hours=args.hours or 720.0,
                               span_s=args.span_s, flurry_min=args.flurry_min, probed_only=args.probed_only,
                               combos=args.combos,
                               helius=fl_helius, max_credits=int(args.max_credits or 20_000), progress=prog, **_tp, **_mp)
            print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_discover(out))
            return
        out = _fl.characterize(_FlJournal(cfg.db_path), _FlWS(_fl_wdp(cfg)), cfg=cfg, hours=args.hours or 720.0,
                               span_s=args.span_s, flurry_min=args.flurry_min, probed_only=args.probed_only,
                               helius=fl_helius, max_credits=int(args.max_credits or 20_000), progress=prog, **_tp, **_mp)
        print(json.dumps(out, indent=2, default=str) if args.json else _fl.render_char(out))
        return

    if args.command == "canary-trades":  # every real canary trade + the pool path after our buy (11-10)
        import os as _ct_os
        from tradingbot import canary_trades
        from tradingbot.journal import Journal as _CtJournal
        from tradingbot.watcher import WatchStore as _CtWS, watch_db_path as _ct_wdp
        ct_helius = None
        if args.fetch:
            import httpx as _ct_httpx
            from tradingbot.data.helius import Helius as _CtHelius
            ct_helius = _CtHelius(client=_ct_httpx.Client(timeout=60))  # research (backup) key, never the live key
            ct_helius.min_interval_s = 1 / max(args.rps, 0.1)
        out = canary_trades.study(_CtWS(_ct_wdp(cfg)), _CtJournal(cfg.db_path), helius=ct_helius,
                                  max_credits=int(args.max_credits or 5_000))
        dest = args.path or _ct_os.path.join(_ct_os.path.dirname(_ct_os.path.abspath(cfg.db_path)), "canary_trades.json")
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(json.dumps(out, indent=2, default=str) if args.json else canary_trades.render(out))
        print(f"\nfull export written to {dest}")
        return

    if args.command == "canary-slots":  # buy slot vs the operator's LP pull slot (11-12)
        import httpx as _cs_httpx
        from tradingbot import canary_slots as _cs
        from tradingbot.data.helius import Helius as _CsHelius
        from tradingbot.journal import Journal as _CsJournal
        from tradingbot.watcher import WatchStore as _CsWS, watch_db_path as _cs_wdp
        cs_helius = _CsHelius(client=_cs_httpx.Client(timeout=60))  # research (backup) key, never the live key
        cs_helius.min_interval_s = 1 / max(args.rps, 0.1)
        out = _cs.slots(_CsWS(_cs_wdp(cfg)), _CsJournal(cfg.db_path), cs_helius, cfg,
                        max_credits=int(args.max_credits or 8_000))
        print(json.dumps(out, indent=2, default=str) if args.json else _cs.render(out))
        return

    if args.command == "canary-package":  # the canary's proposed rule set replayed over the year (11-10)
        import os as _os
        from tradingbot import canary_package
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, hist_db_path, watch_db_path
        if not _os.path.exists(hist_db_path(cfg)):
            print("CANARY PACKAGE\n  no hist.db: run history-backfill first")
            return
        helius = None
        if args.fetch:   # minute-10 states of the 25k10 rule's pools, on the live key (backup when out)
            from tradingbot.data.helius import Helius as _H
            helius = _H()
            helius.min_interval_s = 1 / max(args.rps, 0.1)
        out = canary_package.study(WatchStore(hist_db_path(cfg)), WatchStore(watch_db_path(cfg)), Journal(cfg.db_path),
                                   cfg, fresh=args.fresh, helius=helius, max_credits=int(args.max_credits or 20_000),
                                   progress=lambda i, n, c: print(f"minute-10 states {i}/{n}, {c} credits", flush=True))
        print(canary_package.to_json(out) if args.json else canary_package.render(out))
        return

    if args.command == "dd-limits":  # daily / trailing drawdown limits on a book's life and on the year replay
        from tradingbot import dd_limits
        from tradingbot.journal import Journal as _DdJournal
        from tradingbot.watcher import WatchStore as _DdWS, watch_db_path as _dd_wdp, hist_db_path as _dd_hdp
        import os as _dd_os
        dd_year = args.year and _dd_os.path.exists(_dd_hdp(cfg))
        out = dd_limits.study(_DdJournal(cfg.db_path), book=args.book, hours=args.hours, year=dd_year,
                              hist_store=_DdWS(_dd_hdp(cfg)) if dd_year else None,
                              watch_store=_DdWS(_dd_wdp(cfg)) if dd_year else None, cfg=cfg, fresh=args.fresh)
        if args.year and not dd_year:
            out["year"] = {"error": "no hist.db: run history-backfill first"}
        print(dd_limits.to_json(out) if args.json else dd_limits.render(out))
        return

    if args.command == "flag-tracker":  # what each market flag has meant for each running book (read only)
        from tradingbot import flag_tracker
        from tradingbot.journal import Journal as _FtJournal
        from tradingbot.watcher import WatchStore as _FtWS, watch_db_path as _ft_wdp
        out = flag_tracker.study(_FtWS(_ft_wdp(cfg)), _FtJournal(cfg.db_path), cfg)
        print(json.dumps(out, indent=2, default=str) if args.json else flag_tracker.render(out, args.book))
        return

    if args.command == "missed-signals":  # signals a launch book was sent but didn't buy, and why
        from tradingbot import missed_signals
        from tradingbot.journal import Journal as _MsJournal
        from tradingbot.watcher import WatchStore as _MsWS, watch_db_path as _ms_wdp
        out = missed_signals.study(_MsWS(_ms_wdp(cfg)), _MsJournal(cfg.db_path), book=args.book,
                                   hours=args.hours or 24.0)
        print(json.dumps(out, indent=2, default=str) if args.json else missed_signals.render(out))
        return

    if args.command == "band-prize":  # is there a prize in one band if filtered better? (default $40-80k, whole life)
        from tradingbot import band_prize
        from tradingbot.watcher import WatchStore as _WS, watch_db_path as _wdp
        lo = args.min_liq if args.min_liq != 25_000.0 else band_prize.MIN_LIQ
        bp_helius = None
        if args.fetch:
            import httpx as _bp_httpx
            from tradingbot.data.helius import Helius as _BpHelius
            bp_helius = _BpHelius(client=_bp_httpx.Client(timeout=60))
            bp_helius.min_interval_s = 1 / max(args.rps, 0.1)  # leave the live scorer its share of the plan's rate
        out = band_prize.study(_WS(_wdp(cfg)), hours=args.hours, min_liq=lo,
                               max_liq=args.max_liq or band_prize.MAX_LIQ, helius=bp_helius,
                               max_credits=int(args.max_credits or 20_000))
        print(json.dumps(out, indent=2, default=str) if args.json else band_prize.render(out))
        return
    if args.command == "band-rw-study":  # rug-vs-win trained per pool-size band, walk-forward (default whole life)
        from tradingbot import band_rw
        from tradingbot.watcher import WatchStore, watch_db_path
        out = band_rw.study(WatchStore(watch_db_path(cfg)), hours=args.hours)
        print(json.dumps(out, indent=2, default=str) if args.json else band_rw.render(out))
        return

    if args.command == "rw-bands":  # rug-vs-win and sz by pool size, 8 liquidity bands (default whole life)
        from tradingbot import pool_bands
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        out = pool_bands.study(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path), hours=args.hours)
        print(json.dumps(out, indent=2, default=str) if args.json else pool_bands.render(out))
        return

    if args.command == "canary-rug-check":  # canary rugs: a bad streak, or did our real buys set them off?
        from tradingbot import canary_rugs
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(canary_rugs.rug_check(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path), cfg,
                                               hours=args.hours), indent=2, default=str))
        return

    if args.command == "size-study":  # would sizing by pool liquidity earn more without losing the big winners?
        from tradingbot import launch_live
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(launch_live.size_study(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path),
                                                hours=args.hours), indent=2, default=str))
        return

    if args.command == "fill-gap-study":  # where real launch fills fall short of the scorer's ideal trade
        from tradingbot import launch_live
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(launch_live.fill_gap_study(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path),
                                                    hours=args.hours), indent=2, default=str))
        return

    if args.command == "early-rw-study":  # a rug-vs-win score trained on the early books' own entries
        from tradingbot import early_rw
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(early_rw.study(WatchStore(watch_db_path(cfg)), cfg, args.hours), indent=2, default=str))
        return

    if args.command == "early-rw-check":  # would the rug-vs-win filter help the early books?
        from tradingbot import launch_live
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(launch_live.early_rw_check(WatchStore(watch_db_path(cfg)), Journal(cfg.db_path), cfg,
                                                    args.hours), indent=2, default=str))
        return

    if args.command == "launch-status":
        from tradingbot import launch_live
        from tradingbot.watcher import WatchStore, watch_db_path
        from tradingbot.journal import Journal
        print(json.dumps(launch_live.status(WatchStore(watch_db_path(cfg)), cfg, args.hours or 24,
                                            journal=Journal(cfg.db_path)), indent=2, default=str))
        return

    if args.command == "early-status":
        from tradingbot import model_book
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(model_book.early_status(WatchStore(watch_db_path(cfg)), cfg, args.hours or 8.0,
                                                 journal=Journal(cfg.db_path)), indent=2))
        return

    if args.command == "early-backtest":
        from tradingbot import model_book
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(model_book.early_backtest(WatchStore(watch_db_path(cfg)), cfg, args.hours or 8.0), indent=2))
        return

    if args.command == "early-walk-forward":
        from tradingbot import model_book
        from tradingbot.watcher import WatchStore, watch_db_path
        print(json.dumps(model_book.early_walk_forward(WatchStore(watch_db_path(cfg)), cfg), indent=2))
        return

    if args.command == "launch-backfill":
        import time
        from tradingbot.enrich import Enricher
        from tradingbot.watcher import WatchStore, watch_db_path
        e = Enricher(cfg, WatchStore(watch_db_path(cfg)))
        if not e.helius.enabled:
            print("HELIUS_API_KEY is not set")
            return
        now = time.time()
        out = e.backfill_launch(now - args.days * 86400, now - 3600, args.max_pools, args.max_credits or 60_000,
                                progress=lambda m: print(m, flush=True))
        print(json.dumps(out, indent=2))
        return

    if args.command == "launch-backtest":
        import time
        from tradingbot import launch_analysis
        from tradingbot.journal import Journal
        from tradingbot.watcher import WatchStore, watch_db_path
        store = WatchStore(watch_db_path(cfg))
        mints = {r[0] for r in Journal(cfg.db_path).query(
            "SELECT DISTINCT token FROM events WHERE kind='model_buy' AND strategy LIKE 'early%'")}
        picks = {r["pool"] for r in store.query("SELECT pool, mint FROM watch_pools WHERE dex='pumpswap'")
                 if r["mint"] in mints}
        since = time.time() - args.hours * 3600 if args.hours else 0.0
        if args.export_exit:
            print(json.dumps(launch_analysis.export_exit_rows(launch_analysis.rows(store, cfg, since), args.test_from,
                                                              args.no_name, args.export_exit), separators=(",", ":")))
            return
        print(json.dumps(launch_analysis.backtest(store, cfg, since, picks, test_from=args.test_from,
                                                  only_timing=args.timing, no_name=args.no_name,
                                                  fetch_candles=args.fetch_candles,
                                                  progress=lambda i, n: i % 25 == 0 and print(
                                                      f"candles {i}/{n}", file=sys.stderr, flush=True)), indent=2))
        return

    if args.command == "model-status":
        import time
        from tradingbot import model_book, report
        from tradingbot.journal import Journal
        path = model_book.model_path(cfg)
        meta = json.loads(path.read_text()) if path.exists() else None
        j = Journal(cfg.db_path)
        from tradingbot.engine import STUCK_AFTER_S

        def book_status(name: str) -> dict:
            stuck = {p.id for p in j.open_positions(name)
                     if (b := report._blocked(j, name, p)) and time.time() - b[0] >= STUCK_AFTER_S}
            skips: dict[str, int] = {}
            for r in j.query("SELECT detail FROM events WHERE strategy=? AND kind='entry_skipped'", (name,)):
                key = r["detail"].split(" (")[0].split(" since")[0]
                key = "price ran since the snapshot" if key.startswith("price ran") else key
                key = "round-trip loss too high" if key.startswith("round-trip loss") else key
                key = "executor declined: price impact too high" if "price impact" in key else key
                skips[key] = skips.get(key, 0) + 1
            return {"hold_min": model_book.hold_min(cfg, name),
                    "buys": j.query("SELECT COUNT(*) AS n FROM events WHERE strategy=? AND kind='model_buy'",
                                    (name,))[0]["n"],
                    "open": [{"symbol": p.symbol, "cost_usd": round(p.cost_usd, 2),
                              "hours": round((time.time() - (p.opened_at or time.time())) / 3600, 1),
                              "exit_blocked": report._exit_blocked(j, name, p)} for p in j.open_positions(name)],
                    "skip_reasons": skips, "rugs": model_book.rug_summary(j, stuck, name)}
        print(json.dumps({"model": {k: meta[k] for k in ("threshold", "meta")} if meta else "not trained yet",
                          **{name: book_status(name) for name in model_book.BOOKS}}, indent=2, default=str))
        return

    if args.command == "watch-report-now":
        from datetime import datetime, timezone
        from tradingbot.watcher import Watcher
        w = Watcher.from_config(cfg)
        out = w.write_report(args.path or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        print(f"report written to {out}" if out else "report failed; see the log line above")
        return

    if args.command in ("watch-report", "watch-export"):
        from tradingbot import watch_analysis
        from tradingbot.watcher import WatchStore, watch_db_path
        store = WatchStore(watch_db_path(cfg))
        if args.command == "watch-export":
            if not args.path:
                parser.error("watch-export needs a destination file, e.g. /data/watch-rows.csv")
            print(f"wrote {watch_analysis.export_csv(store, args.path, args.age)} rows to {args.path}")
            return
        if args.label not in ("pump2x", "pump5x", "clean2x"):
            parser.error("--label must be pump2x, pump5x or clean2x")
        if args.paid_value:
            pv = watch_analysis.paid_value(watch_analysis.build_dataset(store, 15), args.label)
            print(json.dumps(pv, indent=2) if args.json else watch_analysis.render_paid_value(pv))
            return
        subset = "control" if args.control else "enriched" if args.enriched else None
        res = watch_analysis.report_by_age(store, (args.label,), (subset,), detail_age=args.age)
        detail, summ = res["detail"][(args.label, subset)], res["summary"][(args.label, subset)]
        print(json.dumps({"summary": summ, "detail": detail}, indent=2, default=str) if args.json
              else watch_analysis.render(detail, summ))
        return

    if args.command == "playbook":
        from tradingbot.journal import Journal
        from tradingbot.learning import playbook
        for l in playbook.dump(Journal(cfg.db_path)):
            print(f"[{l['status']}] {l['tag']} ({l['support']} for / {l['contradict']} against)")
            print(f"  {l['statement']}  (when: {l['applies_when'] or 'always'})")
        return

    if args.command == "postmortems":
        from tradingbot.journal import Journal
        rows = Journal(cfg.db_path).query("SELECT * FROM postmortems ORDER BY ts DESC LIMIT 20")
        for r in rows:
            a = json.loads(r["analysis"])
            ret = "n/a" if r["return_pct"] is None else f"{r['return_pct']:+.1f}%"
            print(f"[{r['kind']}] {r['symbol']} {ret} {a['category']} ({a['decision_quality']} decision)")
            print(f"  {a['what_happened']}")
            if a["lesson"]["statement"]:
                print(f"  lesson [{a['lesson']['tag']}]: {a['lesson']['statement']}")
        return

    if args.command in ("backup-now", "backup-list"):
        from tradingbot import backups
        from tradingbot.journal import Journal
        if args.command == "backup-now":
            out = backups.DailyBackup(Journal(cfg.db_path), cfg).run()
        else:
            out = backups.list_backups(cfg)
        print(json.dumps(out, indent=2))
        raise SystemExit(1 if isinstance(out, dict) and out.get("error") else 0)

    if args.command == "replay":
        from tradingbot import replay
        from tradingbot.journal import Journal
        if args.group not in replay.GROUPS:
            parser.error(f"--group must be one of {replay.GROUPS}")
        overrides = dict(kv.split("=", 1) for kv in args.set)
        print(replay.report(Journal(cfg.db_path), cfg, overrides, args.group, args.size, args.split))
        return

    engine = build_engine(cfg, with_analyst=args.command != "scan" and cfg.claude_trading)
    if args.command == "scan":
        survivors = engine.screen()
        for snap, result, _ in survivors:
            print(f"{result.score:.2f}  {snap.symbol:<12} liq=${snap.liquidity_usd or 0:>10,.0f}  "
                  f"h1={snap.price_change.get('h1', 0):+6.1f}%  {snap.token_address}")
        print(f"{len(survivors)} candidate(s) would be sent to Claude")
    elif args.command == "once":
        engine.tick()
    else:
        if cfg.db_path != ":memory:":
            from tradingbot.learning import codebase
            codebase.record_running(cfg)
        engine.run_forever()
        if engine.restart_reason:
            # Exit so the supervisor (Fly restart policy, systemd, Docker) starts the new code.
            logging.getLogger(__name__).warning("exiting for restart: %s", engine.restart_reason)
            raise SystemExit(75)


if __name__ == "__main__":
    main()
