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
import sys
import time

from tradingbot.config import LIVE_CONFIRM_PHRASE, Config


def build_engine(cfg: Config, with_analyst: bool = True):
    from tradingbot.analyst import Analyst
    from tradingbot.data.dexscreener import DexScreener
    from tradingbot.data.jupiter import Jupiter
    from tradingbot.data.rugcheck import RugCheck
    from tradingbot.engine import Engine
    from tradingbot.journal import Journal

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
    if cfg.launch_canary and cfg.model_book:
        from tradingbot.execution.jupiter import JupiterExecutor
        launch_canary = JupiterExecutor(replace(cfg, max_price_impact_pct=cfg.model_book_max_impact_pct), jupiter,
                                        priority_fee_sol=cfg.launch_canary_priority_fee_sol,
                                        sell_slippage_bps=cfg.launch_canary_sell_slippage_bps)
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
                                            "canary-status", "canary-stop", "canary-start", "canary-smoke", "rw-check", "early-rw-check", "early-rw-study", "fill-gap-study", "size-study", "canary-rug-check", "canary-sweep", "block-compare", "market-monitor", "rw-bands", "band-rw-study", "band-compare", "risk-model", "history-backfill", "hold-study", "band-prize", "sz-whatif", "safeguards", "thin-sizing", "missed-signals", "flag-tracker", "dd-limits", "sz-stops", "canary-package"])
    parser.add_argument("path", nargs="?",
                        help="backup: destination file; code-diff/code-rollback: version (vN, or 'image')")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--all", action="store_true", help="report: every book, not just model5")
    parser.add_argument("--rps", type=float, default=2.0,
                        help="launch-hist-backfill: Helius calls per second (the live scorer shares the plan's limit)")
    parser.add_argument("--json", action="store_true", help="report: print raw JSON")
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
    parser.add_argument("--usd", type=float, default=2.0, help="canary-smoke: USD to buy and sell back")
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

    if args.command in ("canary-status", "canary-stop", "canary-start"):  # the real-money launch canary
        from tradingbot import launch_canary
        from tradingbot.engine import CANARY_HALT
        from tradingbot.journal import Journal
        import time as _time  # a later `import time` in main() makes the bare name local here
        journal = Journal(cfg.db_path)
        if args.command == "canary-stop":
            journal.set_state(CANARY_HALT, repr(_time.time()))
            print("launch canary halted: no new buys (open positions still sell at their pool minute)")
        elif args.command == "canary-start":
            journal.execute("DELETE FROM state WHERE key=?", (CANARY_HALT,))
            print("launch canary kill switch cleared")
        out = {"status": launch_canary.status(journal, cfg), "calibration": launch_canary.calibration(journal, args.hours)}
        print(json.dumps(out, indent=2, default=str))
        return

    if args.command == "canary-smoke":  # one tiny real buy + sell of a deep token
        if cfg.live_confirm != LIVE_CONFIRM_PHRASE:
            raise SystemExit(f"canary-smoke spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        from tradingbot import launch_canary
        from tradingbot.data.jupiter import Jupiter
        from tradingbot.execution.jupiter import JupiterExecutor
        ex = JupiterExecutor(cfg, Jupiter(cfg.slippage_bps), priority_fee_sol=cfg.launch_canary_priority_fee_sol,
                             sell_slippage_bps=cfg.launch_canary_sell_slippage_bps)
        print(json.dumps(launch_canary.smoke(ex, args.path or launch_canary.SMOKE_MINT, args.usd), indent=2, default=str))
        return

    if args.command == "canary-sweep":  # get the rent back from the wallet's empty token accounts
        from tradingbot import model_book as mb
        from tradingbot.data.jupiter import Jupiter
        from tradingbot.execution.jupiter import JupiterExecutor
        from tradingbot.journal import Journal
        if args.apply and cfg.live_confirm != LIVE_CONFIRM_PHRASE:
            raise SystemExit(f"canary-sweep --apply signs transactions; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        journal = Journal(cfg.db_path)
        keep = {p.token_address for b in mb.CANARY_BOOKS for p in journal.open_positions(b)}
        ex = JupiterExecutor(cfg, Jupiter(cfg.slippage_bps))
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
