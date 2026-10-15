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

from tradingbot.config import Config


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
    from tradingbot.execution.sim import SimExecutor
    from tradingbot.shadows import parse_shadows

    # Baselines are always simulated, even when the main book trades live.
    shadow_executors = {name: SimExecutor(cfg, jupiter) for name in parse_shadows(cfg.shadows)}
    canary = None
    if cfg.canary_every_n > 0:
        from tradingbot.execution.jupiter import JupiterExecutor
        canary = JupiterExecutor(cfg, jupiter)
    analyst = Analyst(cfg) if with_analyst else None
    triage = None
    if analyst is not None and cfg.triage:
        from tradingbot.triage import Triage
        triage = Triage(cfg, analyst.client)
    from tradingbot.data.geckoterminal import GeckoTerminal

    return Engine(cfg, DexScreener(), RugCheck(), analyst, executor, Journal(cfg.db_path), jupiter,
                  shadow_executors=shadow_executors, canary_executor=canary, gecko=GeckoTerminal(),
                  triage=triage)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tradingbot")
    parser.add_argument("command", choices=["scan", "once", "run", "report", "replay", "health", "backup",
                                            "backup-now", "backup-list", "learn-now", "postmortems", "playbook", "optimize-now", "settings",
                                            "settings-rollback", "code-versions", "code-diff", "code-rollback",
                                            "propose", "proposals", "watch-status", "watch-report",
                                            "watch-export", "watch-links", "watch-report-now"])
    parser.add_argument("path", nargs="?",
                        help="backup: destination file; code-diff/code-rollback: version (vN, or 'image')")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="report: print raw JSON")
    parser.add_argument("--group", default="bought", help="replay: bought|passed|survivors|filters|all")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="replay: override a setting, e.g. stop_loss_pct=20 (repeatable)")
    parser.add_argument("--split", help="replay: YYYY-MM-DD; tune before it, check after it")
    parser.add_argument("--size", type=float, default=50.0, help="replay: USD per trade")
    parser.add_argument("--age", type=int, default=15, help="watch-report/watch-export: feature age in minutes")
    parser.add_argument("--label", default="pump2x", help="watch-report: pump2x | pump5x | clean2x")
    parser.add_argument("--enriched", action="store_true", help="watch-report: only tokens with Helius/X data")
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

    if args.command == "report":
        from tradingbot import report
        from tradingbot.journal import Journal
        rep = report.build(Journal(cfg.db_path), cfg)
        print(json.dumps(rep, indent=2, default=str) if args.json else report.render(rep))
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

    if args.command == "watch-report-now":
        from datetime import datetime, timezone
        from tradingbot.watcher import Watcher
        w = Watcher.from_config(cfg)
        out = w.write_report(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
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
        subset = "enriched" if args.enriched else None
        detail = watch_analysis.analyse(store, args.age, args.label, subset=subset)
        summ = watch_analysis.summary(store, args.label, subset=subset)
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

    engine = build_engine(cfg, with_analyst=args.command != "scan")
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
