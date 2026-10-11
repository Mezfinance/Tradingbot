"""Daily journal backups: a compressed copy on the volume, and one in object storage.

Once per UTC day (on the first scan cycle after `backup_hour_utc`) the bot:
  1. takes an online SQLite copy (safe while trading continues),
  2. gzips it to <backup dir>/tradingbot-YYYY-MM-DD.db.gz and keeps the newest
     `backup_keep_local` files there,
  3. uploads it to the S3-compatible bucket named by BUCKET_NAME, if configured.

On Fly.io, `fly storage create` creates a Tigris bucket and sets BUCKET_NAME,
AWS_ENDPOINT_URL_S3, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY on the app.
Without them, only the local copies are kept.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from tradingbot.config import Config
from tradingbot.journal import Journal

log = logging.getLogger(__name__)

STATE_KEY = "last_backup"
FILE_PREFIX = "tradingbot-"
FILE_SUFFIX = ".db.gz"


def backup_dir(cfg: Config) -> Path:
    if cfg.backup_dir:
        return Path(cfg.backup_dir)
    return Path(cfg.db_path).resolve().parent / "backups"


def s3_client():
    """S3 client for the configured bucket, or None when object storage isn't set up."""
    if not os.environ.get("BUCKET_NAME") or not os.environ.get("AWS_ACCESS_KEY_ID"):
        return None
    try:
        import boto3  # optional dependency: pip install .[backup]
    except ImportError:
        log.warning("BUCKET_NAME is set but boto3 is not installed; keeping local backups only")
        return None
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"),
        region_name=os.environ.get("AWS_REGION", "auto"),
    )


class DailyBackup:
    def __init__(self, journal: Journal, cfg: Config, client_factory=s3_client):
        self.journal = journal
        self.cfg = cfg
        self.client_factory = client_factory

    def last(self) -> dict | None:
        raw = self.journal.get_state(STATE_KEY)
        return json.loads(raw) if raw else None

    def due(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        last = self.last()
        if last and last.get("date") == now.strftime("%Y-%m-%d"):
            return False
        return now.hour >= self.cfg.backup_hour_utc

    def run_if_due(self) -> dict | None:
        if self.cfg.db_path == ":memory:" or not self.due():  # nothing to back up in tests
            return None
        return self.run()

    def run(self, now: datetime | None = None) -> dict:
        """Make today's backup now. Never raises: a failed backup must not stop trading."""
        now = now or datetime.now(timezone.utc)
        date = now.strftime("%Y-%m-%d")
        result: dict = {"date": date, "ts": time.time(), "file": None, "bytes": 0, "uploaded": False, "error": None}
        try:
            path = self._make_local(date)
            result["file"], result["bytes"] = str(path), path.stat().st_size
            self._prune_local()
            client = self.client_factory()
            if client is not None:
                bucket = os.environ["BUCKET_NAME"]
                key = f"{self.cfg.backup_prefix}{path.name}"
                client.upload_file(str(path), bucket, key)
                result["uploaded"] = f"s3://{bucket}/{key}"
        except Exception as exc:  # noqa: BLE001 (report any failure, keep the bot running)
            result["error"] = f"{type(exc).__name__}: {exc}"
            log.error("daily backup failed: %s", result["error"])
            self.journal.log_event("backup_failed", None, result["error"])
        else:
            where = result["uploaded"] or "volume only (no bucket configured)"
            log.info("daily backup %s: %.1f MB -> %s", date, result["bytes"] / 1e6, where)
        self.journal.set_state(STATE_KEY, json.dumps(result))
        return result

    def _make_local(self, date: str) -> Path:
        out_dir = backup_dir(self.cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = out_dir / f"{FILE_PREFIX}{date}.db"
        final = out_dir / f"{FILE_PREFIX}{date}{FILE_SUFFIX}"
        self.journal.backup(str(raw))
        try:
            with open(raw, "rb") as src, gzip.open(final, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
        finally:
            raw.unlink(missing_ok=True)
        return final

    def _prune_local(self) -> None:
        files = sorted(backup_dir(self.cfg).glob(f"{FILE_PREFIX}*{FILE_SUFFIX}"))
        for old in files[: max(0, len(files) - self.cfg.backup_keep_local)]:
            old.unlink()


def list_backups(cfg: Config, client_factory=s3_client) -> dict:
    local = [p.name for p in sorted(backup_dir(cfg).glob(f"{FILE_PREFIX}*{FILE_SUFFIX}"))]
    remote: list[str] | str = "no bucket configured"
    client = client_factory()
    if client is not None:
        resp = client.list_objects_v2(Bucket=os.environ["BUCKET_NAME"], Prefix=cfg.backup_prefix)
        remote = sorted(o["Key"] for o in resp.get("Contents", []))
    return {"local": local, "remote": remote}
