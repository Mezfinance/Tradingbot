import gzip
import os
import sqlite3
from datetime import datetime, timezone

import pytest

from tradingbot import backups
from tradingbot.journal import Journal


def at(day, hour=1):
    return datetime(2026, 10, day, hour, tzinfo=timezone.utc)


@pytest.fixture
def setup(cfg, tmp_path, monkeypatch):
    for k in ("BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL_S3"):
        monkeypatch.delenv(k, raising=False)
    cfg.db_path = str(tmp_path / "tradingbot.db")
    j = Journal(cfg.db_path)
    j.set_state("cash:claude", "987.5")
    return cfg, j


def test_local_backup_is_a_restorable_gzip(setup):
    cfg, j = setup
    result = backups.DailyBackup(j, cfg).run(at(12))
    assert result["error"] is None and result["uploaded"] is False
    path = backups.backup_dir(cfg) / "tradingbot-2026-10-12.db.gz"
    restored = path.with_suffix("")  # .db
    restored.write_bytes(gzip.decompress(path.read_bytes()))
    row = sqlite3.connect(restored).execute("SELECT value FROM state WHERE key='cash:claude'").fetchone()
    assert row[0] == "987.5"


def test_runs_once_per_day_after_the_hour(setup):
    cfg, j = setup
    cfg.backup_hour_utc = 2
    b = backups.DailyBackup(j, cfg)
    assert not b.due(at(12, hour=1))  # before the hour
    assert b.due(at(12, hour=3))
    b.run(at(12, hour=3))
    assert not b.due(at(12, hour=23))  # already done today
    assert b.due(at(13, hour=2))


def test_keeps_only_newest_local_copies(setup):
    cfg, j = setup
    cfg.backup_keep_local = 3
    b = backups.DailyBackup(j, cfg)
    for day in range(1, 7):
        b.run(at(day))
    names = sorted(p.name for p in backups.backup_dir(cfg).iterdir())
    assert names == [f"tradingbot-2026-10-0{d}.db.gz" for d in (4, 5, 6)]


def test_failure_is_logged_not_raised(setup):
    cfg, j = setup

    class Broken:
        def upload_file(self, *a):
            raise ConnectionError("bucket unreachable")

    os.environ["BUCKET_NAME"] = "b"
    try:
        result = backups.DailyBackup(j, cfg, client_factory=Broken).run(at(12))
    finally:
        del os.environ["BUCKET_NAME"]
    assert "bucket unreachable" in result["error"]
    assert result["file"]  # local copy still made
    assert j.query("SELECT kind FROM events")[0]["kind"] == "backup_failed"


def test_uploads_to_s3_compatible_bucket(setup, monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    cfg, j = setup
    monkeypatch.setenv("BUCKET_NAME", "supermeme-backups")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")  # moto needs a real region name
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="supermeme-backups")
        result = backups.DailyBackup(j, cfg).run(at(12))
        assert result["uploaded"] == "s3://supermeme-backups/journal/tradingbot-2026-10-12.db.gz"
        listing = backups.list_backups(cfg)
        assert listing["remote"] == ["journal/tradingbot-2026-10-12.db.gz"]
        assert listing["local"] == ["tradingbot-2026-10-12.db.gz"]


def test_report_shows_last_backup(setup):
    from tradingbot import report

    cfg, j = setup
    assert "LAST BACKUP: none yet" in report.render(report.build(j, cfg))
    backups.DailyBackup(j, cfg).run(at(12))
    assert "LAST BACKUP: 2026-10-12 -> volume only" in report.render(report.build(j, cfg))


def test_in_memory_database_is_never_backed_up(cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg.db_path = ":memory:"
    assert backups.DailyBackup(Journal(":memory:"), cfg).run_if_due() is None
    assert not (tmp_path / "backups").exists()
