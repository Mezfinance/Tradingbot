"""Self-written code versions: where they live, how they go live, how they are undone.

Layout under `<data dir>/code/`:

  versions/vN/tradingbot/...   full package for version N (plus tests/)
  versions/vN/BASE_HASH        hash of the deployed image's package it was built from
  versions/vN/PARENT           version it was built on ("" = the image code)
  versions/vN/MANIFEST.json    proposal, summary, changed files, timestamps
  versions/vN/DIFF.patch       the change against its parent
  versions/vN/SHA256SUMS       checksum of every file in the version
  active                       name of the version to run ("vN"); absent = image code
  boots                        consecutive start-ups of the active version (crash detection)
  crashed                      versions the entrypoint abandoned (crash loop or bad checksum)

Versions are immutable once installed (files made read-only, checksummed) and are
never deleted, so any of them can be run again with
`python -m tradingbot code-rollback vN` (or `image` for the deployed code). Each one
is also archived to the backup bucket as code/vN.tar.gz when a bucket is configured.

deploy/entrypoint.sh reads these before Python starts: it puts the active version
first on PYTHONPATH; if the version's checksums fail or it crash-loops, it falls back
to that version's parent (and so on down to the image code). A new image (different
BASE_HASH) always wins, so `fly deploy` never runs code built for an older image.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import stat
import tarfile
import time
from pathlib import Path

import tradingbot
from tradingbot.config import Config

log = logging.getLogger(__name__)

HEALTHY_AFTER_S = 600  # uptime after which boots resets to 0


def code_root(cfg: Config) -> Path:
    if cfg.code_dir:
        return Path(cfg.code_dir)
    return Path(cfg.db_path).resolve().parent / "code"


def running_package_dir() -> Path:
    """The tradingbot package currently imported (an overlay version or the image)."""
    return Path(tradingbot.__file__).resolve().parent


def tests_dir(cfg: Config) -> Path:
    """Tests that belong to the running code: the active version's own, else the configured
    copy (TB_TESTS_DIR=/app/tests in the image), else the repo checkout's."""
    sibling = running_package_dir().parent / "tests"
    if current_version() and sibling.exists():
        return sibling
    if cfg.tests_dir:
        return Path(cfg.tests_dir)
    return sibling


def package_hash(pkg: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(pkg.rglob("*.py")):
        h.update(str(f.relative_to(pkg)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def image_base_hash() -> str:
    """Hash of the code baked into the image (TB_BASE_HASH is set by the entrypoint)."""
    return os.environ.get("TB_BASE_HASH") or package_hash(running_package_dir())


def current_version() -> str | None:
    return os.environ.get("TB_CODE_VERSION") or None


def make_workspace(cfg: Config, name: str) -> Path:
    """Fresh, writable copy of the running package and tests to edit."""
    ws = code_root(cfg) / "work" / name
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    shutil.copytree(running_package_dir(), ws / "tradingbot", ignore=shutil.ignore_patterns("__pycache__"))
    src_tests = tests_dir(cfg)
    if src_tests.exists():
        shutil.copytree(src_tests, ws / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    else:
        (ws / "tests").mkdir()
    for f in ws.rglob("*"):  # copies of read-only versions must be editable
        f.chmod(f.stat().st_mode | stat.S_IWUSR)
    return ws


def _checksums(dest: Path) -> str:
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dest)}"
             for f in sorted(dest.rglob("*")) if f.is_file() and f.name != "SHA256SUMS"]
    return "\n".join(lines) + "\n"


def install_version(cfg: Config, workspace: Path, version: str, parent: str | None,
                    manifest: dict, diff: str) -> Path:
    """Copy a tested workspace into an immutable, checksummed version directory."""
    dest = code_root(cfg) / "versions" / version
    if dest.exists():
        raise FileExistsError(f"{version} already exists; versions are immutable")
    shutil.copytree(workspace, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    (dest / "BASE_HASH").write_text(image_base_hash())
    (dest / "PARENT").write_text(parent or "")
    (dest / "DIFF.patch").write_text(diff)
    (dest / "MANIFEST.json").write_text(json.dumps(
        {"version": version, "parent": parent, "created": time.time(), "base_hash": image_base_hash(),
         **manifest}, indent=2))
    (dest / "SHA256SUMS").write_text(_checksums(dest))
    for f in dest.rglob("*"):
        if f.is_file():
            f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return dest


def verify_version(cfg: Config, version: str) -> bool:
    dest = code_root(cfg) / "versions" / version
    sums = dest / "SHA256SUMS"
    return sums.exists() and sums.read_text() == _checksums(dest)


def archive_version(cfg: Config, version: str, client_factory=None) -> str | None:
    """Upload versions/vN as code/vN.tar.gz to the backup bucket, if one is configured."""
    from tradingbot import backups

    client = (client_factory or backups.s3_client)()
    if client is None:
        return None
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(code_root(cfg) / "versions" / version, arcname=version)
    key = f"code/{version}.tar.gz"
    client.put_object(Bucket=os.environ["BUCKET_NAME"], Key=key, Body=buf.getvalue())
    return key


def list_versions(cfg: Config) -> list[dict]:
    root = code_root(cfg) / "versions"
    found = sorted(root.glob("v*/MANIFEST.json"),
                   key=lambda p: int(p.parent.name[1:]) if p.parent.name[1:].isdigit() else 0)
    active = active_version(cfg)
    out = []
    for m in found:
        info = json.loads(m.read_text())
        info["verified"] = verify_version(cfg, m.parent.name)
        info["active"] = m.parent.name == active
        out.append(info)
    return out


def switch_to(cfg: Config, target: str) -> str:
    """Make `target` ("vN" or "image") the code that runs from the next start."""
    if target == "image":
        set_active(cfg, None)
        return "image code"
    if not (code_root(cfg) / "versions" / target / "tradingbot").exists():
        raise ValueError(f"no such version {target}")
    if not verify_version(cfg, target):
        raise ValueError(f"{target} failed its checksum check")
    set_active(cfg, target)
    return target


def set_active(cfg: Config, version: str | None) -> None:
    root = code_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    active = root / "active"
    if version is None:
        active.unlink(missing_ok=True)
    else:
        active.write_text(version)
    (root / "boots").write_text("0")


def active_version(cfg: Config) -> str | None:
    f = code_root(cfg) / "active"
    return f.read_text().strip() or None if f.exists() else None


def mark_healthy(cfg: Config) -> None:
    """Called once the process has run cleanly for a while: the crash counter starts over."""
    f = code_root(cfg) / "boots"
    if f.exists() and f.read_text().strip() != "0":
        f.write_text("0")


def crashed_versions(cfg: Config) -> list[str]:
    f = code_root(cfg) / "crashed"
    return [l.strip() for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
