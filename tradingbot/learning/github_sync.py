"""Optional: push each self-written code version to a GitHub branch.

Self-written versions live on the Fly volume and are dropped when a new image is
deployed. Syncing them to a branch keeps them: merge the branch into the deploy
branch and the next `fly deploy` ships them as part of the image.

Enabled when GITHUB_TOKEN and GITHUB_REPO ("owner/name") are set. Commits go to
GITHUB_BRANCH (default "bot/self-improvements"), created from GITHUB_BASE_BRANCH
(default "main") if it doesn't exist. The bot never pushes to the base branch.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path

import httpx

from tradingbot.config import Config

log = logging.getLogger(__name__)

API = "https://api.github.com"


def changed_files(diff: str) -> list[str]:
    return sorted({m for m in re.findall(r"^\+\+\+ b/(\S+)$", diff, flags=re.M)})


def push_version(cfg: Config, ws: Path, version: str, message: str, diff: str,
                 http: httpx.Client | None = None) -> str | None:
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPO")
    if not token or not repo:
        return None
    branch = os.environ.get("GITHUB_BRANCH", "bot/self-improvements")
    base = os.environ.get("GITHUB_BASE_BRANCH", "main")
    if branch == base:
        raise ValueError("GITHUB_BRANCH must differ from GITHUB_BASE_BRANCH")
    gh = http or httpx.Client(base_url=API, timeout=30, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})

    ref = gh.get(f"/repos/{repo}/git/ref/heads/{branch}")
    if ref.status_code == 404:
        base_ref = gh.get(f"/repos/{repo}/git/ref/heads/{base}")
        base_ref.raise_for_status()
        gh.post(f"/repos/{repo}/git/refs",
                json={"ref": f"refs/heads/{branch}", "sha": base_ref.json()["object"]["sha"]}).raise_for_status()
    else:
        ref.raise_for_status()

    for rel in changed_files(diff):
        path = ws / rel
        if not path.exists():
            continue
        existing = gh.get(f"/repos/{repo}/contents/{rel}", params={"ref": branch})
        body = {"message": f"[bot {version}] {message.splitlines()[0]}\n\n{message}",
                "content": base64.b64encode(path.read_bytes()).decode(), "branch": branch}
        if existing.status_code == 200:
            body["sha"] = existing.json()["sha"]
        gh.put(f"/repos/{repo}/contents/{rel}", json=body).raise_for_status()
    log.info("pushed %s to %s:%s", version, repo, branch)
    return branch
