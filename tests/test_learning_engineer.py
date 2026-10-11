import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.learning import codebase, engineer, versions
from tests.test_analyst_engine import FakeClient, resp
from tests.test_learning_postmortem import tool_use

REPO = Path(__file__).resolve().parents[1]
# Repo checkout layout, or the copy baked into the Docker image. The engineer's workspace has
# neither (the entrypoint is outside the package), so these tests skip there.
ENTRYPOINT = next((p for p in (REPO / "deploy/entrypoint.sh", Path("/entrypoint.sh")) if p.exists()), None)
needs_entrypoint = pytest.mark.skipif(ENTRYPOINT is None, reason="entrypoint.sh not available here")

FAKE_FILES = {
    "tradingbot/__init__.py": "",
    "tradingbot/cli.py": "def main():\n    return 0\n",
    "tradingbot/engine.py": "",
    "tradingbot/risk.py": "STOP = 25\n",
    "tradingbot/analyst.py": "PROMPT = 'old'\n",
}
FAKE_TEST = "from tradingbot.analyst import PROMPT\n\n\ndef test_prompt():\n    assert PROMPT\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tiny stand-in project, so the engineer's gate runs fast and in isolation."""
    src = tmp_path / "src"
    for rel, text in FAKE_FILES.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    (src / "tests").mkdir()
    (src / "tests" / "__init__.py").write_text("")
    (src / "tests" / "test_fake.py").write_text(FAKE_TEST)
    monkeypatch.setattr(codebase, "running_package_dir", lambda: src / "tradingbot")
    monkeypatch.setenv("TB_BASE_HASH", "base123")
    monkeypatch.delenv("TB_CODE_VERSION", raising=False)
    for k in ("BUCKET_NAME", "GITHUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    cfg = Config(db_path=str(tmp_path / "bot.db"), code_dir=str(tmp_path / "code"))
    j = Journal(cfg.db_path)
    versions.ensure(j)
    j.execute("INSERT INTO proposals(ts, source, title, problem, evidence, idea, priority) "
              "VALUES(0, 'optimizer', 'Sharper prompt', 'weak theses', '5 losses', 'reword prompt', 1)")
    restarts = []
    eng = engineer.Engineer(j, cfg, restart=restarts.append)
    return SimpleNamespace(cfg=cfg, j=j, eng=eng, restarts=restarts, src=src)


def done(summary="Reworded the prompt."):
    return resp("end_turn", json.dumps({"status": "done", "summary": summary, "expected_effect": "fewer bad buys"}))


def test_ships_a_tested_change_as_an_immutable_version(env):
    client = FakeClient([
        tool_use("write_file", {"path": "tradingbot/analyst.py", "content": "PROMPT = 'new and better'\n"}),
        tool_use("run_tests", {}, "t2"),
        done(),
    ])
    res = env.eng.attempt(client)
    assert res.outcome == "ok"
    assert "PASSED" in res.transcript[4]["content"][0]["content"]
    assert env.restarts == ["new code version v1"]
    assert codebase.active_version(env.cfg) == "v1"
    vdir = codebase.code_root(env.cfg) / "versions" / "v1"
    assert (vdir / "tradingbot/analyst.py").read_text() == "PROMPT = 'new and better'\n"
    assert (vdir / "BASE_HASH").read_text() == "base123" and (vdir / "PARENT").read_text() == ""
    assert "+PROMPT = 'new and better'" in (vdir / "DIFF.patch").read_text()
    manifest = json.loads((vdir / "MANIFEST.json").read_text())
    assert manifest["files_changed"] == ["tradingbot/analyst.py"] and manifest["title"] == "Sharper prompt"
    assert codebase.verify_version(env.cfg, "v1")
    assert (vdir / "tradingbot/analyst.py").stat().st_mode & 0o777 == 0o444  # read-only (root ignores it)
    [row] = env.j.query("SELECT * FROM code_versions")
    assert row["status"] == "probation" and row["version"] == "v1"
    assert env.j.query("SELECT status FROM proposals")[0]["status"] == "done"
    assert [v["version"] for v in codebase.list_versions(env.cfg)] == ["v1"]


def test_protected_files_cannot_be_written(env):
    client = FakeClient([
        tool_use("write_file", {"path": "tradingbot/risk.py", "content": "STOP = 90\n"}),
        tool_use("write_file", {"path": "tests/test_fake.py", "content": "def test_x():\n    pass\n"}, "t2"),
        tool_use("write_file", {"path": "../escape.py", "content": "x"}, "t3"),
        resp("end_turn", json.dumps({"status": "gave_up", "summary": "blocked", "expected_effect": ""})),
    ])
    env.eng.attempt(client)
    results = [m["content"][0] for m in client.calls[-1]["messages"] if m["role"] == "user"
               and isinstance(m["content"], list)]
    assert all(r.get("is_error") for r in results)
    assert (env.src / "tradingbot/risk.py").read_text() == "STOP = 25\n"
    assert codebase.active_version(env.cfg) is None
    assert env.j.query("SELECT status FROM proposals")[0]["status"] == "rejected"


def test_failing_tests_block_the_release(env):
    client = FakeClient([
        tool_use("write_file", {"path": "tradingbot/analyst.py", "content": "PROMPT = ''\n"}),
        done(),
    ])
    env.eng.attempt(client)
    [row] = env.j.query("SELECT status, test_output FROM code_versions")
    assert row["status"] == "failed" and "tests failed" in row["test_output"]
    assert codebase.active_version(env.cfg) is None and env.restarts == []


def test_changes_touching_secrets_are_refused(env):
    client = FakeClient([
        tool_use("write_file", {"path": "tradingbot/analyst.py",
                                "content": "import os\nPROMPT = os.environ.get('ANTHROPIC_API_KEY', 'x')\n"}),
        done(),
    ])
    env.eng.attempt(client)
    assert "secrets" in env.j.query("SELECT test_output FROM code_versions")[0]["test_output"]


def test_one_attempt_per_interval(env):
    env.eng.attempt(FakeClient([resp("end_turn", json.dumps(
        {"status": "gave_up", "summary": "no", "expected_effect": ""}))]))
    env.j.execute("INSERT INTO proposals(ts, title, problem, evidence, idea) VALUES(0,'b','b','b','b')")
    assert not env.eng.due()


def ship(env):
    env.eng.attempt(FakeClient([
        tool_use("write_file", {"path": "tradingbot/analyst.py", "content": "PROMPT = 'v1'\n"}), done()]))


def test_loop_errors_roll_back_to_parent(env):
    ship(env)
    for _ in range(env.cfg.code_max_errors):
        env.j.log_event("loop_error", None, "boom")
    verdict = env.eng.review()
    assert verdict.startswith("rolled back")
    assert codebase.active_version(env.cfg) is None  # parent = image code
    assert env.restarts[-1] == "rollback of v1"
    assert env.j.query("SELECT status FROM code_versions")[0]["status"] == "rolled_back"


def test_worse_edge_vs_baseline_rolls_back(env):
    ship(env)
    since = env.j.query("SELECT activated_ts FROM code_versions")[0]["activated_ts"]
    for i in range(env.cfg.code_probation_trades):
        env.j.execute("INSERT INTO positions(token, strategy, opened_at, closed_at, cost_usd, realized_usd) "
                      "VALUES(?, 'claude', ?, ?, 50, 35)", (f"C{i}", since + 1 + i, since + 100))
        env.j.execute("INSERT INTO positions(token, strategy, opened_at, closed_at, cost_usd, realized_usd) "
                      "VALUES(?, 'filters_only', ?, ?, 50, 50)", (f"F{i}", since + 1 + i, since + 100))
    assert "edge vs filters_only" in env.eng.review()
    assert codebase.active_version(env.cfg) is None


def test_good_version_is_kept(env):
    ship(env)
    since = env.j.query("SELECT activated_ts FROM code_versions")[0]["activated_ts"]
    for i in range(env.cfg.code_probation_trades):
        env.j.execute("INSERT INTO positions(token, strategy, opened_at, closed_at, cost_usd, realized_usd) "
                      "VALUES(?, 'claude', ?, ?, 50, 60)", (f"C{i}", since + 1 + i, since + 100))
    assert env.eng.review().startswith("kept")
    assert codebase.active_version(env.cfg) == "v1"


def test_manual_rollback_and_tamper_detection(env):
    ship(env)
    assert codebase.switch_to(env.cfg, "image") == "image code"
    assert codebase.switch_to(env.cfg, "v1") == "v1"
    f = codebase.code_root(env.cfg) / "versions/v1/tradingbot/analyst.py"
    f.chmod(0o644)
    f.write_text("PROMPT = 'tampered'\n")
    assert not codebase.verify_version(env.cfg, "v1")
    with pytest.raises(ValueError, match="checksum"):
        codebase.switch_to(env.cfg, "v1")


# --- the shell entrypoint ------------------------------------------------------------

def entry(tmp_path, code_dir, base="base123", cmd='echo "$TB_CODE_VERSION|$PYTHONPATH"'):
    (tmp_path / "BASE_HASH").write_text(base)
    env = {"PATH": os.environ["PATH"], "TB_CODE_DIR": str(code_dir), "TB_BASE_HASH_FILE": str(tmp_path / "BASE_HASH")}
    out = subprocess.run(["sh", str(ENTRYPOINT), "sh", "-c", cmd], env=env,
                         capture_output=True, text=True, timeout=30)
    return out.stdout


@needs_entrypoint
def test_entrypoint_runs_active_version(env, tmp_path):
    ship(env)
    out = entry(tmp_path, codebase.code_root(env.cfg))
    assert out.strip().splitlines()[-1].startswith("v1|") and "versions/v1" in out


@needs_entrypoint
def test_entrypoint_falls_back_on_crash_loop_and_bad_checksum(env, tmp_path):
    ship(env)
    root = codebase.code_root(env.cfg)
    (root / "boots").write_text(str(3))  # next boot would be the 4th in a row
    out = entry(tmp_path, root)
    assert "crashed 3 times" in out and out.strip().splitlines()[-1] == "|"
    assert codebase.crashed_versions(env.cfg) == ["v1"]
    env.eng.review()  # notices the entrypoint's crash record
    assert env.j.query("SELECT status FROM code_versions")[0]["status"] == "crashed"

    codebase.set_active(env.cfg, "v1")
    f = root / "versions/v1/tradingbot/analyst.py"
    f.chmod(0o644)
    f.write_text("tampered")
    assert "checksum mismatch" in entry(tmp_path, root)


@needs_entrypoint
def test_entrypoint_ignores_versions_from_another_image(env, tmp_path):
    ship(env)
    out = entry(tmp_path, codebase.code_root(env.cfg), base="newimage")
    assert "different image" in out and out.strip().splitlines()[-1] == "|"
