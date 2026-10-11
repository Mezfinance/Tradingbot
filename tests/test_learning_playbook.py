import json
import random
from types import SimpleNamespace

from tradingbot.analyst import Analyst
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tradingbot.learning import playbook as pb
from tradingbot.learning.learner import Learner
from tests.test_analyst_engine import BUY_JSON, FakeClient, FakeDex, resp


def analysis(tag="", statement="", supports=(), contradicts=(), confidence=0.6):
    return {"lesson": {"tag": tag, "statement": statement, "applies_when": "x", "confidence": confidence},
            "supports_lessons": list(supports), "contradicts_lessons": list(contradicts)}


def status(j, tag):
    return j.query("SELECT status, support, contradict FROM lessons WHERE tag=?", (tag,))[0]


def test_lesson_lifecycle(cfg):
    j = Journal(":memory:")
    book = pb.Playbook(j, cfg)
    book.record(analysis("Avoid Paid-Shills!", "Skip paid hype."))
    assert status(j, "avoid_paid_shills")["status"] == "candidate"
    book.record(analysis("avoid_paid_shills", "same idea"))
    book.record(analysis(supports=["avoid_paid_shills"]))
    assert dict(status(j, "avoid_paid_shills")) == {"status": "active", "support": 3, "contradict": 0}
    assert "Skip paid hype." in book.render()[0]  # original wording kept

    for _ in range(3):
        book.record(analysis(contradicts=["avoid_paid_shills"]))
    assert status(j, "avoid_paid_shills")["status"] == "retired"
    assert book.render() == ("", "")


def test_low_confidence_or_empty_lessons_are_ignored(cfg):
    j = Journal(":memory:")
    book = pb.Playbook(j, cfg)
    book.record(analysis("weak", "maybe", confidence=0.1))
    book.record(analysis("", ""))
    assert j.query("SELECT COUNT(*) AS n FROM lessons")[0]["n"] == 0


def test_ab_arms(cfg):
    j = Journal(":memory:")
    book = pb.Playbook(j, cfg, rng=random.Random(3))
    assert book.choose_arm() == ("none", "")
    for _ in range(3):
        book.record(analysis("tight_stops_on_spikes", "Use tighter stops after parabolic entries."))
    arms = [book.choose_arm()[0].split(":")[0] for _ in range(2000)]
    share = arms.count("control") / len(arms)
    assert abs(share - cfg.playbook_holdout) < 0.04


def test_analyst_sees_playbook_and_arm_is_journalled(cfg, good_pair, safe):
    cfg.playbook_holdout = 0.0
    j = Journal(":memory:")
    client = FakeClient([resp("end_turn", BUY_JSON)])
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe), Analyst(cfg, client),
                 PaperExecutor(cfg), j)
    for _ in range(3):
        eng.analyst.playbook.record(analysis("organic_only", "Buy only with organic buzz."))
    eng.tick()
    assert "<playbook>" in client.calls[0]["messages"][0]["content"]
    assert "Buy only with organic buzz." in client.calls[0]["messages"][0]["content"]
    arm = j.query("SELECT playbook_arm FROM decisions WHERE stage='analysed'")[0]["playbook_arm"]
    assert arm.startswith("playbook:")


def test_postmortem_feeds_playbook(cfg):
    j = Journal(":memory:")
    learner = Learner(j, cfg, FakeClient([]))
    learner.after_postmortem(analysis("cut_late_entries", "Skip entries after +80% in an hour."))
    assert learner.lessons() == [{"tag": "cut_late_entries", "statement": "Skip entries after +80% in an hour."}]


def test_consolidation_merges_counts(cfg):
    j = Journal(":memory:")
    book = pb.Playbook(j, cfg)
    for tag in ("paid_hype", "paid_shills", "shilled_tokens"):
        for _ in range(2):
            book.record(analysis(tag, f"{tag} statement"))
    book.record(analysis("other", "unrelated"))
    for _ in range(12):
        book.record(analysis(f"filler_{_}", "filler"))
    merge = {"merges": [{"keep_tag": "paid_hype", "merge_tags": ["paid_shills", "shilled_tokens"],
                         "statement": "Avoid tokens driven by paid promotion.", "applies_when": "always"}]}
    cost = book.consolidate_if_due(FakeClient([resp("end_turn", json.dumps(merge))]))
    assert cost > 0
    row = status(j, "paid_hype")
    assert row["support"] == 6 and row["status"] == "active"
    assert not j.query("SELECT 1 FROM lessons WHERE tag IN ('paid_shills','shilled_tokens')")
    assert book.consolidate_if_due(FakeClient([])) == 0  # not again within a day


def test_arm_report(cfg):
    j = Journal(":memory:")
    pb.Playbook(j, cfg)
    j.execute("INSERT INTO decisions(id, ts, token, stage, playbook_arm) VALUES(1,0,'A','analysed','playbook:ab')")
    j.execute("INSERT INTO decisions(id, ts, token, stage, playbook_arm) VALUES(2,0,'B','analysed','control:ab')")
    j.execute("INSERT INTO positions(token, strategy, cost_usd, realized_usd, closed_at, decision_id) "
              "VALUES('A','claude',50,60,1,1), ('B','claude',50,40,1,2)")
    rep = pb.arm_report(j)
    assert rep["playbook"]["total_pnl_usd"] == 10 and rep["control"]["total_pnl_usd"] == -10
