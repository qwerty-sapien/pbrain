"""Strong commitment_persisted / goals_referenced signals for the live judge.

The legacy vault-path-diff heuristic returns False for EVERY real bundle (this
product persists goals/tasks to SQLite, never to a vault path with 'goal'/'task'),
so it was effectively cosmetic. These signals instead read pb's own authoritative
intake records (outcome=persisted + commitment artifact_kind) and pb's actual
output, so the judge is told the truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from swarm.swarm.scoring import evidence_facts as ef
from swarm.swarm.scoring.scorer import build_mechanical_facts


def _bundle(tmp_path: Path, *, intake: list[dict] | None = None,
            commands: list[dict] | None = None, name: str = "p") -> Path:
    b = tmp_path / name
    b.mkdir(parents=True, exist_ok=True)
    if commands is not None:
        (b / "commands.jsonl").write_text("\n".join(json.dumps(c) for c in commands) + "\n")
    if intake is not None:
        idir = b / "_env" / "data" / "intake"
        idir.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(intake):
            (idir / f"{i:03d}.json").write_text(json.dumps(rec))
    return b


class TestCommitmentPersisted:
    def test_true_when_intake_persisted_a_goal(self, tmp_path):
        b = _bundle(tmp_path, intake=[
            {"workflow": "goal", "outcome": "persisted", "metadata": {"artifact_kind": "goal"}},
        ])
        assert ef.commitment_persisted(b) is True

    def test_true_for_todo_and_task_kinds(self, tmp_path):
        for kind in ("todo", "task", "plan"):
            b = _bundle(tmp_path, name=f"p-{kind}", intake=[
                {"outcome": "persisted", "metadata": {"artifact_kind": kind}},
            ])
            assert ef.commitment_persisted(b) is True, kind

    def test_false_for_thought_capture(self, tmp_path):
        b = _bundle(tmp_path, intake=[
            {"outcome": "persisted", "metadata": {"artifact_kind": "thought"}},
        ])
        assert ef.commitment_persisted(b) is False

    def test_false_when_not_persisted(self, tmp_path):
        b = _bundle(tmp_path, intake=[
            {"outcome": "abandoned", "metadata": {"artifact_kind": "goal"}},
        ])
        assert ef.commitment_persisted(b) is False

    def test_false_when_no_intake(self, tmp_path):
        b = _bundle(tmp_path, commands=[{"command": "/x/pb do hi", "stdout_excerpt": "Do"}])
        assert ef.commitment_persisted(b) is False


class TestGoalsReferenced:
    def test_true_when_pb_output_names_a_known_goal(self, tmp_path):
        intake = [{"entries": [
            {"content": {"goal_preview": {"title": "Cambridge B1 German Exam Prep"}}},
        ]}]
        commands = [{"command": "/x/pb next",
                     "stdout_excerpt": "3. Study toward Cambridge B1 German Exam Prep"}]
        b = _bundle(tmp_path, intake=intake, commands=commands)
        assert ef.goals_referenced(b) is True

    def test_true_on_goal_reference_phrase(self, tmp_path):
        commands = [{"command": "/x/pb do hi",
                     "stdout_excerpt": "3. Study X\n   because This goal still needs conceptual progress."}]
        b = _bundle(tmp_path, intake=[], commands=commands)
        assert ef.goals_referenced(b) is True

    def test_false_when_only_the_user_input_mentions_the_goal(self, tmp_path):
        # Goal title appears in the COMMAND (user/brain text) but NOT pb stdout.
        intake = [{"entries": [{"content": {"goal_preview": {"title": "Ship Analytics Dashboard"}}}]}]
        commands = [{"command": "/x/pb do help me with Ship Analytics Dashboard",
                     "stdout_excerpt": "Do\n1. Capture a quick thought"}]
        b = _bundle(tmp_path, intake=intake, commands=commands)
        assert ef.goals_referenced(b) is False

    def test_false_for_generic_output(self, tmp_path):
        commands = [{"command": "/x/pb do hi", "stdout_excerpt": "Do\n1. Capture a quick thought"}]
        b = _bundle(tmp_path, intake=[], commands=commands)
        assert ef.goals_referenced(b) is False

    def test_mcp_handshake_output_excluded(self, tmp_path):
        # A goal phrase appearing only in the MCP probe output must not count.
        commands = [{"command": "/x/productivebrain-mcp --config c.toml",
                     "stdout_excerpt": '{"this goal still needs conceptual": 1}'}]
        b = _bundle(tmp_path, intake=[], commands=commands)
        assert ef.goals_referenced(b) is False


class TestWiredIntoMechanicalFacts:
    def test_build_mechanical_facts_uses_strong_signals(self, tmp_path):
        intake = [{"outcome": "persisted", "metadata": {"artifact_kind": "goal"},
                   "entries": [{"content": {"goal_preview": {"title": "Learn Topology Intuition"}}}]}]
        commands = [{"command": "/x/pb next", "stdout_excerpt": "Continue goal: Learn Topology Intuition"}]
        b = _bundle(tmp_path, intake=intake, commands=commands)
        facts = build_mechanical_facts(b, {"total_actions": 1, "issues": []})
        assert facts["commitment_persisted"] is True
        assert facts["goals_referenced"] is True
