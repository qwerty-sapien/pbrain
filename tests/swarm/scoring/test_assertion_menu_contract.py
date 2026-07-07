"""Fix 3 — assertions must fire on pb's REAL menu output contract.

pb's `do` output is a terse menu ("Do / N. <Mode> <title> / because ..."), not the
conversational prose the original A-01..A-06 phrase lists assumed. The canonical run
proved the assertions physically could not fire on the real shape. These tests pin the
menu-contract behaviour plus the new A-07 echo-as-task P0 (the user's explicit ask).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm.swarm.scoring.assertion_engine import evaluate_assertions
from swarm.swarm.scoring.rubric_loader import load_rubric
from tests.swarm.scoring.test_assertion_engine import _make_bundle


@pytest.fixture(scope="module")
def rubric() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    return load_rubric(repo_root / "rubric.md")


# A real what-now-user turn-1 menu: pb echoed the user's words back as a Study title.
DO_MENU_ECHO = (
    "Do\n"
    "1. Study I only have five minutes, just give me the single most important one.\n"
    "   because study is the safer default for conceptual/internalisation work.\n"
    "2. Capture a quick thought\n"
    "   because Capture it before it disappears."
)


class TestEchoA07:
    def test_echo_fires_A07_p0_routing(self, tmp_path: Path, rubric: dict) -> None:
        bundle = _make_bundle(
            tmp_path,
            "what-now-user",
            pb_responses=[DO_MENU_ECHO],
            brain_utterances=[
                "I only have five minutes, just give me the single most important one."
            ],
        )
        results = evaluate_assertions(bundle, "what-now-user", rubric)
        a07 = [r for r in results if r.assertion_id == "A-07"]
        assert len(a07) == 1, f"A-07 echo must fire. Got: {results}"
        r = a07[0]
        assert r.pillar == "Agentic-routing"
        assert r.severity == "P0"
        assert r.selection == "wrong"
        assert r.narrow_tutor_penalty is True

    def test_legit_study_topic_does_not_fire_A07(self, tmp_path: Path, rubric: dict) -> None:
        bundle = _make_bundle(
            tmp_path,
            "math-hobbyist",
            pb_responses=[
                "Do\n1. Study eigenvalue decomposition\n   because conceptual work."
            ],
            brain_utterances=["I want to understand eigenvalues better"],
        )
        results = evaluate_assertions(bundle, "math-hobbyist", rubric)
        assert [r for r in results if r.assertion_id == "A-07"] == []


class TestMenuContractAssertions:
    def test_overwhelmed_planner_study_menu_fires_A01(self, tmp_path: Path, rubric: dict) -> None:
        bundle = _make_bundle(
            tmp_path,
            "overwhelmed-planner",
            pb_responses=[
                "Do\n1. Study quarterly planning basics\n"
                "   because study is the safer default.\n"
                "2. Capture a quick thought\n   because capture it."
            ],
            brain_utterances=["I'm drowning, show me what matters across my commitments"],
        )
        results = evaluate_assertions(bundle, "overwhelmed-planner", rubric)
        a01 = [r for r in results if r.assertion_id == "A-01"]
        assert len(a01) == 1 and a01[0].pillar == "Productivity", f"Got: {results}"

    def test_what_now_user_study_menu_fires_A03(self, tmp_path: Path, rubric: dict) -> None:
        bundle = _make_bundle(
            tmp_path,
            "what-now-user",
            pb_responses=[
                "Do\n1. Study weekly review habits\n"
                "   because study is the safer default.\n"
                "2. Capture a quick thought\n   because capture it."
            ],
            brain_utterances=["just tell me which of my todos to start"],
        )
        results = evaluate_assertions(bundle, "what-now-user", rubric)
        a03 = [r for r in results if r.assertion_id == "A-03"]
        assert len(a03) == 1 and a03[0].pillar == "Goal-alignment", f"Got: {results}"

    def test_german_practise_menu_without_b1_fires_A06(self, tmp_path: Path, rubric: dict) -> None:
        bundle = _make_bundle(
            tmp_path,
            "german-learner",
            pb_responses=["Do\n1. Practise German conversation\n   because embodied skill work."],
            brain_utterances=["let's keep going"],
        )
        results = evaluate_assertions(bundle, "german-learner", rubric)
        a06 = [r for r in results if r.assertion_id == "A-06"]
        assert len(a06) == 1 and a06[0].pillar == "Goal-alignment", f"Got: {results}"

    def test_german_practise_menu_with_b1_does_not_fire_A06(self, tmp_path: Path, rubric: dict) -> None:
        bundle = _make_bundle(
            tmp_path,
            "german-learner",
            pb_responses=["Do\n1. Practise German for your B1 exam goal\n   because tied to B1."],
            brain_utterances=["let's keep going"],
        )
        results = evaluate_assertions(bundle, "german-learner", rubric)
        assert [r for r in results if r.assertion_id == "A-06"] == []
