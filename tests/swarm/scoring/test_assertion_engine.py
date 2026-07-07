"""Tests for swarm/swarm/scoring/assertion_engine.py

Verifies the deterministic failure-mode assertion evaluation against fixture bundles:
- overwhelmed-planner collapse fires A-01 on Productivity with narrow_tutor_penalty=True
- Learning pillar is NOT in any AssertionResult from A-01 (Pitfall 1 scope-bleed guard)
- A clean prioritised-view response does NOT fire A-01
- FAIL signal in persona utterance (brain_utterance) does NOT fire (Pitfall 4)
- german-learner without B1 reference fires A-06 on Goal-alignment
- german-learner WITH B1 reference does NOT fire A-06 (EVAL-02 Learning-untouched)
- All AssertionResults always have selection == "wrong" (never "right")

All tests load the real rubric.md so assertions are verified against the actual bindings.
No LLM calls.  Follows test_evidence_writer.py fixture-bundle pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm.swarm.scoring.assertion_engine import (
    NARROW_TUTOR_PILLARS,
    AssertionResult,
    evaluate_assertions,
)
from swarm.swarm.scoring.rubric_loader import load_rubric


# ---------------------------------------------------------------------------
# Rubric fixture (real rubric.md — assertions must match actual bindings)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rubric() -> dict:
    """Load the real rubric.md from the repo root.

    Resolves relative to this test file: tests/swarm/scoring/ -> repo_root/rubric.md
    """
    repo_root = Path(__file__).resolve().parents[3]
    rubric_path = repo_root / "rubric.md"
    return load_rubric(rubric_path)


# ---------------------------------------------------------------------------
# Bundle-building helpers
# ---------------------------------------------------------------------------

def _make_transcript(
    persona_id: str,
    pb_responses: list[str],
    brain_utterances: list[str] | None = None,
) -> str:
    """Build a minimal transcript.md in the evidence.py write_transcript_md format.

    The D-18 format is:
        ## Turn [N]
        **Brain said:** {brain_utterance}
        **Command:** `pb do ...`
        Exit: 0 | 123ms
        ```
        {pb_stdout_excerpt}
        ```

    Only pb_responses appear in fenced blocks; brain_utterances go on **Brain said:** lines.
    """
    if brain_utterances is None:
        brain_utterances = [f"Turn {i} input" for i in range(len(pb_responses))]

    lines = [
        f"# Transcript: {persona_id}",
        "Domain: test",
        "Run time: 2026-06-02T17:00:00Z",
        "",
    ]
    for i, (pb_out, brain_in) in enumerate(zip(pb_responses, brain_utterances)):
        lines.append(f"## Turn [{i}]")
        lines.append(f"**Brain said:** {brain_in}")
        lines.append(f"**Command:** `pb do test`")
        lines.append(f"Exit: 0 | 123ms")
        if pb_out:
            lines.append(f"```\n{pb_out}\n```")
        lines.append("")

    return "\n".join(lines)


def _make_bundle(
    tmp_path: Path,
    persona_id: str,
    pb_responses: list[str],
    brain_utterances: list[str] | None = None,
    issues: list[dict] | None = None,
    artifacts_before: dict | None = None,
    artifacts_after: dict | None = None,
) -> Path:
    """Build a minimal D-19 bundle directory under tmp_path.

    Creates the files the assertion engine reads:
        transcript.md, findings.json, artifacts_before.json, artifacts_after.json

    Args:
        tmp_path:          pytest tmp_path fixture.
        persona_id:        Persona id string.
        pb_responses:      List of pb stdout_excerpt strings (one per turn).
        brain_utterances:  Optional list of persona input strings.
        issues:            Optional list of issue dicts for findings.json.
        artifacts_before:  Optional before-snapshot dict.
        artifacts_after:   Optional after-snapshot dict.

    Returns:
        Path to the bundle directory.
    """
    bundle_dir = tmp_path / f"run-001" / persona_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # transcript.md — the main source the engine reads pb responses from
    transcript = _make_transcript(persona_id, pb_responses, brain_utterances)
    (bundle_dir / "transcript.md").write_text(transcript)

    # findings.json
    findings = {
        "persona_id": persona_id,
        "total_actions": len(pb_responses),
        "passed": len(pb_responses),
        "failed": 0,
        "issues": issues or [],
        "artifact_changes": [],
    }
    (bundle_dir / "findings.json").write_text(json.dumps(findings, indent=2))

    # artifacts_before.json
    (bundle_dir / "artifacts_before.json").write_text(
        json.dumps(artifacts_before or {}, indent=2)
    )

    # artifacts_after.json
    (bundle_dir / "artifacts_after.json").write_text(
        json.dumps(artifacts_after or {}, indent=2)
    )

    return bundle_dir


# ---------------------------------------------------------------------------
# Core assertion tests
# ---------------------------------------------------------------------------

class TestOverwhelmedPlannerA01:
    """Tests for A-01: overwhelmed-planner narrowing into a study session."""

    def test_overwhelmed_collapse_fires_A01_on_productivity_only(
        self, tmp_path: Path, rubric: dict
    ) -> None:
        """Locked narrow-tutor collapse: A-01 fires on Productivity, Learning untouched.

        This is the LOCKED example from 08-CONTEXT.md <specifics>:
            mode_chosen=study-session, selection=wrong, execution=pass,
            pillar=Productivity (hit), Learning unaffected, severity=P1

        Also validates Pitfall 1 scope-bleed guard: no result has pillar==Learning.
        """
        bundle = _make_bundle(
            tmp_path,
            persona_id="overwhelmed-planner",
            pb_responses=[
                "Here are some suggestions.",
                "I see you are overwhelmed. Let me start a study session for you to help you focus.",
                "Your German session is ready.",
            ],
            brain_utterances=[
                "I have too much going on.",
                "I don't know what to do first.",
                "OK.",
            ],
            issues=[
                {"action_index": 3, "command": "pb do", "severity": "P1", "issue": "wrong mode"},
                {"action_index": 6, "command": "pb study", "severity": "P1", "issue": "wrong mode"},
            ],
        )

        results = evaluate_assertions(bundle, "overwhelmed-planner", rubric)

        # Must fire exactly one AssertionResult
        assert len(results) == 1, f"Expected exactly 1 result, got {results}"
        r = results[0]

        assert r.assertion_id == "A-01", f"Expected A-01, got {r.assertion_id}"
        assert r.pillar == "Productivity", f"Expected Productivity, got {r.pillar}"
        assert r.selection == "wrong", f"Expected 'wrong', got {r.selection}"
        assert r.narrow_tutor_penalty is True, "A-01 should set narrow_tutor_penalty=True"
        assert r.severity == "P1", f"Expected P1, got {r.severity}"

        # Pitfall 1 — scope bleed guard: NO result should have pillar==Learning
        learning_results = [res for res in results if res.pillar == "Learning"]
        assert learning_results == [], (
            f"A-01 must NOT produce a Learning pillar result (Pitfall 1 scope-bleed). "
            f"Got: {learning_results}"
        )

    def test_clean_prioritised_view_does_not_fire_A01(
        self, tmp_path: Path, rubric: dict
    ) -> None:
        """A-01 must NOT fire when pb produces a clean prioritised view (win condition).

        The overwhelmed-planner win_condition: 'pb produces a prioritised view
        across my actual commitments without starting a study session.'
        """
        bundle = _make_bundle(
            tmp_path,
            persona_id="overwhelmed-planner",
            pb_responses=[
                "Here is your prioritised view across your current commitments:\n"
                "1. Write product spec section 2 (2 days old — high urgency)\n"
                "2. Review German vocab deck (supports your B1 goal)\n"
                "3. Book dentist appointment (stale — low priority)\n"
                "Your MVP goal is your most time-sensitive. Would you like to start with item 1?",
            ],
            brain_utterances=["I'm overwhelmed, what should I do?"],
        )

        results = evaluate_assertions(bundle, "overwhelmed-planner", rubric)

        assert results == [], (
            f"A-01 must NOT fire for a prioritised-view response. Got: {results}"
        )

    def test_fail_signal_in_persona_utterance_does_not_fire(
        self, tmp_path: Path, rubric: dict
    ) -> None:
        """Pitfall 4: FAIL phrase in brain_utterance (persona input) must NOT fire A-01.

        The assertion engine must match ONLY against pb's response text (stdout_excerpt),
        never against the persona's brain_utterance.
        """
        bundle = _make_bundle(
            tmp_path,
            persona_id="overwhelmed-planner",
            pb_responses=[
                # pb gives a clean prioritised view — correct behaviour
                "Your current commitments, ranked by urgency:\n"
                "1. Ship MVP by end of Q3 — most critical\n"
                "2. Pass Cambridge B1 exam in June\n"
                "3. Fitness target before summer\n"
                "Recommendation: focus on item 1 today.",
            ],
            brain_utterances=[
                # Only the PERSONA says "start a study session" — pb does NOT
                "Please start a study session for me, I can't decide what to do."
            ],
        )

        results = evaluate_assertions(bundle, "overwhelmed-planner", rubric)

        assert results == [], (
            f"Pitfall 4: FAIL signal in brain_utterance must NOT fire A-01. "
            f"Got: {results}"
        )


# ---------------------------------------------------------------------------
# German learner A-06 tests (EVAL-02 Learning-untouched)
# ---------------------------------------------------------------------------

class TestGermanLearnerA06:
    """Tests for A-06: german-learner session not linked to B1 goal."""

    def test_german_not_linked_to_b1_fires_A06_goal_alignment(
        self, tmp_path: Path, rubric: dict
    ) -> None:
        """A-06 fires when pb starts a session without referencing the B1 goal.

        Signal (rubric.md A-06): 'pb starts a practice/study session without
        referencing or linking to the B1 exam goal.'
        """
        bundle = _make_bundle(
            tmp_path,
            persona_id="german-learner",
            pb_responses=[
                "Let's practice German grammar. I'll start your session now.\n"
                "Exercise 1: Fill in the correct article...",
            ],
            brain_utterances=["I want to practice German."],
        )

        results = evaluate_assertions(bundle, "german-learner", rubric)

        assert len(results) == 1, f"Expected exactly 1 result for A-06, got {results}"
        r = results[0]

        assert r.assertion_id == "A-06", f"Expected A-06, got {r.assertion_id}"
        assert r.pillar == "Goal-alignment", f"Expected Goal-alignment, got {r.pillar}"
        assert r.selection == "wrong", f"Expected 'wrong', got {r.selection}"
        assert r.narrow_tutor_penalty is True, "A-06 should set narrow_tutor_penalty=True"

    def test_german_linked_to_b1_does_not_fire_and_learning_free(
        self, tmp_path: Path, rubric: dict
    ) -> None:
        """A-06 must NOT fire when pb's response references the B1 goal (EVAL-02).

        When A-06 does not fire, the Learning pillar is free to be scored on its
        own merits by the judge — this is the EVAL-02 'Learning untouched' guarantee.
        """
        bundle = _make_bundle(
            tmp_path,
            persona_id="german-learner",
            pb_responses=[
                "Let's start your German practice session, tied to your B1 exam goal "
                "(Cambridge B1, June). Today we'll focus on Akkusativ/Dativ production "
                "since that's your speaking gap. Sentence drill: 'Ich gehe in den Park.'",
            ],
            brain_utterances=["I want to practice German."],
        )

        results = evaluate_assertions(bundle, "german-learner", rubric)

        assert results == [], (
            f"A-06 must NOT fire when pb references the B1 goal (EVAL-02). Got: {results}"
        )


# ---------------------------------------------------------------------------
# Cross-fixture safety check: selection is always "wrong"
# ---------------------------------------------------------------------------

class TestSelectionAlwaysWrong:
    """No AssertionResult may ever have selection != 'wrong' (D-09 structural override)."""

    def test_results_never_set_selection_right(
        self, tmp_path: Path, rubric: dict
    ) -> None:
        """Across all fixture bundles, no AssertionResult has selection other than 'wrong'."""
        # Collect results from multiple fire scenarios
        all_results: list[AssertionResult] = []

        # A-01 collapse
        bundle1 = _make_bundle(
            tmp_path / "bundle1",
            persona_id="overwhelmed-planner",
            pb_responses=["Let me start a study session for you right away."],
        )
        all_results.extend(evaluate_assertions(bundle1, "overwhelmed-planner", rubric))

        # A-06 not-linked
        bundle2 = _make_bundle(
            tmp_path / "bundle2",
            persona_id="german-learner",
            pb_responses=["Let's practice German. Starting your session now."],
        )
        all_results.extend(evaluate_assertions(bundle2, "german-learner", rubric))

        # At least some results should have fired in the above
        assert len(all_results) > 0, "At least one assertion should have fired across both bundles"

        # Core invariant: every fired result has selection="wrong"
        for r in all_results:
            assert r.selection == "wrong", (
                f"AssertionResult for {r.assertion_id} has selection={r.selection!r}, "
                f"expected 'wrong'. D-09 structural override must always produce 'wrong'."
            )


# ---------------------------------------------------------------------------
# NARROW_TUTOR_PILLARS constant validation
# ---------------------------------------------------------------------------

class TestNarrowTutorPillarsConstant:
    """Verify the NARROW_TUTOR_PILLARS constant is correctly defined."""

    def test_narrow_tutor_pillars_does_not_contain_learning(self) -> None:
        """Learning must NOT be in NARROW_TUTOR_PILLARS (D-10 + EVAL-02)."""
        assert "Learning" not in NARROW_TUTOR_PILLARS, (
            "NARROW_TUTOR_PILLARS must NOT contain 'Learning' — "
            "the Learning pillar must be free when the study session itself is competent (EVAL-02)"
        )

    def test_narrow_tutor_pillars_contains_expected_pillars(self) -> None:
        """Productivity, Accountability, Goal-alignment must be in NARROW_TUTOR_PILLARS."""
        assert "Productivity" in NARROW_TUTOR_PILLARS
        assert "Accountability" in NARROW_TUTOR_PILLARS
        assert "Goal-alignment" in NARROW_TUTOR_PILLARS

    def test_narrow_tutor_pillars_is_frozenset(self) -> None:
        """NARROW_TUTOR_PILLARS must be immutable (frozenset)."""
        assert isinstance(NARROW_TUTOR_PILLARS, frozenset)
