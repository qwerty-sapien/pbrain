"""Integration tests for swarm/swarm/scoring/scorer.py — score_bundle().

Verifies the complete Phase 8 scoring pipeline using fixture bundles with no
live LLM calls (judge_session is monkeypatched throughout).

Tests:
- score.tsv is written with header + >= 1 data row
- LOCKED collapse row: assertion forces selection=wrong even though stubbed judge
  says Productivity selection=right (D-09 assertion-overrides-judge proof)
- Learning pillar row is never selection=wrong (EVAL-02 learning untouched)
- scores.json stays {} after scoring (Pitfall 3 guard)
- score_meta.json records narrow_tutor_penalty=True with A-01 fired
- Every row's confidence is a string from {"high","med","low"}, no float (D-06 guard)
- Every row has P[0-3] severity and a repro containing '#' (EVAL-04)
- score_bundle never raises on a broken/empty bundle (SWARM-07)

Bundle fixture layout follows the D-19 spec:
    <persona_dir>/
        profile.md          YAML frontmatter: id, win_condition, anti_goal, primary_intent
        transcript.md       pb turns with fenced pb-response blocks
        findings.json       issues with action_index values
        artifacts_before.json
        artifacts_after.json
        scores.json         stays {} — Pitfall 3 guard

No real vault access. No live LLM calls.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from swarm.swarm.scoring.judge import JudgeVerdict
from swarm.swarm.scoring.scorer import score_bundle

# ---------------------------------------------------------------------------
# Repo root and rubric path — real rubric.md loaded once
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_RUBRIC = REPO_ROOT / "rubric.md"


# ---------------------------------------------------------------------------
# Fixture helpers — build a minimal D-19 bundle
# ---------------------------------------------------------------------------

def _make_transcript(
    persona_id: str,
    pb_responses: list[str],
    brain_utterances: list[str] | None = None,
) -> str:
    """Build a minimal transcript.md matching the evidence.py write_transcript_md format.

    Each turn has:
        ## Turn [N]
        **Brain said:** {brain_utterance}
        **Command:** `pb do test`
        Exit: 0 | 123ms
        ```
        {pb_stdout_excerpt}
        ```
    """
    if brain_utterances is None:
        brain_utterances = [f"Turn {i} input" for i in range(len(pb_responses))]

    lines = [
        f"# Transcript: {persona_id}",
        "Domain: test",
        "Run time: 2026-06-03T17:00:00Z",
        "",
    ]
    for i, (pb_out, brain_in) in enumerate(zip(pb_responses, brain_utterances)):
        lines.append(f"## Turn [{i}]")
        lines.append(f"**Brain said:** {brain_in}")
        lines.append(f"**Command:** `pb do test`")
        lines.append("Exit: 0 | 123ms")
        lines.append("```")
        lines.append(pb_out)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _make_profile_md(
    persona_id: str,
    win_condition: str = "pb produces a prioritised view",
    anti_goal: str = "pb routes overwhelm into a study session",
    primary_intent: str = "prioritise-across-commitments",
) -> str:
    """Build a profile.md with YAML frontmatter in the standard format."""
    return (
        "---\n"
        f"id: {persona_id}\n"
        f"win_condition: \"{win_condition}\"\n"
        f"anti_goal: \"{anti_goal}\"\n"
        f"primary_intent: \"{primary_intent}\"\n"
        "---\n\n"
        f"## {persona_id}\n\n"
        "This is a test persona.\n"
    )


def _make_bundle(
    tmp_path: Path,
    persona_id: str = "overwhelmed-planner",
    pb_responses: list[str] | None = None,
    brain_utterances: list[str] | None = None,
    issues: list[dict] | None = None,
    artifacts_before: dict | None = None,
    artifacts_after: dict | None = None,
    primary_intent: str = "prioritise-across-commitments",
) -> Path:
    """Build a full minimal D-19 bundle under tmp_path/subagent_runs/run-001/<persona_id>.

    Writes:
        profile.md, transcript.md, findings.json,
        artifacts_before.json, artifacts_after.json, scores.json

    Returns:
        Path to the bundle directory.
    """
    bundle_dir = tmp_path / "subagent_runs" / "run-001" / persona_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if pb_responses is None:
        pb_responses = [
            "Here are some suggestions.",
            "Let me start a study session for you to help you focus.",
            "Your study session is ready.",
        ]

    # profile.md
    (bundle_dir / "profile.md").write_text(
        _make_profile_md(persona_id, primary_intent=primary_intent)
    )

    # transcript.md
    (bundle_dir / "transcript.md").write_text(
        _make_transcript(persona_id, pb_responses, brain_utterances)
    )

    # findings.json — include action_index 3..6 as in the locked collapse spec
    issue_list = issues if issues is not None else [
        {"action_index": 3, "command": "pb do something", "severity": "P1", "issue": "wrong mode"},
        {"action_index": 4, "command": "pb study", "severity": "P1", "issue": "narrow"},
        {"action_index": 5, "command": "pb study", "severity": "P1", "issue": "narrow"},
        {"action_index": 6, "command": "pb study", "severity": "P1", "issue": "narrow"},
    ]
    findings = {
        "persona_id": persona_id,
        "total_actions": len(pb_responses),
        "passed": len(pb_responses),
        "failed": 0,
        "issues": issue_list,
        "artifact_changes": [],
    }
    (bundle_dir / "findings.json").write_text(json.dumps(findings, indent=2))

    # artifacts_before.json and artifacts_after.json
    (bundle_dir / "artifacts_before.json").write_text(
        json.dumps(artifacts_before or {}, indent=2)
    )
    (bundle_dir / "artifacts_after.json").write_text(
        json.dumps(artifacts_after or {}, indent=2)
    )

    # scores.json — the Pitfall 3 guard: must stay {} after scoring
    (bundle_dir / "scores.json").write_text("{}\n")

    return bundle_dir


# ---------------------------------------------------------------------------
# Stub judge verdict — Productivity and Learning both say selection=right
# to prove the assertion overrides the judge (D-09)
# ---------------------------------------------------------------------------

_STUB_VERDICT = JudgeVerdict(
    intent="prioritise-across-commitments",
    mode_chosen="study-session",
    confidence="high",
    pillar_verdicts={
        "Productivity": {"selection": "right", "execution": "pass"},
        "Learning": {"selection": "right", "execution": "pass"},
    },
)


def _stub_judge(*args: Any, **kwargs: Any) -> JudgeVerdict:
    """Stub replacing judge_session — returns a verdict saying Productivity=right.

    The test suite uses this to prove that the assertion_engine's selection=wrong
    for Productivity overrides the judge's selection=right (D-09).
    """
    return _STUB_VERDICT


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreTsvWritten:
    """score.tsv is created and has the correct structure."""

    def test_score_tsv_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_bundle() creates score.tsv with a header and >= 1 data row."""
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)
        bundle = _make_bundle(tmp_path)
        tsv_path = bundle / "score.tsv"

        rows = score_bundle(bundle, tsv_path, rubric_path=REPO_RUBRIC)

        assert tsv_path.exists(), "score.tsv must be written by score_bundle()"
        content = tsv_path.read_text()
        lines = [ln for ln in content.splitlines() if ln.strip()]
        assert len(lines) >= 2, f"Expected header + >= 1 data row, got {len(lines)} lines"
        # Header line check
        assert "selection" in lines[0], f"Header must contain 'selection', got: {lines[0]}"
        assert "pillar" in lines[0], f"Header must contain 'pillar', got: {lines[0]}"


class TestLockedCollapseRow:
    """Assertion forces selection=wrong even when the stubbed judge says right (D-09)."""

    def test_locked_collapse_row(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The LOCKED collapse row is reproduced: mode_chosen=study-session,
        selection=wrong, execution=pass, pillar=Productivity, severity=P1.

        The stubbed judge deliberately returns Productivity selection=right —
        the test proves that the assertion engine's lock overrides the judge (D-09).
        """
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)
        bundle = _make_bundle(tmp_path)
        tsv_path = bundle / "score.tsv"

        score_bundle(bundle, tsv_path, rubric_path=REPO_RUBRIC)

        with open(tsv_path, newline="") as fh:
            reader = list(csv.DictReader(fh, delimiter="\t"))

        # Find the Productivity row
        productivity_rows = [r for r in reader if r.get("pillar") == "Productivity"]
        assert len(productivity_rows) >= 1, (
            f"Expected at least one Productivity row in score.tsv. "
            f"All rows: {[r.get('pillar') for r in reader]}"
        )
        r = productivity_rows[0]

        # LOCKED collapse encoding from 08-CONTEXT.md <specifics>
        assert r["mode_chosen"] == "study-session", (
            f"mode_chosen must be 'study-session', got {r['mode_chosen']!r}"
        )
        assert r["selection"] == "wrong", (
            f"Assertion must force selection='wrong' even though the stubbed judge says "
            f"'right'. Got {r['selection']!r}. This proves D-09 assertion-overrides-judge."
        )
        assert r["execution"] == "pass", (
            f"execution must be 'pass' (study session was competent). Got {r['execution']!r}"
        )
        assert r["severity"] == "P1", (
            f"severity must be 'P1' for A-01. Got {r['severity']!r}"
        )


class TestLearningPillarUntouched:
    """Learning pillar row never gets selection=wrong from a narrow-tutor assertion (EVAL-02)."""

    def test_learning_pillar_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No row should have pillar=Learning and selection=wrong.

        The stub judge says Learning selection=right. Even if the judge said wrong,
        EVAL-02 guarantees the assertion engine only locks Productivity for A-01.
        """
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)
        bundle = _make_bundle(tmp_path)
        tsv_path = bundle / "score.tsv"

        score_bundle(bundle, tsv_path, rubric_path=REPO_RUBRIC)

        with open(tsv_path, newline="") as fh:
            reader = list(csv.DictReader(fh, delimiter="\t"))

        bad_learning_rows = [
            r for r in reader
            if r.get("pillar") == "Learning" and r.get("selection") == "wrong"
        ]
        assert bad_learning_rows == [], (
            f"EVAL-02 violated: Learning row(s) with selection=wrong found: "
            f"{bad_learning_rows}"
        )

        # If a Learning row exists from the judge, it should carry selection=right
        learning_rows = [r for r in reader if r.get("pillar") == "Learning"]
        for lr in learning_rows:
            assert lr["selection"] == "right", (
                f"Learning row from judge should have selection=right, got {lr['selection']!r}"
            )


class TestScoresJsonUnchanged:
    """scores.json must still be {} after score_bundle() runs (Pitfall 3 guard)."""

    def test_scores_json_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """scores.json stays {} — score_bundle() never overwrites it."""
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)
        bundle = _make_bundle(tmp_path)
        tsv_path = bundle / "score.tsv"

        score_bundle(bundle, tsv_path, rubric_path=REPO_RUBRIC)

        scores_json = bundle / "scores.json"
        assert scores_json.exists(), "scores.json must still exist after scoring"
        content = json.loads(scores_json.read_text())
        assert content == {}, (
            f"scores.json must remain {{}} (Pitfall 3 / D-17). Got: {content!r}"
        )


class TestScoreMetaRecordsPenalty:
    """score_meta.json records narrow_tutor_penalty=True with A-01 fired."""

    def test_score_meta_records_penalty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score_meta.json must record narrow_tutor_penalty=True and A-01 in fired_assertions."""
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)
        bundle = _make_bundle(tmp_path)
        tsv_path = bundle / "score.tsv"

        score_bundle(bundle, tsv_path, rubric_path=REPO_RUBRIC)

        meta_path = bundle / "score_meta.json"
        assert meta_path.exists(), "score_meta.json must be written by score_bundle()"
        meta = json.loads(meta_path.read_text())

        assert meta.get("narrow_tutor_penalty") is True, (
            f"narrow_tutor_penalty must be True when A-01 fires. "
            f"score_meta.json = {meta}"
        )
        assert "A-01" in meta.get("fired_assertions", []), (
            f"A-01 must appear in fired_assertions. "
            f"score_meta.json = {meta}"
        )


class TestNoFloatConfidence:
    """Every row's confidence is a bounded string, no float values anywhere (D-06)."""

    def test_no_float_confidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All confidence values are from {'high','med','low'}; no float-pattern strings."""
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)
        bundle = _make_bundle(tmp_path)
        tsv_path = bundle / "score.tsv"

        score_bundle(bundle, tsv_path, rubric_path=REPO_RUBRIC)

        # Pattern that detects float-looking strings (e.g. '0.8', '1.0', '75%')
        float_pattern = re.compile(r"\b0?\.\d+\b|\b1\.0+\b|\d+\.\d+%?")

        with open(tsv_path, newline="") as fh:
            reader = list(csv.DictReader(fh, delimiter="\t"))

        valid_confidence = {"high", "med", "low"}
        for row in reader:
            conf = row.get("confidence", "")
            assert conf in valid_confidence, (
                f"confidence must be in {valid_confidence!r}, got {conf!r}"
            )
            # Check all fields for float-pattern strings (D-06 guard)
            for field_name, value in row.items():
                assert not float_pattern.search(str(value)), (
                    f"Float-pattern value found in field '{field_name}': {value!r} "
                    f"(D-06: no probabilities in score.tsv)"
                )


class TestEveryRowHasSeverityAndRepro:
    """Every row has P[0-3] severity and a repro string containing '#' (EVAL-04)."""

    def test_every_row_has_severity_and_repro(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each row in score.tsv has severity matching P[0-3] and repro containing '#'."""
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)
        bundle = _make_bundle(tmp_path)
        tsv_path = bundle / "score.tsv"

        score_bundle(bundle, tsv_path, rubric_path=REPO_RUBRIC)

        severity_pattern = re.compile(r"^P[0-3]$")

        with open(tsv_path, newline="") as fh:
            reader = list(csv.DictReader(fh, delimiter="\t"))

        assert len(reader) >= 1, "score.tsv must have at least one data row"
        for row in reader:
            severity = row.get("severity", "")
            repro = row.get("repro", "")
            assert severity_pattern.match(severity), (
                f"severity must match P[0-3], got {severity!r} (EVAL-04)"
            )
            assert "#" in repro, (
                f"repro must contain '#' (e.g. 'overwhelmed-planner#a3-a6'), "
                f"got {repro!r} (EVAL-04)"
            )


class TestNeverRaisesOnBrokenBundle:
    """score_bundle never raises on a broken/empty bundle (SWARM-07)."""

    def test_never_raises_on_broken_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pointing score_bundle at an empty bundle dir must return [] and not raise.

        Verifies either:
        - Returns [] and writes score_error.txt, OR
        - Returns [] and writes an empty score.tsv (valid graceful degradation)
        """
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)

        # Intentionally empty bundle — no profile.md, no findings.json, no transcript
        empty_bundle = tmp_path / "subagent_runs" / "run-002" / "empty-persona"
        empty_bundle.mkdir(parents=True, exist_ok=True)
        tsv_path = empty_bundle / "score.tsv"

        # Must not raise — this is the SWARM-07 contract
        result = score_bundle(empty_bundle, tsv_path, rubric_path=REPO_RUBRIC)

        assert isinstance(result, list), (
            f"score_bundle must return a list, got {type(result)}"
        )
        # Either empty list (error path) or a list of rows (successful graceful degradation)
        # The key invariant is: it did not raise
        # On error path: score_error.txt is written
        score_error = empty_bundle / "score_error.txt"
        # score.tsv or score_error.txt must exist (one of the two graceful outcomes)
        assert tsv_path.exists() or score_error.exists(), (
            "Either score.tsv or score_error.txt must be written on broken bundle — "
            "score_bundle must produce some output on failure (SWARM-07)"
        )
