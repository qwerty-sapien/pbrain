"""Tests for pb/mcp/tools/scoring_evidence.py — swarm_evidence_facts MCP tool.

Verifies:
- D-03: The tool returns facts only — no selection, execution, pillar, or
  narrow_tutor_penalty keys in the return dict.
- T-08-11: Path traversal outside subagent_runs/ is refused (ScoringEvidenceError raised).
- JSON leak from findings.json P0 issues is surfaced as json_leak_detected=True.
- commitment_persisted is True when artifacts_after has a new commitment/goal row.
- Missing findings.json returns safe defaults without raising.
"""

import json
from pathlib import Path

import pytest

from swarm.swarm.scoring.evidence_facts import (
    ScoringEvidenceError,
    swarm_evidence_facts,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_bundle(
    tmp_path: Path,
    *,
    run_id: str = "run-001",
    persona_id: str = "overwhelmed-planner",
    findings: dict | None = None,
    artifacts_before: dict | None = None,
    artifacts_after: dict | None = None,
) -> Path:
    """Build a minimal persona bundle under tmp_path/subagent_runs/<run>/<persona>/.

    Returns the path to the persona bundle directory.
    """
    bundle = tmp_path / "subagent_runs" / run_id / persona_id
    bundle.mkdir(parents=True, exist_ok=True)

    if findings is not None:
        (bundle / "findings.json").write_text(json.dumps(findings))
    if artifacts_before is not None:
        (bundle / "artifacts_before.json").write_text(json.dumps(artifacts_before))
    if artifacts_after is not None:
        (bundle / "artifacts_after.json").write_text(json.dumps(artifacts_after))

    return bundle


def _minimal_findings(*, include_leak: bool = False, include_traceback: bool = False) -> dict:
    """Return a findings.json dict suitable for testing."""
    issues = []
    if include_leak:
        issues.append({
            "action_index": 2,
            "command": "pb next",
            "severity": "P0",
            "issue": "possible JSON leak in stdout",
        })
    if include_traceback:
        issues.append({
            "action_index": 3,
            "command": "pb goal add",
            "severity": "P0",
            "issue": "traceback in stderr",
        })
    if not issues:
        # A normal failed command (non-P0)
        issues.append({
            "action_index": 1,
            "command": "pb todo add",
            "severity": "P2",
            "issue": "command failed (exit 1)",
        })
    return {
        "persona_id": "overwhelmed-planner",
        "total_actions": 6,
        "passed": 5,
        "failed": len(issues),
        "issues": issues,
        "artifact_changes": ["goals.db", "vault/tasks.md"],
    }


# ---------------------------------------------------------------------------
# test_facts_are_facts_only — D-03 / T-08-15
# ---------------------------------------------------------------------------

class TestFactsAreFactsOnly:
    def test_returned_keys_are_fact_keys_only(self, tmp_path):
        """The returned dict must contain only fact keys — no verdict columns (D-03)."""
        bundle = _make_bundle(tmp_path, findings=_minimal_findings())
        result = swarm_evidence_facts(str(bundle))

        expected_keys = {
            "json_leak_detected",
            "failed_commands",
            "traceback_detected",
            "artifact_changes",
            "goals_referenced",
            "commitment_persisted",
            "total_actions",
        }
        assert set(result.keys()) == expected_keys, (
            f"Unexpected keys in result: {set(result.keys()) - expected_keys}"
        )

    def test_no_verdict_keys_in_result(self, tmp_path):
        """Verdict columns must never appear in the facts return (D-03 / T-08-15)."""
        bundle = _make_bundle(tmp_path, findings=_minimal_findings())
        result = swarm_evidence_facts(str(bundle))

        forbidden = {"selection", "execution", "pillar", "narrow_tutor_penalty", "confidence"}
        found_forbidden = forbidden & set(result.keys())
        assert not found_forbidden, (
            f"Verdict keys found in facts output (D-03 violation): {found_forbidden}"
        )

    def test_no_verdict_keys_in_failed_commands(self, tmp_path):
        """Each failed_command dict must not contain verdict keys."""
        bundle = _make_bundle(
            tmp_path,
            findings=_minimal_findings(include_leak=True),
        )
        result = swarm_evidence_facts(str(bundle))
        forbidden = {"selection", "execution", "pillar"}
        for fc in result["failed_commands"]:
            found_forbidden = forbidden & set(fc.keys())
            assert not found_forbidden, (
                f"Verdict keys found in failed_command entry: {found_forbidden}"
            )


# ---------------------------------------------------------------------------
# test_json_leak_surfaced_from_findings
# ---------------------------------------------------------------------------

class TestJsonLeakSurfacing:
    def test_json_leak_detected_true_for_p0_leak_issue(self, tmp_path):
        """json_leak_detected must be True when findings has a P0 JSON-leak issue."""
        bundle = _make_bundle(tmp_path, findings=_minimal_findings(include_leak=True))
        result = swarm_evidence_facts(str(bundle))
        assert result["json_leak_detected"] is True

    def test_json_leak_detected_false_when_no_leak(self, tmp_path):
        """json_leak_detected must be False when no JSON-leak issue exists."""
        bundle = _make_bundle(tmp_path, findings=_minimal_findings(include_leak=False))
        result = swarm_evidence_facts(str(bundle))
        assert result["json_leak_detected"] is False

    def test_traceback_detected_true(self, tmp_path):
        """traceback_detected must be True when a traceback issue is present."""
        bundle = _make_bundle(tmp_path, findings=_minimal_findings(include_traceback=True))
        result = swarm_evidence_facts(str(bundle))
        assert result["traceback_detected"] is True

    def test_traceback_detected_false_when_no_traceback(self, tmp_path):
        bundle = _make_bundle(tmp_path, findings=_minimal_findings())
        result = swarm_evidence_facts(str(bundle))
        assert result["traceback_detected"] is False


# ---------------------------------------------------------------------------
# test_commitment_persisted_diff
# ---------------------------------------------------------------------------

class TestCommitmentPersistedDiff:
    def test_commitment_persisted_true_when_new_goal_file(self, tmp_path):
        """commitment_persisted is True when artifacts_after has a new vault/goals/... path.

        Uses the real isolation.snapshot_artifacts flat {path: mtime_ns} shape.
        A new file whose path contains "goal" signals commitment_persisted=True.
        """
        before = {"vault/existing.md": "111"}
        after = {"vault/existing.md": "111", "vault/goals/learn-german.md": "222"}
        bundle = _make_bundle(
            tmp_path,
            findings=_minimal_findings(),
            artifacts_before=before,
            artifacts_after=after,
        )
        result = swarm_evidence_facts(str(bundle))
        assert result["commitment_persisted"] is True

    def test_commitment_persisted_true_when_new_commitment_file(self, tmp_path):
        """commitment_persisted is True when a new vault/commitments/... path is added.

        Uses the real flat {path: mtime_ns} shape.
        """
        before = {}
        after = {"vault/commitments/daily-review.md": "333"}
        bundle = _make_bundle(
            tmp_path,
            findings=_minimal_findings(),
            artifacts_before=before,
            artifacts_after=after,
        )
        result = swarm_evidence_facts(str(bundle))
        assert result["commitment_persisted"] is True

    def test_commitment_persisted_false_when_identical_before_after(self, tmp_path):
        """commitment_persisted is False when artifacts_before and after have identical keys.

        No new top-level path key means no new vault file was created.
        """
        same = {"vault/goals/old-goal.md": "111"}
        bundle = _make_bundle(
            tmp_path,
            findings=_minimal_findings(),
            artifacts_before=same,
            artifacts_after=same,
        )
        result = swarm_evidence_facts(str(bundle))
        assert result["commitment_persisted"] is False

    def test_commitment_persisted_false_when_no_goal_paths(self, tmp_path):
        """commitment_persisted is False when the new path contains no commitment word.

        A new vault/notes/... file is created (new key) but the path does not contain
        "goal", "todo", "task", or "commitment" — so commitment_persisted stays False.
        """
        before = {"vault/notes/note-1.md": "111"}
        after = {"vault/notes/note-1.md": "111", "vault/notes/note-2.md": "222"}
        bundle = _make_bundle(
            tmp_path,
            findings=_minimal_findings(),
            artifacts_before=before,
            artifacts_after=after,
        )
        result = swarm_evidence_facts(str(bundle))
        assert result["commitment_persisted"] is False


# ---------------------------------------------------------------------------
# test_path_traversal_refused — T-08-11 Security guard
# ---------------------------------------------------------------------------

class TestPathTraversalRefused:
    def test_etc_path_refused(self):
        """An absolute path outside subagent_runs/ must raise ScoringEvidenceError."""
        with pytest.raises(ScoringEvidenceError):
            swarm_evidence_facts("/etc")

    def test_parent_traversal_path_refused(self, tmp_path):
        """A path like tmp_path/.. that escapes subagent_runs/ must be refused."""
        with pytest.raises(ScoringEvidenceError):
            swarm_evidence_facts(str(tmp_path / ".."))

    def test_non_subagent_runs_dir_refused(self, tmp_path):
        """A valid directory that does not contain 'subagent_runs' in its path must be refused."""
        safe_dir = tmp_path / "some_other_dir"
        safe_dir.mkdir()
        with pytest.raises(ScoringEvidenceError):
            swarm_evidence_facts(str(safe_dir))

    def test_error_message_mentions_subagent_runs(self, tmp_path):
        """The error message must reference subagent_runs (clear security refusal)."""
        try:
            swarm_evidence_facts(str(tmp_path))
            pytest.fail("Expected ScoringEvidenceError was not raised")
        except ScoringEvidenceError as e:
            assert "subagent_runs" in str(e).lower(), (
                f"Error message should mention subagent_runs: {e}"
            )


# ---------------------------------------------------------------------------
# test_missing_findings_does_not_raise — SWARM-07 style safe defaults
# ---------------------------------------------------------------------------

class TestMissingFindingsDoesNotRaise:
    def test_empty_bundle_returns_safe_defaults(self, tmp_path):
        """A bundle with no findings.json must return safe-default facts without raising."""
        bundle = tmp_path / "subagent_runs" / "run-001" / "no-findings-persona"
        bundle.mkdir(parents=True, exist_ok=True)
        # No files written at all

        result = swarm_evidence_facts(str(bundle))

        # Must not raise, and must return all expected keys with safe defaults
        assert result["json_leak_detected"] is False
        assert result["failed_commands"] == []
        assert result["traceback_detected"] is False
        assert result["artifact_changes"] == []
        assert result["goals_referenced"] is False
        assert result["commitment_persisted"] is False
        assert result["total_actions"] == 0

    def test_corrupt_findings_json_returns_safe_defaults(self, tmp_path):
        """A bundle with malformed findings.json must return safe defaults without raising."""
        bundle = tmp_path / "subagent_runs" / "run-001" / "corrupt-persona"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "findings.json").write_text("NOT VALID JSON {{{")

        result = swarm_evidence_facts(str(bundle))
        assert result["total_actions"] == 0
        assert result["failed_commands"] == []

    def test_total_actions_from_findings(self, tmp_path):
        """total_actions must echo the value from findings.json."""
        findings = {
            "persona_id": "test",
            "total_actions": 42,
            "passed": 40,
            "failed": 2,
            "issues": [],
            "artifact_changes": [],
        }
        bundle = _make_bundle(tmp_path, findings=findings)
        result = swarm_evidence_facts(str(bundle))
        assert result["total_actions"] == 42

    def test_artifact_changes_echoed(self, tmp_path):
        """artifact_changes must be echoed from findings.json."""
        findings = {
            "total_actions": 3,
            "passed": 3,
            "failed": 0,
            "issues": [],
            "artifact_changes": [{"change": "goals.db modified"}],
        }
        bundle = _make_bundle(tmp_path, findings=findings)
        result = swarm_evidence_facts(str(bundle))
        assert result["artifact_changes"] == [{"change": "goals.db modified"}]
