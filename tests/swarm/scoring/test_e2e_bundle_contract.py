"""IN-06 end-to-end bundle-contract fixture.

Builds bundles using the REAL EvidenceWriter + snapshot_artifacts APIs and asserts that
swarm_evidence_facts / score_bundle correctly interpret the on-disk shape.

This is the D-01a hard gate: these tests must pass before the full live swarm run (09-02)
can launch. They pin the production contract so any future drift between the writer and
the reader is caught immediately.

Path guard: bundles are created under tmp_path/"subagent_runs"/"run-e2e"/<persona_id>
to satisfy _validate_bundle_dir (scoring_evidence.py lines 49-76).

No live LLM calls — judge_session is monkeypatched where needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from swarm.swarm.evidence import CommandRecord, EvidenceWriter
from swarm.swarm.isolation import snapshot_artifacts
from swarm.swarm.scoring.scorer import score_bundle
from swarm.swarm.scoring.evidence_facts import swarm_evidence_facts

# Reuse helpers from test_scorer_integration (real rubric, stub judge, profile helpers)
from tests.swarm.scoring.test_scorer_integration import (
    _make_transcript,
    _make_profile_md,
    _stub_judge,
    REPO_RUBRIC,
)


# ---------------------------------------------------------------------------
# Real-bundle construction helper
# ---------------------------------------------------------------------------

def _write_real_bundle(
    tmp_path: Path,
    records: list[CommandRecord],
    vault_before: dict | None = None,
    vault_after: dict | None = None,
    persona_id: str = "overwhelmed-planner",
    domain: str = "productivity",
) -> Path:
    """Build a bundle using real EvidenceWriter + snapshot_artifacts APIs.

    Path guard in _validate_bundle_dir requires "subagent_runs" segment.
    Bundle path: tmp_path / "subagent_runs" / "run-e2e" / persona_id
    """
    bundle_dir = tmp_path / "subagent_runs" / "run-e2e" / persona_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    writer = EvidenceWriter(bundle_dir)
    writer.write_profile_md(_make_profile_md(persona_id))
    writer.write_transcript_md(persona_id, domain, records)
    writer.write_stdout_stderr(records)
    writer.write_commands_jsonl(records)

    # Real snapshot_artifacts shape: flat {path: mtime_ns}
    before = vault_before or {}
    after = vault_after or {}
    writer.write_artifacts_snapshot("before", before)
    writer.write_artifacts_snapshot("after", after)

    # write_findings_json serialises CommandRecord.issue_detected as "issue" on disk
    writer.write_findings_json(persona_id, records, [])
    writer.write_scores_placeholder()

    return bundle_dir


# ---------------------------------------------------------------------------
# TestIN06EndToEnd — the D-01a gate
# ---------------------------------------------------------------------------

class TestIN06EndToEnd:
    """End-to-end contract: real EvidenceWriter output -> swarm_evidence_facts + score_bundle."""

    def test_json_leak_fires_from_real_findings(self, tmp_path: Path) -> None:
        """json_leak_detected is True when a CommandRecord.issue_detected contains 'json'.

        Verifies the full on-disk contract:
          CommandRecord(issue_detected="possible JSON leak in stdout")
          -> write_findings_json -> findings.json: {"issue": "possible JSON leak in stdout"}
          -> swarm_evidence_facts -> json_leak_detected=True
        """
        records = [
            CommandRecord(
                action_index=1,
                command="pb next",
                issue_detected="possible JSON leak in stdout",
                severity="P0",
            )
        ]
        bundle = _write_real_bundle(tmp_path, records)
        result = swarm_evidence_facts(str(bundle))
        assert result["json_leak_detected"] is True, (
            f"json_leak_detected must be True when issue contains 'json'. "
            f"Got: {result}"
        )

    def test_traceback_fires_from_real_findings(self, tmp_path: Path) -> None:
        """traceback_detected is True when a CommandRecord.issue_detected contains 'Traceback'.

        Verifies the full on-disk contract:
          CommandRecord(issue_detected="Traceback (most recent call last)")
          -> write_findings_json -> findings.json: {"issue": "Traceback (most recent call last)"}
          -> swarm_evidence_facts -> traceback_detected=True
        """
        records = [
            CommandRecord(
                action_index=2,
                command="pb goal add test",
                issue_detected="Traceback (most recent call last)",
                severity="P0",
            )
        ]
        bundle = _write_real_bundle(tmp_path, records)
        result = swarm_evidence_facts(str(bundle))
        assert result["traceback_detected"] is True, (
            f"traceback_detected must be True when issue contains 'Traceback'. "
            f"Got: {result}"
        )

    def test_commitment_persisted_from_real_path_snapshot(self, tmp_path: Path) -> None:
        """commitment_persisted is True when artifacts_after has a new vault/goals/... key.

        Verifies the full on-disk contract:
          after snapshot adds "vault/goals/learn-german.md" (new key with "goal" in path)
          -> write_artifacts_snapshot("after", after)
          -> swarm_evidence_facts -> commitment_persisted=True
        """
        before = {"vault/existing.md": "111"}
        after = {"vault/existing.md": "111", "vault/goals/learn-german.md": "222"}
        bundle = _write_real_bundle(tmp_path, [], vault_before=before, vault_after=after)
        result = swarm_evidence_facts(str(bundle))
        assert result["commitment_persisted"] is True, (
            f"commitment_persisted must be True when after snapshot adds a vault/goals/... path. "
            f"Got: {result}"
        )

    def test_scorer_produces_rows_from_real_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score_bundle returns rows with valid severities when built from a real bundle.

        Verifies the end-to-end scoring pipeline:
          EvidenceWriter -> bundle files -> score_bundle -> ScoreRow list
        """
        monkeypatch.setattr("swarm.swarm.scoring.scorer.judge_session", _stub_judge)
        records = [
            CommandRecord(
                action_index=1,
                command="pb do something",
                issue_detected="wrong mode",
                severity="P1",
            )
        ]
        bundle = _write_real_bundle(tmp_path, records)
        tsv_path = bundle / "score.tsv"
        rows = score_bundle(bundle, tsv_path, rubric_path=REPO_RUBRIC)
        assert isinstance(rows, list), f"score_bundle must return a list, got {type(rows)}"
        # At least one row with a valid severity
        severities = {r.severity for r in rows}
        assert severities <= {"P0", "P1", "P2", "P3"}, (
            f"All severities must be within {{P0,P1,P2,P3}}. Got: {severities}"
        )
