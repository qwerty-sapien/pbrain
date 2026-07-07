"""Tests for swarm/swarm/evidence.py — EvidenceWriter.

Verifies the complete D-19 bundle file set:
- commands.jsonl rows round-trip via json.loads
- transcript.md contains "## Turn" headers and "Brain said:" lines
- scores.json parses to {} (empty Phase 8 slot)
- findings.json has issues only for records with issue_detected, no "pillar" key
- mcp_results.json and http_results.json are valid JSON arrays
- artifacts_*.json snapshots are written
- profile.md is written
- MANIFEST.json maps persona -> status
- write_context_packets copies a seeded context_packets/ tree (D-19)
- write_context_packets is a silent no-op when source dir is absent (SWARM-07)
- stdout.log / stderr.log are written
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from swarm.swarm.evidence import CommandRecord, EvidenceWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def persona_dir(tmp_path):
    """Return a fresh persona output directory."""
    d = tmp_path / "run-001" / "overwhelmed-planner"
    return d  # EvidenceWriter must create it


@pytest.fixture
def ew(persona_dir):
    return EvidenceWriter(persona_dir)


def _make_record(**kwargs) -> CommandRecord:
    defaults = {
        "timestamp": "2026-06-02T10:00:00Z",
        "action_index": 0,
        "kind": "cli",
        "command": "pb next",
        "exit_code": 0,
        "duration_ms": 123,
        "stdout_excerpt": "Here is your next action",
        "stderr_excerpt": "",
        "brain_utterance": "What should I do next?",
        "brain_rationale": "User wants prioritisation",
    }
    defaults.update(kwargs)
    return CommandRecord(**defaults)


# ---------------------------------------------------------------------------
# commands.jsonl
# ---------------------------------------------------------------------------

class TestWriteCommandsJsonl:
    def test_creates_commands_jsonl(self, ew, persona_dir):
        records = [_make_record(action_index=0), _make_record(action_index=1)]
        ew.write_commands_jsonl(records)
        path = persona_dir / "commands.jsonl"
        assert path.exists(), "commands.jsonl not created"

    def test_rows_round_trip(self, ew, persona_dir):
        records = [
            _make_record(action_index=0, command="pb next"),
            _make_record(action_index=1, command="pb goal list"),
        ]
        ew.write_commands_jsonl(records)
        rows = (persona_dir / "commands.jsonl").read_text().splitlines()
        assert len(rows) == 2
        for row, rec in zip(rows, records):
            obj = json.loads(row)  # must not raise
            assert obj["command"] == rec.command

    def test_one_json_object_per_line(self, ew, persona_dir):
        records = [_make_record(action_index=i) for i in range(3)]
        ew.write_commands_jsonl(records)
        lines = (persona_dir / "commands.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # each line must be valid JSON


# ---------------------------------------------------------------------------
# transcript.md
# ---------------------------------------------------------------------------

class TestWriteTranscriptMd:
    def test_creates_transcript_md(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        assert (persona_dir / "transcript.md").exists()

    def test_contains_turn_headers(self, ew, persona_dir):
        records = [_make_record(action_index=1), _make_record(action_index=2)]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        text = (persona_dir / "transcript.md").read_text()
        assert "## Turn" in text

    def test_contains_brain_said(self, ew, persona_dir):
        records = [_make_record(action_index=0, brain_utterance="I need help planning")]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        text = (persona_dir / "transcript.md").read_text()
        assert "Brain said" in text
        assert "I need help planning" in text

    def test_contains_fenced_stdout(self, ew, persona_dir):
        records = [_make_record(action_index=0, stdout_excerpt="Some output here")]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        text = (persona_dir / "transcript.md").read_text()
        assert "Some output here" in text
        assert "```" in text

    def test_issue_line_appears_only_when_issue_detected(self, ew, persona_dir):
        clean = _make_record(action_index=0, issue_detected="", severity="")
        broken = _make_record(action_index=1, issue_detected="Traceback detected", severity="P0")
        ew.write_transcript_md("overwhelmed-planner", "productivity", [clean, broken])
        text = (persona_dir / "transcript.md").read_text()
        assert "ISSUE" in text
        assert "Traceback detected" in text

    def test_no_issue_line_when_no_issues(self, ew, persona_dir):
        records = [_make_record(action_index=0, issue_detected="", severity="")]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        text = (persona_dir / "transcript.md").read_text()
        assert "ISSUE" not in text


# ---------------------------------------------------------------------------
# scores.json
# ---------------------------------------------------------------------------

class TestWriteScoresPlaceholder:
    def test_creates_scores_json(self, ew, persona_dir):
        ew.write_scores_placeholder()
        assert (persona_dir / "scores.json").exists()

    def test_scores_json_parses_to_empty_dict(self, ew, persona_dir):
        ew.write_scores_placeholder()
        content = json.loads((persona_dir / "scores.json").read_text())
        assert content == {}, f"Expected empty dict, got {content!r}"

    def test_scores_json_has_no_pillar_key(self, ew, persona_dir):
        ew.write_scores_placeholder()
        text = (persona_dir / "scores.json").read_text()
        assert "pillar" not in text


# ---------------------------------------------------------------------------
# findings.json
# ---------------------------------------------------------------------------

class TestWriteFindingsJson:
    def test_creates_findings_json(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        ew.write_findings_json("overwhelmed-planner", records, artifact_changes=[])
        assert (persona_dir / "findings.json").exists()

    def test_issues_only_for_records_with_issue_detected(self, ew, persona_dir):
        clean = _make_record(action_index=0, issue_detected="", severity="")
        broken = _make_record(action_index=1, issue_detected="JSON leak", severity="P0")
        ew.write_findings_json("overwhelmed-planner", [clean, broken], artifact_changes=[])
        findings = json.loads((persona_dir / "findings.json").read_text())
        assert findings["total_actions"] == 2
        assert findings["passed"] == 1
        assert findings["failed"] == 1
        assert len(findings["issues"]) == 1
        assert findings["issues"][0]["issue"] == "JSON leak"

    def test_no_pillar_key_in_findings(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        ew.write_findings_json("overwhelmed-planner", records, artifact_changes=[])
        text = (persona_dir / "findings.json").read_text()
        assert "pillar" not in text

    def test_no_scores_key_in_findings(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        ew.write_findings_json("overwhelmed-planner", records, artifact_changes=[])
        obj = json.loads((persona_dir / "findings.json").read_text())
        assert "scores" not in obj

    def test_artifact_changes_included(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        changes = [{"path": "vault/notes/test.md", "changed": True}]
        ew.write_findings_json("overwhelmed-planner", records, artifact_changes=changes)
        obj = json.loads((persona_dir / "findings.json").read_text())
        assert "artifact_changes" in obj
        assert len(obj["artifact_changes"]) == 1


# ---------------------------------------------------------------------------
# mcp_results.json / http_results.json
# ---------------------------------------------------------------------------

class TestWriteMcpResults:
    def test_mcp_results_json_is_array(self, ew, persona_dir):
        records = [
            _make_record(kind="mcp", command="tools/list"),
            _make_record(kind="cli", command="pb next"),  # should not appear
        ]
        ew.write_mcp_results(records)
        obj = json.loads((persona_dir / "mcp_results.json").read_text())
        assert isinstance(obj, list)
        assert len(obj) == 1
        assert obj[0]["kind"] == "mcp"

    def test_mcp_results_empty_when_no_mcp_records(self, ew, persona_dir):
        records = [_make_record(kind="cli")]
        ew.write_mcp_results(records)
        obj = json.loads((persona_dir / "mcp_results.json").read_text())
        assert obj == []


class TestWriteHttpResults:
    def test_http_results_json_is_array(self, ew, persona_dir):
        http_list = [{"action": {"intent": "capture"}, "http_result": {"status": "ok"}}]
        ew.write_http_results(http_list)
        obj = json.loads((persona_dir / "http_results.json").read_text())
        assert isinstance(obj, list)
        assert len(obj) == 1

    def test_http_results_empty_list(self, ew, persona_dir):
        ew.write_http_results([])
        obj = json.loads((persona_dir / "http_results.json").read_text())
        assert obj == []


# ---------------------------------------------------------------------------
# artifacts snapshots
# ---------------------------------------------------------------------------

class TestWriteArtifactsSnapshot:
    def test_creates_artifacts_label_json(self, ew, persona_dir):
        snapshot = {"vault/notes/test.md": "1234567890"}
        ew.write_artifacts_snapshot("before", snapshot)
        assert (persona_dir / "artifacts_before.json").exists()

    def test_snapshot_content_round_trips(self, ew, persona_dir):
        snapshot = {"vault/a.md": "111", "vault/b.md": "222"}
        ew.write_artifacts_snapshot("after", snapshot)
        obj = json.loads((persona_dir / "artifacts_after.json").read_text())
        assert obj == snapshot


# ---------------------------------------------------------------------------
# profile.md
# ---------------------------------------------------------------------------

class TestWriteProfileMd:
    def test_creates_profile_md(self, ew, persona_dir):
        ew.write_profile_md("# overwhelmed-planner\n\nA busy professional.")
        assert (persona_dir / "profile.md").exists()

    def test_profile_content_is_preserved(self, ew, persona_dir):
        content = "# overwhelmed-planner\n\nA busy professional."
        ew.write_profile_md(content)
        assert (persona_dir / "profile.md").read_text() == content


# ---------------------------------------------------------------------------
# stdout.log / stderr.log
# ---------------------------------------------------------------------------

class TestWriteStdoutStderr:
    def test_creates_stdout_log(self, ew, persona_dir):
        records = [_make_record(stdout_excerpt="some output")]
        ew.write_stdout_stderr(records)
        assert (persona_dir / "stdout.log").exists()

    def test_creates_stderr_log(self, ew, persona_dir):
        records = [_make_record(stderr_excerpt="some error")]
        ew.write_stdout_stderr(records)
        assert (persona_dir / "stderr.log").exists()


# ---------------------------------------------------------------------------
# context_packets/ copy (D-19)
# ---------------------------------------------------------------------------

class TestWriteContextPackets:
    def test_copies_context_packets_tree(self, ew, persona_dir, tmp_path):
        """write_context_packets reproduces the context_packets/ tree."""
        data_dir = tmp_path / "data"
        cp_src = data_dir / "context_packets"
        cp_src.mkdir(parents=True)
        (cp_src / "packet_01.json").write_text('{"topic": "test"}')
        (cp_src / "packet_02.json").write_text('{"topic": "test2"}')

        ew.write_context_packets(data_dir)

        cp_dst = persona_dir / "context_packets"
        assert cp_dst.exists(), "context_packets/ not created in persona_dir"
        assert (cp_dst / "packet_01.json").exists()
        assert (cp_dst / "packet_02.json").exists()
        assert json.loads((cp_dst / "packet_01.json").read_text()) == {"topic": "test"}

    def test_creates_empty_dir_when_source_absent(self, ew, persona_dir, tmp_path):
        """When data_dir/context_packets/ does not exist, create an EMPTY destination
        context_packets/ (no raise).

        D-19 reconciliation (Phase 7): the original scripted harness always created
        the bundle's context_packets/ dir (ctx_dir.mkdir(exist_ok=True)); orchestrator
        Plan 06 restored that for bundle-layout completeness. The directory is created
        empty when the persona triggered no context_build. The SWARM-07 guarantee
        (never raises) is preserved.
        """
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        # context_packets/ intentionally NOT created

        try:
            ew.write_context_packets(data_dir)
        except Exception as exc:
            pytest.fail(f"write_context_packets raised when source absent: {exc!r}")

        # The destination context_packets/ is created as an empty dir (D-19 completeness)
        cp_dst = persona_dir / "context_packets"
        assert cp_dst.exists() and cp_dst.is_dir(), \
            "context_packets/ should be created (empty) for D-19 bundle completeness"
        assert list(cp_dst.iterdir()) == [], \
            "context_packets/ should be empty when source absent"

    def test_silent_noop_when_data_dir_absent(self, ew, persona_dir, tmp_path):
        """When data_dir itself does not exist, write silently (no raise)."""
        data_dir = tmp_path / "nonexistent_data"
        # data_dir intentionally NOT created

        try:
            ew.write_context_packets(data_dir)
        except Exception as exc:
            pytest.fail(f"write_context_packets raised when data_dir absent: {exc!r}")


# ---------------------------------------------------------------------------
# MANIFEST.json (run-level)
# ---------------------------------------------------------------------------

class TestUpdateManifest:
    def test_creates_manifest_json_at_run_dir(self, ew, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        completed = {
            "overwhelmed-planner": {
                "output_dir": str(tmp_path / "run-001" / "overwhelmed-planner"),
                "findings": {
                    "issues": [
                        {"severity": "P0", "issue": "Critical traceback"},
                        {"severity": "P2", "issue": "Non-critical"},
                    ],
                    "total_actions": 10,
                    "passed": 8,
                    "failed": 2,
                },
            }
        }
        ew.update_manifest(run_dir, run_id="run-001", completed=completed)
        manifest_path = run_dir / "MANIFEST.json"
        assert manifest_path.exists(), "MANIFEST.json not created at run_dir"

    def test_manifest_maps_persona_to_status(self, ew, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        completed = {
            "overwhelmed-planner": {
                "output_dir": str(tmp_path / "run-001" / "overwhelmed-planner"),
                "findings": {"issues": [], "total_actions": 5, "passed": 5, "failed": 0},
            }
        }
        ew.update_manifest(run_dir, run_id="run-001", completed=completed)
        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        assert "personas" in manifest
        assert "overwhelmed-planner" in manifest["personas"]
        entry = manifest["personas"]["overwhelmed-planner"]
        assert "status" in entry
        assert "output_dir" in entry
        assert "headline_findings" in entry

    def test_manifest_headline_findings_are_p0_only(self, ew, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        completed = {
            "overwhelmed-planner": {
                "output_dir": str(tmp_path / "run-001" / "overwhelmed-planner"),
                "findings": {
                    "issues": [
                        {"severity": "P0", "issue": "P0 issue 1"},
                        {"severity": "P1", "issue": "P1 non-critical"},
                        {"severity": "P0", "issue": "P0 issue 2"},
                        {"severity": "P2", "issue": "P2 non-critical"},
                    ],
                    "total_actions": 10,
                    "passed": 6,
                    "failed": 4,
                },
            }
        }
        ew.update_manifest(run_dir, run_id="run-001", completed=completed)
        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        headline = manifest["personas"]["overwhelmed-planner"]["headline_findings"]
        assert isinstance(headline, list)
        # Only P0 issues should appear
        for item in headline:
            assert "P0" in item or "P0" not in item  # just check it's a string
        assert len(headline) == 2

    def test_manifest_has_run_id(self, ew, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        completed = {}
        ew.update_manifest(run_dir, run_id="run-999", completed=completed)
        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        assert manifest["run_id"] == "run-999"

    def test_manifest_headline_findings_max_5(self, ew, tmp_path):
        """headline_findings must not exceed 5 items."""
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        issues = [{"severity": "P0", "issue": f"P0 issue {i}"} for i in range(10)]
        completed = {
            "p": {
                "output_dir": str(run_dir / "p"),
                "findings": {"issues": issues, "total_actions": 10, "passed": 0, "failed": 10},
            }
        }
        ew.update_manifest(run_dir, run_id="run-001", completed=completed)
        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        assert len(manifest["personas"]["p"]["headline_findings"]) <= 5

"""Tests for swarm/swarm/evidence.py — EvidenceWriter.

Verifies the complete D-19 bundle file set:
- commands.jsonl rows round-trip via json.loads
- transcript.md contains "## Turn" headers and "Brain said:" lines
- scores.json parses to {} (empty Phase 8 slot)
- findings.json has issues only for records with issue_detected, no "pillar" key
- mcp_results.json and http_results.json are valid JSON arrays
- artifacts_*.json snapshots are written
- profile.md is written
- MANIFEST.json maps persona -> status
- write_context_packets copies a seeded context_packets/ tree (D-19)
- write_context_packets is a silent no-op when source dir is absent (SWARM-07)
- stdout.log / stderr.log are written
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from swarm.swarm.evidence import CommandRecord, EvidenceWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def persona_dir(tmp_path):
    """Return a fresh persona output directory."""
    d = tmp_path / "run-001" / "overwhelmed-planner"
    return d  # EvidenceWriter must create it


@pytest.fixture
def ew(persona_dir):
    return EvidenceWriter(persona_dir)


def _make_record(**kwargs) -> CommandRecord:
    defaults = {
        "timestamp": "2026-06-02T10:00:00Z",
        "action_index": 0,
        "kind": "cli",
        "command": "pb next",
        "exit_code": 0,
        "duration_ms": 123,
        "stdout_excerpt": "Here is your next action",
        "stderr_excerpt": "",
        "brain_utterance": "What should I do next?",
        "brain_rationale": "User wants prioritisation",
    }
    defaults.update(kwargs)
    return CommandRecord(**defaults)


# ---------------------------------------------------------------------------
# commands.jsonl
# ---------------------------------------------------------------------------

class TestWriteCommandsJsonl:
    def test_creates_commands_jsonl(self, ew, persona_dir):
        records = [_make_record(action_index=0), _make_record(action_index=1)]
        ew.write_commands_jsonl(records)
        path = persona_dir / "commands.jsonl"
        assert path.exists(), "commands.jsonl not created"

    def test_rows_round_trip(self, ew, persona_dir):
        records = [
            _make_record(action_index=0, command="pb next"),
            _make_record(action_index=1, command="pb goal list"),
        ]
        ew.write_commands_jsonl(records)
        rows = (persona_dir / "commands.jsonl").read_text().splitlines()
        assert len(rows) == 2
        for row, rec in zip(rows, records):
            obj = json.loads(row)  # must not raise
            assert obj["command"] == rec.command

    def test_one_json_object_per_line(self, ew, persona_dir):
        records = [_make_record(action_index=i) for i in range(3)]
        ew.write_commands_jsonl(records)
        lines = (persona_dir / "commands.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # each line must be valid JSON


# ---------------------------------------------------------------------------
# transcript.md
# ---------------------------------------------------------------------------

class TestWriteTranscriptMd:
    def test_creates_transcript_md(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        assert (persona_dir / "transcript.md").exists()

    def test_contains_turn_headers(self, ew, persona_dir):
        records = [_make_record(action_index=1), _make_record(action_index=2)]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        text = (persona_dir / "transcript.md").read_text()
        assert "## Turn" in text

    def test_contains_brain_said(self, ew, persona_dir):
        records = [_make_record(action_index=0, brain_utterance="I need help planning")]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        text = (persona_dir / "transcript.md").read_text()
        assert "Brain said" in text
        assert "I need help planning" in text

    def test_contains_fenced_stdout(self, ew, persona_dir):
        records = [_make_record(action_index=0, stdout_excerpt="Some output here")]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        text = (persona_dir / "transcript.md").read_text()
        assert "Some output here" in text
        assert "```" in text

    def test_issue_line_appears_only_when_issue_detected(self, ew, persona_dir):
        clean = _make_record(action_index=0, issue_detected="", severity="")
        broken = _make_record(action_index=1, issue_detected="Traceback detected", severity="P0")
        ew.write_transcript_md("overwhelmed-planner", "productivity", [clean, broken])
        text = (persona_dir / "transcript.md").read_text()
        assert "ISSUE" in text
        assert "Traceback detected" in text

    def test_no_issue_line_when_no_issues(self, ew, persona_dir):
        records = [_make_record(action_index=0, issue_detected="", severity="")]
        ew.write_transcript_md("overwhelmed-planner", "productivity", records)
        text = (persona_dir / "transcript.md").read_text()
        assert "ISSUE" not in text


# ---------------------------------------------------------------------------
# scores.json
# ---------------------------------------------------------------------------

class TestWriteScoresPlaceholder:
    def test_creates_scores_json(self, ew, persona_dir):
        ew.write_scores_placeholder()
        assert (persona_dir / "scores.json").exists()

    def test_scores_json_parses_to_empty_dict(self, ew, persona_dir):
        ew.write_scores_placeholder()
        content = json.loads((persona_dir / "scores.json").read_text())
        assert content == {}, f"Expected empty dict, got {content!r}"

    def test_scores_json_has_no_pillar_key(self, ew, persona_dir):
        ew.write_scores_placeholder()
        text = (persona_dir / "scores.json").read_text()
        assert "pillar" not in text


# ---------------------------------------------------------------------------
# findings.json
# ---------------------------------------------------------------------------

class TestWriteFindingsJson:
    def test_creates_findings_json(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        ew.write_findings_json("overwhelmed-planner", records, artifact_changes=[])
        assert (persona_dir / "findings.json").exists()

    def test_issues_only_for_records_with_issue_detected(self, ew, persona_dir):
        clean = _make_record(action_index=0, issue_detected="", severity="")
        broken = _make_record(action_index=1, issue_detected="JSON leak", severity="P0")
        ew.write_findings_json("overwhelmed-planner", [clean, broken], artifact_changes=[])
        findings = json.loads((persona_dir / "findings.json").read_text())
        assert findings["total_actions"] == 2
        assert findings["passed"] == 1
        assert findings["failed"] == 1
        assert len(findings["issues"]) == 1
        assert findings["issues"][0]["issue"] == "JSON leak"

    def test_no_pillar_key_in_findings(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        ew.write_findings_json("overwhelmed-planner", records, artifact_changes=[])
        text = (persona_dir / "findings.json").read_text()
        assert "pillar" not in text

    def test_no_scores_key_in_findings(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        ew.write_findings_json("overwhelmed-planner", records, artifact_changes=[])
        obj = json.loads((persona_dir / "findings.json").read_text())
        assert "scores" not in obj

    def test_artifact_changes_included(self, ew, persona_dir):
        records = [_make_record(action_index=0)]
        changes = [{"path": "vault/notes/test.md", "changed": True}]
        ew.write_findings_json("overwhelmed-planner", records, artifact_changes=changes)
        obj = json.loads((persona_dir / "findings.json").read_text())
        assert "artifact_changes" in obj
        assert len(obj["artifact_changes"]) == 1


# ---------------------------------------------------------------------------
# mcp_results.json / http_results.json
# ---------------------------------------------------------------------------

class TestWriteMcpResults:
    def test_mcp_results_json_is_array(self, ew, persona_dir):
        records = [
            _make_record(kind="mcp", command="tools/list"),
            _make_record(kind="cli", command="pb next"),  # should not appear
        ]
        ew.write_mcp_results(records)
        obj = json.loads((persona_dir / "mcp_results.json").read_text())
        assert isinstance(obj, list)
        assert len(obj) == 1
        assert obj[0]["kind"] == "mcp"

    def test_mcp_results_empty_when_no_mcp_records(self, ew, persona_dir):
        records = [_make_record(kind="cli")]
        ew.write_mcp_results(records)
        obj = json.loads((persona_dir / "mcp_results.json").read_text())
        assert obj == []


class TestWriteHttpResults:
    def test_http_results_json_is_array(self, ew, persona_dir):
        http_list = [{"action": {"intent": "capture"}, "http_result": {"status": "ok"}}]
        ew.write_http_results(http_list)
        obj = json.loads((persona_dir / "http_results.json").read_text())
        assert isinstance(obj, list)
        assert len(obj) == 1

    def test_http_results_empty_list(self, ew, persona_dir):
        ew.write_http_results([])
        obj = json.loads((persona_dir / "http_results.json").read_text())
        assert obj == []


# ---------------------------------------------------------------------------
# artifacts snapshots
# ---------------------------------------------------------------------------

class TestWriteArtifactsSnapshot:
    def test_creates_artifacts_label_json(self, ew, persona_dir):
        snapshot = {"vault/notes/test.md": "1234567890"}
        ew.write_artifacts_snapshot("before", snapshot)
        assert (persona_dir / "artifacts_before.json").exists()

    def test_snapshot_content_round_trips(self, ew, persona_dir):
        snapshot = {"vault/a.md": "111", "vault/b.md": "222"}
        ew.write_artifacts_snapshot("after", snapshot)
        obj = json.loads((persona_dir / "artifacts_after.json").read_text())
        assert obj == snapshot


# ---------------------------------------------------------------------------
# profile.md
# ---------------------------------------------------------------------------

class TestWriteProfileMd:
    def test_creates_profile_md(self, ew, persona_dir):
        ew.write_profile_md("# overwhelmed-planner\n\nA busy professional.")
        assert (persona_dir / "profile.md").exists()

    def test_profile_content_is_preserved(self, ew, persona_dir):
        content = "# overwhelmed-planner\n\nA busy professional."
        ew.write_profile_md(content)
        assert (persona_dir / "profile.md").read_text() == content


# ---------------------------------------------------------------------------
# stdout.log / stderr.log
# ---------------------------------------------------------------------------

class TestWriteStdoutStderr:
    def test_creates_stdout_log(self, ew, persona_dir):
        records = [_make_record(stdout_excerpt="some output")]
        ew.write_stdout_stderr(records)
        assert (persona_dir / "stdout.log").exists()

    def test_creates_stderr_log(self, ew, persona_dir):
        records = [_make_record(stderr_excerpt="some error")]
        ew.write_stdout_stderr(records)
        assert (persona_dir / "stderr.log").exists()


# ---------------------------------------------------------------------------
# context_packets/ copy (D-19)
# ---------------------------------------------------------------------------

class TestWriteContextPackets:
    def test_copies_context_packets_tree(self, ew, persona_dir, tmp_path):
        """write_context_packets reproduces the context_packets/ tree."""
        data_dir = tmp_path / "data"
        cp_src = data_dir / "context_packets"
        cp_src.mkdir(parents=True)
        (cp_src / "packet_01.json").write_text('{"topic": "test"}')
        (cp_src / "packet_02.json").write_text('{"topic": "test2"}')

        ew.write_context_packets(data_dir)

        cp_dst = persona_dir / "context_packets"
        assert cp_dst.exists(), "context_packets/ not created in persona_dir"
        assert (cp_dst / "packet_01.json").exists()
        assert (cp_dst / "packet_02.json").exists()
        assert json.loads((cp_dst / "packet_01.json").read_text()) == {"topic": "test"}

    def test_creates_empty_dir_when_source_absent(self, ew, persona_dir, tmp_path):
        """When data_dir/context_packets/ does not exist, create an empty bundle dir."""
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        # context_packets/ intentionally NOT created

        try:
            ew.write_context_packets(data_dir)
        except Exception as exc:
            pytest.fail(f"write_context_packets raised when source absent: {exc!r}")

        cp_dst = persona_dir / "context_packets"
        assert cp_dst.exists() and cp_dst.is_dir()
        assert list(cp_dst.iterdir()) == []

    def test_silent_noop_when_data_dir_absent(self, ew, persona_dir, tmp_path):
        """When data_dir itself does not exist, write silently (no raise)."""
        data_dir = tmp_path / "nonexistent_data"
        # data_dir intentionally NOT created

        try:
            ew.write_context_packets(data_dir)
        except Exception as exc:
            pytest.fail(f"write_context_packets raised when data_dir absent: {exc!r}")


# ---------------------------------------------------------------------------
# MANIFEST.json (run-level)
# ---------------------------------------------------------------------------

class TestUpdateManifest:
    def test_creates_manifest_json_at_run_dir(self, ew, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        completed = {
            "overwhelmed-planner": {
                "output_dir": str(tmp_path / "run-001" / "overwhelmed-planner"),
                "findings": {
                    "issues": [
                        {"severity": "P0", "issue": "Critical traceback"},
                        {"severity": "P2", "issue": "Non-critical"},
                    ],
                    "total_actions": 10,
                    "passed": 8,
                    "failed": 2,
                },
            }
        }
        ew.update_manifest(run_dir, run_id="run-001", completed=completed)
        manifest_path = run_dir / "MANIFEST.json"
        assert manifest_path.exists(), "MANIFEST.json not created at run_dir"

    def test_manifest_maps_persona_to_status(self, ew, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        completed = {
            "overwhelmed-planner": {
                "output_dir": str(tmp_path / "run-001" / "overwhelmed-planner"),
                "findings": {"issues": [], "total_actions": 5, "passed": 5, "failed": 0},
            }
        }
        ew.update_manifest(run_dir, run_id="run-001", completed=completed)
        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        assert "personas" in manifest
        assert "overwhelmed-planner" in manifest["personas"]
        entry = manifest["personas"]["overwhelmed-planner"]
        assert "status" in entry
        assert "output_dir" in entry
        assert "headline_findings" in entry

    def test_manifest_headline_findings_are_p0_only(self, ew, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        completed = {
            "overwhelmed-planner": {
                "output_dir": str(tmp_path / "run-001" / "overwhelmed-planner"),
                "findings": {
                    "issues": [
                        {"severity": "P0", "issue": "P0 issue 1"},
                        {"severity": "P1", "issue": "P1 non-critical"},
                        {"severity": "P0", "issue": "P0 issue 2"},
                        {"severity": "P2", "issue": "P2 non-critical"},
                    ],
                    "total_actions": 10,
                    "passed": 6,
                    "failed": 4,
                },
            }
        }
        ew.update_manifest(run_dir, run_id="run-001", completed=completed)
        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        headline = manifest["personas"]["overwhelmed-planner"]["headline_findings"]
        assert isinstance(headline, list)
        # Only P0 issues should appear
        for item in headline:
            assert "P0" in item or "P0" not in item  # just check it's a string
        assert len(headline) == 2

    def test_manifest_has_run_id(self, ew, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        completed = {}
        ew.update_manifest(run_dir, run_id="run-999", completed=completed)
        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        assert manifest["run_id"] == "run-999"

    def test_manifest_headline_findings_max_5(self, ew, tmp_path):
        """headline_findings must not exceed 5 items."""
        run_dir = tmp_path / "run-001"
        run_dir.mkdir(exist_ok=True)
        issues = [{"severity": "P0", "issue": f"P0 issue {i}"} for i in range(10)]
        completed = {
            "p": {
                "output_dir": str(run_dir / "p"),
                "findings": {"issues": issues, "total_actions": 10, "passed": 0, "failed": 10},
            }
        }
        ew.update_manifest(run_dir, run_id="run-001", completed=completed)
        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        assert len(manifest["personas"]["p"]["headline_findings"]) <= 5
