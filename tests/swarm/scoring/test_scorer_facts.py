"""The judge's mechanical facts (json_leak / traceback) must be derived from the
SAME correct per-command scan as the SC#1 report — NOT from the usually-empty
findings.issues heuristic, and NOT counting the harness MCP-handshake probe.

This is the safe half of the WR-01/WR-02 facts-shape blocker: json_leak_detected
and traceback_detected. (commitment_persisted / goals_referenced remain open —
they need vault-taxonomy domain logic and can only be validated by a live run.)
"""

from __future__ import annotations

import json
from pathlib import Path

from swarm.swarm.scoring.scorer import build_mechanical_facts


def _bundle(tmp_path: Path, name: str, records: list[dict], findings: dict | None = None) -> Path:
    b = tmp_path / name
    b.mkdir()
    (b / "commands.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    if findings is not None:
        (b / "findings.json").write_text(json.dumps(findings))
    return b


def test_mcp_handshake_json_is_not_counted_as_leak(tmp_path):
    recs = [
        {"action_index": 0, "command": "/x/pb do hi",
         "stdout_excerpt": "Do\n1. Study X", "stderr_excerpt": ""},
        {"action_index": 4, "command": "/x/productivebrain-mcp --config c.toml",
         "stdout_excerpt": '{"jsonrpc":"2.0","id":1,"result":{}}', "stderr_excerpt": ""},
    ]
    b = _bundle(tmp_path, "p", recs, findings={"total_actions": 5, "issues": []})
    facts = build_mechanical_facts(b, {"total_actions": 5, "issues": []})
    assert facts["json_leak_detected"] is False
    assert facts["traceback_detected"] is False
    assert facts["total_actions"] == 5


def test_real_pb_json_leak_sets_flag(tmp_path):
    recs = [{"action_index": 0, "command": "/x/pb next",
             "stdout_excerpt": '{"tasks": []}', "stderr_excerpt": ""}]
    b = _bundle(tmp_path, "p", recs, findings={"issues": []})
    facts = build_mechanical_facts(b, {"issues": []})
    assert facts["json_leak_detected"] is True


def test_traceback_sets_flag(tmp_path):
    recs = [{"action_index": 0, "command": "/x/pb do hi",
             "stdout_excerpt": "", "stderr_excerpt": "Traceback (most recent call last):\n ..."}]
    b = _bundle(tmp_path, "p", recs, findings={"issues": []})
    facts = build_mechanical_facts(b, {"issues": []})
    assert facts["traceback_detected"] is True


def test_failed_commands_still_carried_from_findings(tmp_path):
    recs = [{"action_index": 0, "command": "/x/pb do hi", "stdout_excerpt": "ok", "stderr_excerpt": ""}]
    findings = {"issues": [{"command": "pb x", "severity": "P0", "issue": "boom"}]}
    b = _bundle(tmp_path, "p", recs, findings=findings)
    facts = build_mechanical_facts(b, findings)
    assert facts["failed_commands"] == [{"command": "pb x", "severity": "P0", "issue": "boom"}]
