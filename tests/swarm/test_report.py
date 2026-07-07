"""Tests for swarm/swarm/report.py — the reusable v1.1 pillar-report module.

Focus is the correctness-critical logic the hand-written "lean inline artifact"
got wrong:
  - SC#1 clean-run attestation must scan PER pb-command and EXCLUDE the harness
    MCP-handshake probe (productivebrain-mcp), whose JSON-RPC output is expected.
    The prior inline scan read the concatenated stdout.log and miscounted that
    handshake as a 10/10 "raw-JSON leak".
  - Per-pillar right/wrong tally aggregated from every bundle's score.tsv.
  - Ship gate = every pillar good-or-better.
"""

from __future__ import annotations

import json
from pathlib import Path

from swarm.swarm import report
from swarm.swarm.scoring.schema import ScoreRow, write_score_tsv


# ---------------------------------------------------------------------------
# MCP-handshake identification + raw-JSON leak heuristic
# ---------------------------------------------------------------------------

class TestMcpHandshakeExclusion:
    def test_identifies_mcp_probe_by_binary_name(self):
        cmd = "/Users/agent/.local/bin/productivebrain-mcp --config /tmp/c.toml"
        assert report.is_mcp_handshake_command(cmd) is True

    def test_pb_command_is_not_mcp(self):
        assert report.is_mcp_handshake_command("/Users/agent/.local/bin/pb do hi") is False

    def test_empty_command_is_not_mcp(self):
        assert report.is_mcp_handshake_command("") is False


class TestRawJsonLeakHeuristic:
    def test_jsonrpc_envelope_is_leak(self):
        assert report.looks_like_raw_json_leak('{"jsonrpc":"2.0","id":1,"result":{"x":1}}')

    def test_plain_json_object_is_leak(self):
        assert report.looks_like_raw_json_leak('{"error": "boom", "code": 500}')

    def test_human_menu_is_not_leak(self):
        menu = "Do\n1. Shape today's learning plan\n2. Study eigenvalues\n"
        assert report.looks_like_raw_json_leak(menu) is False

    def test_prose_with_braces_is_not_leak(self):
        assert report.looks_like_raw_json_leak("Use a HashMap<K, V> { ... } in Rust") is False

    def test_empty_is_not_leak(self):
        assert report.looks_like_raw_json_leak("") is False
        assert report.looks_like_raw_json_leak("   ") is False


# ---------------------------------------------------------------------------
# Per-command bundle scan (the SC#1 fix)
# ---------------------------------------------------------------------------

class TestScanBundleJsonLeaks:
    def _bundle(self, tmp_path: Path, records: list[dict]) -> Path:
        b = tmp_path / "persona-x"
        b.mkdir()
        (b / "commands.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        return b

    def test_mcp_handshake_json_is_excluded(self, tmp_path):
        recs = [
            {"action_index": 0, "command": "/x/pb do hi",
             "stdout_excerpt": "Do\n1. Study X", "stderr_excerpt": ""},
            {"action_index": 4, "command": "/x/productivebrain-mcp --config c.toml",
             "stdout_excerpt": '{"jsonrpc":"2.0","id":1,"result":{}}', "stderr_excerpt": ""},
        ]
        leaks = report.scan_bundle_json_leaks(self._bundle(tmp_path, recs))
        assert leaks == [], f"the only JSON is the excluded MCP probe: {leaks}"

    def test_real_pb_json_leak_is_flagged(self, tmp_path):
        recs = [
            {"action_index": 0, "command": "/x/pb next",
             "stdout_excerpt": '{"tasks": [], "next": null}', "stderr_excerpt": ""},
        ]
        leaks = report.scan_bundle_json_leaks(self._bundle(tmp_path, recs))
        assert len(leaks) == 1
        assert leaks[0].action_index == 0
        assert leaks[0].stream == "stdout"

    def test_json_on_stderr_is_flagged(self, tmp_path):
        recs = [
            {"action_index": 1, "command": "/x/pb review day",
             "stdout_excerpt": "Daily review", "stderr_excerpt": '{"trace": "x"}'},
        ]
        leaks = report.scan_bundle_json_leaks(self._bundle(tmp_path, recs))
        assert len(leaks) == 1 and leaks[0].stream == "stderr"


class TestScanBundleTracebacks:
    def _bundle(self, tmp_path: Path, records: list[dict]) -> Path:
        b = tmp_path / "persona-tb"
        b.mkdir()
        (b / "commands.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        return b

    def test_python_traceback_flagged_even_on_mcp(self, tmp_path):
        # A traceback is always a defect, including from the MCP probe.
        recs = [
            {"action_index": 4, "command": "/x/productivebrain-mcp --config c.toml",
             "stdout_excerpt": "", "stderr_excerpt": "Traceback (most recent call last):\n  File ..."},
        ]
        tbs = report.scan_bundle_tracebacks(self._bundle(tmp_path, recs))
        assert len(tbs) == 1

    def test_clean_output_no_traceback(self, tmp_path):
        recs = [{"action_index": 0, "command": "/x/pb do hi",
                 "stdout_excerpt": "Do\n1. Study X", "stderr_excerpt": ""}]
        assert report.scan_bundle_tracebacks(self._bundle(tmp_path, recs)) == []


# ---------------------------------------------------------------------------
# Pillar aggregation + ship gate
# ---------------------------------------------------------------------------

def _write_bundle_scores(run_dir: Path, persona: str, rows: list[ScoreRow]) -> None:
    b = run_dir / persona
    b.mkdir(parents=True, exist_ok=True)
    write_score_tsv(rows, b / "score.tsv")


def _row(pillar: str, selection: str) -> ScoreRow:
    execution = "pass" if selection == "right" else "fail"
    return ScoreRow(
        input_pattern="do/action-selector", intent="x", mode_chosen="todo",
        selection=selection, execution=execution, confidence="high",
        pillar=pillar, severity="P3" if selection == "right" else "P1",
        repro="p#a0",
    )


class TestAggregatePillars:
    def test_tallies_right_and_wrong_across_bundles(self, tmp_path):
        run = tmp_path / "run"
        _write_bundle_scores(run, "p1", [_row("Productivity", "wrong"), _row("Reliability", "right")])
        _write_bundle_scores(run, "p2", [_row("Productivity", "right"), _row("Reliability", "right")])
        tally = report.aggregate_pillars(run)
        assert tally["Productivity"].right == 1
        assert tally["Productivity"].wrong == 1
        assert tally["Reliability"].right == 2
        assert tally["Reliability"].wrong == 0

    def test_unscored_pillar_has_zero_tally(self, tmp_path):
        run = tmp_path / "run"
        _write_bundle_scores(run, "p1", [_row("Reliability", "right")])
        tally = report.aggregate_pillars(run)
        assert tally["Learning"].right == 0 and tally["Learning"].wrong == 0


class TestShipGate:
    def test_gate_not_met_when_any_pillar_below_good(self, tmp_path):
        run = tmp_path / "run"
        _write_bundle_scores(run, "p1", [_row("Productivity", "wrong"), _row("Reliability", "right")])
        tally = report.aggregate_pillars(run)
        met, good_pillars, _ = report.ship_gate(tally)
        assert met is False
        assert "Reliability" in good_pillars
        assert "Productivity" not in good_pillars

    def test_gate_met_only_when_all_scored_pillars_good(self, tmp_path):
        run = tmp_path / "run"
        # Every pillar all-right => all good. (Unscored pillars are not certifiable,
        # so a run with unscored pillars must NOT pass — guarded separately.)
        for p in report.PILLARS:
            _write_bundle_scores(run, f"b-{p}", [_row(p, "right")])
        tally = report.aggregate_pillars(run)
        met, good_pillars, _ = report.ship_gate(tally)
        assert met is True
        assert set(good_pillars) == set(report.PILLARS)
