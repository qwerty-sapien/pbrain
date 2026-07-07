"""The canonical-run launcher glue: --canonical preset + post-run report writing.

The launcher chains already-tested pieces (orchestrator run -> per-bundle scoring
-> report.py). These tests cover the new glue only: the canonical preset config
and the post-run report writer. The live LLM run itself is covered by the
orchestrator dry-run tests and is not re-exercised here.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SWARM_PARENT = REPO_ROOT if (REPO_ROOT / "swarm").exists() else REPO_ROOT.parent
if str(SWARM_PARENT) not in sys.path:
    sys.path.insert(0, str(SWARM_PARENT))

from swarm.swarm.__main__ import _canonical_config, _write_run_report
from swarm.swarm.scoring.schema import ScoreRow, write_score_tsv


def test_canonical_config_is_all_personas_14_actions_with_report():
    cfg = _canonical_config()
    assert cfg["all"] is True
    assert cfg["max_actions"] == 14
    assert cfg["report"] is True


def _seed_bundle(run_dir: Path, persona: str, rows: list[ScoreRow]) -> None:
    b = run_dir / persona
    b.mkdir(parents=True, exist_ok=True)
    write_score_tsv(rows, b / "score.tsv")
    (b / "commands.jsonl").write_text("")  # so it counts as a bundle


def _row(pillar: str, selection: str) -> ScoreRow:
    return ScoreRow(
        input_pattern="do/action-selector", intent="x", mode_chosen="todo",
        selection=selection, execution="pass" if selection == "right" else "fail",
        confidence="high", pillar=pillar, severity="P3" if selection == "right" else "P1",
        repro="p#a0",
    )


def test_write_run_report_emits_report_and_rollup(tmp_path):
    run = tmp_path / "20260604_000000"
    _seed_bundle(run, "p1", [_row("Reliability", "right"), _row("Productivity", "wrong")])

    verdict = _write_run_report(run)

    report_md = run / "PILLAR_REPORT.md"
    rollup = run / "findings_rollup.tsv"
    assert report_md.exists() and rollup.exists()
    text = report_md.read_text()
    assert "Ship gate" in text
    assert "Per-pillar verdicts" in text
    # The helper returns a one-line verdict string for the CLI to print.
    assert "Ship gate" in verdict
