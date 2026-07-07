"""Fix 1 — a judge blackout must be recorded loudly, never as silent 'clean'.

When judge_session returns None (provider call or verdict parse failed — commonly
429/RESOURCE_EXHAUSTED during scoring), score_bundle MUST:
  - write a judge_error.txt marker into the bundle, and
  - record judge_status="error" in score_meta.json

so the pillar report can count UNSCORED bundles separately from genuinely clean
ones. The canonical run had 6/10 personas come back with zero rows purely because
the judge returned None — indistinguishable from clean under the old code.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from swarm.swarm.scoring.judge import JudgeVerdict
from swarm.swarm.scoring.scorer import score_bundle
from tests.swarm.scoring.test_scorer_integration import (
    REPO_RUBRIC,
    _make_bundle,
    _stub_judge,
)


def _judge_returns_none(*args: Any, **kwargs: Any):
    return None


class TestJudgeStatusRecorded:
    def test_judge_none_writes_error_marker_and_status(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "swarm.swarm.scoring.scorer.judge_session", _judge_returns_none
        )
        bundle = _make_bundle(
            tmp_path,
            persona_id="what-now-user",
            pb_responses=["Do\n1. Study foo\n   because study is the safer default."],
            primary_intent="next-action-selection",
        )
        score_bundle(bundle, bundle / "score.tsv", rubric_path=REPO_RUBRIC)

        meta = json.loads((bundle / "score_meta.json").read_text())
        assert meta["judge_status"] == "error"
        assert (bundle / "judge_error.txt").exists()

    def test_judge_ok_status_and_no_error_marker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "swarm.swarm.scoring.scorer.judge_session", _stub_judge
        )
        bundle = _make_bundle(tmp_path)
        score_bundle(bundle, bundle / "score.tsv", rubric_path=REPO_RUBRIC)

        meta = json.loads((bundle / "score_meta.json").read_text())
        assert meta["judge_status"] == "ok"
        assert not (bundle / "judge_error.txt").exists()

    def test_judge_empty_verdict_distinct_from_error(self, tmp_path, monkeypatch):
        empty = JudgeVerdict(
            intent="x", mode_chosen="study-session", confidence="low", pillar_verdicts={}
        )
        monkeypatch.setattr(
            "swarm.swarm.scoring.scorer.judge_session", lambda *a, **k: empty
        )
        bundle = _make_bundle(
            tmp_path,
            persona_id="what-now-user",
            primary_intent="next-action-selection",
        )
        score_bundle(bundle, bundle / "score.tsv", rubric_path=REPO_RUBRIC)

        meta = json.loads((bundle / "score_meta.json").read_text())
        assert meta["judge_status"] == "empty"
        assert not (bundle / "judge_error.txt").exists()

    def test_stale_error_marker_cleared_on_successful_rescore(self, tmp_path, monkeypatch):
        # First pass: judge blacks out -> error marker written
        monkeypatch.setattr(
            "swarm.swarm.scoring.scorer.judge_session", _judge_returns_none
        )
        bundle = _make_bundle(tmp_path)
        score_bundle(bundle, bundle / "score.tsv", rubric_path=REPO_RUBRIC)
        assert (bundle / "judge_error.txt").exists()

        # Re-score with a working judge -> stale marker must be cleared
        monkeypatch.setattr(
            "swarm.swarm.scoring.scorer.judge_session", _stub_judge
        )
        score_bundle(bundle, bundle / "score.tsv", rubric_path=REPO_RUBRIC)
        meta = json.loads((bundle / "score_meta.json").read_text())
        assert meta["judge_status"] == "ok"
        assert not (bundle / "judge_error.txt").exists()
