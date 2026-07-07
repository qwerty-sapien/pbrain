# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Compact preview layout tests for roadmap-like learner artifacts."""

from __future__ import annotations

from types import SimpleNamespace

from rich.console import Console

from pb.cli.preview import build_step_table, markdown_learning_plan_lines
from pb.llm.drafts import ArtifactPresentationDraft, InstructionStep


def _render_text(renderable, *, width: int = 90) -> str:
    console = Console(width=width, record=True, color_system=None, force_terminal=False)
    console.print(renderable)
    return console.export_text()


def test_markdown_learning_plan_lines_use_bracketed_titles_and_short_bullets() -> None:
    blocks = [
        SimpleNamespace(
            node_id="step-1",
            title="Limits and continuity",
            branch="study",
            duration_minutes=45,
            success_check="Explain the epsilon-delta idea without notes.",
            reason="It unlocks continuity proofs and later derivative definitions.",
            depends_on=[],
        ),
        SimpleNamespace(
            node_id="step-2",
            title="Derivative drills",
            branch="practise",
            duration_minutes=30,
            success_check="Differentiate five mixed functions correctly.",
            reason="This turns symbolic rules into fast retrieval.",
            depends_on=["step-1"],
        ),
        SimpleNamespace(
            node_id="step-3",
            title="Transfer problems",
            branch="study",
            duration_minutes=25,
            success_check="Solve one unfamiliar problem cleanly.",
            reason="It checks whether the idea transfers beyond drills.",
            depends_on=["step-1"],
        ),
    ]

    rendered = _render_text(
        markdown_learning_plan_lines(
            blocks,
            presentation=ArtifactPresentationDraft(density="balanced", accent="cyan"),
        )
    )

    assert "[1] Limits and continuity  study · 45 min" in rendered
    assert "• Check: Explain the epsilon-delta idea without notes." in rendered
    assert "• Why: It unlocks continuity proofs and later derivative definitions." in rendered
    assert "[2] Derivative drills  practise · 30 min" in rendered
    assert "[3] Transfer problems  study · 25 min" in rendered
    assert "• Depends: Step 1" in rendered
    assert "step-1" not in rendered


def test_build_step_table_uses_compact_operator_style() -> None:
    steps = [
        InstructionStep(
            title="Diagnose the gap",
            instruction="Work one blind derivative example and write where you hesitated.",
            success_check="Name the exact rule you forgot before looking it up.",
        ),
        InstructionStep(
            title="Rebuild the rule",
            instruction="Derive the missing rule from a simpler example.",
            success_check="State the rule from memory and use it correctly once.",
        ),
    ]

    rendered = _render_text(
        build_step_table(
            steps,
            presentation=ArtifactPresentationDraft(density="balanced", accent="yellow"),
        )
    )

    assert "[1] Diagnose the gap" in rendered
    assert "• Do: Work one blind derivative example and write where you hesitated." in rendered
    assert "• Check: Name the exact rule you forgot before looking it up." in rendered
    assert "[2] Rebuild the rule" in rendered
