# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio

from pb.core.agent_instruction_judge import (
    active_agent_instruction_patch,
    agent_instruction_suffix,
    apply_agent_instruction_patch,
    create_agent_instruction_patch,
    judge_agent_instruction_fit,
    revert_agent_instruction_patch,
    sweep_agent_instruction_judge,
)
from pb.core.agent_weights import record_agent_weight_event
from pb.core.models import utc_now
from pb.llm.drafts import AgentInstructionJudgeDraft
from pb.storage.database import get_connection


def test_patch_apply_revert_updates_active_agent_instruction(temp_db) -> None:
    record = create_agent_instruction_patch(
        agent_id="review",
        summary="Use concise review prompts.",
        instruction_patch="Keep review prompts concise and action-focused.",
        evidence=["User said reviews feel too long."],
    )

    assert active_agent_instruction_patch("review") == ""

    applied = apply_agent_instruction_patch(record.id)
    assert applied.status == "applied"
    assert active_agent_instruction_patch("review") == (
        "Keep review prompts concise and action-focused."
    )
    assert "Keep review prompts concise" in agent_instruction_suffix("review")

    reverted = revert_agent_instruction_patch(record.id)
    assert reverted.status == "reverted"
    assert active_agent_instruction_patch("review") == ""


def test_judge_auto_applies_high_confidence_small_patch(temp_db, monkeypatch) -> None:
    async def fake_structured_output_call(*args, **kwargs):
        return AgentInstructionJudgeDraft(
            action="patch",
            confidence=0.91,
            summary="Prefer concrete commitments.",
            instruction_patch="When commitment wording is vague, ask for the concrete deliverable first.",
            evidence_citations=["User marked vague accountability as wrong."],
        )

    monkeypatch.setattr(
        "pb.core.agent_instruction_judge.structured_output_call",
        fake_structured_output_call,
    )
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO dispatch_sessions
                (id, agent_id, status, context_json, created_at, updated_at)
            VALUES ('sess-judge', 'accountability', 'complete', '{}', ?, ?)
            """,
            ("2026-06-24T00:00:00", "2026-06-24T00:00:00"),
        )
        conn.commit()

    record = asyncio.run(
        judge_agent_instruction_fit(
            agent_id="accountability",
            session_id="sess-judge",
            feedback_text="The agent accepted a vague commitment.",
            evidence=["intent: keep me accountable"],
        )
    )

    assert record is not None
    assert record.status == "applied"
    assert "concrete deliverable" in active_agent_instruction_patch("accountability")
    with get_connection() as conn:
        row = conn.execute(
            "SELECT judged, judged_at FROM dispatch_sessions WHERE id = 'sess-judge'",
        ).fetchone()
    assert row["judged"] == 1
    assert row["judged_at"]


def test_stable_agent_gets_no_instruction_patch(temp_db, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("stable agents should not call the LLM judge")

    monkeypatch.setattr(
        "pb.core.agent_instruction_judge.structured_output_call",
        fail_if_called,
    )
    for _ in range(4):
        record_agent_weight_event(
            "review",
            "session_completed",
            source_kind="human",
            created_at="2026-06-24T00:00:00",
        )

    record = asyncio.run(
        judge_agent_instruction_fit(
            agent_id="review",
            feedback_text="Reviews are basically working.",
            evidence=["stable positive history"],
        )
    )

    assert record is None


def test_judge_records_clarification_request(temp_db, monkeypatch) -> None:
    async def fake_structured_output_call(*args, **kwargs):
        return AgentInstructionJudgeDraft(
            action="clarify",
            confidence=0.61,
            summary="Feedback is ambiguous.",
            clarifying_question="Should the review agent be shorter or more critical?",
            evidence_citations=["User said review felt off."],
        )

    monkeypatch.setattr(
        "pb.core.agent_instruction_judge.structured_output_call",
        fake_structured_output_call,
    )

    record = asyncio.run(
        judge_agent_instruction_fit(
            agent_id="review",
            feedback_text="The review felt off.",
            evidence=["No concrete patch target."],
        )
    )

    assert record is not None
    assert record.status == "clarify"
    assert "shorter or more critical" in record.clarifying_question
    assert active_agent_instruction_patch("review") == ""


def test_weekly_sweep_proposes_without_auto_apply(temp_db, monkeypatch) -> None:
    async def fake_structured_output_call(*args, **kwargs):
        return AgentInstructionJudgeDraft(
            action="patch",
            confidence=0.88,
            summary="Ask for the preferred review scope.",
            instruction_patch="When review scope is ambiguous, ask whether the user wants day or week.",
            evidence_citations=["Unjudged review session had ambiguous context."],
        )

    monkeypatch.setattr(
        "pb.core.agent_instruction_judge.structured_output_call",
        fake_structured_output_call,
    )
    now = utc_now().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO dispatch_sessions
                (id, agent_id, status, context_json, created_at, updated_at)
            VALUES ('sess-sweep', 'review', 'complete', '{"intent":"review"}', ?, ?)
            """,
            (now, now),
        )
        conn.commit()

    records = asyncio.run(sweep_agent_instruction_judge(cap=1))

    assert len(records) == 1
    assert records[0].status == "proposed"
    assert active_agent_instruction_patch("review") == ""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT judged FROM dispatch_sessions WHERE id = 'sess-sweep'",
        ).fetchone()
    assert row["judged"] == 1
