# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Phase 13 self-improvement judge for specialised-agent instructions.

The dispatcher prompt remains context-guided and is never patched here.  This
module only stores and applies small, reversible instruction patches for
specialised agents such as review/accountability/domain agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import structlog

from pb.core.agent_weights import get_agent_weight_snapshot
from pb.core.models import generate_internal_id, utc_now
from pb.llm.drafts import AgentInstructionJudgeDraft
from pb.llm.structured import structured_output_call
from pb.storage.database import get_connection

logger = structlog.get_logger()

BUILTIN_DOMAIN_PREFIX = "builtin:"
PATCH_CONFIDENCE_THRESHOLD = 0.82
PATCH_MAX_CHARS = 600
STABLE_MIN_EVENTS = 4
STABLE_DISPATCH_PRIOR = 0.80
STABLE_COMPLETION_RATE = 0.80
SWEEP_LOOKBACK_DAYS = 7
SWEEP_PATCH_CAP = 5

_JUDGE_SYSTEM_PROMPT = """\
You are ProductiveBrain's self-improvement judge for specialised agents.

Scope:
- You may propose a small instruction patch for exactly one specialised agent.
- Do NOT patch the dispatcher, `pb do`, or `pb next`; those are context-guided.
- Prefer no patch when the evidence is weak, generic, or already handled by normal routing.

Decision rules:
- Return action="patch" only for a small, concrete, durable instruction change supported by evidence.
- Return action="clarify" when the feedback is real but ambiguous.
- Return action="none" when the agent appears stable or the evidence is not instruction-specific.
- Every patch must be reversible, brief, and phrased as an instruction for this one agent.
"""


@dataclass(frozen=True)
class AgentInstructionPatchRecord:
    """Persisted instruction patch proposal or applied patch."""

    id: str
    agent_id: str
    session_id: str
    status: str
    trigger_kind: str
    confidence: float
    summary: str
    instruction_patch: str
    previous_instruction: str
    clarifying_question: str
    evidence: tuple[str, ...]
    model_tier: str
    created_at: str
    applied_at: str
    reverted_at: str


def _storage_domain_for_agent(agent_id: str) -> str:
    normalized = (agent_id or "").strip()
    if normalized.startswith("domain_"):
        return normalized[len("domain_") :]
    return f"{BUILTIN_DOMAIN_PREFIX}{normalized}"


def _parse_config_json(raw: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dump_config_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _ensure_agent_config_row(conn, agent_id: str) -> dict[str, Any]:
    storage_domain = _storage_domain_for_agent(agent_id)
    row = conn.execute(
        "SELECT id, config_json FROM dispatch_agents WHERE domain = ?",
        (storage_domain,),
    ).fetchone()
    if row is not None:
        return {"id": row["id"], "config_json": row["config_json"] or "{}"}

    row_id = generate_internal_id()
    conn.execute(
        """
        INSERT INTO dispatch_agents (
            id, domain, goal_id, config_json, created_at, interaction_count
        ) VALUES (?, ?, NULL, '{}', ?, 0)
        """,
        (row_id, storage_domain, utc_now().isoformat()),
    )
    return {"id": row_id, "config_json": "{}"}


def active_agent_instruction_patch(agent_id: str, *, conn=None) -> str:
    """Return the active instruction patch text for one specialised agent."""
    storage_domain = _storage_domain_for_agent(agent_id)
    owns_connection = conn is None
    if conn is None:
        context = get_connection()
        conn = context.__enter__()
    else:
        context = None

    try:
        row = conn.execute(
            "SELECT config_json FROM dispatch_agents WHERE domain = ?",
            (storage_domain,),
        ).fetchone()
        if row is None:
            return ""
        payload = _parse_config_json(row["config_json"])
        value = payload.get("agent_instruction_patch")
        return value.strip() if isinstance(value, str) else ""
    finally:
        if owns_connection and context is not None:
            context.__exit__(None, None, None)


def agent_instruction_suffix(agent_id: str) -> str:
    """Render active agent-local guidance for prompt suffixes."""
    patch = active_agent_instruction_patch(agent_id)
    if not patch:
        return ""
    return (
        "\n\nActive user-specific instruction patch for this specialised agent only:\n"
        f"{patch.strip()}\n"
    )


def _patch_row_to_record(row) -> AgentInstructionPatchRecord:
    try:
        evidence = json.loads(row["evidence_json"] or "[]")
    except json.JSONDecodeError:
        evidence = []
    if not isinstance(evidence, list):
        evidence = []
    return AgentInstructionPatchRecord(
        id=row["id"],
        agent_id=row["agent_id"],
        session_id=row["session_id"] or "",
        status=row["status"],
        trigger_kind=row["trigger_kind"],
        confidence=float(row["confidence"] or 0.0),
        summary=row["summary"] or "",
        instruction_patch=row["instruction_patch"] or "",
        previous_instruction=row["previous_instruction"] or "",
        clarifying_question=row["clarifying_question"] or "",
        evidence=tuple(str(item) for item in evidence if str(item).strip()),
        model_tier=row["model_tier"] or "mid",
        created_at=row["created_at"] or "",
        applied_at=row["applied_at"] or "",
        reverted_at=row["reverted_at"] or "",
    )


def _unique_evidence(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        cleaned = (item or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique[:8]


def _mark_session_judged(session_id: str) -> None:
    if not session_id:
        return
    with get_connection() as conn:
        conn.execute(
            "UPDATE dispatch_sessions SET judged = 1, judged_at = ? WHERE id = ?",
            (utc_now().isoformat(), session_id),
        )
        conn.commit()


def _agent_event_summary(agent_id: str) -> dict[str, Any]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT event_kind, source_kind, created_at, metadata_json
            FROM agent_weight_events
            WHERE agent_id = ?
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (agent_id,),
        ).fetchall()

    positive = {"session_completed", "session_continued", "resume_selected"}
    negative = {"wrong", "kickback", "session_paused"}
    positive_count = sum(1 for row in rows if row["event_kind"] in positive)
    negative_count = sum(1 for row in rows if row["event_kind"] in negative)
    total = positive_count + negative_count
    completion_rate = (positive_count / total) if total else 0.0
    return {
        "total_rows": len(rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "completion_rate": completion_rate,
        "recent_events": [
            {
                "event_kind": row["event_kind"],
                "source_kind": row["source_kind"],
                "created_at": row["created_at"],
            }
            for row in rows[:8]
        ],
    }


def agent_is_stable(agent_id: str) -> bool:
    """Return True when Phase 13 should leave a well-fit agent alone."""
    summary = _agent_event_summary(agent_id)
    snapshot = get_agent_weight_snapshot(agent_id)
    return (
        summary["total_rows"] >= STABLE_MIN_EVENTS
        and summary["negative_count"] == 0
        and summary["completion_rate"] >= STABLE_COMPLETION_RATE
        and snapshot.dispatch_prior >= STABLE_DISPATCH_PRIOR
    )


def create_agent_instruction_patch(
    *,
    agent_id: str,
    session_id: str = "",
    trigger_kind: str = "manual",
    confidence: float = 0.0,
    summary: str = "",
    instruction_patch: str = "",
    clarifying_question: str = "",
    evidence: Optional[list[str]] = None,
    status: str = "proposed",
    model_tier: str = "mid",
) -> AgentInstructionPatchRecord:
    """Persist a proposed/clarifying agent-instruction patch record."""
    patch_id = generate_internal_id()
    now = utc_now().isoformat()
    evidence_json = json.dumps(_unique_evidence(evidence or []))
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_instruction_patches (
                id, agent_id, session_id, status, trigger_kind, confidence,
                summary, instruction_patch, previous_instruction,
                clarifying_question, evidence_json, model_tier, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?)
            """,
            (
                patch_id,
                agent_id,
                session_id or None,
                status,
                trigger_kind,
                float(confidence),
                summary.strip(),
                instruction_patch.strip(),
                clarifying_question.strip(),
                evidence_json,
                model_tier,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_instruction_patches WHERE id = ?",
            (patch_id,),
        ).fetchone()
    return _patch_row_to_record(row)


def apply_agent_instruction_patch(patch_id: str) -> AgentInstructionPatchRecord:
    """Apply a proposed patch to the target agent config."""
    now = utc_now().isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM agent_instruction_patches WHERE id = ?",
            (patch_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown agent instruction patch: {patch_id}")
        record = _patch_row_to_record(row)
        if not record.instruction_patch.strip():
            raise ValueError("Cannot apply an empty instruction patch")

        config_row = _ensure_agent_config_row(conn, record.agent_id)
        payload = _parse_config_json(config_row["config_json"])
        previous = str(payload.get("agent_instruction_patch") or "")
        payload["agent_instruction_patch"] = record.instruction_patch.strip()
        payload["agent_instruction_patch_id"] = record.id

        storage_domain = _storage_domain_for_agent(record.agent_id)
        conn.execute(
            "UPDATE dispatch_agents SET config_json = ? WHERE domain = ?",
            (_dump_config_json(payload), storage_domain),
        )
        conn.execute(
            """
            UPDATE agent_instruction_patches
            SET status = 'superseded'
            WHERE agent_id = ? AND status = 'applied' AND id != ?
            """,
            (record.agent_id, record.id),
        )
        conn.execute(
            """
            UPDATE agent_instruction_patches
            SET status = 'applied', previous_instruction = ?, applied_at = ?
            WHERE id = ?
            """,
            (previous, now, record.id),
        )
        conn.commit()
        refreshed = conn.execute(
            "SELECT * FROM agent_instruction_patches WHERE id = ?",
            (record.id,),
        ).fetchone()
    return _patch_row_to_record(refreshed)


def revert_agent_instruction_patch(patch_id: str) -> AgentInstructionPatchRecord:
    """Revert an applied patch using the previous instruction snapshot."""
    now = utc_now().isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM agent_instruction_patches WHERE id = ?",
            (patch_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown agent instruction patch: {patch_id}")
        record = _patch_row_to_record(row)

        config_row = _ensure_agent_config_row(conn, record.agent_id)
        payload = _parse_config_json(config_row["config_json"])
        current_patch_id = payload.get("agent_instruction_patch_id")
        if current_patch_id == record.id:
            if record.previous_instruction.strip():
                payload["agent_instruction_patch"] = record.previous_instruction.strip()
                payload.pop("agent_instruction_patch_id", None)
            else:
                payload.pop("agent_instruction_patch", None)
                payload.pop("agent_instruction_patch_id", None)

        storage_domain = _storage_domain_for_agent(record.agent_id)
        conn.execute(
            "UPDATE dispatch_agents SET config_json = ? WHERE domain = ?",
            (_dump_config_json(payload), storage_domain),
        )
        conn.execute(
            """
            UPDATE agent_instruction_patches
            SET status = 'reverted', reverted_at = ?
            WHERE id = ?
            """,
            (now, record.id),
        )
        conn.commit()
        refreshed = conn.execute(
            "SELECT * FROM agent_instruction_patches WHERE id = ?",
            (record.id,),
        ).fetchone()
    return _patch_row_to_record(refreshed)


def list_agent_instruction_patches(
    *,
    agent_id: str = "",
    limit: int = 20,
) -> list[AgentInstructionPatchRecord]:
    """Return recent instruction patch records for CLI inspection."""
    params: list[object] = []
    where = ""
    if agent_id:
        where = "WHERE agent_id = ?"
        params.append(agent_id)
    params.append(max(1, min(int(limit), 100)))
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM agent_instruction_patches
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_patch_row_to_record(row) for row in rows]


def _unjudged_sessions(
    *,
    agent_id: str = "",
    lookback_days: int = SWEEP_LOOKBACK_DAYS,
    limit: int = 50,
) -> list[dict[str, str]]:
    cutoff = (utc_now() - timedelta(days=max(1, lookback_days))).isoformat()
    params: list[object] = [cutoff]
    agent_filter = ""
    if agent_id:
        agent_filter = "AND agent_id = ?"
        params.append(agent_id)
    params.append(max(1, min(int(limit), 200)))
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, agent_id, status, context_json, created_at, updated_at
            FROM dispatch_sessions
            WHERE judged = 0
              AND updated_at >= ?
              AND agent_id NOT IN ('capture', 'next_action')
              {agent_filter}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [
        {
            "id": row["id"],
            "agent_id": row["agent_id"],
            "status": row["status"],
            "context_json": row["context_json"] or "{}",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


async def sweep_agent_instruction_judge(
    *,
    agent_id: str = "",
    lookback_days: int = SWEEP_LOOKBACK_DAYS,
    cap: int = SWEEP_PATCH_CAP,
) -> list[AgentInstructionPatchRecord]:
    """Review recent unjudged sessions and produce proposed patches.

    Sweep patches are intentionally not auto-applied.  A cron job can run this
    command weekly and leave the resulting digest for human review.
    """
    sessions = _unjudged_sessions(
        agent_id=agent_id,
        lookback_days=lookback_days,
        limit=max(cap * 4, cap, 1),
    )
    records: list[AgentInstructionPatchRecord] = []
    inspected = 0
    for session in sessions:
        if inspected >= cap:
            break
        inspected += 1
        feedback_text = (
            "Weekly instruction-fit sweep for a specialised agent session. "
            "Propose a patch only when the session evidence shows durable "
            "agent-instruction misfit."
        )
        evidence = [
            f"session_id: {session['id']}",
            f"agent_id: {session['agent_id']}",
            f"status: {session['status']}",
            f"updated_at: {session['updated_at']}",
            f"context_json: {session['context_json'][:500]}",
        ]
        record = await judge_agent_instruction_fit(
            agent_id=session["agent_id"],
            session_id=session["id"],
            feedback_text=feedback_text,
            evidence=evidence,
            trigger_kind="weekly_sweep",
            auto_apply=False,
        )
        if record is not None:
            records.append(record)
    return records


def write_agent_instruction_digest(
    vault_path: Path,
    records: list[AgentInstructionPatchRecord],
    *,
    lookback_days: int = SWEEP_LOOKBACK_DAYS,
) -> Path:
    """Write a reviewable Markdown digest for sweep results."""
    digest_dir = vault_path / "direction" / "preferences" / "agent-judge-digests"
    digest_dir.mkdir(parents=True, exist_ok=True)
    path = digest_dir / f"{utc_now().strftime('%Y-%m-%d')}-agent-judge.md"
    lines = [
        "---",
        "type: agent_instruction_judge_digest",
        f"updated: {utc_now().strftime('%Y-%m-%d')}",
        f"lookback_days: {lookback_days}",
        "---",
        "",
        "# Agent Instruction Judge Digest",
        "",
    ]
    if not records:
        lines.append("No agent-instruction patches were proposed this sweep.")
    else:
        for record in records:
            lines.extend(
                [
                    f"## {record.agent_id} - {record.status}",
                    "",
                    f"- Patch id: `{record.id}`",
                    f"- Confidence: {record.confidence:.2f}",
                    f"- Summary: {record.summary or '-'}",
                    f"- Apply: `pb config agents apply {record.id}`",
                    f"- Revert after apply: `pb config agents revert {record.id}`",
                    "",
                    "### Instruction Patch",
                    "",
                    record.instruction_patch or "_No patch text._",
                    "",
                    "### Evidence",
                    "",
                ]
            )
            lines.extend(f"- {item}" for item in record.evidence)
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _build_judge_prompt(
    *,
    agent_id: str,
    session_id: str,
    feedback_text: str,
    evidence: list[str],
) -> str:
    summary = _agent_event_summary(agent_id)
    active_patch = active_agent_instruction_patch(agent_id)
    return (
        f"Agent id: {agent_id}\n"
        f"Session id: {session_id or 'none'}\n"
        f"Current active instruction patch: {active_patch or 'none'}\n"
        f"Feedback or misalignment signal:\n{feedback_text.strip() or 'none'}\n\n"
        f"Evidence:\n"
        + "\n".join(f"- {item}" for item in _unique_evidence(evidence))
        + "\n\n"
        f"Recent agent event summary:\n{json.dumps(summary, sort_keys=True)}\n\n"
        "Decide whether this one specialised agent needs a reversible instruction patch."
    )


def _should_auto_apply(draft: AgentInstructionJudgeDraft) -> bool:
    patch = (draft.instruction_patch or "").strip()
    return (
        draft.action == "patch"
        and draft.confidence >= PATCH_CONFIDENCE_THRESHOLD
        and bool(patch)
        and len(patch) <= PATCH_MAX_CHARS
    )


async def judge_agent_instruction_fit(
    *,
    agent_id: str,
    session_id: str = "",
    feedback_text: str = "",
    evidence: Optional[list[str]] = None,
    trigger_kind: str = "manual",
    auto_apply: bool = True,
) -> Optional[AgentInstructionPatchRecord]:
    """Run the production judge and optionally apply a high-confidence patch.

    Returns None when the agent is stable, evidence is insufficient, or the LLM
    is unavailable/rate-limited.  That degradation is intentional and non-fatal.
    """
    if not agent_id.strip():
        return None
    evidence_items = _unique_evidence([feedback_text, *(evidence or [])])
    if not feedback_text.strip() and not evidence_items:
        return None

    if agent_is_stable(agent_id):
        _mark_session_judged(session_id)
        logger.info("agent_instruction_judge.stable_noop", agent_id=agent_id)
        return None

    prompt = _build_judge_prompt(
        agent_id=agent_id,
        session_id=session_id,
        feedback_text=feedback_text,
        evidence=evidence_items,
    )
    try:
        draft = await structured_output_call(
            prompt,
            AgentInstructionJudgeDraft,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            tier="mid",
        )
    except Exception as exc:
        logger.warning(
            "agent_instruction_judge.call_failed",
            agent_id=agent_id,
            error=str(exc),
        )
        return None

    if draft is None or draft.action == "none":
        _mark_session_judged(session_id)
        return None

    citations = _unique_evidence([*evidence_items, *draft.evidence_citations])
    if draft.action == "clarify":
        record = create_agent_instruction_patch(
            agent_id=agent_id,
            session_id=session_id,
            trigger_kind=trigger_kind,
            confidence=draft.confidence,
            summary=draft.summary,
            clarifying_question=draft.clarifying_question,
            evidence=citations,
            status="clarify",
            model_tier="mid",
        )
        _mark_session_judged(session_id)
        return record

    record = create_agent_instruction_patch(
        agent_id=agent_id,
        session_id=session_id,
        trigger_kind=trigger_kind,
        confidence=draft.confidence,
        summary=draft.summary,
        instruction_patch=draft.instruction_patch,
        evidence=citations,
        status="proposed",
        model_tier="mid",
    )
    if auto_apply and _should_auto_apply(draft):
        record = apply_agent_instruction_patch(record.id)
    _mark_session_judged(session_id)
    return record


def format_patch_announcement(record: AgentInstructionPatchRecord) -> str:
    """Return a concise user-visible patch announcement."""
    if record.status == "applied":
        undo = f"pb config agents revert {record.id}"
        return (
            f"Agent instruction updated for {record.agent_id}: {record.summary}\n"
            f"Evidence: {'; '.join(record.evidence) or 'session evidence'}\n"
            f"Undo: `{undo}`"
        )
    if record.status == "clarify":
        question = record.clarifying_question or "What should this agent do differently?"
        return f"Agent instruction judge needs one clarification: {question}"
    return f"Agent instruction patch proposed for {record.agent_id}: {record.summary}"
