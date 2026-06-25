# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Semantic ProductiveBrain MCP tools."""

from __future__ import annotations

from dataclasses import is_dataclass, asdict
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from pb.cli.commands.anki import _resolve_deck_and_domain
from pb.cli.context_runtime import attach_active_context, ingest_context_source
from pb.cli.commands.notes import _collect_moves, apply_moves, _parse_frontmatter
from pb.cli.commands.review import _collect_review_metrics
from pb.core.action_routing import build_next_candidates, route_learning_intent, suggest_commands_for_intent
from pb.core.clock import utc_now
from pb.core.context_file_intake import (
    active_context_from_bundle,
    active_context_from_sources,
    inspect_context_files,
    plan_context_file_response,
)
from pb.core.feedback_profile import (
    feedback_profile_path,
    load_feedback_guidance,
    normalize_feedback_scope,
    save_feedback_profile,
)
from pb.core.intake import create_task
from pb.core.naming import (
    apply_generated_names,
    apply_generated_title,
    deterministic_names,
    stored_display_title,
)
from pb.core.models import GoalArc, Task
from pb.mcp.context import get_mcp_context, get_runtime_context
from pb.mcp.pending import (
    _bypassing,
    queue_pending,
    queue_response,
    register_impl,
)
from pb.mcp.server import mcp
from pb.sessions.repo import SessionRepoAdapter
from pb.sessions.service import SessionService
from pb.llm.runtime import LLMRuntime
from pb.storage.config import get_quarantine_path
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository


class MCPWriteError(RuntimeError):
    """Raised when a write tool is used without --allow-writes."""


def _safe_dict(obj: Any) -> Any:
    """Coerce dataclasses/enums/datetime/Path into MCP-serializable primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {k: _safe_dict(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe_dict(v) for v in obj]
    # Last resort: try attribute dict
    if hasattr(obj, "__dict__"):
        return {k: _safe_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _require_writes() -> None:
    if not get_mcp_context().allow_writes:
        raise MCPWriteError("This MCP server is running in read-only mode. Restart with --allow-writes.")


def _bootstrap_repo() -> tuple[Any, Repository]:
    runtime = get_runtime_context()
    set_db_path(runtime.db_path)
    init_db(runtime.db_path)
    return runtime, Repository()


def _session_service(repo: Repository) -> SessionService:
    return SessionService(repo=SessionRepoAdapter(repo))


def _context_cmd_ctx(runtime, repo):
    return SimpleNamespace(repo=repo, runtime=runtime, config=runtime.config)


_TOOL_CATALOG: dict[str, str] = {
    "vault_current": "read_only",
    "vault_profile": "read_only",
    "goal_draft_from_text": "read_only",
    "goal_commit_draft": "tier_2_queued_write",
    "goal_list": "read_only",
    "plan_day": "read_only",
    "next_action": "read_only",
    "feedback_capture": "tier_1_write",
    "do_route": "read_only",
    "context_file_inspect": "read_only",
    "context_file_ingest": "tier_1_write",
    "context_file_status": "read_only",
    "source_bundle_list": "read_only",
    "source_bundle_show": "read_only",
    "context_lock": "tier_1_write",
    "context_unlock": "tier_1_write",
    "context_status": "read_only",
    "study_start": "tier_2_queued_write",
    "practise_start": "tier_2_queued_write",
    "teach_start": "tier_2_queued_write",
    "learn_start": "tier_2_queued_write",
    "learn_with_context": "tier_1_write",
    "session_status": "read_only",
    "session_pause": "tier_2_queued_write",
    "session_resume": "tier_2_queued_write",
    "session_finish": "tier_2_queued_write",
    "anki_generate_candidates": "tier_2_queued_write",
    "anki_candidate_list": "read_only",
    "anki_candidate_status_counts": "read_only",
    "anki_candidate_update": "tier_1_write",
    "anki_export_status": "read_only",
    "anki_export": "tier_1_write",
    "context_build": "read_only",
    "vault_link_graph": "read_only",
    "pb_command": "debug_escape_hatch",
}


def _goal_to_dict(goal: GoalArc) -> dict[str, Any]:
    return goal.model_dump(mode="json")


def _session_payload(session: Any, task: Task | None = None, *, service: SessionService | None = None) -> dict[str, Any]:
    payload = _safe_dict(session) if session is not None else {}
    if task is not None:
        payload["task"] = task.model_dump(mode="json")
    if service is not None and session is not None:
        payload["elapsed_minutes"] = service.get_elapsed_minutes()
        payload["remaining_minutes"] = service.get_remaining_minutes()
    return payload


def _matches_domain(*values: str, domain: str) -> bool:
    needle = (domain or "").strip().lower()
    if not needle:
        return True
    return any(needle in (value or "").lower() for value in values)


def _recent_session_rows(repo: Repository, *, domain: str = "", limit: int = 5) -> list[dict[str, Any]]:
    rows: list[tuple[datetime, dict[str, Any]]] = []
    for task in repo.list_tasks():
        task_goals = [repo.get_goal_arc(goal_id) for goal_id in getattr(task, "linked_goal_arc_ids", [])]
        goal_domains = [goal.domain for goal in task_goals if goal is not None and getattr(goal, "domain", "")]
        for session in repo.list_sessions_for_task(task.id):
            if not _matches_domain(
                getattr(task, "title", ""),
                getattr(task, "description", ""),
                getattr(session, "subject_scope", "") or "",
                " ".join(goal_domains),
                domain=domain,
            ):
                continue
            rows.append(
                (
                    session.start_at,
                    {
                        "session_id": session.id,
                        "task_id": task.id,
                        "task_title": task.title,
                        "branch": getattr(session, "branch", "study"),
                        "subject_scope": getattr(session, "subject_scope", "") or task.title,
                        "start_at": session.start_at.isoformat() if getattr(session, "start_at", None) else None,
                        "end_at": session.end_at.isoformat() if getattr(session, "end_at", None) else None,
                        "actual_outcome": getattr(session, "actual_outcome", "") or "",
                        "observed_errors": getattr(session, "observed_errors", "") or "",
                        "next_adjustment": getattr(session, "next_adjustment", "") or "",
                    },
                )
            )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in rows[:limit]]


def _resumeable_tasks(repo: Repository) -> list[Task]:
    tasks: list[Task] = []
    active_session = repo.get_active_session()
    active_task_id = active_session.task_id if active_session is not None else None
    for task in repo.list_tasks():
        if task.archived_at is not None or task.completion >= 100 or task.id == active_task_id:
            continue
        if task.state.value == "paused" or repo.list_sessions_for_task(task.id):
            tasks.append(task)
    return tasks


def _resolve_resume_task(repo: Repository, task_id: str = "") -> Task | None:
    resumable = _resumeable_tasks(repo)
    if not resumable:
        return None
    if not task_id:
        resumable.sort(key=lambda task: task.updated_at or task.created_at, reverse=True)
        return resumable[0]
    exact = next((task for task in resumable if task.id == task_id), None)
    if exact is not None:
        return exact
    matches = [task for task in resumable if task.id.startswith(task_id)]
    if len(matches) == 1:
        return matches[0]
    return None


@mcp.tool()
def vault_current() -> dict[str, Any]:
    """Return the currently served vault profile."""
    runtime, _ = _bootstrap_repo()
    return {
        "vault": runtime.vault_name,
        "vault_path": str(runtime.vault_path),
        "data_dir": str(runtime.data_dir),
        "quarantine_path": str(runtime.quarantine_path),
        "allow_writes": get_mcp_context().allow_writes,
    }


@mcp.tool()
def vault_profile(name: str = "") -> dict[str, Any]:
    """Return one vault profile or the active profile when omitted."""
    runtime, _ = _bootstrap_repo()
    config = runtime.config
    selected = name or runtime.vault_name
    profile = config.vaults[selected]
    return {
        "name": selected,
        "path": profile.path,
        "data_dir": profile.data_dir,
        "quarantine_folder": profile.quarantine_folder,
        "active": selected == runtime.vault_name,
    }


@mcp.tool()
def tool_catalog() -> dict[str, Any]:
    """Return tool classifications for external MCP clients."""
    return {
        "tools": [
            {"name": name, "classification": classification}
            for name, classification in sorted(_TOOL_CATALOG.items())
        ]
    }


@mcp.tool()
def goal_draft_from_text(text: str) -> dict[str, Any]:
    """Create a lightweight goal draft from plain text."""
    normalized = " ".join((text or "").split())
    execution_mode = "mixed"
    lowered = normalized.lower()
    if any(token in lowered for token in ("speak", "perform", "play", "drill", "practise", "practice")):
        execution_mode = "practise"
    elif any(token in lowered for token in ("understand", "learn", "study", "theory", "concept")):
        execution_mode = "study"
    return {
        "title": normalized,
        "description": normalized,
        "execution_mode": execution_mode,
        "status": "draft",
        "preview": True,
    }


def _do_goal_commit_draft(
    title: str,
    description: str = "",
    execution_mode: str = "mixed",
    domain: str = "",
) -> dict[str, Any]:
    """Actual goal-commit implementation; bypasses the confirmation gate."""
    runtime, repo = _bootstrap_repo()
    goal = GoalArc(
        title=title,
        description=description,
        execution_mode=execution_mode,
        domain=domain,
    )
    repo.create_goal_arc(goal)

    note_path = runtime.quarantine_path / f"{title.lower().replace(' ', '-')}-{utc_now().strftime('%Y-%m-%d')}" / "goal.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "---\n"
        f"id: {goal.id}\n"
        "type: goal\n"
        f"title: {goal.title}\n"
        f"domain: {goal.domain}\n"
        f"execution_mode: {goal.execution_mode}\n"
        "---\n\n"
        f"# {goal.title}\n\n{description or ''}\n"
    )
    return {
        "goal": _goal_to_dict(goal),
        "note_path": str(note_path),
    }


register_impl("goal_commit_draft", _do_goal_commit_draft)


@mcp.tool()
def goal_commit_draft(
    title: str,
    description: str = "",
    execution_mode: str = "mixed",
    domain: str = "",
) -> dict[str, Any]:
    """Commit a drafted goal. Tier-2: queues for user confirmation."""
    _require_writes()
    args = {"title": title, "description": description, "execution_mode": execution_mode, "domain": domain}
    if _bypassing():
        return _do_goal_commit_draft(**args)
    pending = queue_pending(
        tool_name="goal_commit_draft",
        args=args,
        summary=f"Commit goal: {title} ({execution_mode})",
        risk="high",
    )
    return queue_response(pending)


@mcp.tool()
def goal_list(status: str = "active") -> list[dict[str, Any]]:
    """List goal arcs as structured objects."""
    _, repo = _bootstrap_repo()
    goals = repo.list_goal_arcs(status=status or None)
    return [_goal_to_dict(goal) for goal in goals]


@mcp.tool()
def plan_day() -> dict[str, Any]:
    """Return today's planned learning blocks and gaps."""
    _, repo = _bootstrap_repo()
    today_blocks = repo.list_time_blocks_for_date(utc_now())
    return {
        "blocks": [
            {
                "task_id": block.task_id,
                "duration_minutes": block.duration_minutes,
                "start_time": block.start_time.isoformat() if getattr(block, "start_time", None) else None,
            }
            for block in today_blocks
        ],
        "goal_count": len(repo.list_goal_arcs()),
        "has_plan": bool(today_blocks),
    }


@mcp.tool()
def next_action(limit: int = 5) -> dict[str, Any]:
    """Return ranked next actions without forcing CLI stdout parsing."""
    _, repo = _bootstrap_repo()
    candidates = build_next_candidates(repo, limit=limit)
    return {
        "next_action": _safe_dict(candidates[0]) if candidates else None,
        "candidates": [_safe_dict(c) for c in candidates],
    }


def _parse_optional_due_date(due_date: Optional[str]) -> datetime | None:
    if not due_date:
        return None
    due_day = date.fromisoformat(due_date)
    return datetime.combine(due_day, time.min)


@mcp.tool()
def feedback_capture(
    surface: str,
    note: str = "",
    more_of: str = "",
    less_of: str = "",
    learner_context: str = "",
    keep_in_mind: str = "",
) -> dict[str, Any]:
    """Capture durable workflow guidance for one stable learning surface."""
    _require_writes()
    runtime, _ = _bootstrap_repo()
    normalized = normalize_feedback_scope(surface)
    note_path = save_feedback_profile(
        runtime.vault_path,
        normalized,
        more_of=more_of,
        less_of=less_of,
        learner_context=learner_context,
        keep_in_mind=keep_in_mind or note,
        focus_note=note,
    )
    return {
        "surface": normalized,
        "note_path": str(note_path),
        "guidance": load_feedback_guidance(runtime.vault_path, normalized),
    }


@mcp.tool()
def do_route(intent: str, limit: int = 5) -> dict[str, Any]:
    """Route a natural-language request into ranked ProductiveBrain commands."""
    _, repo = _bootstrap_repo()
    candidates = suggest_commands_for_intent(repo, intent, limit=limit)
    return {
        "intent": intent,
        "best_command": candidates[0].command if candidates else None,
        "candidates": [_safe_dict(candidate) for candidate in candidates],
    }


@mcp.tool()
def context_file_inspect(paths: list[str], model: str = "") -> dict[str, Any]:
    """Inspect one or more local context files without persisting them."""
    runtime, _ = _bootstrap_repo()
    provider_name, default_model = LLMRuntime(runtime.config).default_binding()
    selected = model.strip() if model.strip() else f"{provider_name}:{default_model}"
    if ":" in selected:
        provider_name, selected_model = selected.split(":", 1)
    else:
        selected_model = selected
    result = inspect_context_files(
        [Path(item).expanduser() for item in paths],
        provider=provider_name.strip().lower(),
        model=selected_model.strip(),
        dryrun=True,
    )
    plan = plan_context_file_response(result)
    return {
        "context_result": result.model_dump(mode="json"),
        "plan": {
            "action": plan.action,
            "can_answer": plan.can_answer,
            "parsed_files_only": plan.parsed_files_only,
            "user_message": plan.user_message,
        },
    }


@mcp.tool()
def context_file_ingest(paths: list[str], model: str = "") -> dict[str, Any]:
    """Persist one or more context files and return stored source refs."""
    _require_writes()
    runtime, repo = _bootstrap_repo()
    cmd_ctx = _context_cmd_ctx(runtime, repo)
    stored_sources: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    messages: list[str] = []
    for item in paths:
        stored, result = ingest_context_source(
            cmd_ctx,
            Path(item).expanduser(),
            model_override=model,
            dryrun=False,
        )
        stored_sources.append(_safe_dict(stored))
        results.append(result.model_dump(mode="json"))
        plan = plan_context_file_response(result)
        if plan.user_message:
            messages.append(plan.user_message)
    return {
        "sources": stored_sources,
        "results": results,
        "messages": messages,
    }


@mcp.tool()
def context_file_status(source_id: str = "") -> dict[str, Any]:
    """Return one stored source record or list all stored context sources."""
    _, repo = _bootstrap_repo()
    if source_id.strip():
        record = repo.find_context_source(source_id)
        return {"source": _safe_dict(record), "found": record is not None}
    return {
        "sources": [_safe_dict(item) for item in repo.list_context_sources()],
    }


@mcp.tool()
def source_bundle_list() -> dict[str, Any]:
    """List stored source bundles."""
    _, repo = _bootstrap_repo()
    return {
        "bundles": [bundle.model_dump(mode="json") for bundle in repo.list_source_bundles()],
    }


@mcp.tool()
def source_bundle_show(name: str) -> dict[str, Any]:
    """Show one stored source bundle by name."""
    _, repo = _bootstrap_repo()
    bundle = repo.get_source_bundle_by_name(name)
    return {
        "found": bundle is not None,
        "bundle": bundle.model_dump(mode="json") if bundle is not None else None,
    }


@mcp.tool()
def context_lock(ref: str) -> dict[str, Any]:
    """Lock the current context to one stored bundle or source."""
    _require_writes()
    _, repo = _bootstrap_repo()
    bundle = repo.get_source_bundle_by_name(ref)
    if bundle is not None:
        scope = active_context_from_bundle(bundle, locked=True)
    else:
        source = repo.find_context_source(ref)
        if source is None:
            return {"locked": False, "error": f"No bundle or source matched `{ref}`."}
        scope = active_context_from_sources(
            [str(source["source_ref"])],
            label=str(source.get("domain_name") or source.get("filename") or "context"),
            domain_id=str(source.get("domain_id", "") or "") or None,
            scope_mode=str(source.get("scope_mode", "unclear")),
            scope_boundary=str(source.get("scope_boundary", "")),
            locked=True,
        )
    repo.set_locked_context(scope)
    return {"locked": True, "scope": scope.model_dump(mode="json")}


@mcp.tool()
def context_unlock() -> dict[str, Any]:
    """Clear the persisted locked context."""
    _require_writes()
    _, repo = _bootstrap_repo()
    repo.clear_locked_context()
    return {"locked": False}


@mcp.tool()
def context_status() -> dict[str, Any]:
    """Return the persisted locked context, if any."""
    _, repo = _bootstrap_repo()
    scope = repo.get_locked_context()
    return {
        "locked": scope is not None,
        "scope": scope.model_dump(mode="json") if scope is not None else None,
    }


def _do_start_learning_session(branch: str, topic: str, *, title_prefix: str | None = None) -> dict[str, Any]:
    """Actual session-start logic; bypasses confirmation."""
    runtime, repo = _bootstrap_repo()
    active_session = repo.get_active_session()
    if active_session is not None:
        active_task = repo.get_task(active_session.task_id)
        return {
            "started": False,
            "error": "Another learning session is already active.",
            "active_session": {
                "session_id": active_session.id,
                "task_id": active_session.task_id,
                "title": stored_display_title(active_task) or getattr(active_session, "subject_scope", "") or "Current session",
                "branch": getattr(active_session, "branch", "study"),
            },
            "options": [
                "continue_current_session",
                "finish_current_session",
                "pause_current_session_and_switch",
                "treat_request_as_session_note",
                "cancel",
            ],
        }
    prefix = title_prefix or branch.capitalize()
    names = deterministic_names(
        f"{branch}_session",
        topic,
        {
            "domain": topic,
            "subject": topic,
            "activity_type": branch,
        },
    )
    task = Task(
        title=f"{prefix}: {topic}",
        description=f"Free {branch} session for {topic}",
        work_type=branch,
    )
    apply_generated_title(task, names, title_key="task_title")
    repo.create_task(task)
    session = _session_service(repo).start_session(
        task_id=task.id,
        mode="focus",
        timer_mode="stopwatch",
        branch=branch,
        subject_scope=topic,
    )
    apply_generated_names(session, names)
    repo.update_session(session)
    return {
        "started": True,
        "task_id": task.id,
        "task_title": stored_display_title(task) or task.title,
        "session_id": session.id,
        "branch": branch,
        "topic": topic,
        "vault": runtime.vault_name,
    }


register_impl("study_start", lambda topic: _do_start_learning_session("study", topic))
register_impl("practise_start", lambda skill: _do_start_learning_session("practise", skill))
register_impl("teach_start", lambda topic: _do_start_learning_session("study", topic, title_prefix="Teach"))


def _do_learn_start(topic: str, branch: str = "") -> dict[str, Any]:
    runtime, repo = _bootstrap_repo()
    resolved_branch = (branch or "").strip().lower()
    if resolved_branch not in {"study", "practise", "teach"}:
        resolved_branch = route_learning_intent(repo, topic).branch
    if resolved_branch == "teach":
        started = _do_start_learning_session("study", topic, title_prefix="Teach")
        started["branch"] = "teach" if started.get("started") else started.get("branch", "teach")
        return started
    return _do_start_learning_session(resolved_branch, topic)


register_impl("learn_start", _do_learn_start)


@mcp.tool()
def study_start(topic: str) -> dict[str, Any]:
    """Start a free study session. Tier-2: queues for confirmation."""
    _require_writes()
    if _bypassing():
        return _do_start_learning_session("study", topic)
    return queue_response(queue_pending(
        tool_name="study_start",
        args={"topic": topic},
        summary=f"Start study session: {topic}",
        risk="high",
    ))


@mcp.tool()
def practise_start(skill: str) -> dict[str, Any]:
    """Start a free deliberate-practice session. Tier-2: queues for confirmation."""
    _require_writes()
    if _bypassing():
        return _do_start_learning_session("practise", skill)
    return queue_response(queue_pending(
        tool_name="practise_start",
        args={"skill": skill},
        summary=f"Start practise session: {skill}",
        risk="high",
    ))


@mcp.tool()
def teach_start(topic: str) -> dict[str, Any]:
    """Start a tracked teaching session scaffold. Tier-2: queues for confirmation."""
    _require_writes()
    if _bypassing():
        return _do_start_learning_session("study", topic, title_prefix="Teach")
    return queue_response(queue_pending(
        tool_name="teach_start",
        args={"topic": topic},
        summary=f"Start teach session: {topic}",
        risk="high",
    ))


@mcp.tool()
def learn_start(topic: str, branch: str = "") -> dict[str, Any]:
    """Route and start a learning session without using the debug shell command surface."""
    _require_writes()
    args = {"topic": topic, "branch": branch}
    if _bypassing():
        return _do_learn_start(**args)
    summary = f"Start learning session: {topic}" if not branch else f"Start {branch} session: {topic}"
    return queue_response(queue_pending(
        tool_name="learn_start",
        args=args,
        summary=summary,
        risk="high",
    ))


@mcp.tool()
def learn_with_context(topic: str, context_paths: list[str], branch: str = "", model: str = "") -> dict[str, Any]:
    """Inspect, persist, and attach context files before starting a learning session."""
    _require_writes()
    runtime, repo = _bootstrap_repo()
    provider_name, default_model = LLMRuntime(runtime.config).default_binding()
    selected = model.strip() if model.strip() else f"{provider_name}:{default_model}"
    if ":" in selected:
        selected_provider, selected_model = selected.split(":", 1)
    else:
        selected_provider, selected_model = provider_name, selected

    aggregate_result = inspect_context_files(
        [Path(item).expanduser() for item in context_paths],
        provider=selected_provider.strip().lower(),
        model=selected_model.strip(),
        dryrun=False,
    )
    aggregate_plan = plan_context_file_response(aggregate_result)
    if not aggregate_plan.can_answer:
        return {
            "started": False,
            "context_result": aggregate_result.model_dump(mode="json"),
            "message": aggregate_plan.user_message,
        }

    cmd_ctx = _context_cmd_ctx(runtime, repo)
    stored_sources: list[dict[str, Any]] = []
    for item in context_paths:
        stored, _ = ingest_context_source(
            cmd_ctx,
            Path(item).expanduser(),
            model_override=selected,
            dryrun=False,
        )
        stored_sources.append(_safe_dict(stored))

    started = _do_learn_start(topic, branch=branch)
    if not started.get("started"):
        return {
            **started,
            "context_result": aggregate_result.model_dump(mode="json"),
            "sources": stored_sources,
        }

    primary = stored_sources[0] if stored_sources else {}
    scope = active_context_from_sources(
        [str(source.get("source_ref", "")) for source in stored_sources if str(source.get("source_ref", "")).strip()],
        label=str(primary.get("domain_name") or primary.get("filename") or "context"),
        domain_id=str(primary.get("domain_id", "") or "") or None,
        scope_mode=str(primary.get("scope_mode", "unclear")),
        scope_boundary=str(primary.get("scope_boundary", "")),
        locked=False,
    )
    session = repo.get_session(str(started.get("session_id", "")))
    if session is not None:
        attach_active_context(session, scope)
        repo.update_session(session)

    return {
        **started,
        "context_result": aggregate_result.model_dump(mode="json"),
        "context_scope": scope.model_dump(mode="json"),
        "sources": stored_sources,
    }


def _do_session_finish(note: str = "", completion_pct: int = 100) -> dict[str, Any]:
    runtime, repo = _bootstrap_repo()
    service = _session_service(repo)
    current = service.get_current_session()
    if current is None:
        return {"finished": False, "error": "No active session."}
    finished = service.finish_session(note=note or "done", completion_pct=completion_pct)
    task = repo.get_task(current.task_id)
    note_path = None
    log_error: Optional[str] = None
    if finished is not None and task is not None:
        try:
            from pb.core.domain_templates import _resolve_domain
            from pb.core.evidence_writer import EvidenceWriter, index_evidence_note
            domain = _resolve_domain(finished, task)
            writer = EvidenceWriter(vault_path=runtime.vault_path)
            note_path = writer.write_evidence(finished, task, assessment=None, domain=domain)
            if note_path:
                index_evidence_note(finished, task, None, note_path, domain)
        except Exception as exc:
            log_error = f"{type(exc).__name__}: {exc}"
    return {
        "finished": finished is not None,
        "session_id": finished.id if finished is not None else None,
        "task_id": current.task_id,
        "summary": getattr(finished, "actual_outcome", "") if finished is not None else "",
        "note_path": str(note_path) if note_path is not None else None,
        "evidence_created": [
            {
                "type": "evidence_note",
                "path": str(note_path),
            }
        ] if note_path is not None else [],
        "log_error": log_error,
    }


register_impl("session_finish", _do_session_finish)


# Phase 2: Evidence & Retry Queue MCP tools


@mcp.tool()
def evidence_list(domain: str = "", date: str = "") -> dict[str, Any]:
    """List evidence notes, optionally filtered by domain and/or date.

    Args:
        domain: Filter by domain (e.g., "math_problem_set", "german_speaking"). Empty = all.
        date: Filter by date (YYYY-MM-DD). Empty = all.

    Returns:
        {"items": [...], "count": N} where each item has domain, date, duration_min, outcome, path, sub_skills.
    """
    _bootstrap_repo()
    try:
        from pb.storage.database import get_connection
        conditions = []
        params: list = []
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if date:
            conditions.append("date = ?")
            params.append(date)
        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT id, domain, date, duration_min, outcome, sub_skills, path, created_at FROM evidence_notes WHERE {where} ORDER BY date DESC, created_at DESC LIMIT 50"
        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            items = [dict(r) for r in rows]
        return {"items": items, "count": len(items)}
    except Exception as exc:
        return {"items": [], "count": 0, "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def retry_queue_list(domain: str = "") -> dict[str, Any]:
    """List pending retry queue items, optionally filtered by domain.

    Args:
        domain: Filter by domain. Empty = all domains.

    Returns:
        {"items": [...], "count": N} where each item has id, domain, item_text, priority, source, cooldown_until.
    """
    _bootstrap_repo()
    try:
        from pb.core.retry_queue import RetryQueueWriter
        writer = RetryQueueWriter()
        items = writer.list_pending(domain=domain, limit=20)
        return {
            "items": [item.model_dump() for item in items],
            "count": len(items),
        }
    except Exception as exc:
        return {"items": [], "count": 0, "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def retry_queue_add(domain: str, item_text: str, priority: int = 3) -> dict[str, Any]:
    """Manually add an item to the retry queue.

    Args:
        domain: Learning domain (e.g., "math_problem_set").
        item_text: Description of what to retry.
        priority: 1=weakness, 2=incomplete, 3=optional (default 3).

    Returns:
        {"ok": True, "id": "..."} on success.
    """
    _require_writes()
    _bootstrap_repo()
    try:
        from pb.core.retry_queue import RetryQueueWriter
        writer = RetryQueueWriter()
        item_id = writer.enqueue(domain=domain, item_text=item_text, source="manual", priority=priority)
        if item_id:
            return {"ok": True, "id": item_id}
        return {"ok": False, "error": "Enqueue failed"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def retry_queue_resolve(item_id: str) -> dict[str, Any]:
    """Mark a retry queue item as resolved.

    Args:
        item_id: The retry item ID to resolve.

    Returns:
        {"ok": True} on success, {"ok": False, "error": "..."} on failure.
    """
    _require_writes()
    _bootstrap_repo()
    try:
        from pb.core.retry_queue import RetryQueueWriter
        writer = RetryQueueWriter()
        success = writer.resolve(item_id)
        return {"ok": success, "error": None if success else "Item not found"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def retry_queue_reschedule(item_id: str, date: str = "") -> dict[str, Any]:
    """Reschedule a retry queue item to resurface on a given date.

    Args:
        item_id: The retry item ID to reschedule.
        date: Target date (YYYY-MM-DD). Empty = tomorrow.

    Returns:
        {"ok": True} on success.
    """
    _require_writes()
    _bootstrap_repo()
    try:
        from pb.core.retry_queue import RetryQueueWriter
        writer = RetryQueueWriter()
        cooldown_date = date if date else None  # None defaults to tomorrow in RetryQueueWriter
        success = writer.reschedule(item_id, cooldown_date=cooldown_date)
        return {"ok": success, "error": None if success else "Item not found"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _do_session_pause(note: str = "") -> dict[str, Any]:
    _, repo = _bootstrap_repo()
    service = _session_service(repo)
    paused = service.pause_session(outcome=note or None)
    if paused is None:
        return {"paused": False, "error": "No active session."}
    task = repo.get_task(paused.task_id)
    return {
        "paused": True,
        "session": _session_payload(paused, task, service=service),
    }


register_impl("session_pause", _do_session_pause)


def _do_session_resume(task_id: str = "") -> dict[str, Any]:
    _, repo = _bootstrap_repo()
    task = _resolve_resume_task(repo, task_id)
    if task is None:
        return {"resumed": False, "error": "No resumable task found."}
    if task.state.value == "paused":
        task.state = type(task.state).ACTIVE
        task.paused_until = None
        task.pause_reason = None
        repo.update_task(task)

    from pb.cli.commands.execute import _task_session_defaults

    defaults = _task_session_defaults(task)
    service = _session_service(repo)
    session = service.start_session(
        task_id=task.id,
        mode="focus",
        duration_minutes=None,
        timer_mode="stopwatch",
        branch=str(defaults["branch"]),
        goal_id=defaults["goal_id"],
        track_id=defaults["track_id"],
        subject_scope=str(defaults["subject_scope"]),
        target_bloom_stage=defaults["target_bloom_stage"],
        practice_stage=defaults["practice_stage"],
        drill_type=defaults["drill_type"],
        constraint=defaults["constraint"],
        feedback_source=defaults["feedback_source"],
        evidence_target=defaults["evidence_target"],
        coach_cues=defaults["coach_cues"],
    )
    return {
        "resumed": True,
        "session": _session_payload(session, task, service=service),
    }


register_impl("session_resume", _do_session_resume)


@mcp.tool()
def session_status() -> dict[str, Any]:
    """Return the current tracked session plus resumable task hints."""
    _, repo = _bootstrap_repo()
    service = _session_service(repo)
    session = service.get_current_session()
    task = service.get_current_task()
    return {
        "active": session is not None,
        "session": _session_payload(session, task, service=service) if session is not None else None,
        "resumable_tasks": [task.model_dump(mode="json") for task in _resumeable_tasks(repo)[:5]],
    }


@mcp.tool()
def session_pause(note: str = "") -> dict[str, Any]:
    """Pause the active learning session. Tier-2: queues for confirmation."""
    _require_writes()
    args = {"note": note}
    if _bypassing():
        return _do_session_pause(**args)
    return queue_response(queue_pending(
        tool_name="session_pause",
        args=args,
        summary="Pause the active learning session",
        risk="high",
    ))


@mcp.tool()
def session_resume(task_id: str = "") -> dict[str, Any]:
    """Resume a prior learning task. Tier-2: queues for confirmation."""
    _require_writes()
    args = {"task_id": task_id}
    if _bypassing():
        return _do_session_resume(**args)
    summary = f"Resume task {task_id}" if task_id else "Resume the most recent resumable task"
    return queue_response(queue_pending(
        tool_name="session_resume",
        args=args,
        summary=summary,
        risk="high",
    ))


@mcp.tool()
def session_finish(note: str = "", completion_pct: int = 100) -> dict[str, Any]:
    """Finish the active learning session. Tier-2: queues for confirmation."""
    _require_writes()
    args = {"note": note, "completion_pct": completion_pct}
    if _bypassing():
        return _do_session_finish(**args)
    return queue_response(queue_pending(
        tool_name="session_finish",
        args=args,
        summary=f"Finish active session ({completion_pct}% complete)",
        risk="high",
    ))


@mcp.tool()
def review_day() -> dict[str, Any]:
    """Return daily learning metrics summary."""
    _, repo = _bootstrap_repo()
    return _collect_review_metrics(repo, days=1)


@mcp.tool()
def review_week() -> dict[str, Any]:
    """Return weekly learning metrics summary."""
    _, repo = _bootstrap_repo()
    return _collect_review_metrics(repo, days=7)


@mcp.tool()
def notes_inbox() -> dict[str, Any]:
    """List quarantined notes waiting for review or organization."""
    runtime = get_runtime_context()
    quarantine_root = get_quarantine_path(runtime.config)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    notes: list[dict[str, Any]] = []
    for note_path in sorted(quarantine_root.rglob("*.md")):
        if not note_path.is_file():
            continue
        frontmatter = _parse_frontmatter(note_path)
        notes.append(
            {
                "path": str(note_path.relative_to(runtime.vault_path)),
                "title": frontmatter.get("title", note_path.stem),
                "type": frontmatter.get("type", ""),
            }
        )
    return {
        "quarantine_path": str(quarantine_root),
        "notes": notes,
    }


@mcp.tool()
def context_build(domain: str = "", days: int = 30) -> dict[str, Any]:
    """Build a lightweight domain-scoped context packet from current vault state."""
    runtime, repo = _bootstrap_repo()
    scope = (domain or "").strip()
    horizon_days = max(1, days)

    goals = [
        _goal_to_dict(goal)
        for goal in repo.list_goal_arcs(status=None)
        if _matches_domain(goal.domain, goal.title, goal.description, domain=scope)
    ]
    todos = [
        task.model_dump(mode="json")
        for task in repo.list_tasks()
        if task.completion < 100
        and getattr(task, "work_type", "") == "todo"
        and _matches_domain(task.title, task.description, domain=scope)
    ]
    sessions = _recent_session_rows(repo, domain=scope, limit=5)
    recurring_errors = [row["observed_errors"] for row in sessions if row.get("observed_errors")]
    useful_notes = _recent_useful_notes(runtime.vault_path, domain=scope, limit=8)
    graph_neighbors = _graph_neighbors_for_notes(runtime.vault_path, useful_notes[:3])
    related_concepts = _related_concepts_from_neighbors(graph_neighbors)
    orphan_notes = _orphan_notes(runtime.vault_path, domain=scope, limit=8)
    stale_notes = _stale_notes(runtime.vault_path, domain=scope, days=horizon_days, limit=8)
    weak_areas = _weak_areas(repo, recurring_errors)
    active_context_scope = _active_context_scope(repo)
    source_bundles = [
        bundle.model_dump(mode="json")
        for bundle in repo.list_source_bundles()[:8]
        if not scope or scope in (bundle.domain_name or "").lower() or scope in bundle.name.lower()
    ]

    pending_review: list[dict[str, Any]] = []
    pending_anki: dict[str, Any] = {"suggested": 0, "export_ready": 0}
    try:
        from pb.vault.anki_client import get_cards_by_status, get_pending_card_count

        suggested_count = len(get_cards_by_status("suggested"))
        export_ready_count = get_pending_card_count()
        pending_review = [
            {"kind": "anki_suggested", "count": suggested_count},
            {"kind": "anki_export_ready", "count": export_ready_count},
        ]
        pending_anki = {"suggested": suggested_count, "export_ready": export_ready_count}
    except Exception:
        pending_review = []

    guidance_paths = []
    for scope_name in ("general", "learn", "study", "practise", "teach", normalize_feedback_scope(scope or "general")):
        path = feedback_profile_path(runtime.vault_path, scope_name)
        if path.exists():
            guidance_paths.append(str(path.relative_to(runtime.vault_path)))

    return {
        "domain": scope or "global",
        "retrieved_at": utc_now().isoformat(),
        "retrieval_policy": {
            "days": horizon_days,
            "thought_source": "none",
            "domain_filter": scope or "global",
        },
        "active_goals": goals,
        "recent_sessions": sessions,
        "high_salience_memories": [],
        "current_weak_areas": weak_areas,
        "weak_areas": weak_areas,
        "recurring_mistakes": recurring_errors[:5],
        "pending_review_items": pending_review,
        "active_todos": todos[:5],
        "useful_notes": useful_notes,
        "related_concepts": related_concepts,
        "graph_neighbors": graph_neighbors,
        "orphan_notes": orphan_notes,
        "stale_notes": stale_notes,
        "source_bundles": source_bundles,
        "pending_anki": pending_anki,
        "active_context_scope": active_context_scope,
        "domain_preferences": guidance_paths,
        "evidence_templates": ["session_log", "recall_note"],
        "private_agent_profile_reference": None,
        "omitted_context_summary": {
            "additional_thoughts": 0,
            "additional_sessions": max(0, len(sessions) - 5),
            "thread_count": 0,
        },
    }


def _recent_useful_notes(vault_path: Path, *, domain: str, limit: int) -> list[dict[str, Any]]:
    knowledge_root = vault_path / "knowledge"
    search_root = knowledge_root / domain if domain and (knowledge_root / domain).exists() else knowledge_root
    if not search_root.exists():
        return []
    rows: list[tuple[float, dict[str, Any]]] = []
    for md_path in search_root.rglob("*.md"):
        if md_path.name.startswith("_"):
            continue
        try:
            stat = md_path.stat()
        except OSError:
            continue
        rows.append(
            (
                stat.st_mtime,
                {
                    "path": str(md_path.relative_to(vault_path)),
                    "title": md_path.stem,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                },
            )
        )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in rows[:limit]]


def _graph_neighbors_for_notes(vault_path: Path, notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from pb.mcp.tools.vault import vault_link_graph

    neighbors: list[dict[str, Any]] = []
    for note in notes:
        try:
            graph = vault_link_graph(note["path"], depth=2)
        except Exception:
            continue
        neighbors.append(
            {
                "path": note["path"],
                "outgoing_count": graph["outgoing_count"],
                "incoming_count": graph["incoming_count"],
                "out1": [item.get("resolved_path") or item.get("target") for item in graph["out1"][:5]],
                "in1": graph["in1"][:5],
                "out2": graph["out2"][:5],
                "in2": graph["in2"][:5],
            }
        )
    return neighbors


def _related_concepts_from_neighbors(neighbors: list[dict[str, Any]]) -> list[str]:
    concepts: list[str] = []
    for item in neighbors:
        for path in [*item.get("out1", []), *item.get("in1", []), *item.get("out2", []), *item.get("in2", [])]:
            if not path:
                continue
            stem = Path(str(path)).stem.replace("-", " ")
            if stem not in concepts:
                concepts.append(stem)
    return concepts[:12]


def _orphan_notes(vault_path: Path, *, domain: str, limit: int) -> list[dict[str, Any]]:
    from pb.core.brain import BrainEngine

    rows = BrainEngine().detect_orphans()
    if domain:
        rows = [row for row in rows if domain in str(row.get("path", "")).lower()]
    return rows[:limit]


def _stale_notes(vault_path: Path, *, domain: str, days: int, limit: int) -> list[dict[str, Any]]:
    cutoff = datetime.now().timestamp() - (days * 86400)
    stale: list[tuple[float, dict[str, Any]]] = []
    for note in _recent_useful_notes(vault_path, domain=domain, limit=500):
        try:
            stat = (vault_path / note["path"]).stat()
        except OSError:
            continue
        if stat.st_mtime > cutoff:
            continue
        stale.append((stat.st_mtime, note))
    stale.sort(key=lambda item: item[0])
    return [payload for _, payload in stale[:limit]]


def _weak_areas(repo: Repository, recurring_errors: list[str]) -> list[dict[str, Any]]:
    areas: list[dict[str, Any]] = []
    try:
        for record in repo.list_concept_confidence()[:8]:
            areas.append(
                {
                    "kind": "concept_confidence",
                    "concept": record.concept_id,
                    "confidence_score": record.confidence_score,
                    "next_review_at": record.next_review_at,
                }
            )
    except Exception:
        pass
    for item in recurring_errors[:5]:
        clean = str(item).strip()
        if not clean:
            continue
        areas.append({"kind": "recurring_error", "concept": clean})
    return areas[:10]


def _active_context_scope(repo: Repository) -> dict[str, Any] | None:
    session = repo.get_active_session()
    if session is not None:
        generated = dict(getattr(session, "generated_names", {}) or {})
        active_context = generated.get("active_context_scope")
        if isinstance(active_context, dict):
            return active_context
    locked = repo.get_locked_context()
    return locked.model_dump(mode="json") if locked is not None else None


def _do_anki_generate(
    domain: Optional[str] = None,
    deck: Optional[str] = None,
    term: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    _, repo = _bootstrap_repo()
    runtime = get_runtime_context()
    from pb.vault.anki_service import AnkiService

    service = AnkiService(vault_path=runtime.vault_path, repo=repo)
    resolved_deck, effective_domain = _resolve_deck_and_domain(deck, domain)
    return service.generate_cards(
        note_slug=term or effective_domain or "vault",
        note_content=term or "",
        domain=effective_domain,
        deck=resolved_deck,
        term=term,
        source="term" if term else "auto",
        note_types=None,
        model=model,
        emulate_existing_deck=False,
    )


register_impl("anki_generate_candidates", _do_anki_generate)


@mcp.tool()
def anki_generate_candidates(
    domain: Optional[str] = None,
    deck: Optional[str] = None,
    term: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Generate Anki candidates. Tier-2: queues for confirmation."""
    _require_writes()
    args = {"domain": domain, "deck": deck, "term": term, "model": model}
    if _bypassing():
        return _do_anki_generate(**args)
    return queue_response(queue_pending(
        tool_name="anki_generate_candidates",
        args=args,
        summary=f"Generate Anki cards (deck={deck or 'auto'}, term={term or 'auto'})",
        risk="high",
    ))


@mcp.tool()
def anki_candidate_list(
    status: str = "suggested",
    domain: Optional[str] = None,
    deck: Optional[str] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List stored Anki candidate cards with lightweight filtering."""
    from pb.vault.anki_client import get_cards_by_status

    cards = get_cards_by_status(status, domain)
    if deck:
        cards = [card for card in cards if str(card.get("deck", "") or "") == deck]
    return {
        "status": status,
        "domain": domain,
        "deck": deck,
        "count": len(cards),
        "items": cards[: max(1, limit)],
    }


@mcp.tool()
def anki_candidate_status_counts(domain: Optional[str] = None) -> dict[str, Any]:
    """Return grouped Anki candidate counts plus export-ready totals."""
    from pb.vault.anki_client import get_card_status_counts, get_pending_card_count

    return {
        "domain": domain,
        "counts": get_card_status_counts(domain),
        "export_ready": get_pending_card_count(domain),
    }


def _do_anki_candidate_update(card_ids: list[str], status: str) -> dict[str, Any]:
    from pb.vault.anki_client import get_card_by_id, update_cards_status

    clean_ids = [str(card_id).strip() for card_id in card_ids if str(card_id).strip()]
    existing = [card_id for card_id in clean_ids if get_card_by_id(card_id) is not None]
    missing = [card_id for card_id in clean_ids if card_id not in existing]
    update_cards_status(existing, status)
    return {
        "updated": existing,
        "missing": missing,
        "status": status,
    }


register_impl("anki_candidate_update", _do_anki_candidate_update)


@mcp.tool()
def anki_candidate_update(card_ids: list[str], status: str) -> dict[str, Any]:
    """Update candidate review state for one or more stored cards."""
    _require_writes()
    args = {"card_ids": list(card_ids), "status": status}
    if _bypassing():
        return _do_anki_candidate_update(**args)
    return queue_response(queue_pending(
        tool_name="anki_candidate_update",
        args=args,
        summary=f"Update {len(card_ids)} Anki candidate(s) to status={status}",
        risk="medium",
    ))


@mcp.tool()
def anki_export_status(domain: Optional[str] = None, deck: Optional[str] = None) -> dict[str, Any]:
    """Report whether candidate cards are ready for export and how they would degrade."""
    from pb.vault.anki_client import get_cards_by_status, is_anki_available

    cards = get_cards_by_status("exportable", domain)
    if deck:
        cards = [card for card in cards if str(card.get("deck", "") or "") == deck]
    return {
        "domain": domain,
        "deck": deck,
        "count": len(cards),
        "anki_available": is_anki_available(),
        "fallback": "csv",
        "cards": cards[:20],
    }


def _do_anki_export(
    domain: Optional[str] = None,
    deck: Optional[str] = None,
    csv_only: bool = False,
) -> dict[str, Any]:
    runtime = get_runtime_context()
    from pb.vault.anki_client import (
        export_cards_to_anki,
        export_cards_to_apkg,
        export_cards_to_csv,
        get_cards_by_status,
        is_anki_available,
    )

    cards = get_cards_by_status("exportable", domain)
    if deck:
        cards = [card for card in cards if str(card.get("deck", "") or "") == deck]
    if not cards:
        return {"ok": False, "message": "No accepted or edited cards are ready for export."}
    if csv_only:
        csv_path = export_cards_to_csv(cards, runtime.vault_path)
        return {"ok": True, "backend": "csv", "path": str(csv_path), "count": len(cards)}

    packaged, package_path, package_msg = export_cards_to_apkg(cards, runtime.vault_path)
    if packaged and package_path is not None:
        result: dict[str, Any] = {
            "ok": True,
            "backend": "apkg",
            "path": str(package_path),
            "message": package_msg,
            "count": len(cards),
        }
        if is_anki_available():
            synced, sync_msg = export_cards_to_anki(cards)
            result["anki_sync"] = {"ok": synced, "message": sync_msg}
        return result

    if is_anki_available():
        synced, sync_msg = export_cards_to_anki(cards)
        if synced:
            return {"ok": True, "backend": "ankiconnect", "message": sync_msg, "count": len(cards)}

    csv_path = export_cards_to_csv(cards, runtime.vault_path)
    return {
        "ok": True,
        "backend": "csv",
        "path": str(csv_path),
        "message": package_msg,
        "count": len(cards),
    }


register_impl("anki_export", _do_anki_export)


@mcp.tool()
def anki_export(
    domain: Optional[str] = None,
    deck: Optional[str] = None,
    csv_only: bool = False,
) -> dict[str, Any]:
    """Export accepted or edited cards, with CSV fallback when Anki is unavailable."""
    _require_writes()
    args = {"domain": domain, "deck": deck, "csv_only": csv_only}
    if _bypassing():
        return _do_anki_export(**args)
    return queue_response(queue_pending(
        tool_name="anki_export",
        args=args,
        summary=f"Export Anki cards{f' for deck {deck}' if deck else ''}",
        risk="high",
    ))


@mcp.tool()
def notes_organise_preview(merge: bool = False) -> dict[str, Any]:
    """Preview conservative quarantine note organization. Read-only."""
    runtime = get_runtime_context()
    moves = _collect_moves(runtime.vault_path, get_quarantine_path(runtime.config), merge=merge)
    return {
        "merge": merge,
        "moves": [
            {
                "source": str(move.source.relative_to(runtime.vault_path)),
                "target": str(move.target.relative_to(runtime.vault_path)),
                "action": move.action,
            }
            for move in moves
        ],
    }


def _do_notes_organise_commit(merge: bool = False, flatten: bool = False) -> dict[str, Any]:
    runtime = get_runtime_context()
    quarantine_root = get_quarantine_path(runtime.config)
    moves = _collect_moves(runtime.vault_path, quarantine_root, merge=merge)
    applied = apply_moves(moves, quarantine_root=quarantine_root, flatten=flatten)
    return {"merge": merge, "flatten": flatten, "applied": applied}


register_impl("notes_organise_commit", _do_notes_organise_commit)


@mcp.tool()
def notes_organise_commit(merge: bool = False, flatten: bool = False) -> dict[str, Any]:
    """Apply quarantine note organization moves. Tier-2: queues for confirmation."""
    _require_writes()
    args = {"merge": merge, "flatten": flatten}
    if _bypassing():
        return _do_notes_organise_commit(**args)
    return queue_response(queue_pending(
        tool_name="notes_organise_commit",
        args=args,
        summary=f"Organise quarantine notes (merge={merge}, flatten={flatten})",
        risk="high",
    ))


def _do_vault_create(name: str, goal: str = "", path: str = "") -> dict[str, Any]:
    """Create a new vault at `path` (or default location), scaffold it, seed PLAN.md from `goal`."""
    from pb.vault.scaffold import scaffold_vault

    if path:
        vault_root = Path(path).expanduser().resolve()
    else:
        vault_root = Path.home() / "Documents" / "vaults" / name
    vault_root.mkdir(parents=True, exist_ok=True)

    created = scaffold_vault(vault_root)

    plan_path = vault_root / "PLAN.md"
    plan_created = False
    if not plan_path.exists():
        goal_text = (goal or "").strip() or "(no goal provided — clarify with the user before proceeding)"
        plan_path.write_text(
            "# PLAN.md\n\n"
            f"_Vault: **{name}** — created {utc_now().strftime('%Y-%m-%d')}_\n\n"
            "## Goal\n\n"
            f"{goal_text}\n\n"
            "## Note Structure\n\n"
            "- Categories follow the numbered prefix scheme (`direction`, `knowledge`, ...).\n"
            "- All notes live at depth ≤ 2.\n"
            "- Each note carries YAML frontmatter (`type:`, `updated:`, and stage tags).\n\n"
            "## Open Clarifications\n\n"
            "- [ ] Confirm primary domains the user wants to track.\n"
            "- [ ] Confirm cadence (per-day or per-week study target).\n"
            "- [ ] Confirm whether evidence is via artefact, recall, or both.\n\n"
            "_Append decisions and refinements below; do not rewrite history._\n"
        )
        plan_created = True
        created.append("PLAN.md")

    return {
        "vault_name": name,
        "vault_path": str(vault_root),
        "created": created,
        "plan_seeded": plan_created,
        "goal": goal,
    }


register_impl("vault_create", _do_vault_create)


@mcp.tool()
def vault_create(name: str, goal: str = "", path: str = "") -> dict[str, Any]:
    """Create a new vault with AGENTS.md, ISSUES.md, and PLAN.md (from goal).
    Tier-2: queues for confirmation."""
    _require_writes()
    args = {"name": name, "goal": goal, "path": path}
    if _bypassing():
        return _do_vault_create(**args)
    return queue_response(queue_pending(
        tool_name="vault_create",
        args=args,
        summary=f"Create vault '{name}' at {path or 'default location'}",
        risk="high",
    ))


@mcp.tool()
def mcp_pending_list() -> dict[str, Any]:
    """List pending tier-2 MCP actions awaiting user confirmation. Read-only."""
    from pb.mcp.pending import list_pending as _list

    return {
        "pending": [
            {
                "id": a.id,
                "tool": a.tool_name,
                "summary": a.summary,
                "risk": a.risk,
                "created_at": a.created_at,
                "args": a.args,
            }
            for a in _list()
        ]
    }
