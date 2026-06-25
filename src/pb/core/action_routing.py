# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Intent routing and next-action recommendation helpers."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from pb.core.agent_weights import sort_commitments_for_next
from pb.core.entity_refs import display_ref
from pb.core.model_policy import resolve_model_binding
from pb.core.models import utc_now
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.scope_resolution import match_goal, match_track
from pb.llm.runtime import LLMRuntime
from pb.storage.config import get_config


@dataclass
class CommandCandidate:
    """A ranked command recommendation."""

    command: str
    reason: str
    score: float = 0.0
    source: str = "heuristic"
    kind: str = "command"
    requires_input: bool = False
    suggested_text: str = ""
    human_label: str = ""
    short_reason: str = ""
    semantic_action: str = ""
    backing_command: str = ""
    confidence: float = 0.0
    agent_id: str = ""

    def __post_init__(self) -> None:
        self.command = self.command.strip()
        self.backing_command = (self.backing_command or self.command).strip()
        self.short_reason = (self.short_reason or self.reason).strip()
        self.reason = self.short_reason
        if self.confidence <= 0.0 and self.score > 0.0:
            self.confidence = self.score
        if self.score <= 0.0 and self.confidence > 0.0:
            self.score = self.confidence
        self.semantic_action = (self.semantic_action or _semantic_action_for_command(self.backing_command)).strip()
        self.human_label = (self.human_label or _human_label_for_command(self.backing_command)).strip()


@dataclass
class LearningRouteDecision:
    """Resolved branch for a learning intent."""

    branch: str
    reason: str
    confidence: float = 0.0
    source: str = "local"


_PRACTISE_KEYWORDS = {
    "archery",
    "crochet",
    "crocheting",
    "drill",
    "parkour",
    "piano",
    "practice",
    "practise",
    "reps",
    "session",
    "skate",
    "skateboarding",
    "swim",
    "swimming",
    "tennis",
}
_STUDY_KEYWORDS = {
    "anki",
    "concept",
    "grammar",
    "internalise",
    "internalize",
    "learn",
    "memory",
    "recall",
    "review cards",
    "study",
    "theory",
    "understand",
    "vocab",
}
_TEACH_KEYWORDS = {
    "teach me",
    "walk me through",
    "explain interactively",
    "socratic",
    "quiz me as you teach",
}


def _generate_policy_json(prompt: dict[str, object], *, operation: str, max_output_tokens: int) -> str:
    """Generate raw JSON text through the configured provider-neutral runtime."""
    runtime = LLMRuntime(get_config())
    binding = resolve_model_binding(runtime.config, operation)
    provider_name, model_name = runtime._resolve_provider_and_model(binding)
    client = runtime._client_for_provider(provider_name)
    if not client.is_available():
        return ""
    try:
        return client.generate_with_model(
            json.dumps(prompt),
            model=model_name,
            timeout=20,
            max_output_tokens=max_output_tokens,
        )
    except Exception:
        return ""


def _quote(text: str) -> str:
    return shlex.quote(text.strip()) if text.strip() else ""


def _display_scope(text: str) -> str:
    cleaned = (text or "").strip().strip("'\"")
    return cleaned or "this learning block"


def _semantic_action_for_command(command: str) -> str:
    normalized = command.strip().lower()
    if normalized == "finish":
        return "finish_session"
    if normalized == "pause":
        return "pause_session"
    if normalized.startswith("next --reminder "):
        return "open_reminder"
    if normalized.startswith("anki export"):
        return "export_recall"
    if normalized.startswith("anki list"):
        return "review_recall"
    if normalized.startswith("plan day"):
        return "plan_day"
    if normalized.startswith("plan week"):
        return "plan_week"
    if normalized == "goal":
        return "clarify_goal"
    if normalized.startswith("study recall"):
        return "generate_recall"
    if normalized.startswith("teach "):
        return "start_teach"
    if normalized.startswith("study "):
        return "start_study"
    if normalized.startswith("practise "):
        return "start_practise"
    if normalized.startswith("resume"):
        return "resume_task"
    if normalized.startswith("start "):
        return "start_task"
    return "command"


def _human_label_for_command(command: str) -> str:
    normalized = command.strip()
    lowered = normalized.lower()
    if lowered == "finish":
        return "Finish the current session"
    if lowered == "pause":
        return "Pause the current session"
    if lowered.startswith("next --reminder "):
        return "Handle the pending reminder"
    if lowered == "goal":
        return "Clarify your learning goal"
    if lowered.startswith("plan day"):
        return "Shape today's learning plan"
    if lowered.startswith("plan week"):
        return "Shape this week's learning plan"
    if lowered.startswith("anki export"):
        return "Export ready recall cards"
    if lowered.startswith("anki list"):
        return "Review suggested recall cards"
    if lowered.startswith("study recall "):
        return f"Turn {_display_scope(normalized[13:])} into recall prompts"
    if lowered == "study recall":
        return "Generate recall prompts from recent study"
    if lowered.startswith("teach "):
        return f"Start a guided teaching session on {_display_scope(normalized[6:])}"
    if lowered.startswith("study "):
        return f"Study {_display_scope(normalized[6:])}"
    if lowered.startswith("practise log "):
        return f"Log deliberate practice for {_display_scope(normalized[13:])}"
    if lowered.startswith("practise "):
        return f"Practise {_display_scope(normalized[9:])}"
    if lowered.startswith("thought "):
        return f"Capture this thought: {_display_scope(normalized[8:])}"
    if lowered == "thought":
        return "Capture a quick thought"
    if lowered.startswith("todo "):
        return f"Capture this upcoming task: {_display_scope(normalized[5:])}"
    if lowered == "todo":
        return "Capture an upcoming task"
    if lowered == "resume":
        return "Resume the most relevant paused task"
    if lowered.startswith("start "):
        return "Start the selected task"
    return normalized or "Run the suggested action"


def _add_candidate(
    candidates: list[CommandCandidate],
    command: str,
    reason: str,
    score: float,
    source: str = "heuristic",
    kind: str = "command",
    requires_input: bool = False,
    suggested_text: str = "",
    *,
    human_label: str = "",
    short_reason: str = "",
    semantic_action: str = "",
    agent_id: str = "",
) -> None:
    normalized = command.strip()
    if not normalized:
        return
    if any(existing.backing_command == normalized for existing in candidates):
        return
    candidates.append(
        CommandCandidate(
            command=normalized,
            reason=reason,
            score=score,
            source=source,
            kind=kind,
            requires_input=requires_input,
            suggested_text=suggested_text,
            human_label=human_label,
            short_reason=short_reason or reason,
            semantic_action=semantic_action,
            backing_command=normalized,
            confidence=score,
            agent_id=agent_id.strip(),
        )
    )


def _todo_start_reason(task) -> str:
    if getattr(task, "due_date", None):
        due_label = task.due_date.strftime("%Y-%m-%d")
        return f"Start todo `{task.title}` before its due date ({due_label})."
    return f"Start todo `{task.title}` while it is still top of mind."


def _best_todo_tasks(repo) -> list:
    todo_tasks = []
    for task in repo.list_tasks():
        if task.completion >= 100 or task.archived_at is not None:
            continue
        if (getattr(task, "work_type", "") or "").lower() != "todo":
            continue
        todo_tasks.append(task)
    return sorted(
        todo_tasks,
        key=lambda item: (
            item.due_date is None,
            item.due_date or datetime.max,
            item.created_at,
        ),
    )


def _task_branch(task, block_kind: str | None = None) -> str:
    meta = parse_learning_task_metadata(task)
    if meta.branch in {"study", "practise"}:
        return meta.branch
    work_type = (getattr(task, "work_type", "") or "").lower()
    block = (block_kind or "").lower()
    if work_type in {"practice", "practise"} or block in {"practice", "practise"}:
        return "practise"
    title = (getattr(task, "title", "") or "").lower()
    if title.startswith("practise:") or title.startswith("practice:"):
        return "practise"
    if title.startswith("teach:"):
        return "study"
    return "study"


def _scope_for_task(task) -> str:
    meta = parse_learning_task_metadata(task)
    return meta.scope or meta.domain or getattr(task, "title", "")


def _learner_command_for_task(task, *, block_kind: str | None = None) -> str:
    branch = _task_branch(task, block_kind)
    scope = _scope_for_task(task).strip()
    meta = parse_learning_task_metadata(task)
    quoted_scope = _quote(scope)
    if scope:
        if branch == "study" and meta.study_mode == "socratic_teach":
            return f"teach {quoted_scope}"
        if branch == "practise":
            return f"practise {quoted_scope}"
        return f"study {quoted_scope}"
    return f"start {display_ref(task, 'task')}"


def _normalized(text: str) -> str:
    return " ".join((text or "").lower().split())


def _token_set(text: str) -> set[str]:
    cleaned = _normalized(text)
    return {token for token in cleaned.replace("/", " ").replace("-", " ").split() if token}


def _topic_without_teach_phrases(intent: str) -> str:
    cleaned = (intent or "").strip()
    lowered = cleaned.lower()
    for phrase in sorted(_TEACH_KEYWORDS, key=len, reverse=True):
        if phrase in lowered:
            start = lowered.find(phrase)
            cleaned = (cleaned[:start] + cleaned[start + len(phrase) :]).strip(" :,-")
            lowered = cleaned.lower()
    return cleaned or (intent or "").strip()


def _topic_without_practise_phrases(intent: str) -> str:
    """Strip request framing so practise suggestions name the skill, not the sentence."""
    cleaned = (intent or "").strip()
    patterns = [
        r"^\s*i\s+(?:want|need|would like)\s+to\s+practi[cs]e\s+",
        r"^\s*please\s+practi[cs]e\s+",
        r"^\s*practi[cs]e\s+",
        r"^\s*drill\s+",
        r"^\s*do\s+reps\s+(?:on|for)\s+",
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" :,-")
        if updated != cleaned:
            cleaned = updated
            break
    return cleaned or (intent or "").strip()


def _text_match_score(intent: str, *haystacks: str) -> float:
    """Return a soft score for how well the intent matches the haystacks."""
    lowered = _normalized(intent)
    if not lowered:
        return 0.0
    intent_tokens = _token_set(lowered)
    best = 0.0
    for haystack in haystacks:
        lowered_haystack = _normalized(haystack)
        if not lowered_haystack:
            continue
        if lowered == lowered_haystack:
            best = max(best, 1.0)
            continue
        if lowered in lowered_haystack or lowered_haystack in lowered:
            best = max(best, 0.82)
        hay_tokens = _token_set(lowered_haystack)
        overlap = len(intent_tokens & hay_tokens)
        if overlap:
            score = overlap / max(1, len(intent_tokens))
            best = max(best, min(0.78, 0.18 + score))
    return best


def _goal_supports_branch(goal, branch: str) -> bool:
    mode = (getattr(goal, "execution_mode", "") or "mixed").lower()
    if branch == "study":
        return mode in {"mixed", "study"}
    return mode in {"mixed", "practise", "practice"}


def _list_knowledge_domains() -> list[str]:
    try:
        from pb.vault.config import get_vault_path

        knowledge_dir = get_vault_path() / "knowledge"
        if not knowledge_dir.exists():
            return []
        return sorted(
            domain_dir.name
            for domain_dir in knowledge_dir.iterdir()
            if domain_dir.is_dir() and not domain_dir.name.startswith(".") and (domain_dir / "_state.md").exists()
        )
    except Exception:
        return []


def _recent_learning_context(repo, *, limit: int = 6) -> list[tuple[datetime, str, str]]:
    rows: list[tuple[datetime, str, str]] = []
    for task in repo.list_tasks():
        for session in repo.list_sessions_for_task(task.id):
            if session.subject_scope:
                rows.append((session.start_at, session.branch or _task_branch(task), session.subject_scope))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows[:limit]


def route_learning_intent(repo, intent: str) -> LearningRouteDecision:
    """Choose study or practise for a free-text learning request."""
    lowered = _normalized(intent)
    if not lowered:
        return LearningRouteDecision(
            branch="study",
            reason="Defaulting to study because no explicit learning target was provided.",
            confidence=0.0,
            source="default",
        )

    scores = {"study": 0.12, "practise": 0.12}
    reasons: dict[str, list[str]] = {"study": [], "practise": []}

    practise_hits = sorted(keyword for keyword in _PRACTISE_KEYWORDS if keyword in lowered)
    study_hits = sorted(keyword for keyword in _STUDY_KEYWORDS if keyword in lowered)
    if practise_hits:
        scores["practise"] += 0.42 + (0.04 * min(3, len(practise_hits)))
        reasons["practise"].append("you asked for drills, reps, or performance improvement.")
    if study_hits:
        scores["study"] += 0.42 + (0.04 * min(3, len(study_hits)))
        reasons["study"].append("you asked to understand, remember, or internalise the idea.")

    best_goal_by_branch: dict[str, tuple[float, str]] = {"study": (0.0, ""), "practise": (0.0, "")}
    for goal in repo.list_goal_arcs(status=None):
        match = _text_match_score(intent, goal.title, getattr(goal, "domain", ""), getattr(goal, "description", ""))
        if match <= 0:
            continue
        if _goal_supports_branch(goal, "study"):
            scores["study"] += 0.38 * match
            if match > best_goal_by_branch["study"][0]:
                best_goal_by_branch["study"] = (match, goal.title)
        if _goal_supports_branch(goal, "practise"):
            scores["practise"] += 0.38 * match
            if match > best_goal_by_branch["practise"][0]:
                best_goal_by_branch["practise"] = (match, goal.title)

    for branch, (match, title) in best_goal_by_branch.items():
        if title:
            reasons[branch].append(f"matches goal '{title}'")

    for track in repo.list_tracks(active_only=True):
        match = _text_match_score(intent, track.name, getattr(track, "description", ""))
        if match <= 0:
            continue
        linked_modes = {"study": 0.0, "practise": 0.0}
        for goal_id in getattr(track, "linked_goal_arc_ids", []):
            goal = repo.get_goal_arc(goal_id)
            if goal is None:
                continue
            if _goal_supports_branch(goal, "study"):
                linked_modes["study"] += 1.0
            if _goal_supports_branch(goal, "practise"):
                linked_modes["practise"] += 1.0
        if linked_modes["study"] == 0 and linked_modes["practise"] == 0:
            scores["study"] += 0.16 * match
            scores["practise"] += 0.16 * match
        else:
            if linked_modes["study"] > 0:
                scores["study"] += 0.24 * match
            if linked_modes["practise"] > 0:
                scores["practise"] += 0.24 * match
        if match >= 0.5:
                reasons["study"].append(f"your active track '{track.name}' points here.")
                reasons["practise"].append(f"your active track '{track.name}' points here.")

    for block in repo.list_time_blocks_for_date(utc_now()):
        task = repo.get_task(block.task_id)
        if task is None:
            continue
        meta = parse_learning_task_metadata(task)
        match = _text_match_score(intent, task.title, task.description, meta.scope, meta.domain)
        if match <= 0:
            continue
        branch = _task_branch(task, getattr(block, "block_kind", "study"))
        scores[branch] += 0.26 * match
        reasons[branch].append(f"today's plan already includes '{task.title}'.")

    for _, branch, scope in _recent_learning_context(repo):
        match = _text_match_score(intent, scope)
        if match <= 0:
            continue
        scores[branch] += 0.18 * match
        reasons[branch].append(f"you recently worked on '{scope}'.")

    for domain in _list_knowledge_domains():
        match = _text_match_score(intent, domain)
        if match <= 0:
            continue
        scores["study"] += 0.22 * match
        reasons["study"].append(f"this matches '{domain}', a knowledge area already in your vault.")

    top_branch = "study" if scores["study"] >= scores["practise"] else "practise"
    other_branch = "practise" if top_branch == "study" else "study"
    gap = scores[top_branch] - scores[other_branch]

    local_reason = reasons[top_branch][0] if reasons[top_branch] else (
        "study is the safer default for conceptual/internalisation work."
        if top_branch == "study"
        else "the request sounds like a drill or practice block."
    )
    if gap >= 0.18 or scores[top_branch] >= 0.72:
        return LearningRouteDecision(
            branch=top_branch,
            reason=local_reason,
            confidence=gap,
            source="local",
        )

    prompt = {
        "intent": intent,
        "local_scores": scores,
        "goals": [
            {
                "title": goal.title,
                "domain": getattr(goal, "domain", ""),
                "mode": getattr(goal, "execution_mode", "mixed"),
            }
            for goal in repo.list_goal_arcs(status=None)[:6]
        ],
        "tracks": [
            {
                "name": track.name,
                "description": getattr(track, "description", ""),
            }
            for track in repo.list_tracks(active_only=True)[:6]
        ],
        "vault_domains": _list_knowledge_domains()[:10],
        "instruction": (
            "Choose the best branch for this learning request. "
            "study = conceptual/internalisation/verifying new knowledge. "
            "practise = deliberate drills, reps, performance improvement. "
            "Return strict JSON with keys branch and reason."
        ),
    }
    raw = _generate_policy_json(prompt, operation="routing", max_output_tokens=4000)
    if not raw:
        return LearningRouteDecision(branch=top_branch, reason=local_reason, confidence=gap, source="local")

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        payload = json.loads(raw[start : end + 1])
        branch = payload.get("branch", "").strip().lower()
        if branch not in {"study", "practise"}:
            raise ValueError(branch)
        reason = payload.get("reason", "").strip() or local_reason
        return LearningRouteDecision(branch=branch, reason=reason, confidence=gap, source="model_policy")
    except Exception:
        return LearningRouteDecision(branch=top_branch, reason=local_reason, confidence=gap, source="local")


def build_next_candidates(repo, *, limit: int = 5) -> list[CommandCandidate]:
    """Build deterministic next-action recommendations from local state."""
    candidates: list[CommandCandidate] = []
    recent_study_scope: str | None = None
    recent_practise_scope: str | None = None
    now = utc_now()
    recent_completed_scopes: list[tuple[str, str]] = []

    active_session = repo.get_active_session()
    if active_session is not None:
        branch = getattr(active_session, "branch", "study") or "study"
        subject = getattr(active_session, "subject_scope", "") or ""
        _add_candidate(
            candidates,
            "finish",
            f"Wrap up the active {branch} session{f' on {subject}' if subject else ''}.",
            1.0,
            "active_session",
            human_label=f"Finish the current {branch} session" if subject else "Finish the current session",
            short_reason=f"You already have {branch} work in flight{f' on {subject}' if subject else ''}.",
            semantic_action="finish_session",
        )
        _add_candidate(
            candidates,
            "pause",
            f"Pause the active {branch} session and come back later.",
            0.98,
            "active_session",
            human_label="Pause the current session",
            short_reason="Keep the session state, but get out cleanly.",
            semantic_action="pause_session",
        )

    for reminder in repo.list_due_action_reminders(now)[:2]:
        _add_candidate(
            candidates,
            f"next --reminder {reminder.id}",
            f"{reminder.title}: {reminder.message}",
            0.96,
            "reminder",
            human_label="Handle the pending reminder",
            short_reason=reminder.message,
            semantic_action="open_reminder",
        )

    for session in repo.list_sessions_in_range(now - timedelta(days=1), now):
        branch = (session.branch or "study").lower()
        scope = (session.subject_scope or "").strip()
        if not scope or getattr(session, "end_at", None) is None:
            continue
        completion = getattr(session, "completion_pct", None)
        if completion is not None and completion < 80:
            continue
        recent_completed_scopes.append(("practise" if branch == "practice" else branch, scope))

    if active_session is None:
        resumable = []
        for task in repo.list_tasks():
            sessions = repo.list_sessions_for_task(task.id)
            if sessions and task.completion < 100 and task.archived_at is None:
                branch = _task_branch(task)
                scope = _scope_for_task(task).strip()
                if any(
                    completed_branch == branch and _text_match_score(scope, completed_scope) >= 0.45
                    for completed_branch, completed_scope in recent_completed_scopes
                ):
                    continue
                resumable.append(task)
        if resumable:
            task = sorted(resumable, key=lambda item: item.updated_at, reverse=True)[0]
            resume_command = _learner_command_for_task(task)
            _add_candidate(
                candidates,
                resume_command,
                f"Resume the next learning block for {task.title}.",
                0.92,
                "resume",
                human_label=f"Continue {task.title}",
                short_reason="You already opened this learning thread and it still needs evidence.",
            )

        for task in _best_todo_tasks(repo)[:2]:
            _add_candidate(
                candidates,
                f"start {display_ref(task, 'task')}",
                _todo_start_reason(task),
                0.945 if getattr(task, "due_date", None) else 0.905,
                "todo",
                human_label=f"Start {task.title}",
                short_reason="It is still top of mind and ready to act on.",
                semantic_action="start_task",
            )

    for block in repo.list_time_blocks_for_date(now):
        task = repo.get_task(block.task_id)
        if task is None or task.completion >= 100 or task.archived_at is not None:
            continue
        branch = _task_branch(task, getattr(block, "block_kind", "study"))
        planned_scope = _scope_for_task(task) or task.title
        if any(
            completed_branch == branch and _text_match_score(planned_scope, completed_scope) >= 0.45
            for completed_branch, completed_scope in recent_completed_scopes
        ):
            continue
        command = _learner_command_for_task(task, block_kind=getattr(block, "block_kind", "study"))
        if command.startswith("teach "):
            reason = f"Start the planned teaching session for {planned_scope}."
        elif branch == "practise":
            reason = f"Start the planned practise block for {planned_scope}."
        else:
            reason = f"Start the planned study block for {planned_scope}."
        _add_candidate(
            candidates,
            command,
            reason,
            0.9,
            "plan",
            human_label=f"Start the planned {branch} block for {planned_scope}",
            short_reason="It is already the next scheduled learning block.",
        )
        break

    for session in repo.list_sessions_in_range(now - timedelta(days=2), now):
        branch = (session.branch or "study").lower()
        scope = (session.subject_scope or "").strip()
        if branch == "study" and scope and recent_study_scope is None:
            recent_study_scope = scope
        if branch in {"practise", "practice"} and scope and recent_practise_scope is None:
            recent_practise_scope = scope
        if recent_study_scope and recent_practise_scope:
            break

    try:
        from pb.vault.anki_client import get_cards_by_status, get_pending_card_count

        suggested_cards = len(get_cards_by_status("suggested"))
        export_ready_cards = get_pending_card_count()
    except Exception:
        suggested_cards = 0
        export_ready_cards = 0

    if suggested_cards:
        _add_candidate(
            candidates,
            "anki list --suggested",
            f"Review {suggested_cards} suggested Anki candidates before they go stale.",
            0.89,
            "anki",
            human_label="Review suggested recall cards",
            short_reason=f"You have {suggested_cards} suggested cards waiting.",
            semantic_action="review_recall",
        )
    if export_ready_cards:
        _add_candidate(
            candidates,
            "anki export",
            f"Export {export_ready_cards} accepted or edited Anki candidates.",
            0.88,
            "anki",
            human_label="Export ready recall cards",
            short_reason=f"{export_ready_cards} cards are ready to leave the vault and reach Anki.",
            semantic_action="export_recall",
        )

    active_goals = repo.list_goal_arcs()
    if not active_goals:
        _add_candidate(
            candidates,
            "goal",
            "Set or refine a goal so study and practise have a clear basis.",
            0.88,
            "goal",
            human_label="Clarify your learning goal",
            short_reason="Direction should come before more study blocks.",
            semantic_action="clarify_goal",
        )
    else:
        _add_candidate(
            candidates,
            "plan day",
            "Turn your active goals into today's study and practise blocks.",
            0.86,
            "plan",
            human_label="Shape today's learning plan",
            short_reason="Translate your active goals into the next concrete blocks.",
            semantic_action="plan_day",
        )
        goal = active_goals[0]
        goal_target = goal.domain or goal.title
        if goal.execution_mode in {"mixed", "study"}:
            _add_candidate(
                candidates,
                f"study {_quote(goal_target)}",
                f"Work on the conceptual side of {goal.title}.",
                0.82,
                "goal",
                human_label=f"Study {goal.title}",
                short_reason="This goal still needs conceptual progress.",
            )
        if goal.execution_mode in {"mixed", "practise", "practice"}:
            _add_candidate(
                candidates,
                f"practise {_quote(goal_target)}",
                f"Do an embodied practice block for {goal.title}.",
                0.81,
                "goal",
                human_label=f"Practise {goal.title}",
                short_reason="This goal needs reps, not just planning.",
            )
        if goal.execution_mode in {"mixed", "study"} and getattr(goal, "target_bloom_stage", None):
            current = getattr(goal, "current_bloom_stage", None)
            if current != goal.target_bloom_stage:
                _add_candidate(
                    candidates,
                    f"study {_quote(goal_target)}",
                    f"{goal.title} still needs deeper conceptual work.",
                    0.84,
                    "goal_gap",
                    human_label=f"Deepen your understanding of {goal.title}",
                    short_reason="The conceptual target is still not met.",
                )
                _add_candidate(
                    candidates,
                    f"teach {_quote(goal_target)}",
                    f"Use a guided teaching loop to deepen and connect {goal.title}.",
                    0.835,
                    "goal_gap",
                    human_label=f"Work through {goal.title} with a teaching loop",
                    short_reason="Explaining it should expose the missing links.",
                )
        if goal.execution_mode in {"mixed", "practise", "practice"} and getattr(goal, "target_practice_stage", None):
            current = getattr(goal, "current_practice_stage", None)
            if current != goal.target_practice_stage:
                _add_candidate(
                    candidates,
                    f"practise {_quote(goal_target)}",
                    f"{goal.title} still needs deliberate practice to reach {getattr(goal.target_practice_stage, 'value', goal.target_practice_stage)}.",
                    0.83,
                    "goal_gap",
                    human_label=f"Practise toward {goal.title}",
                    short_reason="The practice stage target is still not met.",
                )

    if recent_study_scope:
        _add_candidate(
            candidates,
            f"study recall {_quote(recent_study_scope)}",
            f"Generate recall prompts while {recent_study_scope} is still fresh.",
            0.8,
            "recall",
            human_label=f"Generate recall prompts for {recent_study_scope}",
            short_reason="Capture the study gains before they decay.",
            semantic_action="generate_recall",
        )

    # Phase 2: Retry queue items boost priority for retry-worthy sessions (per D-20)
    try:
        from pb.storage.database import get_connection
        today_str = now.strftime("%Y-%m-%d")
        with get_connection() as conn:
            retry_rows = conn.execute(
                "SELECT id, domain, item_text, priority FROM retry_queue "
                "WHERE status = 'pending' AND (cooldown_until IS NULL OR cooldown_until <= ?) "
                "ORDER BY priority ASC LIMIT 3",
                (today_str,),
            ).fetchall()
            for row in retry_rows:
                domain = row["domain"] or "general"
                item_text = row["item_text"]
                priority = row["priority"]
                # Priority scoring: weakness=0.88, incomplete=0.85, optional=0.80
                # These stay below active session (1.0) and reminders (0.96)
                score_map = {1: 0.88, 2: 0.85, 3: 0.80}
                score = score_map.get(priority, 0.80)
                _add_candidate(
                    candidates,
                    f"practise {domain}",
                    f"Retry: {item_text[:80]}",
                    score,
                    "retry_queue",
                    human_label=f"Retry the weak spot in {domain}",
                    short_reason=item_text[:80],
                )
            # Update cooldown for surfaced items (per D-15 Pitfall 4)
            if retry_rows:
                tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
                surfaced_ids = [row["id"] for row in retry_rows]
                placeholders = ",".join("?" for _ in surfaced_ids)
                conn.execute(
                    f"UPDATE retry_queue SET cooldown_until = ? WHERE id IN ({placeholders})",
                    [tomorrow] + surfaced_ids,
                )
                conn.commit()
    except Exception:
        pass  # Non-fatal: retry queue failure must not break pb next

    # Phase 10: surface active commitments in pb next (ACCT-01, D-03 passive+contextual)
    try:
        from pb.storage.database import get_connection
        with get_connection() as conn:
            commitment_rows = conn.execute(
                """
                SELECT c.id, c.description, c.due_date, ds.agent_id
                FROM commitments c
                LEFT JOIN dispatch_sessions ds ON ds.id = c.session_id
                WHERE c.status = 'active'
                """
            ).fetchall()
        active_commitments = sort_commitments_for_next([dict(row) for row in commitment_rows])[:2]
        for c in active_commitments:
            desc = c["description"][:50] if c["description"] else "commitment"
            _add_candidate(
                candidates,
                f'do "check in on: {desc}"',
                f"You committed to: {c['description'][:80]}",
                0.91,
                "commitment",
                human_label=f"Follow up on: {desc}",
                short_reason="This commitment is still open.",
                agent_id=str(c.get("agent_id") or ""),
            )
    except Exception:
        pass  # Non-fatal: commitment surfacing must never break pb next

    # Phase 10: surface neglected goals (ACCT-02, D-04)
    try:
        now_for_neglect = utc_now()
        for goal in active_goals if active_goals else []:
            last_session_at = getattr(goal, "last_session_at", None)
            if last_session_at and (now_for_neglect - last_session_at) > timedelta(days=7):
                days_ago = (now_for_neglect - last_session_at).days
                _add_candidate(
                    candidates,
                    f'do "work on {goal.title[:30]}"',
                    f"{goal.title} hasn't been touched in {days_ago} days.",
                    0.88,
                    "neglected_goal",
                    human_label=f"Neglected: {goal.title[:40]}",
                    short_reason=f"No activity in {days_ago} days.",
                )
    except Exception:
        pass  # Non-fatal

    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    return ordered[:limit]


# Phrases that signal a prioritisation / next-action / status request rather than a
# learning topic. These route to build_next_candidates (the real next-action surface)
# instead of being echoed back as a "Study <request>" task (kills the echo collapse).
_NEXT_INTENT_PHRASES = (
    "next", "what should", "what now", "what do i do", "what to do",
    "most important", "single most", "single thing", "one thing",
    "top priority", "the top one", "priorit",  # prioritise/prioritize/prioritised
    "my todos", "my to-dos", "my tasks", "my list", "my commitments",
    "from my list", "which task", "which of my", "tell me which",
    "show me my", "status report", "the big picture", "overview",
    "where do i start", "where should i start",
)


def _looks_like_topic(intent: str) -> bool:
    """True if *intent* reads like a clean learning TOPIC (safe to scope a study
    session around) rather than a conversational/first-person request.

    Guards the study/practise fallback against echoing a whole user sentence back as
    a "Study <sentence>" task title. "communication - how to speak with charisma"
    returns True; "just give me the single most important one" returns False.
    """
    low = " ".join(intent.strip().lower().split())
    if not low:
        return False
    padded = f" {low} "
    # Sentence prose — questions, exclamations, or a mid-string sentence boundary —
    # is never a clean topic. This is the language-agnostic guard: it catches long
    # declarative and non-English rants that carry no English first-person markers
    # ("Hallo Sofia! ...", "... zero minutes?").
    if "?" in low or "!" in low or ". " in low:
        return False
    conversational_markers = (
        # English first-person / imperative framing
        " i ", " i'm", " i’ve", " i've", " i am ", " my ", " me ", " me,", " me.",
        "give me", "tell me", "show me", " stop ", " just ", " don't", " do not",
        " you ", " your ", " we ", "let's", "let’s",
        # Non-English first-person / request markers (German, Spanish, French) so a
        # short foreign-language rant is not echoed. Restricted to tokens that are
        # not English homographs (no "dame"/"dime"/"io"/"mi") to protect tech topics.
        " ich ", " mir ", " mich ", " gib ", " bitte ", " kannst ",
        " quiero ", " necesito ", " moi ", " donne ",
    )
    if any(marker in padded for marker in conversational_markers):
        return False
    if len(low.split()) > 12:
        return False
    return True


def suggest_commands_for_intent(repo, intent: str, *, limit: int = 5) -> list[CommandCandidate]:
    """Turn a free-text request into one or more likely commands."""
    lowered = " ".join(intent.lower().split())
    if not lowered:
        return build_next_candidates(repo, limit=limit)

    candidates: list[CommandCandidate] = []

    # --- Scope pre-resolution: try known objects before keyword heuristics ---
    matched_goal = match_goal(repo, intent)
    if matched_goal:
        mode = getattr(matched_goal, "execution_mode", "mixed")
        # Let explicit intent keywords override goal mode so "practise X" never routes to study
        _kw_low = " ".join(intent.lower().split())
        _intent_practise = any(kw in _kw_low for kw in _PRACTISE_KEYWORDS)
        _intent_study = any(kw in _kw_low for kw in _STUDY_KEYWORDS)
        if _intent_practise and not _intent_study and mode != "study":
            branch = "practise"
        elif _intent_study and not _intent_practise and mode not in ("practise", "practice"):
            branch = "study"
        else:
            branch = "practise" if mode in ("practise", "practice") else "study"
        domain = getattr(matched_goal, "domain", "") or getattr(matched_goal, "title", "")
        _add_candidate(
            candidates,
            f"{branch} {_quote(domain)}",
            f"Matched existing goal: {matched_goal.title}",
            0.92,
            human_label=f"Continue goal: {matched_goal.title}",
            short_reason=f"Your '{matched_goal.title}' goal suggests {branch}.",
            semantic_action=branch,
        )

    matched_track = match_track(repo, intent)
    if matched_track:
        track_name = getattr(matched_track, "name", intent)
        _add_candidate(
            candidates,
            f"study {_quote(track_name)}",
            f"Matched existing track: {track_name}",
            0.90,
            human_label=f"Study track: {track_name}",
            short_reason=f"Your '{track_name}' track is a direct match.",
            semantic_action="study",
        )
    # --- End scope pre-resolution ---

    if any(phrase in lowered for phrase in _NEXT_INTENT_PHRASES):
        return build_next_candidates(repo, limit=limit)

    if any(term in lowered for term in ("goal", "curriculum", "direction", "north star")):
        _add_candidate(candidates, "goal", "Create or refine a goal first.", 0.95)
    if "plan week" in lowered or ("week" in lowered and "plan" in lowered):
        _add_candidate(candidates, "plan week", "Generate a weekly framework from your goals.", 0.94)
    if "plan day" in lowered or ("today" in lowered and "plan" in lowered):
        _add_candidate(candidates, "plan day", "Generate a daily plan from your active goals.", 0.94)
    if lowered.startswith("plan") or "framework" in lowered:
        _add_candidate(candidates, "plan", "Shape the short-to-medium-term learning framework.", 0.9)
    if any(term in lowered for term in ("review", "reflect", "reflection", "stats", "diary")):
        _add_candidate(candidates, "review day", "Open the optional reflection and progress view.", 0.9)
    if any(term in lowered for term in _TEACH_KEYWORDS):
        topic = _topic_without_teach_phrases(intent)
        _add_candidate(
            candidates,
            f"teach {_quote(topic)}",
            "This sounds like guided conceptual teaching, so `pb teach` is the best fit.",
            0.97,
        )
    if any(term in lowered for term in ("recall", "flashcard", "flashcards", "anki")):
        _add_candidate(candidates, "study recall", "Run a scoped recall flow instead of passive review.", 0.93)
    explicit_practise = any(keyword in lowered for keyword in _PRACTISE_KEYWORDS)
    if explicit_practise:
        topic = _topic_without_practise_phrases(intent)
        if _looks_like_topic(topic):
            _add_candidate(
                candidates,
                f"practise {_quote(topic)}",
                "The request asks for reps or drills, so practise is the right first move.",
                0.98,
            )
    if any(term in lowered for term in ("vocab", "word", "words")):
        _add_candidate(candidates, "study vocab", "Work through vocabulary in the study flow.", 0.93)
    if "resume" in lowered:
        _add_candidate(candidates, "resume", "Resume the most relevant paused task.", 0.91)

    if any(term in lowered for term in ("todo", "deadline", "due", "task", "errand", "follow up", "follow-up", "need to")):
        _add_candidate(
            candidates,
            f"todo {_quote(intent.strip())}" if intent.strip() else "todo",
            "Capture this as a todo so it shows up in your next-action flow.",
            0.9,
            kind="capture",
            requires_input=not bool(intent.strip()),
            suggested_text=intent.strip(),
        )

    # Keyword-branch echo guard (A-07, P0): a conversational rant that merely
    # *contains* a study/practise keyword ("...a 45-minute session...", "...I don't
    # need a learning plan...") must never be echoed back as "study <whole rant>" /
    # "practise <whole rant>". Only route to study/practise when the request reads
    # like a CLEAN skill/learning TOPIC; otherwise fall through to the no-capable-mode
    # graceful next-action fallback below.
    if explicit_practise:
        topic = intent.strip()
        if _looks_like_topic(topic):
            _add_candidate(
                candidates,
                f"practise {_quote(topic)}",
                "This looks like embodied skill work, so practise is the best fit.",
                0.96,
            )
            _add_candidate(
                candidates,
                f"practise log {_quote(topic)}",
                "If you already practised, log the session outcome instead.",
                0.74,
            )

    if any(keyword in lowered for keyword in _STUDY_KEYWORDS):
        topic = intent.strip()
        if _looks_like_topic(topic):
            _add_candidate(
                candidates,
                f"study {_quote(topic)}",
                "This looks like theoretical or internalisation work, so study fits best.",
                0.96,
            )
            _add_candidate(
                candidates,
                f"study debrief {_quote(topic)}",
                "Use a debrief if the learning block already happened and you want consolidation.",
                0.76,
            )

    if "thought" in lowered or "note" in lowered or "capture" in lowered or lowered.startswith("write down"):
        suffix = intent.strip()
        command = f"thought {_quote(suffix)}" if suffix else "thought"
        _add_candidate(
            candidates,
            command,
            "Capture the thought directly in the vault.",
            0.84,
            kind="capture",
            requires_input=not bool(suffix),
            suggested_text=suffix,
        )

    if not candidates:
        topic = intent.strip()
        if topic and _looks_like_topic(topic):
            decision = route_learning_intent(repo, intent)
            _add_candidate(
                candidates,
                f"{decision.branch} {_quote(topic)}",
                decision.reason,
                0.9 if decision.source == "local" else 0.92,
                decision.source,
            )
        else:
            # No mode keyword/goal/track matched AND the request is not a clean
            # learning topic — surface the user's real next actions instead of
            # echoing the raw request back as a "Study <request>" task. This is the
            # no-capable-mode graceful fallback that kills the echo collapse.
            return build_next_candidates(repo, limit=limit)

    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    non_capture = [item for item in ordered if item.kind != "capture"]
    if len(non_capture) <= 1:
        return ordered[:limit]

    reranked = rerank_candidates_with_gemini(intent, ordered)
    return reranked[:limit]


def rerank_candidates_with_gemini(
    intent: str,
    candidates: Iterable[CommandCandidate],
) -> list[CommandCandidate]:
    """Use Gemini Flash to reorder candidates when available."""
    original = list(candidates)
    if len(original) <= 1:
        return original

    prompt = {
        "intent": intent,
        "candidates": [
            {"command": item.backing_command, "label": item.human_label, "reason": item.short_reason}
            for item in original
        ],
        "task": (
            "Reorder these candidate pb CLI commands from best to worst for the user's intent. "
            "Return strict JSON with one key named ordered_commands containing only the command strings."
        ),
    }
    raw = _generate_policy_json(prompt, operation="routing", max_output_tokens=4000)
    if not raw:
        return sorted(original, key=lambda item: item.score, reverse=True)

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        payload = json.loads(raw[start : end + 1])
        ordered_commands = payload.get("ordered_commands", [])
    except Exception:
        return sorted(original, key=lambda item: item.score, reverse=True)

    lookup = {item.backing_command: item for item in original}
    reranked: list[CommandCandidate] = []
    for command in ordered_commands:
        item = lookup.pop(command, None)
        if item is not None:
            reranked.append(
                CommandCandidate(
                    command=item.backing_command,
                    reason=item.short_reason,
                    score=item.confidence,
                    source="gemini",
                    kind=item.kind,
                    requires_input=item.requires_input,
                    suggested_text=item.suggested_text,
                    human_label=item.human_label,
                    short_reason=item.short_reason,
                    semantic_action=item.semantic_action,
                    backing_command=item.backing_command,
                    confidence=item.confidence,
                )
            )
    reranked.extend(sorted(lookup.values(), key=lambda item: item.score, reverse=True))
    return reranked
