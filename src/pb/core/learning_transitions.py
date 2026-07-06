# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""LLM-backed routing after a cleared learning session."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from rich.markup import escape

from pb.cli.console import get_console
from pb.cli.pickers import PickerResult, pick_single_choice
from pb.cli.preview import markdown_learning_plan_lines, preview_decision, render_markdown_preview
from pb.core.agent_lifecycle import AgentLifecycleSuggester
from pb.core.models import LearningTransition, Session, Task, utc_now
from pb.core.renderables import renderable_cli_text
from pb.llm.drafts import AgentSpawnProposalDraft, LearningPlanBlockDraft, LearningTransitionDraft
from pb.llm.runtime import DraftGenerationError


class LearningTransitionUnavailable(RuntimeError):
    """Raised when a tailored transition requires an unavailable LLM."""


@dataclass(frozen=True)
class TransitionCommand:
    """Accepted transition command and summary."""

    command: str
    summary: str
    draft: LearningTransitionDraft | None = None


class LearningTransitionService:
    """Generate, queue, and resume closeout transitions for learning sessions."""

    def __init__(self, repo: Any, runtime: Any | None = None, runtime_ctx: Any | None = None):
        self.repo = repo
        self.runtime = runtime
        self.runtime_ctx = runtime_ctx

    def llm_available(self) -> bool:
        try:
            return bool(self.runtime is not None and self.runtime.health().available)
        except Exception:
            return False

    def enqueue(self, *, route_kind: str, session: Session, task: Task | None, reason: str = "") -> LearningTransition:
        """Persist a deferred transition for future LLM processing."""
        payload = self._payload_for_session(session=session, task=task, reason=reason)
        transition = LearningTransition(
            route_kind=route_kind,
            source_session_id=session.id,
            source_task_id=session.task_id,
            payload_json=payload,
        )
        return self.repo.create_learning_transition(transition)

    def generate(self, *, route_kind: str, session: Session | None = None, task: Task | None = None, transition: LearningTransition | None = None) -> LearningTransitionDraft:
        """Generate a tailored transition draft from either live or queued context."""
        if not self.llm_available():
            raise LearningTransitionUnavailable("LLM unavailable for tailored learning transition.")
        payload = dict(transition.payload_json) if transition is not None else self._payload_for_session(session=session, task=task)
        prompt = self._transition_prompt(route_kind=route_kind, payload=payload)
        try:
            result = self.runtime.generate_draft(
                LearningTransitionDraft,
                prompt,
                source_scope=f"learning_transition:{route_kind}:{payload.get('session_id', '')}",
                max_output_tokens=8000,
            )
        except DraftGenerationError as exc:
            raise LearningTransitionUnavailable(exc.to_user_message()) from exc
        draft = result.payload
        if route_kind in {"next_session", "deferred_next_session"} and draft.next_session is None:
            raise LearningTransitionUnavailable("The model did not return a next-session plan.")
        return draft

    def choose_next_session(self, *, session: Session, task: Task | None) -> TransitionCommand | None:
        """Generate, preview, and accept a tailored next-session command."""
        draft = self.generate(route_kind="next_session", session=session, task=task)
        block = draft.next_session
        if block is None:
            raise LearningTransitionUnavailable("The model did not return a next-session plan.")
        self.render_next_session_preview(draft)
        decision = preview_decision(yes=False, action_label="Start this next session", allow_refinement=False)
        if decision.kind != "accept":
            return None
        return TransitionCommand(
            command=self.command_for_block(block),
            summary=draft.summary or block.reason or f"Next session: {block.subject_scope}",
            draft=draft,
        )

    def choose_continuation_focus(self, *, session: Session, task: Task | None) -> tuple[str, LearningTransitionDraft]:
        """Generate and choose a tailored continuation focus."""
        draft = self.generate(route_kind="continue", session=session, task=task)
        options = [
            (option.instruction or option.label, option.label)
            for option in draft.continuation_options[:5]
            if option.label.strip()
        ]
        details = [
            option.description or option.instruction
            for option in draft.continuation_options[:5]
            if option.label.strip()
        ]
        if not options:
            raise LearningTransitionUnavailable("The model did not return continuation choices.")
        selected = pick_single_choice(
            options,
            title="Continue session",
            text=draft.summary or "Choose how this extension should be conducted.",
            details=details,
            allow_inline_edit=True,
            inline_prompt="Discuss/custom continuation",
            return_result=True,
            allow_back_navigation=True,
        )
        if isinstance(selected, PickerResult):
            if selected.kind == "cancel":
                return "", draft
            if selected.kind == "inline_text":
                return str(selected.value or "").strip(), draft
            return str(selected.value or "").strip(), draft
        return str(selected or "").strip(), draft

    def render_next_session_preview(self, draft: LearningTransitionDraft) -> None:
        block = draft.next_session
        lines = markdown_learning_plan_lines([block], presentation=None) if block is not None else []
        render_markdown_preview(
            title="Next Session Draft",
            rows=[("Summary", draft.summary)],
            sections=[("Plan", lines)],
        )

    def command_for_block(self, block: LearningPlanBlockDraft) -> str:
        branch = "practise" if block.branch == "practise" else "study"
        scope = block.subject_scope or block.title
        argv = [branch, scope]
        if block.duration_minutes:
            argv.extend(["--duration", f"{block.duration_minutes}m"])
        argv.append("--yes")
        return shlex.join(argv)

    def maybe_offer_agent_spawn(
        self,
        *,
        session: Session,
        task: Task | None,
        draft: LearningTransitionDraft | None = None,
    ) -> None:
        """Offer an explicit agent spawn/respawn proposal and activate it when accepted."""
        if not self.llm_available():
            return
        suggester = AgentLifecycleSuggester(self.repo)
        domain = suggester.domain_from_session(session) or "general_learning"
        proposal = self._agent_spawn_proposal(session=session, task=task, draft=draft, domain=domain)
        render_markdown_preview(
            title="Agent Proposal",
            rows=[
                ("Agent", proposal.title or domain.replace("_", " ").title()),
                ("Domain", proposal.domain or domain),
            ],
            sections=[
                ("Why", [proposal.rationale or "Use the just-finished session context to steer this learning thread."]),
                ("Context", [proposal.context_summary or session.subject_scope or getattr(task, "title", "") or "Current learning context."]),
            ],
        )
        decision = preview_decision(yes=False, action_label="Spawn this agent", allow_refinement=False)
        if decision.kind != "accept":
            try:
                suggester.record_skip(domain)
            except Exception:
                pass
            return
        result = suggester.spawn_or_respawn(session=session)
        agent_domain = result.domain or domain
        agent_id = f"domain_{agent_domain}"
        try:
            from pb.agents.domain import create_domain_agent
            from pb.mcp.protocol import create_session

            create_domain_agent(
                agent_id,
                agent_domain.replace("_", " ").title(),
                agent_domain,
                goal_id=getattr(session, "goal_id", None),
            )
            create_session(
                agent_id,
                context={
                    "source": "learning_transition",
                    "learning_session_id": session.id,
                    "task_id": session.task_id,
                    "subject_scope": session.subject_scope,
                },
            )
        except Exception:
            pass
        get_console().print(f"[success]{escape(result.message or f'Activated {agent_domain} agent.')}[/]")

    def _agent_spawn_proposal(
        self,
        *,
        session: Session,
        task: Task | None,
        draft: LearningTransitionDraft | None,
        domain: str,
    ) -> AgentSpawnProposalDraft:
        if draft is not None and draft.agent_spawn is not None:
            proposal = draft.agent_spawn
            if proposal.domain:
                return proposal
        payload = self._payload_for_session(session=session, task=task)
        prompt = (
            "Create a succinct proposal for optionally spawning a ProductiveBrain domain agent.\n"
            "Return a concrete proposal, not a sales pitch. If no niche exists, use general_learning.\n"
            f"Preferred domain: {domain}\n"
            f"Session context JSON: {json.dumps(payload, ensure_ascii=True)[:5000]}\n"
        )
        try:
            return self.runtime.generate_draft(
                AgentSpawnProposalDraft,
                prompt,
                source_scope=f"agent_spawn:{session.id}",
                max_output_tokens=2000,
            ).payload
        except DraftGenerationError:
            return AgentSpawnProposalDraft(
                domain=domain,
                title=f"{domain.replace('_', ' ').title()} Agent",
                rationale="Keep future prompts tailored to the session context and recurring weak spots.",
                context_summary=session.subject_scope or getattr(task, "title", "") or "Current learning session.",
            )

    def _payload_for_session(self, *, session: Session | None, task: Task | None, reason: str = "") -> dict[str, object]:
        if session is None:
            return {"reason": reason}
        lesson_summary: dict[str, object] = {}
        try:
            run = self.repo.get_lesson_run(session.id)
            if run is not None:
                pages = self.repo.list_lesson_pages(run.id)
                questions = self.repo.list_lesson_questions(run.id)
                attempts = self.repo.list_lesson_attempts(run.id)
                lesson_summary = {
                    "lesson_title": getattr(run, "title", ""),
                    "lesson_status": getattr(run, "lesson_status", ""),
                    "total_points": getattr(run, "total_points", 0),
                    "retry_queue": list(getattr(run, "retry_queue", []) or []),
                    "pages": [getattr(page, "title", "") for page in pages[-6:]],
                    "questions": [
                        str(getattr(question, "prompt_json", {}).get("prompt", "") or "")[:180]
                        for question in questions[-12:]
                    ],
                    "attempt_results": [getattr(attempt, "result", "") for attempt in attempts[-12:]],
                }
        except Exception:
            lesson_summary = {}
        goals = []
        try:
            goals = [
                {
                    "id": goal.id,
                    "title": goal.title,
                    "domain": goal.domain,
                    "success": goal.success_definition,
                }
                for goal in self.repo.list_goal_arcs(status=None)[:6]
            ]
        except Exception:
            goals = []
        return {
            "reason": reason,
            "session_id": session.id,
            "task_id": session.task_id,
            "task_title": getattr(task, "title", "") if task is not None else "",
            "task_description": getattr(task, "description", "") if task is not None else "",
            "branch": session.branch,
            "subject_scope": renderable_cli_text(session.subject_scope),
            "actual_outcome": renderable_cli_text(session.actual_outcome or ""),
            "goal_id": session.goal_id or "",
            "track_id": session.track_id or "",
            "lesson": lesson_summary,
            "active_goals": goals,
        }

    def _transition_prompt(self, *, route_kind: str, payload: dict[str, object]) -> str:
        return (
            "You are routing the closeout of a ProductiveBrain learning session.\n"
            "Use the just-finished lesson context to generate tailored options. Do not use generic labels like "
            "'more application-based' or 'more theoretical' unless the session context specifically justifies those words.\n"
            "For next_session, include exactly one next_session block that is distinct from the just-finished session.\n"
            "For continue, include 3-5 continuation_options that extend the lesson rather than repeat cleared questions.\n"
            "Include an agent_spawn proposal when a niche or general learning agent would help.\n"
            f"Route kind: {route_kind}\n"
            f"Context JSON: {json.dumps(payload, ensure_ascii=True)[:9000]}\n"
        )
