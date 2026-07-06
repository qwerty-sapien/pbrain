# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Interactive lesson adapter over the unified lesson engine."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from pb.cli.console import get_console
from pb.cli.input_router import (
    PbCommandResolver,
    RoutedInput,
    classify_interactive_input,
    prompt_answer_or_command,
)
from pb.cli.markdown import render_markdown_to_rich_blocks
from pb.cli.pickers import PickerResult, pick_many_choices, pick_single_choice
from pb.core.lesson_engine import (
    LESSON_FINISH_CONTINUE_SESSION,
    LESSON_FINISH_MENU_ACTION,
    LESSON_FINISH_NEXT_SESSION,
    LessonEngine,
    lesson_finish_review_label,
)
from pb.core.registry import CommandHandler, CommandRegistry
from pb.core.naming import stored_display_title
from pb.core.renderables import renderable_cli_text
from pb.core.session_activity import (
    append_learning_partner_activity,
    format_learning_partner_activity_transcript_item,
    summarize_learning_partner_activity,
)
from pb.llm.drafts import LearningPartnerTurnDraft


def _transcript_path(data_dir: Path, session_id: str, session_slug: str = "") -> Path:
    """Return the durable transcript path for one learning session."""
    filename = session_slug or session_id
    return data_dir / "transcripts" / f"{filename}.json"


def _format_points(value: float) -> str:
    return f"{float(value):g}"


def _rich_block_plain(block: object) -> str:
    plain = getattr(block, "plain", None)
    if plain is None:
        return "renderable"
    return str(plain or "")


def load_session_transcript(data_dir: Path, session_id: str, session_slug: str = "") -> list[dict[str, str]]:
    """Load the durable user/assistant transcript for a session."""
    if not session_id:
        return []
    candidate_paths = []
    if session_slug:
        candidate_paths.append(_transcript_path(data_dir, session_id, session_slug))
    candidate_paths.append(_transcript_path(data_dir, session_id))

    raw = ""
    for path in candidate_paths:
        try:
            raw = path.read_text(encoding="utf-8")
            break
        except FileNotFoundError:
            continue
        except OSError:
            continue
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []

    transcript: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        transcript.append({"role": role, "content": content})
    return transcript


def save_session_transcript(
    data_dir: Path,
    session_id: str,
    transcript: list[dict[str, str]],
    session_slug: str = "",
) -> None:
    """Persist the durable user/assistant transcript for a session atomically."""
    if not session_id:
        return
    path = _transcript_path(data_dir, session_id, session_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(transcript, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    if session_slug:
        legacy = _transcript_path(data_dir, session_id)
        if legacy != path:
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(
                json.dumps(transcript, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )


@dataclass
class PartnerRunResult:
    action: str
    summary: str = ""
    note_path: Path | None = None
    recall_candidates: list[str] = field(default_factory=list)
    detected_gaps: list[str] = field(default_factory=list)
    next_drill: str = ""
    command: str = ""


class LearningPartnerSession:
    """CLI lesson shell built on top of the persistent lesson engine."""

    def __init__(
        self,
        *,
        runtime,
        runtime_ctx,
        repo,
        task,
        session,
        branch: str,
        objective: str,
        topic: str,
        domain: str,
        clarifier_answers: dict[str, str] | None = None,
        mode: str = "",
        verbose: bool = False,
        confidence_level: float = 0.0,
        max_options: int = 5,           # D-16-25: 6/8/10 for practice hard difficulty
        pb_command_resolver: PbCommandResolver | None = None,
    ):
        self.runtime = runtime
        self.runtime_ctx = runtime_ctx
        self.repo = repo
        self.task = task
        self.session = session
        self.branch = branch
        self.objective = objective
        self.topic = topic
        self.domain = domain
        self.mode = mode or branch
        self.verbose = verbose
        self.confidence_level = max(0.0, min(1.0, float(confidence_level)))
        self.max_options = max_options
        self.clarifier_answers = clarifier_answers or {}
        self.console = get_console()
        fallback_root = getattr(runtime_ctx, "quarantine_path", Path("."))
        self.data_dir = Path(getattr(runtime_ctx, "data_dir", fallback_root))
        self.pb_command_resolver = pb_command_resolver or self._default_pb_command_resolver()
        self.generated_names = dict(getattr(self.session, "generated_names", {}) or {})
        self.session_slug = str(self.generated_names.get("session_slug", "") or "").strip()
        self.transcript = load_session_transcript(
            self.data_dir,
            getattr(self.session, "id", ""),
            self.session_slug,
        )
        self.evidence_log = self._load_evidence_log()
        self.collected_recall: list[str] = []
        self.collected_gaps: list[str] = []
        self.collected_corrections: list[str] = []
        self.next_drill: str = ""
        self.current_turn: LearningPartnerTurnDraft | None = None
        self.view_page_slug: str = ""
        self.view_question_slug: str = ""
        self._view_history: list[tuple[str, str]] = []
        self._preserve_next_render: bool = False
        self._feynman_opening: str = ""  # D-16-21: set by open_with_first_move when branch=="teach"
        self.engine = LessonEngine(
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            repo=repo,
            task=task,
            session=session,
            branch=branch,
            objective=objective,
            topic=topic,
            domain=domain,
            mode=mode,
            clarifier_answers=self.clarifier_answers,
            confidence_level=self.confidence_level,
        )
        self.command_registry = self._build_command_registry()
        self._mark_partner_session_used()
        self._sync_session_metadata()

    def start(self) -> PartnerRunResult:
        current_turn = self.open_with_first_move()

        while True:
            self.set_current_turn(current_turn)
            self._render_session_frame(current_turn)
            picker_input = self._render_question_input(current_turn)
            if picker_input is not None:
                if isinstance(picker_input, str):
                    picker_input = RoutedInput(kind="answer", text=picker_input)
                if picker_input.kind == "navigation":
                    self._browse(picker_input.argv or (picker_input.command,))
                    continue
                if picker_input.kind == "lesson_continue":
                    self._reset_view()
                    current_turn = self.continue_after_clear()
                    continue
                if picker_input.kind == "answer":
                    self._reset_view()
                    current_turn = self.respond_once(picker_input.text)
                    continue
                if picker_input.kind == "slash_command":
                    self._reset_view()
                    next_turn = self.run_contextual_command(picker_input.command)
                    if next_turn is not None:
                        current_turn = next_turn
                    continue
                if picker_input.kind in {"slash_ambiguous", "slash_unknown"}:
                    self.explain_contextual_command_error(picker_input)
                    continue
                if picker_input.kind == "pb_command":
                    return self._result_from_command(picker_input.text)
                continue

            try:
                raw = input("You> ").strip()
            except EOFError:
                return self._finalize("pause", "Paused the lesson session.")
            except KeyboardInterrupt:
                self.console.print("")
                continue

            if not raw:
                continue
            decision = classify_interactive_input(
                raw,
                pb_command_resolver=self.pb_command_resolver,
                slash_registry=self.command_registry,
                active_learning=True,
                allow_shell_commands=False,
                allow_nl_dispatch=False,
            )
            if decision.kind == "pb_command":
                return self._result_from_command(decision.text)
            if decision.kind == "navigation":
                self._browse(decision.argv or (decision.command,))
                continue
            if decision.kind == "slash_command":
                self._reset_view()
                next_turn = self.run_contextual_command(decision.command, decision.args)
                if next_turn is not None:
                    current_turn = next_turn
                continue
            if decision.kind in {"slash_ambiguous", "slash_unknown"}:
                self.explain_contextual_command_error(decision)
                continue
            if decision.kind == "answer":
                self._reset_view()
                current_turn = self.respond_once(decision.text)

    def respond_once(self, user_input: str) -> LearningPartnerTurnDraft:
        """Generate and persist one learner-answer turn."""
        self._feynman_opening = ""  # D-16-21: clear after first learner response
        activity_context = self._activity_context_for_current_question()
        previous_attempt_count = self._attempt_count_for_activity_context(activity_context)
        turn = self.engine.answer_current(user_input)
        self.current_turn = turn
        activity = self._record_answer_activity(
            activity_context=activity_context,
            user_input=user_input,
            previous_attempt_count=previous_attempt_count,
        )
        self._record_exchange(user_input, turn, activity=activity)
        if activity:
            self._print_activity_receipt(activity)
        return turn

    def continue_after_clear(self) -> LearningPartnerTurnDraft:
        """Continue a cleared lesson with an implications/applications page."""

        self._feynman_opening = ""
        turn = self.engine.continue_after_clear()
        self.current_turn = turn
        self._append_assistant_turn(self._assistant_log_text(turn))
        self._sync_session_metadata()
        return turn

    def open_with_first_move(self) -> LearningPartnerTurnDraft:
        """Return the active lesson page/question without duplicating state."""
        # D-16-21: Feynman explain-back — learner speaks first, tutor listens.
        # No LLM generation for the first move in teach mode.
        if getattr(self, "branch", "") == "teach":
            first_prompt = (
                f"Please explain {self.topic} in your own words — as if teaching someone "
                "who hasn't seen it before. Don't worry about being perfect; "
                "just share what you know."
            )
            # Direct return: NO LLM call, NO engine.current_turn()
            feynman_turn = LearningPartnerTurnDraft(
                reply=first_prompt,
                question_type="free_text",
                support_cards=[],
                next_action="Explain the concept, then we will identify gaps and explore them.",
            )
            self._feynman_opening = first_prompt  # store for frame renderer
            self.current_turn = feynman_turn
            self._sync_session_metadata()
            if not self.transcript:
                self._append_assistant_turn(first_prompt)
            return feynman_turn
        self._mark_partner_session_used()
        opening = self.engine.current_turn()
        self.current_turn = opening
        self._sync_session_metadata()
        if not self.transcript:
            self._append_assistant_turn(self._assistant_log_text(opening))
        return opening

    def set_current_turn(self, turn: LearningPartnerTurnDraft | None) -> None:
        """Track the currently rendered lesson turn."""
        self.current_turn = turn

    def contextual_command_names(self) -> list[str]:
        """Return the active contextual slash commands for this session."""
        return self.command_registry.command_names()

    def _contextual_command_specs(self) -> list[tuple[str, str]]:
        return [
            ("/hint", "Give a targeted hint without revealing the full answer."),
            ("/answer", "Reveal the answer, mark it as revealed, and queue a retry."),
            ("/harder", "Regenerate the current question at a harder level."),
            ("/easier", "Regenerate the current question at an easier level."),
            ("/intuitive", "Explain the concept intuitively without directly giving the answer."),
            ("/skip", "Move past a revealed or blocked question and keep its retry pinned."),
            ("/recall", "Show compact recall prompts for the current lesson."),
            ("/explain", "Explain the current concept more directly."),
            ("/drill", "Generate a fresh drill on the same underlying concept."),
            ("/context", "Manage context lock and status from inside the lesson."),
            ("/lock", "Lock the current lesson context for future commands."),
            ("/unlock", "Unlock the currently locked context."),
            ("/spawn", "Spawn or revive the best domain agent for this session."),
            ("/forget", "Archive the current domain agent without deleting history."),
        ]

    def _build_command_registry(self) -> CommandRegistry:
        registry = CommandRegistry()
        for command, help_text in self._contextual_command_specs():
            registry.register(
                CommandHandler(
                    name=command,
                    help_text=help_text,
                    handler=lambda args, ctx: None,
                )
            )
        return registry

    def _partner_help_lines(self) -> list[str]:
        return self.command_registry.help_lines()

    def explain_contextual_command_error(self, decision: RoutedInput) -> None:
        """Surface a deterministic error when a contextual slash command is invalid."""
        if decision.kind == "slash_ambiguous" and decision.matches:
            joined = ", ".join(decision.matches)
            self.console.print(f"[warn]Ambiguous command. Matches: {joined}[/]")
            return

        head = (decision.text or "").split()[0].lower()
        if head in {"/finish", "/pause", "/resume"}:
            if head == "/finish":
                self.console.print("[warn]Choose a finish option from the cleared lesson menu.[/]")
            else:
                self.console.print("[warn]Use `pause` or `resume` without a slash.[/]")
            return

        available = ", ".join(self.contextual_command_names())
        self.console.print(f"[warn]Unknown contextual command. Available: {available}[/]")

    def _current_context_scope(self):
        from pb.cli.context_runtime import session_active_context_scope

        return session_active_context_scope(self.session)

    def _render_context_feedback(self, lines: list[str]) -> LearningPartnerTurnDraft:
        self.engine.last_feedback = [line for line in lines if str(line).strip()]
        return self.engine.current_turn()

    def _lock_context_from_ref(self, ref: str):
        from pb.core.context_file_intake import active_context_from_bundle, active_context_from_sources

        bundle = self.repo.get_source_bundle_by_name(ref)
        if bundle is not None:
            return active_context_from_bundle(bundle, locked=True)
        source = self.repo.find_context_source(ref)
        if source is not None:
            return active_context_from_sources(
                [str(source["source_ref"])],
                label=str(source.get("domain_name") or source.get("filename") or "context"),
                domain_id=str(source.get("domain_id", "") or "") or None,
                scope_mode=str(source.get("scope_mode", "unclear")),
                scope_boundary=str(source.get("scope_boundary", "")),
                locked=True,
            )
        return None

    def _context_command(self, args: str) -> LearningPartnerTurnDraft:
        from pb.cli.context_runtime import attach_active_context
        from pb.core.context_file_intake import summarize_context_label

        command_args = (args or "").strip()
        current_scope = self._current_context_scope()
        if not command_args or command_args == "status":
            locked = self.repo.get_locked_context()
            if locked is None:
                return self._render_context_feedback(["No context is currently locked."])
            return self._render_context_feedback(
                [
                    f"Locked context: {summarize_context_label(locked)}",
                    f"Mode: {locked.mode}",
                    f"Scope mode: {locked.scope_mode}",
                    f"Boundary: {locked.scope_boundary or 'None'}",
                ]
            )
        if command_args == "unlock":
            self.repo.clear_locked_context()
            if current_scope is not None:
                current_scope.locked = False
                attach_active_context(self.session, current_scope)
                self.repo.update_session(self.session)
            return self._render_context_feedback(["Context unlocked."])
        if command_args == "lock":
            if current_scope is None:
                return self._render_context_feedback(["There is no active lesson context to lock."])
            current_scope.locked = True
            self.repo.set_locked_context(current_scope)
            attach_active_context(self.session, current_scope)
            self.repo.update_session(self.session)
            return self._render_context_feedback([f"Locked context: {summarize_context_label(current_scope)}"])
        if command_args.startswith("lock "):
            ref = command_args[len("lock "):].strip()
            scope = self._lock_context_from_ref(ref)
            if scope is None:
                return self._render_context_feedback([f"No bundle or source matched `{ref}`."])
            self.repo.set_locked_context(scope)
            attach_active_context(self.session, scope)
            self.repo.update_session(self.session)
            return self._render_context_feedback([f"Locked context: {summarize_context_label(scope)}"])
        return self._render_context_feedback(["Use `/context status`, `/context lock`, `/context lock <bundle>`, or `/context unlock`."])

    def _spawn_command(self) -> LearningPartnerTurnDraft:
        from pb.core.agent_lifecycle import AgentLifecycleSuggester

        result = AgentLifecycleSuggester(self.repo).spawn_or_respawn(session=self.session)
        return self._render_context_feedback([result.message])

    def _forget_agent_command(self) -> LearningPartnerTurnDraft:
        from pb.core.agent_lifecycle import AgentLifecycleSuggester

        result = AgentLifecycleSuggester(self.repo).forget(session=self.session)
        return self._render_context_feedback([result.message])

    def run_contextual_command(self, command: str, args: str = "") -> LearningPartnerTurnDraft | None:
        """Execute one contextual slash command over the current lesson state."""
        activity_context = self._activity_context_for_current_question()
        if command == "/hint":
            turn = self.engine.use_hint()
        elif command == "/answer":
            turn = self.engine.reveal_current_answer()
        elif command == "/harder":
            turn = self.engine.change_difficulty("harder")
        elif command == "/easier":
            turn = self.engine.change_difficulty("easier")
        elif command == "/intuitive":
            turn = self.engine.explain_current(intuitive=True)
        elif command == "/skip":
            turn = self.engine.skip_current_question()
        elif command == "/explain":
            turn = self.engine.explain_current(intuitive=False)
        elif command == "/drill":
            turn = self.engine.drill_current()
        elif command == "/recall":
            self.engine.last_feedback = self.engine.recall_candidates()[:4]
            turn = self.engine.current_turn()
        elif command == "/context":
            turn = self._context_command(args)
        elif command == "/lock":
            turn = self._context_command("lock")
        elif command == "/unlock":
            turn = self._context_command("unlock")
        elif command == "/spawn":
            turn = self._spawn_command()
        elif command == "/forget":
            turn = self._forget_agent_command()
        else:
            return None
        return self._record_contextual_turn(turn, command=command, args=args, activity_context=activity_context)

    def _record_contextual_turn(
        self,
        turn: LearningPartnerTurnDraft,
        *,
        command: str = "",
        args: str = "",
        activity_context: dict[str, object] | None = None,
    ) -> LearningPartnerTurnDraft:
        self.current_turn = turn
        activity = self._record_command_activity(command=command, args=args, activity_context=activity_context)
        if activity:
            self.transcript.append(
                {
                    "role": "assistant",
                    "content": format_learning_partner_activity_transcript_item(activity),
                }
            )
        self._append_assistant_turn(self._assistant_log_text(turn))
        if activity:
            self._print_activity_receipt(activity)
        self._sync_session_metadata()
        return turn

    def _render_session_frame(self, turn: LearningPartnerTurnDraft) -> None:
        """Repaint the current page-oriented lesson frame."""
        self.console = get_console()
        should_clear = sys.stdin.isatty() and not self._preserve_next_render
        self._preserve_next_render = False
        if should_clear:
            try:
                self.console.clear()
            except Exception:
                pass

        snapshot = self._display_snapshot()
        turn = self._display_turn()
        lesson_title = snapshot.run.title or stored_display_title(self.task) or self.topic or "Lesson"
        mode_label = snapshot.run.lesson_mode
        page = snapshot.page
        page_index = (page.sequence_index + 1) if page is not None and page.page_slug != "mistakes" else len(
            [item for item in snapshot.pages if item.page_slug != "mistakes"]
        )
        normal_pages = [item for item in snapshot.pages if item.page_slug != "mistakes"]
        page_total = len(normal_pages) or len(snapshot.pages) or 1

        header = Text()
        header.append(lesson_title, style="bold white")
        try:
            from pb.cli.context_runtime import session_active_context_scope
            from pb.core.context_file_intake import summarize_context_label

            context_scope = session_active_context_scope(self.session)
            context_label = summarize_context_label(context_scope)
        except Exception:
            context_label = ""
        if context_label:
            header.append("  ")
            header.append(context_label, style="bold yellow")
        header.append("  ")
        header.append(mode_label, style=f"branch.{self.branch}")
        header.append("  ")
        header.append(f"{_format_points(snapshot.run.total_points)} pts", style="bold cyan")

        page_line = Text()
        if page is not None:
            page_line.append(f"Page {page_index}/{page_total}", style="bold blue")
            page_line.append("  ")
            page_line.append(page.title, style="bold white")
            if page.page_slug == "mistakes":
                page_line.append("  ")
                page_line.append("retry queue", style="yellow")
        else:
            page_line.append("Lesson complete", style="bold blue")

        progress_line = Text()
        if page is not None:
            progress_line.append(self._page_progress_text(snapshot), style="dim")
        if snapshot.header_note:
            if progress_line.plain.strip():
                progress_line.append("  ")
            progress_line.append(snapshot.header_note, style="yellow")
        if self._is_browsing(snapshot):
            if progress_line.plain.strip():
                progress_line.append("  ")
            progress_line.append("Browsing earlier material; answers still return to the live question.", style="bold yellow")

        question_lines = self._page_question_lines(snapshot)
        feedback_lines = self._feedback_lines(snapshot.feedback_lines)

        question_block: list[object] = []
        # D-16-21: Feynman opening takes priority on the first render in teach mode.
        _feynman_opening = getattr(self, "_feynman_opening", "")
        active_reply = _feynman_opening if _feynman_opening else turn.reply
        if active_reply.strip():
            question_block.append(Text("Question", style="bold bright_white"))
            question_block.extend(
                self._readable_text_blocks(active_reply, style="white", first_style="bold bright_white")
            )

        footer = Text()
        if snapshot.footer_commands:
            footer.append("Commands: ", style="bold white")
            for index, chunk in enumerate(snapshot.footer_commands):
                if index:
                    footer.append("  ", style="dim")
                footer.append(chunk, style="command")

        elements: list[object] = [header, page_line]
        if progress_line.plain.strip():
            elements.append(progress_line)
        if page is not None and page.intro_text.strip():
            elements.extend([Text(), Text("Page focus", style="bold bright_white")])
            elements.extend(self._readable_text_blocks(page.intro_text, style="white", first_style="bold white"))
        if question_lines:
            elements.extend([Text(), Text("Page status", style="bold white"), *question_lines])
        if feedback_lines:
            elements.extend([Text(), Text("Feedback", style="bold white"), *feedback_lines])
        if question_block:
            elements.extend([Text(), *question_block])
        if self.verbose and snapshot.run.active_question_slug:
            elements.extend(
                [
                    Text(),
                    Text(
                        f"Active: {snapshot.run.active_page_slug}/{snapshot.run.active_question_slug}",
                        style="dim",
                    ),
                ]
            )
        if footer.plain.strip():
            elements.extend([Text(), footer])

        self.console.print(
            Panel(
                Group(*elements),
                border_style="panel.border",
                padding=(1, 2),
                expand=True,
            )
        )

    def _readable_text_blocks(
        self,
        value: str,
        *,
        style: str = "white",
        first_style: str = "bold bright_white",
        max_lines_per_paragraph: int = 5,
        line_spacing: int = 1,
    ) -> list[object]:
        """Return Glow-formatted Rich blocks for learner-facing lesson prose."""

        raw = str(value or "")
        if not raw.strip():
            return []
        width = max(44, min(92, int(getattr(self.console, "width", 80) or 80) - 10))
        blocks = render_markdown_to_rich_blocks(raw, width=width)
        while blocks and not _rich_block_plain(blocks[-1]).strip():
            blocks.pop()
        while blocks and not _rich_block_plain(blocks[0]).strip():
            blocks.pop(0)
        return blocks

    def _render_turn(self, turn: LearningPartnerTurnDraft) -> None:
        """Compatibility wrapper for direct render calls."""
        self._render_session_frame(turn)

    def _page_progress_text(self, snapshot) -> str:
        page = snapshot.page
        if page is None:
            return "Lesson ready to finish."
        cleared = sum(1 for item in snapshot.page_questions if item.status in {"correct", "revealed", "skipped"})
        total = len(snapshot.page_questions)
        retry_count = len([item for item in snapshot.page_questions if item.retry_of_question_slug])
        if retry_count:
            return f"{cleared}/{total} cleared on this page | {retry_count} retry item(s)"
        return f"{cleared}/{total} cleared on this page"

    def _page_question_lines(self, snapshot) -> list[Text]:
        active_slug = snapshot.run.active_question_slug
        lines: list[Text] = []
        for index, question in enumerate(snapshot.page_questions, start=1):
            title = (
                str(question.prompt_json.get("title", "") or "").strip()
                or str(question.prompt_json.get("prompt", "") or "").strip().splitlines()[0]
                or question.skill_slug.replace("_", " ")
            )
            line = Text()
            line.append(f"{self._question_marker(question, active_slug)} ", style="dim")
            line.append(f"{index}. ", style="bold white")
            line.append(title, style="white" if question.question_slug == active_slug else "dim")
            if question.retry_of_question_slug:
                line.append("  ")
                line.append("retry", style="yellow")
            lines.append(line)
        return lines

    @staticmethod
    def _question_marker(question, active_slug: str) -> str:
        if question.question_slug == active_slug:
            return ">"
        if question.status == "correct":
            return "x"
        if question.status == "revealed":
            return "r"
        if question.status == "skipped":
            return "-"
        return "."

    def _feedback_lines(self, feedback: list[str]) -> list[Text]:
        rows: list[Text] = []
        for item in feedback[:4]:
            clean = renderable_cli_text(item).strip()
            if not clean:
                continue
            for line in clean.splitlines():
                line = line.strip()
                if not line:
                    continue
                style = "green" if line.startswith("Correct selections:") else "yellow"
                rows.append(Text(line, style=style))
        return rows

    def _render_question_input(self, turn: LearningPartnerTurnDraft) -> RoutedInput | None:
        """Collect structured answers inline when the turn calls for them."""
        question_type = getattr(turn, "question_type", "free_text")
        options = list(dict.fromkeys(str(option).strip() for option in getattr(turn, "mcq_options", []) if str(option).strip()))[:self.max_options]
        cloze_options = list(
            dict.fromkeys(
                str(option).strip()
                for option in getattr(turn, "cloze_blank_options", [])
                if str(option).strip()
            )
        )[:self.max_options]

        if (
            question_type == "mcq"
            and options
            and getattr(turn, "next_action", "") == LESSON_FINISH_MENU_ACTION
        ):
            selected = pick_single_choice(
                [(option, renderable_cli_text(option)) for option in options],
                title="Lesson cleared",
                text="Choose what happens next.",
                return_result=True,
                slash_registry=self.command_registry,
                pb_command_resolver=self.pb_command_resolver,
                allow_back_navigation=True,
            )
            if isinstance(selected, PickerResult):
                if selected.kind == "command" and isinstance(selected.value, RoutedInput):
                    return selected.value
                if selected.kind == "cancel":
                    return None
                selected = str(selected.value or "")
            if not selected:
                return None
            return self._route_finish_menu_selection(selected)

        if question_type == "mcq" and options:
            selected = pick_single_choice(
                [(option, renderable_cli_text(option)) for option in options],
                title="Choose one",
                text="Use arrows or digits, or type your own answer.",
                allow_inline_edit=True,
                inline_prompt="Type your own answer",
                return_result=True,
                slash_registry=self.command_registry,
                pb_command_resolver=self.pb_command_resolver,
                allow_back_navigation=True,
            )
            if isinstance(selected, PickerResult):
                if selected.kind == "command" and isinstance(selected.value, RoutedInput):
                    return selected.value
                if selected.kind in {"cancel"}:
                    return None
                if selected.kind == "inline_text":
                    typed = str(selected.value or "").strip()
                    if not typed:
                        return None
                    routed = self._classify_inline_text(typed)
                    if routed.kind != "answer":
                        return routed
                    return RoutedInput(kind="answer", text=typed)
                selected = str(selected.value or "")
            if not selected:
                return None
            return RoutedInput(kind="answer", text=selected)

        if question_type == "multi_select" and options:
            selected = pick_many_choices(
                [(option, renderable_cli_text(option)) for option in options],
                title="Select all that apply",
                text="Toggle with digits or arrows, then confirm.",
                allow_inline_edit=True,
                inline_prompt="Type your own answer",
                return_result=True,
                slash_registry=self.command_registry,
                pb_command_resolver=self.pb_command_resolver,
                allow_back_navigation=True,
            )
            if isinstance(selected, PickerResult):
                if selected.kind == "command" and isinstance(selected.value, RoutedInput):
                    return selected.value
                if selected.kind == "cancel":
                    return None
                selected = list(selected.value or [])
            cleaned = [str(item).strip() for item in selected if str(item).strip()]
            if cleaned:
                return RoutedInput(kind="answer", text=" | ".join(cleaned))
            return None

        if question_type == "cloze" and cloze_options:
            selected = pick_single_choice(
                [(option, renderable_cli_text(option)) for option in cloze_options],
                title="Fill in the blank",
                text="Choose the best fit for the blank.",
                allow_inline_edit=True,
                inline_prompt="Type your own answer",
                return_result=True,
                slash_registry=self.command_registry,
                pb_command_resolver=self.pb_command_resolver,
                allow_back_navigation=True,
            )
            if isinstance(selected, PickerResult):
                if selected.kind == "command" and isinstance(selected.value, RoutedInput):
                    return selected.value
                if selected.kind == "cancel":
                    return None
                if selected.kind == "inline_text":
                    typed = str(selected.value or "").strip()
                    if not typed:
                        return None
                    routed = self._classify_inline_text(typed)
                    if routed.kind != "answer":
                        return routed
                    return RoutedInput(kind="answer", text=typed)
                selected = str(selected.value or "")
            if not selected:
                return None
            return RoutedInput(kind="answer", text=selected)

        if question_type in {"short_text", "free_production", "error_correction", "reorder", "free_text"}:
            label = "Order> " if question_type == "reorder" else "Answer> "
            return prompt_answer_or_command(
                prompt_label=label,
                registry=self.command_registry,
                pb_command_resolver=self.pb_command_resolver,
                allow_navigation=True,
            )
        return None

    def _route_finish_menu_selection(self, selected: str) -> RoutedInput | None:
        clean = str(selected or "").strip()
        if clean == LESSON_FINISH_NEXT_SESSION:
            return self._finish_and_start_next_input()
        if clean == lesson_finish_review_label(self.mode):
            return RoutedInput(
                kind="pb_command",
                text="finish --debrief",
                argv=("finish", "--debrief"),
                command="finish",
            )
        if clean == LESSON_FINISH_CONTINUE_SESSION:
            return RoutedInput(kind="lesson_continue", text=clean, command="continue")
        return RoutedInput(kind="answer", text=clean)

    def _finish_and_start_next_input(self) -> RoutedInput | None:
        preference = self._pick_next_session_direction()
        if isinstance(preference, RoutedInput):
            return preference
        if preference is None:
            return None
        preference_text = str(preference or "").strip()
        note = "Next session preference"
        if preference_text:
            note = f"{note}: {preference_text}"
            self._store_next_session_preference(preference_text)
        return RoutedInput(
            kind="pb_command",
            text="finish --skip --yes; next --run",
            argv=("finish", "--skip", "--yes", note),
            command="finish",
            args="then:next --run",
        )

    def _pick_next_session_direction(self) -> str | RoutedInput | None:
        weak_focuses = self._challenging_focus_labels()
        weak_summary = self._summarize_focus_labels(weak_focuses)
        application_label = (
            "More conversational / translational"
            if self._looks_like_language_learning()
            else "More application-based / translational"
        )
        options = [
            ("application", application_label),
            ("weak_focus", "Focus on the weakest subtopics"),
            ("theoretical", "More theoretical / proof-first"),
            ("shift", "Shift to a connected topic"),
        ]
        details = [
            "Push the next session toward concrete transfer and use cases.",
            weak_summary or "Probe the concepts with the weakest evidence from this lesson.",
            "Ask for definitions, assumptions, proofs, mechanisms, and edge cases.",
            "Move laterally into a fresh but tightly connected direction.",
        ]
        selected = pick_single_choice(
            options,
            title="Next session direction",
            text="Choose the kind of challenge for the next session.",
            details=details,
            allow_inline_edit=True,
            inline_prompt="Type custom feedback",
            return_result=True,
            slash_registry=self.command_registry,
            pb_command_resolver=self.pb_command_resolver,
            allow_back_navigation=True,
        )
        if isinstance(selected, PickerResult):
            if selected.kind == "command" and isinstance(selected.value, RoutedInput):
                return selected.value
            if selected.kind == "cancel":
                return None
            if selected.kind == "inline_text":
                typed = str(selected.value or "").strip()
                return typed or None
            selected_value = str(selected.value or "").strip()
        else:
            selected_value = str(selected or "").strip()

        if selected_value == "application":
            return application_label
        if selected_value == "weak_focus":
            if len(weak_focuses) > 4:
                return self._pick_focus_cluster(weak_focuses)
            return weak_summary or "Focus on the weakest subtopics from this lesson."
        if selected_value == "theoretical":
            return "Make it more theoretical / proof-first."
        if selected_value == "shift":
            return "Shift to a fresh but tightly connected topic."
        return selected_value or None

    def _pick_focus_cluster(self, labels: list[str]) -> str | RoutedInput | None:
        clusters = self._cluster_focus_labels(labels)
        selected = pick_single_choice(
            [(cluster, cluster) for cluster in clusters],
            title="Focus cluster",
            text="Too many weak directions; choose the cluster for the next session.",
            allow_inline_edit=True,
            inline_prompt="Type a narrower focus",
            return_result=True,
            slash_registry=self.command_registry,
            pb_command_resolver=self.pb_command_resolver,
            allow_back_navigation=True,
        )
        if isinstance(selected, PickerResult):
            if selected.kind == "command" and isinstance(selected.value, RoutedInput):
                return selected.value
            if selected.kind == "cancel":
                return None
            if selected.kind == "inline_text":
                typed = str(selected.value or "").strip()
                return typed or None
            cluster = str(selected.value or "").strip()
        else:
            cluster = str(selected or "").strip()
        return f"Focus on {cluster}" if cluster else None

    def _challenging_focus_labels(self) -> list[str]:
        try:
            run = self.repo.get_lesson_run(self.session.id)
        except Exception:
            run = None
        if run is None:
            return []
        try:
            questions = list(self.repo.list_lesson_questions(run.id))
        except Exception:
            return []
        ranked: list[str] = []
        fallback: list[str] = []
        for question in questions:
            label = (
                str(question.answer_json.get("skill_label", "") or "").strip()
                or str(question.prompt_json.get("title", "") or "").strip()
                or question.skill_slug.replace("_", " ").strip()
            )
            label = renderable_cli_text(label).strip()
            if not label:
                continue
            fallback.append(label)
            if (
                question.status in {"revealed", "skipped", "wrong", "close"}
                or bool(question.retry_of_question_slug)
                or bool(question.queued_retry)
            ):
                ranked.append(label)
        return list(dict.fromkeys(ranked or fallback))

    @staticmethod
    def _summarize_focus_labels(labels: list[str]) -> str:
        clean = [label for label in dict.fromkeys(labels) if label]
        if not clean:
            return ""
        shown = clean[:4]
        suffix = f", +{len(clean) - len(shown)} more" if len(clean) > len(shown) else ""
        return "Focus on " + ", ".join(shown) + suffix + "."

    @staticmethod
    def _cluster_focus_labels(labels: list[str]) -> list[str]:
        clean = [label for label in dict.fromkeys(labels) if label]
        if len(clean) <= 4:
            return clean
        clusters: list[str] = []
        for index in range(0, len(clean), 4):
            chunk = clean[index:index + 4]
            clusters.append(", ".join(chunk))
        return clusters[:5]

    def _looks_like_language_learning(self) -> bool:
        text = f"{self.topic} {self.domain} {getattr(self.task, 'title', '')}".lower()
        markers = {
            "language",
            "conversation",
            "speaking",
            "listening",
            "vocabulary",
            "grammar",
            "spanish",
            "french",
            "german",
            "chinese",
            "mandarin",
            "japanese",
            "korean",
            "arabic",
            "italian",
            "portuguese",
        }
        return any(marker in text for marker in markers)

    def _store_next_session_preference(self, preference: str) -> None:
        generated = dict(getattr(self.session, "generated_names", {}) or {})
        generated["next_session_preference"] = preference
        self.session.generated_names = generated
        try:
            self.repo.update_session(self.session)
        except Exception:
            pass

    def _classify_inline_text(self, typed: str) -> RoutedInput:
        """Classify typed picker text before treating it as a lesson answer."""

        return classify_interactive_input(
            typed,
            pb_command_resolver=self.pb_command_resolver,
            slash_registry=self.command_registry,
            active_learning=True,
            allow_shell_commands=False,
            allow_nl_dispatch=False,
        )

    def _activity_context_for_current_question(self) -> dict[str, object] | None:
        """Capture the visible question state before the engine advances."""

        try:
            snapshot = self.engine.current_snapshot()
        except Exception:
            return None
        question = getattr(snapshot, "question", None)
        if question is None:
            return None
        page = getattr(snapshot, "page", None)
        return {
            "lesson_run_id": getattr(snapshot.run, "id", ""),
            "page_slug": getattr(question, "page_slug", ""),
            "page_title": getattr(page, "title", "") if page is not None else "",
            "question_slug": getattr(question, "question_slug", ""),
            "question_title": self._activity_question_title(question),
            "question_prompt": self._activity_question_prompt(question),
            "question_type": getattr(question, "question_type", ""),
            "skill_slug": getattr(question, "skill_slug", ""),
            "options": self._activity_question_options(question),
            "target_answers": self._activity_target_answers(question),
        }

    def _attempt_count_for_activity_context(self, activity_context: dict[str, object] | None) -> int:
        if not activity_context:
            return 0
        run_id = str(activity_context.get("lesson_run_id", "") or "")
        question_slug = str(activity_context.get("question_slug", "") or "")
        if not run_id or not question_slug:
            return 0
        try:
            return len(self.repo.list_lesson_attempts(run_id, question_slug))
        except Exception:
            return 0

    def _latest_activity_attempt(
        self,
        activity_context: dict[str, object] | None,
        previous_attempt_count: int,
    ):
        if not activity_context:
            return None
        run_id = str(activity_context.get("lesson_run_id", "") or "")
        question_slug = str(activity_context.get("question_slug", "") or "")
        if not run_id or not question_slug:
            return None
        try:
            attempts = self.repo.list_lesson_attempts(run_id, question_slug)
        except Exception:
            return None
        if len(attempts) <= previous_attempt_count:
            return attempts[-1] if attempts else None
        return attempts[-1]

    def _record_answer_activity(
        self,
        *,
        activity_context: dict[str, object] | None,
        user_input: str,
        previous_attempt_count: int,
    ) -> dict[str, object] | None:
        if not activity_context:
            return None
        attempt = self._latest_activity_attempt(activity_context, previous_attempt_count)
        result = str(getattr(attempt, "result", "") or "recorded")
        feedback = [renderable_cli_text(item).strip() for item in list(self.engine.last_feedback or []) if renderable_cli_text(item).strip()]
        record: dict[str, object] = {
            "kind": "answer_attempt",
            "session_id": getattr(self.session, "id", ""),
            "branch": self.branch,
            "page_slug": activity_context.get("page_slug", ""),
            "page_title": activity_context.get("page_title", ""),
            "question_slug": activity_context.get("question_slug", ""),
            "question_title": activity_context.get("question_title", ""),
            "question_prompt": activity_context.get("question_prompt", ""),
            "question_type": activity_context.get("question_type", ""),
            "skill_slug": activity_context.get("skill_slug", ""),
            "options": list(activity_context.get("options", []) or []),
            "selected_text": renderable_cli_text(user_input).strip(),
            "selected_options": self._selected_options_for_activity(activity_context, user_input),
            "target_answers": list(activity_context.get("target_answers", []) or []),
            "result": result,
            "is_correct": result == "correct",
            "correctness": self._correctness_label(result),
            "feedback": feedback,
            "points_delta": float(getattr(attempt, "points_delta", 0.0) or 0.0) if attempt is not None else 0.0,
            "response_ms": int(getattr(attempt, "response_ms", 0) or 0) if attempt is not None else 0,
            "attempt_id": str(getattr(attempt, "id", "") or "") if attempt is not None else "",
            "created_at": str(getattr(attempt, "created_at", "") or ""),
        }
        self._append_activity_record(record)
        return record

    def _record_command_activity(
        self,
        *,
        command: str,
        args: str = "",
        activity_context: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if not command:
            return None
        context = activity_context or {}
        feedback = [renderable_cli_text(item).strip() for item in list(self.engine.last_feedback or []) if renderable_cli_text(item).strip()]
        record: dict[str, object] = {
            "kind": "lesson_command",
            "session_id": getattr(self.session, "id", ""),
            "branch": self.branch,
            "command": command,
            "args": args,
            "page_slug": context.get("page_slug", ""),
            "page_title": context.get("page_title", ""),
            "question_slug": context.get("question_slug", ""),
            "question_title": context.get("question_title", ""),
            "question_prompt": context.get("question_prompt", ""),
            "question_type": context.get("question_type", ""),
            "options": list(context.get("options", []) or []),
            "feedback": feedback,
        }
        self._append_activity_record(record)
        return record

    def _append_activity_record(self, record: dict[str, object]) -> None:
        generated = append_learning_partner_activity(
            dict(getattr(self.session, "generated_names", {}) or {}),
            record,
        )
        self.session.generated_names = generated
        self.generated_names = generated
        self.repo.update_session(self.session)

    @staticmethod
    def _activity_question_title(question) -> str:
        title = str(question.prompt_json.get("title", "") or "").strip()
        if title:
            return renderable_cli_text(title)
        prompt = str(question.prompt_json.get("prompt", "") or "").strip()
        if prompt:
            return renderable_cli_text(prompt.splitlines()[0])
        return str(getattr(question, "skill_slug", "") or "Question").replace("_", " ")

    @staticmethod
    def _activity_question_prompt(question) -> str:
        prompt = str(question.prompt_json.get("prompt", "") or "").strip()
        return renderable_cli_text(prompt)

    @staticmethod
    def _activity_question_options(question) -> list[str]:
        if question.question_type in {"mcq", "multi_select", "cloze"}:
            raw_options = question.prompt_json.get("choices", [])
        elif question.question_type == "reorder":
            raw_options = question.prompt_json.get("display_items", []) or question.answer_json.get("ordered_items", [])
        else:
            raw_options = []
        if not isinstance(raw_options, list):
            return []
        return [renderable_cli_text(item).strip() for item in raw_options if renderable_cli_text(item).strip()]

    @staticmethod
    def _activity_target_answers(question) -> list[str]:
        targets: list[str] = []
        for key in ("correct_choices", "accepted_answers", "ordered_items"):
            raw = question.answer_json.get(key, [])
            if isinstance(raw, list):
                targets.extend(renderable_cli_text(item).strip() for item in raw if renderable_cli_text(item).strip())
        reveal = renderable_cli_text(question.answer_json.get("reveal_answer", "")).strip()
        if reveal:
            targets.append(reveal)
        deduped: list[str] = []
        seen: set[str] = set()
        for target in targets:
            key = target.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(target)
        return deduped

    @staticmethod
    def _correctness_label(result: str) -> str:
        if result == "correct":
            return "right"
        if result == "close":
            return "partly right"
        if result in {"wrong", "revealed"}:
            return "wrong"
        if result == "skipped":
            return "skipped"
        return result or "recorded"

    def _selected_options_for_activity(
        self,
        activity_context: dict[str, object],
        user_input: str,
    ) -> list[str]:
        text = renderable_cli_text(user_input).strip()
        question_type = str(activity_context.get("question_type", "") or "")
        options = [str(item).strip() for item in list(activity_context.get("options", []) or []) if str(item).strip()]
        if question_type == "reorder" and options:
            selected: list[str] = []
            for raw_index in re.findall(r"\d+", text):
                index = int(raw_index)
                if 1 <= index <= len(options):
                    selected.append(f"{index}. {options[index - 1]}")
            return selected or ([text] if text else [])
        if "|" in text:
            return [part.strip() for part in re.split(r"\s*\|\s*", text) if part.strip()]
        return [text] if text else []

    def _print_activity_receipt(self, activity: dict[str, object]) -> None:
        self.console = get_console()
        elements: list[object] = []
        if activity.get("kind") == "answer_attempt":
            result = str(activity.get("result", "") or "recorded")
            style = "green" if result == "correct" else ("yellow" if result == "close" else "red")
            header = Text("Recorded answer", style="bold white")
            header.append("  ")
            header.append(self._correctness_label(result), style=f"bold {style}")
            elements.append(header)
            question = renderable_cli_text(activity.get("question_title", "") or activity.get("question_prompt", "")).strip()
            if question:
                elements.append(Text(f"Question: {question}", style="white"))
            selected = list(activity.get("selected_options", []) or [])
            selected_text = " | ".join(str(item) for item in selected if str(item).strip())
            elements.append(Text(f"Selected: {selected_text or activity.get('selected_text', '')}", style="cyan"))
            options = [str(item).strip() for item in list(activity.get("options", []) or []) if str(item).strip()]
            if options:
                elements.append(Text("Options:", style="bold white"))
                selected_keys = {str(item).strip().lower() for item in selected}
                for index, option in enumerate(options, start=1):
                    marker = " <- selected" if option.lower() in selected_keys else ""
                    elements.append(Text(f"  {index}. {option}{marker}", style="dim" if not marker else "cyan"))
            feedback = [str(item).strip() for item in list(activity.get("feedback", []) or []) if str(item).strip()]
            if feedback:
                elements.append(Text(f"Feedback: {' | '.join(feedback[:3])}", style="yellow" if result != "correct" else "green"))
        elif activity.get("kind") == "lesson_command":
            command = str(activity.get("command", "") or "command")
            args = str(activity.get("args", "") or "").strip()
            elements.append(Text(f"Recorded command: {command} {args}".strip(), style="bold white"))
            feedback = [str(item).strip() for item in list(activity.get("feedback", []) or []) if str(item).strip()]
            if feedback:
                elements.append(Text(f"Response: {' | '.join(feedback[:3])}", style="yellow"))
        if not elements:
            return
        self.console.print(Panel(Group(*elements), border_style="panel.border", padding=(1, 2), expand=True))
        self._preserve_next_render = True

    def _record_exchange(
        self,
        user_text: str,
        turn: LearningPartnerTurnDraft,
        *,
        activity: dict[str, object] | None = None,
    ) -> None:
        self._capture_user_input_evidence(user_text)
        self.transcript.append({"role": "user", "content": user_text})
        if activity:
            self.transcript.append(
                {
                    "role": "assistant",
                    "content": format_learning_partner_activity_transcript_item(activity),
                }
            )
        self.transcript.append({"role": "assistant", "content": self._assistant_log_text(turn)})
        save_session_transcript(
            self.data_dir,
            getattr(self.session, "id", ""),
            self.transcript,
            self.session_slug,
        )
        self._sync_session_metadata()

    def _append_assistant_turn(self, reply: str) -> None:
        self.transcript.append({"role": "assistant", "content": reply})
        save_session_transcript(
            self.data_dir,
            getattr(self.session, "id", ""),
            self.transcript,
            self.session_slug,
        )

    @staticmethod
    def _assistant_log_text(turn: LearningPartnerTurnDraft) -> str:
        parts = [renderable_cli_text(turn.reply).strip()]
        parts.extend(renderable_cli_text(item).strip() for item in turn.corrections[:3] if renderable_cli_text(item).strip())
        return "\n".join(part for part in parts if part)

    def _load_evidence_log(self) -> list[dict[str, str]]:
        stored = self.generated_names.get("learning_partner_evidence")
        if not isinstance(stored, list):
            return []
        evidence: list[dict[str, str]] = []
        for item in stored:
            parsed = self._coerce_evidence_item(item)
            if parsed is not None:
                evidence.append(parsed)
        return evidence

    @staticmethod
    def _coerce_evidence_item(item: object) -> dict[str, str] | None:
        if isinstance(item, dict):
            note = str(item.get("note", "") or item.get("evidence", "") or "").strip()
            subskill = str(item.get("subskill", "")).strip()
            source = str(item.get("source", "")).strip() or "partner"
            if note:
                return {"subskill": subskill, "note": note, "source": source}
            return None
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return {"subskill": "", "note": text, "source": "partner"}
        return None

    def _append_evidence_item(self, item: dict[str, str] | None) -> None:
        if item is None:
            return
        key = (
            item.get("subskill", "").strip().lower(),
            item.get("note", "").strip().lower(),
            item.get("source", "").strip().lower(),
        )
        for existing in self.evidence_log:
            existing_key = (
                existing.get("subskill", "").strip().lower(),
                existing.get("note", "").strip().lower(),
                existing.get("source", "").strip().lower(),
            )
            if existing_key == key:
                return
        self.evidence_log.append(item)

    def _capture_user_input_evidence(self, user_text: str) -> None:
        clean = renderable_cli_text(user_text).strip()
        if not clean or clean.startswith("/"):
            return
        self._append_evidence_item(
            {
                "subskill": "",
                "note": clean[:240],
                "source": "learner_input",
            }
        )

    def _sync_session_metadata(self) -> None:
        snapshot = self.engine.current_snapshot()
        diagnostics = self.engine.skill_diagnostics()
        self.collected_recall = self.engine.recall_candidates()
        self.next_drill = self.engine.next_drill()
        self.collected_gaps = [
            state.skill_slug.replace("_", " ")
            for state in diagnostics
            if state.overall_status != "strong"
        ]
        self.collected_corrections = list(snapshot.feedback_lines)

        generated = dict(getattr(self.session, "generated_names", {}) or {})
        generated["learning_partner_used"] = True
        generated["learning_partner_evidence"] = list(self.evidence_log)
        generated["learning_partner_progress"] = {
            "page_slug": snapshot.run.active_page_slug,
            "question_slug": snapshot.run.active_question_slug,
            "lesson_status": snapshot.run.lesson_status,
            "points": snapshot.run.total_points,
            "ready_to_finish": snapshot.run.ready_to_finish,
        }
        self.session.generated_names = generated
        self.generated_names = generated
        self.repo.update_session(self.session)

    def _mark_partner_session_used(self) -> None:
        generated = dict(getattr(self.session, "generated_names", {}) or {})
        if generated.get("learning_partner_used"):
            return
        generated["learning_partner_used"] = True
        self.session.generated_names = generated
        self.generated_names = generated
        self.repo.update_session(self.session)

    def _default_pb_command_resolver(self) -> PbCommandResolver | None:
        try:
            import typer.main
            from pb.cli.main import app as pb_app

            return PbCommandResolver(typer.main.get_command(pb_app))
        except Exception:
            return None

    def _result_from_command(self, command: str) -> PartnerRunResult:
        self._store_partner_closeout(command=command)
        return PartnerRunResult(
            action="command",
            command=command,
            recall_candidates=list(self.collected_recall),
            detected_gaps=list(self.collected_gaps),
            next_drill=self.next_drill,
        )

    def _finalize(self, action: str, summary: str) -> PartnerRunResult:
        self._store_partner_closeout(action=action, summary=summary)
        return PartnerRunResult(
            action=action,
            summary=summary,
            recall_candidates=list(self.collected_recall),
            detected_gaps=list(self.collected_gaps),
            next_drill=self.next_drill,
        )

    def _store_partner_closeout(
        self,
        *,
        action: str = "",
        command: str = "",
        summary: str = "",
    ) -> None:
        generated = dict(getattr(self.session, "generated_names", {}) or {})
        activity_summary = summarize_learning_partner_activity(generated)
        generated["learning_partner_closeout"] = {
            "action": action,
            "command": command,
            "summary": summary,
            "recall_candidates": list(self.collected_recall),
            "detected_gaps": list(self.collected_gaps),
            "corrections": list(self.collected_corrections),
            "next_drill": self.next_drill,
            "activity_summary": activity_summary,
        }
        self.session.generated_names = generated
        self.generated_names = generated
        self.repo.update_session(self.session)

    def _display_snapshot(self):
        return self.engine.snapshot_for(page_slug=self.view_page_slug, question_slug=self.view_question_slug)

    def _display_turn(self) -> LearningPartnerTurnDraft:
        return self.engine.turn_for(page_slug=self.view_page_slug, question_slug=self.view_question_slug)

    def _reset_view(self) -> None:
        self.view_page_slug = ""
        self.view_question_slug = ""
        self._view_history.clear()

    def _is_browsing(self, snapshot) -> bool:
        return bool(
            (self.view_page_slug and self.view_page_slug != snapshot.run.active_page_slug)
            or (self.view_question_slug and self.view_question_slug != snapshot.run.active_question_slug)
        )

    def _browse(self, directions: tuple[str, ...] | list[str]) -> None:
        for direction in directions:
            self._apply_navigation(str(direction or "").strip().lower())

    def _set_view(self, page_slug: str, question_slug: str) -> None:
        target = (page_slug, question_slug)
        current = (self.view_page_slug, self.view_question_slug)
        if target == current:
            return
        self._view_history.append(current)
        self._view_history = self._view_history[-20:]
        self.view_page_slug, self.view_question_slug = target

    def _apply_navigation(self, direction: str) -> None:
        if direction in {"back", "parent", "tab"}:
            if self._view_history:
                self.view_page_slug, self.view_question_slug = self._view_history.pop()
            else:
                self.view_page_slug = ""
                self.view_question_slug = ""
            return

        snapshot = self._display_snapshot()
        pages = list(snapshot.pages)
        if not pages:
            return
        current_page_slug = snapshot.page.page_slug if snapshot.page is not None else snapshot.run.active_page_slug
        current_question_slug = snapshot.question.question_slug if snapshot.question is not None else snapshot.run.active_question_slug
        page_index = next((index for index, item in enumerate(pages) if item.page_slug == current_page_slug), 0)
        page_questions = list(snapshot.page_questions)
        question_index = next((index for index, item in enumerate(page_questions) if item.question_slug == current_question_slug), 0)

        if direction == "left":
            target_page = pages[(page_index - 1) % len(pages)]
            target_questions = self.repo.list_lesson_questions(snapshot.run.id, target_page.page_slug)
            self._set_view(target_page.page_slug, target_questions[0].question_slug if target_questions else "")
            return
        if direction == "right":
            target_page = pages[(page_index + 1) % len(pages)]
            target_questions = self.repo.list_lesson_questions(snapshot.run.id, target_page.page_slug)
            self._set_view(target_page.page_slug, target_questions[0].question_slug if target_questions else "")
            return
        if direction in {"up", "down"} and page_questions:
            delta = -1 if direction == "up" else 1
            target_question = page_questions[(question_index + delta) % len(page_questions)]
            self._set_view(target_question.page_slug, target_question.question_slug)
