# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Interactive lesson adapter over the unified lesson engine."""

from __future__ import annotations

import json
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
from pb.cli.pickers import PickerResult, pick_many_choices, pick_single_choice
from pb.core.lesson_engine import LessonEngine
from pb.core.registry import CommandHandler, CommandRegistry
from pb.core.renderables import renderable_cli_text
from pb.core.naming import stored_display_title
from pb.llm.drafts import LearningPartnerTurnDraft


def _transcript_path(data_dir: Path, session_id: str, session_slug: str = "") -> Path:
    """Return the durable transcript path for one learning session."""
    filename = session_slug or session_id
    return data_dir / "transcripts" / f"{filename}.json"


def _format_points(value: float) -> str:
    return f"{float(value):g}"


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
        turn = self.engine.answer_current(user_input)
        self.current_turn = turn
        self._record_exchange(user_input, turn)
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
            self.console.print("[warn]Use `finish`, `pause`, or `resume` without a slash.[/]")
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

    def run_contextual_command(self, command: str, args: str = "") -> LearningPartnerTurnDraft | None:
        """Execute one contextual slash command over the current lesson state."""
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
        else:
            return None
        return self._record_contextual_turn(turn)

    def _record_contextual_turn(self, turn: LearningPartnerTurnDraft) -> LearningPartnerTurnDraft:
        self.current_turn = turn
        self._append_assistant_turn(self._assistant_log_text(turn))
        self._sync_session_metadata()
        return turn

    def _render_session_frame(self, turn: LearningPartnerTurnDraft) -> None:
        """Repaint the current page-oriented lesson frame."""
        self.console = get_console()
        if sys.stdin.isatty():
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
            question_block.append(Text("Question", style="bold white"))
            question_block.append(Text(renderable_cli_text(active_reply).strip(), style="white"))

        footer = Text()
        footer.append("Commands: ", style="bold white")
        for index, chunk in enumerate(snapshot.footer_commands):
            if index:
                footer.append("  ", style="dim")
            footer.append(chunk, style="command")

        elements: list[object] = [header, page_line]
        if progress_line.plain.strip():
            elements.append(progress_line)
        if page is not None and page.intro_text.strip():
            elements.extend([Text(), Text(page.intro_text.strip(), style="dim")])
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
        elements.extend([Text(), footer])

        self.console.print(
            Panel(
                Group(*elements),
                border_style="panel.border",
                padding=(1, 2),
                expand=True,
            )
        )

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

        if question_type == "mcq" and options:
            selected = pick_single_choice(
                [(option, option) for option in options],
                title="Choose one",
                text="Use arrows or digits, or type your own answer.",
                allow_inline_edit=True,
                inline_prompt="Type your own answer",
                return_result=True,
                slash_registry=self.command_registry,
                pb_command_resolver=self.pb_command_resolver,
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
                    if typed.startswith("/"):
                        return RoutedInput(kind="slash_command", command=typed)
                    return RoutedInput(kind="answer", text=typed)
                selected = str(selected.value or "")
            if not selected:
                return None
            return RoutedInput(kind="answer", text=selected)

        if question_type == "multi_select" and options:
            selected = pick_many_choices(
                [(option, option) for option in options],
                title="Select all that apply",
                text="Toggle with digits or arrows, then confirm.",
                allow_inline_edit=True,
                inline_prompt="Type your own answer",
                return_result=True,
                slash_registry=self.command_registry,
                pb_command_resolver=self.pb_command_resolver,
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
                [(option, option) for option in cloze_options],
                title="Fill in the blank",
                text="Choose the best fit for the blank.",
                allow_inline_edit=True,
                inline_prompt="Type your own answer",
                return_result=True,
                slash_registry=self.command_registry,
                pb_command_resolver=self.pb_command_resolver,
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
                    if typed.startswith("/"):
                        return RoutedInput(kind="slash_command", command=typed)
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

    def _record_exchange(self, user_text: str, turn: LearningPartnerTurnDraft) -> None:
        self._capture_user_input_evidence(user_text)
        self.transcript.append({"role": "user", "content": user_text})
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
        return PartnerRunResult(
            action="command",
            command=command,
            recall_candidates=list(self.collected_recall),
            detected_gaps=list(self.collected_gaps),
            next_drill=self.next_drill,
        )

    def _finalize(self, action: str, summary: str) -> PartnerRunResult:
        return PartnerRunResult(
            action=action,
            summary=summary,
            recall_candidates=list(self.collected_recall),
            detected_gaps=list(self.collected_gaps),
            next_drill=self.next_drill,
        )

    def _display_snapshot(self):
        return self.engine.snapshot_for(page_slug=self.view_page_slug, question_slug=self.view_question_slug)

    def _display_turn(self) -> LearningPartnerTurnDraft:
        return self.engine.turn_for(page_slug=self.view_page_slug, question_slug=self.view_question_slug)

    def _reset_view(self) -> None:
        self.view_page_slug = ""
        self.view_question_slug = ""

    def _is_browsing(self, snapshot) -> bool:
        return bool(
            (self.view_page_slug and self.view_page_slug != snapshot.run.active_page_slug)
            or (self.view_question_slug and self.view_question_slug != snapshot.run.active_question_slug)
        )

    def _browse(self, directions: tuple[str, ...] | list[str]) -> None:
        for direction in directions:
            self._apply_navigation(str(direction or "").strip().lower())

    def _apply_navigation(self, direction: str) -> None:
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
            self.view_page_slug = target_page.page_slug
            self.view_question_slug = target_questions[0].question_slug if target_questions else ""
            return
        if direction == "right":
            target_page = pages[(page_index + 1) % len(pages)]
            target_questions = self.repo.list_lesson_questions(snapshot.run.id, target_page.page_slug)
            self.view_page_slug = target_page.page_slug
            self.view_question_slug = target_questions[0].question_slug if target_questions else ""
            return
        if direction in {"up", "down"} and page_questions:
            delta = -1 if direction == "up" else 1
            target_question = page_questions[(question_index + delta) % len(page_questions)]
            self.view_page_slug = target_question.page_slug
            self.view_question_slug = target_question.question_slug
