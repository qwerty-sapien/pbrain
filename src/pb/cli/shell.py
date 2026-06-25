# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Interactive shell for pb with prompt_toolkit REPL, vault navigation, and tab completion."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Callable, Optional

import structlog
from rich.markup import escape

# Import prompt_toolkit at module level so tests can patch these names.
# prompt_toolkit is a declared dependency; if unavailable, shell degrades to input() loop.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    _PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PromptSession = None        # type: ignore[assignment]
    Completer = object          # type: ignore[assignment]
    Completion = None           # type: ignore[assignment]
    FileHistory = None          # type: ignore[assignment]
    _PROMPT_TOOLKIT_AVAILABLE = False

from pb.cli.console import get_console, get_err_console
from pb.cli.input_router import (
    EXIT_TOKENS,
    SHELL_COMMANDS,
    PbCommandResolver,
    RoutedInput,
    classify_interactive_input,
    is_natural_language_input as _router_is_natural_language_input,
)
from pb.cli.pickers import pick_single_choice
from pb.core.action_routing import suggest_commands_for_intent
from pb.core.clock import utc_now
from pb.core.error_logging import format_logged_exception, log_error
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.learning_partner import LearningPartnerSession
from pb.core.naming import stored_short_title
from pb.vault.indexer import (
    is_folder_index_stale,
    rebuild_folder_index,
    generate_directory_md,
    search_folder_index,
)
from pb.core.suggestions import MkMvEngine, tier2_confirm
from pb.llm.gemini import get_client
from pb.llm.runtime import LLMRuntime
from pb.vault.graph import update_note_in_graph

_logger = structlog.get_logger()

VAULT_COMMANDS = list(SHELL_COMMANDS[:4])


def _safe_get_active_session(repo):
    """Return the active session without letting repo errors break the shell."""
    try:
        return repo.get_active_session()
    except Exception:
        return None

def _is_natural_language_input(args: list[str], raw_input: str | None) -> bool:
    """Backward-compatible wrapper for the shared NL dispatch heuristic."""
    return _router_is_natural_language_input(args, raw_input)


def _coaching_turn(repo, runtime: LLMRuntime, runtime_ctx, active_session, user_input: str) -> RoutedInput | None:
    """Generate one coaching turn for the current active session."""
    partner = _learning_partner_for_session(repo, runtime, runtime_ctx, active_session)
    if partner is None:
        return None
    turn = partner.respond_once(user_input)
    return _render_partner_turn_chain(partner, turn)


def _learning_partner_for_session(repo, runtime: LLMRuntime, runtime_ctx, active_session) -> LearningPartnerSession | None:
    """Build a blueprint-aware partner for one active learning session."""
    task = repo.get_task(active_session.task_id)
    if task is None:
        return None
    metadata = parse_learning_task_metadata(task)
    branch = getattr(active_session, "branch", "") or metadata.branch or "study"
    if branch not in {"study", "teach", "practise", "practice"}:
        return None
    topic = getattr(active_session, "subject_scope", "") or metadata.scope or task.title
    objective = getattr(active_session, "intended_outcome", "") or metadata.success_check or task.title
    mode = metadata.study_mode or metadata.practice_stage or branch
    return LearningPartnerSession(
        runtime=runtime,
        runtime_ctx=runtime_ctx,
        repo=repo,
        task=task,
        session=active_session,
        branch=branch,
        objective=objective,
        topic=topic,
        domain=metadata.domain or topic,
        mode=mode,
    )


def _contextual_slash_commands(repo, runtime: LLMRuntime, runtime_ctx, active_session) -> list[str]:
    """Return the active contextual slash commands for the current learning session."""
    partner = _learning_partner_for_session(repo, runtime, runtime_ctx, active_session)
    if partner is None:
        return []
    return partner.contextual_command_names()


def _render_partner_turn_chain(partner: LearningPartnerSession, turn) -> RoutedInput | None:
    """Render one partner turn and immediately ingest any structured picker answers."""
    current_turn = turn
    while True:
        partner.set_current_turn(current_turn)
        partner._render_session_frame(current_turn)
        picker_input = partner._render_question_input(current_turn)
        if picker_input is None:
            return
        if isinstance(picker_input, str):
            picker_input = RoutedInput(kind="answer", text=picker_input)
        if picker_input.kind == "navigation":
            partner._browse(picker_input.argv or (picker_input.command,))
            continue
        if picker_input.kind == "answer":
            current_turn = partner.respond_once(picker_input.text)
            continue
        if picker_input.kind == "slash_command":
            next_turn = partner.run_contextual_command(picker_input.command)
            if next_turn is not None:
                current_turn = next_turn
            continue
        if picker_input.kind in {"slash_ambiguous", "slash_unknown"}:
            partner.explain_contextual_command_error(picker_input)
            continue
        if picker_input.kind in {"pb_command", "shell_command"}:
            return picker_input


def _maybe_open_learning_session(repo, runtime: LLMRuntime, runtime_ctx, active_session) -> None:
    """Start a fresh learning session with one useful coaching move in shell mode."""
    partner = _learning_partner_for_session(repo, runtime, runtime_ctx, active_session)
    if partner is None or partner.transcript:
        return
    opening = partner.open_with_first_move()
    _render_partner_turn_chain(partner, opening)


def get_history_path() -> Path:
    """Return XDG-compliant path for persistent shell history (D-24)."""
    xdg_data = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    path = Path(xdg_data) / "pb" / "shell_history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_prompt_callable(
    vault_root: Path,
    get_vault_cwd: Callable[[], Path],
    repo,
) -> Callable[[], str]:
    """Return a callable that builds the prompt string on each keystroke (D-05, D-06, D-07).

    DB-backed fields (session, dispatch label) are cached with a short TTL so
    prompt_toolkit re-renders don't cause a DB round-trip per keystroke.
    """
    import time as _time

    _cache: dict = {
        "ts": 0.0,
        "task_name": "",
        "timer_label": "",
        "dispatch_label": "",
        "context_label": "",
    }
    _CACHE_TTL = 2.0  # seconds

    def _refresh_cache() -> None:
        now = _time.monotonic()
        if now - _cache["ts"] < _CACHE_TTL:
            return
        _cache["ts"] = now

        task_name = ""
        timer_label = ""
        context_label = ""
        try:
            session = repo.get_active_session()
            if session:
                from pb.cli.context_runtime import session_active_context_scope
                from pb.core.context_file_intake import summarize_context_label

                task = repo.get_task(session.task_id)
                task_name = stored_short_title(task) if task else ""
                scope = session_active_context_scope(session)
                context_label = summarize_context_label(scope)
                elapsed_minutes = max(
                    0,
                    int((utc_now() - session.start_at).total_seconds() / 60),
                ) if getattr(session, "start_at", None) else 0
                duration_minutes = getattr(session, "duration_minutes", None)
                if isinstance(duration_minutes, int) and duration_minutes > 0:
                    remaining = duration_minutes - elapsed_minutes
                    timer_label = f"{remaining}m left" if remaining >= 0 else f"+{abs(remaining)}m"
                else:
                    timer_label = f"{elapsed_minutes}m"
            else:
                from pb.core.context_file_intake import summarize_context_label

                locked = repo.get_locked_context()
                context_label = summarize_context_label(locked)
        except Exception:
            pass
        _cache["task_name"] = task_name
        _cache["timer_label"] = timer_label
        _cache["context_label"] = context_label

        dispatch_label = ""
        try:
            from pb.storage.database import get_connection
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT agent_id FROM dispatch_sessions WHERE status='active' ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
            if row:
                raw_label = row["agent_id"]
                try:
                    from pb.agents import resolve_agent
                    if resolve_agent(raw_label) is not None:
                        dispatch_label = raw_label.replace("domain_", "").replace("_agent", "").replace("_", "-")
                except Exception:
                    pass
        except Exception:
            pass
        _cache["dispatch_label"] = dispatch_label

    def get_prompt() -> str:
        _refresh_cache()
        task_name = _cache["task_name"]
        timer_label = _cache["timer_label"]
        dispatch_label = _cache["dispatch_label"]
        context_label = _cache["context_label"]

        vault_cwd = get_vault_cwd()
        try:
            rel = vault_cwd.relative_to(vault_root)
            rel_str = str(rel) if str(rel) != "." else ""
        except ValueError:
            rel_str = ""

        session_label = task_name
        if task_name and context_label and timer_label:
            session_label = f"{task_name} | {context_label} | {timer_label}"
        elif task_name and timer_label:
            session_label = f"{task_name} | {timer_label}"
        elif task_name and context_label:
            session_label = f"{task_name} | {context_label}"

        if session_label and rel_str:
            return f"pb [{session_label}] {rel_str}> "
        elif session_label:
            return f"pb [{session_label}]> "
        elif context_label:
            if rel_str:
                return f"pb[{context_label}] {rel_str}> "
            return f"pb[{context_label}]> "
        elif not session_label and dispatch_label:
            if rel_str:
                return f"pb[{dispatch_label}] {rel_str}> "
            return f"pb[{dispatch_label}]> "
        elif rel_str:
            return f"pb {rel_str}> "
        else:
            return "pb> "

    return get_prompt


class VaultCompleter(Completer):
    """Context-aware completer for vault paths and pb subcommands (D-18 to D-22)."""

    def __init__(
        self,
        vault_root: Path,
        get_vault_cwd: Callable[[], Path],
        pb_commands: list[str],
        get_contextual_commands: Callable[[], list[str]] | None = None,
    ):
        self.vault_root = vault_root
        self.get_vault_cwd = get_vault_cwd   # callable — avoids stale closure (Pitfall 1)
        self.get_contextual_commands = get_contextual_commands or (lambda: [])
        self.all_commands = VAULT_COMMANDS + list(EXIT_TOKENS) + pb_commands + ["?", "mkmv", "deactivate"]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()

        # Position 0: complete command names (D-19)
        if len(words) == 0 or (len(words) == 1 and not text.endswith(" ")):
            prefix = words[0] if words else ""
            commands = list(dict.fromkeys(self.all_commands + self.get_contextual_commands()))
            for cmd in commands:
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix))
            return

        first = words[0].lower()
        if first == "cd":
            partial = "" if text.endswith(" ") else (words[-1] if len(words) > 1 else "")
            yield from self._complete_paths(partial, folders_only=True)   # D-20
        elif first in ("cat", "grep"):
            partial = "" if text.endswith(" ") else (words[-1] if len(words) > 1 else "")
            yield from self._complete_paths(partial, folders_only=False)  # D-21

    def _complete_paths(self, partial: str, folders_only: bool):
        vault_cwd = self.get_vault_cwd()  # D-22: always fresh cwd
        base = vault_cwd / partial if partial else vault_cwd
        if not base.is_dir():
            base = base.parent
            prefix = Path(partial).name if partial else ""
        else:
            prefix = ""
        try:
            for entry in sorted(base.iterdir()):
                if entry.name.startswith("."):
                    continue
                if folders_only and not entry.is_dir():
                    continue
                name = entry.name + ("/" if entry.is_dir() else "")
                if name.startswith(prefix):
                    yield Completion(name, start_position=-len(prefix))
        except PermissionError:
            pass


def _maybe_refresh_folder_index(new_cwd: Path, vault_root: Path) -> None:
    """Trigger lazy index rebuild if stale. Silent on cache hit (D-09)."""
    try:
        if is_folder_index_stale(new_cwd):
            count = rebuild_folder_index(new_cwd, vault_root)
            # Regenerate directory summary (D-05, D-07)
            summary = generate_directory_md(new_cwd)
            (new_cwd / ".pb-directory.md").write_text(summary)
            try:
                rel = str(new_cwd.relative_to(vault_root))
            except ValueError:
                rel = new_cwd.name
            console = get_console()
            console.print(f"[dim]indexed {escape(rel)} ({count} notes)[/]")
    except Exception as e:
        _logger.debug("vault.index_refresh_failed", folder=str(new_cwd), error=str(e))


def cmd_ls(vault_cwd: Path) -> None:
    """D-11, D-12: List contents with type indicators."""
    console = get_console()
    try:
        entries = sorted(vault_cwd.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                console.print(f"  [header]{escape(entry.name)}/[/]")
            else:
                console.print(f"  {escape(entry.name)}")
    except PermissionError as e:
        get_err_console().print(f"[error]ls: {escape(str(e))}[/]")


def _list_dirs(directory: Path) -> list[Path]:
    """Return visible subdirectories sorted alphabetically."""
    try:
        return sorted(
            [e for e in directory.iterdir() if e.is_dir() and not e.name.startswith(".")],
            key=lambda p: p.name.lower(),
        )
    except PermissionError:
        return []


def _unique_note_path(base_path: Path) -> Path:
    """Return a collision-safe note path by appending numeric suffixes."""
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    counter = 2
    while True:
        candidate = base_path.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _pick_numbered(dirs: list[Path], vault_root: Path) -> Optional[Path]:
    """Show numbered directory list; return selected Path or None."""
    console = get_console()
    for i, d in enumerate(dirs, 1):
        try:
            rel = d.relative_to(vault_root)
        except ValueError:
            rel = d
        console.print(f"  [dim]{i}.[/] [header]{escape(str(rel))}/[/]")
    try:
        raw = input("select [1-{}]: ".format(len(dirs)))
    except (EOFError, KeyboardInterrupt):
        return None
    try:
        idx = int(raw.strip()) - 1
        if 0 <= idx < len(dirs):
            return dirs[idx]
    except ValueError:
        pass
    get_err_console().print("[error]cd: invalid selection[/]")
    return None


def _find_matching_dirs(name: str, vault_root: Path) -> list[Path]:
    """Find all directories under vault_root whose name matches (case-insensitive)."""
    name_lower = name.lower()
    matches = []
    for dirpath in vault_root.rglob("*/"):
        if not dirpath.is_dir():
            continue
        if any(part.startswith(".") for part in dirpath.relative_to(vault_root).parts):
            continue
        if dirpath.name.lower() == name_lower or dirpath.name.lower().startswith(name_lower):
            matches.append(dirpath)
    return sorted(matches, key=lambda p: (len(p.parts), p.name.lower()))


def cmd_cd(args: list[str], vault_cwd: Path, vault_root: Path) -> Path:
    """Navigate vault directories with numbered picker and fuzzy matching."""
    err_console = get_err_console()
    if not args:
        dirs = _list_dirs(vault_cwd)
        if not dirs:
            err_console.print("[error]cd: no subfolders here[/]")
            return vault_cwd
        picked = _pick_numbered(dirs, vault_root)
        return picked if picked else vault_cwd

    arg = args[0]

    if arg == "..":
        parent = vault_cwd.parent.resolve()
        try:
            parent.relative_to(vault_root)
            return parent
        except ValueError:
            return vault_root

    if arg in ("/", "~"):
        return vault_root

    direct = (vault_cwd / arg).resolve()
    try:
        direct.relative_to(vault_root)
        if direct.is_dir():
            return direct
    except ValueError:
        pass

    matches = _find_matching_dirs(arg, vault_root)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console = get_console()
        console.print(f"cd: multiple matches for '{escape(arg)}':")
        picked = _pick_numbered(matches, vault_root)
        return picked if picked else vault_cwd
    else:
        err_console.print(f"[error]cd: no such folder: {escape(arg)}[/]")
        return vault_cwd


def _grep_content(pattern: re.Pattern, vault_cwd: Path) -> None:
    """Walk vault_cwd for *.md files and print line matches grouped by file."""
    console = get_console()
    first_group = True
    for filepath in sorted(vault_cwd.rglob("*.md")):
        # Skip hidden files and files in hidden directories
        if any(part.startswith(".") for part in filepath.parts):
            continue
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append((lineno, line.rstrip()))
        if hits:
            if not first_group:
                console.print("")
            rel = filepath.relative_to(vault_cwd)
            console.print(escape(str(rel)))
            for lineno, text_line in hits:
                console.print(f"  [dim]{lineno}:[/] {escape(text_line)}")
            first_group = False


def _grep_filenames(pattern: re.Pattern, vault_cwd: Path) -> None:
    """Walk vault_cwd and print filenames matching pattern, sorted by relative path."""
    console = get_console()
    matches = []
    for filepath in vault_cwd.rglob("*"):
        # Skip hidden files and files in hidden directories
        if any(part.startswith(".") for part in filepath.parts):
            continue
        if filepath.is_file() and pattern.search(filepath.name):
            matches.append(filepath.relative_to(vault_cwd))
    for rel in sorted(matches):
        console.print(f"  [success]{escape(str(rel))}[/]")


def cmd_grep(args: list[str], vault_cwd: Path) -> None:
    """D-13/D-14/D-15: Filename or content search."""
    err_console = get_err_console()
    if not args:
        err_console.print("[error]grep: missing pattern[/]")
        return
    content_mode = args[0] == "-n"
    if content_mode:
        if len(args) < 2:
            err_console.print("[error]grep: missing pattern after -n[/]")
            return
        pattern_str = args[1]
    else:
        pattern_str = args[0]
    try:
        pattern = re.compile(pattern_str, re.IGNORECASE)
    except re.error as e:
        err_console.print(f"[error]grep: invalid pattern: {escape(str(e))}[/]")
        return
    if content_mode:
        db_path = vault_cwd / ".pb-index.db"
        if db_path.exists():
            results = search_folder_index(vault_cwd, pattern_str)
            if results is not None:  # None = FTS5 error, fall back
                console = get_console()
                for path, snippet in results:
                    console.print(f"[dim]{escape(str(path))}:[/] {escape(snippet)}")
                return
        _grep_content(pattern, vault_cwd)  # fallback per D-12
    else:
        _grep_filenames(pattern, vault_cwd)


def cmd_cat(args: list[str], vault_cwd: Path, vault_root: Path) -> None:
    """D-16, D-17: Print note contents. Includes boundary check (T-10-02 mitigation)."""
    err_console = get_err_console()
    if not args:
        err_console.print("[error]cat: missing filename[/]")
        return
    target = (vault_cwd / args[0]).resolve()
    try:
        target.relative_to(vault_root)
    except ValueError:
        err_console.print(f"[error]cat: access denied: {escape(args[0])}[/]")
        return
    if not target.is_file():
        err_console.print(f"[error]cat: no such file: {escape(args[0])}[/]")
        return
    try:
        console = get_console()
        console.print(target.read_text(encoding="utf-8", errors="replace"), highlight=False)
    except OSError as e:
        err_console.print(f"[error]cat: {escape(str(e))}[/]")


def run_shell(click_app, vault_root: Path, repo, on_cd: Callable[[], None] | None = None, runtime_ctx=None) -> None:
    """Prompt_toolkit REPL replacing _interactive_shell() (D-08 through D-27).

    If prompt_toolkit is unavailable, falls back to the original input()-based loop
    with vault commands added (graceful degradation).

    on_cd: optional callback invoked after each cd command (e.g. to invalidate chat prefix).
    """
    vault_cwd = vault_root                        # D-08: reset to root on launch
    _cwd_ref = [vault_cwd]                        # mutable container for closure sharing
    runtime = LLMRuntime(runtime_ctx.config) if runtime_ctx is not None else None
    pb_command_resolver = PbCommandResolver(click_app)
    os.environ["PB_IN_SHELL"] = "1"

    def get_vault_cwd() -> Path:
        return _cwd_ref[0]

    console = get_console()
    shell_test_mode = os.environ.get("PRODUCTIVEBRAIN_SHELL_TEST_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # D-27: habit insights on launch — identical to existing main.py lines 194-206
    try:
        from pb.core.insights import InsightEngine
        from pb.storage.database import get_connection
        with get_connection() as conn:
            engine = InsightEngine(conn)
            insights = engine.get_insights(max_count=2)
        for msg in insights:
            console.print(f"  [dim]· {escape(msg)}[/]")
        if insights:
            console.print("")
    except Exception:
        pass  # Non-fatal: insights never break shell launch

    # Log shell entry so idle detection stays accurate
    try:
        from pb.storage.database import log_usage
        log_usage("shell", 0)
    except Exception:
        pass

    active_session = None
    try:
        active_session = _safe_get_active_session(repo)
    except Exception:
        active_session = None
    active_label = "none"
    if active_session is not None:
        active_label = getattr(active_session, "subject_scope", "") or getattr(active_session, "branch", "study") or "active"
    console.print(
        f"pb shell  vault={escape(vault_root.name)}  active={escape(active_label)}  exit=exit|quit|Ctrl-D"
    )
    if not shell_test_mode:
        _show_startup_picker(
            click_app,
            vault_root,
            repo,
            _cwd_ref,
            on_cd=on_cd,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            pb_command_resolver=pb_command_resolver,
        )

    if _PROMPT_TOOLKIT_AVAILABLE and not shell_test_mode:
        _run_shell_prompt_toolkit(
            click_app,
            vault_root,
            repo,
            _cwd_ref,
            get_vault_cwd,
            on_cd=on_cd,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            pb_command_resolver=pb_command_resolver,
        )
    else:
        _run_shell_fallback(
            click_app,
            vault_root,
            repo,
            _cwd_ref,
            on_cd=on_cd,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            pb_command_resolver=pb_command_resolver,
        )


def _show_startup_picker(
    click_app,
    vault_root: Path,
    repo,
    _cwd_ref: list,
    on_cd: Callable[[], None] | None = None,
    runtime: LLMRuntime | None = None,
    runtime_ctx=None,
    pb_command_resolver: PbCommandResolver | None = None,
) -> None:
    """Offer a first action before dropping into the shell prompt."""
    choice = pick_single_choice(
        [
            ("__prompt_only__", "Open the learning shell"),
            ("next", "Choose the best next learning action"),
            ("learn", "Start the next study or practise block"),
            ("thought", "Capture a quick thought"),
        ],
        title="Start here",
        text="Pick a first move, or drop into the learning shell.",
        details=[
            "Go straight to the learning shell prompt.",
            "See the most relevant next move from your local learning context.",
            "Drop straight into the learning loop.",
            "Log an idea or friction point without ceremony.",
        ],
    )
    if not choice or choice == "__prompt_only__":
        return
    try:
        selected_args = shlex.split(choice)
    except ValueError:
        selected_args = choice.split()
    if selected_args:
        _dispatch(
            selected_args,
            click_app,
            vault_root,
            _cwd_ref,
            on_cd=on_cd,
            repo=repo,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            raw_input=choice,
            pb_command_resolver=pb_command_resolver,
        )


def _run_shell_prompt_toolkit(
    click_app,
    vault_root: Path,
    repo,
    _cwd_ref: list,
    get_vault_cwd: Callable[[], Path],
    on_cd: Callable[[], None] | None = None,
    runtime: LLMRuntime | None = None,
    runtime_ctx=None,
    pb_command_resolver: PbCommandResolver | None = None,
) -> None:
    """prompt_toolkit-powered interactive loop (main path)."""
    history_path = get_history_path()
    pb_commands = pb_command_resolver.command_names if pb_command_resolver is not None else list(getattr(click_app, "commands", {}).keys())

    completer = VaultCompleter(
        vault_root,
        get_vault_cwd,
        list(pb_commands),
        get_contextual_commands=lambda: _contextual_slash_commands(
            repo,
            runtime,
            runtime_ctx,
            _safe_get_active_session(repo),
        ) if repo is not None and runtime is not None and runtime_ctx is not None else [],
    )
    session = PromptSession(
        history=FileHistory(str(history_path)),
        completer=completer,
        complete_while_typing=False,              # Pitfall 5: don't trigger on every keystroke
    )
    get_prompt = make_prompt_callable(vault_root, get_vault_cwd, repo)

    while True:
        try:
            line = session.prompt(get_prompt)     # Pitfall 2: session created ONCE outside loop
        except KeyboardInterrupt:
            continue                               # D-25: Ctrl-C cancels current line
        except EOFError:
            get_console().print("")
            break                                 # D-26: Ctrl-D exits

        stripped = line.strip()
        if not stripped:
            continue
        if stripped in EXIT_TOKENS:
            break

        try:
            args = shlex.split(stripped)
        except ValueError:
            get_err_console().print("[warn]Note: unmatched quotes detected, interpreting literally.[/]")
            args = stripped.split()

        _dispatch(
            args,
            click_app,
            vault_root,
            _cwd_ref,
            on_cd=on_cd,
            repo=repo,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            raw_input=stripped,
            pb_command_resolver=pb_command_resolver,
        )


def _run_shell_fallback(
    click_app,
    vault_root: Path,
    repo,
    _cwd_ref: list,
    on_cd: Callable[[], None] | None = None,
    runtime: LLMRuntime | None = None,
    runtime_ctx=None,
    pb_command_resolver: PbCommandResolver | None = None,
) -> None:
    """Fallback input()-based loop when prompt_toolkit is unavailable."""
    while True:
        try:
            _session = _safe_get_active_session(repo)
            if _session:
                _task = repo.get_task(_session.task_id)
                _task_name = stored_short_title(_task) if _task else ""
                prompt_str = f"pb [{_task_name}]> "
            else:
                prompt_str = "pb> "
        except Exception:
            prompt_str = "pb> "

        try:
            user_input = input(prompt_str)
        except KeyboardInterrupt:
            continue                               # D-25: Ctrl-C cancels current line
        except EOFError:
            get_console().print("")
            break                                 # D-26: Ctrl-D exits

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped in EXIT_TOKENS:
            break

        try:
            args = shlex.split(stripped)
        except ValueError:
            get_err_console().print("[warn]Note: unmatched quotes detected, interpreting literally.[/]")
            args = stripped.split()

        _dispatch(
            args,
            click_app,
            vault_root,
            _cwd_ref,
            on_cd=on_cd,
            repo=repo,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            raw_input=stripped,
            pb_command_resolver=pb_command_resolver,
        )


def _handle_suggest(
    args: list[str],
    vault_root: Path,
    _cwd_ref: list,
    repo,
    click_app,
    on_cd=None,
    runtime: LLMRuntime | None = None,
    runtime_ctx=None,
    pb_command_resolver: PbCommandResolver | None = None,
) -> None:
    """Handle '? <intent>' via the multi-candidate do-style router."""
    console = get_console()
    intent = " ".join(args)
    if not intent:
        console.print("Usage: ? <what you want to do>")
        return

    candidates = suggest_commands_for_intent(repo, intent, limit=5)
    if not candidates:
        console.print("Didn't understand. Try `pb help`")
        return

    if len(candidates) == 1:
        command = candidates[0].backing_command
    else:
        choice = pick_single_choice(
            [(item.backing_command, item.human_label) for item in candidates],
            title="Choose what to do",
            text=intent,
            details=[item.short_reason for item in candidates],
        )
        if not choice:
            return
        command = choice

    # Execute the confirmed suggestion by re-dispatching through _dispatch
    try:
        suggested_args = shlex.split(command)
    except ValueError:
        suggested_args = command.split()
    # Strip leading "pb" if present (shell doesn't need it)
    if suggested_args and suggested_args[0].lower() == "pb":
        suggested_args = suggested_args[1:]
    if suggested_args:
        _dispatch(
            suggested_args,
            click_app,
            vault_root,
            _cwd_ref,
            on_cd=on_cd,
            repo=repo,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            raw_input=command,
            pb_command_resolver=pb_command_resolver,
        )


def _handle_mkmv(args: list[str], vault_root: Path, _cwd_ref: list) -> None:
    """Handle 'mkmv <description>' — find notes about a topic and collate them.

    Default: creates a collection note linking to all matching notes.
    With --folder: physically moves matching notes into a target folder.
    """
    console = get_console()
    move_mode = "--folder" in args
    filtered_args = [a for a in args if a != "--folder"]
    description = " ".join(filtered_args)
    if not description:
        console.print("mkmv: missing topic")
        console.print("Usage: mkmv <topic>          — create collection note")
        console.print("       mkmv --folder <topic>  — move notes into folder")
        return

    # Check API availability and inform user (AISG-04, D-09)
    client = get_client()
    if not client.is_available():
        console.print("  [dim]AI features unavailable -- filtering and folder suggestions will be manual.[/]")

    mkmv = MkMvEngine(vault_root=vault_root)

    # Step 1: Find matching notes across the vault
    console.print(f"  Searching for notes about: {escape(description)}")
    candidates = mkmv.find_matching_notes(description)
    if not candidates:
        console.print("  [dim]No matching notes found.[/]")
        return

    # Step 2: AI filter if available, otherwise show all
    try:
        selected_paths = mkmv.ai_filter_notes(description, candidates)
    except Exception:
        selected_paths = [p for p, _ in candidates]

    if not selected_paths:
        console.print("  [dim]I don't know which note you mean with enough confidence.[/]")
        for path, _ in candidates[:3]:
            console.print(f"    [dim]- {escape(str(path.relative_to(vault_root)))}[/]")
        return

    # Show matches
    console.print(f"\n  Found {len(selected_paths)} note(s):")
    for p in selected_paths:
        console.print(f"    [dim]{escape(str(p.relative_to(vault_root)))}[/]")

    if move_mode:
        _mkmv_move(selected_paths, description, vault_root)
    else:
        _mkmv_collection(selected_paths, description, vault_root, _cwd_ref)


def _mkmv_collection(
    notes: list[Path], description: str, vault_root: Path, _cwd_ref: list
) -> None:
    """Create a collection note that links to all matched notes."""
    console = get_console()
    err_console = get_err_console()
    lines = [f"# {description}\n"]
    for p in notes:
        rel = p.relative_to(vault_root)
        stem = p.stem.replace("-", " ").replace("_", " ").title()
        lines.append(f"- [[{rel}|{stem}]]")
    content = "\n".join(lines) + "\n"

    preview = "\n".join(lines[:8])
    if len(lines) > 8:
        preview += f"\n  ... ({len(lines) - 8} more)"
    if not tier2_confirm(f"Create collection: {description}", preview):
        return

    # Determine target folder -- top-level only (D-06, D-10, AISG-05)
    top_folders = _list_dirs(vault_root)
    if not top_folders:
        err_console.print("[error]mkmv: no folders in vault root[/]")
        return

    mkmv = MkMvEngine(vault_root=vault_root)
    folder_names = [f.name for f in top_folders]
    target_folder = None
    try:
        ranked = mkmv.rank_folder(description, folder_names)
        if ranked:
            target_folder = next((f for f in top_folders if f.name == ranked), None)
    except Exception:
        pass

    if target_folder:
        try:
            target_rel = str(target_folder.relative_to(vault_root))
        except ValueError:
            target_rel = target_folder.name
        if not tier2_confirm(f"Place in: {target_rel}/", ""):
            return
    else:
        console.print("  Pick target folder:")
        target_folder = _pick_numbered(top_folders, vault_root)
        if not target_folder:
            return

    slug = re.sub(r"[^\w]+", "-", description.lower())[:40].strip("-")
    note_path = _unique_note_path(target_folder / f"{slug}.md")
    note_path.write_text(content, encoding="utf-8")
    console.print(f"  [success]Created: {escape(str(note_path.relative_to(vault_root)))}[/]")

    note_rel_path = str(note_path.relative_to(vault_root))
    try:
        update_note_in_graph(vault_root, note_rel_path, content)
        rebuild_folder_index(target_folder, vault_root)
        summary = generate_directory_md(target_folder)
        (target_folder / ".pb-directory.md").write_text(summary, encoding="utf-8")
    except Exception as e:
        _logger.debug("mkmv.collection_post_write", error=str(e))
    # Phase 17: enforce bidirectional backlinks (D-09) + warn if no outgoing links (GRPH-03)
    # + log interaction (D-02) + check promotion with Rich notification (D-12)
    try:
        from pb.vault.graph import enforce_backlinks_for_note, check_no_outgoing_links
        from pb.vault.lifecycle import log_interaction, check_promotion
        _console = get_console()

        enforce_backlinks_for_note(vault_root, note_rel_path, content)

        warning = check_no_outgoing_links(content, note_rel_path)
        if warning:
            _console.print(warning)

        log_interaction(note_path=note_rel_path, event_type="read")

        # D-12: check if this interaction triggers a stage promotion
        promo_msg = check_promotion(note_rel_path, vault_root)
        if promo_msg:
            _console.print(promo_msg)
    except Exception:
        pass  # Non-fatal: hooks must never break vault write


def _mkmv_move(notes: list[Path], description: str, vault_root: Path) -> None:
    """Physically move matched notes into a target folder."""
    console = get_console()
    err_console = get_err_console()
    top_folders = _list_dirs(vault_root)
    if not top_folders:
        err_console.print("[error]mkmv: no folders in vault root[/]")
        return

    # AI-suggest or manual pick
    mkmv = MkMvEngine(vault_root=vault_root)
    folder_names = [f.name for f in top_folders]
    target = None
    try:
        ranked = mkmv.rank_folder(description, folder_names)
        if ranked:
            target = next((f for f in top_folders if f.name == ranked), None)
    except Exception:
        pass

    if target:
        try:
            target_rel = str(target.relative_to(vault_root))
        except ValueError:
            target_rel = target.name
        if not tier2_confirm(f"Move {len(notes)} note(s) to: {target_rel}/", ""):
            return
    else:
        console.print("  Pick target folder:")
        target = _pick_numbered(top_folders, vault_root)
        if not target:
            return

    # Move files, skip if already in target
    moved = 0
    for src in notes:
        if src.parent == target:
            continue
        dest = target / src.name
        if dest.exists():
            console.print(f"  [dim]skip (exists): {escape(src.name)}[/]")
            continue
        src.rename(dest)
        moved += 1
        # Update graph: remove old, add new
        dest_rel_path = str(dest.relative_to(vault_root))
        dest_content = dest.read_text(encoding="utf-8", errors="replace")
        try:
            update_note_in_graph(vault_root, dest_rel_path, dest_content)
        except Exception as e:
            _logger.debug("mkmv.move_graph_update", error=str(e))
        # Phase 17: enforce bidirectional backlinks (D-09) + warn if no outgoing links (GRPH-03)
        # + log interaction (D-02) + check promotion with Rich notification (D-12)
        try:
            from pb.vault.graph import enforce_backlinks_for_note, check_no_outgoing_links
            from pb.vault.lifecycle import log_interaction, check_promotion
            _console = get_console()

            enforce_backlinks_for_note(vault_root, dest_rel_path, dest_content)

            warning = check_no_outgoing_links(dest_content, dest_rel_path)
            if warning:
                _console.print(warning)

            log_interaction(note_path=dest_rel_path, event_type="read")

            # D-12: check if this interaction triggers a stage promotion
            promo_msg = check_promotion(dest_rel_path, vault_root)
            if promo_msg:
                _console.print(promo_msg)
        except Exception:
            pass  # Non-fatal: hooks must never break vault write

    console.print(f"  [success]Moved {moved} note(s) to {escape(target.name)}/[/]")

    # Rebuild indexes for affected folders
    affected = {target} | {n.parent for n in notes if n.parent != target}
    for folder in affected:
        try:
            rebuild_folder_index(folder, vault_root)
            summary = generate_directory_md(folder)
            (folder / ".pb-directory.md").write_text(summary, encoding="utf-8")
        except Exception as e:
            _logger.debug("mkmv.move_reindex", error=str(e))


def _dispatch(
    args: list[str],
    click_app,
    vault_root: Path,
    _cwd_ref: list,
    on_cd: Callable[[], None] | None = None,
    repo=None,
    runtime: LLMRuntime | None = None,
    runtime_ctx=None,
    raw_input: str | None = None,
    pb_command_resolver: PbCommandResolver | None = None,
) -> None:
    """Dispatch a parsed command to vault commands or the Click app."""
    if not args:
        return

    raw_text = (raw_input or " ".join(args)).strip()
    resolver = pb_command_resolver or PbCommandResolver(click_app)
    active_session = _safe_get_active_session(repo) if repo is not None else None
    partner = None
    slash_registry = None
    if (
        active_session is not None
        and repo is not None
        and runtime is not None
        and runtime_ctx is not None
    ):
        partner = _learning_partner_for_session(repo, runtime, runtime_ctx, active_session)
        if partner is not None:
            slash_registry = partner.command_registry

    decision = classify_interactive_input(
        raw_text,
        pb_command_resolver=resolver,
        slash_registry=slash_registry,
        active_learning=active_session is not None,
        allow_shell_commands=True,
        allow_nl_dispatch=active_session is None,
    )

    if decision.kind == "empty":
        return

    if decision.kind == "answer":
        nested = _coaching_turn(repo, runtime, runtime_ctx, active_session, decision.text)
        if nested is not None:
            _dispatch(
                list(nested.argv),
                click_app,
                vault_root,
                _cwd_ref,
                on_cd=on_cd,
                repo=repo,
                runtime=runtime,
                runtime_ctx=runtime_ctx,
                raw_input=nested.text,
                pb_command_resolver=resolver,
            )
        return

    if decision.kind == "navigation":
        if partner is not None:
            partner._browse(decision.argv or (decision.command,))
            nested = _render_partner_turn_chain(partner, partner.current_turn or partner.open_with_first_move())
            if nested is not None:
                _dispatch(
                    list(nested.argv),
                    click_app,
                    vault_root,
                    _cwd_ref,
                    on_cd=on_cd,
                    repo=repo,
                    runtime=runtime,
                    runtime_ctx=runtime_ctx,
                    raw_input=nested.text,
                    pb_command_resolver=resolver,
                )
            return
        get_err_console().print("[error]Arrow navigation is only available inside an active learning session.[/]")
        return

    if decision.kind == "slash_command":
        if partner is None:
            get_err_console().print("[error]No active learning context for contextual slash commands.[/]")
            return
        next_turn = partner.run_contextual_command(decision.command)
        if next_turn is not None:
            nested = _render_partner_turn_chain(partner, next_turn)
            if nested is not None:
                _dispatch(
                    list(nested.argv),
                    click_app,
                    vault_root,
                    _cwd_ref,
                    on_cd=on_cd,
                    repo=repo,
                    runtime=runtime,
                    runtime_ctx=runtime_ctx,
                    raw_input=nested.text,
                    pb_command_resolver=resolver,
                )
        return

    if decision.kind in {"slash_ambiguous", "slash_unknown"}:
        if partner is not None:
            partner.explain_contextual_command_error(decision)
        else:
            get_err_console().print("[error]Unknown slash command.[/]")
        return

    dispatched_args = list(decision.argv)
    first = decision.command.lower() if decision.command else (dispatched_args[0].lower() if dispatched_args else "")

    if decision.kind == "shell_command" and first == "ls":
        cmd_ls(_cwd_ref[0])
        return
    if decision.kind == "shell_command" and first == "cd":
        _cwd_ref[0] = cmd_cd(args[1:], _cwd_ref[0], vault_root)
        _maybe_refresh_folder_index(_cwd_ref[0], vault_root)
        if on_cd is not None:
            try:
                on_cd()
            except Exception:
                pass  # Never break shell navigation on chat engine failure
        return
    if decision.kind == "shell_command" and first == "grep":
        cmd_grep(args[1:], _cwd_ref[0])
        return
    if decision.kind == "shell_command" and first == "cat":
        cmd_cat(args[1:], _cwd_ref[0], vault_root)
        return
    if decision.kind == "shell_command" and first == "?":
        _handle_suggest(
            args[1:],
            vault_root,
            _cwd_ref,
            repo,
            click_app,
            on_cd=on_cd,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            pb_command_resolver=resolver,
        )
        return
    if decision.kind == "shell_command" and first == "mkmv":
        _handle_mkmv(args[1:], vault_root, _cwd_ref)
        return
    if decision.kind == "shell_command" and first == "deactivate":
        from pb.mcp.protocol import deactivate_all_sessions
        count = deactivate_all_sessions()
        console = get_console()
        if count > 0:
            console.print(f"[dim]Cleared {count} active dispatch session(s).[/]")
        else:
            console.print("[dim]No active dispatch sessions to clear.[/]")
        return

    if decision.kind not in {"pb_command", "dispatch"}:
        get_err_console().print(
            f"[error]Unknown command: {escape(first or raw_text)}[/]\n"
            f"  [dim]Type [bold]do {escape(raw_text)}[/bold] to dispatch, or [bold]?[/bold] for help.[/]"
        )
        return

    # FIX-02: propagate shell's virtual cwd so pb learn can find _state.md
    os.environ["PB_SHELL_VAULT_CWD"] = str(_cwd_ref[0])
    prior_active_session_id = None
    if repo is not None:
        previous_session = _safe_get_active_session(repo)
        prior_active_session_id = getattr(previous_session, "id", None)
    try:
        click_app(dispatched_args, standalone_mode=False)   # D-03: existing routing pattern
    except SystemExit:
        pass
    except Exception as e:
        log_ref = log_error(
            event="shell.dispatch_exception",
            message=str(e),
            exc=e,
            data_dir=getattr(runtime_ctx, "data_dir", None),
            command="shell",
            raw_input=raw_text,
            status="shell",
            extra={
                "exception_type": e.__class__.__name__,
                "parsed_args": dispatched_args,
                "vault_cwd": str(_cwd_ref[0]),
            },
        )
        get_err_console().print(f"[error]Error: {escape(format_logged_exception(e, log_ref))}[/]")
    finally:
        # FIX-01: flush stdout/stderr so prompt_toolkit REPL gets clean terminal state
        # after commands that call input()/typer.prompt() (e.g., pb finish)
        import sys as _sys

        _sys.stdout.flush()
        _sys.stderr.flush()
    if repo is not None and runtime is not None and runtime_ctx is not None:
        active_after = _safe_get_active_session(repo)
        if active_after is not None and getattr(active_after, "id", None) != prior_active_session_id:
            _maybe_open_learning_session(repo, runtime, runtime_ctx, active_after)
