# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared interactive input routing for shell and learning-session prompts."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Callable, Iterable

from pb.core.registry import CommandRegistry

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.key_binding import KeyBindings

    _PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback
    PromptSession = None  # type: ignore[assignment]
    Completer = object  # type: ignore[assignment]
    Completion = None  # type: ignore[assignment]
    KeyBindings = None  # type: ignore[assignment]
    _PROMPT_TOOLKIT_AVAILABLE = False


SHELL_COMMANDS = ("ls", "cd", "grep", "cat", "?", "mkmv", "deactivate")
EXIT_TOKENS = {"exit", "quit", "/exit", "/quit"}
_NL_TRIGGER_PREFIXES = ("i ", "i'm ", "what ", "how ", "why ", "help ", "can ", "please ", "do ")
_NAVIGATION_SEQUENCES = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "^[[A": "up",
    "^[[B": "down",
    "^[[C": "right",
    "^[[D": "left",
}


@dataclass(frozen=True)
class RoutedInput:
    """Minimal routed input decision shared across interactive surfaces."""

    kind: str
    text: str = ""
    argv: tuple[str, ...] = ()
    command: str = ""
    args: str = ""
    matches: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class QuestionCommandBuffer:
    """Attached slash-command editing state for a live lesson question."""

    active: bool = False
    text: str = ""

    def activate(self) -> None:
        self.active = True
        self.text = "/"

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        if not self.active:
            self.activate()
        self.text += chunk

    def backspace(self) -> None:
        if not self.active:
            return
        self.text = self.text[:-1]
        if not self.text:
            self.clear()

    def clear(self) -> None:
        self.active = False
        self.text = ""


def _extract_navigation_inputs(raw_text: str) -> list[str]:
    remaining = str(raw_text or "").strip()
    if not remaining:
        return []
    directions: list[str] = []
    while remaining:
        matched = False
        for token, direction in _NAVIGATION_SEQUENCES.items():
            if remaining.startswith(token):
                directions.append(direction)
                remaining = remaining[len(token):].strip()
                matched = True
                break
        if not matched:
            return []
    return directions


def split_interactive_input(raw_text: str) -> tuple[list[str], bool]:
    """Split one interactive line into argv, falling back on literal whitespace."""
    stripped = (raw_text or "").strip()
    if not stripped:
        return [], True
    try:
        return shlex.split(stripped), True
    except ValueError:
        return stripped.split(), False


class PbCommandResolver:
    """Non-executing resolver for real ``pb`` commands and root flags."""

    def __init__(self, click_app) -> None:
        self.click_app = click_app
        self.command_names = tuple(getattr(click_app, "commands", {}).keys())
        self._root_options = self._build_root_option_map()

    def _build_root_option_map(self) -> dict[str, bool]:
        option_map: dict[str, bool] = {"--help": False, "-h": False}
        for param in getattr(self.click_app, "params", []):
            opts = getattr(param, "opts", []) or []
            secondary = getattr(param, "secondary_opts", []) or []
            takes_value = not bool(getattr(param, "is_flag", False))
            for token in [*opts, *secondary]:
                option_map[str(token)] = takes_value
        return option_map

    def root_option_tokens(self) -> list[str]:
        """Return the declared root option tokens."""
        return sorted(self._root_options.keys())

    def resolve(self, argv: Iterable[str]) -> RoutedInput | None:
        """Resolve argv to a real pb command or root-flag invocation."""
        normalized = [str(token).strip() for token in argv if str(token).strip()]
        if normalized and normalized[0].lower() == "pb":
            normalized = normalized[1:]
        if not normalized:
            return None

        index = 0
        while index < len(normalized):
            token = normalized[index]
            if token == "--":
                index += 1
                break
            if not token.startswith("-"):
                break
            option_token = token.split("=", 1)[0]
            if option_token not in self._root_options:
                return None
            takes_value = self._root_options[option_token]
            if takes_value and "=" not in token:
                next_index = index + 1
                if next_index >= len(normalized):
                    return None
                if normalized[next_index].startswith("-") and normalized[next_index] not in self.command_names:
                    return None
                index += 2
                continue
            index += 1

        remaining = normalized[index:]
        if remaining:
            if remaining[0] not in self.command_names:
                return None
        elif not any(token.startswith("-") for token in normalized):
            return None

        return RoutedInput(
            kind="pb_command",
            text=" ".join(normalized),
            argv=tuple(normalized),
            command=remaining[0] if remaining else "",
        )


class SlashCommandCompleter(Completer):
    """Prompt-toolkit completer for contextual slash commands."""

    def __init__(self, get_commands: Callable[[], list[str]]) -> None:
        self.get_commands = get_commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        prefix = text.strip()
        if not prefix.startswith("/"):
            return
        for command in self.get_commands():
            if command.startswith(prefix):
                yield Completion(command, start_position=-len(prefix))


def prompt_answer_or_command(
    *,
    prompt_label: str,
    registry: CommandRegistry | None,
    pb_command_resolver: PbCommandResolver | None = None,
    allow_navigation: bool = False,
) -> RoutedInput:
    """Read one inline answer field while still intercepting slash commands."""
    get_commands = (lambda: registry.command_names()) if registry is not None else (lambda: [])

    while True:
        if _PROMPT_TOOLKIT_AVAILABLE and os.environ.get("TERM"):
            navigation_result: RoutedInput | None = None
            bindings = KeyBindings() if allow_navigation else None
            if bindings is not None:
                for direction in ("left", "right", "up", "down"):
                    @bindings.add(direction)
                    def _navigate(event, direction=direction) -> None:
                        nonlocal navigation_result
                        navigation_result = RoutedInput(
                            kind="navigation",
                            text=direction,
                            argv=(direction,),
                            command=direction,
                        )
                        event.app.exit(result="")
            session = PromptSession(
                completer=SlashCommandCompleter(get_commands),
                complete_while_typing=False,
            )
            try:
                raw = session.prompt(prompt_label, key_bindings=bindings)
            except EOFError:
                return RoutedInput(kind="empty")
            except KeyboardInterrupt:
                return RoutedInput(kind="empty")
            if navigation_result is not None:
                return navigation_result
        else:  # pragma: no cover - exercised in TTY fallback paths
            try:
                raw = input(prompt_label)
            except EOFError:
                return RoutedInput(kind="empty")
            except KeyboardInterrupt:
                return RoutedInput(kind="empty")

        decision = classify_interactive_input(
            raw,
            pb_command_resolver=pb_command_resolver,
            slash_registry=registry,
            active_learning=True,
            allow_shell_commands=False,
            allow_nl_dispatch=False,
        )
        if decision.kind == "slash_ambiguous":
            matches = ", ".join(decision.matches)
            print(f"Matching commands: {matches}")
            continue
        return decision


def is_natural_language_input(args: list[str], raw_input: str | None) -> bool:
    """Return True when input should route to ``pb do`` instead of erroring."""
    try:
        from pb.storage.database import get_connection

        with get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM dispatch_sessions WHERE status='active' LIMIT 1"
            ).fetchone()
        if row is not None:
            return True
    except Exception:
        pass

    text = raw_input or " ".join(args)
    if len(args) >= 3:
        return True
    if text.startswith(('"', "'")):
        return True

    lowered = text.lower()
    return any(lowered.startswith(prefix) for prefix in _NL_TRIGGER_PREFIXES)


def classify_interactive_input(
    raw_text: str,
    *,
    pb_command_resolver: PbCommandResolver | None,
    slash_registry: CommandRegistry | None,
    active_learning: bool,
    allow_shell_commands: bool,
    allow_nl_dispatch: bool,
) -> RoutedInput:
    """Classify one interactive line before any agent or tutor sees it."""
    stripped = (raw_text or "").strip()
    if not stripped:
        return RoutedInput(kind="empty")

    if active_learning:
        navigation = _extract_navigation_inputs(raw_text)
        if navigation:
            return RoutedInput(
                kind="navigation",
                text=" ".join(navigation),
                argv=tuple(navigation),
                command=navigation[-1],
            )

    argv, _ = split_interactive_input(stripped)
    if allow_shell_commands and argv:
        first = argv[0].lower()
        if first == "pb" and len(argv) == 1:
            return RoutedInput(kind="empty")
        if first in SHELL_COMMANDS or first in EXIT_TOKENS:
            return RoutedInput(
                kind="shell_command",
                text=" ".join(argv),
                argv=tuple(argv),
                command=first,
            )

    if pb_command_resolver is not None:
        pb_match = pb_command_resolver.resolve(argv)
        if pb_match is not None:
            return pb_match

    if stripped.startswith("/model"):
        model_argv, _ = split_interactive_input(f"model {stripped[len('/model'):].strip()}".strip())
        pb_match = pb_command_resolver.resolve(model_argv) if pb_command_resolver is not None else None
        if pb_match is not None:
            return pb_match

    if stripped.startswith("/"):
        if slash_registry is None:
            return RoutedInput(kind="slash_unknown", text=stripped)
        resolution = slash_registry.resolve(stripped)
        if resolution.status in {"exact", "unique_prefix"}:
            return RoutedInput(
                kind="slash_command",
                text=stripped,
                command=resolution.command,
                args=resolution.args,
            )
        if resolution.status == "ambiguous_prefix":
            return RoutedInput(
                kind="slash_ambiguous",
                text=stripped,
                matches=resolution.matches,
            )
        return RoutedInput(kind="slash_unknown", text=stripped)

    if active_learning:
        return RoutedInput(kind="answer", text=stripped)

    if allow_nl_dispatch and is_natural_language_input(argv, stripped):
        return RoutedInput(
            kind="dispatch",
            text=stripped,
            argv=tuple(["do", *argv]),
            command="do",
        )

    return RoutedInput(
        kind="unknown",
        text=stripped,
        argv=tuple(argv),
        command=argv[0].lower() if argv else "",
    )
