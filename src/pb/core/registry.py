# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Command registry for the pb chat REPL (D-11).

Provides an extensible dispatch table that replaces the if/elif chain in chat.py.
Each /command is a registered CommandHandler with name, help text, optional aliases,
and a callable handler.

Usage:
    registry = CommandRegistry()
    registry.register(CommandHandler(
        name="/help",
        help_text="Show available commands",
        handler=lambda args, ctx: typer.echo("..."),
        aliases=["/h"],
    ))
    if registry.dispatch(user_input, ctx):
        continue  # command was handled
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CommandHandler:
    """Metadata and handler for a registered in-chat command (D-11).

    Attributes:
        name: Primary command token, e.g. '/help'. Used for dispatch and help display.
        help_text: One-line description shown in /help listing.
        handler: Callable invoked as handler(args: str, ctx: dict) -> None.
        aliases: Optional alternative tokens that also dispatch to this handler.
    """

    name: str
    help_text: str
    handler: Callable[..., None]
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CommandResolution:
    """Resolution result for one command-like input line."""

    status: str
    command: str = ""
    args: str = ""
    handler: CommandHandler | None = None
    matches: tuple[str, ...] = ()


class CommandRegistry:
    """Dispatch table for /commands in the chat REPL (D-11).

    Replaces the if/elif chain in chat.py. Each command is a registered
    CommandHandler with name, help text, and handler callable.

    Dispatch is case-insensitive. Aliases are stored as separate keys pointing
    to the same handler. help_lines() returns one line per unique handler in
    registration order (aliases do not produce duplicate entries).

    Example::

        registry = CommandRegistry()
        registry.register(CommandHandler("/help", "Show help", my_help_fn))
        registry.register(CommandHandler("/new", "Fresh session", my_new_fn, aliases=["/n"]))

        handled = registry.dispatch(user_input, ctx)
    """

    def __init__(self) -> None:
        # Maps command token (lower-cased) -> CommandHandler
        self._handlers: dict[str, CommandHandler] = {}
        # Tracks primary names in registration order for help display
        self._order: list[str] = []

    def register(self, handler: CommandHandler) -> None:
        """Register a command handler.

        The handler's primary name and all aliases are registered as dispatch keys.
        Aliases point to the same CommandHandler object as the primary name.
        Registering the same name twice overwrites the previous handler.

        Args:
            handler: CommandHandler instance to register.
        """
        self._handlers[handler.name.lower()] = handler
        if handler.name not in self._order:
            self._order.append(handler.name)
        for alias in handler.aliases:
            self._handlers[alias.lower()] = handler

    def resolve(self, line: str) -> CommandResolution:
        """Resolve a /command line without executing it.

        Status values:
            - ``empty``: line was empty or whitespace
            - ``exact``: exact command or alias match
            - ``unique_prefix``: one unique prefix match
            - ``ambiguous_prefix``: multiple distinct commands match the prefix
            - ``missing``: no command matches
        """
        if not line or not line.strip():
            return CommandResolution(status="empty")

        parts = line.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handler = self._handlers.get(cmd)
        if handler is not None:
            return CommandResolution(
                status="exact",
                command=handler.name,
                args=args,
                handler=handler,
            )

        matching_handlers: dict[str, CommandHandler] = {}
        for token, candidate in self._handlers.items():
            if token.startswith(cmd):
                matching_handlers.setdefault(candidate.name, candidate)

        if not matching_handlers:
            return CommandResolution(status="missing")

        if len(matching_handlers) == 1:
            resolved_handler = next(iter(matching_handlers.values()))
            return CommandResolution(
                status="unique_prefix",
                command=resolved_handler.name,
                args=args,
                handler=resolved_handler,
            )

        return CommandResolution(
            status="ambiguous_prefix",
            matches=tuple(sorted(matching_handlers.keys())),
        )

    def dispatch(self, line: str, ctx: dict) -> bool:
        """Dispatch a /command line to its registered handler.

        Splits line into (command_token, args). Command lookup is case-insensitive.
        The handler is called as handler(args, ctx).

        Args:
            line: Raw input line from the REPL.
            ctx: Context dict passed through to the handler (e.g. engine state).

        Returns:
            True if a registered handler was found and called.
            False if the line is empty, whitespace-only, or no matching handler found.
        """
        resolution = self.resolve(line)
        if resolution.status not in {"exact", "unique_prefix"} or resolution.handler is None:
            return False
        resolution.handler.handler(resolution.args, ctx)
        return True

    def command_names(self) -> list[str]:
        """Return primary command names in registration order."""
        return list(self._order)

    def exact_match(self, token: str) -> CommandHandler | None:
        """Return the handler for an exact command token or alias."""
        if not token or not token.strip():
            return None
        return self._handlers.get(token.strip().lower())

    def has_command(self, token: str) -> bool:
        """Return True when a token is registered exactly."""
        return self.exact_match(token) is not None

    def prefix_matches(self, token: str) -> list[str]:
        """Return sorted primary command names matching a token prefix."""
        resolution = self.resolve(token)
        if resolution.status == "ambiguous_prefix":
            return list(resolution.matches)
        if resolution.status in {"exact", "unique_prefix"} and resolution.command:
            return [resolution.command]
        return []

    def help_lines(self) -> list[str]:
        """Return formatted help lines, one per unique handler, in registration order.

        Aliases do not appear as separate lines — only the primary handler name is shown.
        Each line is left-padded with two spaces and the name is padded to 16 characters.

        Returns:
            List of formatted strings, e.g. ['  /help             Show help'].
        """
        lines: list[str] = []
        for name in self._order:
            h = self._handlers.get(name.lower())
            if h:
                lines.append(f"  {h.name:<16} {h.help_text}")
        return lines
