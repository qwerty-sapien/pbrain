# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed command context — replaces scattered ctx.obj['...'] dict access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import typer
from rich.console import Console

from pb.cli.console import get_console, get_err_console
from pb.core.exceptions import ConfigError
from pb.runtime import RuntimeContext
from pb.storage.repository import Repository


@dataclass
class CommandContext:
    """Validated, typed context for command handlers."""

    repo: Repository
    runtime: RuntimeContext
    config: Any
    factory: dict
    console: Console
    err_console: Console
    yes: bool
    verbose: bool

    @staticmethod
    def from_typer(ctx: typer.Context) -> CommandContext:
        obj = ctx.obj or {}
        runtime = obj.get("runtime")
        repo = obj.get("repo")
        if runtime is None or repo is None:
            raise ConfigError("Runtime not initialized. Run `pb init` first.")
        return CommandContext(
            repo=repo,
            runtime=runtime,
            config=obj.get("config"),
            factory=obj.get("factory", {}),
            console=get_console(),
            err_console=get_err_console(),
            yes=bool(obj.get("yes", False)),
            verbose=bool(obj.get("verbose", False)),
        )

    def service(self, name: str):
        """Get a service from the lazy factory by name."""
        builder = self.factory.get(name)
        if builder is None:
            raise ConfigError(f"Service '{name}' not available.")
        return builder()
