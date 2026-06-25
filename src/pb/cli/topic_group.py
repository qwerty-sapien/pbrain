# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Custom Typer group helpers for CLI surfaces with free-form topic fallback."""

from __future__ import annotations

import click
from typer.core import TyperGroup


class TopicFallbackGroup(TyperGroup):
    """Treat unknown would-be subcommands as free-form topic text."""

    def invoke(self, ctx: click.Context):
        if ctx._protected_args:
            args = [*ctx._protected_args, *ctx.args]
            if args and args[0] not in self.commands:
                ctx.args = args
                ctx._protected_args = []
                ctx.invoked_subcommand = None
                with ctx:
                    return click.Command.invoke(self, ctx)
        return super().invoke(ctx)
