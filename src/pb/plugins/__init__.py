# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Plugin registry -- static list in v4.0, dynamic discovery deferred to v5.0.

Plugins are registered manually in ENABLED_PLUGINS.
register_all() is called at app startup to mount plugin commands and services.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pb.plugins.base import Plugin
    import typer

# Static registry. Populated as plugins graduate from shelf.
# Example: from pb.plugins.people import PeoplePlugin; ENABLED_PLUGINS.append(PeoplePlugin())
ENABLED_PLUGINS: list = []


def register_all(app: "typer.Typer", ctx: dict) -> None:
    """Register all enabled plugins with the typer app and service context.

    Called once at startup. Each plugin adds its commands and services.
    """
    for plugin in ENABLED_PLUGINS:
        plugin.register_commands(app)
        plugin.register_services(ctx)
