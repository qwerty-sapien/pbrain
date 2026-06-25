# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Plugin Protocol -- Flask Blueprint-inspired registration interface.

Every plugin must satisfy this protocol:
- name: unique plugin identifier
- register_commands(app): add typer commands to the app
- register_services(ctx): add service instances to the context

Static registry in Phase 21; dynamic discovery deferred to v5.0 (PLUG-F01).
"""
from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import typer


class Plugin(Protocol):
    """Protocol all plugins must satisfy.

    Flask Blueprint analogy: plugin.register_commands(app) is like
    blueprint.register(app). Services are registered into a context
    dict rather than a global.
    """
    name: str

    def register_commands(self, app: "typer.Typer") -> None: ...

    def register_services(self, ctx: dict) -> None: ...
