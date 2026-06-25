# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Base service classes and cross-cutting mixins.

PyTorch nn.Module-inspired: deps declared in __init__, sub-services
registered and accessible, no global state, no singletons.
"""

from __future__ import annotations

from typing import Optional


class BaseService:
    """PyTorch nn.Module-inspired service base.

    - Deps declared in __init__
    - Sub-services registered and accessible via _register/_get
    - No global state, no singletons
    """

    def __init__(self, **kwargs):
        self._sub_services: dict[str, BaseService] = {}

    def _register(self, name: str, service: BaseService) -> None:
        """Register a sub-service (like nn.Module sub-module registration)."""
        self._sub_services[name] = service

    def _get(self, name: str) -> Optional[BaseService]:
        """Retrieve a registered sub-service by name."""
        return self._sub_services.get(name)


class LoggableMixin:
    """Log-usage cross-cutting concern (React hooks-inspired).

    Mix in to record command invocations. Stub in Phase 21 --
    implementation fills in when usage logging migrates.
    """

    def log_usage(self, command: str, **kwargs) -> None:
        pass  # stub -- implementation fills in later


class AIMixin:
    """Flash Lite cross-cutting concern (React hooks-inspired).

    Mix in to access LLM capabilities. Stub in Phase 21 --
    returns GeminiClient when AI service migrates.
    """

    def get_ai_client(self):
        return None  # stub -- returns GeminiClient in later phases
