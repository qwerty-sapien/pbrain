# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""AI integration service: chat, probing, suggestions."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pb.ai.service import AIService


def __getattr__(name: str):
    if name == "AIService":
        from pb.ai.service import AIService
        globals()[name] = AIService
        return AIService
    raise AttributeError(f"module 'pb.ai' has no attribute {name!r}")


__all__ = ["AIService"]
