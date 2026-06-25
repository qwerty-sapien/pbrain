# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Goal alignment and tracking service."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pb.goals.service import GoalsService

def __getattr__(name: str):
    if name == "GoalsService":
        from pb.goals.service import GoalsService
        globals()[name] = GoalsService
        return GoalsService
    raise AttributeError(f"module 'pb.goals' has no attribute {name!r}")

__all__ = ["GoalsService"]
