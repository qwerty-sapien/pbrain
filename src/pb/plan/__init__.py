# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Planning and scheduling service."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pb.plan.service import PlanService

def __getattr__(name: str):
    if name == "PlanService":
        from pb.plan.service import PlanService
        globals()[name] = PlanService
        return PlanService
    raise AttributeError(f"module 'pb.plan' has no attribute {name!r}")

__all__ = ["PlanService"]
