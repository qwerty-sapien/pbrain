# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Task and skill management service."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pb.tasks.service import TaskService


def __getattr__(name: str):
    if name == "TaskService":
        from pb.tasks.service import TaskService
        globals()[name] = TaskService
        return TaskService
    raise AttributeError(f"module 'pb.tasks' has no attribute {name!r}")


__all__ = ["TaskService"]
