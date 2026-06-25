# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Session lifecycle service."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pb.sessions.service import SessionService


def __getattr__(name: str):
    if name == "SessionService":
        from pb.sessions.service import SessionService
        globals()[name] = SessionService
        return SessionService
    raise AttributeError(f"module 'pb.sessions' has no attribute {name!r}")


__all__ = ["SessionService"]
