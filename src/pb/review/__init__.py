# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Review and reporting service."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pb.review.service import ReviewService

def __getattr__(name: str):
    if name == "ReviewService":
        from pb.review.service import ReviewService
        globals()[name] = ReviewService
        return ReviewService
    raise AttributeError(f"module 'pb.review' has no attribute {name!r}")

__all__ = ["ReviewService"]
