# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Duration helpers for learner-facing artifacts."""

from __future__ import annotations

from datetime import datetime


def elapsed_minutes_and_label(start_at: datetime | None, end_at: datetime | None) -> tuple[int, str]:
    """Return numeric elapsed minutes plus a friendly display label."""

    if not start_at or not end_at:
        return 0, "0 min"
    elapsed_seconds = max(0, int((end_at - start_at).total_seconds()))
    if elapsed_seconds == 0:
        return 0, "0 min"
    if elapsed_seconds < 60:
        return 1, "<1 min"
    minutes = int(elapsed_seconds / 60)
    return minutes, f"{minutes} min"
