# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Small clock helpers for UTC-compatible timestamps.

The wider codebase still stores mostly naive UTC datetimes, so these helpers
avoid deprecated `datetime.utcnow()` calls while preserving the existing
comparison semantics until a broader timezone-aware migration lands.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return a naive UTC datetime without using deprecated utcnow()."""
    return datetime.now(UTC).replace(tzinfo=None)
