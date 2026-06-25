# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared time-based attenuation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import exp, floor, log

_SECONDS_PER_DAY = 86400.0
_WEEK_DAYS = 7


def days_elapsed_since(captured_at: datetime, *, now: datetime | None = None) -> float:
    """Return elapsed wall-clock days, clamped at zero."""
    baseline = now or datetime.utcnow()
    delta = baseline - captured_at
    return max(0.0, delta.total_seconds() / _SECONDS_PER_DAY)


def thought_weight_bucket(days_elapsed: float) -> int:
    """Return the 7-day attenuation bucket for an elapsed-day value."""
    clamped = max(0.0, days_elapsed)
    return _WEEK_DAYS * floor(clamped / _WEEK_DAYS)


def thought_time_weight(x: float) -> float:
    """Return the bounded time weight for a thought captured x days ago."""
    w = thought_weight_bucket(x)
    u = -log(2) * w / 10
    p = -0.57
    r = exp(u)
    g = (1 + 0.0058333 * (w - 2)) ** p
    return min(1.0, max(0.05, 0.6 * r + 0.4 * g))


def thought_weight_state(
    captured_at: datetime,
    *,
    now: datetime | None = None,
) -> tuple[int, float, datetime]:
    """Return (bucket_days, weight, next_recompute_at) for a thought."""
    baseline = now or datetime.utcnow()
    elapsed = days_elapsed_since(captured_at, now=baseline)
    bucket_days = thought_weight_bucket(elapsed)
    next_recompute_at = captured_at + timedelta(days=bucket_days + _WEEK_DAYS)
    return bucket_days, thought_time_weight(elapsed), next_recompute_at
