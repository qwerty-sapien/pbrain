# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from datetime import datetime, timedelta

from pb.core.durations import elapsed_minutes_and_label


def test_elapsed_duration_formats_positive_short_sessions_as_less_than_one_minute():
    start = datetime(2026, 6, 25, 12, 0, 0)

    assert elapsed_minutes_and_label(start, start + timedelta(seconds=20)) == (1, "<1 min")


def test_elapsed_duration_formats_whole_minutes_normally():
    start = datetime(2026, 6, 25, 12, 0, 0)

    assert elapsed_minutes_and_label(start, start + timedelta(minutes=12, seconds=30)) == (12, "12 min")
