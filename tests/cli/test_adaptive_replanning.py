# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from unittest.mock import MagicMock
from datetime import datetime

from pb.cli.commands.execute import _detect_session_overrun


def test_detect_overrun():
    repo = MagicMock()
    session = MagicMock()
    session.task_id = "t1"
    session.start_at = datetime(2026, 5, 27, 9, 0)
    session.end_at = datetime(2026, 5, 27, 10, 15)

    block = MagicMock(task_id="t1", duration_minutes=60)
    repo.list_time_blocks_for_date.return_value = [block]

    assert _detect_session_overrun(repo, session) == 15


def test_no_overrun():
    repo = MagicMock()
    session = MagicMock()
    session.task_id = "t1"
    session.start_at = datetime(2026, 5, 27, 9, 0)
    session.end_at = datetime(2026, 5, 27, 9, 55)

    block = MagicMock(task_id="t1", duration_minutes=60)
    repo.list_time_blocks_for_date.return_value = [block]

    assert _detect_session_overrun(repo, session) is None
