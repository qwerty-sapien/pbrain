# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from unittest.mock import MagicMock
from pb.cli.commands.study import _todays_plan_blocks
from pb.cli.commands.execute import _next_session_recommendation


def test_full_day_plan_flow():
    repo = MagicMock()

    b1 = MagicMock(task_id="t1", duration_minutes=60, sub_index=None)
    b2a = MagicMock(task_id="t2a", duration_minutes=30, sub_index="2a")
    b2b = MagicMock(task_id="t2b", duration_minutes=30, sub_index="2b")
    repo.list_time_blocks_created_for_date.return_value = [b1, b2a, b2b]

    t1 = MagicMock(id="t1", archived_at=None, completion=0, title="German", description="")
    t2a = MagicMock(id="t2a", archived_at=None, completion=0, title="Calc Core", description="")
    t2b = MagicMock(id="t2b", archived_at=None, completion=0, title="LinAlg SR", description="")
    repo.get_task = lambda tid: {"t1": t1, "t2a": t2a, "t2b": t2b}.get(tid)

    # Plan items have correct codes
    items = _todays_plan_blocks(repo)
    assert [code for code, _, _ in items] == ["1", "2a", "2b"]

    # Finish t1 → recommend 2a
    t1.completion = 100
    assert _next_session_recommendation(repo, "t1") == "pb study 2a"

    # Finish t2a → recommend 2b
    t2a.completion = 100
    assert _next_session_recommendation(repo, "t2a") == "pb study 2b"

    # Finish t2b → all done
    t2b.completion = 100
    assert _next_session_recommendation(repo, "t2b") is None
