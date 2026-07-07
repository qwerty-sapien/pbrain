# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from unittest.mock import MagicMock
from pb.cli.commands.study import _todays_plan_blocks


def test_todays_plan_blocks_ordered():
    repo = MagicMock()

    block1 = MagicMock(task_id="t1", duration_minutes=60, sub_index=None)
    block2 = MagicMock(task_id="t2", duration_minutes=30, sub_index="2a")
    repo.list_time_blocks_created_for_date.return_value = [block1, block2]

    task1 = MagicMock(id="t1", archived_at=None, completion=0, title="German", description="")
    task2 = MagicMock(id="t2", archived_at=None, completion=0, title="Calc", description="")
    repo.get_task = lambda tid: {"t1": task1, "t2": task2}.get(tid)

    items = _todays_plan_blocks(repo)
    assert len(items) == 2
    assert items[0][0] == "1"
    assert items[1][0] == "2a"


def test_todays_plan_blocks_skips_completed():
    repo = MagicMock()

    block1 = MagicMock(task_id="t1", duration_minutes=60, sub_index=None)
    repo.list_time_blocks_created_for_date.return_value = [block1]

    task1 = MagicMock(id="t1", archived_at=None, completion=100, title="Done", description="")
    repo.get_task = lambda tid: task1

    items = _todays_plan_blocks(repo)
    assert len(items) == 0


def test_study_delete_by_codes():
    repo = MagicMock()
    b1 = MagicMock(task_id="t1", sub_index=None)
    b2 = MagicMock(task_id="t2", sub_index=None)
    b3 = MagicMock(task_id="t3", sub_index=None)
    repo.list_time_blocks_created_for_date.return_value = [b1, b2, b3]

    t1 = MagicMock(id="t1", archived_at=None, completion=0, title="A", description="")
    t2 = MagicMock(id="t2", archived_at=None, completion=0, title="B", description="")
    t3 = MagicMock(id="t3", archived_at=None, completion=0, title="C", description="")
    repo.get_task = lambda tid: {"t1": t1, "t2": t2, "t3": t3}[tid]

    items = _todays_plan_blocks(repo)
    codes_to_delete = {"2", "3"}
    target_ids = {t.id for code, t, _ in items if code in codes_to_delete}
    assert target_ids == {"t2", "t3"}
