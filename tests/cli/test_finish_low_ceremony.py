# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from unittest.mock import MagicMock

from pb.cli.commands.execute import _next_session_recommendation


def test_recommends_next_after_finish():
    repo = MagicMock()
    b1 = MagicMock(task_id="t1", sub_index=None)
    b2 = MagicMock(task_id="t2", sub_index=None)
    repo.list_time_blocks_created_for_date.return_value = [b1, b2]

    t1 = MagicMock(id="t1", archived_at=None, completion=100)
    t2 = MagicMock(id="t2", archived_at=None, completion=0)
    repo.get_task = lambda tid: {"t1": t1, "t2": t2}[tid]

    assert _next_session_recommendation(repo, "t1") == "pb study 2"


def test_recommends_sub_indexed_sibling():
    repo = MagicMock()
    ba = MagicMock(task_id="ta", sub_index="2a")
    bb = MagicMock(task_id="tb", sub_index="2b")
    repo.list_time_blocks_created_for_date.return_value = [ba, bb]

    ta = MagicMock(id="ta", archived_at=None, completion=100)
    tb = MagicMock(id="tb", archived_at=None, completion=0)
    repo.get_task = lambda tid: {"ta": ta, "tb": tb}[tid]

    assert _next_session_recommendation(repo, "ta") == "pb study 2b"


def test_returns_none_all_complete():
    repo = MagicMock()
    b1 = MagicMock(task_id="t1", sub_index=None)
    repo.list_time_blocks_created_for_date.return_value = [b1]

    t1 = MagicMock(id="t1", archived_at=None, completion=100)
    repo.get_task = lambda tid: t1

    assert _next_session_recommendation(repo, "t1") is None
