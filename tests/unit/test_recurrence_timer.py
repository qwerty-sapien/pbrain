"""Tests for TimeBlock recurrence support and timer auto-trigger (Plan 02-02).

Covers:
- TimeBlock model with series_id and recurrence_rule fields (D-10)
- Database migration idempotency for time_blocks recurrence columns
- Repository persistence and reading of recurrence fields
- Planner.generate_recurrence_instances() for daily/weekly rules
- Planner.fork_series() for "edit this and future" semantics
- TimerManager auto-trigger via at-job scheduling (_schedule_auto_finish / _cancel_auto_finish) (D-11)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pb.domain.models import TimeBlock, generate_internal_id
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage import config as config_module
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository
from pb.core.planner import Planner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_dir(tmp_path):
    """Set up a temporary database for each test."""
    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)

    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    config = Config(
        general=GeneralConfig(vault_path=str(vault_path)),
        storage=StorageConfig(data_dir=str(tmp_path)),
    )
    config_module._config = config

    yield db_path

    db_module._db_path = None
    config_module._config = None


@pytest.fixture
def repo(temp_db_dir):
    return Repository()


@pytest.fixture
def planner(repo):
    return Planner(repo)


@pytest.fixture
def persisted_task(repo):
    """Return a persisted task to use as parent for time blocks."""
    from pb.domain.models import Task
    from pb.domain.enums import TaskState, Horizon
    task = Task(title="Recurrence test task", state=TaskState.ACTIVE, horizon=Horizon.TODAY)
    return repo.create_task(task)


# ===========================================================================
# Task 1a: TimeBlock model — series_id and recurrence_rule fields
# ===========================================================================

class TestTimeBlockModel:
    def test_timeblock_defaults_series_id_none(self):
        block = TimeBlock(task_id="x", duration_minutes=30)
        assert block.series_id is None

    def test_timeblock_defaults_recurrence_rule_none(self):
        block = TimeBlock(task_id="x", duration_minutes=30)
        assert block.recurrence_rule is None

    def test_timeblock_with_series_id_and_recurrence_rule(self):
        block = TimeBlock(task_id="x", duration_minutes=30, series_id="ABC", recurrence_rule="daily")
        assert block.series_id == "ABC"
        assert block.recurrence_rule == "daily"

    def test_timeblock_series_id_is_optional_str(self):
        block = TimeBlock(task_id="x", duration_minutes=30, series_id="SID-123")
        assert isinstance(block.series_id, str)

    def test_timeblock_recurrence_rule_accepts_weekly(self):
        block = TimeBlock(task_id="x", duration_minutes=60, recurrence_rule="weekly")
        assert block.recurrence_rule == "weekly"


# ===========================================================================
# Task 1b: Database migration — _migrate_time_blocks_recurrence
# ===========================================================================

class TestMigrateTimeBlocksRecurrence:
    def test_migration_function_exists(self):
        from pb.storage.database import _migrate_time_blocks_recurrence
        assert callable(_migrate_time_blocks_recurrence)

    def test_migration_adds_columns(self, temp_db_dir):
        import sqlite3
        conn = sqlite3.connect(str(temp_db_dir))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(time_blocks)").fetchall()}
            assert "series_id" in cols
            assert "recurrence_rule" in cols
        finally:
            conn.close()

    def test_migration_is_idempotent(self, temp_db_dir):
        import sqlite3
        from pb.storage.database import _migrate_time_blocks_recurrence
        conn = sqlite3.connect(str(temp_db_dir))
        try:
            # Run migration a second time — should not raise
            _migrate_time_blocks_recurrence(conn)
            _migrate_time_blocks_recurrence(conn)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(time_blocks)").fetchall()}
            assert "series_id" in cols
            assert "recurrence_rule" in cols
        finally:
            conn.close()

    def test_init_db_calls_migration(self, tmp_path):
        """init_db() must call _migrate_time_blocks_recurrence so fresh DBs have the columns."""
        import sqlite3
        db_path = tmp_path / "fresh.db"
        init_db(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(time_blocks)").fetchall()}
            assert "series_id" in cols, "series_id column missing after init_db"
            assert "recurrence_rule" in cols, "recurrence_rule column missing after init_db"
        finally:
            conn.close()
            db_module._db_path = None


# ===========================================================================
# Task 1c: Repository — persistence and reading of recurrence fields
# ===========================================================================

class TestRepositoryRecurrenceFields:
    def test_create_time_block_persists_series_id(self, repo, persisted_task):
        block = TimeBlock(
            task_id=persisted_task.id,
            duration_minutes=30,
            series_id="SID-001",
        )
        repo.create_time_block(block)
        fetched = repo.get_time_block(block.id)
        assert fetched is not None
        assert fetched.series_id == "SID-001"

    def test_create_time_block_persists_recurrence_rule(self, repo, persisted_task):
        block = TimeBlock(
            task_id=persisted_task.id,
            duration_minutes=30,
            recurrence_rule="daily",
        )
        repo.create_time_block(block)
        fetched = repo.get_time_block(block.id)
        assert fetched is not None
        assert fetched.recurrence_rule == "daily"

    def test_create_time_block_null_recurrence_fields(self, repo, persisted_task):
        block = TimeBlock(task_id=persisted_task.id, duration_minutes=45)
        repo.create_time_block(block)
        fetched = repo.get_time_block(block.id)
        assert fetched is not None
        assert fetched.series_id is None
        assert fetched.recurrence_rule is None

    def test_row_to_time_block_reads_series_id(self, repo, persisted_task):
        sid = generate_internal_id()
        block = TimeBlock(task_id=persisted_task.id, duration_minutes=60, series_id=sid)
        repo.create_time_block(block)
        fetched = repo.get_time_block(block.id)
        assert fetched.series_id == sid

    def test_update_time_block_persists_series_id(self, repo, persisted_task):
        block = TimeBlock(task_id=persisted_task.id, duration_minutes=30)
        repo.create_time_block(block)
        block.series_id = "UPDATED-SID"
        block.recurrence_rule = "weekly"
        repo.update_time_block(block)
        fetched = repo.get_time_block(block.id)
        assert fetched.series_id == "UPDATED-SID"
        assert fetched.recurrence_rule == "weekly"

    def test_list_time_blocks_by_series(self, repo, persisted_task):
        """list_time_blocks_by_series returns all blocks sharing a series_id."""
        sid = generate_internal_id()
        now = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
        b1 = TimeBlock(task_id=persisted_task.id, duration_minutes=30, series_id=sid, start_time=now)
        b2 = TimeBlock(task_id=persisted_task.id, duration_minutes=30, series_id=sid, start_time=now + timedelta(days=1))
        b3 = TimeBlock(task_id=persisted_task.id, duration_minutes=30, series_id="OTHER")
        repo.create_time_block(b1)
        repo.create_time_block(b2)
        repo.create_time_block(b3)
        result = repo.list_time_blocks_by_series(sid)
        assert len(result) == 2
        ids = {b.id for b in result}
        assert b1.id in ids
        assert b2.id in ids
        assert b3.id not in ids


# ===========================================================================
# Task 1d: Planner — generate_recurrence_instances
# ===========================================================================

class TestGenerateRecurrenceInstances:
    def _parent_block(self, task_id: str, rule: str) -> TimeBlock:
        """Helper: parent block starting at 9:00 today with given rule."""
        now = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
        sid = generate_internal_id()
        return TimeBlock(
            task_id=task_id,
            start_time=now,
            duration_minutes=30,
            series_id=sid,
            recurrence_rule=rule,
        )

    def test_daily_rule_produces_7_instances(self, planner, repo, persisted_task):
        parent = self._parent_block(persisted_task.id, "daily")
        repo.create_time_block(parent)
        instances = planner.generate_recurrence_instances(parent, days_ahead=7)
        assert len(instances) == 7

    def test_weekly_rule_produces_1_instance_in_7_days(self, planner, repo, persisted_task):
        parent = self._parent_block(persisted_task.id, "weekly")
        repo.create_time_block(parent)
        instances = planner.generate_recurrence_instances(parent, days_ahead=7)
        assert len(instances) == 1

    def test_no_recurrence_rule_returns_empty(self, planner, repo, persisted_task):
        parent = TimeBlock(task_id=persisted_task.id, duration_minutes=30)
        repo.create_time_block(parent)
        instances = planner.generate_recurrence_instances(parent, days_ahead=7)
        assert instances == []

    def test_instances_share_parent_series_id(self, planner, repo, persisted_task):
        parent = self._parent_block(persisted_task.id, "daily")
        repo.create_time_block(parent)
        instances = planner.generate_recurrence_instances(parent, days_ahead=3)
        for inst in instances:
            assert inst.series_id == parent.series_id

    def test_instances_have_unique_ids(self, planner, repo, persisted_task):
        parent = self._parent_block(persisted_task.id, "daily")
        repo.create_time_block(parent)
        instances = planner.generate_recurrence_instances(parent, days_ahead=3)
        ids = [inst.id for inst in instances]
        assert len(ids) == len(set(ids))

    def test_instances_are_persisted_to_repo(self, planner, repo, persisted_task):
        parent = self._parent_block(persisted_task.id, "daily")
        repo.create_time_block(parent)
        instances = planner.generate_recurrence_instances(parent, days_ahead=3)
        for inst in instances:
            fetched = repo.get_time_block(inst.id)
            assert fetched is not None, f"Instance {inst.id} not persisted"

    def test_daily_instances_skip_past_dates(self, planner, repo, persisted_task):
        """Instances must be in the future (days 1-N from today), never today itself."""
        parent = self._parent_block(persisted_task.id, "daily")
        repo.create_time_block(parent)
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        instances = planner.generate_recurrence_instances(parent, days_ahead=7)
        for inst in instances:
            if inst.start_time is not None:
                assert inst.start_time > today_start, (
                    f"Instance start {inst.start_time} is not in the future"
                )

    def test_generate_preserves_parent_duration(self, planner, repo, persisted_task):
        parent = self._parent_block(persisted_task.id, "daily")
        parent.duration_minutes = 45
        repo.create_time_block(parent)
        instances = planner.generate_recurrence_instances(parent, days_ahead=2)
        for inst in instances:
            assert inst.duration_minutes == 45


# ===========================================================================
# Task 1e: Planner — fork_series
# ===========================================================================

class TestForkSeries:
    def test_fork_series_returns_new_series_id(self, planner, repo, persisted_task):
        sid = generate_internal_id()
        now = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
        b1 = TimeBlock(task_id=persisted_task.id, duration_minutes=30, series_id=sid, start_time=now)
        b2 = TimeBlock(task_id=persisted_task.id, duration_minutes=30, series_id=sid, start_time=now + timedelta(days=1))
        repo.create_time_block(b1)
        repo.create_time_block(b2)

        new_sid = planner.fork_series(b2)
        assert new_sid != sid

    def test_fork_series_updates_future_blocks_series_id(self, planner, repo, persisted_task):
        sid = generate_internal_id()
        now = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
        b1 = TimeBlock(task_id=persisted_task.id, duration_minutes=30, series_id=sid, start_time=now)
        b2 = TimeBlock(task_id=persisted_task.id, duration_minutes=30, series_id=sid, start_time=now + timedelta(days=1))
        b3 = TimeBlock(task_id=persisted_task.id, duration_minutes=30, series_id=sid, start_time=now + timedelta(days=2))
        repo.create_time_block(b1)
        repo.create_time_block(b2)
        repo.create_time_block(b3)

        new_sid = planner.fork_series(b2)

        # b1 should keep original series_id
        fetched_b1 = repo.get_time_block(b1.id)
        assert fetched_b1.series_id == sid

        # b2 and b3 should have new series_id
        fetched_b2 = repo.get_time_block(b2.id)
        fetched_b3 = repo.get_time_block(b3.id)
        assert fetched_b2.series_id == new_sid
        assert fetched_b3.series_id == new_sid


# ===========================================================================
# Task 2: Timer auto-trigger — _schedule_auto_finish / _cancel_auto_finish
# ===========================================================================

class TestScheduleAutoFinish:
    def test_schedule_auto_finish_calls_at_command(self, tmp_path):
        """_schedule_auto_finish must invoke subprocess with 'at now + N minutes'."""
        from pb.core.timer import _schedule_auto_finish, AT_JOB_ID_FILE

        mock_proc = MagicMock()
        mock_proc.stderr = "job 42 at Sun Apr 26 09:30:00 2026"

        with patch("pb.core.timer.AT_JOB_ID_FILE", tmp_path / "at_job.id"):
            with patch("pb.core.timer.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="",
                    stderr="job 42 at Sun Apr 26 09:30:00 2026",
                    returncode=0,
                )
                _schedule_auto_finish(30)

                # First call should be 'which pb'
                calls = mock_run.call_args_list
                assert any(
                    "at" in str(c) for c in calls
                ), "Expected 'at' command in subprocess calls"

    def test_schedule_auto_finish_writes_job_id_file(self, tmp_path):
        """Job ID parsed from at stderr must be written to AT_JOB_ID_FILE."""
        from pb.core.timer import _schedule_auto_finish

        at_job_id_file = tmp_path / "at_job.id"

        with patch("pb.core.timer.AT_JOB_ID_FILE", at_job_id_file), \
             patch("pb.core.timer._has_terminal_notifier", return_value=False), \
             patch("pb.core.timer.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="", stderr="job 99 at Mon Apr 27 10:00:00 2026", returncode=0),
            ]
            _schedule_auto_finish(45)

            assert at_job_id_file.exists(), "AT_JOB_ID_FILE should be written"
            assert at_job_id_file.read_text().strip() == "99"

    def test_schedule_auto_finish_graceful_on_failure(self, tmp_path):
        """If subprocess raises, _schedule_auto_finish should not propagate the exception."""
        from pb.core.timer import _schedule_auto_finish

        at_job_id_file = tmp_path / "at_job.id"
        with patch("pb.core.timer.AT_JOB_ID_FILE", at_job_id_file):
            with patch("pb.core.timer.subprocess.run", side_effect=FileNotFoundError("no at")):
                # Should not raise
                _schedule_auto_finish(30)


class TestCancelAutoFinish:
    def test_cancel_auto_finish_calls_atrm(self, tmp_path):
        """_cancel_auto_finish must call atrm with the stored job ID."""
        from pb.core.timer import _cancel_auto_finish

        at_job_id_file = tmp_path / "at_job.id"
        at_job_id_file.write_text("42")

        with patch("pb.core.timer.AT_JOB_ID_FILE", at_job_id_file):
            with patch("pb.core.timer.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                _cancel_auto_finish()
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "atrm" in args
                assert "42" in args

    def test_cancel_auto_finish_removes_job_id_file(self, tmp_path):
        """_cancel_auto_finish must delete AT_JOB_ID_FILE after calling atrm."""
        from pb.core.timer import _cancel_auto_finish

        at_job_id_file = tmp_path / "at_job.id"
        at_job_id_file.write_text("55")

        with patch("pb.core.timer.AT_JOB_ID_FILE", at_job_id_file):
            with patch("pb.core.timer.subprocess.run", return_value=MagicMock(returncode=0)):
                _cancel_auto_finish()
                assert not at_job_id_file.exists()

    def test_cancel_auto_finish_no_op_when_no_file(self, tmp_path):
        """_cancel_auto_finish must not raise when AT_JOB_ID_FILE doesn't exist."""
        from pb.core.timer import _cancel_auto_finish

        at_job_id_file = tmp_path / "at_job.id"
        assert not at_job_id_file.exists()

        with patch("pb.core.timer.AT_JOB_ID_FILE", at_job_id_file):
            with patch("pb.core.timer.subprocess.run") as mock_run:
                _cancel_auto_finish()
                mock_run.assert_not_called()


class TestTimerManagerAutoFinishIntegration:
    """Verify TimerManager integrates _schedule_auto_finish / _cancel_auto_finish."""

    def test_start_session_timers_stores_duration(self):
        from pb.core.timer import TimerManager

        manager = TimerManager()
        manager.start_session_timers("sess-1", 30, "My Task")
        assert manager.state is not None
        assert manager.state.duration_minutes == 30

    def test_start_session_timers_does_not_schedule_when_no_duration(self):
        from pb.core.timer import TimerManager

        manager = TimerManager()
        with patch("pb.core.timer._schedule_auto_finish") as mock_schedule, \
             patch("pb.core.timer.send_notification"), \
             patch.object(manager.break_reminder, "start"), \
             patch.object(manager.caffeinate, "start"):
            manager.start_session_timers("sess-1", None, "My Task")
            mock_schedule.assert_not_called()

    def test_stop_session_timers_calls_cancel(self):
        from pb.core.timer import TimerManager

        manager = TimerManager()
        with patch("pb.core.timer._cancel_auto_finish") as mock_cancel, \
             patch.object(manager.break_reminder, "stop"), \
             patch.object(manager.caffeinate, "stop"):
            manager.stop_session_timers()
            mock_cancel.assert_called_once()
