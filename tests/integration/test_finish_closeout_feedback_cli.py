from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from pb.cli.main import app
from pb.core.models import Session, Task
from pb.core.enums import SessionMode, TaskState
from pb.storage.repository import Repository


def test_finish_no_progress_captures_feedback_without_raw_json(temp_db, temp_config):
    runner = CliRunner()
    repo = Repository()
    task = Task(title="German conjugation recovery", state=TaskState.ACTIVE, work_type="study")
    repo.create_task(task)
    repo.create_session(
        Session(
            task_id=task.id,
            mode=SessionMode.FOCUS,
            branch="study",
            subject_scope="German conjugation",
        )
    )

    with patch("pb.sessions.service.TimerManager.stop_session_timers", return_value=None):
        result = runner.invoke(
            app,
            ["finish", "i", "didnt", "do", "anything,", "this", "tool", "is", "useless"],
        )

    assert result.exit_code == 0, result.output
    assert "Quick learning check-in" not in result.output
    assert '{"error":' not in result.output
    assert '"event":' not in result.output

    proposal_dir = Path(temp_config.vaults[temp_config.general.active_vault].path) / "direction" / "preferences" / "proposals"
    proposals = list(proposal_dir.glob("*.md"))
    assert proposals
    assert "frank_no_flattery" not in result.output
