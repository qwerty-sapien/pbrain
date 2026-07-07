"""Integration tests for the simplified capture -> start workflow.

Tests that captured tasks are ACTIVE and can be started immediately.
"""

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.domain.enums import TaskState
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository


runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path):
    """Set up CLI test environment."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)

    config = Config(
        general=GeneralConfig(vault_path=str(vault_path)),
        storage=StorageConfig(data_dir=str(tmp_path)),
    )
    config_module._config = config

    yield tmp_path

    config_module._config = None
    db_module._db_path = None


class TestCaptureCreatesActive:
    """Captured tasks are ACTIVE immediately."""

    def test_capture_creates_active_task(self, cli_env):
        """Default capture creates task in ACTIVE state."""
        result = runner.invoke(app, ["add", "Test task"])
        assert result.exit_code == 0
        assert "Added" in result.output

        repo = Repository()
        tasks = repo.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].state == TaskState.ACTIVE


class TestCaptureAndStart:
    """Capture -> start workflow."""

    def test_start_captured_task_directly(self, cli_env):
        """Captured tasks can be started immediately."""
        result = runner.invoke(app, ["add", "Quick task"])
        assert result.exit_code == 0

        repo = Repository()
        task = repo.list_tasks()[0]
        assert task.state == TaskState.ACTIVE

        result = runner.invoke(app, ["start", task.id[:8]])
        assert result.exit_code == 0, f"Failed with: {result.output}"
        assert "Started" in result.output

    def test_capture_and_start_workflow(self, cli_env):
        """Complete capture -> start workflow works."""
        result = runner.invoke(app, ["add", "Quick task"])
        task_id = result.output.split()[1].rstrip("...")

        result = runner.invoke(app, ["start", task_id])
        assert result.exit_code == 0
        assert "Started" in result.output
