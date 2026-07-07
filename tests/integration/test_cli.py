"""Integration tests for CLI commands.

Tests CLI invocation and exit codes.
"""

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import init_db, set_db_path


runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path):
    """Set up CLI test environment."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    config_path = tmp_path / "config.toml"
    config_path.write_text(f'''
[general]
vault_path = "{vault_path}"

[storage]
data_dir = "{tmp_path}"
''')

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


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_help(self, cli_env):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "pb" in result.output

    def test_capture(self, cli_env):
        result = runner.invoke(app, ["add", "Test task"])
        assert result.exit_code == 0
        assert "Added" in result.output

    def test_add_creates_active_task(self, cli_env):
        """Added tasks are ACTIVE immediately."""
        runner.invoke(app, ["add", "Task 1"])
        from pb.storage.repository import Repository
        repo = Repository()
        tasks = repo.list_tasks()
        assert len(tasks) == 1
        from pb.domain.enums import TaskState
        assert tasks[0].state == TaskState.ACTIVE


class TestExecutionCommands:
    """Test start/pause/finish commands."""

    def test_start_active_task(self, cli_env):
        """Captured tasks are ACTIVE and can be started immediately."""
        result = runner.invoke(app, ["add", "Test"])
        task_id = result.output.split()[1].rstrip("...")

        result = runner.invoke(app, ["start", task_id])
        assert result.exit_code == 0
        assert "Started" in result.output

    def test_now_shows_active(self, cli_env):
        runner.invoke(app, ["add", "Active task"])

        from pb.storage.repository import Repository
        repo = Repository()
        tasks = repo.list_tasks()
        task_id = tasks[0].id

        runner.invoke(app, ["start", task_id[:8]])

        result = runner.invoke(app, ["now"])
        assert "Active task" in result.output

    def test_pause(self, cli_env):
        runner.invoke(app, ["add", "Pause test"])

        from pb.storage.repository import Repository
        repo = Repository()
        tasks = repo.list_tasks()
        task_id = tasks[0].id

        runner.invoke(app, ["start", task_id[:8]])

        result = runner.invoke(app, ["pause", "--note", "Break time"])
        assert result.exit_code == 0
        assert "Paused" in result.output


class TestPlanningCommands:
    """Test planning commands."""

    def test_plan_day(self, cli_env):
        runner.invoke(app, ["goal", "add", "--yes", "Plan test"])

        result = runner.invoke(app, ["plan", "day", "--quick", "--yes"])

        assert result.exit_code == 0
        assert "Day Plan" in result.output
        assert "pb study plan" in result.output

    def test_block_list(self, cli_env):
        result = runner.invoke(app, ["plan", "block", "list"])
        assert result.exit_code == 0


class TestBlockCommands:
    """Test block subcommands (add, list, rm, edit)."""

    def _create_ready_task(self):
        """Helper to create a ready task and return its ID prefix."""
        from pb.storage.repository import Repository
        runner.invoke(app, ["add", "Block test task"])
        repo = Repository()
        task = repo.list_tasks()[0]
        return task.id[:8]

    def test_block_add_basic(self, cli_env):
        """Test adding a block with standard HH:MM format."""
        task_prefix = self._create_ready_task()
        result = runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])
        assert result.exit_code == 0
        assert "Scheduled" in result.output
        assert "09:00" in result.output
        assert "60m" in result.output

    def test_block_add_flexible_hhmm(self, cli_env):
        """Test adding a block with HHMM format (D-01)."""
        task_prefix = self._create_ready_task()
        result = runner.invoke(app, ["plan", "block", "add", task_prefix, "0900", "60"])
        assert result.exit_code == 0
        assert "Scheduled" in result.output
        assert "09:00" in result.output

    def test_block_add_flexible_hmm(self, cli_env):
        """Test adding a block with H:MM format (D-01)."""
        task_prefix = self._create_ready_task()
        result = runner.invoke(app, ["plan", "block", "add", task_prefix, "9:00", "60"])
        assert result.exit_code == 0
        assert "Scheduled" in result.output

    def test_block_add_invalid_format(self, cli_env):
        """Test that invalid time format is rejected with actionable error (D-02)."""
        task_prefix = self._create_ready_task()
        result = runner.invoke(app, ["plan", "block", "add", task_prefix, "25:00", "60"])
        assert result.exit_code == 1
        assert "Invalid time format" in result.output
        assert "HH:MM" in result.output

    def test_block_add_overlap_warns(self, cli_env):
        """Test that overlapping blocks emit warning but succeed (D-03, D-04)."""
        task_prefix = self._create_ready_task()

        result1 = runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])
        assert result1.exit_code == 0

        result2 = runner.invoke(app, ["plan", "block", "add", task_prefix, "09:30", "60"])
        assert result2.exit_code == 0
        assert "Note: overlaps with existing block" in result2.output
        assert "09:00-10:00" in result2.output

    def test_block_list_shows_numbered_entries(self, cli_env):
        """Test that block list shows numbered entries per D-09 UX redesign."""
        import re
        task_prefix = self._create_ready_task()
        runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])

        result = runner.invoke(app, ["plan", "block", "list"])
        assert result.exit_code == 0
        # Per D-09: numbered selection format (Rich Table column, no parens)
        assert re.search(r"\d+\s+\d{2}:\d{2}", result.output)

    def test_block_list_shows_end_time(self, cli_env):
        """Test that block list shows HH:MM-HH:MM time range (BLCK-02)."""
        task_prefix = self._create_ready_task()
        runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])

        result = runner.invoke(app, ["plan", "block", "list"])
        assert result.exit_code == 0
        assert "09:00-10:00" in result.output

    def test_block_rm_removes_block(self, cli_env):
        """Test removing a block by number (D-05, D-09)."""
        task_prefix = self._create_ready_task()
        runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])

        # Per D-09: use numbered selection (block 1)
        rm_result = runner.invoke(app, ["plan", "block", "rm", "1"])
        assert rm_result.exit_code == 0
        assert "Removed:" in rm_result.output

        final_list = runner.invoke(app, ["plan", "block", "list"])
        assert "No blocks scheduled" in final_list.output

    def test_block_rm_invalid_number(self, cli_env):
        """Test that rm with invalid number fails gracefully (D-09)."""
        task_prefix = self._create_ready_task()
        runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])

        result = runner.invoke(app, ["plan", "block", "rm", "99"])
        assert result.exit_code == 1
        assert "Invalid block number" in result.output

    def test_block_edit_start(self, cli_env):
        """Test editing block start time by number (D-06, D-07, D-09)."""
        task_prefix = self._create_ready_task()
        runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])

        # Per D-09: use numbered selection (block 1)
        edit_result = runner.invoke(app, ["plan", "block", "edit", "1", "--start", "10:00"])
        assert edit_result.exit_code == 0
        assert "Updated:" in edit_result.output
        assert "10:00" in edit_result.output

        final_list = runner.invoke(app, ["plan", "block", "list"])
        assert "10:00-11:00" in final_list.output

    def test_block_edit_duration(self, cli_env):
        """Test editing block duration by number (D-06, D-07, D-09)."""
        task_prefix = self._create_ready_task()
        runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])

        # Per D-09: use numbered selection (block 1)
        edit_result = runner.invoke(app, ["plan", "block", "edit", "1", "--duration", "90"])
        assert edit_result.exit_code == 0
        assert "Updated:" in edit_result.output
        assert "90m" in edit_result.output

        final_list = runner.invoke(app, ["plan", "block", "list"])
        assert "09:00-10:30" in final_list.output
        assert "90m" in final_list.output

    def test_block_edit_invalid_number(self, cli_env):
        """Test that edit with invalid block number fails gracefully (D-09)."""
        task_prefix = self._create_ready_task()
        runner.invoke(app, ["plan", "block", "add", task_prefix, "09:00", "60"])

        edit_result = runner.invoke(app, ["plan", "block", "edit", "99", "--start", "10:00"])
        assert edit_result.exit_code == 1
        assert "Invalid block number" in edit_result.output


class TestReviewCommands:
    """Test review commands."""

    def test_review_day(self, cli_env):
        result = runner.invoke(app, ["review", "day", "--skip"])

        assert result.exit_code == 0
        assert "Daily Review" in result.output

    def test_review_week(self, cli_env):
        result = runner.invoke(app, ["review", "week"])

        assert result.exit_code == 0
        # Default mode now shows structured reflection; legacy mode shows old table
        assert "Weekly Reflection" in result.output or "Weekly Review" in result.output


class TestGoalsCommands:
    """Test goals and tracks commands."""

    def test_goals_empty(self, cli_env):
        result = runner.invoke(app, ["goal"])
        assert result.exit_code == 0
        assert "No goals" in result.output

    def test_goals_add(self, cli_env):
        result = runner.invoke(app, ["goal", "add", "--yes", "Learn German"])
        assert result.exit_code == 0
        assert "Created goal" in result.output

    def test_tracks_empty(self, cli_env):
        result = runner.invoke(app, ["goal", "tracks"])
        assert result.exit_code == 0
        assert "No tracks" in result.output

    def test_track_add(self, cli_env):
        result = runner.invoke(app, ["goal", "track", "add", "German"])
        assert result.exit_code == 0
        assert "Created track" in result.output


class TestExitCodes:
    """Test exit codes per spec."""

    def test_success_exit_code(self, cli_env):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_user_error_exit_code(self, cli_env):
        from pb.domain.exceptions import ExitCode
        result = runner.invoke(app, ["start", "nonexistent"])
        assert result.exit_code == ExitCode.NOT_FOUND

    def test_invalid_outcome_exit_code(self, cli_env):
        # Phase 23: --outcome flag removed from pb finish.
        # Passing it now produces exit code 2 (typer unknown option).
        runner.invoke(app, ["add", "Test"])

        from pb.storage.repository import Repository
        repo = Repository()
        tasks = repo.list_tasks()
        task_id = tasks[0].id

        runner.invoke(app, ["start", task_id[:8]])

        result = runner.invoke(app, ["finish", "--outcome", "invalid"])
        # --outcome is an unrecognised flag (removed); typer returns exit 2
        assert result.exit_code == 2


class TestInitCommand:
    """Test pb init command - covers ONBD-01, ONBD-03."""

    @staticmethod
    def _init_input(*lines: str) -> str:
        return "\n".join(lines) + "\n"

    @staticmethod
    def _setup_init_env(tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        config_home = tmp_path / "config"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
        config_module._config = None
        return home, config_home

    def test_init_creates_config(self, tmp_path, monkeypatch):
        """ONBD-01: pb init creates config.toml at XDG path."""
        home, config_home = self._setup_init_env(tmp_path, monkeypatch)

        # Create a vault directory for the test
        vault = home / "vault"
        vault.mkdir()

        result = runner.invoke(
            app,
            ["init"],
            input=self._init_input("1", str(vault), "", "", "", "", ""),
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Config created" in result.output
        assert "Anki auto-open: not approved" in result.output
        assert "[main]" not in result.output
        assert "[gemini]" not in result.output
        assert "[y/N]" not in result.output
        assert "[Y/n]" not in result.output

        # Verify config file was created
        config_file = config_home / "productivebrain" / "config.toml"
        assert config_file.exists()

    def test_init_can_create_new_vault(self, tmp_path, monkeypatch):
        """pb init can create a new vault after choosing a parent directory."""
        home, config_home = self._setup_init_env(tmp_path, monkeypatch)

        result = runner.invoke(
            app,
            ["init"],
            input=self._init_input("3", "1", "CourseVault", "", "", "", "", ""),
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Config created" in result.output

        config_file = config_home / "productivebrain" / "config.toml"
        assert config_file.exists()
        assert (home / "CourseVault").exists()

    def test_init_existing_config_decline_recreate(self, tmp_path, monkeypatch):
        """ONBD-01: pb init with existing config, user declines recreate (default N)."""
        home, _ = self._setup_init_env(tmp_path, monkeypatch)

        vault = home / "vault"
        vault.mkdir()

        # First init - create config
        result1 = runner.invoke(
            app,
            ["init"],
            input=self._init_input("1", str(vault), "", "", "", "", ""),
        )
        assert result1.exit_code == 0

        # Clear config cache before second init
        config_module._config = None

        # Second init - press Enter to accept default N (decline recreate)
        result2 = runner.invoke(app, ["init"], input="\n")

        assert result2.exit_code == 0
        assert "No changes made" in result2.output

    def test_init_existing_config_accept_recreate(self, tmp_path, monkeypatch):
        """ONBD-01: pb init with existing config, user accepts recreate."""
        home, config_home = self._setup_init_env(tmp_path, monkeypatch)

        vault1 = home / "vault1"
        vault1.mkdir()
        vault2 = home / "vault2"
        vault2.mkdir()

        # First init with vault1
        result1 = runner.invoke(
            app,
            ["init"],
            input=self._init_input("1", str(vault1), "", "", "", "", ""),
        )
        assert result1.exit_code == 0

        config_module._config = None

        # Second init - accept recreate with Y, choose vault2 via the browser.
        result2 = runner.invoke(
            app,
            ["init"],
            input=self._init_input("y", "1", str(vault2), "", "", "", "", ""),
        )

        assert result2.exit_code == 0
        assert "Config created" in result2.output
        assert str(vault2) in result2.output

        config_file = config_home / "productivebrain" / "config.toml"
        assert config_file.exists()
        assert str(vault2) in config_file.read_text()

    def test_no_config_error_message(self, tmp_path, monkeypatch):
        """ONBD-03: any command without config shows actionable message."""
        # Point to non-existent config directory
        config_home = tmp_path / "no_config_here"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
        config_module._config = None

        from pb.domain.exceptions import ExitCode
        result = runner.invoke(app, ["add", "Test"])

        assert result.exit_code == ExitCode.CONFIG_ERROR
        assert "pb init" in result.output
        assert "Config not found" in result.output
