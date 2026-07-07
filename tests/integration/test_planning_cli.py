"""Integration tests for pb plan week and pb plan day CLI commands (Plan 02-08).

Tests:
- pb plan week --hours N generates capacity-aware weekly plan
- pb plan day --quick generates energy-aware daily plan without prompts
- pb plan day --quick --budget 4h includes budget section
- pb plan day --budget abc exits with code 1
- pb plan week --hours 0 exits with code 1
- Interactive day plan falls back to defaults on non-interactive input
"""

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.domain.enums import TaskState, WorkType
from pb.domain.models import Task
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import DB_FILENAME, init_db, set_db_path
from pb.storage.repository import Repository


runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path):
    """Set up CLI test environment with config and DB."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    data_dir = tmp_path / "data"
    db_path = data_dir / DB_FILENAME
    set_db_path(db_path)
    init_db(db_path)

    config = Config(
        general=GeneralConfig(vault_path=str(vault_path)),
        storage=StorageConfig(data_dir=str(data_dir)),
    )
    config_module._config = config

    yield tmp_path

    config_module._config = None
    db_module._db_path = None


@pytest.fixture
def cli_env_with_tasks(cli_env):
    """CLI env pre-populated with scored tasks."""
    repo = Repository()
    for i in range(3):
        task = Task(
            title=f"Scored task {i + 1}",
            impact=4,
            urgency_score=4,
            strategic_value=4,
            effort=2,
            important=True,
            urgent=True,
            energy_required=3,
            work_type=WorkType.DEEP.value,
            estimated_minutes=60,
            state=TaskState.ACTIVE,
        )
        repo.create_task(task)
    # Add one shallow/admin task
    admin_task = Task(
        title="Admin task",
        impact=2,
        urgency_score=2,
        strategic_value=2,
        effort=2,
        important=False,
        urgent=True,
        energy_required=1,
        work_type=WorkType.ADMIN.value,
        estimated_minutes=30,
        state=TaskState.ACTIVE,
    )
    repo.create_task(admin_task)
    yield cli_env


class TestWeekPlan:
    def test_week_plan_with_hours_flag(self, cli_env):
        """pb plan week --hours 40 outputs weekly plan."""
        result = runner.invoke(app, ["plan", "week", "--hours", "40"])
        assert result.exit_code == 0, result.output
        assert "Weekly Plan" in result.output
        assert "60%" in result.output

    def test_week_plan_shows_deep_work_section(self, cli_env):
        result = runner.invoke(app, ["plan", "week", "--hours", "40"])
        assert result.exit_code == 0
        assert "Deep work" in result.output

    def test_week_plan_shows_allocation_numbers(self, cli_env):
        result = runner.invoke(app, ["plan", "week", "--hours", "40"])
        assert result.exit_code == 0
        # 40h is shown in the available line
        assert "40" in result.output

    def test_week_plan_custom_hours(self, cli_env):
        result = runner.invoke(app, ["plan", "week", "--hours", "20"])
        assert result.exit_code == 0
        assert "20" in result.output

    def test_week_plan_accepts_flexible_duration_hours(self, cli_env):
        result = runner.invoke(app, ["plan", "week", "--hours", "1h 10min"])
        assert result.exit_code == 0
        assert "Weekly Plan" in result.output

    def test_week_plan_no_placeholder(self, cli_env):
        """Ensure placeholder message is gone."""
        result = runner.invoke(app, ["plan", "week", "--hours", "40"])
        assert "coming in v1.5" not in result.output

    def test_week_plan_zero_hours_fails(self, cli_env):
        result = runner.invoke(app, ["plan", "week", "--hours", "0"])
        assert result.exit_code == 1

    def test_week_plan_with_tasks(self, cli_env_with_tasks):
        result = runner.invoke(app, ["plan", "week", "--hours", "40"])
        assert result.exit_code == 0
        assert "Scored task" in result.output


class TestDayPlan:
    def test_day_plan_quick_mode(self, cli_env):
        """pb plan day --quick exits 0 and shows all sections."""
        result = runner.invoke(app, ["plan", "day", "--quick"])
        assert result.exit_code == 0, result.output
        assert "Daily Plan" in result.output
        assert "Top 1" in result.output
        assert "Top 3" in result.output
        assert "Batch List" in result.output

    def test_day_plan_quick_shows_suggested_blocks(self, cli_env):
        result = runner.invoke(app, ["plan", "day", "--quick"])
        assert result.exit_code == 0
        assert "Suggested Blocks" in result.output

    def test_day_plan_skip_scoring_alias_uses_quick_mode(self, cli_env):
        result = runner.invoke(app, ["plan", "day", "--skip-scoring"])
        assert result.exit_code == 0
        assert "Daily Plan" in result.output

    def test_day_plan_quick_with_budget(self, cli_env):
        """--budget flag adds Budget section."""
        result = runner.invoke(app, ["plan", "day", "--quick", "--budget", "4h"])
        assert result.exit_code == 0
        assert "Budget" in result.output

    def test_day_plan_quick_budget_hours_and_minutes(self, cli_env):
        result = runner.invoke(app, ["plan", "day", "--quick", "--budget", "2h30m"])
        assert result.exit_code == 0
        assert "Budget" in result.output

    def test_day_plan_accepts_flexible_hours_input(self, cli_env):
        result = runner.invoke(app, ["plan", "day", "--quick", "--hours", "1h 10min"])
        assert result.exit_code == 0
        assert "Budget" in result.output or "Daily Plan" in result.output

    def test_day_plan_invalid_budget_exits_1(self, cli_env):
        """Invalid budget string causes exit code 1."""
        result = runner.invoke(app, ["plan", "day", "--budget", "abc"])
        assert result.exit_code == 1

    def test_day_plan_interactive_fallback_on_eof(self, cli_env):
        """Without --quick, EOF input falls back to defaults and still outputs plan."""
        result = runner.invoke(app, ["plan", "day"], input="\n\n\n\n")
        assert result.exit_code == 0
        assert "Daily Plan" in result.output

    def test_day_plan_with_tasks_shows_task_in_top1(self, cli_env_with_tasks):
        result = runner.invoke(app, ["plan", "day", "--quick"])
        assert result.exit_code == 0
        assert "Day Plan Draft" in result.output or "Daily Plan" in result.output
        assert "Scored task" in result.output or "Admin task" in result.output

    def test_day_plan_short_flag(self, cli_env):
        """Short flag -q also works."""
        result = runner.invoke(app, ["plan", "day", "-q"])
        assert result.exit_code == 0
        assert "Daily Plan" in result.output
