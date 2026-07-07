from unittest.mock import MagicMock, patch
from io import StringIO
from rich.console import Console

from pb.cli.commands.plan import _render_plan_rows


def test_render_plan_rows_column_headers():
    block = MagicMock()
    block.branch = "study"
    block.duration_minutes = 60
    block.subject_scope = "German: Verb Conjugation, Case Usage"
    block.title = "German Conjugation"
    block.sub_index = None

    task = MagicMock()
    task.title = "Study: German [Consolidate]"

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=200)

    with patch("pb.cli.commands.plan.get_console", return_value=console):
        _render_plan_rows([(block, task)], 60)

    rendered = output.getvalue()
    assert "MODE" in rendered
    assert "BRANCH" not in rendered
    assert "German Conjugation" in rendered


def test_render_plan_rows_code_column():
    block = MagicMock()
    block.branch = "study"
    block.duration_minutes = 30
    block.subject_scope = "Calc core"
    block.title = "Calc Core"
    block.sub_index = "2a"

    task = MagicMock()
    task.title = "Study: Math"

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=200)

    with patch("pb.cli.commands.plan.get_console", return_value=console):
        _render_plan_rows([(block, task)], 30)

    rendered = output.getvalue()
    assert "2a" in rendered


def test_render_plan_rows_budget_wording_and_long_text_not_ellipsized():
    block = MagicMock()
    block.branch = "study"
    block.duration_minutes = 30
    block.subject_scope = "A very long conceptual scope about Rust async cancellation and cooperative task shutdown"
    block.title = "Trace cancellation from request drop through cleanup"
    block.sub_index = "1a"

    task = MagicMock()
    task.title = "Study: Rust async cancellation"

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=90)

    with patch("pb.cli.commands.plan.get_console", return_value=console):
        _render_plan_rows([(block, task)], 45)

    rendered = output.getvalue()
    assert "budget 45 min" in rendered
    assert "Planned 30 of 45 budgeted minutes" in rendered
    assert "..." not in rendered
