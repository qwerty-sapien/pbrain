from unittest.mock import MagicMock, patch


def test_compact_scoring_3_questions():
    from pb.cli.task_scoring import score_task_interactively

    repo = MagicMock()
    task = MagicMock()
    task.title = "Study German"
    task.description = ""
    task.impact = None
    task.urgency_score = None
    task.effort = None
    task.linked_goal_arc_ids = []
    task.generated_names = {}

    calls = []

    def mock_prompt(label, default=""):
        calls.append(label)
        return "3"

    with patch("pb.cli.task_scoring.prompt_text", side_effect=mock_prompt), \
         patch("pb.cli.task_scoring.get_console", return_value=MagicMock()):
        result = score_task_interactively(repo, task)

    assert result is True
    assert len(calls) == 3
    assert task.important is True
    assert task.urgent is True
    assert task.strategic_value == 3
    assert task.energy_required == 3


def test_low_impact_not_important():
    from pb.cli.task_scoring import score_task_interactively

    repo = MagicMock()
    task = MagicMock()
    task.title = "Low priority"
    task.description = ""
    task.impact = None
    task.urgency_score = None
    task.effort = None
    task.linked_goal_arc_ids = []
    task.generated_names = {}

    answers = iter(["2", "1", "4"])

    with patch("pb.cli.task_scoring.prompt_text", side_effect=lambda *a, **kw: next(answers)), \
         patch("pb.cli.task_scoring.get_console", return_value=MagicMock()):
        score_task_interactively(repo, task)

    assert task.important is False
    assert task.urgent is False
