from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

from pb.core.action_routing import CommandCandidate, build_next_candidates, suggest_commands_for_intent
from pb.core.models import Session, Task, TimeBlock


def test_do_routing_prefers_direct_intent_over_unrelated_recent_work(repo):
    task = Task(
        title="Study: Machine Learning: Scikit-Learn Model Implementation",
        description="Recent unrelated work.",
        work_type="study",
    )
    repo.create_task(task)
    repo.create_session(
        Session(
            task_id=task.id,
            branch="study",
            subject_scope="Machine Learning: Scikit-Learn Model Implementation",
            start_at=datetime.utcnow() - timedelta(hours=2),
            end_at=datetime.utcnow() - timedelta(hours=1),
        )
    )

    candidates = suggest_commands_for_intent(repo, "communication - how to speak with charisma", limit=5)

    assert candidates
    assert candidates[0].command.startswith("study ")
    assert "charisma" in candidates[0].command
    assert all("scikit" not in candidate.command.lower() for candidate in candidates)
    assert all(candidate.command != "plan day" for candidate in candidates)


def test_do_routing_skips_gemini_rerank_for_single_direct_candidate(repo):
    with patch(
        "pb.core.action_routing.rerank_candidates_with_gemini",
        side_effect=AssertionError("rerank should not be used for a single direct candidate"),
    ):
        candidates = suggest_commands_for_intent(repo, "communication - how to speak with charisma", limit=5)

    assert candidates
    assert candidates[0].command.startswith("study ")


def test_do_routing_explicit_practise_word_problems_beats_vocab(repo):
    candidates = suggest_commands_for_intent(
        repo,
        "I want to practise Bayes rule word problems, especially converting base-rate stories into equations",
        limit=5,
    )

    assert candidates
    assert candidates[0].command.startswith("practise ")
    assert "Bayes rule word problems" in candidates[0].command
    assert "I want to practise" not in candidates[0].command
    assert candidates[0].command != "study vocab"


def test_do_routing_moulds_lets_learn_into_study_scope(repo):
    candidates = suggest_commands_for_intent(repo, "lets learn multivariate calculus", limit=5)

    assert candidates
    assert candidates[0].command == "study 'multivariate calculus'"
    assert candidates[0].human_label == "Study multivariate calculus"


def test_do_routing_handles_apostrophe_without_unbalanced_shell_quote(repo):
    candidates = suggest_commands_for_intent(repo, "let's learn Euler's theorem", limit=5)

    assert candidates
    assert candidates[0].command == "study 'Euler'\"'\"'s theorem'"
    assert candidates[0].human_label == "Study Euler's theorem"


def test_study_debrief_label_does_not_show_shell_quote():
    candidate = CommandCandidate(
        command="study debrief 'multivariate calculus'",
        reason="Use a debrief if the learning block already happened.",
        score=0.76,
    )

    assert candidate.human_label == "Study debrief multivariate calculus"
    assert "Study debrief '" not in candidate.human_label


def test_next_candidates_skip_planned_block_recently_completed_by_scope(repo):
    task = Task(title="rust", description="Planned rust study block.", work_type="study")
    repo.create_task(task)
    repo.create_time_block(
        TimeBlock(
            task_id=task.id,
            start_time=datetime.utcnow(),
            duration_minutes=45,
            block_kind="study",
        )
    )
    repo.create_session(
        Session(
            task_id=task.id,
            branch="study",
            subject_scope="Rust async cancellation",
            start_at=datetime.utcnow() - timedelta(minutes=20),
            end_at=datetime.utcnow() - timedelta(minutes=5),
            completion_pct=100,
        )
    )

    candidates = build_next_candidates(repo, limit=5)

    assert all(candidate.source != "plan" for candidate in candidates)
    assert all(candidate.source != "resume" for candidate in candidates)
    assert all("planned study block for rust" not in candidate.short_reason.lower() for candidate in candidates)
