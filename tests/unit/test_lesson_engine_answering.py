# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

import random
from types import SimpleNamespace

from pb.core.learning_partner import LearningPartnerSession
from pb.core.lesson_engine import (
    LessonAttemptRecord,
    LessonEngine,
    _limit_lesson_question_choices,
    LessonQuestionRecord,
    _normalize_lesson_hints,
    _randomize_lesson_question_choices,
    _split_answers,
)
from pb.llm.drafts import LessonQuestionDraft


def _engine() -> LessonEngine:
    engine = object.__new__(LessonEngine)
    engine.runtime = SimpleNamespace(health=lambda: SimpleNamespace(available=False))
    return engine


def _question(
    *,
    question_type: str = "multi_select",
    choices: list[str] | None = None,
    correct: list[str] | None = None,
) -> LessonQuestionRecord:
    choices = choices or ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    correct = correct or [choices[0], choices[1]]
    return LessonQuestionRecord(
        id="run:q1",
        lesson_run_id="run",
        session_id="session",
        page_slug="page",
        question_slug="q1",
        skill_slug="skill",
        question_type=question_type,
        prompt_json={"prompt": "Select all that apply.", "choices": choices},
        answer_json={
            "accepted_answers": list(correct),
            "correct_choices": list(correct),
            "hints": ["First hint", "Second hint", "Third hint"],
            "reveal_answer": "",
        },
        sequence_index=0,
    )


def _attempt(points: float, result: str = "wrong") -> LessonAttemptRecord:
    return LessonAttemptRecord(
        id=f"a{abs(points)}{result}",
        lesson_run_id="run",
        session_id="session",
        page_slug="page",
        question_slug="q1",
        result=result,
        points_delta=points,
    )


def test_multiselect_selected_option_with_commas_is_not_split_apart() -> None:
    long_correct = (
        "In quantum field theory, we use regularization to introduce a cutoff for "
        "high-frequency modes, and renormalization to subtract the infinite background."
    )
    question = _question(
        choices=[
            "Regularization only.",
            "Renormalization only.",
            long_correct,
            "Purely mathematical artifact.",
            "Integer-valued energy states.",
        ],
        correct=[long_correct],
    )

    assert _split_answers(long_correct) == [long_correct]
    assert _engine()._evaluate_locally(question=question, raw_answer=long_correct).result == "correct"
    assert _engine()._evaluate_locally(question=question, raw_answer="3").result == "correct"


def test_wrong_mcq_feedback_omits_repetitive_discrimination_sentence() -> None:
    question = _question(question_type="mcq", choices=["A", "B"], correct=["A"])

    result = _engine()._evaluate_locally(question=question, raw_answer="B")

    assert result is not None
    assert result.result == "wrong"
    assert result.feedback == "Not quite."
    assert "That option misses the main discrimination" not in result.feedback


def test_multiselect_points_follow_first_try_partial_wrong_and_cap_rules() -> None:
    question = _question(
        choices=["A", "B", "C", "D", "E"],
        correct=["A", "B", "C"],
    )
    engine = _engine()

    assert engine._points_delta(
        question=question,
        result="correct",
        response_ms=1000,
        attempts=[],
        raw_answer="1 | 2 | 3",
    ) == 2
    assert engine._points_delta(
        question=question,
        result="close",
        response_ms=1000,
        attempts=[],
        raw_answer="1 | 4",
    ) == -1
    assert engine._points_delta(
        question=question,
        result="wrong",
        response_ms=1000,
        attempts=[],
        raw_answer="4",
    ) == -3
    assert engine._points_delta(
        question=question,
        result="wrong",
        response_ms=1000,
        attempts=[_attempt(-3)],
        raw_answer="4",
    ) == -2
    assert engine._points_delta(
        question=question,
        result="wrong",
        response_ms=1000,
        attempts=[_attempt(-3), _attempt(-2)],
        raw_answer="4",
    ) == 0


def test_hints_lower_positive_point_ceiling_by_half_point_each() -> None:
    engine = _engine()
    mcq = _question(question_type="mcq", correct=["Alpha"])
    mcq.hint_level = 1

    assert engine._points_delta(
        question=mcq,
        result="correct",
        response_ms=1000,
        attempts=[],
        raw_answer="1",
    ) == 2.5

    mcq.hint_level = 3
    assert engine._points_delta(
        question=mcq,
        result="correct",
        response_ms=1000,
        attempts=[],
        raw_answer="1",
    ) == 1.5

    mcq.hint_level = 1
    assert engine._points_delta(
        question=mcq,
        result="correct",
        response_ms=60_000,
        attempts=[],
        raw_answer="1",
    ) == 1

    multi = _question(choices=["A", "B", "C", "D", "E"], correct=["A", "B", "C"])
    multi.hint_level = 1
    assert engine._points_delta(
        question=multi,
        result="correct",
        response_ms=1000,
        attempts=[],
        raw_answer="1 | 2 | 3",
    ) == 1.5


def test_non_multiselect_negative_caps_distinguish_free_response() -> None:
    engine = _engine()
    short_text = _question(question_type="short_text")
    mcq = _question(question_type="mcq", correct=["A"])

    assert engine._points_delta(
        question=short_text,
        result="wrong",
        response_ms=1000,
        attempts=[],
    ) == -1
    assert engine._points_delta(
        question=short_text,
        result="wrong",
        response_ms=1000,
        attempts=[_attempt(-3)],
    ) == 0
    assert engine._points_delta(
        question=mcq,
        result="wrong",
        response_ms=1000,
        attempts=[_attempt(-2)],
    ) == -2
    assert engine._points_delta(
        question=mcq,
        result="wrong",
        response_ms=1000,
        attempts=[_attempt(-4)],
    ) == 0


def test_hint_ladder_normalizes_to_three_distinct_hints() -> None:
    hints = _normalize_lesson_hints(["Look at the contrast.", "Look at the contrast."], prompt="Why?", skill_label="Bayes")

    assert len(hints) == 3
    assert len(set(hints)) == 3
    assert hints[0] == "Look at the contrast."


def test_lesson_choice_randomization_moves_correct_answer_off_first_slot() -> None:
    question = LessonQuestionDraft(
        prompt="Which option is correct?",
        question_type="mcq",
        choices=["Correct mechanism", "Surface wording", "Neighboring idea", "Completion marker"],
        accepted_answers=["Correct mechanism"],
        correct_choices=["Correct mechanism"],
        reveal_answer="Correct mechanism",
    )

    _randomize_lesson_question_choices(question, rng=random.Random(7))

    assert set(question.choices) == {
        "Correct mechanism",
        "Surface wording",
        "Neighboring idea",
        "Completion marker",
    }
    assert question.choices[0] != "Correct mechanism"
    assert question.correct_choices == ["Correct mechanism"]


def test_lesson_choice_randomization_maps_ordinal_answer_keys_before_shuffle() -> None:
    question = LessonQuestionDraft(
        prompt="Select all metric axioms.",
        question_type="multi_select",
        choices=["Non-negativity", "Symmetry", "Triangle inequality", "Strict inequality", "Continuity"],
        accepted_answers=["1, 2, 3"],
        correct_choices=["1", "2", "3"],
        reveal_answer="1 | 2 | 3",
    )

    _randomize_lesson_question_choices(question, rng=random.Random(4))

    assert question.correct_choices == ["Non-negativity", "Symmetry", "Triangle inequality"]
    assert question.accepted_answers == ["Non-negativity", "Symmetry", "Triangle inequality"]
    assert question.reveal_answer == "Non-negativity | Symmetry | Triangle inequality"
    displayed_correct_positions = [
        str(question.choices.index(choice) + 1)
        for choice in question.correct_choices
    ]
    persisted = _question(choices=question.choices, correct=question.correct_choices)

    assert _engine()._evaluate_locally(
        question=persisted,
        raw_answer=" ".join(displayed_correct_positions),
    ).result == "correct"


def test_lesson_choice_quality_collapses_explanatory_duplicate_answers() -> None:
    long_choice = (
        r"The largest radius is $\epsilon = r - d(x, y)$, which represents "
        r"the distance from x to the boundary of $B_r(y)$."
    )
    terse_choice = r"$\epsilon = r - d(x, y)$"
    question = LessonQuestionDraft(
        prompt=r"What is the largest guaranteed radius $\epsilon$?",
        question_type="mcq",
        choices=[
            r"$\epsilon = d(x, y)$",
            long_choice,
            terse_choice,
            r"$\epsilon = r + d(x, y)$",
            r"$\epsilon = r$",
        ],
        accepted_answers=[long_choice],
        correct_choices=[long_choice],
        reveal_answer=long_choice,
    )

    _limit_lesson_question_choices(question)

    assert question.choices == [
        "ε = 𝑟 - 𝑑(𝑥, 𝑦)",
        "ε = 𝑑(𝑥, 𝑦)",
        "ε = 𝑟 + 𝑑(𝑥, 𝑦)",
        "ε = 𝑟",
    ]
    assert question.correct_choices == ["ε = 𝑟 - 𝑑(𝑥, 𝑦)"]
    assert question.accepted_answers == ["ε = 𝑟 - 𝑑(𝑥, 𝑦)"]
    assert all("which" not in choice and "represents" not in choice for choice in question.choices)


def test_lesson_choice_quality_keeps_mathematically_distinct_similar_formulas() -> None:
    question = LessonQuestionDraft(
        prompt=r"What is the largest guaranteed radius $\epsilon$?",
        question_type="mcq",
        choices=[
            r"$\epsilon = r - d(x, y)$",
            r"$\epsilon = r + d(x, y)$",
            r"$\epsilon = d(x, y)$",
        ],
        accepted_answers=[r"$\epsilon = r - d(x, y)$"],
        correct_choices=[r"$\epsilon = r - d(x, y)$"],
    )

    _limit_lesson_question_choices(question)

    assert "ε = 𝑟 - 𝑑(𝑥, 𝑦)" in question.choices
    assert "ε = 𝑟 + 𝑑(𝑥, 𝑦)" in question.choices
    assert len(question.choices) == 3


def test_correct_multiselect_feedback_line_is_green() -> None:
    rows = LearningPartnerSession._feedback_lines(
        object(),
        ["Partly right.\nCorrect selections: regularization; renormalization"],
    )

    assert [row.plain for row in rows] == [
        "Partly right.",
        "Correct selections: regularization; renormalization",
    ]
    assert rows[1].style == "green"
