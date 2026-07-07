from __future__ import annotations

import pb.core.clarifier as clarifier_mod

from pathlib import Path
from types import SimpleNamespace

from pb.cli.pickers import PickerResult
from pb.core.clarifier import ClarifierAnswerBundle, ClarifierAnswerRecord, ClarifierService
from pb.core.product_control import ControlState, FeedbackEvent
from pb.llm.drafts import ClarifierQuestionDraft


def _runtime_unavailable():
    return SimpleNamespace(
        config=SimpleNamespace(model_roles=SimpleNamespace(fast_inference="", fast="", default="")),
        generate_draft=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )


def test_clarifier_fallback_uses_richer_fields_and_options():
    service = ClarifierService(_runtime_unavailable())

    questions = service._fallback_questions(
        "learn ricci flow and its application in general relativity",
        {"mode": "study"},
        control_state=ControlState(scope="artifact"),
        prior_question_signatures=[],
    )

    assert questions
    first = questions[0]
    assert "theoretical focus" in first.question
    assert first.option_candidates
    assert any("definitions" in option.lower() for option in first.option_candidates)
    assert first.why_this_matters
    assert first.downstream_effect
    assert 0.0 <= first.confidence <= 1.0


def test_clarifier_fallback_uses_prior_foundational_signal():
    service = ClarifierService(_runtime_unavailable())
    state = ControlState(
        scope="artifact",
        signal_counts={"foundational": 1},
        feedback_events=[
            FeedbackEvent(
                kind="needs_prerequisite",
                artifact_kind="study_block",
                artifact_id="ricci",
                free_text="start earlier",
            )
        ],
    )

    questions = service._fallback_questions(
        "ricci curvature for general relativity",
        {"mode": "study"},
        control_state=state,
        prior_question_signatures=[],
    )

    rendered = " ".join(question.question for question in questions)
    assert "shaky prerequisite" in rendered


def test_clarifier_filters_repeated_question_signatures():
    service = ClarifierService(_runtime_unavailable())

    questions = service.generate_questions(
        "learn tensors",
        {"mode": "study"},
        scope="study",
        control_state=ControlState(scope="artifact"),
        prior_question_signatures=["wrong_scope:which theoretical focus should the next study block clarify first for learn tensors?"],
    )

    assert all("Which theoretical focus should the next study block clarify first for learn tensors?" != q.question for q in questions)


def test_ask_clarifier_questions_uses_selected_option_without_reprompt(monkeypatch):
    console_lines: list[str] = []

    class _Console:
        def print(self, message):
            console_lines.append(str(message))

    monkeypatch.setattr(clarifier_mod, "get_console", lambda: _Console())
    monkeypatch.setattr(
        clarifier_mod,
        "pick_single_choice",
        lambda *args, **kwargs: "Explain one concept clearly",
    )
    monkeypatch.setattr(
        clarifier_mod,
        "prompt_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompt_text should not run for picked answers")),
    )

    bundle = clarifier_mod.ask_clarifier_questions(
        [
            ClarifierQuestionDraft(
                question="What exact result should the next block produce?",
                option_candidates=["Explain one concept clearly", "Work one guided example"],
                inferred_signal_type="wrong_scope",
            )
        ]
    )

    assert bundle.answers == {"What exact result should the next block produce?": "Explain one concept clearly"}
    assert bundle.records[0].answer_kind == "selected_option"
    assert any("[blue]" in line and "Explain one concept clearly" in line for line in console_lines)


def test_ask_clarifier_questions_marks_custom_text_separately(monkeypatch):
    console_lines: list[str] = []

    class _Console:
        def print(self, message):
            console_lines.append(str(message))

    monkeypatch.setattr(clarifier_mod, "get_console", lambda: _Console())
    monkeypatch.setattr(
        clarifier_mod,
        "pick_single_choice",
        lambda *args, **kwargs: PickerResult(kind="inline_text", value="Focus on linear maps first"),
    )

    bundle = clarifier_mod.ask_clarifier_questions(
        [
            ClarifierQuestionDraft(
                question="Which starting layer should come first?",
                option_candidates=["Vectors and matrices", "Span, basis, and dimension"],
                inferred_signal_type="needs_prerequisite",
            )
        ]
    )

    assert bundle.answers == {"Which starting layer should come first?": "Focus on linear maps first"}
    assert bundle.records[0].answer_kind == "custom_text"
    assert any("[green]" in line and "Focus on linear maps first" in line for line in console_lines)


def test_ask_clarifier_questions_treats_none_as_explicit_exclusion(monkeypatch):
    monkeypatch.setattr(
        clarifier_mod,
        "pick_single_choice",
        lambda *args, **kwargs: PickerResult(kind="inline_text", value="none"),
    )

    bundle = clarifier_mod.ask_clarifier_questions(
        [
            ClarifierQuestionDraft(
                question="Which pragmatic particle is the most frequent source of confusion?",
                option_candidates=["o", "la", "le", "me"],
                inferred_signal_type="naturalness_focus",
            )
        ]
    )

    assert bundle.answers["Which pragmatic particle is the most frequent source of confusion?"] == "none"
    assert bundle.records[0].answer_kind == "explicit_exclusion"
    assert bundle.records[0].excluded_options == ["o", "la", "le", "me"]


def test_trim_context_option_candidates_reserves_discuss_and_inline_slots():
    trimmed = clarifier_mod.trim_context_option_candidates(
        ["one", "two", "three", "four", "five", "six"],
    )

    assert trimmed == ["one", "two", "three", "four"]


def test_ask_clarifier_questions_trims_visible_options_before_picker(monkeypatch):
    captured: dict[str, object] = {}

    def fake_pick(options, *args, **kwargs):
        captured["options"] = options
        return "one"

    monkeypatch.setattr(clarifier_mod, "pick_single_choice", fake_pick)

    clarifier_mod.ask_clarifier_questions(
        [
            ClarifierQuestionDraft(
                question="Pick one",
                option_candidates=["one", "two", "three", "four", "five", "six"],
                inferred_signal_type="wrong_scope",
            )
        ]
    )

    assert captured["options"] == [
        ("one", "one"),
        ("two", "two"),
        ("three", "three"),
        ("four", "four"),
        (clarifier_mod.DISCUSS_SENTINEL, "Let's discuss this"),
    ]


def test_clarifier_prompt_block_serializes_exclusions_and_answer_kind():
    bundle = ClarifierAnswerBundle(
        answers={"Particles": "none"},
        records=[
            ClarifierAnswerRecord(
                question="Particles",
                answer="none",
                answer_kind="explicit_exclusion",
                options_presented=["o", "la", "le", "me"],
                excluded_options=["o", "la", "le", "me"],
                inferred_signal_type="naturalness_focus",
            )
        ],
    )

    prompt = clarifier_mod.clarifier_prompt_block(bundle)

    assert "Structured clarifier answers JSON:" in prompt
    assert '"answer_kind": "explicit_exclusion"' in prompt
    assert '"excluded_options": ["o", "la", "le", "me"]' in prompt
    assert "explicitly out of scope" in prompt


def test_learning_intent_style_guidance_defaults_to_performance_not_academia():
    guidance = clarifier_mod.learning_intent_style_guidance()

    assert "performance coaching" in guidance
    assert "sociolinguistics" in guidance
    assert "unrelated dialect or regional comparisons" in guidance


def test_study_prompt_includes_performance_first_guidance_for_accent_requests():
    from pb.cli.commands.study import _build_study_prompt

    prompt = _build_study_prompt(
        topic_text="Help with Taiwan accent",
        domain_hint="Taiwan accent",
        matched_goal=None,
        requested_minutes=30,
        stage_hint=None,
        steps=False,
        vault_path=Path("."),
        clarifier_bundle=None,
    )

    assert "performance coaching" in prompt
    assert "sociolinguistics" in prompt
    assert "unrelated dialect" in prompt
    assert "not a performance drill" in prompt
