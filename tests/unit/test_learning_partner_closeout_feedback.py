from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pb.cli.input_router import RoutedInput, classify_interactive_input
from pb.cli.pickers import PickerResult
from pb.core.closeout import CloseoutService
from pb.core.feedback_proposals import FeedbackProposalService
from pb.core.learning_metadata import build_learning_task_description
from pb.core.learning_partner import (
    LearningPartnerSession,
    load_session_transcript,
    save_session_transcript,
)
from pb.core.lesson_engine import (
    LESSON_FINISH_CONTINUE_SESSION,
    LESSON_FINISH_MENU_ACTION,
    LESSON_FINISH_NEXT_SESSION,
    lesson_finish_action_options,
    lesson_finish_review_label,
)
from pb.core.models import Session, Task
from pb.core.renderables import renderable_cli_text
from pb.domain.enums import SessionMode, TaskState
from pb.llm.drafts import LearningPartnerTurnDraft


def _runtime(tmp_path: Path):
    learning_policy = SimpleNamespace(
        answer_check="",
        small_retry="",
        session_explain="",
        lesson_hint_intuitive="",
        drill_generation="",
        complex_free_response_eval="",
        lesson_planning="",
    )
    return SimpleNamespace(
        health=lambda: SimpleNamespace(available=False),
        role_bindings=lambda: {},
        config=SimpleNamespace(
            model_roles=SimpleNamespace(default="gemini:test", fast_inference="gemini:test"),
            learning=SimpleNamespace(model_policy=learning_policy),
            preferences={},
        ),
    )


def _runtime_ctx(tmp_path: Path, temp_config):
    data_dir = tmp_path / ".pb-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = tmp_path / "Learning" / "Inbox" / "pb"
    quarantine_path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        vault_path=Path(temp_config.general.vault_path),
        quarantine_path=quarantine_path,
        data_dir=data_dir,
    )


def _create_partner(
    repo,
    *,
    tmp_path: Path,
    temp_config,
    branch: str = "study",
    topic: str = "Bayes rule",
    domain: str = "probability",
    objective: str = "Use the concept on one fresh case.",
    study_mode: str = "",
    practice_stage: str = "",
    confidence_level: float = 0.0,
):
    task = repo.create_task(
        Task(
            title=topic,
            state=TaskState.ACTIVE,
            work_type="practice" if branch == "practise" else "study",
            description=build_learning_task_description(
                branch="study" if "teach" in study_mode else branch,
                scope=topic,
                domain=domain,
                study_mode=study_mode,
                practice_stage=practice_stage,
                success_check=objective,
            ),
            generated_names={"display_title": topic},
        )
    )
    session = repo.create_session(
        Session(
            task_id=task.id,
            mode=SessionMode.PRACTICE if branch == "practise" else SessionMode.FOCUS,
            branch="study" if "teach" in study_mode else branch,
            subject_scope=topic,
            intended_outcome=objective,
        )
    )
    runtime_ctx = _runtime_ctx(tmp_path, temp_config)
    partner = LearningPartnerSession(
        runtime=_runtime(tmp_path),
        runtime_ctx=runtime_ctx,
        repo=repo,
        task=task,
        session=session,
        branch="study" if "teach" in study_mode else branch,
        objective=objective,
        topic=topic,
        domain=domain,
        mode=study_mode or practice_stage or branch,
        confidence_level=confidence_level,
    )
    return partner, task, session, runtime_ctx


def _active_question(repo, session):
    run = repo.get_lesson_run(session.id)
    assert run is not None
    question = repo.get_lesson_question(run.id, run.active_question_slug)
    assert question is not None
    return run, question


def _correct_answer(question) -> str:
    if question.question_type == "reorder":
        ordered_items = list(question.answer_json.get("ordered_items", []) or [])
        display_items = list(question.prompt_json.get("display_items", []) or ordered_items)
        return " ".join(str(display_items.index(item) + 1) for item in ordered_items)
    if question.question_type == "multi_select":
        choices = list(question.answer_json.get("correct_choices", []) or question.answer_json.get("accepted_answers", []))
        return " | ".join(str(item) for item in choices)
    accepted = list(question.answer_json.get("accepted_answers", []) or question.answer_json.get("correct_choices", []))
    if accepted:
        return str(accepted[0])
    return str(question.answer_json.get("reveal_answer", "") or "answer")


def _wrong_answer(question) -> str:
    choices = [str(item) for item in question.prompt_json.get("choices", []) if str(item).strip()]
    correct = {str(item) for item in question.answer_json.get("correct_choices", [])}
    for choice in choices:
        if choice not in correct:
            return choice
    return "wrong answer"


def _advance_until_mistakes(partner, repo, session, *, max_steps: int = 30) -> None:
    for _ in range(max_steps):
        run, question = _active_question(repo, session)
        if question.page_slug == "mistakes" or run.ready_to_finish:
            return
        partner.respond_once(_correct_answer(question))


def _clear_lesson(partner, repo, session, *, max_steps: int = 80):
    partner.open_with_first_move()
    for _ in range(max_steps):
        run = repo.get_lesson_run(session.id)
        assert run is not None
        if run.ready_to_finish:
            return run
        question = repo.get_lesson_question(run.id, run.active_question_slug)
        assert question is not None
        partner.respond_once(_correct_answer(question))
    raise AssertionError("lesson did not clear")


def test_feedback_proposal_extracts_learning_scoped_patches(tmp_path):
    service = FeedbackProposalService()
    proposal = service.generate_proposal(
        "these questions are mechanical, use GPT OSS 120B for simple naming, "
        "never flatter me, and give me a proper study partner",
        scope="learn",
    )

    assert proposal.preference_patches["prefer_contextual_questions"] is True
    assert proposal.preference_patches["avoid_generic_clarifiers"] is True
    assert proposal.preference_patches["coach_tone"] == "frank_no_flattery"
    assert proposal.preference_patches["simple_inference_model_role"] == "fast_inference"
    assert proposal.preference_patches["study_session_mode"] == "agentic_partner"

    note_path = service.write_proposal(tmp_path, "this tool is useless", proposal, scope="learn")
    assert note_path.exists()
    assert "Feedback Proposal" in note_path.read_text(encoding="utf-8")


def test_closeout_service_detects_frustration_feedback():
    decision = CloseoutService().generate_closeout(
        SimpleNamespace(subject_scope="German conjugation"),
        "i didnt do anything, this tool is useless",
        {"title": "German conjugation recovery"},
    )

    assert decision.status == "frustration_feedback"
    assert decision.discard_recommended is True
    assert "No meaningful learning evidence" in decision.summary


def test_learning_partner_transcript_round_trip(tmp_path):
    transcript = [
        {"role": "user", "content": "how should i approach this"},
        {"role": "assistant", "content": "State the core idea first."},
    ]

    save_session_transcript(tmp_path, "session-1", transcript)

    loaded = load_session_transcript(tmp_path, "session-1")
    assert loaded == transcript


def test_cleared_lesson_renders_finish_action_menu(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)

    _clear_lesson(partner, repo, session)

    snapshot = partner.engine.snapshot_for()
    turn = partner.engine.current_turn()

    assert "/finish" not in snapshot.header_note
    assert "/finish" not in turn.reply
    assert snapshot.footer_commands == []
    assert turn.question_type == "mcq"
    assert turn.next_action == LESSON_FINISH_MENU_ACTION
    assert turn.mcq_options == lesson_finish_action_options("study")


def test_finish_menu_routes_to_commands_or_continuation(repo, temp_config, temp_dir, monkeypatch):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    _clear_lesson(partner, repo, session)

    class FakeTransitionService:
        def __init__(self, *_args, **_kwargs):
            pass

        def choose_next_session(self, *, session, task):
            return SimpleNamespace(
                command="study 'Bayes transfer drill' --duration 25m --yes",
                summary="Tailored Bayes transfer plan",
                draft=SimpleNamespace(),
            )

        def choose_continuation_focus(self, *, session, task):
            return "Extend Bayes with a fresh false-positive case", SimpleNamespace()

        def maybe_offer_agent_spawn(self, **_kwargs):
            return None

    monkeypatch.setattr("pb.core.learning_partner.LearningTransitionService", FakeTransitionService)

    next_input = partner._route_finish_menu_selection(LESSON_FINISH_NEXT_SESSION)
    review_input = partner._route_finish_menu_selection(lesson_finish_review_label("study"))
    continue_input = partner._route_finish_menu_selection(LESSON_FINISH_CONTINUE_SESSION)

    assert next_input is not None
    assert next_input.action == "command"
    assert next_input.command == "finish --skip --yes"
    assert next_input.follow_up_command == "study 'Bayes transfer drill' --duration 25m --yes"
    assert partner.session.generated_names["next_session_preference"] == "Tailored Bayes transfer plan"
    assert review_input is not None
    assert review_input.action == "command"
    assert review_input.command == "finish"
    assert review_input.follow_up_command == ""
    assert continue_input is not None
    assert continue_input.kind == "lesson_continue"
    assert continue_input.text == "Extend Bayes with a fresh false-positive case"


def test_session_control_slash_commands_are_available_during_lessons(repo, temp_config, temp_dir):
    partner, _, _, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)

    assert "/pause" in partner.contextual_command_names()
    assert "/finish" in partner.contextual_command_names()
    assert "/next" in partner.contextual_command_names()

    decision = classify_interactive_input(
        "/finish",
        pb_command_resolver=partner.pb_command_resolver,
        slash_registry=partner.command_registry,
        active_learning=True,
        allow_shell_commands=False,
        allow_nl_dispatch=False,
    )

    assert decision.kind == "slash_command"
    assert decision.command == "/finish"


def test_session_control_slash_commands_finalize_from_current_lesson(repo, temp_config, temp_dir, monkeypatch):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()

    class FakeTransitionService:
        def __init__(self, *_args, **_kwargs):
            pass

        def choose_next_session(self, *, session, task):
            return SimpleNamespace(
                command="study 'Bayes transfer drill' --duration 25m --yes",
                summary="Tailored Bayes transfer plan",
                draft=SimpleNamespace(),
            )

    monkeypatch.setattr("pb.core.learning_partner.LearningTransitionService", FakeTransitionService)

    pause_result = partner.run_contextual_command("/pause", "break")
    finish_result = partner.run_contextual_command("/finish")
    next_result = partner.run_contextual_command("/next", "more proofs")

    assert pause_result is not None
    assert pause_result.action == "pause"
    assert pause_result.summary == "break"
    assert finish_result is not None
    assert finish_result.action == "finish"
    assert next_result is not None
    assert next_result.action == "command"
    assert next_result.follow_up_command == "study 'Bayes transfer drill' --duration 25m --yes"
    assert next_result.skip_finish_assessment is True
    assert session.generated_names["next_session_preference"] == "Tailored Bayes transfer plan"


def test_continue_from_finish_menu_adds_extension_page(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    _clear_lesson(partner, repo, session)

    turn = partner.continue_after_clear()
    run = repo.get_lesson_run(session.id)
    assert run is not None
    active_page = repo.get_lesson_page(run.id, run.active_page_slug)
    active_question = repo.get_lesson_question(run.id, run.active_question_slug)

    assert run.ready_to_finish is False
    assert active_page is not None
    assert active_page.title == "Implications and applications"
    assert active_question is not None
    assert active_question.page_slug == active_page.page_slug
    assert turn.next_action != LESSON_FINISH_MENU_ACTION


def test_unified_runtime_covers_teach_study_and_practise(repo, temp_config, temp_dir):
    teach_partner, _, teach_session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        topic="Konjunktiv II",
        domain="german",
        study_mode="socratic_teach",
    )
    study_partner, _, study_session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        topic="Bayes rule",
        domain="probability",
        study_mode="active_recall",
    )
    practise_partner, _, practise_session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        branch="practise",
        topic="Biddle grip",
        domain="cardistry",
        practice_stage="integrate",
    )

    assert teach_partner.engine.lesson_mode == "teach"
    assert study_partner.engine.lesson_mode == "study"
    assert practise_partner.engine.lesson_mode == "practise"
    assert repo.get_lesson_run(teach_session.id) is not None
    assert repo.get_lesson_run(study_session.id) is not None
    assert repo.get_lesson_run(practise_session.id) is not None


def test_learning_partner_forwards_confidence_level_to_lesson_engine(repo, temp_config, temp_dir):
    partner, _, _, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        confidence_level=0.72,
    )

    assert partner.confidence_level == 0.72
    assert partner.engine.confidence_level == 0.72


def test_lesson_pages_and_question_state_persist_on_resume(repo, temp_config, temp_dir):
    partner, _, session, runtime_ctx = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    _, question = _active_question(repo, session)
    partner.respond_once(_correct_answer(question))

    resumed = LearningPartnerSession(
        runtime=_runtime(temp_dir),
        runtime_ctx=runtime_ctx,
        repo=repo,
        task=repo.get_task(session.task_id),
        session=repo.get_session(session.id),
        branch="study",
        objective="Use the concept on one fresh case.",
        topic="Bayes rule",
        domain="probability",
        mode="active_recall",
    )
    run = repo.get_lesson_run(session.id)
    assert run is not None
    assert resumed.engine.current_snapshot().run.active_question_slug == run.active_question_slug
    assert load_session_transcript(runtime_ctx.data_dir, session.id)


def test_varied_question_types_exist_across_modes(repo, temp_config, temp_dir):
    teach_partner, _, teach_session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        topic="Konjunktiv II",
        domain="german",
        study_mode="socratic_teach",
    )
    study_partner, _, study_session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        topic="Bayes rule",
        domain="probability",
        study_mode="active_recall",
    )
    practise_partner, _, practise_session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        branch="practise",
        topic="Biddle grip",
        domain="cardistry",
        practice_stage="integrate",
    )

    types = {
        question.question_type
        for session in (teach_session, study_session, practise_session)
        for question in repo.list_lesson_questions(session.id)
    }
    assert types >= {
        "mcq",
        "multi_select",
        "cloze",
        "short_text",
        "free_production",
        "error_correction",
        "reorder",
    }


def test_learning_partner_caps_mcq_and_cloze_options_and_uses_inline_edit(repo, temp_config, temp_dir, monkeypatch):
    partner, _, _, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    captured: list[tuple[list[tuple[str, str]], bool]] = []

    def fake_pick(options, *args, **kwargs):
        captured.append((options, kwargs.get("allow_inline_edit", False)))
        return PickerResult(kind="inline_text", value="custom answer")

    monkeypatch.setattr("pb.core.learning_partner.pick_single_choice", fake_pick)

    mcq_result = partner._render_question_input(
        LearningPartnerTurnDraft(
            reply="Question",
            question_type="mcq",
            mcq_options=["1", "2", "3", "4", "5", "6", "7"],
        )
    )
    cloze_result = partner._render_question_input(
        LearningPartnerTurnDraft(
            reply="Question",
            question_type="cloze",
            cloze_blank_options=["a", "b", "c", "d", "e", "f"],
        )
    )

    assert isinstance(mcq_result, RoutedInput) and mcq_result.text == "custom answer"
    assert isinstance(cloze_result, RoutedInput) and cloze_result.text == "custom answer"
    assert len(captured) == 2
    assert all(len(options) == 5 for options, _ in captured)
    assert all(allow_inline_edit is True for _, allow_inline_edit in captured)
    assert all("discuss" not in label.lower() for options, _ in captured for _, label in options)


def test_learning_partner_renders_option_labels_but_preserves_raw_values(repo, temp_config, temp_dir, monkeypatch):
    partner, _, _, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    captured: dict[str, object] = {}

    def fake_pick(options, *args, **kwargs):
        captured["options"] = options
        return PickerResult(kind="selection", value=r"\(\mathbb{R}^n\)")

    monkeypatch.setattr("pb.core.learning_partner.pick_single_choice", fake_pick)

    result = partner._render_question_input(
        LearningPartnerTurnDraft(
            reply="Question",
            question_type="mcq",
            mcq_options=[r"\(\mathbb{R}^n\)", "plain distractor"],
        )
    )

    assert isinstance(result, RoutedInput)
    assert result.text == r"\(\mathbb{R}^n\)"
    options = captured["options"]
    assert options[0][0] == r"\(\mathbb{R}^n\)"
    assert options[0][1] == renderable_cli_text(r"\(\mathbb{R}^n\)")
    assert r"\mathbb" not in options[0][1]


def test_learning_partner_caps_multi_select_options_without_discuss(repo, temp_config, temp_dir, monkeypatch):
    partner, _, _, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    captured: dict[str, object] = {}

    def fake_pick(options, *args, **kwargs):
        captured["options"] = options
        captured["allow_inline_edit"] = kwargs.get("allow_inline_edit")
        return PickerResult(kind="selection", value=["tone sandhi", "retroflex reduction"])

    monkeypatch.setattr("pb.core.learning_partner.pick_many_choices", fake_pick)

    result = partner._render_question_input(
        LearningPartnerTurnDraft(
            reply="Question",
            question_type="multi_select",
            mcq_options=[
                "tone sandhi",
                "retroflex reduction",
                "particle timing",
                "vowel quality",
                "rhythm",
                "intonation",
            ],
        )
    )

    assert isinstance(result, RoutedInput)
    assert result.text == "tone sandhi | retroflex reduction"
    assert captured["allow_inline_edit"] is True
    assert len(captured["options"]) == 5
    assert all("discuss" not in label.lower() for _, label in captured["options"])


def test_first_miss_uses_hint_without_revealing_answer(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    _, question = _active_question(repo, session)

    turn = partner.respond_once(_wrong_answer(question))

    refreshed = repo.get_lesson_question(session.id, question.question_slug)
    assert refreshed is not None
    assert refreshed.status == "pending"
    assert refreshed.hint_level == 1
    joined = "\n".join(turn.corrections)
    assert refreshed.answer_json["reveal_answer"] not in joined


def test_answer_command_marks_revealed_and_enqueues_retry(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    _, question = _active_question(repo, session)
    before_points = repo.get_lesson_run(session.id).total_points
    before_attempts = len(repo.list_lesson_attempts(session.id, question.question_slug))

    partner.run_contextual_command("/answer")

    refreshed = repo.get_lesson_question(session.id, question.question_slug)
    retries = repo.list_lesson_questions(session.id, "mistakes")
    assert refreshed is not None
    assert refreshed.status == "revealed"
    assert refreshed.revealed is True
    assert refreshed.mastered is False
    assert retries
    assert repo.get_lesson_run(session.id).total_points == before_points
    assert len(repo.list_lesson_attempts(session.id, question.question_slug)) == before_attempts


def test_wrong_mcq_creates_transformed_retry_not_identical(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        topic="Konjunktiv II",
        domain="german",
        study_mode="socratic_teach",
    )
    partner.open_with_first_move()
    _, question = _active_question(repo, session)

    partner.respond_once(_wrong_answer(question))

    retries = repo.list_lesson_questions(session.id, "mistakes")
    assert retries
    assert retries[0].prompt_json.get("choices", []) != question.prompt_json.get("choices", [])


def test_three_misses_reveal_answer_and_pin_easier_retry(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    run, question = _active_question(repo, session)
    wrong = _wrong_answer(question)

    partner.respond_once(wrong)
    partner.respond_once(wrong)
    turn = partner.respond_once(wrong)

    run = repo.get_lesson_run(session.id)
    assert run is not None
    refreshed = repo.get_lesson_question(run.id, question.question_slug)
    retries = repo.list_lesson_questions(run.id, "mistakes")
    assert refreshed is not None
    assert refreshed.status == "revealed"
    assert refreshed.revealed is True
    assert run.active_question_slug == question.question_slug
    assert any(line.startswith("Answer:") for line in turn.corrections)
    assert retries
    assert retries[0].metadata_json.get("retry_strategy") == "easier_then_verify"

    partner.run_contextual_command("/skip")

    skipped = repo.get_lesson_question(run.id, question.question_slug)
    assert skipped is not None
    assert skipped.status == "skipped"
    assert skipped.revealed is True


def test_retry_question_slugs_stay_unique_across_repeated_transformed_retries(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        topic="Konjunktiv II",
        domain="german",
        study_mode="socratic_teach",
    )
    partner.open_with_first_move()
    _, question = _active_question(repo, session)
    partner.respond_once(_wrong_answer(question))
    partner.respond_once(_correct_answer(question))

    _advance_until_mistakes(partner, repo, session)
    _, retry_question = _active_question(repo, session)
    assert retry_question.page_slug == "mistakes"

    partner.respond_once(_wrong_answer(retry_question))

    slugs = [item.question_slug for item in repo.list_lesson_questions(session.id)]
    assert len(slugs) == len(set(slugs))


def test_mistake_queue_runs_after_normal_pages(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    _, question = _active_question(repo, session)
    partner.respond_once(_wrong_answer(question))
    partner.respond_once(_correct_answer(question))

    _advance_until_mistakes(partner, repo, session)

    run, active = _active_question(repo, session)
    assert run.ready_to_finish is False
    assert active.page_slug == "mistakes"
    assert active.retry_of_question_slug == question.question_slug


def test_points_are_separate_from_mastery(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    _, first_question = _active_question(repo, session)
    partner.respond_once(_correct_answer(first_question))

    _, second_question = _active_question(repo, session)
    before_points = repo.get_lesson_run(session.id).total_points
    partner.run_contextual_command("/hint")
    partner.respond_once(_correct_answer(second_question))

    refreshed = repo.get_lesson_question(session.id, second_question.question_slug)
    after_points = repo.get_lesson_run(session.id).total_points
    assert refreshed is not None
    assert refreshed.status == "correct"
    assert refreshed.mastered is False
    assert after_points - before_points == 2.5


def test_easier_command_changes_question_without_scoring_or_attempt(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    run, question = _active_question(repo, session)
    before_prompt = str(question.prompt_json.get("prompt", "") or "")
    before_points = run.total_points
    before_attempts = len(repo.list_lesson_attempts(session.id, question.question_slug))

    turn = partner.run_contextual_command("/easier")

    refreshed = repo.get_lesson_question(session.id, question.question_slug)
    assert refreshed is not None
    assert turn is not None
    assert str(refreshed.prompt_json.get("prompt", "") or "") != before_prompt
    assert repo.get_lesson_run(session.id).total_points == before_points
    assert len(repo.list_lesson_attempts(session.id, question.question_slug)) == before_attempts
    assert refreshed.metadata_json["last_transform"] == "easier"
    assert refreshed.metadata_json["original_prompt_json"]["prompt"] == before_prompt


def test_harder_and_hint_and_intuitive_commands_do_not_score(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    _, question = _active_question(repo, session)
    before_points = repo.get_lesson_run(session.id).total_points
    before_attempts = len(repo.list_lesson_attempts(session.id, question.question_slug))

    harder_turn = partner.run_contextual_command("/harder")
    hint_turn = partner.run_contextual_command("/hint")
    intuitive_turn = partner.run_contextual_command("/intuitive")

    refreshed = repo.get_lesson_question(session.id, question.question_slug)
    assert refreshed is not None
    assert harder_turn is not None and hint_turn is not None and intuitive_turn is not None
    assert refreshed.metadata_json["last_transform"] == "harder"
    assert repo.get_lesson_run(session.id).total_points == before_points
    assert len(repo.list_lesson_attempts(session.id, question.question_slug)) == before_attempts


def test_skill_diagnostics_track_recognition_vs_production(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        topic="Konjunktiv II",
        domain="german",
        study_mode="socratic_teach",
    )
    partner.open_with_first_move()
    _, first_question = _active_question(repo, session)
    partner.respond_once(_correct_answer(first_question))
    _, second_question = _active_question(repo, session)
    partner.run_contextual_command("/answer")

    states = partner.engine.skill_diagnostics()
    state = next(item for item in states if item.skill_slug == second_question.skill_slug)
    assert state.recognition_status == "strong"
    assert state.production_status == "needs_repair"
    assert state.overall_status == "needs_repair"


def test_short_slugs_are_used_for_pages_questions_and_attempts(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    _, question = _active_question(repo, session)
    partner.respond_once(_correct_answer(question))

    run = repo.get_lesson_run(session.id)
    assert run is not None
    pages = repo.list_lesson_pages(run.id)
    questions = repo.list_lesson_questions(run.id)
    attempts = repo.list_lesson_attempts(run.id)

    for page in pages:
        assert len(page.page_slug) <= 27
        assert page.page_slug.count("_") <= 1
    for item in questions:
        assert len(item.question_slug) <= 27
        assert item.question_slug.count("_") <= 1
    for attempt in attempts:
        assert len(attempt.id) <= 27
        assert attempt.id.count("_") <= 1


def test_contextual_commands_and_typed_inputs_still_work(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    names = partner.contextual_command_names()
    assert names[:5] == ["/hint", "/answer", "/harder", "/easier", "/intuitive"]
    assert {"/recall", "/explain", "/drill"} <= set(names)

    turn = partner.run_contextual_command("/intuitive")
    assert turn is not None
    assert turn.corrections

    with patch("pb.core.learning_partner.prompt_answer_or_command", return_value=RoutedInput(kind="answer", text="2 1 3")) as mocked:
        routed = partner._render_question_input(
            SimpleNamespace(
                question_type="reorder",
                mcq_options=[],
                cloze_blank_options=[],
            )
        )
    assert routed.kind == "answer"
    assert mocked.call_args.kwargs["prompt_label"] == "Order> "


def test_review_signal_differs_for_recognition_and_production(repo, temp_config, temp_dir):
    recognise_partner, _, recognise_session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        topic="Konjunktiv II",
        domain="german",
        study_mode="socratic_teach",
    )
    practise_partner, _, practise_session, _ = _create_partner(
        repo,
        tmp_path=temp_dir,
        temp_config=temp_config,
        branch="practise",
        topic="Biddle grip",
        domain="cardistry",
        practice_stage="integrate",
    )

    recognise_partner.open_with_first_move()
    _, recognition_question = _active_question(repo, recognise_session)
    recognise_partner.respond_once(_correct_answer(recognition_question))
    recognition_done = repo.get_lesson_question(recognise_session.id, recognition_question.question_slug)

    practise_partner.open_with_first_move()
    _, production_question = _active_question(repo, practise_session)
    practise_partner.respond_once(_correct_answer(production_question))
    production_done = repo.get_lesson_question(practise_session.id, production_question.question_slug)

    assert recognition_done is not None and production_done is not None
    assert production_done.next_review_at > recognition_done.next_review_at


def test_unknown_slash_commands_list_available_session_controls(repo, temp_config, temp_dir):
    partner, _, _, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    with patch.object(partner.console, "print") as mock_print:
        partner.explain_contextual_command_error(RoutedInput(kind="slash_unknown", text="/resume"))

    message = mock_print.call_args.args[0]
    assert message.startswith("[warn]Unknown contextual command. Available:")
    assert "/pause" in message
    assert "/finish" in message
    assert "/next" in message


def test_latex_survives_question_display_and_contextual_commands(repo, temp_config, temp_dir):
    partner, _, session, _ = _create_partner(repo, tmp_path=temp_dir, temp_config=temp_config)
    partner.open_with_first_move()
    _, question = _active_question(repo, session)
    question.prompt_json["prompt"] = (
        r"Map $f: M \to N$ and explain \(x^2 + y^2 = 1\). "
        r"Then justify \[ \int_M \omega = \int_{\partial M} \eta \]."
    )
    question.answer_json["hints"] = [r"Think in \mathbb{R}^n with \partial, \nabla, and \wedge."]
    question.answer_json["reveal_answer"] = r"\[ \int_M \omega = \int_{\partial M} \eta \]"
    repo.update_lesson_question(question)

    opening = partner.engine.current_turn()
    easier = partner.run_contextual_command("/easier")
    harder = partner.run_contextual_command("/harder")
    hint = partner.run_contextual_command("/hint")
    with patch.object(
        partner.engine,
        "_generate_explanation",
        return_value=r"Use \(x^2 + y^2 = 1\) and \nabla on \mathbb{R}^n.",
    ):
        intuitive = partner.run_contextual_command("/intuitive")
    answer = partner.run_contextual_command("/answer")

    assert "→" in renderable_cli_text(opening.reply)
    assert "² = 1" in renderable_cli_text(opening.reply)
    assert "∫" in renderable_cli_text(easier.reply)
    assert "∫" in renderable_cli_text(harder.reply)
    assert "ℝⁿ" in renderable_cli_text(hint.corrections[0])
    assert "∇" in renderable_cli_text(intuitive.corrections[0])
    assert "∫" in renderable_cli_text(answer.corrections[0])
