from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from pb.core.finish_assessment import AssessmentResult, FinishAssessmentAgent, SubSkillScore
from pb.core.session_blueprints import resolve_learning_session_blueprint


def _session(**generated_names):
    now = datetime.utcnow()
    return SimpleNamespace(
        id="sess-1",
        task_id="task-1",
        branch="study",
        subject_scope="Riemann zeta function",
        start_at=now - timedelta(minutes=20),
        end_at=now,
        actual_outcome="Worked through analytic continuation.",
        observed_errors="Mixed up the functional equation domain.",
        next_adjustment="Retry one derivation without notes.",
        generated_names=generated_names,
    )


def _task():
    return SimpleNamespace(
        id="task-1",
        title="Study: Riemann zeta analytic continuation",
        description="",
        work_type="study",
    )


def test_finish_assessment_has_no_interactive_subskill_selector():
    assert not hasattr(FinishAssessmentAgent, "_select_sub_skills")


def test_assessment_targets_include_blueprint_evidence_and_self_reports():
    resolution = resolve_learning_session_blueprint(
        branch="study",
        domain="math",
        topic="analytic continuation",
    )
    session = _session(
        session_blueprint=resolution.blueprint.model_dump(mode="json"),
        learning_partner_evidence=[
            {
                "source": "learner_input",
                "subskill": "contour_choice",
                "note": "I explained why the branch cut matters.",
            }
        ],
        learning_partner_compact={
            "unknowns": ["functional equation setup"],
            "detected_gaps": ["pole at s=1"],
        },
        learner_self_reports=[
            {
                "topic": "analytic continuation",
                "level": 3,
                "confidence": 2,
            }
        ],
    )
    agent = FinishAssessmentAgent()
    targets = agent._assessment_targets(
        session=session,
        task=_task(),
        domain="math",
        template=SimpleNamespace(sub_skill_taxonomy=["problem_setup", "verification"]),
        blueprint=resolution.blueprint,
    )

    assert "problem_setup" in targets
    assert "contour_choice" in targets
    assert "functional equation setup" in targets
    assert "analytic continuation" in targets


def test_run_assessment_uses_evidence_without_prompting():
    resolution = resolve_learning_session_blueprint(
        branch="study",
        domain="math",
        topic="analytic continuation",
    )
    session = _session(
        domain_pack_id=resolution.pack_id,
        session_blueprint=resolution.blueprint.model_dump(mode="json"),
        learning_partner_evidence=[
            {
                "source": "learner_input",
                "subskill": "verification",
                "note": "I caught the missing pole check.",
            }
        ],
    )

    class FakeRuntime:
        def __init__(self):
            self.config = SimpleNamespace(
                model_roles=SimpleNamespace(
                    fast_inference="gemini:fast",
                    default="gemini:default",
                )
            )
            self.prompts: list[str] = []

        def generate_draft(self, schema_cls, prompt, **kwargs):
            self.prompts.append(prompt)
            return SimpleNamespace(
                payload=AssessmentResult(
                    sub_skill_scores=[
                        SubSkillScore(name="verification", score=2, is_weak=True),
                    ],
                    critique="Verification is still fragile.",
                    retry_items=[],
                ),
                raw_response="",
                model="gemini:fast",
            )

    fake_runtime = FakeRuntime()

    with patch("pb.core.finish_assessment.sys.stdin.isatty", return_value=True), \
         patch.object(FinishAssessmentAgent, "is_available", return_value=True), \
         patch("pb.llm.runtime.LLMRuntime", return_value=fake_runtime), \
         patch("pb.cli.commands.execute.typer.prompt") as prompt_mock:
        result = FinishAssessmentAgent().run(session, _task(), "math")

    prompt_mock.assert_not_called()
    assert result is not None
    assert result.retry_items == ["math: verification needs targeted practice (scored 2/5)"]
    assert "EVIDENCE SIGNALS" in fake_runtime.prompts[0]
    assert "missing pole check" in fake_runtime.prompts[0]
