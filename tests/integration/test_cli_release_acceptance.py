"""Release-style CLI acceptance journey for the learning-first workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from pb.cli.main import app
from pb.llm.drafts import (
    DailyReviewDraft,
    GoalDraft,
    GoalRoadmapDraft,
    GoalRoadmapNodeDraft,
    LearningPlanBlockDraft,
    PractisePlanDraft,
    RecallPromptDraft,
    RecallPromptItem,
    StudyPlanDraft,
    WeeklyReviewDraft,
)
from pb.llm.runtime import GeneratedDraft


runner = CliRunner()


@dataclass
class CommandLog:
    command: list[str]
    exit_code: int
    output: str


def _fake_generate_draft(self, schema_cls, prompt, **kwargs):
    source_scope = kwargs.get("source_scope", "test")
    if schema_cls is GoalDraft:
        raw = source_scope.split(":", 1)[-1].strip() or "Learning goal"
        title = " ".join(word.capitalize() for word in raw.split())
        domain = raw.split()[0].lower()
        payload = GoalDraft(
            title=title,
            description=f"Structured goal for {raw}.",
            domain=domain,
            execution_mode="mixed" if len(raw.split()) <= 3 else "study",
            horizon="quarter",
            framework="Bloom-first CLI loop",
            study_framework="bloom_retrieval",
            target_bloom_stage="apply",
            practice_framework="deliberate_practice",
            target_practice_stage="integrate",
            success_definition=f"Make concrete progress on {raw}.",
            feedback_source="artifact",
            evidence_type="artifact",
        )
    elif schema_cls is StudyPlanDraft:
        topic = source_scope.split(":", 1)[-1].strip() or "study topic"
        payload = StudyPlanDraft(
            summary="",
            blocks=[
                LearningPlanBlockDraft(
                    branch="study",
                    subject_scope=topic,
                    duration_minutes=30,
                    target_bloom_stage="apply",
                    study_mode="active recall",
                    success_check=f"Explain the key ideas behind {topic}.",
                    reason=f"Advance conceptual understanding of {topic}.",
                )
            ],
        )
    elif schema_cls is PractisePlanDraft:
        skill = source_scope.split(":", 1)[-1].strip() or "practice topic"
        payload = PractisePlanDraft(
            summary="",
            blocks=[
                LearningPlanBlockDraft(
                    branch="practise",
                    subject_scope=skill,
                    duration_minutes=25,
                    practice_stage="integrate",
                    drill_type=skill,
                    constraint="No hints on the first rep.",
                    feedback_source="artifact",
                    evidence_target=f"One concrete artifact for {skill}.",
                    coach_cues="Slow down before speeding up.",
                    success_check=f"Finish one clean deliberate-practice block for {skill}.",
                    reason=f"Advance deliberate practice for {skill}.",
                )
            ],
        )
    elif schema_cls is RecallPromptDraft:
        scope = source_scope.split(":", 1)[-1].strip() or "scope"
        payload = RecallPromptDraft(
            scope=scope,
            summary="",
            prompts=[
                RecallPromptItem(
                    prompt=f"What matters most about {scope}?",
                    answer=f"The core principle behind {scope}.",
                    difficulty="medium",
                    source_note=f"knowledge/{scope}/_index.md",
                )
            ],
        )
    elif schema_cls is DailyReviewDraft:
        payload = DailyReviewDraft(
            summary="A compact daily learning review.",
            progress_signals=["Completed a concrete study block."],
            friction_patterns=["Keep scopes tighter when energy dips."],
            evidence_captured=["Captured at least one practise artifact."],
            next_adjustments=["Start tomorrow with one explicit block."],
        )
    elif schema_cls is WeeklyReviewDraft:
        payload = WeeklyReviewDraft(
            summary="A compact weekly learning review.",
            wins=["Maintained both study and practise loops."],
            stalls=["Some sessions still need tighter scopes."],
            evidence_progress=["Recall and practise evidence both moved forward."],
            friction_patterns=["Ambiguity cost energy on broader topics."],
            next_week_focus=["Choose one explicit block before opening new tabs."],
        )
    elif schema_cls is GoalRoadmapDraft:
        topic = source_scope.split(":", 1)[-1].strip() or "topic"
        payload = GoalRoadmapDraft(
            summary=f"Roadmap for {topic}.",
            project_title=topic.title(),
            nodes=[GoalRoadmapNodeDraft(
                node_id="n1", title=f"Foundation: {topic}",
                branch="study", scope=topic,
                success_check=f"Explain core concepts of {topic}.",
            )],
        )
    else:
        from pb.llm.drafts import GeneratedNamesDraft
        if schema_cls is GeneratedNamesDraft:
            topic = source_scope.split(":", 1)[-1].strip() or "topic"
            payload = GeneratedNamesDraft(
                display_title=topic.title(),
                short_title=topic[:20],
                slug=topic.lower().replace(" ", "-")[:30],
                note_title=topic.title(),
                session_title=topic.title(),
                task_title=topic.title(),
            )
        else:
            payload = schema_cls()

    return GeneratedDraft(
        payload=payload,
        model="gemini:gemini-3-flash-preview",
        source_scope=source_scope,
        prompt_template_version="test",
        raw_response="{}",
    )


def _invoke(command: list[str], transcript: list[CommandLog]) -> str:
    result = runner.invoke(app, command)
    transcript.append(CommandLog(command=command, exit_code=result.exit_code, output=result.output))
    assert result.exit_code == 0, f"{command} failed:\n{result.output}"
    return result.output


def test_cli_release_acceptance_journey(tmp_path, monkeypatch, temp_db, temp_config):
    """Exercise 30+ CLI invocations across the core learning workflow."""

    config_path = tmp_path / "config.toml"
    vault_path = tmp_path / "vault"
    monkeypatch.setenv("PRODUCTIVEBRAIN_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    transcript: list[CommandLog] = []

    timer_patches = (
        patch("pb.sessions.service.TimerManager.start_session_timers", return_value=None),
        patch("pb.sessions.service.TimerManager.stop_session_timers", return_value=None),
        patch("pb.cli.commands.next.schedule_actionable_notification", return_value=True),
        patch("pb.llm.runtime.LLMRuntime.generate_draft", new=_fake_generate_draft),
    )

    with timer_patches[0], timer_patches[1], timer_patches[2], timer_patches[3]:
        _invoke(
            [
                "init",
                "--non-interactive",
                "--vault-name",
                "main",
                "--vault-path",
                str(vault_path),
                "--provider",
                "gemini",
                "--model",
                "gemini-3-flash-preview",
                "--yes",
            ],
            transcript,
        )
        _invoke(["vault", "current", "--json"], transcript)
        _invoke(["thought", "Need", "a", "clean", "ML", "note"], transcript)
        _invoke(["thought", "Potential", "book", "chapter", "hook"], transcript)
        _invoke(["todo", "Email advisor /due 2026-05-21"], transcript)
        _invoke(["todo", "Outline practice schedule"], transcript)
        _invoke(["notes", "inbox"], transcript)
        _invoke(["goal", "add", "--yes", "ML foundations"], transcript)
        _invoke(["goal", "tracks"], transcript)
        _invoke(["next"], transcript)
        _invoke(["do", "what should I learn next"], transcript)
        _invoke(["plan", "day", "--quick", "--yes"], transcript)
        _invoke(["learn", "--yes", "--study", "linear", "algebra"], transcript)
        _invoke(["now"], transcript)
        _invoke(["finish", "understood", "matrix", "rank"], transcript)
        _invoke(["learn", "--yes", "-p", "piano", "scales"], transcript)
        _invoke(["now"], transcript)
        _invoke(["finish", "partial", "needs", "cleaner", "transitions"], transcript)
        _invoke(["review", "day"], transcript)
        _invoke(["review", "week"], transcript)
        _invoke(["study", "--yes", "probability", "basics"], transcript)
        _invoke(["finish", "recalled", "Bayes", "rule"], transcript)
        _invoke(["practise", "--yes", "piano", "arpeggios"], transcript)
        _invoke(["finish", "better", "tempo", "control"], transcript)
        _invoke(["study", "recall", "ml", "--yes"], transcript)
        _invoke(["next", "--schedule", "15"], transcript)
        _invoke(["next"], transcript)
        _invoke(["goal", "add", "--yes", "Writing craft"], transcript)
        _invoke(["plan", "day", "--quick", "--yes"], transcript)
        _invoke(["do", "capture a thought about a chapter structure"], transcript)
        _invoke(["thought", "Chapter structure should pivot earlier"], transcript)
        _invoke(["todo", "Draft intro @[2026-05-22]"], transcript)

    assert len(transcript) >= 30
    assert any("Created goal" in entry.output for entry in transcript)
    assert any("Started:" in entry.output for entry in transcript)
    assert any("Finished:" in entry.output for entry in transcript)
    assert any("Daily Review" in entry.output for entry in transcript)
    assert any("Weekly Reflection" in entry.output or "Weekly Review" in entry.output for entry in transcript)
    assert (vault_path / "Learning" / "Inbox" / "pb" / "thoughts").exists()
