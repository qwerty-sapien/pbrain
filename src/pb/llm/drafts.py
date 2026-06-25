# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Validated draft schemas for LLM-backed learning workflows."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from pb.core.enums import (
    BloomStage,
    EvidenceContract,
    EvidenceType,
    FeedbackSource,
    PracticeStage,
    SessionFeedbackSource,
    SessionFrame,
    SkillKind,
)
from pb.core.renderables import RenderableText


class InstructionStep(BaseModel):
    """One ordered study or practice instruction."""

    title: str
    instruction: RenderableText = Field(default_factory=RenderableText)
    success_check: RenderableText = Field(default_factory=RenderableText)


class ArtifactPresentationDraft(BaseModel):
    """Curated terminal-safe presentation metadata for learner-facing previews."""

    variant: Literal["operator", "editorial", "minimal"] = "operator"
    density: Literal["compact", "balanced", "relaxed"] = "balanced"
    accent: Literal["cyan", "blue", "green", "yellow", "magenta"] = "cyan"
    roadmap_layout: Literal["dag_legend", "sequence"] = "dag_legend"
    heading_style: Literal["ruled", "bracketed", "plain"] = "ruled"


def artifact_presentation_prompt(*, include_dependency_layout: bool = False) -> str:
    """Return prompt guidance for safe CLI presentation metadata."""

    lines = [
        "Optional presentation metadata: include a `presentation` object only if it helps terminal readability.",
        "Choose enum values only. Do not emit ANSI escape codes, CSS, HTML, Markdown layout instructions, or prose about styling.",
        "`presentation.variant` must be one of: operator, editorial, minimal.",
        "`presentation.density` must be one of: compact, balanced, relaxed.",
        "`presentation.accent` must be one of: cyan, blue, green, yellow, magenta.",
        "`presentation.heading_style` must be one of: ruled, bracketed, plain.",
    ]
    if include_dependency_layout:
        lines.append(
            "`presentation.roadmap_layout` must be one of: dag_legend, sequence. Prefer dag_legend unless a linear sequence is clearly better."
        )
    else:
        lines.append("If you include `presentation.roadmap_layout`, keep it as dag_legend.")
    return "\n".join(lines) + "\n"


class GoalDraft(BaseModel):
    """Structured goal proposal derived from messy user input."""

    title: str
    description: str = ""
    domain: str
    execution_mode: Literal["study", "practise", "mixed"]
    horizon: Literal["month", "quarter", "six_month"] = "six_month"
    framework: str = ""
    study_framework: Optional[Literal["bloom_retrieval"]] = None
    current_bloom_stage: Optional[BloomStage] = None
    target_bloom_stage: Optional[BloomStage] = None
    practice_framework: Optional[Literal["deliberate_practice"]] = None
    current_practice_stage: Optional[PracticeStage] = None
    target_practice_stage: Optional[PracticeStage] = None
    success_definition: str
    primary_metric: Optional[str] = None
    feedback_source: Optional[FeedbackSource] = None
    evidence_type: Optional[EvidenceType] = None


class GoalRoadmapNodeDraft(BaseModel):
    """One narrow project phase / task in a goal roadmap."""

    node_id: str
    title: str
    branch: Literal["study", "practise"] = "study"
    scope: str = ""
    milestone: str = ""
    success_check: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    clarification_prompts: list[str] = Field(default_factory=list)


class GoalRoadmapDraft(BaseModel):
    """A reviewed goal roadmap before task materialization."""

    summary: str = ""
    progression_mode: Literal["adaptive", "aggressive", "conservative"] = "adaptive"
    project_title: str = ""
    presentation: ArtifactPresentationDraft = Field(default_factory=ArtifactPresentationDraft)
    nodes: list[GoalRoadmapNodeDraft] = Field(default_factory=list)


class LearningSessionBlueprintDraft(BaseModel):
    """Resolved session blueprint that governs a live learning interaction."""

    domain: str
    topic: str
    skill_kind: SkillKind
    primary_frame: SessionFrame
    secondary_frames: list[SessionFrame] = Field(default_factory=list)
    subskills: list[str] = Field(default_factory=list)
    evidence_contract: list[EvidenceContract] = Field(default_factory=list)
    feedback_sources: list[SessionFeedbackSource] = Field(default_factory=list)
    opening_move: str = ""
    stop_condition: str = ""
    coach_rules: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


class ChallengeAssessmentDraft(BaseModel):
    """Verdict for a challenge/placement attempt against a future phase."""

    target_node_id: str
    verdict: Literal["advance", "hold", "remediate"]
    competency_score: float = Field(ge=0.0, le=1.0, default=0.0)
    bypass_node_ids: list[str] = Field(default_factory=list)
    remediation_requirements: list[str] = Field(default_factory=list)


class LearningPlanBlockDraft(BaseModel):
    """One executable study or practise block."""

    goal_id: Optional[str] = None
    branch: Literal["study", "practise"]
    title: str = ""
    subject_scope: str
    duration_minutes: int = Field(ge=5, le=240)
    target_bloom_stage: Optional[BloomStage] = None
    study_mode: Optional[str] = None
    practice_stage: Optional[PracticeStage] = None
    drill_type: Optional[str] = None
    constraint: str = ""
    feedback_source: Optional[FeedbackSource] = None
    evidence_target: str = ""
    coach_cues: str = ""
    success_check: str = ""
    reason: str = ""
    steps: list[InstructionStep] = Field(default_factory=list)
    node_id: Optional[str] = None
    depends_on: list[str] = Field(default_factory=list)
    sub_index: Optional[str] = None
    domain_pack_id: str = ""
    session_blueprint: Optional[LearningSessionBlueprintDraft] = None


class StudyPlanDraft(BaseModel):
    """A conceptual-study plan proposal."""

    summary: str = ""
    presentation: ArtifactPresentationDraft = Field(default_factory=ArtifactPresentationDraft)
    blocks: list[LearningPlanBlockDraft]


class PractisePlanDraft(BaseModel):
    """A deliberate-practice plan proposal."""

    summary: str = ""
    presentation: ArtifactPresentationDraft = Field(default_factory=ArtifactPresentationDraft)
    blocks: list[LearningPlanBlockDraft]


class MixedPlanDraft(BaseModel):
    """A mixed study and practise plan proposal."""

    summary: str = ""
    presentation: ArtifactPresentationDraft = Field(default_factory=ArtifactPresentationDraft)
    blocks: list[LearningPlanBlockDraft]


class CurriculumPlanDraft(BaseModel):
    """A clarified multi-step learning plan with linked dependencies."""

    summary: str = ""
    learner_state: str = ""
    presentation: ArtifactPresentationDraft = Field(default_factory=ArtifactPresentationDraft)
    blocks: list[LearningPlanBlockDraft]


class RecallPromptItem(BaseModel):
    """One recall prompt."""

    prompt: RenderableText = Field(default_factory=RenderableText)
    answer: RenderableText = Field(default_factory=RenderableText)
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    source_note: str = ""


class RecallPromptDraft(BaseModel):
    """Scoped recall prompts for a study loop."""

    scope: str
    summary: str = ""
    prompts: list[RecallPromptItem]


class AnkiCandidateItem(BaseModel):
    """One candidate Anki card."""

    front: RenderableText = Field(default_factory=RenderableText)
    back: RenderableText = Field(default_factory=RenderableText)
    note_type: str = "Basic"
    sub_deck: str = "Concepts"
    rationale: str = ""


class AnkiCandidateDraft(BaseModel):
    """LLM-proposed Anki candidates for a scoped topic."""

    scope: str
    deck: str
    summary: str = ""
    cards: list[AnkiCandidateItem]


class StudyDebriefDraft(BaseModel):
    """Structured closeout for conceptual work."""

    summary: str
    key_insights: list[str] = Field(default_factory=list)
    recurring_gaps: list[str] = Field(default_factory=list)
    recommended_recall: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class PractiseDebriefDraft(BaseModel):
    """Structured closeout for deliberate practice."""

    summary: str
    what_happened: str
    degraded: str = ""
    helpful_cue: str = ""
    next_adjustment: str
    recurring_errors: list[str] = Field(default_factory=list)


class DailyReviewDraft(BaseModel):
    """Daily synthesis over deterministic learning metrics."""

    summary: str
    progress_signals: list[str] = Field(default_factory=list)
    friction_patterns: list[str] = Field(default_factory=list)
    evidence_captured: list[str] = Field(default_factory=list)
    next_adjustments: list[str] = Field(default_factory=list)


class WeeklyReviewDraft(BaseModel):
    """Weekly synthesis over deterministic learning metrics."""

    summary: str
    wins: list[str] = Field(default_factory=list)
    stalls: list[str] = Field(default_factory=list)
    evidence_progress: list[str] = Field(default_factory=list)
    friction_patterns: list[str] = Field(default_factory=list)
    next_week_focus: list[str] = Field(default_factory=list)


class NameConfidenceDraft(BaseModel):
    """Confidence metadata for lightweight routing and naming."""

    score: float = Field(ge=0.0, le=1.0, default=0.0)
    info_density: float = Field(ge=0.0, le=1.0, default=0.0)
    routing_relevance: float = Field(ge=0.0, le=1.0, default=0.0)


class GeneratedNamesDraft(BaseModel):
    """Structured titles and routing metadata for persisted learning artifacts."""

    display_title: str = ""
    short_title: str = ""
    slug: str = ""
    note_title: str = ""
    folder_name: str = ""
    session_title: str = ""
    task_title: str = ""
    plan_title: str = ""
    goal_title: str = ""
    frontmatter: dict[str, object] = Field(default_factory=dict)
    confidence: NameConfidenceDraft = Field(default_factory=NameConfidenceDraft)


class ClarifierQuestionDraft(BaseModel):
    """One contextual clarifying question."""

    question: str
    reason: str = ""
    answer_type: str = "short_text"
    optional: bool = False
    option_candidates: list[str] = Field(default_factory=list)
    why_this_matters: str = ""
    inferred_signal_type: str = ""
    downstream_effect: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ClarifierQuestionSetDraft(BaseModel):
    """Small batch of contextual clarifying questions."""

    questions: list[ClarifierQuestionDraft] = Field(default_factory=list)


class LearningPartnerTurnDraft(BaseModel):
    """One conversational turn from the learning partner."""

    reply: str
    corrections: list[str] = Field(default_factory=list)
    detected_gaps: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    friction: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    recall_candidates: list[str] = Field(default_factory=list)
    next_drill: str = ""
    next_action: str = ""
    current_step_index: int = 0
    total_steps: int = 0
    current_step_title: str = ""
    current_objective: str = ""
    success_check: str = ""
    retry_focus: str = ""
    step_status: str = ""
    advance_step: bool = False
    support_cards: list[str] = Field(default_factory=list)
    # Question type routing for MCQ/cloze per D-14 through D-17
    question_type: Literal[
        "free_text",
        "mcq",
        "multi_select",
        "cloze",
        "short_text",
        "free_production",
        "error_correction",
        "reorder",
    ] = "free_text"
    mcq_options: list[str] = Field(default_factory=list)
    cloze_blank_options: list[str] = Field(default_factory=list)


class LessonQuestionDraft(BaseModel):
    """One question planned for a unified lesson page."""

    title: str = ""
    prompt: str
    question_type: Literal[
        "mcq",
        "multi_select",
        "cloze",
        "short_text",
        "free_production",
        "error_correction",
        "reorder",
    ]
    skill_slug: str = ""
    skill_label: str = ""
    choices: list[str] = Field(default_factory=list)
    accepted_answers: list[str] = Field(default_factory=list)
    correct_choices: list[str] = Field(default_factory=list)
    ordered_items: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    reveal_answer: str = ""
    error_tags: list[str] = Field(default_factory=list)
    evaluator_notes: str = ""
    page_intro: str = ""


class LessonPageDraft(BaseModel):
    """One page in a unified lesson plan."""

    title: str
    focus: str = ""
    intro: str = ""
    questions: list[LessonQuestionDraft] = Field(default_factory=list)


class LessonPlanDraft(BaseModel):
    """Planned page/question structure for a live lesson session."""

    lesson_title: str
    summary: str = ""
    mode: Literal["teach", "study", "practise"]
    pages: list[LessonPageDraft] = Field(default_factory=list)


class LessonEvaluationDraft(BaseModel):
    """Structured evaluation for ambiguous short or free responses."""

    result: Literal["correct", "close", "wrong"]
    feedback: str = ""
    hint: str = ""
    stronger_hint: str = ""
    reveal_answer: str = ""
    error_tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LessonNoteDraft(BaseModel):
    """Structured final note for teach/study lesson write-ups."""

    title: str
    summary: str = ""
    intuitive_explanation: str = ""
    key_points: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    next_moves: list[str] = Field(default_factory=list)


class CloseoutDecisionDraft(BaseModel):
    """Adaptive finish classification for learning sessions."""

    status: Literal[
        "completed",
        "partial",
        "blocked",
        "abandoned",
        "no_progress",
        "frustration_feedback",
        "accidental_start",
    ]
    summary: str = ""
    recovery_step: str = ""
    feedback_note: str = ""
    discard_recommended: bool = False


class FeedbackProposalDraft(BaseModel):
    """Previewable preference and workflow patch suggestions from user feedback."""

    summary: str = ""
    preference_patches: dict[str, object] = Field(default_factory=dict)
    workflow_patches: list[str] = Field(default_factory=list)


class AgentInstructionJudgeDraft(BaseModel):
    """Structured Phase 13 decision for specialised-agent instruction patches."""

    action: Literal["patch", "clarify", "none"] = "none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    instruction_patch: str = Field(default="", max_length=900)
    clarifying_question: str = Field(default="", max_length=300)
    evidence_citations: list[str] = Field(default_factory=list)
