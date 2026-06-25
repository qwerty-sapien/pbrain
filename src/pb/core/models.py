# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Domain models for the productivity tool."""

from __future__ import annotations

import re
import random
from datetime import datetime
from typing import Optional
from uuid import uuid4 as _uuid4

from pydantic import BaseModel, Field, model_validator

from pb.core.enums import (
    BloomStage,
    EvidenceType,
    EnergyType,
    FeedbackSource,
    Horizon,
    PacketType,
    PracticeStage,
    ProjectStatus,
    ProjectType,
    SessionMode,
    TaskState,
)


def generate_slug(title: str) -> str:
    """Generate slug ID: snake_case title (<=40 chars) + 5-digit random suffix (D-09)."""
    normalized = re.sub(r"[^a-z0-9]+", "_", title.lower().strip())
    normalized = normalized.strip("_")[:40].rstrip("_") or "task"
    suffix = str(random.randint(0, 99999)).zfill(5)
    return f"{normalized}_{suffix}"


def generate_internal_id() -> str:
    """Generate UUID4 for internal entities (sessions, time_blocks, goals, tracks) (D-11)."""
    return str(_uuid4())


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.utcnow()


class GoalArc(BaseModel):
    """Long-horizon intent spanning months."""

    id: str = Field(default_factory=generate_internal_id)
    title: str
    domain: str = ""
    execution_mode: str = "mixed"
    study_framework: Optional[str] = None
    current_bloom_stage: Optional[BloomStage] = None
    target_bloom_stage: Optional[BloomStage] = None
    practice_framework: Optional[str] = None
    current_practice_stage: Optional[PracticeStage] = None
    target_practice_stage: Optional[PracticeStage] = None
    horizon: Horizon = Horizon.SIX_MONTH
    description: str = ""
    success_definition: str = ""
    framework: str = ""
    primary_metric: Optional[str] = None
    feedback_source: Optional[FeedbackSource] = None
    evidence_type: Optional[EvidenceType] = None
    metric_type: Optional[str] = None
    target_value: Optional[float] = None
    start_date: Optional[datetime] = None
    target_date: Optional[datetime] = None
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    generated_names: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Track(BaseModel):
    """Persistent development stream (e.g., German, Rust, Piano)."""

    id: str = Field(default_factory=generate_internal_id)
    name: str
    description: str = ""
    linked_goal_arc_ids: list[str] = Field(default_factory=list)
    cadence: Optional[str] = None
    priority_weight: float = 1.0
    active: bool = True
    generated_names: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Project(BaseModel):
    """Bounded execution domain."""

    id: str = Field(default_factory=lambda: generate_slug("project"))
    name: str
    project_type: ProjectType = ProjectType.BUILD
    track_id: Optional[str] = None
    repo_path: Optional[str] = None
    packet_path: str  # Required - enforced by INV-2
    status: ProjectStatus = ProjectStatus.READY
    next_review_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    generated_names: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Task(BaseModel):
    """Executable work item."""

    id: str = Field(default_factory=lambda: generate_slug("task"))
    project_id: Optional[str] = None
    title: str
    description: str = ""
    horizon: Horizon = Horizon.TODAY
    state: TaskState = TaskState.ACTIVE
    completion: int = 0
    paused_until: Optional[datetime] = None
    pause_reason: Optional[str] = None
    estimate_minutes: Optional[int] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    energy_type: EnergyType = EnergyType.DEEP
    # Phase 2: Priority model (D-22)
    impact: Optional[int] = None             # 1-5
    urgency_score: Optional[int] = None      # 1-5 (renamed to avoid clash with urgent bool)
    strategic_value: Optional[int] = None    # 1-5
    effort: Optional[int] = None             # 1-5
    important: Optional[bool] = None         # Eisenhower axis
    urgent: Optional[bool] = None            # Eisenhower axis
    energy_required: Optional[int] = None    # 1-5
    work_type: Optional[str] = None          # WorkType enum value or None
    due_date: Optional[datetime] = None
    scheduled_date: Optional[datetime] = None
    estimated_minutes: Optional[int] = None
    actual_minutes: Optional[int] = None

    @model_validator(mode="after")
    def _migrate_estimate_minutes(self) -> "Task":
        """Migrate legacy estimate_minutes into canonical estimated_minutes (WR-01)."""
        if self.estimated_minutes is None and self.estimate_minutes is not None:
            self.estimated_minutes = self.estimate_minutes
        return self
    linked_goal_arc_ids: list[str] = Field(default_factory=list)
    linked_track_ids: list[str] = Field(default_factory=list)
    packet_path: Optional[str] = None
    generated_names: dict[str, object] = Field(default_factory=dict)
    interruption_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None


class Session(BaseModel):
    """One execution interval or block."""

    id: str = Field(default_factory=generate_internal_id)
    task_id: str
    start_at: datetime = Field(default_factory=utc_now)
    end_at: Optional[datetime] = None
    mode: SessionMode = SessionMode.FOCUS
    branch: str = "study"
    goal_id: Optional[str] = None
    track_id: Optional[str] = None
    subject_scope: str = ""
    bloom_stage: Optional[BloomStage] = None
    target_bloom_stage: Optional[BloomStage] = None
    practice_stage: Optional[PracticeStage] = None
    drill_type: Optional[str] = None
    constraint: Optional[str] = None
    feedback_source: Optional[FeedbackSource] = None
    evidence_target: Optional[str] = None
    coach_cues: Optional[str] = None
    observed_errors: Optional[str] = None
    quality_rating: Optional[int] = None
    difficulty_rating: Optional[int] = None
    next_adjustment: Optional[str] = None
    intended_outcome: str = ""
    actual_outcome: Optional[str] = None
    generated_names: dict[str, object] = Field(default_factory=dict)
    interruption_count: int = 0
    llm_summary_used: bool = False
    # Phase 8: pre-session expectation and post-mortem data
    expectation: Optional[str] = None       # <=10-word pre-session goal
    completion_pct: Optional[int] = None    # 0-100
    distraction: Optional[int] = None       # 1-5


class Packet(BaseModel):
    """Markdown artifact for project or task context."""

    path: str
    packet_type: PacketType
    linked_entity_id: str
    updated_at: datetime = Field(default_factory=utc_now)


class Clip(BaseModel):
    """Captured web material or notes."""

    id: str = Field(default_factory=generate_internal_id)
    source_url: Optional[str] = None
    title: Optional[str] = None
    captured_text: str
    summary: Optional[str] = None
    linked_project_id: Optional[str] = None
    linked_task_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class TimeBlock(BaseModel):
    """Scheduled time block for planning."""

    id: str = Field(default_factory=generate_internal_id)
    task_id: str
    start_time: Optional[datetime] = None
    duration_minutes: int
    block_kind: str = "study"
    created_at: datetime = Field(default_factory=utc_now)
    # Phase 2: recurrence support (D-10)
    series_id: Optional[str] = None        # shared by all instances in a series
    recurrence_rule: Optional[str] = None  # 'daily' | 'weekly' | None


class DailyReviewResponse(BaseModel):
    """Response to a daily review question (per D-17)."""

    id: str = Field(default_factory=generate_internal_id)
    review_date: str  # ISO date string YYYY-MM-DD
    question_id: str  # energy, presence, best_window, blockers, alignment
    numeric_score: int  # 1-10, assigned by user or LLM (per D-06, D-12)
    text_response: Optional[str] = None  # NULL for direct numeric, populated for chat mode
    llm_rationale: Optional[str] = None  # NULL for direct numeric, populated when LLM scores
    created_at: datetime = Field(default_factory=utc_now)


class DailyDebrief(BaseModel):
    """5-section daily debrief per D-30."""

    id: str = Field(default_factory=generate_internal_id)
    review_date: str  # YYYY-MM-DD

    # A. Completion check
    top1_completed: Optional[str] = None  # "yes", "no", "partial"
    top3_completed: list[str] = Field(default_factory=list)  # titles of completed tasks
    what_shipped: Optional[str] = None  # one short answer

    # B. Friction check
    biggest_blocker: Optional[str] = None  # from predefined list
    blocker_note: Optional[str] = None  # optional one-liner

    # C. Energy check
    energy_morning: Optional[int] = None  # 1-5
    energy_midday: Optional[int] = None   # 1-5
    energy_evening: Optional[int] = None  # 1-5
    energy_task_match: Optional[str] = None  # "yes", "no", "partial"

    # D. Learning check
    learning_question: Optional[str] = None   # which rotating question
    learning_answer: Optional[str] = None     # free text
    learning_score: Optional[int] = None      # LLM-assigned 1-10
    learning_rationale: Optional[str] = None  # LLM rationale

    # E. Tomorrow setup
    tomorrow_top1: Optional[str] = None       # tentative top 1 task title
    tomorrow_next_action: Optional[str] = None  # one next action

    created_at: datetime = Field(default_factory=utc_now)


class ActionReminder(BaseModel):
    """Queued actionable reminder for click-to-open practice/study prompts."""

    id: str = Field(default_factory=generate_internal_id)
    title: str
    message: str
    target_command: str
    status: str = "pending"
    remind_at: datetime = Field(default_factory=utc_now)
    source_kind: str = "manual"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class GenerationProvenance(BaseModel):
    """Minimal audit record for LLM-generated artifacts."""

    id: str = Field(default_factory=generate_internal_id)
    artifact_kind: str
    artifact_id: str
    generated_by_model: str
    prompt_template_version: str
    source_scope: str = ""
    accepted_by_user: bool = False
    created_at: datetime = Field(default_factory=utc_now)


# -- Roadmap SC-1 convenience types (Phase 21 gap closure) --

# Goal is an alias for GoalArc — the roadmap SC says
# `import pb.core.models` must resolve Goal.
Goal = GoalArc


class Note(BaseModel):
    """Vault note metadata for the knowledge pipeline.

    Represents a note's identity and learning state as tracked by
    the vault indexer and lifecycle modules. Content lives on disk
    as markdown; this model holds the metadata shadow.
    """

    path: str  # relative to vault root, e.g. "piano/scales.md"
    title: str = ""
    domain: str = ""  # top-level vault folder, e.g. "piano"
    learning_stage: str = "new"  # new | learning | learnt | stale | archive
    source: Optional[str] = None  # e.g. "socratic", "scaffold", "manual"
    interaction_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Domain(BaseModel):
    """Knowledge domain grouping (e.g., piano, German, ML).

    Maps to a top-level vault folder with _state.md and _index.md.
    Used by study planner for decay thresholds and by scoring for
    domain-specific retrieval.
    """

    name: str  # e.g. "piano"
    vault_folder: str = ""  # relative path from vault root
    decay_days: int = 7  # domain-specific staleness threshold
    description: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class StudySession(BaseModel):
    """Structured study session view over the generic session record."""

    session_id: str
    topic: str = ""
    current_stage: Optional[BloomStage] = None
    target_stage: Optional[BloomStage] = None
    goal_id: Optional[str] = None
    domain: str = ""


class PractiseSession(BaseModel):
    """Structured practise session view over the generic session record."""

    session_id: str
    skill: str = ""
    drill_type: str = ""
    goal_id: Optional[str] = None
    domain: str = ""
    coach_cues: str = ""
