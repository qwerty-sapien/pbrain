# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Domain enumerations for the productivity tool."""

from enum import Enum


class TaskState(str, Enum):
    """Task lifecycle states: active (workable), paused (postponed N days), done (complete)."""

    ACTIVE = "active"
    PAUSED = "paused"
    DONE = "done"


class SessionMode(str, Enum):
    """Session execution modes."""

    FOCUS = "focus"
    SUPERVISORY = "supervisory"
    REVIEW = "review"
    PRACTICE = "practice"


class BloomStage(str, Enum):
    """Bloom's Taxonomy stages for study-oriented work."""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class PracticeStage(str, Enum):
    """Deliberate-practice stages for skill execution work."""

    ORIENT = "orient"
    ISOLATE = "isolate"
    INTEGRATE = "integrate"
    PERFORM = "perform"
    ADAPT = "adapt"


class SkillKind(str, Enum):
    """Compact cross-domain skill families for session behavior."""

    CONCEPTUAL = "conceptual"
    PROCEDURAL_COGNITIVE = "procedural_cognitive"
    PROCEDURAL_MOTOR = "procedural_motor"
    PERCEPTUAL = "perceptual"
    LANGUAGE = "language"
    CREATIVE_ARTIFACT = "creative_artifact"
    ENGINEERING_BUILD_DEBUG = "engineering_build_debug"
    EXPERIMENTAL_LAB = "experimental_lab"
    MIXED = "mixed"


class SessionFrame(str, Enum):
    """Reusable learning-session structures across domains."""

    RETRIEVAL_PROBE = "retrieval_probe"
    WORKED_EXAMPLE_FADING = "worked_example_fading"
    ERROR_ANALYSIS = "error_analysis"
    MECHANISM_TRACING = "mechanism_tracing"
    MODEL_OR_PROOF_BUILDING = "model_or_proof_building"
    CASE_APPLICATION = "case_application"
    DELIBERATE_REP_LOOP = "deliberate_rep_loop"
    FAILURE_POINT_ISOLATION = "failure_point_isolation"
    PERCEPTUAL_DISCRIMINATION = "perceptual_discrimination"
    ARTIFACT_RECREATION = "artifact_recreation"
    PERFORMANCE_SIMULATION = "performance_simulation"
    EXPERIMENT_VARIABLE_ISOLATION = "experiment_variable_isolation"


class EvidenceContract(str, Enum):
    """Concrete forms of learner evidence that a session can require."""

    FREE_TEXT_ANSWER = "free_text_answer"
    WORKED_SOLUTION = "worked_solution"
    MCQ = "mcq"
    CLOZE = "cloze"
    REP_COUNT = "rep_count"
    TIMED_HOLD = "timed_hold"
    RECORDING = "recording"
    ARTIFACT_PATH = "artifact_path"
    BEFORE_AFTER_COMPARISON = "before_after_comparison"
    SENSORY_LOG = "sensory_log"
    DEBUG_TRACE = "debug_trace"
    SELF_RUBRIC = "self_rubric"
    QUIZ_SCORE = "quiz_score"


class SessionFeedbackSource(str, Enum):
    """Feedback sources for a blueprint-driven learning session."""

    SELF_CHECK = "self_check"
    LLM_CHECK = "llm_check"
    ANSWER_KEY = "answer_key"
    COMPILER_OR_TEST = "compiler_or_test"
    RECORDING_REVIEW = "recording_review"
    COACH_OR_PEER = "coach_or_peer"
    PHYSICAL_FEEDBACK = "physical_feedback"
    EXTERNAL_METRIC = "external_metric"


class FeedbackSource(str, Enum):
    """Primary feedback source for a practice or learning loop."""

    SELF = "self"
    COACH = "coach"
    TESTS = "tests"
    RECORDING = "recording"
    PEER = "peer"
    AUTOMATED = "automated"
    ARTIFACT = "artifact"


class EvidenceType(str, Enum):
    """Primary evidence type expected from a learning loop."""

    RECALL = "recall"
    ANKI = "anki"
    TEST = "test"
    ARTIFACT = "artifact"
    RECORDING = "recording"
    REPS = "reps"
    RUBRIC = "rubric"
    TIMEBOXED_OUTPUT = "timeboxed_output"


class EnergyType(str, Enum):
    """Energy classification for tasks."""

    DEEP = "deep"
    SHALLOW = "shallow"
    SUPERVISORY = "supervisory"
    ADMIN = "admin"
    PRACTICE = "practice"


class Horizon(str, Enum):
    """Planning horizons."""

    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    SIX_MONTH = "six_month"


class ProjectType(str, Enum):
    """Project classification."""

    BUILD = "build"
    STUDY = "study"
    PRACTICE = "practice"
    ADMIN = "admin"
    RESEARCH = "research"


class ProjectStatus(str, Enum):
    """Project lifecycle status."""

    READY = "ready"
    ACTIVE = "active"
    WAITING = "waiting"
    BLOCKED = "blocked"
    ARCHIVED = "archived"


class PacketType(str, Enum):
    """Knowledge packet types."""

    PROJECT = "project"
    TASK = "task"
    HANDOFF = "handoff"
    REVIEW = "review"
    CLIP = "clip"


class TaskOutcome(str, Enum):
    """Outcome classification for completed tasks."""

    DONE = "done"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


class WorkType(str, Enum):
    """Work type classification per D-22."""

    DEEP = "deep"
    SHALLOW = "shallow"
    ADMIN = "admin"
    MEETING = "meeting"
    RECOVERY = "recovery"
    PLANNING = "planning"


class EisenhowerClass(str, Enum):
    """Eisenhower matrix classification per D-24."""

    DO_TODAY = "do_today"
    SCHEDULE_DEEP_WORK = "schedule_deep_work"
    BATCH_DELEGATE_OR_AUTOMATE = "batch_delegate_or_automate"
    DELETE_OR_DEFER = "delete_or_defer"


class PriorityAction(str, Enum):
    """Priority action thresholds per D-25."""

    SCHEDULE_FIRST = "schedule_first"
    SCHEDULE_IF_CAPACITY = "schedule_if_capacity"
    BATCH_DELEGATE_SIMPLIFY = "batch_delegate_simplify"
    DROP_OR_DEFER = "drop_or_defer"
