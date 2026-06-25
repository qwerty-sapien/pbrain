# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Session lifecycle service.

Migrated from pb.core.sessions.SessionManager in Phase 23.
All methods fully implemented; Phase 21 stubs removed.
INV-4: no typer or rich imports.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol

import structlog

from pb.core.base import BaseService, LoggableMixin
from pb.core.enums import BloomStage, FeedbackSource, PracticeStage
from pb.core.models import Session, Task
from pb.core.timer import TimerManager
from pb.core.enums import SessionMode, TaskState
from pb.core.rules import RuleViolation


class SessionRepo(Protocol):
    """Protocol for session persistence."""

    def get_active_session(self) -> Session | None: ...
    def create_session(self, session: Session) -> Session: ...
    def update_session(self, session: Session) -> Session: ...
    def delete_session(self, session_id: str) -> bool: ...
    def delete_sessions_for_task(self, task_id: str) -> int: ...
    def list_sessions(self, task_id: str | None = None) -> list[Session]: ...
    def get_task(self, task_id: str) -> Task | None: ...
    def update_task(self, task: Task) -> None: ...
    def force_delete_task(self, task_id: str) -> bool: ...
    def list_sessions_for_task(self, task_id: str) -> list[Session]: ...
    def delete_time_blocks_for_task(self, task_id: str) -> int: ...
    def delete_generation_provenance(
        self,
        *,
        artifact_kind: str | None = None,
        artifact_id: str | None = None,
    ) -> int: ...
    def resume_pause_interval(self, session_id: str) -> None: ...
    def list_time_blocks_for_date(self, dt: object) -> list: ...


class SessionService(BaseService, LoggableMixin):
    """Manages work session lifecycle: start, pause, finish, timing.

    Constructor takes explicit deps per D-05. Timer is optional --
    defaults to TimerManager() if not provided.
    """

    def __init__(self, repo: SessionRepo, timer: Optional[TimerManager] = None):
        super().__init__()
        self.repo = repo
        self.timer = timer or TimerManager()
        self._log = structlog.get_logger()

    def start_session(
        self,
        task_id: str,
        mode: str = "focus",
        duration_minutes: int | None = None,
        timer_mode: str = "stopwatch",
        branch: str = "study",
        goal_id: str | None = None,
        track_id: str | None = None,
        subject_scope: str = "",
        bloom_stage: BloomStage | str | None = None,
        target_bloom_stage: BloomStage | str | None = None,
        practice_stage: PracticeStage | str | None = None,
        drill_type: str | None = None,
        constraint: str | None = None,
        feedback_source: FeedbackSource | str | None = None,
        evidence_target: str | None = None,
        coach_cues: str | None = None,
    ) -> Session:
        """Start a new work session for the given task.

        Args:
            task_id: ID of the task to work on.
            mode: Session mode string (default 'focus').
            duration_minutes: Timer countdown duration; None = stopwatch.
            timer_mode: 'timer' | 'stopwatch' (stored on session row).

        Returns:
            The created Session.

        Raises:
            RuleViolation: If another session is active, task not found,
                           task already complete, or task is snoozed.
        """
        # Auto-close stale paused sessions first (D-13)
        self._cancel_stale_pauses()

        task = self.repo.get_task(task_id)
        if task is None:
            raise RuleViolation(f"Task not found: {task_id}")

        if task.completion >= 100:
            raise RuleViolation("Task is already complete.")

        if task.state == TaskState.PAUSED:
            raise RuleViolation(
                f"Task is paused until {task.paused_until}. Un-pause it first."
            )

        active_session = self.repo.get_active_session()
        if active_session is not None:
            raise RuleViolation(
                "Cannot start session: another session is already active. "
                "Pause or finish the current session first."
            )

        # Close open pause interval from a previous session pause (non-fatal)
        try:
            task_sessions = self.repo.list_sessions_for_task(task_id)
            if task_sessions:
                last_session = task_sessions[-1]
                self.repo.resume_pause_interval(last_session.id)
        except Exception:
            pass

        session = Session(
            task_id=task_id,
            mode=SessionMode[mode.upper()],
            branch=branch,
            goal_id=goal_id,
            track_id=track_id,
            subject_scope=subject_scope,
            bloom_stage=BloomStage(bloom_stage) if isinstance(bloom_stage, str) and bloom_stage else bloom_stage,
            target_bloom_stage=BloomStage(target_bloom_stage) if isinstance(target_bloom_stage, str) and target_bloom_stage else target_bloom_stage,
            practice_stage=PracticeStage(practice_stage) if isinstance(practice_stage, str) and practice_stage else practice_stage,
            drill_type=drill_type,
            constraint=constraint,
            feedback_source=FeedbackSource(feedback_source) if isinstance(feedback_source, str) and feedback_source else feedback_source,
            evidence_target=evidence_target,
            coach_cues=coach_cues,
            intended_outcome="",
        )

        # Store Phase 23 timer fields as dynamic attributes.
        # The DB columns (duration_minutes, timer_mode) exist from the migration;
        # Repository maps them if they are present on the model via __dict__ access.
        # Pydantic models allow extra fields when model_config allows it; we use
        # object.__setattr__ to bypass pydantic validation for these extra columns.
        try:
            object.__setattr__(session, "duration_minutes", duration_minutes)
            object.__setattr__(session, "timer_mode", timer_mode)
        except Exception:
            pass  # Non-fatal if model is immutable

        self.repo.create_session(session)

        # Activate task
        task.state = TaskState.ACTIVE
        self.repo.update_task(task)

        # Start OS-level timers (caffeinate + at-job notification)
        self.timer.start_session_timers(
            session_id=session.id,
            duration_minutes=duration_minutes,
            task_title=task.title,
        )

        self._log.info(
            "session.started",
            task_id=task_id,
            session_id=session.id,
            duration_minutes=duration_minutes,
            timer_mode=timer_mode,
        )
        return session

    def pause_session(self, outcome: str | None = None) -> Session | None:
        """Pause the current active session.

        Args:
            outcome: Optional note on what was accomplished so far.

        Returns:
            The paused session, or None if no active session.
        """
        session = self.repo.get_active_session()
        if session is None:
            return None

        session.end_at = datetime.utcnow()
        session.actual_outcome = outcome
        self.repo.update_session(session)
        self.timer.stop_session_timers()

        # Record pause interval for trend tracking (D-12 compat, non-fatal)
        try:
            self.repo.resume_pause_interval(session.id)
        except Exception:
            pass

        return session

    def finish_session(
        self,
        note: str | None = None,
        completion_pct: int = 100,
    ) -> Session | None:
        """Finish the current active session.

        Args:
            note: Optional one-line note capturing what happened.
            completion_pct: Completion percentage 0-100 (default 100).

        Returns:
            The finished session, or None if no active session.
        """
        session = self.repo.get_active_session()
        if session is None:
            return None

        session.end_at = datetime.utcnow()
        session.actual_outcome = note or "done"
        session.completion_pct = completion_pct
        self.repo.update_session(session)

        task = self.repo.get_task(session.task_id)
        if task is not None:
            task.completion = completion_pct
            if completion_pct >= 100:
                task.state = TaskState.DONE
                task.completed_at = datetime.utcnow()
            self.repo.update_task(task)

        self.timer.stop_session_timers()

        self._log.info(
            "session.finished",
            session_id=session.id,
            completion_pct=completion_pct,
        )
        return session

    def get_current_session(self) -> Session | None:
        """Get the currently active session."""
        return self.repo.get_active_session()

    def get_current_task(self) -> Task | None:
        """Get the task for the currently active session."""
        session = self.repo.get_active_session()
        if session is None:
            return None
        return self.repo.get_task(session.task_id)

    def discard_session(self) -> Session | None:
        """Forget the current active session without changing task progress."""
        session = self.repo.get_active_session()
        if session is None:
            return None
        self.timer.stop_session_timers()
        self.repo.delete_session(session.id)
        return session

    def reset_task_for_later(self, task_id: str) -> Task | None:
        """Forget task runtime history, reset progress, and postpone indefinitely."""
        active_session = self.repo.get_active_session()
        if active_session is not None and active_session.task_id == task_id:
            self.timer.stop_session_timers()

        task = self.repo.get_task(task_id)
        if task is None:
            return None

        self.repo.delete_sessions_for_task(task_id)
        self.repo.delete_time_blocks_for_task(task_id)
        self.repo.delete_generation_provenance(artifact_id=task_id)

        task.completion = 0
        task.completed_at = None
        task.state = TaskState.PAUSED
        task.paused_until = None
        task.pause_reason = "Later"
        self.repo.update_task(task)
        return task

    def delete_task_permanently(self, task_id: str) -> Task | None:
        """Delete a task and its runtime history."""
        active_session = self.repo.get_active_session()
        if active_session is not None and active_session.task_id == task_id:
            self.timer.stop_session_timers()

        task = self.repo.get_task(task_id)
        if task is None:
            return None

        self.repo.force_delete_task(task_id)
        return task

    def get_elapsed_minutes(self) -> int | None:
        """Get elapsed minutes since session start.

        Tries in-process timer first; falls back to DB computation
        from session.start_at for cross-process safety.
        """
        in_process = self.timer.get_elapsed_minutes()
        if in_process is not None:
            return in_process

        session = self.repo.get_active_session()
        if session is None:
            return None
        return int((datetime.utcnow() - session.start_at).total_seconds() / 60)

    def get_remaining_minutes(self) -> int | None:
        """Get remaining minutes if a duration was set.

        Tries in-process timer first; falls back to DB using session.duration_minutes
        (stored via Phase 23 migration column) for cross-process safety.
        Returns None in stopwatch mode (no duration set).
        """
        in_process = self.timer.get_remaining_minutes()
        if in_process is not None:
            return in_process

        session = self.repo.get_active_session()
        if session is None:
            return None

        duration = getattr(session, "duration_minutes", None)
        if duration is None:
            return None

        elapsed = self.get_elapsed_minutes()
        if elapsed is None:
            return None

        return max(0, duration - elapsed)

    def resume_session(self, session_id: str | None = None) -> Session:
        """Resume a paused session by starting a new session on the same task.

        Args:
            session_id: ID of a previous session whose task should be resumed.

        Returns:
            A new Session on the task.

        Raises:
            RuleViolation: If session_id is None or session not found.
        """
        if session_id is None:
            raise RuleViolation("session_id required to resume a session.")

        # Find the task from the most recent session for this session_id
        # We have to find the task_id by listing sessions
        # Use list_sessions_for_task is by task_id, not session_id;
        # instead get active session or fall back to a broad session list
        # and match by session.id
        all_sessions = self.repo.list_sessions(task_id=None)
        target = next((s for s in all_sessions if s.id == session_id), None)
        if target is None:
            raise RuleViolation(f"Session not found: {session_id}")

        return self.start_session(task_id=target.task_id)

    def list_sessions(
        self,
        task_id: str | None = None,
        limit: int = 20,
    ) -> list[Session]:
        """List sessions, optionally filtered by task.

        Args:
            task_id: Optional task ID filter.
            limit: Maximum number of sessions to return (default 20).

        Returns:
            List of sessions, most recent first (sliced to limit).
        """
        sessions = self.repo.list_sessions(task_id=task_id)
        return sessions[:limit]

    def suggest_duration(
        self,
        task_id: str,
        min_samples: int = 3,
    ) -> int | None:
        """Suggest a session duration based on historical completed sessions.

        Returns the median actual duration when >= min_samples completed
        sessions exist for the task. Returns None silently when data is
        insufficient (per D-05: no output when data is insufficient).

        Args:
            task_id: Task to analyse.
            min_samples: Minimum completed sessions required (default 3).

        Returns:
            Median duration in minutes, or None.
        """
        sessions = self.repo.list_sessions_for_task(task_id)
        completed = [s for s in sessions if s.end_at is not None]
        if len(completed) < min_samples:
            return None

        durations = [
            (s.end_at - s.start_at).total_seconds() / 60
            for s in completed
        ]
        sorted_durations = sorted(durations)
        median = sorted_durations[len(sorted_durations) // 2]
        return int(median)

    def _cancel_stale_pauses(self, max_hours: int = 3) -> list[str]:
        """Auto-close sessions paused longer than max_hours (D-13).

        Non-fatal: swallows all exceptions. Returns list of affected task IDs.
        """
        closed: list[str] = []
        try:
            # Use broad session list and filter in-memory
            # (get_stale_pauses is a Repository method not in SessionRepo Protocol)
            all_sessions = self.repo.list_sessions(task_id=None)
            cutoff = datetime.utcnow()
            from datetime import timedelta
            threshold = timedelta(hours=max_hours)

            for session in all_sessions:
                if session.end_at is None:
                    continue
                # A "stale pause" is a session ended more than max_hours ago
                # with no subsequent active session
                if (cutoff - session.end_at) > threshold:
                    try:
                        self.repo.resume_pause_interval(session.id)
                    except Exception:
                        pass
                    closed.append(session.task_id)
                    self._log.info(
                        "sessions.stale_pause_closed",
                        task_id=session.task_id,
                        end_at=session.end_at.isoformat(),
                    )
        except Exception as exc:
            self._log.warning("sessions.cancel_stale_pauses_failed", error=str(exc))
        return closed
