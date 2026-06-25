# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Session lifecycle management.

Handles starting, pausing, and finishing work sessions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog

from pb.core.timer import TimerManager

logger = structlog.get_logger()
from pb.domain.enums import SessionMode, TaskState
from pb.domain.models import Session, Task
from pb.domain.rules import RuleViolation
from pb.storage.repository import Repository


class SessionManager:
    """Manages work session lifecycle."""

    def __init__(self, repo: Repository, timer_manager: Optional[TimerManager] = None):
        self.repo = repo
        self.timer_manager = timer_manager or TimerManager()

    def start_session(
        self,
        task: Task,
        mode: SessionMode = SessionMode.FOCUS,
        intended_outcome: str = "",   # keep for backward-compat with resume_task
        expectation: Optional[str] = None,
        is_resume: bool = False,
    ) -> Session:
        """
        Start a new work session.

        Args:
            task: Task to work on (must be in valid state)
            mode: Session mode
            intended_outcome: What we intend to accomplish
            is_resume: Whether this is resuming a paused task (uses timew continue)

        Returns:
            The created session

        Raises:
            RuleViolation: If task cannot be started
        """
        # D-13: Auto-close stale paused sessions
        self.cancel_stale_pauses()

        if task.completion >= 100:
            raise RuleViolation("Task is already complete.")

        if task.state == TaskState.PAUSED:
            raise RuleViolation(
                f"Task is paused until {task.paused_until}. Un-pause it first."
            )

        active_session = self.repo.get_active_session()
        if active_session:
            raise RuleViolation(
                "Cannot start session: another session is already active. "
                "Pause or finish the current session first."
            )

        # Close open pause interval from a previous session pause
        try:
            task_sessions = self.repo.list_sessions_for_task(task.id)
            if task_sessions:
                last_session = task_sessions[-1]
                self.repo.resume_pause_interval(last_session.id)
        except Exception:
            pass  # Non-fatal

        session = Session(
            task_id=task.id,
            mode=mode,
            intended_outcome=expectation or intended_outcome or "",
            expectation=expectation,
        )
        self.repo.create_session(session)

        # Per D-03: Timer activation is implicit with session start
        # Look up duration from TimeBlock if one exists for this task
        duration_minutes = None
        blocks = self.repo.list_time_blocks_for_date(datetime.utcnow())
        for block in blocks:
            if block.task_id == task.id and block.duration_minutes:
                duration_minutes = block.duration_minutes
                break

        self.timer_manager.start_session_timers(
            session_id=session.id,
            duration_minutes=duration_minutes,
            task_title=task.title,
        )

        return session

    def pause_session(self, actual_outcome: Optional[str] = None) -> Optional[Session]:
        """
        Pause the current active session.

        Args:
            actual_outcome: What was accomplished

        Returns:
            The paused session, or None if no active session
        """
        session = self.repo.get_active_session()
        if session is None:
            return None

        session.end_at = datetime.utcnow()
        session.actual_outcome = actual_outcome
        self.repo.update_session(session)

        # D-12: Record pause interval for trend tracking
        try:
            self.repo.create_pause_interval(session.id, session.end_at or datetime.utcnow())
        except Exception:
            pass  # Non-fatal

        # Per D-13: Stop timers and allow sleep when paused
        self.timer_manager.stop_session_timers()

        return session

    def cancel_stale_pauses(self, max_hours: int = 3) -> list[str]:
        """Auto-close sessions paused longer than max_hours (D-13).

        Closes the session and pause interval. Does NOT change task state —
        task stays active and user can start a new session.
        Returns list of affected task IDs.
        """
        closed = []
        try:
            stale = self.repo.get_stale_pauses(max_hours=max_hours)
            for entry in stale:
                try:
                    self.repo.resume_pause_interval(entry["session_id"])
                except Exception:
                    pass
                logger.info(
                    "sessions.stale_pause_closed",
                    task_id=entry["task_id"],
                    pause_start=entry["pause_start"],
                )
                closed.append(entry["task_id"])
        except Exception as e:
            logger.warning("sessions.cancel_stale_pauses_failed", error=str(e))
        return closed

    def finish_session(
        self,
        outcome: str,
        actual_outcome_note: Optional[str] = None,
        completion_pct: Optional[int] = None,
        distraction: Optional[int] = None,
        next_steps: Optional[list] = None,
    ) -> Optional[Session]:
        """
        Finish the current active session and mark task done.

        Args:
            outcome: Outcome classification (done, partial, blocked, abandoned)
            actual_outcome_note: Description of what happened
            completion_pct: How much was completed (0-100)
            distraction: Distraction rating (1-5)
            next_steps: List of next step strings for GraphWriter wikilinks

        Returns:
            The finished session, or None if no active session
        """
        session = self.repo.get_active_session()
        if session is None:
            return None

        session.end_at = datetime.utcnow()
        session.actual_outcome = actual_outcome_note or outcome
        session.completion_pct = completion_pct
        session.distraction = distraction
        self.repo.update_session(session)

        task = self.repo.get_task(session.task_id)
        if task is None:
            logger.warning("finish_session.task_not_found", task_id=session.task_id)
            self.timer_manager.stop_session_timers()
            return session

        if outcome == "done":
            task.completion = 100
            task.state = TaskState.DONE
            task.completed_at = datetime.utcnow()
        elif outcome == "partial" and completion_pct is not None:
            task.completion = min(completion_pct, 99)
        self.repo.update_task(task)

        # Per D-13: Stop timers and allow sleep when finished
        self.timer_manager.stop_session_timers()

        # D-01: Auto-trigger graph note writing on every pb finish
        task = self.repo.get_task(session.task_id)
        if task:
            try:
                from pb.core.graph_writer import GraphWriter, make_slug
                writer = GraphWriter()
                project = None
                if task.project_id:
                    project = self.repo.get_project(task.project_id)
                date_str = session.end_at.strftime("%Y-%m-%d") if session.end_at else datetime.utcnow().strftime("%Y-%m-%d")
                writer.write_task_note(session, task, project, next_steps or [])
                if project:
                    slug = make_slug(task.title)
                    writer.upsert_project_note(project, slug, date_str)
            except Exception as e:
                import structlog as _structlog
                _logger = _structlog.get_logger()
                _logger.warning("graph_writer.skipped", error=str(e))

            # Phase 3 D-01: Auto-write session log to vault on pb finish
            try:
                from pb.core.session_log_writer import SessionLogWriter
                log_writer = SessionLogWriter()
                log_writer.write_session_log(session, task, project, next_steps or [])
            except Exception as e:
                import structlog as _structlog
                _structlog.get_logger().warning("session_log_writer.skipped", error=str(e))

        return session

    def discard_session(self) -> Optional[Session]:
        """Discard the active session entirely (D-08).

        No graph note, no time tracked — as if the session never happened.
        Stops timers and hard-deletes the session record. Task state unchanged.

        Returns:
            The discarded session object, or None if no active session.
        """
        session = self.repo.get_active_session()
        if session is None:
            return None

        self.timer_manager.stop_session_timers()
        self.repo.delete_session(session.id)

        return session

    def log_interruption(self) -> bool:
        """
        Log an interruption to the current session.

        Returns:
            True if interruption logged, False if no active session
        """
        session = self.repo.get_active_session()
        if session is None:
            return False

        session.interruption_count += 1
        self.repo.update_session(session)

        task = self.repo.get_task(session.task_id)
        if task:
            task.interruption_count += 1
            self.repo.update_task(task)

        return True

    def get_current_session(self) -> Optional[Session]:
        """Get the currently active session."""
        return self.repo.get_active_session()

    def get_current_task(self) -> Optional[Task]:
        """Get the task for the currently active session."""
        session = self.repo.get_active_session()
        if session is None:
            return None
        return self.repo.get_task(session.task_id)

    def get_elapsed_minutes(self) -> Optional[int]:
        """Get elapsed minutes since session start (per D-11).

        Falls back to DB query when timer state is not in-process
        (e.g., calling pb now in a new process after pb start).
        """
        # Try in-process timer first (works if called within same process)
        in_process = self.timer_manager.get_elapsed_minutes()
        if in_process is not None:
            return in_process

        # DB fallback: compute from session.start_at
        session = self.repo.get_active_session()
        if session is None:
            return None
        elapsed = datetime.utcnow() - session.start_at
        return int(elapsed.total_seconds() / 60)

    def get_remaining_minutes(self) -> Optional[int]:
        """Get remaining minutes if duration set (per D-11).

        Falls back to DB query when timer state is not in-process.
        """
        # Try in-process timer first
        in_process = self.timer_manager.get_remaining_minutes()
        if in_process is not None:
            return in_process

        # DB fallback: look up duration from active session's time block
        session = self.repo.get_active_session()
        if session is None:
            return None

        blocks = self.repo.list_time_blocks_for_date(datetime.utcnow())
        duration_minutes = None
        for block in blocks:
            if block.task_id == session.task_id and block.duration_minutes:
                duration_minutes = block.duration_minutes
                break

        if duration_minutes is None:
            return None

        elapsed = self.get_elapsed_minutes()
        if elapsed is None:
            return None

        return max(0, duration_minutes - elapsed)
