# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Repository for CRUD operations on domain entities.

Implements data access while enforcing invariants via rules module.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pb.core.entity_refs import (
    dedupe_visible_ref,
    derive_visible_ref,
    display_ref,
    is_uuid_like,
    visible_ref_key,
)
from pb.core.context_file_intake import ActiveContextScope, FileSupportDecision, SourceBundle, SourceBundleItem
from pb.domain.enums import (
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
from pb.domain.models import (
    ActionReminder,
    DailyDebrief,
    DailyReviewResponse,
    GenerationProvenance,
    GoalArc,
    Packet,
    Project,
    Session,
    Task,
    TimeBlock,
    Track,
)
from pb.domain.rules import (
    RuleViolation,
    validate_project_has_packet,
)
from pb.storage.database import get_connection
from pb.storage.yaml_io import dump_compact_yaml, load_yaml_text


def _serialize_list(items: list) -> str:
    return dump_compact_yaml(items)


def _deserialize_list(data: str) -> list:
    if not data:
        return []
    loaded = load_yaml_text(data, [])
    return loaded if isinstance(loaded, list) else []


def _serialize_dict(data: dict[str, object]) -> str:
    return json.dumps(data or {}, ensure_ascii=True, sort_keys=True)


def _deserialize_dict(data: str | None) -> dict[str, object]:
    if not data:
        return {}
    try:
        loaded = json.loads(data)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _serialize_json_list(items: list[object]) -> str:
    return json.dumps(items or [], ensure_ascii=True)


def _deserialize_json_list(data: str | None) -> list[object]:
    if not data:
        return []
    try:
        loaded = json.loads(data)
    except Exception:
        return []
    return loaded if isinstance(loaded, list) else []


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _bloom_to_str(stage: Optional[BloomStage]) -> Optional[str]:
    if stage is None:
        return None
    return stage.value if hasattr(stage, "value") else str(stage)


def _str_to_bloom(value: Optional[str]) -> Optional[BloomStage]:
    if not value:
        return None
    try:
        return BloomStage(value)
    except ValueError:
        return None


def _practice_stage_to_str(stage: Optional[PracticeStage]) -> Optional[str]:
    if stage is None:
        return None
    return stage.value if hasattr(stage, "value") else str(stage)


def _str_to_practice_stage(value: Optional[str]) -> Optional[PracticeStage]:
    if not value:
        return None
    try:
        return PracticeStage(value)
    except ValueError:
        return None


def _feedback_source_to_str(source: Optional[FeedbackSource]) -> Optional[str]:
    if source is None:
        return None
    return source.value if hasattr(source, "value") else str(source)


def _str_to_feedback_source(value: Optional[str]) -> Optional[FeedbackSource]:
    if not value:
        return None
    try:
        return FeedbackSource(value)
    except ValueError:
        return None


def _evidence_type_to_str(value: Optional[EvidenceType]) -> Optional[str]:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def _str_to_evidence_type(value: Optional[str]) -> Optional[EvidenceType]:
    if not value:
        return None
    try:
        return EvidenceType(value)
    except ValueError:
        return None


class Repository:
    """Data access repository for all domain entities."""

    def _entity_aliases_for(self, conn, entity_kind: str) -> set[str]:
        rows = conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_kind = ?",
            (entity_kind,),
        ).fetchall()
        return {
            str(row["alias"]).strip()
            for row in rows
            if row is not None and str(row["alias"]).strip()
        }

    def _upsert_entity_alias(
        self,
        conn,
        *,
        entity_kind: str,
        entity_id: str,
        alias: str,
        alias_kind: str,
    ) -> None:
        normalized = (alias or "").strip().lower()
        if not normalized:
            return
        conn.execute(
            """
            INSERT INTO entity_aliases (entity_kind, entity_id, alias, alias_kind, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(entity_kind, alias) DO UPDATE SET
                entity_id = excluded.entity_id,
                alias_kind = excluded.alias_kind
            """,
            (
                entity_kind,
                entity_id,
                normalized,
                alias_kind,
                _dt_to_str(datetime.utcnow()),
            ),
        )

    def _ensure_visible_ref(self, conn, entity_kind: str, entity: Any, *, parent_ref: str = "") -> str:
        generated = dict(getattr(entity, "generated_names", {}) or {})
        key = visible_ref_key(entity_kind)
        stored = str(generated.get(key, "") or "").strip().lower()

        if not stored:
            title = (
                getattr(entity, "title", "")
                or getattr(entity, "name", "")
                or getattr(entity, "subject_scope", "")
                or getattr(entity, "intended_outcome", "")
            )
            fallback = str(getattr(entity, "id", "") or entity_kind)
            base = derive_visible_ref(
                entity_kind,
                title=title,
                parent_ref=parent_ref,
                fallback=fallback,
            )
            existing = self._entity_aliases_for(conn, entity_kind)
            stored = dedupe_visible_ref(base, existing)
            generated[key] = stored
            entity.generated_names = generated

        self._upsert_entity_alias(
            conn,
            entity_kind=entity_kind,
            entity_id=str(getattr(entity, "id", "") or ""),
            alias=stored,
            alias_kind="visible_ref",
        )
        entity_id = str(getattr(entity, "id", "") or "")
        if is_uuid_like(entity_id):
            self._upsert_entity_alias(
                conn,
                entity_kind=entity_kind,
                entity_id=entity_id,
                alias=entity_id,
                alias_kind="legacy_uuid",
            )
        return stored

    def _resolve_entity_ref(
        self,
        *,
        entity_kind: str,
        ref: str,
        direct_getter,
        prefix_query: str,
    ):
        query = (ref or "").strip()
        if not query:
            return None

        direct = direct_getter(query)
        if direct is not None:
            return direct

        with get_connection() as conn:
            alias_row = conn.execute(
                """
                SELECT entity_id FROM entity_aliases
                WHERE entity_kind = ? AND alias = ?
                """,
                (entity_kind, query.lower()),
            ).fetchone()
            if alias_row is not None:
                resolved = direct_getter(str(alias_row["entity_id"]))
                if resolved is not None:
                    return resolved

            rows = conn.execute(prefix_query, (f"{query}%",)).fetchall()
            if len(rows) == 1:
                return direct_getter(str(rows[0]["id"]))
        return None

    def resolve_task_ref(self, ref: str, include_archived: bool = False) -> Optional[Task]:
        """Resolve a task by internal id, visible ref, legacy uuid, or unique prefix."""
        if include_archived:
            direct_getter = lambda value: self.get_task(value)
            prefix_query = "SELECT id FROM tasks WHERE id LIKE ?"
        else:
            def direct_getter(value: str) -> Optional[Task]:
                task = self.get_task(value)
                if task is None or task.archived_at is not None:
                    return None
                return task
            prefix_query = "SELECT id FROM tasks WHERE id LIKE ? AND archived_at IS NULL"
        return self._resolve_entity_ref(
            entity_kind="task",
            ref=ref,
            direct_getter=direct_getter,
            prefix_query=prefix_query,
        )

    def resolve_session_ref(self, ref: str) -> Optional[Session]:
        """Resolve a session by internal id, visible ref, legacy uuid, or unique prefix."""
        return self._resolve_entity_ref(
            entity_kind="session",
            ref=ref,
            direct_getter=self.get_session,
            prefix_query="SELECT id FROM sessions WHERE id LIKE ?",
        )

    def resolve_goal_ref(self, ref: str) -> Optional[GoalArc]:
        """Resolve a goal by internal id, visible ref, legacy uuid, or unique prefix."""
        return self._resolve_entity_ref(
            entity_kind="goal",
            ref=ref,
            direct_getter=self.get_goal_arc,
            prefix_query="SELECT id FROM goal_arcs WHERE id LIKE ?",
        )

    # --- Tasks ---

    def create_task(self, task: Task) -> Task:
        """Create a new task."""
        with get_connection() as conn:
            self._ensure_visible_ref(conn, "task", task)
            conn.execute(
                """
                INSERT INTO tasks (
                    id, project_id, title, description, horizon, state,
                    completion, paused_until, pause_reason,
                    estimate_minutes, scheduled_start, scheduled_end,
                    energy_type, linked_goal_arc_ids, linked_track_ids,
                    packet_path, generated_names_json, interruption_count, created_at, updated_at, completed_at, archived_at,
                    impact, urgency_score, strategic_value, effort,
                    important, urgent, energy_required, work_type,
                    due_date, scheduled_date, estimated_minutes, actual_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.project_id,
                    task.title,
                    task.description,
                    task.horizon.value,
                    task.state.value,
                    task.completion,
                    _dt_to_str(task.paused_until),
                    task.pause_reason,
                    task.estimate_minutes,
                    _dt_to_str(task.scheduled_start),
                    _dt_to_str(task.scheduled_end),
                    task.energy_type.value,
                    _serialize_list(task.linked_goal_arc_ids),
                    _serialize_list(task.linked_track_ids),
                    task.packet_path,
                    _serialize_dict(task.generated_names),
                    task.interruption_count,
                    _dt_to_str(task.created_at),
                    _dt_to_str(task.updated_at),
                    _dt_to_str(task.completed_at),
                    _dt_to_str(task.archived_at),
                    task.impact,
                    task.urgency_score,
                    task.strategic_value,
                    task.effort,
                    1 if task.important is True else (0 if task.important is False else None),
                    1 if task.urgent is True else (0 if task.urgent is False else None),
                    task.energy_required,
                    task.work_type,
                    _dt_to_str(task.due_date),
                    _dt_to_str(task.scheduled_date),
                    task.estimated_minutes,
                    task.actual_minutes,
                ),
            )
            conn.commit()
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_task(row)

    def list_tasks(
        self, state: Optional[TaskState] = None, include_archived: bool = False
    ) -> list[Task]:
        """List all tasks, optionally filtered by state.

        Args:
            state: Filter by task state (optional)
            include_archived: If False (default), exclude archived tasks per D-03
        """
        with get_connection() as conn:
            if state is not None and not include_archived:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE state = ? AND archived_at IS NULL ORDER BY created_at",
                    (state.value,),
                ).fetchall()
            elif state is not None and include_archived:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE state = ? ORDER BY created_at",
                    (state.value,),
                ).fetchall()
            elif state is None and not include_archived:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE archived_at IS NULL ORDER BY created_at"
                ).fetchall()
            else:  # state is None and include_archived
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at"
                ).fetchall()
            return [self._row_to_task(row) for row in rows]

    def update_task(self, task: Task) -> Task:
        """Update a task."""
        task.updated_at = datetime.utcnow()

        with get_connection() as conn:
            self._ensure_visible_ref(conn, "task", task)
            conn.execute(
                """
                UPDATE tasks SET
                    project_id = ?, title = ?, description = ?, horizon = ?,
                    state = ?, completion = ?, paused_until = ?, pause_reason = ?,
                    estimate_minutes = ?, scheduled_start = ?,
                    scheduled_end = ?, energy_type = ?, linked_goal_arc_ids = ?,
                    linked_track_ids = ?, packet_path = ?, generated_names_json = ?, interruption_count = ?,
                    updated_at = ?, completed_at = ?,
                    impact = ?, urgency_score = ?, strategic_value = ?, effort = ?,
                    important = ?, urgent = ?, energy_required = ?, work_type = ?,
                    due_date = ?, scheduled_date = ?, estimated_minutes = ?, actual_minutes = ?
                WHERE id = ?
                """,
                (
                    task.project_id,
                    task.title,
                    task.description,
                    task.horizon.value,
                    task.state.value,
                    task.completion,
                    _dt_to_str(task.paused_until),
                    task.pause_reason,
                    task.estimate_minutes,
                    _dt_to_str(task.scheduled_start),
                    _dt_to_str(task.scheduled_end),
                    task.energy_type.value,
                    _serialize_list(task.linked_goal_arc_ids),
                    _serialize_list(task.linked_track_ids),
                    task.packet_path,
                    _serialize_dict(task.generated_names),
                    task.interruption_count,
                    _dt_to_str(task.updated_at),
                    _dt_to_str(task.completed_at),
                    task.impact,
                    task.urgency_score,
                    task.strategic_value,
                    task.effort,
                    1 if task.important is True else (0 if task.important is False else None),
                    1 if task.urgent is True else (0 if task.urgent is False else None),
                    task.energy_required,
                    task.work_type,
                    _dt_to_str(task.due_date),
                    _dt_to_str(task.scheduled_date),
                    task.estimated_minutes,
                    task.actual_minutes,
                    task.id,
                ),
            )
            conn.commit()
        return task

    def get_active_task(self) -> Optional[Task]:
        """Get the task with an active session (at most one per INV-1)."""
        session = self.get_active_session()
        if session is None:
            return None
        return self.get_task(session.task_id)

    def archive_task(self, task_id: str) -> Optional[Task]:
        """
        Archive a task (soft delete) per QUAL-02.

        Sets archived_at timestamp; task remains in database.
        Returns archived task or None if not found.
        """
        task = self.get_task(task_id)
        if task is None:
            return None

        archived_at = datetime.utcnow()

        with get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET archived_at = ?, updated_at = ? WHERE id = ?",
                (_dt_to_str(archived_at), _dt_to_str(archived_at), task_id),
            )
            conn.commit()

        task.archived_at = archived_at
        task.updated_at = archived_at
        return task

    def restore_task(self, task_id: str) -> Optional[Task]:
        """
        Restore an archived task per D-04.

        Clears archived_at timestamp.
        Returns restored task or None if not found.
        """
        # Must find task even if archived - use direct query
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None

            now = datetime.utcnow()
            conn.execute(
                "UPDATE tasks SET archived_at = NULL, updated_at = ? WHERE id = ?",
                (_dt_to_str(now), task_id),
            )
            conn.commit()

        task = self._row_to_task(row)
        task.archived_at = None
        task.updated_at = now
        return task

    def auto_activate_paused_tasks(self) -> int:
        """Activate paused tasks whose paused_until date has passed. Returns count."""
        now = _dt_to_str(datetime.utcnow())
        with get_connection() as conn:
            cursor = conn.execute(
                "UPDATE tasks SET state = 'active', paused_until = NULL, pause_reason = NULL WHERE state = 'paused' AND paused_until IS NOT NULL AND paused_until <= ?",
                (now,),
            )
            conn.commit()
            return cursor.rowcount

    def hard_delete_task(self, task_id: str) -> bool:
        """Hard delete task if no sessions; archive if sessions exist.

        Returns True if hard-deleted, False if archived instead.
        Per D-09: auto-routes to archive when sessions exist.
        Per Pitfall #6: also deletes orphaned time_blocks (SQLite FK not enforced).
        """
        sessions = self.list_sessions_for_task(task_id)
        if sessions:
            self.archive_task(task_id)
            return False
        with get_connection() as conn:
            conn.execute("DELETE FROM time_blocks WHERE task_id = ?", (task_id,))
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
            return cursor.rowcount > 0

    def force_delete_task(self, task_id: str) -> bool:
        """Delete a task and all runtime history, even when sessions exist."""
        with get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM time_blocks WHERE task_id = ?", (task_id,))
            conn.execute(
                "DELETE FROM generation_provenance WHERE artifact_id = ?",
                (task_id,),
            )
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_task(self, row) -> Task:
        keys = row.keys() if hasattr(row, "keys") else []
        state_val = row["state"]
        try:
            state = TaskState(state_val)
        except ValueError:
            state = TaskState.ACTIVE
        return Task(
            id=row["id"],
            project_id=row["project_id"],
            title=row["title"],
            description=row["description"],
            horizon=Horizon(row["horizon"]),
            state=state,
            completion=row["completion"] if "completion" in keys and row["completion"] is not None else 0,
            paused_until=_str_to_dt(row["paused_until"]) if "paused_until" in keys else None,
            pause_reason=row["pause_reason"] if "pause_reason" in keys else None,
            estimate_minutes=row["estimate_minutes"],
            scheduled_start=_str_to_dt(row["scheduled_start"]),
            scheduled_end=_str_to_dt(row["scheduled_end"]),
            energy_type=EnergyType(row["energy_type"]),
            linked_goal_arc_ids=_deserialize_list(row["linked_goal_arc_ids"]),
            linked_track_ids=_deserialize_list(row["linked_track_ids"]),
            packet_path=row["packet_path"],
            generated_names=_deserialize_dict(row["generated_names_json"]) if "generated_names_json" in keys else {},
            interruption_count=row["interruption_count"],
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
            completed_at=_str_to_dt(row["completed_at"]),
            archived_at=_str_to_dt(row["archived_at"]),
            impact=row["impact"] if "impact" in keys else None,
            urgency_score=row["urgency_score"] if "urgency_score" in keys else None,
            strategic_value=row["strategic_value"] if "strategic_value" in keys else None,
            effort=row["effort"] if "effort" in keys else None,
            important=bool(row["important"]) if "important" in keys and row["important"] is not None else None,
            urgent=bool(row["urgent"]) if "urgent" in keys and row["urgent"] is not None else None,
            energy_required=row["energy_required"] if "energy_required" in keys else None,
            work_type=row["work_type"] if "work_type" in keys else None,
            due_date=_str_to_dt(row["due_date"]) if "due_date" in keys else None,
            scheduled_date=_str_to_dt(row["scheduled_date"]) if "scheduled_date" in keys else None,
            estimated_minutes=row["estimated_minutes"] if "estimated_minutes" in keys else None,
            actual_minutes=row["actual_minutes"] if "actual_minutes" in keys else None,
        )

    # --- Projects ---

    def create_project(self, project: Project) -> Project:
        """Create a new project, enforcing packet_path requirement."""
        validate_project_has_packet(project)

        with get_connection() as conn:
            self._ensure_visible_ref(conn, "project", project)
            conn.execute(
                """
                INSERT INTO projects (
                    id, name, project_type, track_id, repo_path, packet_path,
                    status, next_review_at, tags, generated_names_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.name,
                    project.project_type.value,
                    project.track_id,
                    project.repo_path,
                    project.packet_path,
                    project.status.value,
                    _dt_to_str(project.next_review_at),
                    _serialize_list(project.tags),
                    _serialize_dict(project.generated_names),
                    _dt_to_str(project.created_at),
                    _dt_to_str(project.updated_at),
                ),
            )
            conn.commit()
        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_project(row)

    def list_projects(self, status: Optional[ProjectStatus] = None) -> list[Project]:
        """List all projects, optionally filtered by status."""
        with get_connection() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM projects WHERE status = ? ORDER BY created_at",
                    (status.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM projects ORDER BY created_at"
                ).fetchall()
            return [self._row_to_project(row) for row in rows]

    def update_project(self, project: Project) -> Project:
        """Update a project (used for marking reviewed via next_review_at)."""
        project.updated_at = datetime.utcnow()

        with get_connection() as conn:
            self._ensure_visible_ref(conn, "project", project)
            conn.execute(
                """
                UPDATE projects SET
                    name = ?, project_type = ?, track_id = ?, repo_path = ?,
                    packet_path = ?, status = ?, next_review_at = ?, tags = ?, generated_names_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    project.name,
                    project.project_type.value,
                    project.track_id,
                    project.repo_path,
                    project.packet_path,
                    project.status.value,
                    _dt_to_str(project.next_review_at),
                    _serialize_list(project.tags),
                    _serialize_dict(project.generated_names),
                    _dt_to_str(project.updated_at),
                    project.id,
                ),
            )
            conn.commit()
        return project

    def _row_to_project(self, row) -> Project:
        keys = row.keys() if hasattr(row, "keys") else []
        return Project(
            id=row["id"],
            name=row["name"],
            project_type=ProjectType(row["project_type"]),
            track_id=row["track_id"],
            repo_path=row["repo_path"],
            packet_path=row["packet_path"],
            status=ProjectStatus(row["status"]),
            next_review_at=_str_to_dt(row["next_review_at"]),
            tags=_deserialize_list(row["tags"]),
            generated_names=_deserialize_dict(row["generated_names_json"]) if "generated_names_json" in keys else {},
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )

    # --- Sessions ---

    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        with get_connection() as conn:
            parent_task = self.get_task(session.task_id)
            parent_ref = display_ref(parent_task, "task") if parent_task is not None else ""
            self._ensure_visible_ref(conn, "session", session, parent_ref=parent_ref)
            conn.execute(
                """
                INSERT INTO sessions (
                    id, task_id, start_at, end_at, mode, branch, goal_id, track_id, subject_scope,
                    bloom_stage, target_bloom_stage, practice_stage, drill_type, constraint_text,
                    feedback_source, evidence_target, coach_cues, observed_errors,
                    quality_rating, difficulty_rating, next_adjustment,
                    intended_outcome, actual_outcome, generated_names_json, interruption_count,
                    llm_summary_used, expectation, completion_pct, distraction,
                    duration_minutes, timer_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.task_id,
                    _dt_to_str(session.start_at),
                    _dt_to_str(session.end_at),
                    session.mode.value,
                    session.branch,
                    session.goal_id,
                    session.track_id,
                    session.subject_scope,
                    _bloom_to_str(session.bloom_stage),
                    _bloom_to_str(session.target_bloom_stage),
                    _practice_stage_to_str(session.practice_stage),
                    session.drill_type,
                    session.constraint,
                    _feedback_source_to_str(session.feedback_source),
                    session.evidence_target,
                    session.coach_cues,
                    session.observed_errors,
                    session.quality_rating,
                    session.difficulty_rating,
                    session.next_adjustment,
                    session.intended_outcome,
                    session.actual_outcome,
                    _serialize_dict(session.generated_names),
                    session.interruption_count,
                    1 if session.llm_summary_used else 0,
                    session.expectation,
                    session.completion_pct,
                    session.distraction,
                    getattr(session, "duration_minutes", None),
                    getattr(session, "timer_mode", "stopwatch"),
                ),
            )
            conn.commit()
        return session

    def update_session(self, session: Session) -> Session:
        """Update a session."""
        with get_connection() as conn:
            parent_task = self.get_task(session.task_id)
            parent_ref = display_ref(parent_task, "task") if parent_task is not None else ""
            self._ensure_visible_ref(conn, "session", session, parent_ref=parent_ref)
            conn.execute(
                """
                UPDATE sessions SET
                    end_at = ?, branch = ?, goal_id = ?, track_id = ?, subject_scope = ?, bloom_stage = ?,
                    target_bloom_stage = ?, practice_stage = ?, drill_type = ?, constraint_text = ?,
                    feedback_source = ?, evidence_target = ?, coach_cues = ?, observed_errors = ?,
                    quality_rating = ?, difficulty_rating = ?, next_adjustment = ?,
                    actual_outcome = ?, generated_names_json = ?, interruption_count = ?, llm_summary_used = ?,
                    expectation = ?, completion_pct = ?, distraction = ?,
                    duration_minutes = ?, timer_mode = ?
                WHERE id = ?
                """,
                (
                    _dt_to_str(session.end_at),
                    session.branch,
                    session.goal_id,
                    session.track_id,
                    session.subject_scope,
                    _bloom_to_str(session.bloom_stage),
                    _bloom_to_str(session.target_bloom_stage),
                    _practice_stage_to_str(session.practice_stage),
                    session.drill_type,
                    session.constraint,
                    _feedback_source_to_str(session.feedback_source),
                    session.evidence_target,
                    session.coach_cues,
                    session.observed_errors,
                    session.quality_rating,
                    session.difficulty_rating,
                    session.next_adjustment,
                    session.actual_outcome,
                    _serialize_dict(session.generated_names),
                    session.interruption_count,
                    1 if session.llm_summary_used else 0,
                    session.expectation,
                    session.completion_pct,
                    session.distraction,
                    getattr(session, "duration_minutes", None),
                    getattr(session, "timer_mode", "stopwatch"),
                    session.id,
                ),
            )
            conn.commit()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_session(row)

    def list_sessions_for_task(self, task_id: str) -> list[Session]:
        """List all sessions for a task."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE task_id = ? ORDER BY start_at",
                (task_id,),
            ).fetchall()
            return [self._row_to_session(row) for row in rows]

    def list_sessions_in_range(
        self, start: datetime, end: datetime
    ) -> list[Session]:
        """List sessions within a date range."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions
                WHERE start_at >= ? AND start_at < ?
                ORDER BY start_at
                """,
                (_dt_to_str(start), _dt_to_str(end)),
            ).fetchall()
            return [self._row_to_session(row) for row in rows]

    def get_active_session(self) -> Optional[Session]:
        """Get the current active (unended) session."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE end_at IS NULL ORDER BY start_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return self._row_to_session(row)

    def _row_to_session(self, row) -> Session:
        keys = row.keys() if hasattr(row, "keys") else []
        session = Session(
            id=row["id"],
            task_id=row["task_id"],
            start_at=_str_to_dt(row["start_at"]),
            end_at=_str_to_dt(row["end_at"]),
            mode=SessionMode(row["mode"]),
            branch=row["branch"] if "branch" in keys and row["branch"] is not None else "study",
            goal_id=row["goal_id"] if "goal_id" in keys else None,
            track_id=row["track_id"] if "track_id" in keys else None,
            subject_scope=row["subject_scope"] if "subject_scope" in keys and row["subject_scope"] is not None else "",
            bloom_stage=_str_to_bloom(row["bloom_stage"]) if "bloom_stage" in keys else None,
            target_bloom_stage=_str_to_bloom(row["target_bloom_stage"]) if "target_bloom_stage" in keys else None,
            practice_stage=_str_to_practice_stage(row["practice_stage"]) if "practice_stage" in keys else None,
            drill_type=row["drill_type"] if "drill_type" in keys else None,
            constraint=(
                row["constraint_text"]
                if "constraint_text" in keys
                else row["constraint"] if "constraint" in keys else None
            ),
            feedback_source=_str_to_feedback_source(row["feedback_source"]) if "feedback_source" in keys else None,
            evidence_target=row["evidence_target"] if "evidence_target" in keys else None,
            coach_cues=row["coach_cues"] if "coach_cues" in keys else None,
            observed_errors=row["observed_errors"] if "observed_errors" in keys else None,
            quality_rating=row["quality_rating"] if "quality_rating" in keys else None,
            difficulty_rating=row["difficulty_rating"] if "difficulty_rating" in keys else None,
            next_adjustment=row["next_adjustment"] if "next_adjustment" in keys else None,
            intended_outcome=row["intended_outcome"],
            actual_outcome=row["actual_outcome"],
            generated_names=_deserialize_dict(row["generated_names_json"]) if "generated_names_json" in keys else {},
            interruption_count=row["interruption_count"],
            llm_summary_used=bool(row["llm_summary_used"]),
            expectation=row["expectation"],
            completion_pct=row["completion_pct"],
            distraction=row["distraction"],
        )
        try:
            object.__setattr__(session, "duration_minutes", row["duration_minutes"] if "duration_minutes" in keys else None)
            object.__setattr__(session, "timer_mode", row["timer_mode"] if "timer_mode" in keys and row["timer_mode"] else "stopwatch")
        except Exception:
            pass
        return session

    def delete_session(self, session_id: str) -> bool:
        """Hard delete a session by ID. Returns True if deleted, False if not found."""
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_sessions_for_task(self, task_id: str) -> int:
        """Delete all sessions for one task. Returns deleted row count."""
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE task_id = ?",
                (task_id,),
            )
            conn.commit()
            return cursor.rowcount

    # --- Lesson Runtime ---

    def create_lesson_run(self, run) -> object:
        """Persist one top-level lesson runtime row."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO lesson_runs (
                    id, session_id, task_id, branch, lesson_mode, title, lesson_status,
                    active_page_slug, active_question_slug, active_page_index, active_question_index,
                    total_points, ready_to_finish, note_path, retry_queue_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.session_id,
                    run.task_id,
                    run.branch,
                    run.lesson_mode,
                    run.title,
                    run.lesson_status,
                    run.active_page_slug,
                    run.active_question_slug,
                    run.active_page_index,
                    run.active_question_index,
                    run.total_points,
                    1 if run.ready_to_finish else 0,
                    run.note_path,
                    _serialize_json_list(list(run.retry_queue or [])),
                    run.created_at,
                    run.updated_at,
                ),
            )
            conn.commit()
        return run

    def update_lesson_run(self, run) -> object:
        """Update one lesson runtime row."""
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE lesson_runs SET
                    task_id = ?, branch = ?, lesson_mode = ?, title = ?, lesson_status = ?,
                    active_page_slug = ?, active_question_slug = ?, active_page_index = ?, active_question_index = ?,
                    total_points = ?, ready_to_finish = ?, note_path = ?, retry_queue_json = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    run.task_id,
                    run.branch,
                    run.lesson_mode,
                    run.title,
                    run.lesson_status,
                    run.active_page_slug,
                    run.active_question_slug,
                    run.active_page_index,
                    run.active_question_index,
                    run.total_points,
                    1 if run.ready_to_finish else 0,
                    run.note_path,
                    _serialize_json_list(list(run.retry_queue or [])),
                    run.updated_at,
                    run.session_id,
                ),
            )
            conn.commit()
        return run

    def get_lesson_run(self, session_id: str):
        """Return the persisted lesson runtime for one session."""
        if not session_id:
            return None
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM lesson_runs WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_lesson_run(row)

    def create_lesson_page(self, page) -> object:
        """Persist one lesson page row."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO lesson_pages (
                    id, lesson_run_id, session_id, page_slug, title, intro_text,
                    sequence_index, status, question_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    page.id,
                    page.lesson_run_id,
                    page.session_id,
                    page.page_slug,
                    page.title,
                    page.intro_text,
                    page.sequence_index,
                    page.status,
                    page.question_count,
                    page.created_at,
                    page.updated_at,
                ),
            )
            conn.commit()
        return page

    def update_lesson_page(self, page) -> object:
        """Update one lesson page row."""
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE lesson_pages SET
                    title = ?, intro_text = ?, sequence_index = ?, status = ?,
                    question_count = ?, updated_at = ?
                WHERE lesson_run_id = ? AND page_slug = ?
                """,
                (
                    page.title,
                    page.intro_text,
                    page.sequence_index,
                    page.status,
                    page.question_count,
                    page.updated_at,
                    page.lesson_run_id,
                    page.page_slug,
                ),
            )
            conn.commit()
        return page

    def get_lesson_page(self, lesson_run_id: str, page_slug: str):
        """Return one lesson page by run and page slug."""
        if not lesson_run_id or not page_slug:
            return None
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM lesson_pages
                WHERE lesson_run_id = ? AND page_slug = ?
                """,
                (lesson_run_id, page_slug),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_lesson_page(row)

    def list_lesson_pages(self, lesson_run_id: str) -> list[object]:
        """List lesson pages for one run in sequence order."""
        if not lesson_run_id:
            return []
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM lesson_pages
                WHERE lesson_run_id = ?
                ORDER BY sequence_index, created_at
                """,
                (lesson_run_id,),
            ).fetchall()
        return [self._row_to_lesson_page(row) for row in rows]

    def create_lesson_question(self, question) -> object:
        """Persist one lesson question row."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO lesson_questions (
                    id, lesson_run_id, session_id, page_slug, question_slug, skill_slug, question_type,
                    prompt_json, answer_json, metadata_json, sequence_index, status, hint_level, revealed, mastered,
                    queued_retry, retry_of_question_slug, retry_generation, next_review_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question.id,
                    question.lesson_run_id,
                    question.session_id,
                    question.page_slug,
                    question.question_slug,
                    question.skill_slug,
                    question.question_type,
                    _serialize_dict(dict(question.prompt_json or {})),
                    _serialize_dict(dict(question.answer_json or {})),
                    _serialize_dict(dict(question.metadata_json or {})),
                    question.sequence_index,
                    question.status,
                    question.hint_level,
                    1 if question.revealed else 0,
                    1 if question.mastered else 0,
                    1 if question.queued_retry else 0,
                    question.retry_of_question_slug,
                    question.retry_generation,
                    question.next_review_at,
                    question.created_at,
                    question.updated_at,
                ),
            )
            conn.commit()
        return question

    def update_lesson_question(self, question) -> object:
        """Update one lesson question row."""
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE lesson_questions SET
                    page_slug = ?, skill_slug = ?, question_type = ?, prompt_json = ?, answer_json = ?, metadata_json = ?,
                    sequence_index = ?, status = ?, hint_level = ?, revealed = ?, mastered = ?,
                    queued_retry = ?, retry_of_question_slug = ?, retry_generation = ?, next_review_at = ?,
                    updated_at = ?
                WHERE lesson_run_id = ? AND question_slug = ?
                """,
                (
                    question.page_slug,
                    question.skill_slug,
                    question.question_type,
                    _serialize_dict(dict(question.prompt_json or {})),
                    _serialize_dict(dict(question.answer_json or {})),
                    _serialize_dict(dict(question.metadata_json or {})),
                    question.sequence_index,
                    question.status,
                    question.hint_level,
                    1 if question.revealed else 0,
                    1 if question.mastered else 0,
                    1 if question.queued_retry else 0,
                    question.retry_of_question_slug,
                    question.retry_generation,
                    question.next_review_at,
                    question.updated_at,
                    question.lesson_run_id,
                    question.question_slug,
                ),
            )
            conn.commit()
        return question

    def get_lesson_question(self, lesson_run_id: str, question_slug: str):
        """Return one lesson question by run and slug."""
        if not lesson_run_id or not question_slug:
            return None
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM lesson_questions
                WHERE lesson_run_id = ? AND question_slug = ?
                """,
                (lesson_run_id, question_slug),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_lesson_question(row)

    def list_lesson_questions(self, lesson_run_id: str, page_slug: str | None = None) -> list[object]:
        """List lesson questions, optionally filtered to one page."""
        if not lesson_run_id:
            return []
        with get_connection() as conn:
            if page_slug:
                rows = conn.execute(
                    """
                    SELECT * FROM lesson_questions
                    WHERE lesson_run_id = ? AND page_slug = ?
                    ORDER BY sequence_index, created_at
                    """,
                    (lesson_run_id, page_slug),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT q.*
                    FROM lesson_questions AS q
                    LEFT JOIN lesson_pages AS p
                      ON p.lesson_run_id = q.lesson_run_id
                     AND p.page_slug = q.page_slug
                    WHERE q.lesson_run_id = ?
                    ORDER BY
                        CASE WHEN q.page_slug = 'mistakes' THEN 1 ELSE 0 END,
                        COALESCE(p.sequence_index, 0),
                        q.sequence_index,
                        q.created_at
                    """,
                    (lesson_run_id,),
                ).fetchall()
        return [self._row_to_lesson_question(row) for row in rows]

    def create_lesson_attempt(self, attempt) -> object:
        """Persist one attempt-level lesson evidence row."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO lesson_attempts (
                    id, lesson_run_id, session_id, page_slug, question_slug, skill_slug,
                    answer_text, result, response_ms, hint_level, points_delta, error_tags_json,
                    evaluator_confidence, model_used, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.id,
                    attempt.lesson_run_id,
                    attempt.session_id,
                    attempt.page_slug,
                    attempt.question_slug,
                    attempt.skill_slug,
                    attempt.answer_text,
                    attempt.result,
                    attempt.response_ms,
                    attempt.hint_level,
                    attempt.points_delta,
                    _serialize_json_list(list(attempt.error_tags or [])),
                    attempt.evaluator_confidence,
                    attempt.model_used,
                    attempt.created_at,
                ),
            )
            conn.commit()
        return attempt

    def list_lesson_attempts(self, lesson_run_id: str, question_slug: str | None = None) -> list[object]:
        """List persisted lesson attempts in insertion order."""
        if not lesson_run_id:
            return []
        with get_connection() as conn:
            if question_slug:
                rows = conn.execute(
                    """
                    SELECT * FROM lesson_attempts
                    WHERE lesson_run_id = ? AND question_slug = ?
                    ORDER BY row_id
                    """,
                    (lesson_run_id, question_slug),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM lesson_attempts
                    WHERE lesson_run_id = ?
                    ORDER BY row_id
                    """,
                    (lesson_run_id,),
                ).fetchall()
        return [self._row_to_lesson_attempt(row) for row in rows]

    def create_lesson_skill_state(self, state) -> object:
        """Persist one aggregated lesson skill-state row."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO lesson_skill_states (
                    id, lesson_run_id, session_id, skill_slug, recognition_status,
                    production_status, overall_status, error_tags_json, attempt_count,
                    next_review_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.id,
                    state.lesson_run_id,
                    state.session_id,
                    state.skill_slug,
                    state.recognition_status,
                    state.production_status,
                    state.overall_status,
                    _serialize_json_list(list(state.error_tags or [])),
                    state.attempt_count,
                    state.next_review_at,
                    state.updated_at,
                ),
            )
            conn.commit()
        return state

    def update_lesson_skill_state(self, state) -> object:
        """Update one aggregated lesson skill-state row."""
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE lesson_skill_states SET
                    recognition_status = ?, production_status = ?, overall_status = ?,
                    error_tags_json = ?, attempt_count = ?, next_review_at = ?, updated_at = ?
                WHERE lesson_run_id = ? AND skill_slug = ?
                """,
                (
                    state.recognition_status,
                    state.production_status,
                    state.overall_status,
                    _serialize_json_list(list(state.error_tags or [])),
                    state.attempt_count,
                    state.next_review_at,
                    state.updated_at,
                    state.lesson_run_id,
                    state.skill_slug,
                ),
            )
            conn.commit()
        return state

    def get_lesson_skill_state(self, lesson_run_id: str, skill_slug: str):
        """Return one lesson skill-state row."""
        if not lesson_run_id or not skill_slug:
            return None
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM lesson_skill_states
                WHERE lesson_run_id = ? AND skill_slug = ?
                """,
                (lesson_run_id, skill_slug),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_lesson_skill_state(row)

    def list_lesson_skill_states(self, lesson_run_id: str) -> list[object]:
        """List lesson skill diagnostics for one run."""
        if not lesson_run_id:
            return []
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM lesson_skill_states
                WHERE lesson_run_id = ?
                ORDER BY skill_slug
                """,
                (lesson_run_id,),
            ).fetchall()
        return [self._row_to_lesson_skill_state(row) for row in rows]

    def list_concept_confidence(self, concept_id: str | None = None) -> list[object]:
        """List confidence records, optionally filtered to one concept (D-16-17).

        Returns all rows ordered by confidence_score ASC (weakest first) when
        concept_id is None — matches the adapter's needs for full-table scans.
        Returns an empty list if no rows match (not an error).
        """
        with get_connection() as conn:
            if concept_id:
                rows = conn.execute(
                    "SELECT * FROM concept_confidence WHERE concept_id = ?",
                    (concept_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM concept_confidence ORDER BY confidence_score"
                ).fetchall()
        return [self._row_to_concept_confidence(row) for row in rows]

    def upsert_concept_confidence(
        self,
        concept_id: str,
        *,
        confidence_score: float,
        card_weight: float | None = None,
        next_review_at: str = "",
        last_evidence_at: str = "",
        burst_active: int = 0,
        burst_streak: int = 0,
    ) -> object:
        """Create or update confidence for one concept (D-16-17).

        card_weight defaults to max(0.0, 1.0 - confidence_score) per D-16-18.
        Uses ON CONFLICT DO UPDATE to avoid read-modify-write races (L5 landmine).
        """
        from pb.core.confidence_model import clamp_score

        now = datetime.utcnow().isoformat()
        clamped = clamp_score(confidence_score)
        effective_weight = card_weight if card_weight is not None else max(0.0, 1.0 - clamped)
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO concept_confidence
                    (concept_id, confidence_score, card_weight, next_review_at,
                     last_evidence_at, burst_active, burst_streak, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(concept_id) DO UPDATE SET
                    confidence_score = excluded.confidence_score,
                    card_weight      = excluded.card_weight,
                    next_review_at   = excluded.next_review_at,
                    last_evidence_at = excluded.last_evidence_at,
                    burst_active     = excluded.burst_active,
                    burst_streak     = excluded.burst_streak,
                    updated_at       = excluded.updated_at
                """,
                (concept_id, clamped, effective_weight,
                 next_review_at, last_evidence_at,
                 burst_active, burst_streak, now, now),
            )
            conn.commit()
        records = self.list_concept_confidence(concept_id)
        return records[0] if records else None

    def _row_to_concept_confidence(self, row) -> object:
        """Convert a concept_confidence db row to ConceptConfidenceRecord."""
        from pb.core.confidence_model import ConceptConfidenceRecord

        return ConceptConfidenceRecord(
            concept_id=row["concept_id"],
            confidence_score=row["confidence_score"],
            card_weight=row["card_weight"],
            next_review_at=row["next_review_at"] or "",
            last_evidence_at=row["last_evidence_at"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            burst_active=int(row["burst_active"] or 0),
            burst_streak=int(row["burst_streak"] or 0),
        )

    def _row_to_lesson_run(self, row):
        from pb.core.lesson_engine import LessonRunRecord

        return LessonRunRecord(
            id=row["id"],
            session_id=row["session_id"],
            task_id=row["task_id"],
            branch=row["branch"],
            lesson_mode=row["lesson_mode"],
            title=row["title"],
            lesson_status=row["lesson_status"],
            active_page_slug=row["active_page_slug"],
            active_question_slug=row["active_question_slug"],
            active_page_index=row["active_page_index"],
            active_question_index=row["active_question_index"],
            total_points=row["total_points"],
            ready_to_finish=bool(row["ready_to_finish"]),
            note_path=row["note_path"],
            retry_queue=[str(item) for item in _deserialize_json_list(row["retry_queue_json"]) if str(item).strip()],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_lesson_page(self, row):
        from pb.core.lesson_engine import LessonPageRecord

        return LessonPageRecord(
            id=row["id"],
            lesson_run_id=row["lesson_run_id"],
            session_id=row["session_id"],
            page_slug=row["page_slug"],
            title=row["title"],
            intro_text=row["intro_text"],
            sequence_index=row["sequence_index"],
            status=row["status"],
            question_count=row["question_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_lesson_question(self, row):
        from pb.core.lesson_engine import LessonQuestionRecord

        return LessonQuestionRecord(
            id=row["id"],
            lesson_run_id=row["lesson_run_id"],
            session_id=row["session_id"],
            page_slug=row["page_slug"],
            question_slug=row["question_slug"],
            skill_slug=row["skill_slug"],
            question_type=row["question_type"],
            prompt_json=_deserialize_dict(row["prompt_json"]),
            answer_json=_deserialize_dict(row["answer_json"]),
            metadata_json=_deserialize_dict(row["metadata_json"]) if "metadata_json" in row.keys() else {},
            sequence_index=row["sequence_index"],
            status=row["status"],
            hint_level=row["hint_level"],
            revealed=bool(row["revealed"]),
            mastered=bool(row["mastered"]),
            queued_retry=bool(row["queued_retry"]),
            retry_of_question_slug=row["retry_of_question_slug"],
            retry_generation=row["retry_generation"],
            next_review_at=row["next_review_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_lesson_attempt(self, row):
        from pb.core.lesson_engine import LessonAttemptRecord

        return LessonAttemptRecord(
            id=row["id"],
            lesson_run_id=row["lesson_run_id"],
            session_id=row["session_id"],
            page_slug=row["page_slug"],
            question_slug=row["question_slug"],
            skill_slug=row["skill_slug"],
            answer_text=row["answer_text"],
            result=row["result"],
            response_ms=row["response_ms"],
            hint_level=row["hint_level"],
            points_delta=row["points_delta"],
            error_tags=[str(item) for item in _deserialize_json_list(row["error_tags_json"]) if str(item).strip()],
            evaluator_confidence=row["evaluator_confidence"],
            model_used=row["model_used"],
            created_at=row["created_at"],
        )

    def _row_to_lesson_skill_state(self, row):
        from pb.core.lesson_engine import LessonSkillStateRecord

        return LessonSkillStateRecord(
            id=row["id"],
            lesson_run_id=row["lesson_run_id"],
            session_id=row["session_id"],
            skill_slug=row["skill_slug"],
            recognition_status=row["recognition_status"],
            production_status=row["production_status"],
            overall_status=row["overall_status"],
            error_tags=[str(item) for item in _deserialize_json_list(row["error_tags_json"]) if str(item).strip()],
            attempt_count=row["attempt_count"],
            next_review_at=row["next_review_at"],
            updated_at=row["updated_at"],
        )

    # --- Time Blocks ---

    def create_time_block(self, block: TimeBlock) -> TimeBlock:
        """Create a new time block."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO time_blocks (
                    id, task_id, start_time, duration_minutes, block_kind,
                    created_at, series_id, recurrence_rule
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block.id,
                    block.task_id,
                    _dt_to_str(block.start_time),
                    block.duration_minutes,
                    block.block_kind,
                    _dt_to_str(block.created_at),
                    block.series_id,
                    block.recurrence_rule,
                ),
            )
            conn.commit()
        return block

    def list_time_blocks_for_date(self, date: datetime) -> list[TimeBlock]:
        """List time blocks for a specific date.

        Includes both scheduled blocks (by start_time) and unscheduled
        duration-only blocks (by created_at) so callers like plan.list_blocks
        and execute.start_task can find them.
        """
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59)
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM time_blocks
                WHERE (start_time >= ? AND start_time <= ?)
                   OR (start_time IS NULL AND created_at >= ? AND created_at <= ?)
                ORDER BY start_time
                """,
                (_dt_to_str(start), _dt_to_str(end),
                 _dt_to_str(start), _dt_to_str(end)),
            ).fetchall()
            return [self._row_to_time_block(row) for row in rows]

    def get_time_block(self, block_id: str) -> Optional[TimeBlock]:
        """Get a time block by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM time_blocks WHERE id = ?", (block_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_time_block(row)

    def delete_time_block(self, block_id: str) -> bool:
        """Delete a time block by ID. Returns True if deleted, False if not found."""
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM time_blocks WHERE id = ?", (block_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_time_blocks_for_task(self, task_id: str) -> int:
        """Delete all time blocks for one task. Returns deleted row count."""
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM time_blocks WHERE task_id = ?",
                (task_id,),
            )
            conn.commit()
            return cursor.rowcount

    def update_time_block(self, block: TimeBlock) -> TimeBlock:
        """Update a time block's start_time, duration_minutes, series_id, and recurrence_rule."""
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE time_blocks SET start_time = ?, duration_minutes = ?,
                block_kind = ?, series_id = ?, recurrence_rule = ?
                WHERE id = ?
                """,
                (
                    _dt_to_str(block.start_time),
                    block.duration_minutes,
                    block.block_kind,
                    block.series_id,
                    block.recurrence_rule,
                    block.id,
                ),
            )
            conn.commit()
        return block

    def _row_to_time_block(self, row) -> TimeBlock:
        keys = row.keys() if hasattr(row, "keys") else []
        return TimeBlock(
            id=row["id"],
            task_id=row["task_id"],
            start_time=_str_to_dt(row["start_time"]),
            duration_minutes=row["duration_minutes"],
            block_kind=row["block_kind"] if "block_kind" in keys and row["block_kind"] else "study",
            created_at=_str_to_dt(row["created_at"]),
            series_id=row["series_id"] if "series_id" in keys else None,
            recurrence_rule=row["recurrence_rule"] if "recurrence_rule" in keys else None,
        )

    def list_time_blocks_created_for_date(self, date: datetime) -> list[TimeBlock]:
        """List all time blocks created on a specific date (includes unscheduled blocks).

        Unlike list_time_blocks_for_date (which filters by start_time), this method
        filters by created_at so that unscheduled blocks with start_time=None are
        included in the result. Used for budget calculations.
        """
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59)
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM time_blocks
                WHERE created_at >= ? AND created_at <= ?
                ORDER BY created_at
                """,
                (_dt_to_str(start), _dt_to_str(end)),
            ).fetchall()
            return [self._row_to_time_block(row) for row in rows]

    def list_time_blocks_by_series(self, series_id: str) -> list[TimeBlock]:
        """List all time blocks in a recurrence series."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM time_blocks WHERE series_id = ? ORDER BY start_time",
                (series_id,),
            ).fetchall()
            return [self._row_to_time_block(row) for row in rows]

    # --- Tracks ---

    def create_track(self, track: Track) -> Track:
        """Create a new track."""
        with get_connection() as conn:
            self._ensure_visible_ref(conn, "track", track)
            conn.execute(
                """
                INSERT INTO tracks (
                    id, name, description, linked_goal_arc_ids, cadence,
                    priority_weight, active, generated_names_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    track.id,
                    track.name,
                    track.description,
                    _serialize_list(track.linked_goal_arc_ids),
                    track.cadence,
                    track.priority_weight,
                    1 if track.active else 0,
                    _serialize_dict(track.generated_names),
                    _dt_to_str(track.created_at),
                    _dt_to_str(track.updated_at),
                ),
            )
            conn.commit()
        return track

    def list_tracks(self, active_only: bool = True) -> list[Track]:
        """List all tracks."""
        with get_connection() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM tracks WHERE active = 1 ORDER BY name"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tracks ORDER BY name"
                ).fetchall()
            return [self._row_to_track(row) for row in rows]

    def get_track(self, track_id: str) -> Optional[Track]:
        """Get a track by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tracks WHERE id = ?", (track_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_track(row)

    def _row_to_track(self, row) -> Track:
        keys = row.keys() if hasattr(row, "keys") else []
        return Track(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            linked_goal_arc_ids=_deserialize_list(row["linked_goal_arc_ids"]),
            cadence=row["cadence"],
            priority_weight=row["priority_weight"],
            active=bool(row["active"]),
            generated_names=_deserialize_dict(row["generated_names_json"]) if "generated_names_json" in keys else {},
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )

    # --- Goal Arcs ---

    def create_goal_arc(self, goal: GoalArc) -> GoalArc:
        """Create a new goal arc."""
        with get_connection() as conn:
            self._ensure_visible_ref(conn, "goal", goal)
            conn.execute(
                """
                INSERT INTO goal_arcs (
                    id, title, domain, execution_mode, study_framework, current_bloom_stage,
                    target_bloom_stage, practice_framework, current_practice_stage,
                    target_practice_stage, horizon, description, success_definition,
                    framework, primary_metric, feedback_source, evidence_type,
                    metric_type, target_value, start_date, target_date,
                    status, tags, generated_names_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal.id,
                    goal.title,
                    goal.domain,
                    goal.execution_mode,
                    goal.study_framework,
                    _bloom_to_str(goal.current_bloom_stage),
                    _bloom_to_str(goal.target_bloom_stage),
                    goal.practice_framework,
                    _practice_stage_to_str(goal.current_practice_stage),
                    _practice_stage_to_str(goal.target_practice_stage),
                    goal.horizon.value,
                    goal.description,
                    goal.success_definition,
                    goal.framework,
                    goal.primary_metric,
                    _feedback_source_to_str(goal.feedback_source),
                    _evidence_type_to_str(goal.evidence_type),
                    goal.metric_type,
                    goal.target_value,
                    _dt_to_str(goal.start_date),
                    _dt_to_str(goal.target_date),
                    goal.status,
                    _serialize_list(goal.tags),
                    _serialize_dict(goal.generated_names),
                    _dt_to_str(goal.created_at),
                    _dt_to_str(goal.updated_at),
                ),
            )
            conn.commit()
        return goal

    def update_goal_arc(self, goal: GoalArc) -> GoalArc:
        """Update a goal arc."""
        goal.updated_at = datetime.utcnow()
        with get_connection() as conn:
            self._ensure_visible_ref(conn, "goal", goal)
            conn.execute(
                """
                UPDATE goal_arcs SET
                    title = ?, domain = ?, execution_mode = ?, study_framework = ?, current_bloom_stage = ?,
                    target_bloom_stage = ?, practice_framework = ?, current_practice_stage = ?,
                    target_practice_stage = ?, horizon = ?, description = ?, success_definition = ?, framework = ?,
                    primary_metric = ?, feedback_source = ?, evidence_type = ?,
                    metric_type = ?, target_value = ?, start_date = ?,
                    target_date = ?, status = ?, tags = ?, generated_names_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    goal.title,
                    goal.domain,
                    goal.execution_mode,
                    goal.study_framework,
                    _bloom_to_str(goal.current_bloom_stage),
                    _bloom_to_str(goal.target_bloom_stage),
                    goal.practice_framework,
                    _practice_stage_to_str(goal.current_practice_stage),
                    _practice_stage_to_str(goal.target_practice_stage),
                    goal.horizon.value,
                    goal.description,
                    goal.success_definition,
                    goal.framework,
                    goal.primary_metric,
                    _feedback_source_to_str(goal.feedback_source),
                    _evidence_type_to_str(goal.evidence_type),
                    goal.metric_type,
                    goal.target_value,
                    _dt_to_str(goal.start_date),
                    _dt_to_str(goal.target_date),
                    goal.status,
                    _serialize_list(goal.tags),
                    _serialize_dict(goal.generated_names),
                    _dt_to_str(goal.updated_at),
                    goal.id,
                ),
            )
            conn.commit()
        return goal

    def hard_delete_goal_arc(self, goal_id: str) -> bool:
        """Hard-delete a goal arc by ID. Used for transactional rollback only."""
        with get_connection() as conn:
            cursor = conn.execute("DELETE FROM goal_arcs WHERE id = ?", (goal_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_goal_arcs(self, status: Optional[str] = "active") -> list[GoalArc]:
        """List all goal arcs."""
        with get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM goal_arcs WHERE status = ? ORDER BY created_at",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM goal_arcs ORDER BY created_at"
                ).fetchall()
            return [self._row_to_goal_arc(row) for row in rows]

    def get_goal_arc(self, goal_id: str) -> Optional[GoalArc]:
        """Get a goal arc by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM goal_arcs WHERE id = ?", (goal_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_goal_arc(row)

    def _row_to_goal_arc(self, row) -> GoalArc:
        keys = row.keys() if hasattr(row, "keys") else []
        return GoalArc(
            id=row["id"],
            title=row["title"],
            domain=row["domain"] if "domain" in keys and row["domain"] is not None else "",
            execution_mode=row["execution_mode"] if "execution_mode" in keys and row["execution_mode"] else "mixed",
            study_framework=row["study_framework"] if "study_framework" in keys else None,
            current_bloom_stage=_str_to_bloom(row["current_bloom_stage"]) if "current_bloom_stage" in keys else None,
            target_bloom_stage=_str_to_bloom(row["target_bloom_stage"]) if "target_bloom_stage" in keys else None,
            practice_framework=row["practice_framework"] if "practice_framework" in keys else None,
            current_practice_stage=_str_to_practice_stage(row["current_practice_stage"]) if "current_practice_stage" in keys else None,
            target_practice_stage=_str_to_practice_stage(row["target_practice_stage"]) if "target_practice_stage" in keys else None,
            horizon=Horizon(row["horizon"]),
            description=row["description"],
            success_definition=row["success_definition"],
            framework=row["framework"] if "framework" in keys and row["framework"] is not None else "",
            primary_metric=row["primary_metric"] if "primary_metric" in keys else None,
            feedback_source=_str_to_feedback_source(row["feedback_source"]) if "feedback_source" in keys else None,
            evidence_type=_str_to_evidence_type(row["evidence_type"]) if "evidence_type" in keys else None,
            metric_type=row["metric_type"],
            target_value=row["target_value"],
            start_date=_str_to_dt(row["start_date"]),
            target_date=_str_to_dt(row["target_date"]),
            status=row["status"],
            tags=_deserialize_list(row["tags"]),
            generated_names=_deserialize_dict(row["generated_names_json"]) if "generated_names_json" in keys else {},
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )

    # --- Generation Provenance ---

    def create_generation_provenance(self, provenance: GenerationProvenance) -> GenerationProvenance:
        """Persist a minimal audit record for an LLM-generated artifact."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO generation_provenance (
                    id, artifact_kind, artifact_id, generated_by_model,
                    prompt_template_version, source_scope, accepted_by_user, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provenance.id,
                    provenance.artifact_kind,
                    provenance.artifact_id,
                    provenance.generated_by_model,
                    provenance.prompt_template_version,
                    provenance.source_scope,
                    1 if provenance.accepted_by_user else 0,
                    _dt_to_str(provenance.created_at),
                ),
            )
            conn.commit()
        return provenance

    def list_generation_provenance(
        self,
        artifact_kind: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> list[GenerationProvenance]:
        """List provenance rows, optionally filtered to one artifact."""
        with get_connection() as conn:
            if artifact_kind and artifact_id:
                rows = conn.execute(
                    """
                    SELECT * FROM generation_provenance
                    WHERE artifact_kind = ? AND artifact_id = ?
                    ORDER BY created_at DESC
                    """,
                    (artifact_kind, artifact_id),
                ).fetchall()
            elif artifact_kind:
                rows = conn.execute(
                    """
                    SELECT * FROM generation_provenance
                    WHERE artifact_kind = ?
                    ORDER BY created_at DESC
                    """,
                    (artifact_kind,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM generation_provenance ORDER BY created_at DESC"
                ).fetchall()
            return [self._row_to_generation_provenance(row) for row in rows]

    def delete_generation_provenance(
        self,
        *,
        artifact_kind: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> int:
        """Delete provenance rows and return the number removed."""
        with get_connection() as conn:
            if artifact_kind and artifact_id:
                cursor = conn.execute(
                    """
                    DELETE FROM generation_provenance
                    WHERE artifact_kind = ? AND artifact_id = ?
                    """,
                    (artifact_kind, artifact_id),
                )
            elif artifact_id:
                cursor = conn.execute(
                    "DELETE FROM generation_provenance WHERE artifact_id = ?",
                    (artifact_id,),
                )
            elif artifact_kind:
                cursor = conn.execute(
                    "DELETE FROM generation_provenance WHERE artifact_kind = ?",
                    (artifact_kind,),
                )
            else:
                cursor = conn.execute("DELETE FROM generation_provenance")
            conn.commit()
            return cursor.rowcount

    def _row_to_generation_provenance(self, row) -> GenerationProvenance:
        return GenerationProvenance(
            id=row["id"],
            artifact_kind=row["artifact_kind"],
            artifact_id=row["artifact_id"],
            generated_by_model=row["generated_by_model"],
            prompt_template_version=row["prompt_template_version"],
            source_scope=row["source_scope"],
            accepted_by_user=bool(row["accepted_by_user"]),
            created_at=_str_to_dt(row["created_at"]),
        )

    # --- Action Reminders ---

    def create_action_reminder(self, reminder: ActionReminder) -> ActionReminder:
        """Create a queued actionable reminder."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reminder_queue (
                    id, title, message, target_command, status, remind_at,
                    source_kind, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reminder.id,
                    reminder.title,
                    reminder.message,
                    reminder.target_command,
                    reminder.status,
                    _dt_to_str(reminder.remind_at),
                    reminder.source_kind,
                    _dt_to_str(reminder.created_at),
                    _dt_to_str(reminder.updated_at),
                ),
            )
            conn.commit()
        return reminder

    def get_action_reminder(self, reminder_id: str) -> Optional[ActionReminder]:
        """Get an actionable reminder by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM reminder_queue WHERE id = ?",
                (reminder_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_action_reminder(row)

    def list_action_reminders(self, status: Optional[str] = None) -> list[ActionReminder]:
        """List queued reminders ordered by due time."""
        with get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM reminder_queue WHERE status = ? ORDER BY remind_at, created_at",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM reminder_queue ORDER BY remind_at, created_at"
                ).fetchall()
            return [self._row_to_action_reminder(row) for row in rows]

    def list_due_action_reminders(self, now: Optional[datetime] = None) -> list[ActionReminder]:
        """List pending reminders due now or earlier."""
        current = now or datetime.utcnow()
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reminder_queue
                WHERE status = 'pending' AND remind_at <= ?
                ORDER BY remind_at, created_at
                """,
                (_dt_to_str(current),),
            ).fetchall()
            return [self._row_to_action_reminder(row) for row in rows]

    def update_action_reminder(self, reminder: ActionReminder) -> ActionReminder:
        """Update a queued reminder."""
        reminder.updated_at = datetime.utcnow()
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE reminder_queue SET
                    title = ?, message = ?, target_command = ?, status = ?,
                    remind_at = ?, source_kind = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    reminder.title,
                    reminder.message,
                    reminder.target_command,
                    reminder.status,
                    _dt_to_str(reminder.remind_at),
                    reminder.source_kind,
                    _dt_to_str(reminder.updated_at),
                    reminder.id,
                ),
            )
            conn.commit()
        return reminder

    def delete_action_reminder(self, reminder_id: str) -> bool:
        """Delete a queued reminder."""
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM reminder_queue WHERE id = ?",
                (reminder_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_action_reminder(self, row) -> ActionReminder:
        return ActionReminder(
            id=row["id"],
            title=row["title"],
            message=row["message"],
            target_command=row["target_command"],
            status=row["status"],
            remind_at=_str_to_dt(row["remind_at"]),
            source_kind=row["source_kind"],
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )

    # --- Daily Review Responses ---

    def _row_to_review_response(self, row) -> DailyReviewResponse:
        return DailyReviewResponse(
            id=row["id"],
            review_date=row["review_date"],
            question_id=row["question_id"],
            numeric_score=row["numeric_score"],
            text_response=row["text_response"],
            llm_rationale=row["llm_rationale"],
            created_at=_str_to_dt(row["created_at"]),
        )

    def create_review_response(self, response: DailyReviewResponse) -> DailyReviewResponse:
        """Create or update a daily review response (upsert per UNIQUE constraint)."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_review_responses (
                    id, review_date, question_id, numeric_score,
                    text_response, llm_rationale, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_date, question_id) DO UPDATE SET
                    numeric_score = excluded.numeric_score,
                    text_response = excluded.text_response,
                    llm_rationale = excluded.llm_rationale
                """,
                (
                    response.id,
                    response.review_date,
                    response.question_id,
                    response.numeric_score,
                    response.text_response,
                    response.llm_rationale,
                    _dt_to_str(response.created_at),
                ),
            )
            conn.commit()
        return response

    def get_review_responses_for_date(self, review_date: str) -> list[DailyReviewResponse]:
        """Get all review responses for a specific date."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_review_responses WHERE review_date = ? ORDER BY created_at",
                (review_date,),
            ).fetchall()
            return [self._row_to_review_response(row) for row in rows]

    def get_yesterday_response(self, question_id: str, today: str) -> Optional[DailyReviewResponse]:
        """Get yesterday's response for a question (for trend calculation per D-16)."""
        from datetime import datetime, timedelta
        today_dt = datetime.fromisoformat(today)
        yesterday = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_review_responses WHERE review_date = ? AND question_id = ?",
                (yesterday, question_id),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_review_response(row)

    # --- Daily Debriefs ---

    def create_daily_debrief(self, debrief: DailyDebrief) -> DailyDebrief:
        """Store a daily debrief (INSERT OR REPLACE to handle re-running review same day)."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_debriefs (
                    id, review_date, top1_completed, top3_completed, what_shipped,
                    biggest_blocker, blocker_note,
                    energy_morning, energy_midday, energy_evening, energy_task_match,
                    learning_question, learning_answer, learning_score, learning_rationale,
                    tomorrow_top1, tomorrow_next_action, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    debrief.id,
                    debrief.review_date,
                    debrief.top1_completed,
                    _serialize_list(debrief.top3_completed),
                    debrief.what_shipped,
                    debrief.biggest_blocker,
                    debrief.blocker_note,
                    debrief.energy_morning,
                    debrief.energy_midday,
                    debrief.energy_evening,
                    debrief.energy_task_match,
                    debrief.learning_question,
                    debrief.learning_answer,
                    debrief.learning_score,
                    debrief.learning_rationale,
                    debrief.tomorrow_top1,
                    debrief.tomorrow_next_action,
                    _dt_to_str(debrief.created_at),
                ),
            )
            conn.commit()
        return debrief

    def get_daily_debrief(self, review_date: str) -> Optional[DailyDebrief]:
        """Get debrief for a specific date."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_debriefs WHERE review_date = ?",
                (review_date,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_daily_debrief(row)

    def list_tasks_deferred_this_week(self, week_start: datetime) -> list[Task]:
        """Find tasks that were not completed during the given week.

        Returns tasks that:
        - Were created before the end of the week (week_start + 7 days)
        - Have completion < 100
        - Are not archived

        Used by compute_weekly_metrics to surface persistent carry-forward tasks.
        """
        from datetime import timedelta

        week_end = week_start + timedelta(days=7)
        tasks = self.list_tasks()  # excludes archived by default
        return [
            t for t in tasks
            if t.completion < 100
            and t.created_at is not None
            and t.created_at < week_end
        ]

    def list_daily_debriefs(self, days: int = 7) -> list[DailyDebrief]:
        """Get recent debriefs ordered by review_date descending."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM daily_debriefs
                ORDER BY review_date DESC
                LIMIT ?
                """,
                (days,),
            ).fetchall()
            return [self._row_to_daily_debrief(row) for row in rows]

    def _row_to_daily_debrief(self, row) -> DailyDebrief:
        return DailyDebrief(
            id=row["id"],
            review_date=row["review_date"],
            top1_completed=row["top1_completed"],
            top3_completed=_deserialize_list(row["top3_completed"] or "[]"),
            what_shipped=row["what_shipped"],
            biggest_blocker=row["biggest_blocker"],
            blocker_note=row["blocker_note"],
            energy_morning=row["energy_morning"],
            energy_midday=row["energy_midday"],
            energy_evening=row["energy_evening"],
            energy_task_match=row["energy_task_match"],
            learning_question=row["learning_question"],
            learning_answer=row["learning_answer"],
            learning_score=row["learning_score"],
            learning_rationale=row["learning_rationale"],
            tomorrow_top1=row["tomorrow_top1"],
            tomorrow_next_action=row["tomorrow_next_action"],
            created_at=_str_to_dt(row["created_at"]),
        )

    # --- Pause Intervals (D-12, D-13) ---

    def create_pause_interval(self, session_id: str, pause_start: datetime) -> str:
        """Record a pause interval start (D-12)."""
        from pb.domain.models import generate_internal_id

        interval_id = generate_internal_id()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO pause_intervals (id, session_id, pause_start) VALUES (?, ?, ?)",
                (interval_id, session_id, pause_start.isoformat()),
            )
            conn.commit()
        return interval_id

    def resume_pause_interval(self, session_id: str) -> None:
        """Mark the open pause interval as resumed (D-12)."""
        with get_connection() as conn:
            conn.execute(
                "UPDATE pause_intervals SET resume_at = ? WHERE session_id = ? AND resume_at IS NULL",
                (datetime.utcnow().isoformat(), session_id),
            )
            conn.commit()

    def list_pause_intervals(self, session_id: str) -> list[dict]:
        """List all pause intervals for a session."""
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, session_id, pause_start, resume_at FROM pause_intervals WHERE session_id = ? ORDER BY pause_start",
                (session_id,),
            )
            return [
                {"id": row[0], "session_id": row[1], "pause_start": row[2], "resume_at": row[3]}
                for row in cursor.fetchall()
            ]

    def get_stale_pauses(self, max_hours: int = 3) -> list[dict]:
        """Find paused sessions with open intervals older than max_hours (D-13).

        Returns list of dicts with session_id, task_id, pause_start.
        """
        from datetime import timedelta

        cutoff = (datetime.utcnow() - timedelta(hours=max_hours)).isoformat()
        with get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT pi.session_id, s.task_id, pi.pause_start
                FROM pause_intervals pi
                JOIN sessions s ON pi.session_id = s.id
                WHERE pi.resume_at IS NULL
                AND pi.pause_start < ?
                """,
                (cutoff,),
            )
            return [
                {"session_id": row[0], "task_id": row[1], "pause_start": row[2]}
                for row in cursor.fetchall()
            ]

    # --- Product Control ---

    def append_feedback_event(self, event: dict[str, Any]) -> None:
        """Persist one learner feedback event."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feedback_events (
                    id, scope_key, scope, kind, artifact_kind, artifact_id, node_id,
                    label, free_text, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("id", ""),
                    event.get("scope_key", ""),
                    event.get("scope", "artifact"),
                    event.get("kind", ""),
                    event.get("artifact_kind", ""),
                    event.get("artifact_id", ""),
                    event.get("node_id", ""),
                    event.get("label", ""),
                    event.get("free_text", ""),
                    _serialize_dict(event.get("metadata", {}) if isinstance(event.get("metadata", {}), dict) else {}),
                    event.get("timestamp", datetime.utcnow().isoformat()),
                ),
            )
            conn.commit()

    def list_feedback_events(
        self,
        *,
        scope_key: str | None = None,
        artifact_kind: str | None = None,
        artifact_id: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List learner feedback events with optional filters."""
        query = "SELECT * FROM feedback_events"
        clauses: list[str] = []
        params: list[Any] = []
        if scope_key:
            clauses.append("scope_key = ?")
            params.append(scope_key)
        if artifact_kind:
            clauses.append("artifact_kind = ?")
            params.append(artifact_kind)
        if artifact_id:
            clauses.append("artifact_id = ?")
            params.append(artifact_id)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            {
                "id": row["id"],
                "scope_key": row["scope_key"],
                "scope": row["scope"],
                "kind": row["kind"],
                "artifact_kind": row["artifact_kind"],
                "artifact_id": row["artifact_id"],
                "node_id": row["node_id"],
                "label": row["label"],
                "free_text": row["free_text"],
                "metadata": _deserialize_dict(row["metadata_json"]),
                "timestamp": row["created_at"],
            }
            for row in rows
        ]

    def get_control_state_snapshot(self, scope_key: str) -> Optional[dict[str, Any]]:
        """Return the latest stored control-state snapshot."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT state_json FROM control_states WHERE scope_key = ?",
                (scope_key,),
            ).fetchone()
        if row is None:
            return None
        return _deserialize_dict(row["state_json"])

    def save_control_state_snapshot(
        self,
        scope_key: str,
        scope: str,
        state: dict[str, Any],
    ) -> None:
        """Upsert a persisted control-state snapshot."""
        goal_id = str(state.get("goal_id", "") or "")
        task_id = str(state.get("task_id", "") or "")
        session_id = str(state.get("session_id", "") or "")
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO control_states (
                    scope_key, scope, goal_id, task_id, session_id, state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_key,
                    scope,
                    goal_id,
                    task_id,
                    session_id,
                    _serialize_dict(state),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    # --- Context Runtime ---

    def create_context_source(self, source: dict[str, object]) -> dict[str, object]:
        """Persist one durable source-file record."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO context_sources (
                    id, filename, original_path, stored_path, normalized_path, mime_type,
                    canonical_class, source_utility, scope_mode, domain_id, domain_name,
                    scope_boundary, source_ref, ingest_result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(source.get("id", "")),
                    str(source.get("filename", "")),
                    str(source.get("original_path", "")),
                    str(source.get("stored_path", "")),
                    str(source.get("normalized_path", "")),
                    str(source.get("mime_type", "application/octet-stream")),
                    str(source.get("canonical_class", "unknown")),
                    str(source.get("source_utility", "unknown")),
                    str(source.get("scope_mode", "unclear")),
                    str(source.get("domain_id", "") or "") or None,
                    str(source.get("domain_name", "") or "") or None,
                    str(source.get("scope_boundary", "")),
                    str(source.get("source_ref", "")),
                    _serialize_dict(source.get("ingest_result", {}) if isinstance(source.get("ingest_result", {}), dict) else {}),
                    str(source.get("created_at", datetime.utcnow().isoformat())),
                    str(source.get("updated_at", datetime.utcnow().isoformat())),
                ),
            )
            conn.commit()
        return self.get_context_source(str(source.get("id", ""))) or dict(source)

    def update_context_source(self, source: dict[str, object]) -> dict[str, object]:
        """Update one durable source-file record."""
        updated_at = datetime.utcnow().isoformat()
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE context_sources SET
                    filename = ?, original_path = ?, stored_path = ?, normalized_path = ?, mime_type = ?,
                    canonical_class = ?, source_utility = ?, scope_mode = ?, domain_id = ?, domain_name = ?,
                    scope_boundary = ?, source_ref = ?, ingest_result_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    str(source.get("filename", "")),
                    str(source.get("original_path", "")),
                    str(source.get("stored_path", "")),
                    str(source.get("normalized_path", "")),
                    str(source.get("mime_type", "application/octet-stream")),
                    str(source.get("canonical_class", "unknown")),
                    str(source.get("source_utility", "unknown")),
                    str(source.get("scope_mode", "unclear")),
                    str(source.get("domain_id", "") or "") or None,
                    str(source.get("domain_name", "") or "") or None,
                    str(source.get("scope_boundary", "")),
                    str(source.get("source_ref", "")),
                    _serialize_dict(source.get("ingest_result", {}) if isinstance(source.get("ingest_result", {}), dict) else {}),
                    updated_at,
                    str(source.get("id", "")),
                ),
            )
            conn.commit()
        return self.get_context_source(str(source.get("id", ""))) or {**source, "updated_at": updated_at}

    def get_context_source(self, source_id: str) -> Optional[dict[str, object]]:
        """Return one durable source-file record."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM context_sources WHERE id = ?",
                (source_id,),
            ).fetchone()
        return self._row_to_context_source(row) if row is not None else None

    def find_context_source(self, ref: str) -> Optional[dict[str, object]]:
        """Resolve a source by id, source_ref, original path, filename, or filename stem."""
        query = (ref or "").strip()
        if not query:
            return None
        query_name = Path(query).name
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM context_sources
                WHERE id IN (?, ?)
                   OR source_ref IN (?, ?)
                   OR original_path IN (?, ?)
                   OR filename IN (?, ?)
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (query, query_name, query, query_name, query, query_name, query, query_name),
            ).fetchone()
            if row is None:
                rows = conn.execute(
                    "SELECT * FROM context_sources ORDER BY updated_at DESC, created_at DESC"
                ).fetchall()
        if row is not None:
            return self._row_to_context_source(row)
        query_stem = Path(query_name).stem
        if not query_stem:
            return None
        for candidate in rows:
            filename = str(candidate["filename"] or "")
            original_path = str(candidate["original_path"] or "")
            if Path(filename).stem == query_stem or Path(original_path).stem == query_stem:
                return self._row_to_context_source(candidate)
        return None

    def list_context_sources(self) -> list[dict[str, object]]:
        """List durable source-file records in newest-first order."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM context_sources ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._row_to_context_source(row) for row in rows]

    def delete_context_source(self, ref: str) -> Optional[dict[str, object]]:
        """Delete one durable source-file record by id, ref, path, or filename."""
        source = self.find_context_source(ref)
        if source is None:
            return None
        source_id = str(source.get("id", ""))
        with get_connection() as conn:
            conn.execute("DELETE FROM source_bundle_items WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM context_sources WHERE id = ?", (source_id,))
            conn.commit()
        return source

    def create_source_bundle(self, bundle: SourceBundle) -> SourceBundle:
        """Persist one named source bundle."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO source_bundles (
                    id, name, domain_id, domain_name, scope_mode, scope_boundary,
                    source_refs_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bundle.id,
                    bundle.name,
                    bundle.domain_id,
                    bundle.domain_name,
                    bundle.scope_mode,
                    bundle.scope_boundary,
                    _serialize_json_list(list(bundle.source_refs)),
                    bundle.created_at,
                    bundle.updated_at,
                ),
            )
            conn.commit()
        for item in bundle.items:
            self.add_source_bundle_item(item)
        return self.get_source_bundle(bundle.id) or bundle

    def update_source_bundle(self, bundle: SourceBundle) -> SourceBundle:
        """Update one named source bundle."""
        bundle.updated_at = datetime.utcnow().isoformat()
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE source_bundles SET
                    name = ?, domain_id = ?, domain_name = ?, scope_mode = ?,
                    scope_boundary = ?, source_refs_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    bundle.name,
                    bundle.domain_id,
                    bundle.domain_name,
                    bundle.scope_mode,
                    bundle.scope_boundary,
                    _serialize_json_list(list(bundle.source_refs)),
                    bundle.updated_at,
                    bundle.id,
                ),
            )
            conn.commit()
        return self.get_source_bundle(bundle.id) or bundle

    def get_source_bundle(self, bundle_id: str) -> Optional[SourceBundle]:
        """Return one source bundle by id."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM source_bundles WHERE id = ?",
                (bundle_id,),
            ).fetchone()
        if row is None:
            return None
        bundle = self._row_to_source_bundle(row)
        bundle.items = self.list_source_bundle_items(bundle.id)
        return bundle

    def get_source_bundle_by_name(self, name: str) -> Optional[SourceBundle]:
        """Return one source bundle by exact name."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM source_bundles WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        bundle = self._row_to_source_bundle(row)
        bundle.items = self.list_source_bundle_items(bundle.id)
        return bundle

    def list_source_bundles(self) -> list[SourceBundle]:
        """List all stored bundles in newest-first order."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM source_bundles ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        bundles = [self._row_to_source_bundle(row) for row in rows]
        for bundle in bundles:
            bundle.items = self.list_source_bundle_items(bundle.id)
        return bundles

    def add_source_bundle_item(self, item: SourceBundleItem) -> SourceBundleItem:
        """Add or replace one source bundle item."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO source_bundle_items (
                    id, bundle_id, source_id, position, source_ref, filename, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.bundle_id,
                    item.source_id,
                    item.position,
                    item.source_ref,
                    item.filename,
                    item.created_at,
                ),
            )
            conn.commit()
        bundle = self.get_source_bundle(item.bundle_id)
        if bundle is not None:
            bundle.source_refs = [bundle_item.source_ref for bundle_item in bundle.items]
            self.update_source_bundle(bundle)
        return item

    def remove_source_bundle_sources(self, bundle_id: str, source_ids: Iterable[str]) -> int:
        """Remove selected sources from a bundle."""
        targets = [str(source_id).strip() for source_id in source_ids if str(source_id).strip()]
        if not targets:
            return 0
        placeholders = ", ".join("?" for _ in targets)
        with get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM source_bundle_items WHERE bundle_id = ? AND source_id IN ({placeholders})",
                (bundle_id, *targets),
            )
            conn.commit()
        bundle = self.get_source_bundle(bundle_id)
        if bundle is not None:
            bundle.source_refs = [bundle_item.source_ref for bundle_item in bundle.items]
            self.update_source_bundle(bundle)
        return int(cursor.rowcount)

    def list_source_bundle_items(self, bundle_id: str) -> list[SourceBundleItem]:
        """List items for one source bundle in stable order."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM source_bundle_items
                WHERE bundle_id = ?
                ORDER BY position, created_at
                """,
                (bundle_id,),
            ).fetchall()
        return [self._row_to_source_bundle_item(row) for row in rows]

    def set_locked_context(self, scope: ActiveContextScope) -> ActiveContextScope:
        """Persist the current locked context scope."""
        now = datetime.utcnow().isoformat()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_lock_state (
                    scope_key, mode, locked, label, label_max_chars, scope_mode,
                    source_bundle_id, source_refs_json, domain_id, scope_boundary, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "locked",
                    scope.mode,
                    1 if scope.locked else 0,
                    scope.label,
                    scope.label_max_chars,
                    scope.scope_mode,
                    scope.source_bundle_id,
                    _serialize_json_list(list(scope.source_refs)),
                    scope.domain_id,
                    scope.scope_boundary,
                    now,
                ),
            )
            conn.commit()
        return self.get_locked_context() or scope

    def get_locked_context(self) -> Optional[ActiveContextScope]:
        """Return the persisted locked context scope, if any."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM context_lock_state WHERE scope_key = ?",
                ("locked",),
            ).fetchone()
        if row is None or not bool(row["locked"]):
            return None
        return self._row_to_active_context_scope(row)

    def clear_locked_context(self) -> None:
        """Remove the persisted locked context scope."""
        with get_connection() as conn:
            conn.execute("DELETE FROM context_lock_state WHERE scope_key = ?", ("locked",))
            conn.commit()

    def upsert_provider_capability_rule(self, rule: FileSupportDecision) -> FileSupportDecision:
        """Persist one provider capability rule."""
        now = datetime.utcnow().isoformat()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO provider_capability_rules (
                    id, provider, model, endpoint, delivery, canonical_class, exact_mimes_json,
                    exact_extensions_json, max_file_size_mb, documented_support, probe_status,
                    support_mode, notes, created_at, updated_at
                ) VALUES (
                    COALESCE(
                        (SELECT id FROM provider_capability_rules
                         WHERE provider = ? AND model = ? AND endpoint = ? AND delivery = ? AND canonical_class = ?),
                        NULL
                    ),
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                        (SELECT created_at FROM provider_capability_rules
                         WHERE provider = ? AND model = ? AND endpoint = ? AND delivery = ? AND canonical_class = ?),
                        ?
                    ), ?
                )
                """,
                (
                    rule.provider,
                    rule.model,
                    rule.endpoint,
                    rule.delivery,
                    rule.canonical_class,
                    rule.provider,
                    rule.model,
                    rule.endpoint,
                    rule.delivery,
                    rule.canonical_class,
                    _serialize_json_list(list(rule.exact_mimes)),
                    _serialize_json_list(list(rule.exact_extensions)),
                    rule.max_file_size_mb,
                    1 if rule.documented_support else 0,
                    rule.probe_status,
                    rule.support_mode,
                    rule.notes,
                    rule.provider,
                    rule.model,
                    rule.endpoint,
                    rule.delivery,
                    rule.canonical_class,
                    now,
                    now,
                ),
            )
            conn.commit()
        return rule

    def list_provider_capability_rules(self, provider: str = "", model: str = "") -> list[FileSupportDecision]:
        """List persisted provider capability rules."""
        query = "SELECT * FROM provider_capability_rules"
        clauses: list[str] = []
        params: list[object] = []
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if model:
            clauses.append("model = ?")
            params.append(model)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY provider, model, canonical_class"
        with get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_provider_capability_rule(row) for row in rows]

    def _row_to_context_source(self, row) -> dict[str, object]:
        return {
            "id": row["id"],
            "filename": row["filename"],
            "original_path": row["original_path"],
            "stored_path": row["stored_path"],
            "normalized_path": row["normalized_path"],
            "mime_type": row["mime_type"],
            "canonical_class": row["canonical_class"],
            "source_utility": row["source_utility"],
            "scope_mode": row["scope_mode"],
            "domain_id": row["domain_id"],
            "domain_name": row["domain_name"],
            "scope_boundary": row["scope_boundary"],
            "source_ref": row["source_ref"],
            "ingest_result": _deserialize_dict(row["ingest_result_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_source_bundle_item(self, row) -> SourceBundleItem:
        return SourceBundleItem(
            id=row["id"],
            bundle_id=row["bundle_id"],
            source_id=row["source_id"],
            position=row["position"],
            source_ref=row["source_ref"],
            filename=row["filename"],
            created_at=row["created_at"],
        )

    def _row_to_source_bundle(self, row) -> SourceBundle:
        return SourceBundle(
            id=row["id"],
            name=row["name"],
            domain_id=row["domain_id"],
            domain_name=row["domain_name"],
            scope_mode=row["scope_mode"],
            scope_boundary=row["scope_boundary"],
            source_refs=[str(item) for item in _deserialize_json_list(row["source_refs_json"]) if str(item).strip()],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_active_context_scope(self, row) -> ActiveContextScope:
        return ActiveContextScope(
            mode=row["mode"],
            locked=bool(row["locked"]),
            label=row["label"],
            label_max_chars=row["label_max_chars"],
            scope_mode=row["scope_mode"],
            source_bundle_id=row["source_bundle_id"],
            source_refs=[str(item) for item in _deserialize_json_list(row["source_refs_json"]) if str(item).strip()],
            domain_id=row["domain_id"],
            scope_boundary=row["scope_boundary"],
        )

    def _row_to_provider_capability_rule(self, row) -> FileSupportDecision:
        return FileSupportDecision(
            provider=row["provider"],
            model=row["model"],
            endpoint=row["endpoint"],
            delivery=row["delivery"],
            canonical_class=row["canonical_class"],
            exact_mimes=[str(item) for item in _deserialize_json_list(row["exact_mimes_json"]) if str(item).strip()],
            exact_extensions=[str(item) for item in _deserialize_json_list(row["exact_extensions_json"]) if str(item).strip()],
            max_file_size_mb=row["max_file_size_mb"],
            documented_support=bool(row["documented_support"]),
            probe_status=row["probe_status"],
            support_mode=row["support_mode"],
            notes=row["notes"],
        )
