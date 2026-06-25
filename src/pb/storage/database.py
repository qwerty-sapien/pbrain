# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""SQLite database setup and connection management.

Uses WAL mode for better concurrent read performance.
Schema matches domain models from DATA_MODEL.md.
"""

import datetime as _dt
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from pb.storage.config import get_data_dir

DB_FILENAME = "productivebrain.db"

_db_path: Optional[Path] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS goal_arcs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    domain TEXT DEFAULT '',
    execution_mode TEXT DEFAULT 'mixed',
    study_framework TEXT DEFAULT NULL,
    current_bloom_stage TEXT DEFAULT NULL,
    target_bloom_stage TEXT DEFAULT NULL,
    practice_framework TEXT DEFAULT NULL,
    current_practice_stage TEXT DEFAULT NULL,
    target_practice_stage TEXT DEFAULT NULL,
    horizon TEXT NOT NULL DEFAULT 'six_month',
    description TEXT DEFAULT '',
    success_definition TEXT DEFAULT '',
    framework TEXT DEFAULT '',
    primary_metric TEXT DEFAULT NULL,
    feedback_source TEXT DEFAULT NULL,
    evidence_type TEXT DEFAULT NULL,
    metric_type TEXT,
    target_value REAL,
    start_date TEXT,
    target_date TEXT,
    status TEXT DEFAULT 'active',
    tags TEXT DEFAULT '[]',
    generated_names_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    linked_goal_arc_ids TEXT DEFAULT '[]',
    cadence TEXT,
    priority_weight REAL DEFAULT 1.0,
    active INTEGER DEFAULT 1,
    generated_names_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    project_type TEXT NOT NULL DEFAULT 'build',
    track_id TEXT,
    repo_path TEXT,
    packet_path TEXT NOT NULL,
    status TEXT DEFAULT 'ready',
    next_review_at TEXT,
    tags TEXT DEFAULT '[]',
    generated_names_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (track_id) REFERENCES tracks(id)
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    horizon TEXT DEFAULT 'today',
    state TEXT DEFAULT 'inbox',
    estimate_minutes INTEGER,
    scheduled_start TEXT,
    scheduled_end TEXT,
    energy_type TEXT DEFAULT 'deep',
    linked_goal_arc_ids TEXT DEFAULT '[]',
    linked_track_ids TEXT DEFAULT '[]',
    packet_path TEXT,
    generated_names_json TEXT DEFAULT '{}',
    interruption_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    archived_at TEXT DEFAULT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    start_at TEXT NOT NULL,
    end_at TEXT,
    mode TEXT DEFAULT 'focus',
    branch TEXT DEFAULT 'study',
    goal_id TEXT DEFAULT NULL,
    track_id TEXT DEFAULT NULL,
    subject_scope TEXT DEFAULT '',
    bloom_stage TEXT,
    target_bloom_stage TEXT,
    practice_stage TEXT,
    drill_type TEXT,
    constraint_text TEXT,
    feedback_source TEXT,
    evidence_target TEXT,
    coach_cues TEXT,
    observed_errors TEXT,
    quality_rating INTEGER DEFAULT NULL,
    difficulty_rating INTEGER DEFAULT NULL,
    next_adjustment TEXT,
    intended_outcome TEXT DEFAULT '',
    actual_outcome TEXT,
    generated_names_json TEXT DEFAULT '{}',
    interruption_count INTEGER DEFAULT 0,
    handoff_path TEXT,
    llm_summary_used INTEGER DEFAULT 0,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS time_blocks (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    start_time TEXT,
    duration_minutes INTEGER NOT NULL,
    block_kind TEXT DEFAULT 'study',
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS packets (
    path TEXT PRIMARY KEY,
    packet_type TEXT NOT NULL,
    linked_entity_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    source_url TEXT,
    title TEXT,
    captured_text TEXT NOT NULL,
    summary TEXT,
    linked_project_id TEXT,
    linked_task_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (linked_project_id) REFERENCES projects(id),
    FOREIGN KEY (linked_task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS daily_review_responses (
    id TEXT PRIMARY KEY,
    review_date TEXT NOT NULL,
    question_id TEXT NOT NULL,
    numeric_score INTEGER NOT NULL,
    text_response TEXT,
    llm_rationale TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(review_date, question_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_task ON sessions(task_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_at);
CREATE INDEX IF NOT EXISTS idx_review_responses_date ON daily_review_responses(review_date);

CREATE TABLE IF NOT EXISTS reminder_queue (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    target_command TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    remind_at TEXT NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminder_queue_status ON reminder_queue(status);
CREATE INDEX IF NOT EXISTS idx_reminder_queue_remind_at ON reminder_queue(remind_at);

CREATE TABLE IF NOT EXISTS generation_provenance (
    id TEXT PRIMARY KEY,
    artifact_kind TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    generated_by_model TEXT NOT NULL,
    prompt_template_version TEXT NOT NULL,
    source_scope TEXT DEFAULT '',
    accepted_by_user INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generation_provenance_artifact ON generation_provenance(artifact_kind, artifact_id);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_kind TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    alias_kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (entity_kind, alias)
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_kind, entity_id);
"""


def get_db_path() -> Path:
    """Get the database file path."""
    global _db_path
    if _db_path is None:
        data_dir = get_data_dir()
        _db_path = data_dir / DB_FILENAME
    return _db_path


def set_db_path(path: Path) -> None:
    """Set a custom database path (for testing)."""
    global _db_path
    _db_path = path


def _migrate_sessions_phase8(conn: sqlite3.Connection) -> None:
    """Add Phase 8 fields to sessions table if not present (idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    new_cols = [
        ("expectation", "TEXT DEFAULT NULL"),
        ("completion_pct", "INTEGER DEFAULT NULL"),
        ("distraction", "INTEGER DEFAULT NULL"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
    if "constraint" in existing and "constraint_text" in {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}:
        conn.execute(
            "UPDATE sessions SET constraint_text = constraint "
            "WHERE constraint_text IS NULL AND constraint IS NOT NULL"
        )
    conn.commit()


def _migrate_time_blocks_recurrence(conn: sqlite3.Connection) -> None:
    """Add Phase 2 recurrence fields to time_blocks table if not present (idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(time_blocks)").fetchall()}
    new_cols = [
        ("series_id", "TEXT DEFAULT NULL"),
        ("recurrence_rule", "TEXT DEFAULT NULL"),
        ("block_kind", "TEXT DEFAULT 'study'"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE time_blocks ADD COLUMN {col_name} {col_def}")
    conn.commit()


def _migrate_goal_arc_focus(conn: sqlite3.Connection) -> None:
    """Add goal focus columns for goal-first CLI flow."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(goal_arcs)").fetchall()}
    new_cols = [
        ("domain", "TEXT DEFAULT ''"),
        ("execution_mode", "TEXT DEFAULT 'mixed'"),
        ("study_framework", "TEXT DEFAULT NULL"),
        ("current_bloom_stage", "TEXT DEFAULT NULL"),
        ("target_bloom_stage", "TEXT DEFAULT NULL"),
        ("practice_framework", "TEXT DEFAULT NULL"),
        ("current_practice_stage", "TEXT DEFAULT NULL"),
        ("target_practice_stage", "TEXT DEFAULT NULL"),
        ("framework", "TEXT DEFAULT ''"),
        ("primary_metric", "TEXT DEFAULT NULL"),
        ("feedback_source", "TEXT DEFAULT NULL"),
        ("evidence_type", "TEXT DEFAULT NULL"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE goal_arcs ADD COLUMN {col_name} {col_def}")
    conn.commit()


def _migrate_sessions_goal_practise(conn: sqlite3.Connection) -> None:
    """Add study/practise session metadata columns if not present."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    new_cols = [
        ("branch", "TEXT DEFAULT 'study'"),
        ("goal_id", "TEXT DEFAULT NULL"),
        ("track_id", "TEXT DEFAULT NULL"),
        ("subject_scope", "TEXT DEFAULT ''"),
        ("bloom_stage", "TEXT DEFAULT NULL"),
        ("target_bloom_stage", "TEXT DEFAULT NULL"),
        ("practice_stage", "TEXT DEFAULT NULL"),
        ("drill_type", "TEXT DEFAULT NULL"),
        ("constraint_text", "TEXT DEFAULT NULL"),
        ("feedback_source", "TEXT DEFAULT NULL"),
        ("evidence_target", "TEXT DEFAULT NULL"),
        ("coach_cues", "TEXT DEFAULT NULL"),
        ("observed_errors", "TEXT DEFAULT NULL"),
        ("quality_rating", "INTEGER DEFAULT NULL"),
        ("difficulty_rating", "INTEGER DEFAULT NULL"),
        ("next_adjustment", "TEXT DEFAULT NULL"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
    conn.commit()


def _migrate_reminder_queue(conn: sqlite3.Connection) -> None:
    """Create queued actionable reminder table if not present."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reminder_queue (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            target_command TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            remind_at TEXT NOT NULL,
            source_kind TEXT NOT NULL DEFAULT 'manual',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reminder_queue_status ON reminder_queue(status);
        CREATE INDEX IF NOT EXISTS idx_reminder_queue_remind_at ON reminder_queue(remind_at);
    """)
    conn.commit()


def _migrate_generation_provenance(conn: sqlite3.Connection) -> None:
    """Create a lightweight audit table for LLM-generated artifacts."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS generation_provenance (
            id TEXT PRIMARY KEY,
            artifact_kind TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            generated_by_model TEXT NOT NULL,
            prompt_template_version TEXT NOT NULL,
            source_scope TEXT DEFAULT '',
            accepted_by_user INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_generation_provenance_artifact
            ON generation_provenance(artifact_kind, artifact_id);
    """)
    conn.commit()


def _migrate_daily_debriefs(conn: sqlite3.Connection) -> None:
    """Create daily_debriefs table if not exists (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_debriefs (
            id TEXT PRIMARY KEY,
            review_date TEXT NOT NULL UNIQUE,
            top1_completed TEXT,
            top3_completed TEXT DEFAULT '[]',
            what_shipped TEXT,
            biggest_blocker TEXT,
            blocker_note TEXT,
            energy_morning INTEGER,
            energy_midday INTEGER,
            energy_evening INTEGER,
            energy_task_match TEXT,
            learning_question TEXT,
            learning_answer TEXT,
            learning_score INTEGER,
            learning_rationale TEXT,
            tomorrow_top1 TEXT,
            tomorrow_next_action TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()


def _migrate_pause_intervals(conn: sqlite3.Connection) -> None:
    """Create pause_intervals table if not exists (D-12, idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pause_intervals (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            pause_start TEXT NOT NULL,
            resume_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()


def _migrate_usage_log(conn: sqlite3.Connection) -> None:
    """Add usage_log table if not present (Phase 9, D-03, idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            command   TEXT NOT NULL,
            exit_code INTEGER NOT NULL DEFAULT 0,
            error     TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_usage_log_timestamp ON usage_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_usage_log_command   ON usage_log(command);
    """)
    conn.commit()


def _migrate_task_skills(conn: sqlite3.Connection) -> None:
    """Add task_skills junction table if not present (Phase 12, D-05, idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_skills (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            tagged_at  TEXT NOT NULL,
            source     TEXT NOT NULL DEFAULT 'manual',
            UNIQUE(task_id, skill_name),
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        );
        CREATE INDEX IF NOT EXISTS idx_task_skills_task  ON task_skills(task_id);
        CREATE INDEX IF NOT EXISTS idx_task_skills_skill ON task_skills(skill_name);
    """)
    conn.commit()


def _migrate_task_completion(conn: sqlite3.Connection) -> None:
    """Add completion score and pause fields; collapse 8 states to 3 (Phase 12.1, idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    new_cols = [
        ("completion", "INTEGER DEFAULT 0"),
        ("paused_until", "TEXT DEFAULT NULL"),
        ("pause_reason", "TEXT DEFAULT NULL"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {col_name} {col_def}")
    conn.execute("UPDATE tasks SET completion = 100 WHERE state IN ('done', 'cancelled') AND (completion IS NULL OR completion = 0)")
    conn.execute("UPDATE tasks SET state = 'done' WHERE state = 'cancelled'")
    conn.execute("UPDATE tasks SET state = 'active' WHERE state IN ('inbox', 'ready', 'waiting', 'blocked')")
    conn.execute("UPDATE tasks SET state = 'active' WHERE state = 'paused' AND paused_until IS NULL")
    conn.commit()


def _migrate_interactions(conn: sqlite3.Connection) -> None:
    """Add interactions table for learning lifecycle (Phase 17, idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS interactions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            note_path TEXT NOT NULL,
            event_type TEXT NOT NULL,
            weight    REAL NOT NULL DEFAULT 1.0,
            ts        TEXT NOT NULL,
            domain    TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_interactions_note  ON interactions(note_path);
        CREATE INDEX IF NOT EXISTS idx_interactions_ts    ON interactions(ts);
        CREATE INDEX IF NOT EXISTS idx_interactions_event ON interactions(event_type);
    """)
    conn.commit()


def _migrate_product_control(conn: sqlite3.Connection) -> None:
    """Add learner-control persistence tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback_events (
            id TEXT PRIMARY KEY,
            scope_key TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'artifact',
            kind TEXT NOT NULL,
            artifact_kind TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            node_id TEXT DEFAULT '',
            label TEXT DEFAULT '',
            free_text TEXT DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_events_scope_key ON feedback_events(scope_key);
        CREATE INDEX IF NOT EXISTS idx_feedback_events_artifact ON feedback_events(artifact_kind, artifact_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_events_kind ON feedback_events(kind);
        CREATE INDEX IF NOT EXISTS idx_feedback_events_created_at ON feedback_events(created_at);

        CREATE TABLE IF NOT EXISTS control_states (
            scope_key TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            goal_id TEXT DEFAULT '',
            task_id TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            state_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_control_states_scope ON control_states(scope);
        CREATE INDEX IF NOT EXISTS idx_control_states_goal ON control_states(goal_id);
        CREATE INDEX IF NOT EXISTS idx_control_states_task ON control_states(task_id);
        CREATE INDEX IF NOT EXISTS idx_control_states_session ON control_states(session_id);
    """)
    conn.commit()


def _migrate_agent_weights(conn: sqlite3.Connection) -> None:
    """Add agent-weight event and cache tables (Phase 12, idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_weight_events (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            session_id TEXT NOT NULL DEFAULT '',
            event_kind TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            base_weight REAL NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_agent_weight_events_agent
            ON agent_weight_events(agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_weight_events_session
            ON agent_weight_events(session_id);
        CREATE INDEX IF NOT EXISTS idx_agent_weight_events_created_at
            ON agent_weight_events(created_at);

        CREATE TABLE IF NOT EXISTS agent_weight_cache (
            agent_id TEXT PRIMARY KEY,
            frecency_score REAL NOT NULL,
            dispatch_prior REAL NOT NULL,
            short_frecency REAL NOT NULL,
            medium_frecency REAL NOT NULL,
            long_frecency REAL NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_agent_weight_cache_updated_at
            ON agent_weight_cache(updated_at);

        CREATE TABLE IF NOT EXISTS agent_frecency_scores (
            agent_id TEXT PRIMARY KEY,
            frecency_score REAL NOT NULL,
            dispatch_prior REAL NOT NULL,
            short_frecency REAL NOT NULL,
            medium_frecency REAL NOT NULL,
            long_frecency REAL NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_agent_frecency_scores_updated_at
            ON agent_frecency_scores(updated_at);
    """)
    conn.commit()


def _migrate_agent_instruction_judge(conn: sqlite3.Connection) -> None:
    """Add agent-instruction patch proposals (Phase 13, idempotent)."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_instruction_patches (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            session_id TEXT DEFAULT NULL,
            status TEXT NOT NULL DEFAULT 'proposed',
            trigger_kind TEXT NOT NULL DEFAULT 'manual',
            confidence REAL NOT NULL DEFAULT 0.0,
            summary TEXT NOT NULL DEFAULT '',
            instruction_patch TEXT NOT NULL DEFAULT '',
            previous_instruction TEXT NOT NULL DEFAULT '',
            clarifying_question TEXT NOT NULL DEFAULT '',
            evidence_json TEXT NOT NULL DEFAULT '[]',
            model_tier TEXT NOT NULL DEFAULT 'mid',
            created_at TEXT NOT NULL,
            applied_at TEXT DEFAULT NULL,
            reverted_at TEXT DEFAULT NULL,
            FOREIGN KEY (session_id) REFERENCES dispatch_sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_instruction_patches_agent_status
            ON agent_instruction_patches(agent_id, status);
        CREATE INDEX IF NOT EXISTS idx_agent_instruction_patches_session
            ON agent_instruction_patches(session_id);
        """
    )
    conn.commit()


def _migrate_anki_cards(conn: sqlite3.Connection) -> None:
    """Add anki_cards table for Phase 18 Anki pipeline (idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS anki_cards (
            id          TEXT PRIMARY KEY,
            note_slug   TEXT NOT NULL,
            front       TEXT NOT NULL,
            back        TEXT NOT NULL,
            card_type   TEXT NOT NULL DEFAULT 'auto',
            status      TEXT NOT NULL DEFAULT 'pending',
            deck        TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '[]',
            anki_model  TEXT NOT NULL DEFAULT 'Basic',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_anki_cards_note   ON anki_cards(note_slug);
        CREATE INDEX IF NOT EXISTS idx_anki_cards_status ON anki_cards(status);
        CREATE INDEX IF NOT EXISTS idx_anki_cards_deck   ON anki_cards(deck);
    """)
    conn.commit()


def _migrate_anki_revlog(conn: sqlite3.Connection) -> None:
    """Add anki_revlog table for Phase 18 revlog tracking (idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS anki_revlog (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            deck          TEXT NOT NULL,
            cards_total   INTEGER NOT NULL DEFAULT 0,
            reviews_total INTEGER NOT NULL DEFAULT 0,
            pulled_at     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_anki_revlog_deck   ON anki_revlog(deck);
        CREATE INDEX IF NOT EXISTS idx_anki_revlog_pulled ON anki_revlog(pulled_at);
    """)
    conn.commit()


def _migrate_domain_weekly_stats(conn: sqlite3.Connection) -> None:
    """Add domain_weekly_stats table for Phase 19 analytics (idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS domain_weekly_stats (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            domain            TEXT NOT NULL,
            week_start        TEXT NOT NULL,
            notes_created     INTEGER NOT NULL DEFAULT 0,
            links_added       INTEGER NOT NULL DEFAULT 0,
            anki_exported     INTEGER NOT NULL DEFAULT 0,
            socratic_sessions INTEGER NOT NULL DEFAULT 0,
            stage_new         INTEGER NOT NULL DEFAULT 0,
            stage_learning    INTEGER NOT NULL DEFAULT 0,
            stage_learnt      INTEGER NOT NULL DEFAULT 0,
            stage_stale       INTEGER NOT NULL DEFAULT 0,
            snapshot_at       TEXT NOT NULL,
            UNIQUE(domain, week_start)
        );
        CREATE INDEX IF NOT EXISTS idx_domain_weekly_domain ON domain_weekly_stats(domain);
        CREATE INDEX IF NOT EXISTS idx_domain_weekly_week   ON domain_weekly_stats(week_start);
    """)
    conn.commit()


def _migrate_sessions_phase23(conn: sqlite3.Connection) -> None:
    """Add Phase 23 timer fields to sessions table if not present (idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    new_cols = [
        ("duration_minutes", "INTEGER DEFAULT NULL"),
        ("timer_mode", "TEXT DEFAULT 'stopwatch'"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
    conn.commit()


def _migrate_anki_cards_phase26(conn: sqlite3.Connection) -> None:
    """Add Phase 26 columns to anki_cards (idempotent). D-12."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(anki_cards)").fetchall()}
    new_cols = [
        ("domain",       "VARCHAR DEFAULT NULL"),
        ("exported_at",  "TEXT DEFAULT NULL"),
        ("anki_note_id", "INTEGER DEFAULT NULL"),
        ("run_id",       "TEXT DEFAULT NULL"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE anki_cards ADD COLUMN {col_name} {col_def}")
    conn.commit()


def _migrate_generation_run_log(conn: sqlite3.Connection) -> None:
    """Create generation_run_log table for pb anki history/rollback (idempotent). D-13."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation_run_log (
            run_id     TEXT PRIMARY KEY,
            note_slug  TEXT,
            term       TEXT,
            card_count INTEGER,
            source     TEXT,
            created_at TEXT
        )
    """)
    conn.commit()


def _migrate_anki_card_statuses(conn: sqlite3.Connection) -> None:
    """Normalize legacy Anki card statuses to the learning-evidence lifecycle."""
    conn.execute(
        "UPDATE anki_cards SET status = 'accepted' WHERE status = 'pending'"
    )
    conn.execute(
        "UPDATE anki_cards SET status = 'edited' WHERE status = 'reviewed'"
    )
    conn.commit()


def _migrate_tasks_priority(conn: sqlite3.Connection) -> None:
    """Add Phase 2 priority fields to tasks table if not present (idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    new_cols = [
        ("impact", "INTEGER DEFAULT NULL"),
        ("urgency_score", "INTEGER DEFAULT NULL"),
        ("strategic_value", "INTEGER DEFAULT NULL"),
        ("effort", "INTEGER DEFAULT NULL"),
        ("important", "INTEGER DEFAULT NULL"),
        ("urgent", "INTEGER DEFAULT NULL"),
        ("energy_required", "INTEGER DEFAULT NULL"),
        ("work_type", "TEXT DEFAULT NULL"),
        ("due_date", "TEXT DEFAULT NULL"),
        ("scheduled_date", "TEXT DEFAULT NULL"),
        ("estimated_minutes", "INTEGER DEFAULT NULL"),
        ("actual_minutes", "INTEGER DEFAULT NULL"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {col_name} {col_def}")
    conn.commit()


def _migrate_metric_tables(conn: sqlite3.Connection) -> None:
    """Create per-vault metric tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metric_definitions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            unit TEXT DEFAULT '',
            domain TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metric_assignments (
            id TEXT PRIMARY KEY,
            metric_id TEXT NOT NULL,
            goal_id TEXT DEFAULT '',
            goal_title TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (metric_id) REFERENCES metric_definitions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_metric_assignments_goal ON metric_assignments(goal_id);
    """)
    conn.commit()


def _migrate_vault_note_index(conn: sqlite3.Connection) -> None:
    """Create a lightweight mirrored markdown index for sync operations."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vault_notes (
            id TEXT PRIMARY KEY,
            note_type TEXT DEFAULT '',
            slug TEXT DEFAULT '',
            path TEXT NOT NULL UNIQUE,
            title TEXT DEFAULT '',
            domain TEXT DEFAULT '',
            updated_at TEXT NOT NULL,
            source_ref TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_vault_notes_slug ON vault_notes(slug);
        CREATE INDEX IF NOT EXISTS idx_vault_notes_type ON vault_notes(note_type);
    """)
    conn.commit()


def _migrate_evidence_phase2(conn: sqlite3.Connection) -> None:
    """Add evidence_notes and retry_queue tables (Phase 2 -- evidence system)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS evidence_notes (
            id          TEXT PRIMARY KEY,
            path        TEXT NOT NULL UNIQUE,
            domain      TEXT NOT NULL DEFAULT '',
            date        TEXT NOT NULL,
            slug        TEXT NOT NULL DEFAULT '',
            duration_min INTEGER DEFAULT 0,
            outcome     TEXT DEFAULT '',
            sub_skills  TEXT DEFAULT '[]',
            retry_count INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_evidence_domain ON evidence_notes(domain);
        CREATE INDEX IF NOT EXISTS idx_evidence_date   ON evidence_notes(date);
        CREATE INDEX IF NOT EXISTS idx_evidence_outcome ON evidence_notes(outcome);

        CREATE TABLE IF NOT EXISTS retry_queue (
            id          TEXT PRIMARY KEY,
            domain      TEXT NOT NULL DEFAULT '',
            item_text   TEXT NOT NULL,
            source      TEXT NOT NULL DEFAULT 'manual',
            priority    INTEGER NOT NULL DEFAULT 1,
            status      TEXT NOT NULL DEFAULT 'pending',
            cooldown_until TEXT DEFAULT NULL,
            evidence_id TEXT DEFAULT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_retry_domain   ON retry_queue(domain);
        CREATE INDEX IF NOT EXISTS idx_retry_status   ON retry_queue(status);
        CREATE INDEX IF NOT EXISTS idx_retry_cooldown ON retry_queue(cooldown_until);
    """)
    conn.commit()


def _migrate_thought_runtime(conn: sqlite3.Connection) -> None:
    """Create hidden runtime/cache tables for learner-signal thoughts."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS thought_runtime_state (
            thought_id TEXT PRIMARY KEY,
            thought_path TEXT NOT NULL UNIQUE,
            raw_text_hash TEXT NOT NULL,
            domain_candidates TEXT NOT NULL DEFAULT '[]',
            goal_candidates TEXT NOT NULL DEFAULT '[]',
            embedding_status TEXT NOT NULL DEFAULT 'missing',
            embedding_updated_at TEXT DEFAULT NULL,
            weight_bucket_days INTEGER NOT NULL DEFAULT 0,
            time_weight REAL NOT NULL DEFAULT 1.0,
            next_weight_recompute_at TEXT DEFAULT NULL,
            last_clustered_at TEXT DEFAULT NULL,
            last_surfaced_at TEXT DEFAULT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_thought_runtime_path ON thought_runtime_state(thought_path);
        CREATE INDEX IF NOT EXISTS idx_thought_runtime_recompute ON thought_runtime_state(next_weight_recompute_at);

        CREATE TABLE IF NOT EXISTS thought_clusters (
            cluster_id TEXT PRIMARY KEY,
            scope_kind TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            theme_label TEXT NOT NULL,
            cluster_mass REAL NOT NULL DEFAULT 0.0,
            member_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_thought_clusters_scope ON thought_clusters(scope_kind, scope_key);

        CREATE TABLE IF NOT EXISTS thought_cluster_members (
            cluster_id TEXT NOT NULL,
            thought_id TEXT NOT NULL,
            similarity REAL NOT NULL DEFAULT 0.0,
            time_weight_snapshot REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (cluster_id, thought_id),
            FOREIGN KEY (cluster_id) REFERENCES thought_clusters(cluster_id)
        );
        CREATE INDEX IF NOT EXISTS idx_thought_cluster_members_thought ON thought_cluster_members(thought_id);
    """)
    conn.commit()


def _migrate_generated_names(conn: sqlite3.Connection) -> None:
    """Add generated-name payload columns for user-facing entities."""
    table_columns = {
        "goal_arcs": "generated_names_json",
        "tasks": "generated_names_json",
        "sessions": "generated_names_json",
        "tracks": "generated_names_json",
        "projects": "generated_names_json",
    }
    for table_name, column_name in table_columns.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name not in existing:
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} TEXT DEFAULT '{{}}'"
            )
    conn.commit()


def _migrate_entity_aliases(conn: sqlite3.Connection) -> None:
    """Create the visible-ref alias table."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entity_aliases (
            entity_kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            alias_kind TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (entity_kind, alias)
        );
        CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_kind, entity_id);
        """
    )
    conn.commit()


def _migrate_dispatch_phase10(conn: sqlite3.Connection) -> None:
    """Create dispatch subsystem tables (Phase 10, idempotent)."""
    conn.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS dispatch_sessions (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            context_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            judged INTEGER DEFAULT 0,
            judged_at TEXT DEFAULT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_dispatch_sessions_status
            ON dispatch_sessions(status);

        CREATE TABLE IF NOT EXISTS commitments (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            created_at TEXT NOT NULL,
            due_date TEXT DEFAULT NULL,
            status TEXT DEFAULT 'active',
            session_id TEXT DEFAULT NULL,
            FOREIGN KEY (session_id) REFERENCES dispatch_sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status);
        CREATE INDEX IF NOT EXISTS idx_commitments_due_date ON commitments(due_date);

        CREATE TABLE IF NOT EXISTS dispatch_agents (
            id TEXT PRIMARY KEY,
            domain TEXT NOT NULL,
            goal_id TEXT DEFAULT NULL,
            config_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            interaction_count INTEGER DEFAULT 0,
            FOREIGN KEY (goal_id) REFERENCES goal_arcs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_dispatch_agents_domain ON dispatch_agents(domain);
    """)
    conn.commit()


def _migrate_lesson_runtime(conn: sqlite3.Connection) -> None:
    """Create unified lesson runtime persistence tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lesson_runs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL,
            branch TEXT NOT NULL DEFAULT 'study',
            lesson_mode TEXT NOT NULL DEFAULT 'study',
            title TEXT NOT NULL DEFAULT '',
            lesson_status TEXT NOT NULL DEFAULT 'active',
            active_page_slug TEXT NOT NULL DEFAULT '',
            active_question_slug TEXT NOT NULL DEFAULT '',
            active_page_index INTEGER NOT NULL DEFAULT 0,
            active_question_index INTEGER NOT NULL DEFAULT 0,
            total_points REAL NOT NULL DEFAULT 0,
            ready_to_finish INTEGER NOT NULL DEFAULT 0,
            note_path TEXT NOT NULL DEFAULT '',
            retry_queue_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        );
        CREATE INDEX IF NOT EXISTS idx_lesson_runs_session
            ON lesson_runs(session_id);

        CREATE TABLE IF NOT EXISTS lesson_pages (
            id TEXT PRIMARY KEY,
            lesson_run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            page_slug TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            intro_text TEXT NOT NULL DEFAULT '',
            sequence_index INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            question_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(lesson_run_id, page_slug),
            FOREIGN KEY (lesson_run_id) REFERENCES lesson_runs(id),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_lesson_pages_run
            ON lesson_pages(lesson_run_id, sequence_index);

        CREATE TABLE IF NOT EXISTS lesson_questions (
            id TEXT PRIMARY KEY,
            lesson_run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            page_slug TEXT NOT NULL,
            question_slug TEXT NOT NULL,
            skill_slug TEXT NOT NULL DEFAULT '',
            question_type TEXT NOT NULL,
            prompt_json TEXT NOT NULL DEFAULT '{}',
            answer_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            sequence_index INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            hint_level INTEGER NOT NULL DEFAULT 0,
            revealed INTEGER NOT NULL DEFAULT 0,
            mastered INTEGER NOT NULL DEFAULT 0,
            queued_retry INTEGER NOT NULL DEFAULT 0,
            retry_of_question_slug TEXT NOT NULL DEFAULT '',
            retry_generation INTEGER NOT NULL DEFAULT 0,
            next_review_at TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(lesson_run_id, question_slug),
            FOREIGN KEY (lesson_run_id) REFERENCES lesson_runs(id),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_lesson_questions_page
            ON lesson_questions(lesson_run_id, page_slug, sequence_index);
        CREATE INDEX IF NOT EXISTS idx_lesson_questions_skill
            ON lesson_questions(lesson_run_id, skill_slug);

        CREATE TABLE IF NOT EXISTS lesson_attempts (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL,
            lesson_run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            page_slug TEXT NOT NULL,
            question_slug TEXT NOT NULL,
            skill_slug TEXT NOT NULL DEFAULT '',
            answer_text TEXT NOT NULL DEFAULT '',
            result TEXT NOT NULL,
            response_ms INTEGER NOT NULL DEFAULT 0,
            hint_level INTEGER NOT NULL DEFAULT 0,
            points_delta REAL NOT NULL DEFAULT 0,
            error_tags_json TEXT NOT NULL DEFAULT '[]',
            evaluator_confidence REAL DEFAULT NULL,
            model_used TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(lesson_run_id, question_slug, id),
            FOREIGN KEY (lesson_run_id) REFERENCES lesson_runs(id),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_lesson_attempts_question
            ON lesson_attempts(lesson_run_id, question_slug, row_id);
        CREATE INDEX IF NOT EXISTS idx_lesson_attempts_skill
            ON lesson_attempts(lesson_run_id, skill_slug, row_id);

        CREATE TABLE IF NOT EXISTS lesson_skill_states (
            id TEXT PRIMARY KEY,
            lesson_run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            skill_slug TEXT NOT NULL,
            recognition_status TEXT NOT NULL DEFAULT 'fragile',
            production_status TEXT NOT NULL DEFAULT 'fragile',
            overall_status TEXT NOT NULL DEFAULT 'fragile',
            error_tags_json TEXT NOT NULL DEFAULT '[]',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_review_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            UNIQUE(lesson_run_id, skill_slug),
            FOREIGN KEY (lesson_run_id) REFERENCES lesson_runs(id),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_lesson_skill_states_run
            ON lesson_skill_states(lesson_run_id, skill_slug);
        """
    )
    conn.commit()


def _migrate_concept_confidence(conn: sqlite3.Connection) -> None:
    """Add concept_confidence table (Phase 16, D-16-17, idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS concept_confidence (
            concept_id        TEXT PRIMARY KEY,
            confidence_score  REAL NOT NULL DEFAULT 0.0,
            card_weight       REAL NOT NULL DEFAULT 1.0,
            next_review_at    TEXT NOT NULL DEFAULT '',
            last_evidence_at  TEXT NOT NULL DEFAULT '',
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_concept_confidence_score
            ON concept_confidence(confidence_score);
        CREATE INDEX IF NOT EXISTS idx_concept_confidence_review
            ON concept_confidence(next_review_at);
    """)
    conn.commit()


def _migrate_concept_confidence_burst(conn: sqlite3.Connection) -> None:
    """Add burst_active and burst_streak columns to concept_confidence (Phase 16, D-16-27, idempotent)."""
    try:
        conn.execute(
            "ALTER TABLE concept_confidence ADD COLUMN burst_active INTEGER NOT NULL DEFAULT 0"
        )
    except Exception:
        pass  # column already exists
    try:
        conn.execute(
            "ALTER TABLE concept_confidence ADD COLUMN burst_streak INTEGER NOT NULL DEFAULT 0"
        )
    except Exception:
        pass  # column already exists
    conn.commit()


def _migrate_lesson_question_metadata(conn: sqlite3.Connection) -> None:
    """Add metadata_json to lesson_questions when upgrading older runtimes."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(lesson_questions)").fetchall()}
    if "metadata_json" not in existing:
        conn.execute("ALTER TABLE lesson_questions ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
        conn.commit()


def _migrate_context_runtime(conn: sqlite3.Connection) -> None:
    """Create durable context source, bundle, lock, and capability tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS context_sources (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            stored_path TEXT NOT NULL DEFAULT '',
            normalized_path TEXT NOT NULL DEFAULT '',
            mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
            canonical_class TEXT NOT NULL DEFAULT 'unknown',
            source_utility TEXT NOT NULL DEFAULT 'unknown',
            scope_mode TEXT NOT NULL DEFAULT 'unclear',
            domain_id TEXT DEFAULT NULL,
            domain_name TEXT DEFAULT NULL,
            scope_boundary TEXT NOT NULL DEFAULT '',
            source_ref TEXT NOT NULL UNIQUE,
            ingest_result_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_context_sources_domain
            ON context_sources(domain_name);
        CREATE INDEX IF NOT EXISTS idx_context_sources_utility
            ON context_sources(source_utility);

        CREATE TABLE IF NOT EXISTS source_bundles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            domain_id TEXT DEFAULT NULL,
            domain_name TEXT DEFAULT NULL,
            scope_mode TEXT NOT NULL DEFAULT 'unclear',
            scope_boundary TEXT NOT NULL DEFAULT '',
            source_refs_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_source_bundles_name
            ON source_bundles(name);

        CREATE TABLE IF NOT EXISTS source_bundle_items (
            id TEXT PRIMARY KEY,
            bundle_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            position INTEGER NOT NULL DEFAULT 0,
            source_ref TEXT NOT NULL DEFAULT '',
            filename TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(bundle_id, source_id),
            FOREIGN KEY (bundle_id) REFERENCES source_bundles(id),
            FOREIGN KEY (source_id) REFERENCES context_sources(id)
        );
        CREATE INDEX IF NOT EXISTS idx_source_bundle_items_bundle
            ON source_bundle_items(bundle_id, position);

        CREATE TABLE IF NOT EXISTS context_lock_state (
            scope_key TEXT PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'none',
            locked INTEGER NOT NULL DEFAULT 0,
            label TEXT NOT NULL DEFAULT '',
            label_max_chars INTEGER NOT NULL DEFAULT 20,
            scope_mode TEXT NOT NULL DEFAULT 'unclear',
            source_bundle_id TEXT DEFAULT NULL,
            source_refs_json TEXT NOT NULL DEFAULT '[]',
            domain_id TEXT DEFAULT NULL,
            scope_boundary TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS provider_capability_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            delivery TEXT NOT NULL,
            canonical_class TEXT NOT NULL,
            exact_mimes_json TEXT NOT NULL DEFAULT '[]',
            exact_extensions_json TEXT NOT NULL DEFAULT '[]',
            max_file_size_mb REAL DEFAULT NULL,
            documented_support INTEGER NOT NULL DEFAULT 0,
            probe_status TEXT NOT NULL DEFAULT 'unknown',
            support_mode TEXT NOT NULL DEFAULT 'unsupported',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(provider, model, endpoint, delivery, canonical_class)
        );

        CREATE TABLE IF NOT EXISTS provider_capability_probes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            canonical_class TEXT NOT NULL,
            probe_status TEXT NOT NULL DEFAULT 'unknown',
            detail TEXT NOT NULL DEFAULT '',
            checked_at TEXT NOT NULL,
            UNIQUE(provider, model, canonical_class)
        );
        """
    )
    conn.commit()


def init_db(path: Optional[Path] = None) -> None:
    """
    Initialize the database with schema.

    Args:
        path: Optional custom path. Defaults to the active vault data dir.
    """
    explicit_path = path is not None
    if explicit_path:
        set_db_path(path)

    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        _migrate_sessions_phase8(conn)
        _migrate_time_blocks_recurrence(conn)
        _migrate_goal_arc_focus(conn)
        _migrate_sessions_goal_practise(conn)
        _migrate_reminder_queue(conn)
        _migrate_generation_provenance(conn)
        _migrate_tasks_priority(conn)
        _migrate_daily_debriefs(conn)
        _migrate_pause_intervals(conn)
        _migrate_usage_log(conn)
        _migrate_task_skills(conn)
        _migrate_task_completion(conn)
        _migrate_interactions(conn)      # Phase 17
        _migrate_product_control(conn)
        _migrate_anki_cards(conn)        # Phase 18
        _migrate_anki_revlog(conn)       # Phase 18
        _migrate_domain_weekly_stats(conn)  # Phase 19
        _migrate_sessions_phase23(conn)     # Phase 23
        _migrate_anki_cards_phase26(conn)   # Phase 26 — D-12
        _migrate_generation_run_log(conn)   # Phase 26 — D-13
        _migrate_anki_card_statuses(conn)
        _migrate_metric_tables(conn)
        _migrate_vault_note_index(conn)
        _migrate_thought_runtime(conn)
        _migrate_evidence_phase2(conn)   # Phase 2 -- evidence system
        _migrate_generated_names(conn)
        _migrate_entity_aliases(conn)
        _migrate_dispatch_phase10(conn)  # Phase 10: dispatch sessions, commitments, agents
        _migrate_agent_weights(conn)     # Phase 12: agent frecency
        _migrate_agent_instruction_judge(conn)  # Phase 13: self-improvement judge
        _migrate_lesson_runtime(conn)
        _migrate_lesson_question_metadata(conn)
        _migrate_context_runtime(conn)
        _migrate_concept_confidence(conn)        # Phase 16 — concept confidence substrate (D-16-17)
        _migrate_concept_confidence_burst(conn)  # Phase 16 — drill-burst recovery columns (D-16-27)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Get a database connection as a context manager.

    Yields:
        SQLite connection with row factory set
    """
    db_path = get_db_path()
    if not db_path.exists():
        init_db()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def log_usage(command: str, exit_code: int, error_first_line: str = "") -> None:
    """Write one usage log entry and prune entries older than 30 days (D-02, D-04).

    Non-fatal: silently swallows all exceptions so usage logging
    never breaks the command being executed.
    """
    now = _dt.datetime.utcnow().isoformat()
    cutoff = (_dt.datetime.utcnow() - _dt.timedelta(days=30)).isoformat()
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO usage_log (timestamp, command, exit_code, error) VALUES (?, ?, ?, ?)",
                (now, command, exit_code, error_first_line),
            )
            conn.execute("DELETE FROM usage_log WHERE timestamp < ?", (cutoff,))
            conn.commit()
    except Exception:
        pass  # Non-fatal: usage logging must never break a command


def get_command_counts(conn: sqlite3.Connection, days: int = 7) -> dict[str, int]:
    """Return command usage counts for the past N days (D-05).

    Args:
        conn: Active SQLite connection (caller manages lifecycle).
        days: Number of days to look back (default 7).

    Returns:
        Dict mapping command name to count, ordered by count descending.
    """
    cutoff = (_dt.datetime.utcnow() - _dt.timedelta(days=days)).isoformat()
    try:
        rows = conn.execute(
            "SELECT command, COUNT(*) as cnt FROM usage_log "
            "WHERE timestamp >= ? GROUP BY command ORDER BY cnt DESC",
            (cutoff,),
        ).fetchall()
        return {row["command"]: row["cnt"] for row in rows}
    except Exception:
        return {}
