# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pathlib import Path

from pb.core.entity_refs import display_ref, is_uuid_like
from pb.core.learning_partner import load_session_transcript, save_session_transcript
from pb.domain.models import GoalArc, Session, Task
from pb.storage.database import get_connection


def test_repository_persists_short_visible_refs_for_task_goal_and_session(repo):
    task = repo.create_task(Task(title="Konj 2 würde"))
    goal = repo.create_goal_arc(GoalArc(title="German Konjunktiv II"))
    session = repo.create_session(Session(task_id=task.id, subject_scope="Konjunktiv II"))

    assert display_ref(task, "task") == task.generated_names["slug"]
    assert display_ref(goal, "goal") == goal.generated_names["slug"]
    assert display_ref(session, "session") == session.generated_names["session_slug"]

    assert len(task.generated_names["slug"]) < 27
    assert len(goal.generated_names["slug"]) < 27
    assert len(session.generated_names["session_slug"]) < 27
    assert not is_uuid_like(task.generated_names["slug"])
    assert not is_uuid_like(goal.generated_names["slug"])


def test_repository_resolves_visible_refs_and_legacy_ids(repo):
    task = repo.create_task(Task(title="Wenn clauses"))
    goal = repo.create_goal_arc(GoalArc(title="German grammar"))
    session = repo.create_session(Session(task_id=task.id, subject_scope="wenn"))

    assert repo.resolve_task_ref(task.generated_names["slug"]).id == task.id
    assert repo.resolve_goal_ref(goal.generated_names["slug"]).id == goal.id
    assert repo.resolve_session_ref(session.generated_names["session_slug"]).id == session.id

    assert repo.resolve_goal_ref(goal.id).id == goal.id
    assert repo.resolve_session_ref(session.id).id == session.id

    with get_connection() as conn:
        alias_kinds = {
            row["alias_kind"]
            for row in conn.execute(
                "SELECT alias_kind FROM entity_aliases WHERE entity_kind = 'session' AND entity_id = ?",
                (session.id,),
            ).fetchall()
        }

    assert {"visible_ref", "legacy_uuid"} <= alias_kinds


def test_task_visible_ref_collisions_use_short_numeric_suffixes(repo):
    first = repo.create_task(Task(title="Konj 2"))
    second = repo.create_task(Task(title="Konj 2"))

    assert first.generated_names["slug"] == "konj_2"
    assert second.generated_names["slug"] == "konj_22"
    assert len(second.generated_names["slug"]) < 27


def test_learning_partner_transcripts_prefer_session_slug(tmp_path):
    transcript = [{"role": "assistant", "content": "Try one more rep."}]
    data_dir = Path(tmp_path)

    save_session_transcript(data_dir, "123e4567-e89b-12d3-a456-426614174000", transcript, "sess_konj2")

    slug_path = data_dir / "transcripts" / "sess_konj2.json"
    legacy_path = data_dir / "transcripts" / "123e4567-e89b-12d3-a456-426614174000.json"

    assert slug_path.exists()
    assert legacy_path.exists()
    assert load_session_transcript(
        data_dir,
        "123e4567-e89b-12d3-a456-426614174000",
        "sess_konj2",
    ) == transcript
    assert load_session_transcript(
        data_dir,
        "123e4567-e89b-12d3-a456-426614174000",
    ) == transcript
