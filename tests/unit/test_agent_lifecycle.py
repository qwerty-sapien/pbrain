from __future__ import annotations

import json
from datetime import datetime

from pb.core.agent_lifecycle import AgentLifecycleSuggester, dispatch_agent_disabled
from pb.core.models import Session, Task
from pb.core.refinement_memory import record_refinement_memory, refinement_memory_prompt_suffix
from pb.storage.database import get_connection


def test_agent_lifecycle_spawn_forget_and_respawn_reuses_archived_agent(repo):
    task = repo.create_task(Task(title="Study music theory"))
    session = repo.create_session(
        Session(
            task_id=task.id,
            branch="study",
            subject_scope="Music Theory harmony",
            start_at=datetime.utcnow(),
        )
    )
    lifecycle = AgentLifecycleSuggester(repo)

    spawned = lifecycle.spawn_or_respawn(session=session)
    assert spawned.action == "spawn"
    assert spawned.domain == "music_theory"
    assert dispatch_agent_disabled("music_theory") is False

    forgotten = lifecycle.forget(session=session)
    assert forgotten.action == "forget"
    assert dispatch_agent_disabled("music_theory") is True

    revived = lifecycle.spawn_or_respawn(session=session)
    assert revived.action == "respawn"
    assert revived.reused_archived is True
    assert revived.agent_id == spawned.agent_id
    assert dispatch_agent_disabled("music_theory") is False

    with get_connection() as conn:
        row = conn.execute("SELECT config_json FROM dispatch_agents WHERE domain = ?", ("music_theory",)).fetchone()
    config = json.loads(row["config_json"])
    assert config["disabled"] is False
    assert config["subject_scope"] == "Music Theory harmony"


def test_general_refinement_promotes_to_matching_plan_niche(repo):
    record_refinement_memory(
        repo,
        surface="plan",
        topic="causal graph learning",
        refinement="Use formal SCM notation and keep equations in LaTeX.",
        prefer_niche=False,
    )

    suffix = refinement_memory_prompt_suffix(
        surface="plan",
        topic="causal graph learning intervention policy",
    )

    assert "high-weight soft guidance" in suffix
    assert "Use formal SCM notation" in suffix
    with get_connection() as conn:
        general_row = conn.execute(
            "SELECT config_json FROM dispatch_agents WHERE domain = ?",
            ("user_refinement:general",),
        ).fetchone()
    general_config = json.loads(general_row["config_json"])
    assert general_config["refinements"] == []
