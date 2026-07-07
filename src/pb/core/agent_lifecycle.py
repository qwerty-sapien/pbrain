# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Intentional spawn/respawn/forget lifecycle for domain agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from pb.core.interest_hierarchy import classify_interest_label
from pb.core.models import generate_internal_id, utc_now
from pb.storage.database import get_connection


def _slug(text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return clean or "general_learning"


def _load_config(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _dump_config(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


@dataclass(frozen=True)
class AgentLifecycleResult:
    action: str
    domain: str
    agent_id: str = ""
    message: str = ""
    reused_archived: bool = False


class AgentLifecycleSuggester:
    """Create, revive, and archive domain agents with explicit user intent."""

    def __init__(self, repo: Any):
        self.repo = repo

    def domain_from_session(self, session: Any) -> str:
        scope = str(getattr(session, "subject_scope", "") or "").strip()
        goal = self._session_goal(session)
        track = self._session_track(session)
        basis = scope or getattr(goal, "domain", "") or getattr(goal, "title", "") or getattr(track, "name", "")
        _top, parent, leaf = classify_interest_label(basis)
        label = leaf if leaf and leaf != "General Learning" else parent or basis
        return _slug(label)

    def spawn_or_respawn(self, *, session: Any, force: bool = True) -> AgentLifecycleResult:
        domain = self.domain_from_session(session)
        if not domain:
            return AgentLifecycleResult("none", "", message="No clear domain for an agent yet.")
        goal = self._session_goal(session)
        track = self._session_track(session)
        now = utc_now().isoformat()
        context = {
            "lifecycle_status": "active",
            "disabled": False,
            "spawned_at": now,
            "last_spawned_at": now,
            "spawn_source": "/spawn" if force else "suggestion",
            "subject_scope": getattr(session, "subject_scope", "") or "",
            "branch": getattr(session, "branch", "") or "",
            "track_id": getattr(track, "id", "") if track is not None else "",
        }
        try:
            generated = dict(getattr(session, "generated_names", {}) or {})
            active_context = generated.get("active_context_scope")
            if isinstance(active_context, dict):
                context["active_context_scope"] = active_context
        except Exception:
            pass
        with get_connection() as conn:
            archived = conn.execute(
                "SELECT * FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
                (domain,),
            ).fetchone()
            if archived is not None:
                config = _load_config(archived["config_json"])
                was_archived = bool(config.get("disabled") or config.get("lifecycle_status") in {"archived", "disabled"})
                config.update(context)
                conn.execute(
                    "UPDATE dispatch_agents SET goal_id = ?, config_json = ?, interaction_count = ? WHERE id = ?",
                    (
                        getattr(goal, "id", None),
                        _dump_config(config),
                        max(2, int(archived["interaction_count"] or 0)),
                        archived["id"],
                    ),
                )
                conn.commit()
                return AgentLifecycleResult(
                    "respawn" if was_archived else "spawn",
                    domain,
                    agent_id=str(archived["id"]),
                    reused_archived=was_archived,
                    message=(
                        f"Respawned {domain} agent with this session's context."
                        if was_archived
                        else f"{domain} agent is active and updated with this session's context."
                    ),
                )
            agent_id = generate_internal_id()
            conn.execute(
                """
                INSERT INTO dispatch_agents (
                    id, domain, goal_id, config_json, created_at, interaction_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (agent_id, domain, getattr(goal, "id", None), _dump_config(context), now, 2),
            )
            conn.commit()
        return AgentLifecycleResult("spawn", domain, agent_id=agent_id, message=f"Spawned {domain} agent.")

    def forget(self, *, session: Any, reason: str = "user_requested") -> AgentLifecycleResult:
        domain = self.domain_from_session(session)
        if not domain:
            return AgentLifecycleResult("none", "", message="No clear domain agent to forget.")
        now = utc_now().isoformat()
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
                (domain,),
            ).fetchone()
            if row is None:
                return AgentLifecycleResult("none", domain, message=f"No {domain} agent exists yet.")
            config = _load_config(row["config_json"])
            config.update(
                {
                    "lifecycle_status": "archived",
                    "disabled": True,
                    "forgotten_at": now,
                    "forget_reason": reason,
                }
            )
            conn.execute("UPDATE dispatch_agents SET config_json = ? WHERE id = ?", (_dump_config(config), row["id"]))
            conn.commit()
        return AgentLifecycleResult("forget", domain, agent_id=str(row["id"]), message=f"Forgot {domain} agent for now.")

    def cooldown_hint(self, *, session: Any) -> str:
        domain = self.domain_from_session(session)
        if not domain:
            return ""
        if not self._cooldown_active(domain):
            return ""
        return f"A {domain} agent could improve this session. Use /spawn when ready."

    def record_skip(self, domain: str, *, days: int = 7) -> None:
        until = (utc_now() + timedelta(days=days)).isoformat()
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
                (domain,),
            ).fetchone()
            if row is None:
                return
            config = _load_config(row["config_json"])
            config["spawn_skip_cooldown_until"] = until
            conn.execute("UPDATE dispatch_agents SET config_json = ? WHERE id = ?", (_dump_config(config), row["id"]))
            conn.commit()

    def _cooldown_active(self, domain: str) -> bool:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT config_json FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
                (domain,),
            ).fetchone()
        if row is None:
            return False
        config = _load_config(row["config_json"])
        try:
            until = datetime.fromisoformat(str(config.get("spawn_skip_cooldown_until", "")))
        except Exception:
            return False
        return until > utc_now()

    def _session_goal(self, session: Any):
        goal_id = getattr(session, "goal_id", None)
        if goal_id:
            try:
                return self.repo.get_goal_arc(goal_id)
            except Exception:
                return None
        return None

    def _session_track(self, session: Any):
        track_id = getattr(session, "track_id", None)
        if track_id:
            try:
                return self.repo.get_track(track_id)
            except Exception:
                return None
        return None


def dispatch_agent_disabled(domain: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT config_json FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
            (domain,),
        ).fetchone()
    if row is None:
        return False
    config = _load_config(row["config_json"])
    return bool(config.get("disabled") or config.get("lifecycle_status") in {"archived", "disabled"})
