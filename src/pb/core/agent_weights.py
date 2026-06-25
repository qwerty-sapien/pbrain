# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Deterministic agent weight and frecency scoring for dispatch routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

from pb.core.models import generate_internal_id
from pb.storage.database import get_connection

HALF_LIFE_DAYS = 7.0
PIN_OVERRIDE = "pin"
SUPPRESS_OVERRIDE = "suppress"
BUILTIN_DOMAIN_PREFIX = "builtin:"

EVENT_BASE_WEIGHTS: dict[str, float] = {
    "dispatch_selected": 0.2,
    "session_continued": 0.6,
    "session_completed": 1.0,
    "resume_selected": 0.8,
    "commitment_followup_selected": 0.7,
    "session_paused": -0.2,
    "kickback": -0.5,
    "wrong": -1.0,
}

SOURCE_MULTIPLIERS: dict[str, float] = {
    "human": 1.0,
    "agent": 0.5,
}

WINDOW_DAYS = {
    "short": 7,
    "medium": 30,
    "long": 90,
}


@dataclass(frozen=True)
class AgentWeightSnapshot:
    """Persisted or computed weight snapshot for one agent."""

    agent_id: str
    frecency_score: float = 0.0
    dispatch_prior: float = 0.5
    short_frecency: float = 0.0
    medium_frecency: float = 0.0
    long_frecency: float = 0.0
    updated_at: str = ""
    override: str = ""


def _utc_now() -> datetime:
    return datetime.utcnow()


def _storage_domain_for_agent(agent_id: str) -> str:
    normalized = (agent_id or "").strip()
    if normalized.startswith("domain_"):
        return normalized[len("domain_"):]
    return f"{BUILTIN_DOMAIN_PREFIX}{normalized}"


def _agent_id_for_storage_domain(domain: str) -> str:
    normalized = (domain or "").strip()
    if normalized.startswith(BUILTIN_DOMAIN_PREFIX):
        return normalized[len(BUILTIN_DOMAIN_PREFIX) :]
    return f"domain_{normalized}"


def _normalize_override(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {PIN_OVERRIDE, SUPPRESS_OVERRIDE}:
        return normalized
    return ""


def _parse_config_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dump_config_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _override_from_config(raw: str | None) -> str:
    payload = _parse_config_json(raw)
    return _normalize_override(payload.get("weight_override"))


def _maybe_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _decayed_contribution(base_weight: float, source_kind: str, event_time: datetime, now: datetime) -> float:
    age_seconds = max(0.0, (now - event_time).total_seconds())
    age_days = age_seconds / 86400.0
    decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
    multiplier = SOURCE_MULTIPLIERS.get(source_kind, SOURCE_MULTIPLIERS["human"])
    return base_weight * multiplier * decay


def _read_cache_row(conn, agent_id: str) -> Optional[AgentWeightSnapshot]:
    row = conn.execute(
        """
        SELECT agent_id, frecency_score, dispatch_prior, short_frecency,
               medium_frecency, long_frecency, updated_at
        FROM agent_frecency_scores
        WHERE agent_id = ?
        """,
        (agent_id,),
    ).fetchone()
    if row is None:
        row = conn.execute(
            """
            SELECT agent_id, frecency_score, dispatch_prior, short_frecency,
                   medium_frecency, long_frecency, updated_at
            FROM agent_weight_cache
            WHERE agent_id = ?
            """,
            (agent_id,),
        ).fetchone()
    if row is None:
        return None
    override = get_weight_override(agent_id, conn=conn)
    return AgentWeightSnapshot(
        agent_id=row["agent_id"],
        frecency_score=float(row["frecency_score"] or 0.0),
        dispatch_prior=float(row["dispatch_prior"] or 0.5),
        short_frecency=float(row["short_frecency"] or 0.0),
        medium_frecency=float(row["medium_frecency"] or 0.0),
        long_frecency=float(row["long_frecency"] or 0.0),
        updated_at=row["updated_at"] or "",
        override=override,
    )


def _agent_has_events(conn, agent_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM agent_weight_events WHERE agent_id = ? LIMIT 1",
        (agent_id,),
    ).fetchone()
    return row is not None


def _upsert_cache_row(conn, snapshot: AgentWeightSnapshot) -> None:
    params = (
        snapshot.agent_id,
        snapshot.frecency_score,
        snapshot.dispatch_prior,
        snapshot.short_frecency,
        snapshot.medium_frecency,
        snapshot.long_frecency,
        snapshot.updated_at,
    )
    for table_name in ("agent_frecency_scores", "agent_weight_cache"):
        conn.execute(
            f"""
            INSERT INTO {table_name} (
                agent_id, frecency_score, dispatch_prior, short_frecency,
                medium_frecency, long_frecency, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                frecency_score = excluded.frecency_score,
                dispatch_prior = excluded.dispatch_prior,
                short_frecency = excluded.short_frecency,
                medium_frecency = excluded.medium_frecency,
                long_frecency = excluded.long_frecency,
                updated_at = excluded.updated_at
            """,
            params,
        )


def refresh_agent_weight_cache(agent_id: str, *, conn=None) -> AgentWeightSnapshot:
    """Recompute and persist the cache row for one agent."""
    owns_connection = conn is None
    if conn is None:
        context = get_connection()
        conn = context.__enter__()
    else:
        context = None

    try:
        rows = conn.execute(
            """
            SELECT source_kind, base_weight, created_at
            FROM agent_weight_events
            WHERE agent_id = ?
            ORDER BY created_at ASC
            """,
            (agent_id,),
        ).fetchall()
        now = _utc_now()
        short_frecency = 0.0
        medium_frecency = 0.0
        long_frecency = 0.0
        positive_events = 0
        negative_events = 0

        for row in rows:
            event_time = _maybe_datetime(row["created_at"])
            if event_time is None:
                continue
            base_weight = float(row["base_weight"] or 0.0)
            source_kind = (row["source_kind"] or "human").strip().lower()
            age_days = max(0.0, (now - event_time).total_seconds()) / 86400.0
            contribution = _decayed_contribution(base_weight, source_kind, event_time, now)

            if age_days <= WINDOW_DAYS["short"]:
                short_frecency += contribution
            if age_days <= WINDOW_DAYS["medium"]:
                medium_frecency += contribution
            if age_days <= WINDOW_DAYS["long"]:
                long_frecency += contribution
            if base_weight > 0:
                positive_events += 1
            elif base_weight < 0:
                negative_events += 1

        dispatch_prior = (positive_events + 1) / (positive_events + negative_events + 2)
        frecency_score = (
            0.30 * short_frecency
            + 0.45 * medium_frecency
            + 0.15 * long_frecency
            + 0.10 * dispatch_prior
        )
        snapshot = AgentWeightSnapshot(
            agent_id=agent_id,
            frecency_score=frecency_score,
            dispatch_prior=dispatch_prior,
            short_frecency=short_frecency,
            medium_frecency=medium_frecency,
            long_frecency=long_frecency,
            updated_at=now.isoformat(),
            override=get_weight_override(agent_id, conn=conn),
        )
        _upsert_cache_row(conn, snapshot)
        if owns_connection:
            conn.commit()
        return snapshot
    finally:
        if context is not None:
            context.__exit__(None, None, None)


def record_agent_weight_event(
    agent_id: str,
    event_kind: str,
    *,
    source_kind: str = "human",
    session_id: str = "",
    metadata: Optional[dict[str, Any]] = None,
    created_at: Optional[str] = None,
    base_weight: Optional[float] = None,
) -> AgentWeightSnapshot:
    """Persist one weight event and refresh the agent cache immediately."""
    normalized_event = (event_kind or "").strip().lower()
    normalized_source = (source_kind or "human").strip().lower()
    if normalized_source not in SOURCE_MULTIPLIERS:
        raise ValueError(f"Unsupported source_kind: {source_kind}")
    if base_weight is None:
        if normalized_event not in EVENT_BASE_WEIGHTS:
            raise ValueError(f"Unsupported event_kind: {event_kind}")
        resolved_base_weight = EVENT_BASE_WEIGHTS[normalized_event]
    else:
        resolved_base_weight = float(base_weight)

    payload = json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True)
    event_id = generate_internal_id()
    event_timestamp = created_at or _utc_now().isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_weight_events (
                id, agent_id, session_id, event_kind, source_kind,
                base_weight, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                agent_id,
                session_id or "",
                normalized_event,
                normalized_source,
                resolved_base_weight,
                event_timestamp,
                payload,
            ),
        )
        snapshot = refresh_agent_weight_cache(agent_id, conn=conn)
        conn.commit()
        return snapshot


def get_agent_weight_snapshot(agent_id: str) -> AgentWeightSnapshot:
    """Return the cached snapshot, recomputing on demand when needed."""
    with get_connection() as conn:
        snapshot = _read_cache_row(conn, agent_id)
        if snapshot is not None:
            return snapshot
        if _agent_has_events(conn, agent_id):
            snapshot = refresh_agent_weight_cache(agent_id, conn=conn)
            conn.commit()
            return snapshot
        return AgentWeightSnapshot(
            agent_id=agent_id,
            override=get_weight_override(agent_id, conn=conn),
        )


def get_weight_override(agent_id: str, *, conn=None) -> str:
    """Return the stored manual weight override, if any."""
    storage_domain = _storage_domain_for_agent(agent_id)
    owns_connection = conn is None
    if conn is None:
        context = get_connection()
        conn = context.__enter__()
    else:
        context = None

    try:
        row = conn.execute(
            "SELECT config_json FROM dispatch_agents WHERE domain = ?",
            (storage_domain,),
        ).fetchone()
        if row is None:
            return ""
        return _override_from_config(row["config_json"])
    finally:
        if context is not None:
            context.__exit__(None, None, None)


def set_weight_override(agent_id: str, override: str | None) -> str:
    """Persist a manual override in dispatch_agents.config_json."""
    normalized_override = _normalize_override(override)
    storage_domain = _storage_domain_for_agent(agent_id)
    now = _utc_now().isoformat()

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, config_json, goal_id, interaction_count
            FROM dispatch_agents
            WHERE domain = ?
            """,
            (storage_domain,),
        ).fetchone()
        if row is None:
            config_payload: dict[str, Any] = {}
            if normalized_override:
                config_payload["weight_override"] = normalized_override
            conn.execute(
                """
                INSERT INTO dispatch_agents (
                    id, domain, goal_id, config_json, created_at, interaction_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    generate_internal_id(),
                    storage_domain,
                    None,
                    _dump_config_json(config_payload),
                    now,
                    0,
                ),
            )
        else:
            config_payload = _parse_config_json(row["config_json"])
            if normalized_override:
                config_payload["weight_override"] = normalized_override
            else:
                config_payload.pop("weight_override", None)
            conn.execute(
                "UPDATE dispatch_agents SET config_json = ? WHERE domain = ?",
                (_dump_config_json(config_payload), storage_domain),
            )
        conn.commit()
    return normalized_override


def get_effective_weight(agent_id: str) -> float:
    """Return the effective ordering weight with manual overrides applied."""
    snapshot = get_agent_weight_snapshot(agent_id)
    if snapshot.override == PIN_OVERRIDE:
        return 1_000_000_000.0
    if snapshot.override == SUPPRESS_OVERRIDE:
        return -1_000_000_000.0
    return snapshot.frecency_score


def normalized_effective_weights(agent_ids: Iterable[str]) -> dict[str, float]:
    """Return 0..1 normalized values while preserving pin/raw/suppress ordering."""
    normalized_ids = list(dict.fromkeys(agent_id for agent_id in agent_ids if agent_id))
    if not normalized_ids:
        return {}

    snapshots = {agent_id: get_agent_weight_snapshot(agent_id) for agent_id in normalized_ids}
    pinned = [agent_id for agent_id, snapshot in snapshots.items() if snapshot.override == PIN_OVERRIDE]
    suppressed = [agent_id for agent_id, snapshot in snapshots.items() if snapshot.override == SUPPRESS_OVERRIDE]
    normal = [
        agent_id for agent_id, snapshot in snapshots.items()
        if snapshot.override not in {PIN_OVERRIDE, SUPPRESS_OVERRIDE}
    ]

    result: dict[str, float] = {}
    for agent_id in pinned:
        result[agent_id] = 1.0
    for agent_id in suppressed:
        result[agent_id] = 0.0

    if not normal:
        return result

    raw_values = [snapshots[agent_id].frecency_score for agent_id in normal]
    min_raw = min(raw_values)
    max_raw = max(raw_values)
    if abs(max_raw - min_raw) < 1e-9:
        for agent_id in normal:
            result[agent_id] = 0.5
        return result

    for agent_id in normal:
        raw = snapshots[agent_id].frecency_score
        scaled = (raw - min_raw) / (max_raw - min_raw)
        result[agent_id] = 0.25 + (0.50 * scaled)
    return result


def sort_sessions_by_weight(sessions: list) -> list:
    """Sort sessions by effective agent weight, then most recent update."""
    return sorted(
        sessions,
        key=lambda session: (
            -get_effective_weight(session.agent_id),
            -getattr(session.updated_at, "timestamp", lambda: 0.0)(),
        ),
    )


def choose_lowest_weight_session(sessions: Iterable) -> Optional[Any]:
    """Return the lowest-weight session, breaking ties by oldest update."""
    candidates = list(sessions)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda session: (
            get_effective_weight(session.agent_id),
            getattr(session.updated_at, "timestamp", lambda: 0.0)(),
            getattr(session.created_at, "timestamp", lambda: 0.0)(),
        ),
    )


def sort_commitments_for_next(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort commitment rows by due date, then linked agent weight descending."""
    def _due_key(value: str | None) -> tuple[int, datetime]:
        parsed = _maybe_datetime(value)
        if parsed is None:
            return (1, datetime.max)
        return (0, parsed)

    return sorted(
        list(rows),
        key=lambda row: (
            _due_key(row.get("due_date")),
            -get_effective_weight(str(row.get("agent_id") or "")),
        ),
    )


def list_agent_weights() -> list[dict[str, Any]]:
    """Return agent-weight diagnostics for config inspection."""
    rows: list[dict[str, Any]] = []
    with get_connection() as conn:
        cache_rows = conn.execute(
            """
            SELECT agent_id, frecency_score, dispatch_prior, short_frecency,
                   medium_frecency, long_frecency, updated_at
            FROM agent_frecency_scores
            ORDER BY frecency_score DESC, agent_id ASC
            """
        ).fetchall()
        overrides = {
            _agent_id_for_storage_domain(row["domain"]): _override_from_config(row["config_json"])
            for row in conn.execute(
                "SELECT domain, config_json FROM dispatch_agents"
            ).fetchall()
        }
        known_ids = set(overrides)
        known_ids.update(row["agent_id"] for row in cache_rows)
        try:
            from pb.agents import list_agents

            known_ids.update(list_agents())
        except Exception:
            pass

        cached_by_id = {
            row["agent_id"]: dict(row)
            for row in cache_rows
        }
        for agent_id in sorted(
            known_ids,
            key=lambda candidate: (
                -(float(cached_by_id.get(candidate, {}).get("frecency_score", 0.0) or 0.0)),
                candidate,
            ),
        ):
            cached = cached_by_id.get(agent_id)
            rows.append(
                {
                    "agent_id": agent_id,
                    "frecency_score": float(cached["frecency_score"] or 0.0) if cached else 0.0,
                    "dispatch_prior": float(cached["dispatch_prior"] or 0.5) if cached else 0.5,
                    "short_frecency": float(cached["short_frecency"] or 0.0) if cached else 0.0,
                    "medium_frecency": float(cached["medium_frecency"] or 0.0) if cached else 0.0,
                    "long_frecency": float(cached["long_frecency"] or 0.0) if cached else 0.0,
                    "override": overrides.get(agent_id, ""),
                    "updated_at": cached["updated_at"] if cached else "",
                }
            )
    return rows


def scorer_health() -> dict[str, Any]:
    """Return non-blocking health diagnostics for the scorer tables."""
    with get_connection() as conn:
        existing_tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        required_tables = {"agent_weight_events", "agent_weight_cache", "agent_frecency_scores"}
        tables_present = required_tables.issubset(existing_tables)
        if not tables_present:
            return {
                "tables_present": False,
                "cache_readable": False,
                "event_rows": 0,
                "cache_rows": 0,
                "missing_cache_agents": 0,
                "stale_cache_agents": 0,
                "anomalies": ["missing_tables"],
            }

        event_rows = int(
            conn.execute("SELECT COUNT(*) FROM agent_weight_events").fetchone()[0]
        )
        cache_rows = int(
            conn.execute("SELECT COUNT(*) FROM agent_frecency_scores").fetchone()[0]
        )
        missing_cache_agents = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT e.agent_id
                    FROM agent_weight_events e
                    LEFT JOIN agent_frecency_scores c ON c.agent_id = e.agent_id
                    WHERE c.agent_id IS NULL
                )
                """
            ).fetchone()[0]
        )
        stale_cache_agents = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT e.agent_id
                    FROM agent_weight_events e
                    JOIN agent_frecency_scores c ON c.agent_id = e.agent_id
                    GROUP BY e.agent_id
                    HAVING MAX(e.created_at) > c.updated_at
                )
                """
            ).fetchone()[0]
        )
        anomalies: list[str] = []
        if missing_cache_agents:
            anomalies.append("missing_cache_rows")
        if stale_cache_agents:
            anomalies.append("stale_cache_rows")
        return {
            "tables_present": True,
            "cache_readable": True,
            "event_rows": event_rows,
            "cache_rows": cache_rows,
            "missing_cache_agents": missing_cache_agents,
            "stale_cache_agents": stale_cache_agents,
            "anomalies": anomalies,
        }
