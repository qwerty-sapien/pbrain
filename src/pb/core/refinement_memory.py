# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Soft user-refinement memory for generated learning drafts."""

from __future__ import annotations

import json
import re
from typing import Any

from pb.core.models import generate_internal_id, utc_now
from pb.storage.database import get_connection

_GENERAL_DOMAIN = "user_refinement:general"
_MAX_REFINEMENTS = 12
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "from",
    "in",
    "it",
    "of",
    "on",
    "or",
    "plan",
    "study",
    "the",
    "this",
    "to",
    "with",
}


def _normalize_surface(surface: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_:-]+", "_", (surface or "").strip().lower()).strip("_")
    return cleaned or "general"


def _topic_tokens(topic: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (topic or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _topic_slug(topic: str) -> str:
    tokens = list(dict.fromkeys(re.findall(r"[a-z0-9]+", (topic or "").lower())))
    tokens = [token for token in tokens if token not in _STOPWORDS]
    return "_".join(tokens[:8]) or "general"


def _niche_domain(surface: str, topic: str) -> str:
    return f"user_refinement:{_normalize_surface(surface)}:{_topic_slug(topic)}"


def _load_config(raw: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dump_config(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _record(*, surface: str, topic: str, refinement: str, source: str) -> dict[str, str]:
    return {
        "created_at": utc_now().isoformat(),
        "surface": _normalize_surface(surface),
        "topic": " ".join((topic or "").split()),
        "refinement": " ".join((refinement or "").split()),
        "source": source,
    }


def _records_match(record: dict[str, Any], *, surface: str, topic: str) -> bool:
    record_surface = _normalize_surface(str(record.get("surface", "")))
    wanted_surface = _normalize_surface(surface)
    if record_surface not in {wanted_surface, "general"}:
        return False
    record_topic = str(record.get("topic", ""))
    if not record_topic or not topic:
        return False
    left = _topic_tokens(record_topic)
    right = _topic_tokens(topic)
    if not left or not right:
        return False
    left_text = " ".join(sorted(left))
    right_text = " ".join(sorted(right))
    if left_text in right_text or right_text in left_text:
        return True
    overlap = len(left & right)
    return overlap >= max(1, min(len(left), len(right)) // 2)


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        refinement = str(record.get("refinement", "")).strip()
        if not refinement:
            continue
        key = (
            _normalize_surface(str(record.get("surface", ""))),
            _topic_slug(str(record.get("topic", ""))),
            refinement.lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped[-_MAX_REFINEMENTS:]


def _upsert_domain_records(conn, *, domain: str, surface: str, topic: str, records: list[dict[str, Any]]) -> None:
    row = conn.execute(
        "SELECT id, config_json, interaction_count FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
        (domain,),
    ).fetchone()
    if row is None:
        payload: dict[str, Any] = {}
        row_id = generate_internal_id()
        interaction_count = 0
    else:
        payload = _load_config(row["config_json"])
        row_id = str(row["id"])
        interaction_count = int(row["interaction_count"] or 0)

    existing = payload.get("refinements")
    existing_records = existing if isinstance(existing, list) else []
    payload.update(
        {
            "type": "user_refinement_memory",
            "lifecycle_status": "active",
            "disabled": False,
            "surface": _normalize_surface(surface),
            "topic": " ".join((topic or "").split()),
            "updated_at": utc_now().isoformat(),
            "refinements": _dedupe_records([*existing_records, *records]),
        }
    )
    if row is None:
        conn.execute(
            """
            INSERT INTO dispatch_agents (
                id, domain, goal_id, config_json, created_at, interaction_count
            ) VALUES (?, ?, NULL, ?, ?, ?)
            """,
            (row_id, domain, _dump_config(payload), utc_now().isoformat(), max(1, len(records))),
        )
    else:
        conn.execute(
            "UPDATE dispatch_agents SET config_json = ?, interaction_count = ? WHERE id = ?",
            (_dump_config(payload), interaction_count + max(1, len(records)), row_id),
        )


def _extract_matching_general_records(conn, *, surface: str, topic: str) -> list[dict[str, Any]]:
    row = conn.execute(
        "SELECT id, config_json FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
        (_GENERAL_DOMAIN,),
    ).fetchone()
    if row is None:
        return []
    payload = _load_config(row["config_json"])
    raw_records = payload.get("refinements")
    records = raw_records if isinstance(raw_records, list) else []
    matching = [record for record in records if isinstance(record, dict) and _records_match(record, surface=surface, topic=topic)]
    if not matching:
        return []
    remaining = [record for record in records if record not in matching]
    payload["refinements"] = remaining
    payload["updated_at"] = utc_now().isoformat()
    conn.execute("UPDATE dispatch_agents SET config_json = ? WHERE id = ?", (_dump_config(payload), row["id"]))
    return matching


def _matching_general_records(conn, *, surface: str, topic: str) -> list[dict[str, Any]]:
    row = conn.execute(
        "SELECT config_json FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
        (_GENERAL_DOMAIN,),
    ).fetchone()
    if row is None:
        return []
    payload = _load_config(row["config_json"])
    raw_records = payload.get("refinements")
    records = raw_records if isinstance(raw_records, list) else []
    return [record for record in records if isinstance(record, dict) and _records_match(record, surface=surface, topic=topic)]


def _domain_records(conn, domain: str) -> list[dict[str, Any]]:
    row = conn.execute(
        "SELECT config_json FROM dispatch_agents WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
        (domain,),
    ).fetchone()
    if row is None:
        return []
    payload = _load_config(row["config_json"])
    raw_records = payload.get("refinements")
    records = raw_records if isinstance(raw_records, list) else []
    return [record for record in records if isinstance(record, dict)]


def record_refinement_memory(
    repo: Any,
    *,
    surface: str,
    topic: str,
    refinement: str,
    prefer_niche: bool = True,
) -> str:
    """Persist a learner refinement as soft future guidance.

    Returns the storage domain used for the memory row.
    """

    cleaned = " ".join((refinement or "").split())
    if not cleaned:
        return ""
    normalized_surface = _normalize_surface(surface)
    domain = _niche_domain(normalized_surface, topic) if prefer_niche and topic.strip() else _GENERAL_DOMAIN
    record = _record(
        surface=normalized_surface,
        topic=topic,
        refinement=cleaned,
        source="preview_refinement",
    )
    with get_connection() as conn:
        records = [record]
        if domain != _GENERAL_DOMAIN:
            records = [*_extract_matching_general_records(conn, surface=normalized_surface, topic=topic), record]
        _upsert_domain_records(conn, domain=domain, surface=normalized_surface, topic=topic, records=records)
        conn.commit()

    if repo is not None and hasattr(repo, "append_feedback_event"):
        try:
            repo.append_feedback_event(
                {
                    "id": generate_internal_id(),
                    "scope_key": f"agent:{domain}",
                    "scope": "agent" if domain != _GENERAL_DOMAIN else "global",
                    "kind": "preview_refinement",
                    "artifact_kind": normalized_surface,
                    "artifact_id": topic or "general",
                    "label": "User typed preview refinement",
                    "free_text": cleaned,
                    "metadata": {"storage_domain": domain, "prefer_niche": prefer_niche},
                    "timestamp": utc_now().isoformat(),
                }
            )
        except Exception:
            pass
    return domain


def refinement_memory_prompt_suffix(*, surface: str, topic: str) -> str:
    """Return relevant refinement memory as a prompt suffix."""

    normalized_surface = _normalize_surface(surface)
    topic = " ".join((topic or "").split())
    with get_connection() as conn:
        niche_records: list[dict[str, Any]] = []
        if topic:
            domain = _niche_domain(normalized_surface, topic)
            promoted = _extract_matching_general_records(conn, surface=normalized_surface, topic=topic)
            if promoted:
                _upsert_domain_records(conn, domain=domain, surface=normalized_surface, topic=topic, records=promoted)
                conn.commit()
            niche_records = _domain_records(conn, domain)
        general_records = [] if niche_records else _matching_general_records(conn, surface=normalized_surface, topic=topic)

    lines: list[str] = []
    if niche_records:
        lines.append("User refinement memory for this same action/topic. Treat this as high-weight soft guidance:")
        for record in niche_records[-5:]:
            lines.append(f"- {record.get('refinement', '')}")
    if general_records:
        if lines:
            lines.append("")
        lines.append("Loose general user refinement memory for a related subject. Consider it only when it fits:")
        for record in general_records[-4:]:
            record_topic = str(record.get("topic", "")).strip()
            prefix = f"{record_topic}: " if record_topic else ""
            lines.append(f"- {prefix}{record.get('refinement', '')}")
    if not lines:
        return ""
    return "\n" + "\n".join(lines).strip() + "\n"
