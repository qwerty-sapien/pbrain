# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Durable scoped feedback profiles for learner-facing LLM flows."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pb.core.clock import utc_now
from pb.vault.lifecycle import read_frontmatter, write_frontmatter


SUPPORTED_FEEDBACK_SCOPES = {
    "anki",
    "diagnostic",
    "general",
    "goal",
    "learn",
    "plan",
    "practise",
    "practice",
    "review",
    "study",
    "teach",
}


def normalize_feedback_scope(scope: str) -> str:
    """Normalize aliases to the canonical scope label."""
    normalized = (scope or "").strip().lower()
    if normalized == "practice":
        return "practise"
    return normalized or "general"


def feedback_profile_dir(vault_path: Path) -> Path:
    path = vault_path / "direction" / "preferences"
    path.mkdir(parents=True, exist_ok=True)
    return path


def feedback_profile_path(vault_path: Path, scope: str) -> Path:
    normalized = normalize_feedback_scope(scope)
    safe = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "general"
    return feedback_profile_dir(vault_path) / f"{safe}.md"


def save_feedback_profile(
    vault_path: Path,
    scope: str,
    *,
    more_of: str = "",
    less_of: str = "",
    learner_context: str = "",
    keep_in_mind: str = "",
    focus_note: str = "",
) -> Path:
    """Write one scoped feedback profile note into the vault."""
    normalized = normalize_feedback_scope(scope)
    note_path = feedback_profile_path(vault_path, normalized)
    fm = {
        "type": "feedback_profile",
        "scope": normalized,
        "updated": utc_now().strftime("%Y-%m-%d"),
    }

    lines = [
        f"# Feedback Profile: {normalized}",
        "",
        "## Working Guidance",
        "",
    ]
    if more_of.strip():
        lines.append(f"- More of: {more_of.strip()}")
    if less_of.strip():
        lines.append(f"- Less of: {less_of.strip()}")
    if learner_context.strip():
        lines.append(f"- Learner context: {learner_context.strip()}")
    if keep_in_mind.strip():
        lines.append(f"- Keep in mind: {keep_in_mind.strip()}")
    if focus_note.strip():
        lines.append(f"- Scoped note: {focus_note.strip()}")
    if lines[-1] == "":
        lines.append("- No explicit guidance captured yet.")

    lines.extend(
        [
            "",
            "## Prompt Snippet",
            "",
            "Use this as durable user guidance for the scoped workflow.",
        ]
    )
    if more_of.strip():
        lines.append(f"Lean toward: {more_of.strip()}")
    if less_of.strip():
        lines.append(f"Avoid or reduce: {less_of.strip()}")
    if learner_context.strip():
        lines.append(f"Assume this learner context: {learner_context.strip()}")
    if keep_in_mind.strip():
        lines.append(f"Keep this preference in mind: {keep_in_mind.strip()}")
    if focus_note.strip():
        lines.append(f"Scope-specific instruction: {focus_note.strip()}")
    lines.append("")

    note_path.write_text(
        write_frontmatter(fm, "\n".join(lines).rstrip() + "\n"),
        encoding="utf-8",
    )
    return note_path


def load_feedback_guidance(vault_path: Path, scope: str) -> str:
    """Load combined general and scoped guidance for prompt injection."""
    normalized = normalize_feedback_scope(scope)
    candidate_scopes = ["general"]
    if normalized != "general":
        candidate_scopes.append(normalized)

    bodies: list[str] = []
    seen: set[str] = set()
    for candidate in candidate_scopes:
        if candidate in seen:
            continue
        seen.add(candidate)
        note_path = feedback_profile_path(vault_path, candidate)
        if not note_path.exists():
            continue
        try:
            _, body = read_frontmatter(note_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cleaned = body.strip()
        if cleaned:
            bodies.append(cleaned)
    return "\n\n".join(bodies).strip()


def feedback_prompt_suffix(vault_path: Path, scope: str) -> str:
    """Render scoped feedback as a prompt suffix."""
    guidance = load_feedback_guidance(vault_path, scope)
    if not guidance:
        return ""
    return f"\nUser feedback guidance for `{normalize_feedback_scope(scope)}`:\n{guidance}\n"


def learner_level_assertions_path(vault_path: Path) -> Path:
    """Return the durable learner self-report note path."""

    return feedback_profile_dir(vault_path) / "learner-levels.md"


def append_learner_level_assertion(
    vault_path: Path,
    *,
    topic: str,
    level: int | None = None,
    confidence: int | None = None,
    evidence: str = "",
    note: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Append an explicit learner understanding assertion to the vault."""

    now = datetime.utcnow().isoformat()
    record = {
        "created_at": now,
        "topic": topic.strip(),
        "level": level,
        "confidence": confidence,
        "evidence": evidence.strip(),
        "note": note.strip(),
    }
    path = learner_level_assertions_path(vault_path)
    existing_body = ""
    if path.exists():
        try:
            _, existing_body = read_frontmatter(path.read_text(encoding="utf-8"))
        except Exception:
            existing_body = path.read_text(encoding="utf-8", errors="replace")

    if not existing_body.strip():
        lines = [
            "# Learner Level Assertions",
            "",
            "Explicit self-reports. Treat these as useful learner claims, not proof of mastery.",
            "",
            "## Entries",
            "",
        ]
    else:
        lines = [existing_body.rstrip(), ""]

    level_text = str(level) if level is not None else "unspecified"
    confidence_text = str(confidence) if confidence is not None else "unspecified"
    parts = [
        f"created_at={now}",
        f"topic={record['topic']}",
        f"level={level_text}",
        f"confidence={confidence_text}",
    ]
    if record["evidence"]:
        parts.append(f"evidence={record['evidence']}")
    if record["note"]:
        parts.append(f"note={record['note']}")
    lines.append("- " + " | ".join(parts))
    fm = {
        "type": "learner_level_assertions",
        "updated": utc_now().strftime("%Y-%m-%d"),
    }
    path.write_text(write_frontmatter(fm, "\n".join(lines).rstrip() + "\n"), encoding="utf-8")
    return path, record


def load_learner_level_assertions(vault_path: Path, *, limit: int = 8) -> list[dict[str, str]]:
    """Load recent learner level assertions from the vault note."""

    path = learner_level_assertions_path(vault_path)
    if not path.exists():
        return []
    try:
        _, body = read_frontmatter(path.read_text(encoding="utf-8"))
    except Exception:
        body = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    for line in body.splitlines():
        text = line.strip()
        if not text.startswith("- "):
            continue
        payload = text[2:]
        row: dict[str, str] = {}
        for part in payload.split(" | "):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row.get("topic"):
            rows.append(row)
    return rows[-limit:]
