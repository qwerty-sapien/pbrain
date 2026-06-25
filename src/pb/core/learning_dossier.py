# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Canonical subtopic dossier persistence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from pb.core.graph_writer import make_slug
from pb.vault.lifecycle import read_frontmatter, write_frontmatter


@dataclass(frozen=True)
class DossierKey:
    """Stable dossier identity for one scoped learning topic."""

    domain_slug: str
    subtopic_slug: str
    domain_title: str
    subtopic_title: str
    aliases: tuple[str, ...] = ()

    @property
    def relative_path(self) -> Path:
        return Path("knowledge") / self.domain_slug / f"{self.subtopic_slug}.md"

    def absolute_path(self, vault_path: Path) -> Path:
        return Path(vault_path) / self.relative_path


@dataclass(frozen=True)
class LessonDossierSignals:
    """Compact lesson-level signals worth preserving in the dossier."""

    summary: str = ""
    explanation: str = ""
    key_points: tuple[str, ...] = ()
    misconceptions: tuple[str, ...] = ()
    next_moves: tuple[str, ...] = ()
    question_patterns: tuple[str, ...] = ()
    fragile_concepts: tuple[str, ...] = ()


@dataclass(frozen=True)
class PartnerDossierSignals:
    """Compact partner-memory signals worth preserving in the dossier."""

    summary: str = ""
    knowns: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    detected_gaps: tuple[str, ...] = ()
    recall_candidates: tuple[str, ...] = ()
    corrections: tuple[str, ...] = ()
    next_drill: str = ""
    next_action: str = ""
    control_signals: tuple[str, ...] = ()
    escalation_level: int = 0


def _clean_title(text: object, fallback: str) -> str:
    value = " ".join(str(text or "").strip().split())
    return value or fallback


def _slug(text: object, fallback: str) -> str:
    return make_slug(_clean_title(text, fallback)) or fallback


def _session_date(session) -> str:
    value = getattr(session, "end_at", None) or getattr(session, "start_at", None)
    if isinstance(value, datetime):
        return value.date().isoformat()
    raw = str(value or "").strip()
    return raw[:10] if raw else ""


def _coerce_frontmatter_payload(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique_strings(values: Iterable[object], *, limit: int = 16) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        clean = " ".join(str(item or "").strip().split())
        if not clean:
            continue
        lowered = clean.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(clean)
        if len(result) >= limit:
            break
    return result


def _merge_string_lists(*groups: Iterable[object], limit: int = 16) -> list[str]:
    flattened: list[object] = []
    for group in groups:
        flattened.extend(list(group or []))
    return _unique_strings(flattened, limit=limit)


def _merge_refs(existing: Iterable[object], additions: Iterable[object], *, key: str) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in [*list(existing or []), *list(additions or [])]:
        if not isinstance(raw, dict):
            continue
        identifier = str(raw.get(key, "") or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        merged.append({field: " ".join(str(value or "").split()) for field, value in raw.items() if str(value or "").strip()})
    return merged


def _merge_paths(existing: Iterable[object], additions: Iterable[object], *, limit: int = 24) -> list[str]:
    return _merge_string_lists(existing, additions, limit=limit)


def _relpath(base: Path, value: Optional[Path]) -> str:
    if value is None:
        return ""
    try:
        return str(value.relative_to(base))
    except Exception:
        return str(value)


def resolve_subtopic_dossier_key(
    *,
    session=None,
    task=None,
    domain: str = "",
    subtopic: str = "",
) -> DossierKey:
    """Resolve the canonical dossier path for one scoped learning topic."""

    task_title = _clean_title(getattr(task, "title", ""), "topic")
    subtopic_title = _clean_title(
        subtopic or getattr(session, "subject_scope", "") or task_title,
        "topic",
    )
    domain_title = _clean_title(
        domain or getattr(session, "branch", "") or getattr(session, "subject_scope", "") or task_title,
        "general",
    )
    aliases = _unique_strings([subtopic_title, getattr(session, "subject_scope", ""), task_title], limit=8)
    return DossierKey(
        domain_slug=_slug(domain_title, "general"),
        subtopic_slug=_slug(subtopic_title, "topic"),
        domain_title=domain_title,
        subtopic_title=subtopic_title,
        aliases=tuple(aliases),
    )


def list_learning_dossiers(vault_path: Path) -> list[dict[str, Any]]:
    """Return structured dossier frontmatter payloads from the knowledge tree."""

    root = Path(vault_path) / "knowledge"
    if not root.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.md")):
        try:
            frontmatter, _ = read_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        payload = _coerce_frontmatter_payload(frontmatter)
        if str(payload.get("type", "")).strip() != "learning_dossier":
            continue
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


class LearningDossierUpdater:
    """Idempotently upsert one canonical knowledge dossier per subtopic."""

    def __init__(self, vault_path: Path):
        self.vault_path = Path(vault_path)

    def _load_payload(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            frontmatter, _ = read_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            return {}
        return _coerce_frontmatter_payload(frontmatter)

    def _render_body(self, payload: dict[str, Any]) -> str:
        title = str(payload.get("title", "") or payload.get("subtopic", "") or "Learning dossier").strip()
        aliases = [str(item).strip() for item in payload.get("aliases", []) if str(item).strip()]
        strengths = [str(item).strip() for item in payload.get("strengths", []) if str(item).strip()]
        weaknesses = [str(item).strip() for item in payload.get("weaknesses", []) if str(item).strip()]
        patterns = [str(item).strip() for item in payload.get("question_patterns", []) if str(item).strip()]
        fragile = [str(item).strip() for item in payload.get("fragile_concepts", []) if str(item).strip()]
        drills = [str(item).strip() for item in payload.get("next_drills", []) if str(item).strip()]
        recall = [str(item).strip() for item in payload.get("recall_cues", []) if str(item).strip()]
        tasks = [item for item in payload.get("task_refs", []) if isinstance(item, dict)]
        sessions = [item for item in payload.get("session_refs", []) if isinstance(item, dict)]
        evidence_refs = [str(item).strip() for item in payload.get("evidence_refs", []) if str(item).strip()]
        transcript_refs = [str(item).strip() for item in payload.get("transcript_refs", []) if str(item).strip()]

        lines = [f"# {title}", ""]
        if aliases:
            lines.extend(["## Aliases", ""])
            lines.extend(f"- {item}" for item in aliases)
            lines.append("")

        latest_summary = str(payload.get("latest_summary", "") or "").strip()
        if latest_summary:
            lines.extend(["## Latest Summary", "", latest_summary, ""])

        def _section(title_text: str, rows: list[str], empty: str) -> None:
            lines.extend([f"## {title_text}", ""])
            if rows:
                lines.extend(f"- {row}" for row in rows)
            else:
                lines.append(f"- {empty}")
            lines.append("")

        _section("Strengths", strengths, "No clear strengths captured yet.")
        _section("Weaknesses", weaknesses, "No persistent weaknesses captured yet.")
        _section("Recurring Misses", patterns, "No recurring miss pattern captured yet.")
        _section("Fragile Concepts", fragile, "No fragile concepts captured yet.")
        _section("Next Drills", drills, "No next drill captured yet.")
        _section("Recall Cues", recall, "No recall cue captured yet.")

        lines.extend(["## Recent Sessions", ""])
        if sessions:
            for item in sessions[:8]:
                date = str(item.get("date", "") or "").strip()
                branch = str(item.get("branch", "") or "").strip()
                objective = str(item.get("objective", "") or "").strip()
                session_id = str(item.get("id", "") or "").strip()
                text = " · ".join(part for part in [date, branch, session_id] if part)
                if objective:
                    text = f"{text} · {objective}" if text else objective
                lines.append(f"- {text or session_id}")
        else:
            lines.append("- No session history captured yet.")
        lines.append("")

        lines.extend(["## Linked Tasks", ""])
        if tasks:
            for item in tasks:
                task_id = str(item.get("id", "") or "").strip()
                title_text = str(item.get("title", "") or "").strip()
                lines.append(f"- {' · '.join(part for part in [task_id, title_text] if part)}")
        else:
            lines.append("- No linked tasks captured yet.")
        lines.append("")

        lines.extend(["## Supporting Records", ""])
        supporting_rows = [f"Evidence: {item}" for item in evidence_refs[:8]]
        supporting_rows.extend(f"Transcript: {item}" for item in transcript_refs[:8])
        if supporting_rows:
            lines.extend(f"- {row}" for row in supporting_rows)
        else:
            lines.append("- No supporting records linked yet.")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def upsert(
        self,
        *,
        key: DossierKey,
        session=None,
        task=None,
        lesson: LessonDossierSignals | None = None,
        partner: PartnerDossierSignals | None = None,
        evidence_paths: Iterable[Path] | None = None,
        transcript_path: Path | None = None,
    ) -> Path:
        """Merge one session's durable signals into the canonical dossier."""

        path = key.absolute_path(self.vault_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._load_payload(path)
        created_at = str(payload.get("created", "") or "").strip() or datetime.now(UTC).isoformat()

        session_refs: list[dict[str, str]] = []
        if session is not None:
            session_refs.append(
                {
                    "id": str(getattr(session, "id", "") or "").strip(),
                    "date": _session_date(session),
                    "branch": str(getattr(session, "branch", "") or "").strip(),
                    "objective": str(
                        getattr(session, "intended_outcome", "") or getattr(session, "subject_scope", "") or ""
                    ).strip(),
                }
            )

        task_refs: list[dict[str, str]] = []
        if task is not None:
            task_refs.append(
                {
                    "id": str(getattr(task, "id", "") or "").strip(),
                    "title": _clean_title(getattr(task, "title", ""), "task"),
                }
            )

        lesson = lesson or LessonDossierSignals()
        partner = partner or PartnerDossierSignals()
        latest_summary = partner.summary.strip() or lesson.summary.strip() or str(payload.get("latest_summary", "") or "").strip()

        merged = {
            "type": "learning_dossier",
            "domain": key.domain_title,
            "domain_slug": key.domain_slug,
            "subtopic": key.subtopic_title,
            "subtopic_slug": key.subtopic_slug,
            "title": key.subtopic_title,
            "aliases": _merge_string_lists(payload.get("aliases", []), key.aliases, limit=12),
            "created": created_at,
            "updated": datetime.now(UTC).isoformat(),
            "latest_summary": latest_summary,
            "task_refs": _merge_refs(payload.get("task_refs", []), task_refs, key="id"),
            "session_refs": _merge_refs(payload.get("session_refs", []), session_refs, key="id"),
            "strengths": _merge_string_lists(
                payload.get("strengths", []),
                lesson.key_points,
                partner.knowns,
                limit=16,
            ),
            "weaknesses": _merge_string_lists(
                payload.get("weaknesses", []),
                lesson.misconceptions,
                partner.unknowns,
                partner.detected_gaps,
                partner.corrections,
                limit=20,
            ),
            "question_patterns": _merge_string_lists(
                payload.get("question_patterns", []),
                lesson.question_patterns,
                limit=16,
            ),
            "fragile_concepts": _merge_string_lists(
                payload.get("fragile_concepts", []),
                lesson.fragile_concepts,
                partner.detected_gaps,
                partner.unknowns,
                limit=16,
            ),
            "next_drills": _merge_string_lists(
                payload.get("next_drills", []),
                lesson.next_moves,
                [partner.next_drill, partner.next_action],
                limit=12,
            ),
            "recall_cues": _merge_string_lists(
                payload.get("recall_cues", []),
                partner.recall_candidates,
                limit=16,
            ),
            "evidence_refs": _merge_paths(
                payload.get("evidence_refs", []),
                [_relpath(self.vault_path, path_item) for path_item in list(evidence_paths or [])],
                limit=24,
            ),
            "transcript_refs": _merge_paths(
                payload.get("transcript_refs", []),
                [_relpath(self.vault_path.parent, transcript_path)] if transcript_path is not None else [],
                limit=24,
            ),
            "lesson_summary": lesson.summary.strip(),
            "lesson_explanation": lesson.explanation.strip(),
            "partner_summary": partner.summary.strip(),
            "control_signals": _merge_string_lists(payload.get("control_signals", []), partner.control_signals, limit=12),
            "escalation_level": max(int(payload.get("escalation_level", 0) or 0), int(partner.escalation_level or 0)),
        }

        merged["session_refs"] = sorted(
            merged["session_refs"],
            key=lambda item: (str(item.get("date", "")), str(item.get("id", ""))),
            reverse=True,
        )[:12]
        merged["task_refs"] = merged["task_refs"][:12]

        path.write_text(write_frontmatter(merged, self._render_body(merged)), encoding="utf-8")
        return path


def question_pattern_summary(attempts: Iterable[object], questions: Iterable[object], *, limit: int = 8) -> list[str]:
    """Summarize recurring non-correct question patterns for one lesson run."""

    titles = {}
    for question in questions:
        slug = str(getattr(question, "question_slug", "") or "").strip()
        prompt_json = getattr(question, "prompt_json", {}) or {}
        title = str(prompt_json.get("title", "") or prompt_json.get("prompt", "") or "").strip().splitlines()[0]
        titles[slug] = title or str(getattr(question, "skill_slug", "") or "").replace("_", " ").strip()

    counter: Counter[str] = Counter()
    for attempt in attempts:
        if str(getattr(attempt, "result", "") or "").strip() not in {"wrong", "close", "revealed", "skipped"}:
            continue
        slug = str(getattr(attempt, "question_slug", "") or "").strip()
        label = titles.get(slug) or slug
        if label:
            counter[label] += 1

    results: list[str] = []
    for label, count in counter.most_common(limit):
        suffix = f" ({count} misses)" if count > 1 else ""
        results.append(f"{label}{suffix}")
    return results
