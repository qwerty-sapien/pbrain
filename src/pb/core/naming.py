# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Generated naming and routing helpers for learning-first artifacts."""

from __future__ import annotations

import re
from typing import Any

from pb.llm.drafts import GeneratedNamesDraft, NameConfidenceDraft
from pb.llm.runtime import DraftGenerationError, LLMRuntime


def _clean_words(text: str) -> list[str]:
    return [word for word in re.split(r"\s+", (text or "").strip()) if word]


def _sentence_case(text: str) -> str:
    cleaned = " ".join(_clean_words(text))
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def _title_case(text: str) -> str:
    return " ".join(word.capitalize() for word in _clean_words(text))


def _safe_slug(text: str, *, fallback: str = "item", limit: int = 24) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    if not slug:
        return fallback
    if len(slug) <= limit:
        return slug
    parts: list[str] = []
    for token in slug.split("_"):
        if not token:
            continue
        candidate = "_".join(parts + [token])
        if len(candidate) > limit:
            break
        parts.append(token)
    return "_".join(parts) or slug[:limit].strip("_") or fallback


def deterministic_names(kind: str, raw_intent: str, context: dict[str, Any] | None = None) -> GeneratedNamesDraft:
    """Build a safe fallback naming bundle without calling the LLM."""
    context = context or {}
    cleaned = " ".join(_clean_words(raw_intent)) or kind.replace("_", " ")
    short_words = _clean_words(cleaned)[:4]
    short_title = _title_case(" ".join(short_words)) or _title_case(kind)
    display_title = _sentence_case(cleaned) or _title_case(kind)
    base_slug = _safe_slug(cleaned, fallback=kind.replace("-", "_"))
    domain_hint = context.get("domain") or context.get("subject") or context.get("topic") or cleaned
    folder_name = _safe_slug(str(domain_hint), fallback=base_slug)
    frontmatter = {
        "subject": context.get("subject") or cleaned,
        "domain": context.get("domain") or "",
        "activity_type": kind,
    }
    return GeneratedNamesDraft(
        display_title=display_title,
        short_title=short_title,
        slug=base_slug,
        note_title=display_title,
        folder_name=folder_name,
        session_title=display_title,
        task_title=display_title,
        plan_title=display_title,
        goal_title=display_title,
        frontmatter=frontmatter,
        confidence=NameConfidenceDraft(score=0.35, info_density=0.2, routing_relevance=0.5),
    )


def stored_short_title(entity: Any) -> str:
    """Return the stored short title when available."""
    generated = getattr(entity, "generated_names", {}) or {}
    short_title = generated.get("short_title")
    if isinstance(short_title, str) and short_title.strip():
        return short_title.strip()
    title = getattr(entity, "title", "") or getattr(entity, "name", "") or ""
    if not title.strip():
        return ""
    return deterministic_names("display", title).short_title


def stored_display_title(entity: Any) -> str:
    """Return the stored display title when available."""
    generated = getattr(entity, "generated_names", {}) or {}
    display_title = generated.get("display_title")
    if isinstance(display_title, str) and display_title.strip():
        return display_title.strip()
    title = getattr(entity, "title", "") or getattr(entity, "name", "") or ""
    if not title.strip():
        return ""
    return deterministic_names("display", title).display_title


class NameService:
    """Generate concise titles and routing metadata via model roles."""

    def __init__(self, runtime: LLMRuntime):
        self.runtime = runtime

    def _role_binding(self) -> str:
        roles = self.runtime.config.model_roles
        return (
            roles.namer
            or roles.fast_inference
            or roles.fast
            or roles.default
        )

    def generate_names(
        self,
        kind: str,
        raw_intent: str,
        context: dict[str, Any] | None = None,
    ) -> GeneratedNamesDraft:
        """Generate a persisted naming bundle for a learning artifact."""
        context = context or {}
        fallback = deterministic_names(kind, raw_intent, context)
        health = self.runtime.health()
        if not health.available:
            return fallback

        prompt = (
            "You are naming and routing a learning artifact for ProductiveBrain.\n"
            "Return concise, human-readable names and lightweight routing metadata.\n"
            "Rules:\n"
            "- short_title should fit naturally in a shell prompt and never be raw truncation.\n"
            "- slug and folder_name must use lowercase snake_case only.\n"
            "- folder_name should stay stable enough to group related study files later.\n"
            "- Use the raw request, goal/session context, and learner context when present.\n"
            "- Do not mention model providers or internal IDs.\n"
            f"Artifact kind: {kind}\n"
            f"Raw intent: {raw_intent}\n"
            f"Context JSON: {context}\n"
        )
        try:
            draft_result = self.runtime.generate_draft(
                GeneratedNamesDraft,
                prompt,
                source_scope=f"names:{kind}:{raw_intent[:80]}",
                model=self._role_binding(),
                max_output_tokens=4000,
            )
        except DraftGenerationError:
            return fallback
        generated = draft_result.payload
        if not hasattr(generated, "folder_name") or not hasattr(generated, "short_title"):
            return fallback

        if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", generated.folder_name or ""):
            generated.folder_name = fallback.folder_name
        if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", generated.slug or ""):
            generated.slug = fallback.slug
        if not generated.short_title.strip():
            generated.short_title = fallback.short_title
        if not generated.display_title.strip():
            generated.display_title = fallback.display_title
        if not generated.task_title.strip():
            generated.task_title = generated.display_title
        if not generated.session_title.strip():
            generated.session_title = generated.display_title
        if not generated.note_title.strip():
            generated.note_title = generated.display_title
        if not generated.goal_title.strip():
            generated.goal_title = generated.display_title
        if not generated.plan_title.strip():
            generated.plan_title = generated.display_title
        return generated


def apply_generated_names(entity: Any, names: GeneratedNamesDraft) -> None:
    """Persist the generated naming bundle onto a model instance."""
    payload = names.model_dump(mode="python")
    existing = getattr(entity, "generated_names", {}) or {}
    merged = dict(existing)
    merged.update(payload)
    setattr(entity, "generated_names", merged)


def apply_generated_title(
    entity: Any,
    names: GeneratedNamesDraft,
    *,
    title_key: str = "display_title",
    attr: str = "title",
) -> None:
    """Apply one generated title field onto the entity after storing its bundle."""
    apply_generated_names(entity, names)
    title = getattr(names, title_key, "") or getattr(names, "display_title", "")
    if isinstance(title, str) and title.strip():
        setattr(entity, attr, title.strip())
