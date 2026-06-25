# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared preview and resource helpers for study/practise blocks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from pb.cli.pickers import pick_single_choice
from pb.cli.preview import markdown_step_lines
from pb.llm.gemini import FLASH_MODEL, PRO_MODEL, get_client
from pb.llm.json_utils import extract_json_block


class LearningResourceItem(BaseModel):
    """One learner-facing external resource."""

    title: str
    url: str
    resource_type: str = ""
    why: str = ""


class LearningResourceBundle(BaseModel):
    """Curated external resources for one learning block."""

    summary: str = ""
    resources: list[LearningResourceItem] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)


class LearningResourceQc(BaseModel):
    """Flash QC decision over grounded resources."""

    approved_urls: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@dataclass
class ResourceFetchResult:
    """Outcome of an optional grounded resource lookup."""

    bundle: LearningResourceBundle | None = None
    qc_notes: list[str] = field(default_factory=list)
    warning: str = ""

    def __post_init__(self) -> None:
        if self.bundle is not None and isinstance(self.bundle, dict):
            self.bundle = LearningResourceBundle.model_validate(self.bundle)

    @property
    def search_terms(self) -> list[str]:
        if self.bundle is None:
            return []
        return list(self.bundle.search_terms or [])


def choose_learning_block_action(action_label: str) -> str | None:
    """Offer a small set of preview-time actions."""

    return pick_single_choice(
        [
            ("start", f"[Start] {action_label}"),
            ("revise", "[Revise] Change this draft"),
            ("cancel", "[Cancel] Do not start this block"),
        ],
        title="Preview options",
    )


def build_learning_session_markdown(
    *,
    task_title: str,
    steps: list[object] | None = None,
    resources: ResourceFetchResult | None = None,
) -> str | None:
    """Render the once-per-session guide shown immediately after start."""

    lines = ["# Session Guide", "", f"**{task_title}**", ""]
    added_content = False

    step_lines = markdown_step_lines(list(steps or []))
    if step_lines:
        lines.extend(step_lines)
        lines.append("")
        added_content = True

    resource_sections = resource_preview_sections(resources)
    for title, section_lines in resource_sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(section_lines)
        lines.append("")
        added_content = True

    if not added_content:
        return None
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def resource_preview_sections(resources: ResourceFetchResult | None) -> list[tuple[str, list[str]]]:
    """Return markdown-ready preview sections for fetched resources."""

    if resources is None:
        return []

    sections: list[tuple[str, list[str]]] = []
    bundle = resources.bundle
    if bundle is not None:
        resource_lines: list[str] = []
        if bundle.summary.strip():
            resource_lines.append(bundle.summary.strip())
            resource_lines.append("")
        for item in bundle.resources:
            label = f"[{item.title}]({item.url})"
            suffix_bits = [item.resource_type.strip(), item.why.strip()]
            suffix = ": " + " | ".join(bit for bit in suffix_bits if bit) if any(suffix_bits) else ""
            resource_lines.append(f"- {label}{suffix}")
        if resource_lines:
            while resource_lines and resource_lines[-1] == "":
                resource_lines.pop()
            sections.append(("Resources", resource_lines))

    if resources.qc_notes:
        sections.append(("Resource notes", [f"- {note}" for note in resources.qc_notes if note.strip()]))
    if bundle is not None and bundle.search_terms:
        sections.append(("Search help", [f"- {term}" for term in bundle.search_terms if term.strip()]))
    return sections


def fetch_grounded_learning_resources(
    *,
    topic: str,
    branch: str,
    block_payload: dict[str, Any],
    max_results: int = 4,
) -> ResourceFetchResult:
    """Fetch optional supporting resources with Gemini Pro grounding plus Flash QC."""

    search_terms = _fallback_search_terms(topic=topic, branch=branch, block_payload=block_payload)
    client = get_client()
    if not client.is_available():
        return ResourceFetchResult(
            bundle=LearningResourceBundle(search_terms=search_terms),
            warning=(
                "Gemini grounding is unavailable right now. If you want resources, try another model and start with the search help below."
            ),
        )

    prompt_payload = {
        "topic": topic,
        "branch": branch,
        "block": block_payload,
        "task": (
            "Use Google Search grounding to curate the smallest set of truly useful resources for this exact learning block. "
            "Return strict JSON with keys summary, resources, and search_terms. "
            f"resources should contain at most {max_results} items, each with title, url, resource_type, and why. "
            "Prefer canonical docs, strong tutorials, relevant papers, and practical tools. "
            "Include videos only when video instruction would materially help."
        ),
    }
    grounded_raw = client.generate_with_grounding(json.dumps(prompt_payload), PRO_MODEL)
    if not grounded_raw:
        return ResourceFetchResult(
            bundle=LearningResourceBundle(search_terms=search_terms),
            warning=(
                "Gemini Pro grounding could not fetch resources for this block. Try another model and use the search help below."
            ),
        )

    try:
        bundle = LearningResourceBundle.model_validate_json(extract_json_block(grounded_raw))
    except (ValidationError, ValueError, json.JSONDecodeError):
        return ResourceFetchResult(
            bundle=LearningResourceBundle(search_terms=search_terms),
            warning=(
                "Gemini Pro grounding returned an unusable resource bundle. Try another model and use the search help below."
            ),
        )

    deduped_resources: list[LearningResourceItem] = []
    seen_urls: set[str] = set()
    for item in bundle.resources:
        url = item.url.strip()
        if not url or not url.startswith(("http://", "https://")) or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_resources.append(item)
    bundle.resources = deduped_resources[:max_results]
    bundle.search_terms = list(dict.fromkeys([*bundle.search_terms, *search_terms]))

    if not bundle.resources:
        return ResourceFetchResult(
            bundle=bundle,
            warning=(
                "Gemini Pro grounding did not return any clean links for this block. Try another model and use the search help below."
            ),
        )

    qc_payload = {
        "topic": topic,
        "branch": branch,
        "block": block_payload,
        "resources": [item.model_dump(mode="json") for item in bundle.resources],
        "task": (
            "Review these resources for exact fit. Return strict JSON with keys approved_urls and notes. "
            "Reject anything that is off-topic, low-signal, redundant, or too generic for the block."
        ),
    }
    qc_raw = client.generate_with_model(
        json.dumps(qc_payload),
        model=FLASH_MODEL,
        timeout=25,
        max_output_tokens=4000,
    )
    if not qc_raw:
        return ResourceFetchResult(
            bundle=bundle,
            warning=(
                "Gemini Flash QC did not complete. These links may still help, but sanity-check them and use the search help if needed."
            ),
        )

    try:
        qc = LearningResourceQc.model_validate_json(extract_json_block(qc_raw))
    except (ValidationError, ValueError, json.JSONDecodeError):
        return ResourceFetchResult(
            bundle=bundle,
            warning=(
                "Gemini Flash QC returned an unusable review. These links may still help, but sanity-check them and use the search help if needed."
            ),
        )

    approved = set(url.strip() for url in qc.approved_urls if url.strip())
    if approved:
        bundle.resources = [item for item in bundle.resources if item.url in approved]
    if not bundle.resources:
        return ResourceFetchResult(
            bundle=LearningResourceBundle(search_terms=bundle.search_terms),
            qc_notes=list(qc.notes),
            warning=(
                "Gemini Flash QC rejected the fetched resources for this block. Try another model and use the search help below."
            ),
        )
    return ResourceFetchResult(bundle=bundle, qc_notes=list(qc.notes))
def _fallback_search_terms(*, topic: str, branch: str, block_payload: dict[str, Any]) -> list[str]:
    scope = str(block_payload.get("subject_scope", "") or topic).strip()
    success = str(block_payload.get("success_check", "")).strip()
    drill = str(block_payload.get("drill_type", "")).strip()
    focus = drill or scope or topic
    quoted_scope = _quoted(scope or topic or "the topic")
    quoted_focus = _quoted(focus or topic or "the skill")

    search_terms = [
        f'Web: {quoted_scope} tutorial OR guide -reddit -quora -pinterest',
        f'YouTube: {quoted_focus} exercise OR walkthrough -shorts -podcast -reaction',
    ]
    if branch == "study":
        search_terms.append(f'Papers/docs: {quoted_scope} review OR documentation filetype:pdf -slides')
    else:
        search_terms.append(f'Practice drill: {quoted_focus} coaching OR drill OR feedback -highlights -compilation')
    if success:
        search_terms.append(f'Outcome-focused: {quoted_scope} "{success[:80]}"')
    return list(dict.fromkeys(term.strip() for term in search_terms if term.strip()))


def _quoted(text: str) -> str:
    cleaned = " ".join((text or "").split()).strip().strip('"')
    return f'"{cleaned}"' if cleaned else '""'
