# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Helpers for storing lightweight learning metadata inside task descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from pb.core.renderables import renderable_cli_text, renderable_payload


_PREFIX = "PB_"


@dataclass
class LearningTaskMetadata:
    """Structured learning metadata recovered from a task."""

    branch: str = ""
    scope: str = ""
    domain: str = ""
    bloom_target: str = ""
    study_mode: str = ""
    practice_stage: str = ""
    drill: str = ""
    constraint: str = ""
    feedback_source: str = ""
    evidence_target: str = ""
    success_check: str = ""
    cues: str = ""
    domain_pack_id: str = ""
    session_blueprint: dict[str, Any] | None = None
    steps: list[dict[str, Any]] | None = None
    plan_id: str = ""
    plan_title: str = ""
    plan_node_id: str = ""
    depends_on_task_ids: list[str] | None = None
    roadmap_id: str = ""
    roadmap_node_id: str = ""
    parent_node_id: str = ""
    sequence_level: int = 0
    sequence_status: str = ""
    goal_project_title: str = ""
    goal_progress_mode: str = ""


def build_learning_task_description(
    summary: str = "",
    *,
    branch: str = "",
    scope: str = "",
    domain: str = "",
    bloom_target: str = "",
    study_mode: str = "",
    practice_stage: str = "",
    drill: str = "",
    constraint: str = "",
    feedback_source: str = "",
    evidence_target: str = "",
    success_check: str = "",
    cues: str = "",
    domain_pack_id: str = "",
    session_blueprint: dict[str, Any] | None = None,
    steps: list[Any] | None = None,
    plan_id: str = "",
    plan_title: str = "",
    plan_node_id: str = "",
    depends_on_task_ids: list[str] | None = None,
    roadmap_id: str = "",
    roadmap_node_id: str = "",
    parent_node_id: str = "",
    sequence_level: int = 0,
    sequence_status: str = "",
    goal_project_title: str = "",
    goal_progress_mode: str = "",
) -> str:
    """Serialize learning metadata into a task description."""
    lines: list[str] = []
    if summary.strip():
        lines.append(summary.strip())

    metadata = {
        "BRANCH": branch,
        "SCOPE": scope,
        "DOMAIN": domain,
        "BLOOM_TARGET": bloom_target,
        "STUDY_MODE": study_mode,
        "PRACTICE_STAGE": practice_stage,
        "DRILL": drill,
        "CONSTRAINT": constraint,
        "FEEDBACK_SOURCE": feedback_source,
        "EVIDENCE_TARGET": evidence_target,
        "SUCCESS_CHECK": success_check,
        "CUES": cues,
        "DOMAIN_PACK_ID": domain_pack_id,
    }
    if session_blueprint:
        metadata["SESSION_BLUEPRINT_JSON"] = json.dumps(
            session_blueprint,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    if steps:
        metadata["STEPS_JSON"] = json.dumps(_serialize_steps(steps), separators=(",", ":"), ensure_ascii=True)
    if depends_on_task_ids:
        metadata["DEPENDS_ON_TASK_IDS_JSON"] = json.dumps(
            [item for item in depends_on_task_ids if item],
            separators=(",", ":"),
            ensure_ascii=True,
        )
    if plan_id.strip():
        metadata["PLAN_ID"] = plan_id
    if plan_title.strip():
        metadata["PLAN_TITLE"] = plan_title
    if plan_node_id.strip():
        metadata["PLAN_NODE_ID"] = plan_node_id
    if roadmap_id.strip():
        metadata["ROADMAP_ID"] = roadmap_id
    if roadmap_node_id.strip():
        metadata["ROADMAP_NODE_ID"] = roadmap_node_id
    if parent_node_id.strip():
        metadata["PARENT_NODE_ID"] = parent_node_id
    if sequence_level > 0:
        metadata["SEQUENCE_LEVEL"] = str(sequence_level)
    if sequence_status.strip():
        metadata["SEQUENCE_STATUS"] = sequence_status
    if goal_project_title.strip():
        metadata["GOAL_PROJECT_TITLE"] = goal_project_title
    if goal_progress_mode.strip():
        metadata["GOAL_PROGRESS_MODE"] = goal_progress_mode
    meta_lines = [f"{_PREFIX}{key}: {value.strip()}" for key, value in metadata.items() if value and value.strip()]
    if meta_lines:
        if lines:
            lines.append("")
        lines.extend(meta_lines)
    step_section = _steps_markdown(steps or [])
    if step_section:
        if lines:
            lines.append("")
        lines.extend(step_section)
    return "\n".join(lines).strip()


def parse_learning_task_metadata(task: Any) -> LearningTaskMetadata:
    """Recover learning metadata from a task-like object."""
    description = getattr(task, "description", "") or ""
    raw: dict[str, str] = {}
    for line in description.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_PREFIX) or ":" not in stripped:
            continue
        key, value = stripped[len(_PREFIX) :].split(":", 1)
        raw[key.strip().upper()] = value.strip()

    title = getattr(task, "title", "") or ""
    work_type = (getattr(task, "work_type", "") or "").lower()

    branch = raw.get("BRANCH", "")
    if not branch:
        if work_type in {"practice", "practise"} or title.lower().startswith(("practise:", "practice:")):
            branch = "practise"
        elif work_type == "study" or title.lower().startswith("study:"):
            branch = "study"

    scope = raw.get("SCOPE", "")
    if not scope:
        lowered = title.lower()
        if lowered.startswith("study:"):
            scope = title.split(":", 1)[1].strip()
        elif lowered.startswith("teach:"):
            scope = title.split(":", 1)[1].strip()
        elif lowered.startswith("practise:"):
            scope = title.split(":", 1)[1].strip()
        elif lowered.startswith("practice:"):
            scope = title.split(":", 1)[1].strip()
        if scope and "[" in scope and scope.endswith("]"):
            scope = scope.rsplit("[", 1)[0].strip()

    return LearningTaskMetadata(
        branch=branch,
        scope=scope,
        domain=raw.get("DOMAIN", ""),
        bloom_target=raw.get("BLOOM_TARGET", ""),
        study_mode=raw.get("STUDY_MODE", ""),
        practice_stage=raw.get("PRACTICE_STAGE", ""),
        drill=raw.get("DRILL", ""),
        constraint=raw.get("CONSTRAINT", ""),
        feedback_source=raw.get("FEEDBACK_SOURCE", ""),
        evidence_target=raw.get("EVIDENCE_TARGET", ""),
        success_check=raw.get("SUCCESS_CHECK", ""),
        cues=raw.get("CUES", ""),
        domain_pack_id=raw.get("DOMAIN_PACK_ID", ""),
        session_blueprint=_parse_json_object(raw.get("SESSION_BLUEPRINT_JSON", "")),
        steps=_parse_steps(raw.get("STEPS_JSON", "")),
        plan_id=raw.get("PLAN_ID", ""),
        plan_title=raw.get("PLAN_TITLE", ""),
        plan_node_id=raw.get("PLAN_NODE_ID", ""),
        depends_on_task_ids=_parse_string_list(raw.get("DEPENDS_ON_TASK_IDS_JSON", "")),
        roadmap_id=raw.get("ROADMAP_ID", ""),
        roadmap_node_id=raw.get("ROADMAP_NODE_ID", ""),
        parent_node_id=raw.get("PARENT_NODE_ID", ""),
        sequence_level=_parse_int(raw.get("SEQUENCE_LEVEL", "")),
        sequence_status=raw.get("SEQUENCE_STATUS", ""),
        goal_project_title=raw.get("GOAL_PROJECT_TITLE", ""),
        goal_progress_mode=raw.get("GOAL_PROGRESS_MODE", ""),
    )


def _serialize_steps(steps: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for step in steps:
        title = str(getattr(step, "title", "") or (step.get("title", "") if isinstance(step, dict) else "")).strip()
        if not title:
            continue
        instruction_value = getattr(step, "instruction", "") if not isinstance(step, dict) else step.get("instruction", "")
        success_value = getattr(step, "success_check", "") if not isinstance(step, dict) else step.get("success_check", "")
        serialized.append(
            {
                "title": title,
                "instruction": renderable_payload(instruction_value),
                "success_check": renderable_payload(success_value),
            }
        )
    return serialized


def _steps_markdown(steps: list[Any]) -> list[str]:
    serialized = _serialize_steps(steps)
    if not serialized:
        return []
    lines = ["## Steps", ""]
    for index, step in enumerate(serialized, start=1):
        lines.append(f"{index}.\t{step['title']}")
        instruction_text = renderable_cli_text(step["instruction"]).strip()
        success_text = renderable_cli_text(step["success_check"]).strip()
        if instruction_text:
            lines.append(f"\t{instruction_text}")
        if success_text:
            lines.append(f"\tSuccess:\t{success_text}")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    return lines


def _parse_steps(raw: str) -> list[dict[str, Any]] | None:
    payload = (raw or "").strip()
    if not payload:
        return None
    try:
        loaded = json.loads(payload)
    except Exception:
        return None
    if not isinstance(loaded, list):
        return None
    parsed: list[dict[str, Any]] = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        instruction = item.get("instruction", "")
        success_check = item.get("success_check", "")
        parsed.append(
            {
                "title": title,
                "instruction": instruction if isinstance(instruction, dict) else str(instruction).strip(),
                "success_check": success_check if isinstance(success_check, dict) else str(success_check).strip(),
            }
        )
    return parsed or None


def _parse_string_list(raw: str) -> list[str] | None:
    payload = (raw or "").strip()
    if not payload:
        return None
    try:
        loaded = json.loads(payload)
    except Exception:
        return None
    if not isinstance(loaded, list):
        return None
    values = [str(item).strip() for item in loaded if str(item).strip()]
    return values or None


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    payload = (raw or "").strip()
    if not payload:
        return None
    try:
        loaded = json.loads(payload)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _parse_int(raw: str) -> int:
    try:
        return int((raw or "").strip())
    except Exception:
        return 0
