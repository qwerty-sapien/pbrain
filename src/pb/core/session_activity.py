# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared formatting for durable learning-session activity records."""

from __future__ import annotations

from collections import Counter

from pb.core.renderables import renderable_cli_text


MAX_LEARNING_PARTNER_ACTIVITY_ITEMS = 240


def _clean_text(value: object, *, max_len: int = 900) -> str:
    text = renderable_cli_text(str(value or "")).strip()
    text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def _clean_list(value: object, *, max_items: int = 12, max_len: int = 900) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value[:max_items]:
        text = _clean_text(item, max_len=max_len)
        if text:
            cleaned.append(text)
    return cleaned


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return _clean_text(value)


def coerce_learning_partner_activity(source: object) -> list[dict[str, object]]:
    """Return normalized learning partner activity items from generated names or a raw list."""

    raw = source.get("learning_partner_activity") if isinstance(source, dict) else source
    if not isinstance(raw, list):
        return []
    items: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = _json_safe(item)
        if isinstance(normalized, dict):
            items.append(normalized)
    return items


def append_learning_partner_activity(
    generated_names: dict[str, object] | None,
    item: dict[str, object],
    *,
    max_items: int = MAX_LEARNING_PARTNER_ACTIVITY_ITEMS,
) -> dict[str, object]:
    """Append one activity item and refresh the compact summary payload."""

    updated = dict(generated_names or {})
    items = coerce_learning_partner_activity(updated)
    normalized = _json_safe(item)
    if isinstance(normalized, dict):
        items.append(normalized)
    if max_items > 0 and len(items) > max_items:
        items = items[-max_items:]
    updated["learning_partner_activity"] = items
    updated["learning_partner_activity_summary"] = summarize_learning_partner_activity(items)
    return updated


def summarize_learning_partner_activity(source: object) -> dict[str, object]:
    """Build a small count summary suitable for generated_names and prompts."""

    items = coerce_learning_partner_activity(source)
    attempts = [item for item in items if item.get("kind") == "answer_attempt"]
    commands = [item for item in items if item.get("kind") == "lesson_command"]
    result_counts = Counter(str(item.get("result", "") or "recorded") for item in attempts)
    last = items[-1] if items else {}
    return {
        "attempt_count": len(attempts),
        "command_count": len(commands),
        "correct": int(result_counts.get("correct", 0)),
        "close": int(result_counts.get("close", 0)),
        "wrong": int(result_counts.get("wrong", 0)),
        "revealed": int(result_counts.get("revealed", 0)),
        "skipped": int(result_counts.get("skipped", 0)),
        "last_kind": str(last.get("kind", "") or ""),
        "last_question": str(last.get("question_title", "") or last.get("question_prompt", "") or "")[:160],
        "last_result": str(last.get("result", "") or ""),
    }


def format_learning_partner_activity_markdown(
    source: object,
    *,
    limit: int | None = None,
    title: str = "Session Activity",
) -> str:
    """Render activity records as Markdown for evidence notes, session logs, and LLM context."""

    items = coerce_learning_partner_activity(source)
    if not items:
        return ""

    summary = summarize_learning_partner_activity(items)
    lines = [
        f"## {title}",
        "",
        (
            f"- Attempts: {summary['attempt_count']} "
            f"(correct {summary['correct']}, close {summary['close']}, wrong {summary['wrong']}, "
            f"revealed {summary['revealed']}, skipped {summary['skipped']})"
        ),
    ]
    if summary["command_count"]:
        lines.append(f"- Learning commands used: {summary['command_count']}")
    lines.append("")

    display_items = items
    omitted = 0
    if limit is not None and limit >= 0 and len(items) > limit:
        omitted = len(items) - limit
        display_items = items[-limit:]
        lines.append(f"_Showing the last {limit} activity item(s); {omitted} earlier item(s) omitted._")
        lines.append("")

    start_index = omitted + 1
    for offset, item in enumerate(display_items, start=start_index):
        if item.get("kind") == "answer_attempt":
            lines.extend(_format_attempt_item(item, offset))
        elif item.get("kind") == "lesson_command":
            lines.extend(_format_command_item(item, offset))
        else:
            lines.extend(_format_generic_item(item, offset))
        lines.append("")
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def format_learning_partner_activity_transcript_item(item: dict[str, object]) -> str:
    """Format one item for insertion into the durable assistant/user transcript."""

    rendered = format_learning_partner_activity_markdown([item], title="Recorded Interaction")
    return rendered.strip()


def _format_attempt_item(item: dict[str, object], index: int) -> list[str]:
    title = _clean_text(item.get("question_title")) or _clean_text(item.get("question_prompt")) or "Question"
    result = _clean_text(item.get("result")) or "recorded"
    status = _clean_text(item.get("correctness")) or ("right" if item.get("is_correct") is True else "wrong")
    lines = [f"### {index}. {title}", f"- Result: {result} ({status})"]
    page = _clean_text(item.get("page_title"))
    if page:
        lines.append(f"- Page: {page}")
    prompt = _clean_text(item.get("question_prompt"))
    if prompt and prompt != title:
        lines.append(f"- Prompt: {prompt}")

    selected = _clean_list(item.get("selected_options")) or _clean_list(item.get("selected_text"))
    if not selected:
        selected_text = _clean_text(item.get("selected_text"))
        selected = [selected_text] if selected_text else []
    lines.append(f"- Selected: {' | '.join(selected) if selected else '_No answer captured_'}")

    options = _clean_list(item.get("options"), max_items=16)
    if options:
        selected_keys = {option.lower() for option in selected}
        lines.append("- Options:")
        for option_index, option in enumerate(options, start=1):
            marker = " [selected]" if option.lower() in selected_keys else ""
            lines.append(f"  - {option_index}. {option}{marker}")

    feedback = _clean_list(item.get("feedback"), max_items=6)
    if feedback:
        lines.append(f"- Feedback: {' | '.join(feedback)}")

    target = _clean_list(item.get("target_answers"), max_items=8)
    if target and result in {"correct", "revealed", "skipped"}:
        lines.append(f"- Stored target: {' | '.join(target)}")
    return lines


def _format_command_item(item: dict[str, object], index: int) -> list[str]:
    command = _clean_text(item.get("command")) or "command"
    args = _clean_text(item.get("args"))
    label = f"{command} {args}".strip()
    lines = [f"### {index}. Command: `{label}`"]
    title = _clean_text(item.get("question_title")) or _clean_text(item.get("question_prompt"))
    if title:
        lines.append(f"- Active question: {title}")
    feedback = _clean_list(item.get("feedback"), max_items=6)
    if feedback:
        lines.append(f"- Response: {' | '.join(feedback)}")
    return lines


def _format_generic_item(item: dict[str, object], index: int) -> list[str]:
    kind = _clean_text(item.get("kind")) or "activity"
    text = _clean_text(item.get("text")) or _clean_text(item)
    return [f"### {index}. {kind}", text]
