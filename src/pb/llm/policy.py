# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared request policy and slow-call UX helpers for LLM traffic."""

from __future__ import annotations

import os
import sys
import threading
import time as _time
from contextlib import contextmanager
from typing import Iterator


DEFAULT_LONG_MODEL_TIMEOUT_SECONDS = 90
DEFAULT_THINKING_NOTICE_SECONDS = 5.0
_thinking_notice_lock = threading.Lock()
_thinking_notice_visible = False

# Burst dedup: suppress repeated "Thinking…" notices within the same command.
# After _THINKING_BURST_IDLE_SECONDS of no LLM activity the burst resets and
# the next call is eligible to print again.
_THINKING_BURST_IDLE_SECONDS = 3.0
_thinking_burst_printed = False
_thinking_burst_last_activity = 0.0


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _thinking_notice_seconds() -> float:
    raw = os.environ.get("PB_LLM_THINKING_NOTICE_SECONDS")
    if raw:
        try:
            parsed = float(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_THINKING_NOTICE_SECONDS


def _normalized_model(provider: str, model: str) -> str:
    raw = str(model or "").strip()
    if provider in {"openai", "openrouter"} and "/" in raw:
        _, raw = raw.split("/", 1)
    return raw


def _needs_long_timeout(provider: str, model: str) -> bool:
    normalized = _normalized_model(provider, model).lower()
    if provider == "gemini":
        return "pro" in normalized
    if provider == "anthropic":
        return "opus" in normalized
    if provider in {"openai", "openrouter"}:
        return normalized.startswith("gpt-5.4") or normalized.startswith("gpt-5.5")
    return False


def model_display_label(provider: str, model: str) -> str:
    """Return a short user-facing model label for slow-call notices."""
    normalized = _normalized_model(provider, model)
    lowered = normalized.lower()
    if provider == "gemini":
        if "flash-lite" in lowered:
            return "Gemini Flash Lite"
        if "flash" in lowered:
            return "Gemini Flash"
        if "pro" in lowered:
            return "Gemini Pro"
        return f"Gemini {normalized}".strip()
    if provider == "anthropic":
        if "sonnet" in lowered:
            return "Claude Sonnet"
        if "opus" in lowered:
            return "Claude Opus"
        return "Claude"
    if provider in {"openai", "openrouter"}:
        if lowered.startswith("gpt-5.5"):
            return "GPT-5.5"
        if lowered.startswith("gpt-5.4"):
            return "GPT-5.4"
        if lowered.startswith("gpt-5"):
            return normalized.replace("-", " ").upper()
    return normalized or provider.title()


def resolve_timeout(provider: str, model: str, timeout: int, *, config=None) -> int:
    """Extend timeouts for slower premium models without shortening caller budgets."""
    resolved = _coerce_positive_int(timeout, 30)
    if not _needs_long_timeout(provider, model):
        return resolved

    cfg = config
    if cfg is None:
        try:
            from pb.storage.config import get_config

            cfg = get_config()
        except Exception:
            cfg = None
    long_timeout = DEFAULT_LONG_MODEL_TIMEOUT_SECONDS
    if cfg is not None:
        long_timeout = _coerce_positive_int(
            getattr(getattr(cfg, "llm", None), "long_model_timeout_seconds", long_timeout),
            long_timeout,
        )
    return max(resolved, long_timeout)


def _can_render_notice() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stderr.isatty())
    except Exception:
        return False


def _print_notice(message: str) -> None:
    try:
        from pb.cli.console import get_err_console

        get_err_console().print(f"[dim]{message}[/]")
    except Exception:
        print(message, file=sys.stderr)


@contextmanager
def slow_thinking_notice(provider: str, model: str) -> Iterator[None]:
    """Emit a delayed thinking notice when an LLM call exceeds the threshold.

    Only one notice is printed per command burst.  Sequential LLM calls within
    the same command (clarifier → block draft → lesson plan) share the same
    burst window and suppress duplicates.  After _THINKING_BURST_IDLE_SECONDS
    of inactivity the burst resets so the next command can print again.
    """
    global _thinking_burst_printed, _thinking_burst_last_activity

    if not _can_render_notice():
        yield
        return

    # Check / update burst state before yielding
    with _thinking_notice_lock:
        now = _time.monotonic()
        if now - _thinking_burst_last_activity > _THINKING_BURST_IDLE_SECONDS:
            _thinking_burst_printed = False  # new command burst
        _thinking_burst_last_activity = now
        burst_already_printed = _thinking_burst_printed

    if burst_already_printed:
        yield
        with _thinking_notice_lock:
            _thinking_burst_last_activity = _time.monotonic()
        return

    stop = threading.Event()
    printed = False

    def _worker() -> None:
        nonlocal printed
        if stop.wait(timeout=_thinking_notice_seconds()):
            return
        global _thinking_notice_visible, _thinking_burst_printed
        with _thinking_notice_lock:
            if _thinking_notice_visible:
                return
            _thinking_notice_visible = True
            _thinking_burst_printed = True
            printed = True
        _print_notice(f"Thinking with {model_display_label(provider, model)}...")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=0.05)
        with _thinking_notice_lock:
            _thinking_burst_last_activity = _time.monotonic()
            if printed:
                _thinking_notice_visible = False
