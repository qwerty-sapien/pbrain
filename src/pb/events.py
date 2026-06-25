# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Structured event bus for lifecycle logging and diagnostics.

Events are generic lifecycle signals (app started, LLM request finished, etc.)
written as JSONL to a configurable file. Useful for debugging, performance
profiling, and command auditing.

Configuration:
    PB_EVENT_LOG=1          Enable event logging (off by default).
    PB_EVENT_LOG_PATH=...   Override the default log file location.

Default log location (when enabled):
    macOS:   ~/Library/Logs/ProductiveBrain/events.jsonl
    Linux:   $XDG_STATE_HOME/productivebrain/events.jsonl
             (fallback: ~/.local/state/productivebrain/events.jsonl)
    Other:   ~/.productivebrain/logs/events.jsonl
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

_SECRET_ENV_PREFIXES = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AWS_SECRET",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)

_MAX_LOG_BYTES = 50 * 1024 * 1024  # 50 MB


class EventSink(Protocol):
    def emit(self, event: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class NoOpSink:
    def emit(self, event: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


def _default_log_path() -> Path:
    override = os.environ.get("PB_EVENT_LOG_PATH")
    if override:
        return Path(override).expanduser()

    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Logs" / "ProductiveBrain" / "events.jsonl"
    if system == "Linux":
        state_home = os.environ.get("XDG_STATE_HOME", "")
        if not state_home:
            state_home = str(Path.home() / ".local" / "state")
        return Path(state_home) / "productivebrain" / "events.jsonl"
    return Path.home() / ".productivebrain" / "logs" / "events.jsonl"


def _scrub_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove values that look like secrets."""
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        key_upper = key.upper()
        if any(key_upper.startswith(prefix) or key_upper.endswith(("_KEY", "_SECRET", "_TOKEN", "_CREDENTIALS", "_PASSWORD"))
               for prefix in ("",)):
            if any(key_upper.startswith(s) or key_upper == s for s in _SECRET_ENV_PREFIXES):
                clean[key] = "<redacted>"
                continue
            if any(key_upper.endswith(suffix) for suffix in ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIALS")):
                clean[key] = "<redacted>"
                continue
        if isinstance(value, dict):
            clean[key] = _scrub_secrets(value)
        else:
            clean[key] = value
    return clean


class JsonlFileSink:
    """Append-only JSONL file sink with basic size rotation."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _default_log_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._maybe_rotate()
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def _maybe_rotate(self) -> None:
        try:
            if self._path.exists() and self._path.stat().st_size > _MAX_LOG_BYTES:
                rotated = self._path.with_suffix(".jsonl.1")
                if rotated.exists():
                    rotated.unlink()
                self._path.rename(rotated)
        except OSError:
            pass

    def close(self) -> None:
        pass


class EventBus:
    """Central event dispatcher. Thread-safe."""

    def __init__(self, sink: EventSink | None = None) -> None:
        self._sink: EventSink = sink or NoOpSink()

    @property
    def sink(self) -> EventSink:
        return self._sink

    def emit(self, event_type: str, **payload: Any) -> None:
        record = {
            "event": event_type,
            "ts": datetime.now(timezone.utc).isoformat(),
            "mono": time.monotonic(),
            **_scrub_secrets(payload),
        }
        self._sink.emit(record)

    def close(self) -> None:
        self._sink.close()


_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the global EventBus, creating it on first call."""
    global _bus
    if _bus is not None:
        return _bus
    with _bus_lock:
        if _bus is not None:
            return _bus
        if os.environ.get("PB_EVENT_LOG", "").strip().lower() in ("1", "true", "yes", "on"):
            _bus = EventBus(sink=JsonlFileSink())
        else:
            _bus = EventBus()
        return _bus


def reset_event_bus(bus: EventBus | None = None) -> None:
    """Replace the global bus (for testing)."""
    global _bus
    with _bus_lock:
        if _bus is not None:
            _bus.close()
        _bus = bus


def emit(event_type: str, **payload: Any) -> None:
    """Convenience: emit on the global bus."""
    get_event_bus().emit(event_type, **payload)
