# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Centralized daily error logging for CLI and provider failures."""

from __future__ import annotations

import io
import json
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TEXT_LIMIT = 20000


@dataclass(frozen=True)
class ErrorLogReference:
    """Location of one appended error entry."""

    path: Path
    line_number: int

    @property
    def filename(self) -> str:
        return self.path.name


def _resolve_error_log_dir(*, data_dir: str | Path | None = None, config: Any | None = None) -> Path:
    if data_dir is not None:
        base = Path(data_dir).expanduser()
    else:
        try:
            from pb.storage.config import get_config, get_data_dir

            cfg = config or get_config()
            base = get_data_dir(cfg)
        except Exception:
            base = Path(tempfile.gettempdir()) / "productivebrain"
    log_dir = base / "error-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _truncate(value: Any, *, limit: int = _TEXT_LIMIT) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        omitted = len(value) - limit
        return f"{value[:limit]}\n... <truncated {omitted} chars>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _truncate(item, limit=limit) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_truncate(item, limit=limit) for item in value]
    if isinstance(value, (int, float, bool)):
        return value
    return _truncate(repr(value), limit=limit)


def _json_block(value: Any) -> str:
    return json.dumps(_truncate(value), indent=2, ensure_ascii=True, sort_keys=True)


def _stream_meta(stream: Any) -> dict[str, Any]:
    if stream is None:
        return {"present": False}
    meta: dict[str, Any] = {
        "present": True,
        "class": stream.__class__.__name__,
        "encoding": getattr(stream, "encoding", None),
        "name": getattr(stream, "name", None),
        "closed": getattr(stream, "closed", None),
    }
    try:
        meta["isatty"] = bool(stream.isatty())
    except Exception:
        meta["isatty"] = None
    return meta


def _safe_stdin_preview(stream: Any, *, limit: int = 4000) -> str | None:
    if stream is None:
        return None
    try:
        if stream.isatty():
            return None
    except Exception:
        return None
    if not isinstance(stream, io.StringIO):
        return None
    try:
        pos = stream.tell()
        preview = stream.read(limit)
        stream.seek(pos)
    except Exception:
        return None
    return preview or None


def _traceback_text(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def log_error(
    *,
    event: str,
    message: str = "",
    exc: BaseException | None = None,
    data_dir: str | Path | None = None,
    config: Any | None = None,
    command: str = "",
    raw_input: str = "",
    argv: list[str] | None = None,
    status: int | str | None = None,
    request_body: Any | None = None,
    response_status: int | str | None = None,
    response_body: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> ErrorLogReference:
    """Append one error entry to the current day's log file."""

    now = datetime.now(timezone.utc)
    try:
        log_path = _resolve_error_log_dir(data_dir=data_dir, config=config) / f"{now:%Y-%m-%d}.txt"
    except Exception:
        fallback_dir = Path(tempfile.gettempdir()) / "productivebrain" / "error-logs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        log_path = fallback_dir / f"{now:%Y-%m-%d}.txt"

    def _line_count(path: Path) -> int:
        try:
            return path.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        except FileNotFoundError:
            return 1
        except OSError:
            return 1

    start_line = _line_count(log_path)

    header_message = (message or (str(exc).strip() if exc else "") or event).strip()
    stdin_preview = _safe_stdin_preview(sys.stdin)
    payload_lines = [
        f"[{now.isoformat()}] {event}",
        f"message: {header_message or '(empty)'}",
        f"status: {status if status is not None else ''}",
        f"command: {command}",
        f"raw_input: {raw_input}",
        f"argv: {_json_block(argv or sys.argv)}",
        f"cwd: {os.getcwd()}",
        f"pid: {os.getpid()}",
        f"python: {platform.python_version()}",
        "stdin:",
        _json_block(_stream_meta(sys.stdin)),
        "stdout:",
        _json_block(_stream_meta(sys.stdout)),
        "stderr:",
        _json_block(_stream_meta(sys.stderr)),
    ]
    if stdin_preview:
        payload_lines.extend(
            [
                "stdin_preview:",
                stdin_preview,
            ]
        )
    if request_body is not None:
        payload_lines.extend(
            [
                "request_body:",
                _json_block(request_body),
            ]
        )
    if response_status is not None or response_body is not None:
        payload_lines.extend(
            [
                "response:",
                _json_block(
                    {
                        "status": response_status,
                        "body": response_body,
                    }
                ),
            ]
        )
    if extra:
        payload_lines.extend(
            [
                "extra:",
                _json_block(extra),
            ]
        )
    traceback_text = _traceback_text(exc)
    if traceback_text:
        payload_lines.extend(
            [
                "traceback:",
                traceback_text.rstrip(),
            ]
        )
    payload_lines.append("")
    entry = "\n".join(payload_lines)
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "productivebrain" / "error-logs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        log_path = fallback_dir / f"{now:%Y-%m-%d}.txt"
        start_line = _line_count(log_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    return ErrorLogReference(path=log_path, line_number=start_line)


def format_logged_exception(
    exc: BaseException,
    log_ref: ErrorLogReference,
    *,
    inline_limit: int = 300,
) -> str:
    """Return the user-facing error message for an uncaught exception."""

    message = (str(exc).strip() or exc.__class__.__name__).replace("\n", " ").strip()
    if len(message) < inline_limit:
        return message
    return f"Error logged in Line {log_ref.line_number}, of {log_ref.filename}"
