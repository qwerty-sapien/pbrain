# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Per-action confirmation queue for tier-2 MCP mutations.

Tier-2 tools queue a PendingAction instead of executing; the user reviews and
confirms via `pb mcp pending` / `pb mcp confirm <id>` / `pb mcp reject <id>`.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


_bypass_state = threading.local()


def _bypassing() -> bool:
    return bool(getattr(_bypass_state, "active", False))


class _Bypass:
    def __enter__(self) -> None:
        _bypass_state.active = True

    def __exit__(self, *_exc: object) -> None:
        _bypass_state.active = False


def bypass_confirmation() -> _Bypass:
    """Context manager: tier-2 tools execute their real body when active."""
    return _Bypass()


_IMPL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_impl(tool_name: str, impl: Callable[..., Any]) -> Callable[..., Any]:
    """Register a tool's concrete implementation so confirmed actions can run it."""
    _IMPL_REGISTRY[tool_name] = impl
    return impl


def resolve_impl(tool_name: str) -> Optional[Callable[..., Any]]:
    return _IMPL_REGISTRY.get(tool_name)


@dataclass
class PendingAction:
    id: str
    tool_name: str
    args: dict[str, Any]
    summary: str
    risk: str = "high"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_path(cls, path: Path) -> "PendingAction":
        data = json.loads(path.read_text())
        return cls(**data)


def _pending_dir() -> Path:
    from pb.mcp.context import get_runtime_context

    base = get_runtime_context().data_dir / "mcp_pending"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _new_id() -> str:
    return secrets.token_hex(4)


def queue_pending(
    tool_name: str,
    args: dict[str, Any],
    summary: str,
    *,
    risk: str = "high",
) -> PendingAction:
    """Persist a pending action and return its record."""
    pending = PendingAction(id=_new_id(), tool_name=tool_name, args=args, summary=summary, risk=risk)
    (_pending_dir() / f"{pending.id}.json").write_text(pending.to_json())
    return pending


def list_pending() -> list[PendingAction]:
    out: list[PendingAction] = []
    for path in sorted(_pending_dir().glob("*.json")):
        try:
            out.append(PendingAction.from_path(path))
        except Exception:
            continue
    out.sort(key=lambda a: a.created_at)
    return out


def get_pending(action_id: str) -> Optional[PendingAction]:
    path = _pending_dir() / f"{action_id}.json"
    if not path.exists():
        return None
    return PendingAction.from_path(path)


def delete_pending(action_id: str) -> bool:
    path = _pending_dir() / f"{action_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def execute_pending(action_id: str) -> dict[str, Any]:
    """Look up the queued tool, invoke its real impl with bypass active, then delete."""
    action = get_pending(action_id)
    if action is None:
        return {"ok": False, "error": f"No pending action with id {action_id}"}
    impl = resolve_impl(action.tool_name)
    if impl is None:
        return {"ok": False, "error": f"No registered impl for tool: {action.tool_name}"}
    with bypass_confirmation():
        try:
            result = impl(**action.args)
        except Exception as exc:
            return {"ok": False, "tool": action.tool_name, "error": f"{type(exc).__name__}: {exc}"}
    delete_pending(action_id)
    return {"ok": True, "tool": action.tool_name, "result": result}


def queue_response(action: PendingAction) -> dict[str, Any]:
    """Standard MCP response shape when a tool defers to the queue."""
    return {
        "status": "pending",
        "pending_id": action.id,
        "tool": action.tool_name,
        "summary": action.summary,
        "risk": action.risk,
        "message": (
            f"Awaiting user confirmation. Run `pb mcp confirm {action.id}` to execute, "
            f"`pb mcp reject {action.id}` to dismiss."
        ),
    }
