# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Backward-compatibility shim with lazy exports from `pb.core`."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "TaskState": ("pb.core.enums", "TaskState"),
    "SessionMode": ("pb.core.enums", "SessionMode"),
    "EnergyType": ("pb.core.enums", "EnergyType"),
    "Horizon": ("pb.core.enums", "Horizon"),
    "ProjectType": ("pb.core.enums", "ProjectType"),
    "ProjectStatus": ("pb.core.enums", "ProjectStatus"),
    "PacketType": ("pb.core.enums", "PacketType"),
    "GoalArc": ("pb.core.models", "GoalArc"),
    "Track": ("pb.core.models", "Track"),
    "Project": ("pb.core.models", "Project"),
    "Task": ("pb.core.models", "Task"),
    "Session": ("pb.core.models", "Session"),
    "Packet": ("pb.core.models", "Packet"),
    "Clip": ("pb.core.models", "Clip"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module attribute behavior
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
