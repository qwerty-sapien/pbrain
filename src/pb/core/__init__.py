# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Core public API with lazy re-exports to avoid import-time cycles."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "PacketEngine": ("pb.core.packet_engine", "PacketEngine"),
    "Planner": ("pb.core.planner", "Planner"),
    "SessionManager": ("pb.core.sessions", "SessionManager"),
    "ReviewEngine": ("pb.core.review_engine", "ReviewEngine"),
    "TimerManager": ("pb.core.timer", "TimerManager"),
    "send_notification": ("pb.core.timer", "send_notification"),
    "TaskState": ("pb.core.enums", "TaskState"),
    "SessionMode": ("pb.core.enums", "SessionMode"),
    "EnergyType": ("pb.core.enums", "EnergyType"),
    "Horizon": ("pb.core.enums", "Horizon"),
    "ProjectType": ("pb.core.enums", "ProjectType"),
    "ProjectStatus": ("pb.core.enums", "ProjectStatus"),
    "PacketType": ("pb.core.enums", "PacketType"),
    "TaskOutcome": ("pb.core.enums", "TaskOutcome"),
    "WorkType": ("pb.core.enums", "WorkType"),
    "EisenhowerClass": ("pb.core.enums", "EisenhowerClass"),
    "PriorityAction": ("pb.core.enums", "PriorityAction"),
    "ExitCode": ("pb.core.exceptions", "ExitCode"),
    "UserError": ("pb.core.exceptions", "UserError"),
    "NotFoundError": ("pb.core.exceptions", "NotFoundError"),
    "ValidationError": ("pb.core.exceptions", "ValidationError"),
    "ConflictError": ("pb.core.exceptions", "ConflictError"),
    "PbSystemError": ("pb.core.exceptions", "PbSystemError"),
    "DatabaseError": ("pb.core.exceptions", "DatabaseError"),
    "ConfigError": ("pb.core.exceptions", "ConfigError"),
    "Domain": ("pb.core.models", "Domain"),
    "Goal": ("pb.core.models", "Goal"),
    "GoalArc": ("pb.core.models", "GoalArc"),
    "Note": ("pb.core.models", "Note"),
    "Track": ("pb.core.models", "Track"),
    "Project": ("pb.core.models", "Project"),
    "Task": ("pb.core.models", "Task"),
    "Session": ("pb.core.models", "Session"),
    "Packet": ("pb.core.models", "Packet"),
    "Clip": ("pb.core.models", "Clip"),
    "TimeBlock": ("pb.core.models", "TimeBlock"),
    "DailyDebrief": ("pb.core.models", "DailyDebrief"),
    "DailyReviewResponse": ("pb.core.models", "DailyReviewResponse"),
    "generate_slug": ("pb.core.models", "generate_slug"),
    "generate_internal_id": ("pb.core.models", "generate_internal_id"),
    "utc_now": ("pb.core.models", "utc_now"),
    "RuleViolation": ("pb.core.rules", "RuleViolation"),
    "validate_no_learning_without_socratic": ("pb.core.rules", "validate_no_learning_without_socratic"),
    "validate_project_has_packet": ("pb.core.rules", "validate_project_has_packet"),
    "validate_single_active_task": ("pb.core.rules", "validate_single_active_task"),
    "BaseService": ("pb.core.base", "BaseService"),
    "LoggableMixin": ("pb.core.base", "LoggableMixin"),
    "AIMixin": ("pb.core.base", "AIMixin"),
    "ContextFileIngestResult": ("pb.core.context_file_intake", "ContextFileIngestResult"),
    "ContextFileResponsePlan": ("pb.core.context_file_intake", "ContextFileResponsePlan"),
    "plan_context_file_response": ("pb.core.context_file_intake", "plan_context_file_response"),
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
