# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Provider-neutral model policy helpers for learning flows."""

from __future__ import annotations

from typing import Any

_DEFAULT_ROLE_BY_OPERATION = {
    "routing": "fast_inference",
    "command_repair": "fast_inference",
    "mcq_or_cloze_repair": "fast_inference",
    "answer_check": "fast_inference",
    "small_retry": "fast_inference",
    "recall_inline": "fast_inference",
    "session_explain": "default",
    "lesson_hint_intuitive": "default",
    "drill_generation": "default",
    "complex_free_response_eval": "default",
    "lesson_planning": "planner",
    "scoped_recall_generation": "recall",
}


def resolve_model_binding(config: Any, operation: str) -> str:
    """Resolve one learning operation to a configured provider:model binding."""
    learning_cfg = getattr(config, "learning", None)
    policy = getattr(learning_cfg, "model_policy", None)
    roles = getattr(config, "model_roles", None)

    role_name = str(getattr(policy, operation, "") or "").strip()
    if not role_name:
        role_name = _DEFAULT_ROLE_BY_OPERATION.get(operation, "default")
    if role_name and roles is not None:
        binding = str(getattr(roles, role_name, "") or "").strip()
        if binding:
            return binding

    default_binding = str(getattr(getattr(config, "model_roles", None), "default", "") or "").strip()
    if default_binding:
        return default_binding

    provider = str(getattr(getattr(config, "llm", None), "provider", "gemini") or "gemini")
    model = str(getattr(getattr(config, "llm", None), "default_model", "") or "").strip()
    return f"{provider}:{model}" if model else provider


def unique_model_sequence(config: Any, *operations: str) -> list[str]:
    """Resolve one or more operations to a de-duplicated fallback sequence."""
    bindings: list[str] = []
    for operation in operations:
        binding = resolve_model_binding(config, operation)
        if binding and binding not in bindings:
            bindings.append(binding)
    fallback = resolve_model_binding(config, "session_explain")
    if fallback and fallback not in bindings:
        bindings.append(fallback)
    return bindings
