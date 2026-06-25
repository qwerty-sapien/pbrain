# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared runtime helpers for context-file ingest, scope inheritance, and prompts."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import typer

from pb.cli.context import CommandContext
from pb.core.context_file_intake import (
    ActiveContextScope,
    ContextFileIngestResult,
    active_context_from_sources,
    compatibility_message,
    inspect_context_files,
    plan_context_file_response,
)
from pb.core.models import generate_internal_id
from pb.llm.runtime import LLMRuntime


@dataclass(frozen=True)
class PreparedContextScope:
    """Prepared invocation context for one command."""

    scope: ActiveContextScope | None
    sources: tuple[dict[str, object], ...] = ()
    results: tuple[ContextFileIngestResult, ...] = ()
    messages: tuple[str, ...] = ()
    blocking: bool = False


def provider_and_model(cmd_ctx: CommandContext, override: str = "") -> tuple[str, str]:
    """Resolve the effective provider:model binding for context intake."""

    if override.strip():
        if ":" in override:
            provider, model = override.split(":", 1)
            return provider.strip().lower(), model.strip()
        runtime = LLMRuntime(cmd_ctx.config)
        default_provider, _ = runtime.default_binding()
        return default_provider, override.strip()
    runtime = LLMRuntime(cmd_ctx.config)
    return runtime.default_binding()


def persist_context_source(
    cmd_ctx: CommandContext,
    path: Path,
    *,
    inspect_result: ContextFileIngestResult,
    domain_override: str = "",
    scope_override: str = "",
) -> dict[str, object]:
    """Persist one inspected source under `vault/sources/` and record it in SQLite."""

    runtime = cmd_ctx.runtime
    repo = cmd_ctx.repo
    existing = repo.find_context_source(str(path))
    source_id = str(existing.get("id")) if existing is not None else generate_internal_id()
    source_dir = runtime.vault_path / "sources" / source_id
    source_dir.mkdir(parents=True, exist_ok=True)

    stored_path = source_dir / f"original{path.suffix.lower()}"
    inspect_json_path = source_dir / "ingest-result.json"
    shutil.copy2(path, stored_path)
    payload = inspect_result.model_dump(mode="json")
    inspect_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if str(payload.get("source_utility", "")) == "mixed_archive":
        manifest_path = source_dir / "archive-manifest.txt"
        parsed_names = [
            str(item.get("filename", ""))
            for item in payload.get("parsed_files", [])
            if isinstance(item, dict)
        ]
        manifest_path.write_text("\n".join(parsed_names) + ("\n" if parsed_names else ""), encoding="utf-8")

    domain_resolution = payload.get("domain_resolution", {})
    record = {
        "id": source_id,
        "filename": path.name,
        "original_path": str(path),
        "stored_path": str(stored_path),
        "normalized_path": str(inspect_json_path),
        "mime_type": str(
            (payload.get("parsed_files") or payload.get("failed_files") or [{}])[0].get(
                "mime_type",
                "application/octet-stream",
            )
        ),
        "canonical_class": str(
            (payload.get("parsed_files") or payload.get("failed_files") or [{}])[0].get(
                "canonical_class",
                "unknown",
            )
        ),
        "source_utility": str(payload.get("source_utility", "unknown")),
        "scope_mode": str(payload.get("scope_mode", "unclear")),
        "domain_id": domain_resolution.get("domain_id"),
        "domain_name": domain_override or domain_resolution.get("domain_name") or domain_resolution.get("new_domain_name"),
        "scope_boundary": scope_override or domain_resolution.get("scope_boundary") or "",
        "source_ref": f"vault://sources/{source_id}/{path.name}",
        "ingest_result": payload,
    }
    if existing is not None:
        return repo.update_context_source(record)
    return repo.create_context_source(record)


def ingest_context_source(
    cmd_ctx: CommandContext,
    path: Path,
    *,
    model_override: str = "",
    domain_override: str = "",
    scope_override: str = "",
    dryrun: bool,
) -> tuple[dict[str, object], ContextFileIngestResult]:
    """Inspect one source and persist it unless dry-run mode is active."""

    provider, model = provider_and_model(cmd_ctx, model_override)
    result = inspect_context_files([path], provider=provider, model=model, dryrun=dryrun)
    payload = result.model_dump(mode="json")
    if dryrun:
        domain_resolution = payload.get("domain_resolution", {})
        return {
            "id": f"dryrun:{path.name}",
            "filename": path.name,
            "original_path": str(path),
            "stored_path": "",
            "normalized_path": "",
            "mime_type": "",
            "canonical_class": "",
            "source_utility": payload.get("source_utility", "unknown"),
            "scope_mode": payload.get("scope_mode", "unclear"),
            "domain_id": domain_resolution.get("domain_id"),
            "domain_name": domain_override or domain_resolution.get("domain_name") or domain_resolution.get("new_domain_name"),
            "scope_boundary": scope_override or domain_resolution.get("scope_boundary") or "",
            "source_ref": f"dryrun://{path.name}",
            "ingest_result": payload,
        }, result
    return persist_context_source(
        cmd_ctx,
        path,
        inspect_result=result,
        domain_override=domain_override,
        scope_override=scope_override,
    ), result


def prepare_context_scope(
    ctx: typer.Context,
    direct_paths: list[Path],
    *,
    model_override: str = "",
) -> PreparedContextScope:
    """Prepare direct or locked context for a learning invocation."""

    cached = ctx.obj.get("_prepared_context_scope")
    if cached is not None and not direct_paths:
        return cached

    cmd_ctx = CommandContext.from_typer(ctx)
    if not direct_paths:
        locked = cmd_ctx.repo.get_locked_context()
        prepared = PreparedContextScope(scope=locked)
        ctx.obj["_prepared_context_scope"] = prepared
        return prepared

    dryrun = bool(ctx.obj.get("dryrun", False))
    sources: list[dict[str, object]] = []
    results: list[ContextFileIngestResult] = []
    messages: list[str] = []

    for path in direct_paths:
        source, result = ingest_context_source(
            cmd_ctx,
            path,
            model_override=model_override,
            dryrun=dryrun,
        )
        plan = plan_context_file_response(result)
        if plan.action == "ask_scope_clarification" and sys.stdin.isatty():
            question = result.scope_clarification.suggested_question or "Which part of this source should define the learning scope?"
            answer = str(typer.prompt(question, default="", show_default=False)).strip()
            if answer:
                result.scope_clarification.needed = False
                result.scope_mode = "reference_only" if "reference" in answer.lower() else "corpus_first"
                result.domain_resolution.scope_boundary = answer
                if not dryrun:
                    source = persist_context_source(cmd_ctx, path, inspect_result=result)
        sources.append(source)
        results.append(result)
        plan = plan_context_file_response(result)
        if plan.user_message:
            messages.append(plan.user_message)

    plans = [plan_context_file_response(result) for result in results]
    can_answer_any = any(plan.can_answer for plan in plans)
    blocking = any(plan.action == "ask_scope_clarification" for plan in plans) or not can_answer_any

    primary = sources[0] if sources else {}
    scope = active_context_from_sources(
        [str(source.get("source_ref", "")) for source in sources if str(source.get("source_ref", "")).strip()],
        label=str(primary.get("domain_name") or primary.get("filename") or "context"),
        domain_id=str(primary.get("domain_id", "") or "") or None,
        scope_mode=str(primary.get("scope_mode", "unclear")),
        scope_boundary=str(primary.get("scope_boundary", "")),
        locked=False,
    ) if sources else None
    prepared = PreparedContextScope(
        scope=scope,
        sources=tuple(sources),
        results=tuple(results),
        messages=tuple(messages),
        blocking=blocking,
    )
    ctx.obj["_prepared_context_scope"] = prepared
    return prepared


def session_active_context_scope(session) -> ActiveContextScope | None:
    """Deserialize the active context scope stored in session metadata."""

    generated = dict(getattr(session, "generated_names", {}) or {})
    payload = generated.get("active_context_scope")
    if not isinstance(payload, dict):
        return None
    try:
        return ActiveContextScope.model_validate(payload)
    except Exception:
        return None


def attach_active_context(entity, scope: ActiveContextScope | None) -> None:
    """Persist one active context scope into an entity's generated_names."""

    if scope is None:
        return
    generated = dict(getattr(entity, "generated_names", {}) or {})
    generated["active_context_scope"] = scope.model_dump(mode="json")
    entity.generated_names = generated


def context_prompt_contract(scope: ActiveContextScope | None) -> str:
    """Return a prompt-safe scope contract for learning drafts."""

    if scope is None or scope.mode == "none":
        return ""
    source_refs = ", ".join(scope.source_refs[:6])
    boundary = scope.scope_boundary or "Stay within the uploaded source scope unless the learner asks for outside material."
    return (
        "Active context scope contract:\n"
        f"- Mode: {scope.mode}\n"
        f"- Locked: {'yes' if scope.locked else 'no'}\n"
        f"- Scope mode: {scope.scope_mode}\n"
        f"- Boundary: {boundary}\n"
        f"- Source refs: {source_refs}\n"
        "- Treat these parsed/uploaded files as authoritative for this request.\n"
        "- Do not silently widen the scope beyond these sources.\n"
        "- If outside material is needed, label it as outside uploaded source scope.\n"
    )


def raise_for_blocking_context(prepared: PreparedContextScope) -> None:
    """Stop the current learning invocation when context intake requires action first."""

    if not prepared.blocking:
        return
    message = "\n\n".join(item for item in prepared.messages if item.strip())
    if message:
        typer.echo(message)
    raise typer.Exit(code=1)


def compatibility_only_message(result: ContextFileIngestResult) -> str:
    """Return the compatibility stop message for one intake result."""

    return compatibility_message(result)
