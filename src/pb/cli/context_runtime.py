# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared runtime helpers for context-file ingest, scope inheritance, and prompts."""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import typer

from pb.cli.context import CommandContext
from pb.core.context_classifier import classify_context_source
from pb.core.context_file_intake import (
    ActiveContextScope,
    CONTENT_PLACEHOLDER_SUMMARIES,
    ContextFileIngestResult,
    SourceBundleItem,
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


SOURCE_REF_PREFIX = "vault://source/"
OLD_SOURCE_REF_PREFIX = "vault://sources/"


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


def _safe_source_filename(filename: str) -> str:
    """Return a readable filesystem-safe source filename."""

    clean = re.sub(r"[\\/]+", "-", (filename or "").strip())
    clean = re.sub(r"[\x00-\x1f:]+", "-", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" .")
    if not clean:
        clean = "source"
    stem = Path(clean).stem.strip(" .") or "source"
    suffix = Path(clean).suffix.lower()
    return f"{stem}{suffix}"


def _source_ref(filename: str) -> str:
    return f"{SOURCE_REF_PREFIX}{filename}"


def _source_filename_from_ref(source_ref: str) -> str:
    ref = (source_ref or "").strip()
    if ref.startswith(SOURCE_REF_PREFIX):
        return Path(ref.removeprefix(SOURCE_REF_PREFIX)).name
    if ref.startswith(OLD_SOURCE_REF_PREFIX):
        return Path(ref).name
    return ""


def _taken_source_filenames(repo, *, excluding_source_id: str = "") -> set[str]:
    taken: set[str] = set()
    for row in repo.list_context_sources():
        if excluding_source_id and str(row.get("id", "")) == excluding_source_id:
            continue
        ref_name = _source_filename_from_ref(str(row.get("source_ref", "")))
        stored_name = Path(str(row.get("stored_path", ""))).name
        if ref_name:
            taken.add(ref_name)
        if stored_name and stored_name != "original":
            taken.add(stored_name)
    return taken


def _dedupe_source_filename(source_dir: Path, filename: str, *, taken: set[str], current_path: Path | None = None) -> str:
    safe = _safe_source_filename(filename)
    stem = Path(safe).stem
    suffix = Path(safe).suffix
    candidate = safe
    index = 2
    while True:
        candidate_path = source_dir / candidate
        occupied = candidate in taken or (candidate_path.exists() and (current_path is None or candidate_path != current_path))
        if not occupied:
            return candidate
        candidate = f"{stem}-{index}{suffix}"
        index += 1


def _context_result_payload(result: ContextFileIngestResult) -> dict[str, object]:
    return result.model_dump(mode="json")


def _payload_with_stored_ref(payload: dict[str, object], *, filename: str, source_ref: str) -> dict[str, object]:
    updated = dict(payload)
    parsed_files = []
    for item in list(updated.get("parsed_files", []) or []):
        if not isinstance(item, dict):
            continue
        parsed = dict(item)
        if parsed.get("filename") == filename and not str(parsed.get("source_ref", "")).startswith("archive://"):
            parsed["source_ref"] = source_ref
        parsed_files.append(parsed)
    updated["parsed_files"] = parsed_files
    return updated


def _payload_has_pdf_failure(payload: dict[str, object]) -> bool:
    for item in list(payload.get("failed_files", []) or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("extension", "")).lower() == "pdf" or str(item.get("canonical_class", "")) == "document.pdf":
            return True
    return False


def _old_source_path(runtime, row: dict[str, object]) -> Path | None:
    candidates = [
        Path(str(row.get("stored_path", ""))),
        Path(str(row.get("original_path", ""))),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists() and candidate.is_file():
            return candidate
    source_id = str(row.get("id", ""))
    filename = str(row.get("filename", ""))
    suffix = Path(filename).suffix.lower()
    legacy = runtime.vault_path / "sources" / source_id / f"original{suffix}"
    if legacy.exists() and legacy.is_file():
        return legacy
    return None


def _cleanup_legacy_source_path(runtime, old_path: Path, new_path: Path) -> None:
    legacy_root = runtime.vault_path / "sources"
    try:
        old_path.relative_to(legacy_root)
    except ValueError:
        return
    if old_path == new_path:
        return
    try:
        old_path.unlink(missing_ok=True)
    except OSError:
        return
    parent = old_path.parent
    for _ in range(2):
        if parent == legacy_root.parent:
            break
        try:
            parent.rmdir()
        except OSError:
            break
        if parent == legacy_root:
            break
        parent = parent.parent


def normalize_context_source_storage(cmd_ctx: CommandContext) -> int:
    """Normalize older context source rows into readable `vault/source/` storage."""

    runtime = cmd_ctx.runtime
    repo = cmd_ctx.repo
    source_dir = runtime.vault_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    changed_refs: dict[str, str] = {}
    normalized_count = 0

    for row in repo.list_context_sources():
        source_id = str(row.get("id", ""))
        filename = _safe_source_filename(str(row.get("filename", "")) or "source")
        old_ref = str(row.get("source_ref", ""))
        old_path = _old_source_path(runtime, row)
        stored_path = Path(str(row.get("stored_path", "")))
        already_new = old_ref.startswith(SOURCE_REF_PREFIX) and stored_path.parent == source_dir
        if already_new and not _payload_has_pdf_failure(dict(row.get("ingest_result", {}) or {})):
            continue

        current_path = stored_path if stored_path.parent == source_dir else None
        target_name = _dedupe_source_filename(
            source_dir,
            filename,
            taken=_taken_source_filenames(repo, excluding_source_id=source_id),
            current_path=current_path,
        )
        target_path = source_dir / target_name
        if old_path is not None and old_path != target_path:
            shutil.copy2(old_path, target_path)
            _cleanup_legacy_source_path(runtime, old_path, target_path)

        new_ref = _source_ref(target_path.name)
        provider, model = provider_and_model(cmd_ctx)
        raw_payload = dict(row.get("ingest_result", {}) or {})
        if target_path.exists() and (not raw_payload or _payload_has_pdf_failure(raw_payload)):
            result = inspect_context_files([target_path], provider=provider, model=model, dryrun=False)
            payload = _context_result_payload(result)
        else:
            try:
                payload = _context_result_payload(ContextFileIngestResult.model_validate(raw_payload))
            except Exception:
                payload = raw_payload
        payload = _payload_with_stored_ref(payload, filename=target_path.name, source_ref=new_ref)
        metadata_path = source_dir / f"{target_path.stem}.ingest-result.json"
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

        updated = {
            **row,
            "filename": target_path.name,
            "stored_path": str(target_path),
            "normalized_path": str(metadata_path),
            "source_ref": new_ref,
            "ingest_result": payload,
        }
        repo.update_context_source(updated)
        if old_ref and old_ref != new_ref:
            changed_refs[old_ref] = new_ref
        normalized_count += 1

    if changed_refs:
        for bundle in repo.list_source_bundles():
            touched = False
            updated_items: list[SourceBundleItem] = []
            for item in bundle.items:
                source_ref = changed_refs.get(item.source_ref, item.source_ref)
                if source_ref != item.source_ref:
                    touched = True
                    item = item.model_copy(update={"source_ref": source_ref})
                updated_items.append(item)
            if touched:
                for item in updated_items:
                    repo.add_source_bundle_item(item)
                bundle.source_refs = [item.source_ref for item in updated_items]
                repo.update_source_bundle(bundle)

        locked = repo.get_locked_context()
        if locked is not None:
            new_refs = [changed_refs.get(ref, ref) for ref in locked.source_refs]
            if new_refs != locked.source_refs:
                repo.set_locked_context(locked.model_copy(update={"source_refs": new_refs}))

    return normalized_count


def persist_context_source(
    cmd_ctx: CommandContext,
    path: Path,
    *,
    inspect_result: ContextFileIngestResult,
    domain_override: str = "",
    scope_override: str = "",
) -> dict[str, object]:
    """Persist one inspected source under `vault/source/` and record it in SQLite."""

    runtime = cmd_ctx.runtime
    repo = cmd_ctx.repo
    normalize_context_source_storage(cmd_ctx)
    existing = repo.find_context_source(str(path))
    source_id = str(existing.get("id")) if existing is not None else generate_internal_id()
    source_dir = runtime.vault_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    existing_stored = Path(str(existing.get("stored_path", ""))) if existing is not None else None
    current_path = existing_stored if existing_stored is not None and existing_stored.parent == source_dir else None
    source_filename = _dedupe_source_filename(
        source_dir,
        path.name,
        taken=_taken_source_filenames(repo, excluding_source_id=source_id),
        current_path=current_path,
    )
    stored_path = source_dir / source_filename
    source_ref = _source_ref(source_filename)
    inspect_json_path = source_dir / f"{Path(source_filename).stem}.ingest-result.json"
    shutil.copy2(path, stored_path)
    payload = _context_result_payload(inspect_result)
    payload = _payload_with_stored_ref(payload, filename=path.name, source_ref=source_ref)
    inspect_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if str(payload.get("source_utility", "")) == "mixed_archive":
        manifest_path = source_dir / f"{Path(source_filename).stem}.archive-manifest.txt"
        parsed_names = [
            str(item.get("filename", ""))
            for item in payload.get("parsed_files", [])
            if isinstance(item, dict)
        ]
        manifest_path.write_text("\n".join(parsed_names) + ("\n" if parsed_names else ""), encoding="utf-8")

    domain_resolution = payload.get("domain_resolution", {})
    record = {
        "id": source_id,
        "filename": source_filename,
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
        "source_ref": source_ref,
        "ingest_result": payload,
    }
    if existing is not None:
        return repo.update_context_source(record)
    return repo.create_context_source(record)


def enrich_result_with_classification(
    cmd_ctx: CommandContext,
    result: ContextFileIngestResult,
    *,
    domain_override: str = "",
    scope_override: str = "",
) -> None:
    """Replace filename-guessed domain/scope with real LLM content classification.

    Best-effort and fail-safe: when offline, unconfigured, or on any error the
    ``result`` is left exactly as the deterministic intake produced it, so
    ``pb context add`` never breaks because of the classifier.
    """

    if domain_override and scope_override:
        return
    primary = None
    for parsed in result.parsed_files:
        summary = (parsed.content_summary or "").strip()
        if summary and summary not in CONTENT_PLACEHOLDER_SUMMARIES:
            primary = parsed
            break
    if primary is None:
        return
    try:
        classification = classify_context_source(
            LLMRuntime(cmd_ctx.config),
            filename=primary.filename,
            excerpt=primary.content_summary,
        )
    except Exception:
        classification = None
    if classification is None:
        return

    topic = classification.topic_summary.strip()
    if topic:
        primary.content_summary = topic
    resolution = result.domain_resolution
    if not domain_override:
        domain = classification.domain_name.strip()
        if domain:
            resolution.domain_name = domain
            if resolution.new_domain_name:
                resolution.new_domain_name = domain
    if not scope_override:
        boundary = classification.scope_boundary.strip()
        if boundary:
            resolution.scope_boundary = boundary


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
    enrich_result_with_classification(
        cmd_ctx,
        result,
        domain_override=domain_override,
        scope_override=scope_override,
    )
    payload = _context_result_payload(result)
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
