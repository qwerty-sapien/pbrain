# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Context source, bundle, and lock management commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pb.cli.context import CommandContext
from pb.cli.context_runtime import (
    ingest_context_source,
    provider_and_model,
)
from pb.core.context_file_intake import (
    ActiveContextScope,
    SourceBundle,
    SourceBundleItem,
    active_context_from_bundle,
    active_context_from_sources,
    compatibility_message,
    inspect_context_files,
    plan_context_file_response,
    summarize_context_label,
)


app = typer.Typer(no_args_is_help=True)
bundle_app = typer.Typer(no_args_is_help=True)
app.add_typer(bundle_app, name="bundle")


def _show_result_summary(payload: dict[str, object]) -> None:
    typer.echo(f"Status: {payload.get('status')}")
    typer.echo(f"Source utility: {payload.get('source_utility')}")
    typer.echo(f"Scope mode: {payload.get('scope_mode')}")
    domain_resolution = payload.get("domain_resolution", {})
    if isinstance(domain_resolution, dict):
        domain_name = domain_resolution.get("domain_name") or domain_resolution.get("new_domain_name")
        if domain_name:
            typer.echo(f"Domain: {domain_name}")
        if domain_resolution.get("scope_boundary"):
            typer.echo(f"Scope boundary: {domain_resolution['scope_boundary']}")
    parsed = payload.get("parsed_files", [])
    failed = payload.get("failed_files", [])
    typer.echo(f"Parsed files: {len(parsed) if isinstance(parsed, list) else 0}")
    typer.echo(f"Failed files: {len(failed) if isinstance(failed, list) else 0}")


def _source_label(row: dict[str, object]) -> str:
    return f"{row['id']}  {row['filename']}  [{row['source_utility']}]  {row.get('domain_name') or '-'}  {row['scope_mode']}"


def _latest_context_scope(cmd_ctx: CommandContext) -> ActiveContextScope | None:
    bundles = cmd_ctx.repo.list_source_bundles()
    if bundles:
        return active_context_from_bundle(bundles[0], locked=True)
    sources = cmd_ctx.repo.list_context_sources()
    if not sources:
        return None
    source = sources[0]
    return active_context_from_sources(
        [str(source["source_ref"])],
        label=str(source.get("domain_name") or source.get("filename") or "context"),
        domain_id=str(source.get("domain_id", "") or "") or None,
        scope_mode=str(source.get("scope_mode", "unclear")),
        scope_boundary=str(source.get("scope_boundary", "")),
        locked=True,
    )


@app.command("inspect")
def context_inspect(
    ctx: typer.Context,
    file: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True),
    model: str = typer.Option("", "--model", help="Optional provider:model override"),
) -> None:
    """Inspect one file and show deterministic context-intake results."""
    cmd_ctx = CommandContext.from_typer(ctx)
    provider, resolved_model = provider_and_model(cmd_ctx, model)
    result = inspect_context_files([file], provider=provider, model=resolved_model, dryrun=True)
    payload = result.model_dump(mode="json")
    _show_result_summary(payload)
    plan = plan_context_file_response(result)
    if plan.user_message:
        typer.echo("")
        typer.echo(plan.user_message)


@app.command("add")
def context_add(
    ctx: typer.Context,
    file: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True),
    domain: str = typer.Option("", "--domain", help="Optional explicit domain name"),
    scope: str = typer.Option("", "--scope", help="Optional explicit scope boundary"),
    model: str = typer.Option("", "--model", help="Optional provider:model override"),
    force: bool = typer.Option(False, "--force", help="Add even when context is locked"),
) -> None:
    """Persist one durable source file and its intake metadata."""
    cmd_ctx = CommandContext.from_typer(ctx)
    locked = cmd_ctx.repo.get_locked_context()
    if locked is not None and not force:
        typer.echo(f"Context is locked: {summarize_context_label(locked)}")
        typer.echo("Run `pb context unlock` before adding another source, or pass --force.")
        raise typer.Exit(code=1)
    dryrun = bool(ctx.obj.get("dryrun", False))
    source, result = ingest_context_source(
        cmd_ctx,
        file,
        model_override=model,
        domain_override=domain,
        scope_override=scope,
        dryrun=dryrun,
    )
    payload = result.model_dump(mode="json")
    _show_result_summary(payload)
    plan = plan_context_file_response(result)
    typer.echo("")
    if dryrun:
        typer.echo(f"Dry run source ref: {source['source_ref']}")
    else:
        typer.echo(f"Stored source: {source['id']} -> {source['source_ref']}")
    if plan.user_message:
        typer.echo(plan.user_message)


@app.command("list")
def context_list(ctx: typer.Context) -> None:
    """List durable source files."""
    cmd_ctx = CommandContext.from_typer(ctx)
    rows = cmd_ctx.repo.list_context_sources()
    if not rows:
        typer.echo("No context sources stored yet.")
        return
    for row in rows:
        typer.echo(_source_label(row))


@app.command("show")
def context_show(
    ctx: typer.Context,
    source_id: str = typer.Argument("", help="Source id, source_ref, original path, or filename"),
) -> None:
    """Show one stored source file record."""
    cmd_ctx = CommandContext.from_typer(ctx)
    if not source_id:
        locked = cmd_ctx.repo.get_locked_context()
        if locked is not None:
            typer.echo(f"Locked context: {summarize_context_label(locked)}")
            typer.echo(f"Source refs: {len(locked.source_refs)}")
            if locked.scope_boundary:
                typer.echo(f"Boundary: {locked.scope_boundary}")
            return
        rows = cmd_ctx.repo.list_context_sources()
        if not rows:
            typer.echo("No context sources stored yet.")
            return
        typer.echo("Stored context sources:")
        for row in rows:
            typer.echo(_source_label(row))
        return
    row = cmd_ctx.repo.find_context_source(source_id)
    if row is None:
        raise typer.BadParameter(f"No stored source matched `{source_id}`.")
    typer.echo(json.dumps(row, indent=2, ensure_ascii=True))


@app.command("remove")
def context_remove(
    ctx: typer.Context,
    source_id: str = typer.Argument(..., help="Source id, source_ref, original path, or filename"),
) -> None:
    """Remove one stored source file record."""
    cmd_ctx = CommandContext.from_typer(ctx)
    removed = cmd_ctx.repo.delete_context_source(source_id)
    if removed is None:
        raise typer.BadParameter(f"No stored source matched `{source_id}`.")
    locked = cmd_ctx.repo.get_locked_context()
    if locked is not None and str(removed.get("source_ref", "")) in set(locked.source_refs):
        cmd_ctx.repo.clear_locked_context()
        typer.echo("Context lock cleared because the locked source was removed.")
    typer.echo(f"Removed source: {removed.get('filename')}")


@bundle_app.command("create")
def bundle_create(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Bundle name"),
    files: list[Path] = typer.Argument(None, exists=True, dir_okay=False, resolve_path=True),
) -> None:
    """Create a named source bundle from one or more files."""
    cmd_ctx = CommandContext.from_typer(ctx)
    if cmd_ctx.repo.get_source_bundle_by_name(name) is not None:
        raise typer.BadParameter(f"Bundle `{name}` already exists.")
    stored_sources: list[dict[str, object]] = []
    for path in files or []:
        stored, _ = ingest_context_source(cmd_ctx, path, dryrun=bool(ctx.obj.get("dryrun", False)))
        stored_sources.append(stored)
    primary = stored_sources[0] if stored_sources else {}
    bundle = SourceBundle(
        name=name,
        domain_id=str(primary.get("domain_id", "") or "") or None,
        domain_name=str(primary.get("domain_name", "") or "") or None,
        scope_mode=str(primary.get("scope_mode", "unclear")),
        scope_boundary=str(primary.get("scope_boundary", "")),
        source_refs=[str(item.get("source_ref", "")) for item in stored_sources if str(item.get("source_ref", "")).strip()],
        items=[
            SourceBundleItem(
                bundle_id="",
                source_id=str(source["id"]),
                position=index,
                source_ref=str(source["source_ref"]),
                filename=str(source["filename"]),
            )
            for index, source in enumerate(stored_sources)
        ],
    )
    bundle.items = [item.model_copy(update={"bundle_id": bundle.id}) for item in bundle.items]
    cmd_ctx.repo.create_source_bundle(bundle)
    typer.echo(f"Created bundle `{bundle.name}` with {len(bundle.items)} source(s).")


@bundle_app.command("add")
def bundle_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Bundle name"),
    files: list[Path] = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True),
) -> None:
    """Add one or more files to an existing bundle."""
    cmd_ctx = CommandContext.from_typer(ctx)
    bundle = cmd_ctx.repo.get_source_bundle_by_name(name)
    if bundle is None:
        raise typer.BadParameter(f"Bundle `{name}` does not exist.")
    existing_source_ids = {item.source_id for item in bundle.items}
    added = 0
    for path in files:
        stored, _ = ingest_context_source(cmd_ctx, path, dryrun=bool(ctx.obj.get("dryrun", False)))
        if str(stored["id"]) in existing_source_ids:
            continue
        cmd_ctx.repo.add_source_bundle_item(
            SourceBundleItem(
                bundle_id=bundle.id,
                source_id=str(stored["id"]),
                position=len(bundle.items) + added,
                source_ref=str(stored["source_ref"]),
                filename=str(stored["filename"]),
            )
        )
        added += 1
    typer.echo(f"Added {added} source(s) to `{bundle.name}`.")


@bundle_app.command("remove")
def bundle_remove(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Bundle name"),
    files: list[str] = typer.Argument(..., help="Source ids, source refs, or filenames to remove"),
) -> None:
    """Remove one or more stored sources from a bundle."""
    cmd_ctx = CommandContext.from_typer(ctx)
    bundle = cmd_ctx.repo.get_source_bundle_by_name(name)
    if bundle is None:
        raise typer.BadParameter(f"Bundle `{name}` does not exist.")
    lookup = {item.source_id: item.source_id for item in bundle.items}
    lookup.update({item.source_ref: item.source_id for item in bundle.items})
    lookup.update({item.filename: item.source_id for item in bundle.items})
    lookup.update({Path(item.filename).name: item.source_id for item in bundle.items})
    normalized_files = [item if item in lookup else Path(item).name for item in files]
    source_ids = [lookup[item] for item in normalized_files if item in lookup]
    removed = cmd_ctx.repo.remove_source_bundle_sources(bundle.id, source_ids)
    typer.echo(f"Removed {removed} source(s) from `{bundle.name}`.")


@bundle_app.command("list")
def bundle_list(ctx: typer.Context) -> None:
    """List all stored source bundles."""
    cmd_ctx = CommandContext.from_typer(ctx)
    bundles = cmd_ctx.repo.list_source_bundles()
    if not bundles:
        typer.echo("No bundles stored yet.")
        return
    for bundle in bundles:
        typer.echo(
            f"{bundle.id}  {bundle.name}  "
            f"[{bundle.scope_mode}]  sources={len(bundle.items)}"
        )


@bundle_app.command("show")
def bundle_show(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Bundle name"),
) -> None:
    """Show one stored source bundle."""
    cmd_ctx = CommandContext.from_typer(ctx)
    bundle = cmd_ctx.repo.get_source_bundle_by_name(name)
    if bundle is None:
        raise typer.BadParameter(f"Bundle `{name}` does not exist.")
    typer.echo(json.dumps(bundle.model_dump(mode="json"), indent=2, ensure_ascii=True))


def _lock_scope_from_ref(cmd_ctx: CommandContext, ref: str) -> ActiveContextScope:
    bundle = cmd_ctx.repo.get_source_bundle_by_name(ref)
    if bundle is not None:
        return active_context_from_bundle(bundle, locked=True)
    source = cmd_ctx.repo.find_context_source(ref)
    if source is not None:
        return active_context_from_sources(
            [str(source["source_ref"])],
            label=str(source.get("domain_name") or source.get("filename") or "context"),
            domain_id=str(source.get("domain_id", "") or "") or None,
            scope_mode=str(source.get("scope_mode", "unclear")),
            scope_boundary=str(source.get("scope_boundary", "")),
            locked=True,
        )
    raise typer.BadParameter(f"No bundle or source matched `{ref}`.")


@app.command("lock")
def context_lock(
    ctx: typer.Context,
    ref: str = typer.Argument("", help="Bundle name or source id/ref; omit to lock the newest bundle/source"),
) -> None:
    """Lock the active context to one bundle or source."""
    cmd_ctx = CommandContext.from_typer(ctx)
    scope = _lock_scope_from_ref(cmd_ctx, ref) if ref else _latest_context_scope(cmd_ctx)
    if scope is None:
        raise typer.BadParameter("No context source or bundle is available to lock.")
    cmd_ctx.repo.set_locked_context(scope)
    typer.echo(f"Locked context: {summarize_context_label(scope)}")


@app.command("unlock")
def context_unlock(ctx: typer.Context) -> None:
    """Clear the persisted locked context."""
    cmd_ctx = CommandContext.from_typer(ctx)
    cmd_ctx.repo.clear_locked_context()
    typer.echo("Context unlocked.")


@app.command("status")
def context_status(ctx: typer.Context) -> None:
    """Show the currently locked context, if any."""
    cmd_ctx = CommandContext.from_typer(ctx)
    scope = cmd_ctx.repo.get_locked_context()
    if scope is None:
        typer.echo("No context is currently locked.")
        return
    typer.echo(f"Locked: {summarize_context_label(scope)}")
    typer.echo(f"Mode: {scope.mode}")
    typer.echo(f"Scope mode: {scope.scope_mode}")
    typer.echo(f"Bundle: {scope.source_bundle_id or '-'}")
    typer.echo(f"Source refs: {len(scope.source_refs)}")
    if scope.scope_boundary:
        typer.echo(f"Boundary: {scope.scope_boundary}")


@app.command("doctor")
def context_doctor(
    ctx: typer.Context,
    file: Path | None = typer.Argument(None, exists=True, dir_okay=False, resolve_path=True),
    model: str = typer.Option("", "--model", help="Optional provider:model override"),
) -> None:
    """Explain whether the current model can use this file cleanly."""
    cmd_ctx = CommandContext.from_typer(ctx)
    if file is None:
        locked = cmd_ctx.repo.get_locked_context()
        source_count = len(cmd_ctx.repo.list_context_sources())
        bundle_count = len(cmd_ctx.repo.list_source_bundles())
        typer.echo("Context doctor")
        typer.echo(f"Sources: {source_count}")
        typer.echo(f"Bundles: {bundle_count}")
        typer.echo(f"Locked: {summarize_context_label(locked) if locked is not None else 'none'}")
        return
    provider, resolved_model = provider_and_model(cmd_ctx, model)
    result = inspect_context_files([file], provider=provider, model=resolved_model, dryrun=True)
    payload = result.model_dump(mode="json")
    _show_result_summary(payload)
    typer.echo("")
    typer.echo(compatibility_message(result) if result.status == "failed" else plan_context_file_response(result).user_message or "The current route can use this file set.")


def _context_inference_guidance(ctx: typer.Context, request_words: list[str] | None = None) -> None:
    """Explain how to apply stored context through a learning command."""
    cmd_ctx = CommandContext.from_typer(ctx)
    request = " ".join(request_words or []).strip()
    locked = cmd_ctx.repo.get_locked_context()
    sources = cmd_ctx.repo.list_context_sources()
    label = ""
    if locked is not None:
        label = summarize_context_label(locked)
    elif sources:
        label = str(sources[0].get("domain_name") or sources[0].get("filename") or "stored context")

    typer.echo("Use context through a learning command so the answer becomes a study or practice step.")
    if request:
        typer.echo(f"Request: {request}")
    if locked is not None:
        topic = request or "your question"
        typer.echo(f"Locked context: {label}")
        typer.echo(f"Try: pb learn \"{topic}\"")
        typer.echo("To switch sources, run `pb context unlock` and then `pb context lock <source-or-bundle>`.")
        return
    if not sources:
        typer.echo("No stored or locked context is available yet.")
        typer.echo("First run `pb context add <file>`, or inspect file fit with `pb context inspect <file>`.")
        return
    first = sources[0]
    topic = request or "your question"
    source_id = str(first.get("id") or first.get("filename") or "").strip()
    typer.echo(f"Stored context: {label}")
    if source_id:
        typer.echo(f"Lock it once: pb context lock {source_id}")
    typer.echo(f"Then try: pb learn \"{topic}\"")
    typer.echo("For a new file, check fit first with: pb context doctor <file>")


@app.command("infer")
def context_infer(
    ctx: typer.Context,
    request_words: list[str] = typer.Argument(None, help="Question or learning request to apply to the active context"),
) -> None:
    """Show how to apply stored context to a learning request."""
    _context_inference_guidance(ctx, request_words)


@app.command("ask")
def context_ask(
    ctx: typer.Context,
    request_words: list[str] = typer.Argument(None, help="Question or learning request to apply to the active context"),
) -> None:
    """Alias for `pb context infer`."""
    _context_inference_guidance(ctx, request_words)
