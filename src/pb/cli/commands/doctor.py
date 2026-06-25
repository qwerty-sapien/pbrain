# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Environment diagnostics for the ProductiveBrain CLI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import typer

from pydantic import BaseModel

from pb.cli.console import get_console, get_err_console
from pb.core.agent_weights import scorer_health
from pb.core.exceptions import ExitCode
from pb.llm.runtime import DraftGenerationError, LLMRuntime
from pb.storage.config import (
    get_config,
    get_config_path,
    get_data_dir,
    get_quarantine_path,
    get_vault_path,
    save_config,
)
from pb.storage.database import get_db_path


@dataclass
class Check:
    label: str
    status: str
    detail: str
    required: bool = True


STATUS_RENDER = {
    "OK": "[success]OK[/]",
    "FAIL": "[error]FAIL[/]",
    "WARN": "[warn]WARN[/]",
    "NA": "[warn]NA[/]",
}


def _llm_setup_hint(health) -> str:
    """Return a short setup hint for the configured LLM state."""

    if not health.provider or not health.default_model:
        return "Set a model with `pb model use gemini:gemini-3-flash-preview`, then run `pb doctor --llm`."
    if health.credential_source == "none":
        return "Set `GEMINI_API_KEY`, then run `pb doctor --llm` to test a real request."
    return "Run `pb doctor --llm` to test a real request."


def _run_checks(checks: list[Check], *, json_out: bool, console, extra_json: Optional[dict] = None) -> int:
    """Render checks and return exit code. Required failures → 53, optional-only → 0."""
    if json_out:
        payload = {
            "checks": [
                {"label": c.label, "status": c.status, "ok": c.status == "OK", "detail": c.detail, "required": c.required}
                for c in checks
            ]
        }
        if extra_json:
            payload.update(extra_json)
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.rule("[header]Doctor[/]")
        for c in checks:
            tag = STATUS_RENDER.get(c.status, f"[dim]{c.status}[/]")
            console.print(f"{tag} {c.label}: {c.detail}")

    required_failures = sum(1 for c in checks if c.status == "FAIL" and c.required)
    return ExitCode.CONFIG_ERROR if required_failures > 0 else ExitCode.SUCCESS


def _verify_tiers(config, runtime, *, json_out: bool, console, debug: bool = False) -> int:
    """Probe each distinct model-role tier binding and report OK/FAIL per tier.

    De-duplicates identical bindings (probe once, cost control — T-07-06).
    On failure, surfaces ONLY the broken tier's re-set hint — never a full
    reconfigure (D-04, T-07-07).  Renders only probe.message, never raw keys
    (T-07-05).
    """
    roles = config.model_roles
    role_names = ["default", "planner", "reviewer", "recall", "fast", "fast_inference", "namer"]

    # Build binding → list-of-roles mapping; skip empty bindings
    binding_roles: dict[str, list[str]] = {}
    for role in role_names:
        binding = getattr(roles, role, "") or ""
        if binding:
            binding_roles.setdefault(binding, []).append(role)

    if not binding_roles:
        checks = [
            Check(
                "tier bindings",
                "WARN",
                "No model-role bindings configured — run `pb model set-role <role> <provider:model>`",
                required=False,
            )
        ]
        return _run_checks(checks, json_out=json_out, console=console)

    checks: list[Check] = []
    probe_results: list[dict] = []

    for binding, roles_list in binding_roles.items():
        role_summary = ",".join(roles_list)
        tier_label = (
            f"tier {roles_list[0]} ({role_summary})" if len(roles_list) > 1 else f"tier {roles_list[0]}"
        )

        probe = runtime.live_probe(model=binding, timeout=12)

        status = "OK" if probe.available else "FAIL"
        # Render only sanitised user message — never raw key (T-07-05)
        detail = f"{probe.provider}:{probe.model} — {probe.message}"
        checks.append(Check(tier_label, status, detail))

        probe_results.append({
            "binding": binding,
            "roles": roles_list,
            "provider": probe.provider,
            "model": probe.model,
            "available": probe.available,
            "category": probe.category,
            "message": probe.message,
            **({"debug_message": probe.debug_message} if debug and probe.debug_message else {}),
        })

    extra_json = {"mode": "tiers", "tiers": probe_results} if json_out else None
    exit_code = _run_checks(checks, json_out=json_out, console=console, extra_json=extra_json)

    # Surgical failure hints — only broken tiers, never "reconfigure everything" (D-04)
    if not json_out and exit_code != 0:
        for result in probe_results:
            if not result["available"]:
                role_summary = ",".join(result["roles"])
                first_role = result["roles"][0]
                console.print(
                    f"[warn]Tier '{role_summary}' failed:[/] {result['message']}. "
                    f"Re-set just this tier with: pb model set-role {first_role} <provider:model>"
                )
        if debug:
            for result in probe_results:
                if not result["available"] and result.get("debug_message"):
                    console.print(f"[dim]Debug ({result['binding']}):[/] {result['debug_message']}")
        else:
            console.print("[dim]Use `pb doctor --tiers --debug` to show raw provider details for failed tiers.[/]")

    return exit_code


class _PingDraft(BaseModel):
    ok: bool


def _run_draft_probe(runtime: LLMRuntime) -> tuple[bool, str, str]:
    """Try generating a minimal structured draft. Returns (ok, message, detail)."""
    try:
        result = runtime.generate_draft(
            _PingDraft,
            'Return {"ok": true}.',
            source_scope="doctor:draft-probe",
            timeout=20,
            max_output_tokens=64,
        )
        ok_val = getattr(result.payload, "ok", None)
        if ok_val is True:
            return True, "structured draft OK", f"model={result.model}"
        return False, "draft parsed but payload unexpected", f"payload={result.payload!r}"
    except DraftGenerationError as exc:
        details = exc.debug_details()
        return False, exc.to_user_message().splitlines()[0], details


def doctor_command(
    ctx: typer.Context,
    *,
    fix: bool = False,
    json_out: bool = False,
    llm: bool = False,
    tiers: bool = False,
    debug: bool = False,
) -> None:
    """Check the LLM runtime plus local vault/state dependencies."""
    console = get_console()
    err_console = get_err_console()

    try:
        config = ctx.obj.get("config") if ctx.obj and ctx.obj.get("config") is not None else get_config(force_reload=True)
    except FileNotFoundError:
        path = get_config_path()
        err_console.print("[error]Config not loaded.[/]")
        console.print(f"[dim]Expected config path:[/] {path}")
        console.print("[dim]Run `pb init` to create a ProductiveBrain configuration.[/]")
        raise typer.Exit(code=ExitCode.CONFIG_ERROR)

    runtime = LLMRuntime(config)
    health = runtime.health()
    probe = runtime.live_probe() if llm else None

    vault_path = get_vault_path(config)
    data_dir = get_data_dir(config)
    quarantine_path = get_quarantine_path(config)
    db_path = get_db_path()

    if fix:
        save_config(config)
        data_dir.mkdir(parents=True, exist_ok=True)
        if vault_path.exists():
            quarantine_path.mkdir(parents=True, exist_ok=True)
        if not json_out:
            console.print("[dim]Applied safe local fixes where possible (config rewrite, profile dirs, quarantine folder).[/]")

    # --- Tiers-only mode (D-04: per-tier verification) ---
    if tiers:
        exit_code = _verify_tiers(config, runtime, json_out=json_out, console=console, debug=debug)
        if exit_code != 0:
            raise typer.Exit(code=exit_code)
        return

    # --- LLM-only mode ---
    if llm:
        draft_ok, draft_message, draft_detail = (False, "skipped — provider unavailable", "") if probe is None or not probe.available else _run_draft_probe(runtime)
        checks = [
            Check("provider configured", "OK" if health.provider else "FAIL", health.provider or "missing"),
            Check("credentials present", "OK" if health.credential_source != "none" else "FAIL", health.credential_source or "missing"),
            Check("default model set", "OK" if health.default_model else "FAIL", health.default_model or "missing"),
            Check("LLM live probe", "OK" if probe is not None and probe.available else "FAIL",
                  probe.message if probe is not None else "Probe unavailable"),
            Check("structured draft", "OK" if draft_ok else "FAIL", draft_message),
        ]
        extra = None
        if json_out and probe is not None:
            extra = {
                "mode": "llm",
                "probe": {
                    "available": probe.available, "provider": probe.provider,
                    "backend": probe.backend, "model": probe.model,
                    "category": probe.category, "message": probe.message,
                    "http_status": probe.http_status,
                    **({"debug_message": probe.debug_message} if debug and probe.debug_message else {}),
                },
                "draft_probe": {"ok": draft_ok, "message": draft_message, "detail": draft_detail},
            }
        exit_code = _run_checks(checks, json_out=json_out, console=console, extra_json=extra)

        if not json_out:
            if probe is not None and debug and probe.debug_message:
                console.print(f"[dim]Debug details:[/] {probe.debug_message}")
            if debug and draft_detail:
                console.print(f"[dim]Draft probe detail:[/] {draft_detail}")
            if not draft_ok and not debug:
                console.print(f"[dim]Next: {_llm_setup_hint(health)}[/]")
                console.print("[dim]Use `pb doctor --llm --debug` to show raw provider details.[/]")

        if exit_code != 0:
            raise typer.Exit(code=exit_code)
        return

    # --- Full diagnostic mode ---
    # Required checks: vault and SQLite health
    checks: list[Check] = [
        Check("vault reachable", "OK" if vault_path.exists() else "FAIL", str(vault_path)),
        Check("vault writable",
              "OK" if vault_path.exists() and vault_path.is_dir() and os.access(vault_path, os.W_OK) else "FAIL",
              str(vault_path)),
        Check("quarantine ready", "OK" if quarantine_path.exists() or vault_path.exists() else "FAIL", str(quarantine_path)),
        Check("SQLite path ready", "OK" if db_path.parent.exists() else "FAIL", str(db_path)),
        Check("data dir ready", "OK" if data_dir.exists() else "FAIL", str(data_dir)),
    ]

    # Optional checks: LLM and Anki — warnings only, never block exit 0
    llm_setup_detail = (
        f"{health.message} {_llm_setup_hint(health)}"
        if health.available
        else f"{health.message} {_llm_setup_hint(health)}"
    )
    checks.extend([
        Check("provider configured", "OK" if health.provider else "WARN", health.provider or "missing", required=False),
        Check("credentials present", "OK" if health.credential_source != "none" else "WARN",
              health.credential_source or "missing", required=False),
        Check("default model set", "OK" if health.default_model else "WARN", health.default_model or "missing", required=False),
        Check("structured output", "OK" if health.structured_output else "WARN",
              "available" if health.structured_output else "missing", required=False),
        Check("LLM request setup", "WARN", llm_setup_detail, required=False),
    ])

    try:
        from pb.vault.anki_client import is_anki_available
        anki_ok = is_anki_available()
        checks.append(Check(
            "Anki integration",
            "OK" if anki_ok else "WARN",
            (
                "AnkiConnect online"
                if anki_ok
                else "AnkiConnect offline. Open Anki with the AnkiConnect add-on, or export CSV with `pb anki export --csv-only`."
            ),
            required=False,
        ))
    except Exception as exc:
        checks.append(Check("Anki integration", "NA", str(exc), required=False))

    agent_weight_status = scorer_health()
    checks.extend([
        Check(
            "agent-weight tables",
            "OK" if agent_weight_status["tables_present"] else "WARN",
            "ready" if agent_weight_status["tables_present"] else "missing",
            required=False,
        ),
        Check(
            "agent-weight cache",
            "OK" if agent_weight_status["cache_readable"] else "WARN",
            (
                f"{agent_weight_status['cache_rows']} cache rows / "
                f"{agent_weight_status['event_rows']} events"
            ),
            required=False,
        ),
        Check(
            "agent-weight drift",
            "OK" if not agent_weight_status["anomalies"] else "WARN",
            ", ".join(agent_weight_status["anomalies"]) if agent_weight_status["anomalies"] else "none",
            required=False,
        ),
    ])

    exit_code = _run_checks(
        checks,
        json_out=json_out,
        console=console,
        extra_json={"agent_weights": agent_weight_status} if json_out else None,
    )

    if not json_out and exit_code != 0:
        console.print("\n[dim]Run `pb init`, `pb model doctor`, or `pb doctor --llm` to repair runtime setup.[/]")

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def doctor_root(
    ctx: typer.Context,
    fix: bool = typer.Option(False, "--fix", help="Repair safe local setup issues when possible."),
    llm: bool = typer.Option(False, "--llm", help="Run a cheap live LLM probe against the configured default model."),
    tiers: bool = typer.Option(False, "--tiers", help="Verify each configured model-role tier's endpoint + API key (surgical re-prompt on failure)."),
    debug: bool = typer.Option(False, "--debug", help="Show raw provider details for the live LLM probe."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run environment diagnostics."""
    doctor_command(ctx, fix=fix, json_out=json_out, llm=llm, tiers=tiers, debug=debug)
