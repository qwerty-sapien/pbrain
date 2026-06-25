# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""LLM provider and model-role management commands."""

from __future__ import annotations

import json

import typer

from pb.cli.console import get_console
from pb.cli.helpers import prompt_text
from pb.cli.pickers import pick_single_choice
from pb.llm.gemini import FLASH_LITE_MODEL, FLASH_MODEL, PRO_MODEL
from pb.llm.runtime import LLMRuntime
from pb.storage.config import (
    get_config,
    set_default_model_binding,
    set_model_role,
    upsert_provider,
)


app = typer.Typer(no_args_is_help=False, invoke_without_command=True)

_KNOWN_PROVIDER_MODELS = {
    "gemini": [FLASH_LITE_MODEL, FLASH_MODEL, PRO_MODEL],
    "vertex": [FLASH_LITE_MODEL, FLASH_MODEL, PRO_MODEL],
    "openai": ["gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"],
    "anthropic": ["claude-sonnet-4-0", "claude-opus-4-0", "claude-3-7-sonnet-latest"],
}


def _runtime(ctx: typer.Context) -> LLMRuntime:
    config = ctx.obj.get("config") if ctx.obj else get_config()
    return LLMRuntime(config)


def _configured_models_for_provider(provider: str, config) -> list[str]:
    clean = (provider or "").strip().lower()
    models = list(_KNOWN_PROVIDER_MODELS.get(clean, []))
    provider_cfg = config.providers.get(clean)
    if provider_cfg and provider_cfg.default_model:
        models.insert(0, provider_cfg.default_model)
    for binding in config.model_roles.model_dump(mode="python").values():
        bound_provider, model = str(binding).split(":", 1) if ":" in str(binding) else ("gemini", str(binding))
        if bound_provider.strip().lower() == clean and model.strip():
            models.append(model.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for model in models:
        lowered = model.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(model)
    return deduped


def _interactive_model_selection(ctx: typer.Context) -> None:
    config = ctx.obj.get("config") if ctx.obj else get_config()
    current_provider = str(config.llm.provider or "gemini").strip().lower() or "gemini"
    provider = pick_single_choice(
        [(name, name) for name in dict.fromkeys([current_provider, *config.providers.keys(), *_KNOWN_PROVIDER_MODELS.keys()])],
        title="Choose provider",
        text="Pick a provider, then choose a known model or type any exact model ID.",
    )
    if not provider:
        return

    known_models = _configured_models_for_provider(provider, config)
    if known_models:
        selected_model = pick_single_choice(
            [(model, model) for model in known_models],
            title="Choose model",
            text="Select a known model, or type an exact model ID.",
            allow_inline_edit=True,
            inline_prompt="Model ID",
        )
    else:
        selected_model = prompt_text("Model ID", default="")
    selected_model = str(selected_model or "").strip()
    if not selected_model:
        return

    binding = f"{provider}:{selected_model}"
    cfg = set_default_model_binding(binding)
    typer.echo(f"Preferred model set to {binding}")
    typer.echo(f"Fast role: {cfg.model_roles.fast}")
    typer.echo(f"Fast inference role: {cfg.model_roles.fast_inference}")


@app.callback(invoke_without_command=True)
def model_root(ctx: typer.Context) -> None:
    """Open the interactive model selector when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        _interactive_model_selection(ctx)


@app.command("status")
def model_status(ctx: typer.Context, json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON")) -> None:
    """Show the configured default LLM runtime."""
    console = get_console()
    runtime = _runtime(ctx)
    health = runtime.health()
    payload = {
        "provider": health.provider,
        "backend": health.backend,
        "default_model": health.default_model,
        "credentials": health.credential_source,
        "structured_output": health.structured_output,
        "available": health.available,
        "message": health.message,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return

    console.rule("[header]Model Status[/]")
    console.print(f"Provider: {health.provider}")
    console.print(f"Backend: {health.backend}")
    console.print(f"Default model: {health.default_model}")
    console.print(f"Credentials: {health.credential_source}")
    console.print(f"Structured output: {'yes' if health.structured_output else 'no'}")
    console.print(f"Request path configured: {'yes' if health.available else 'no'}")
    console.print(f"[dim]{health.message}[/]")


@app.command("list")
def model_list(ctx: typer.Context, json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON")) -> None:
    """List configured providers and role bindings."""
    runtime = _runtime(ctx)
    config = runtime.config
    payload = {
        "providers": {
            name: provider.model_dump(mode="json")
            for name, provider in config.providers.items()
        },
        "roles": runtime.role_bindings(),
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return

    console = get_console()
    console.rule("[header]Configured Models[/]")
    for name, provider in config.providers.items():
        console.print(
            f"{name}: default={provider.default_model or '(unset)'} "
            f"env={provider.api_key_env or '(unset)'} "
            f"base={provider.base_url or '(default)'}"
        )
    console.print("")
    console.print("Roles:")
    for role, binding in runtime.role_bindings().items():
        console.print(f"  {role}: {binding}")


@app.command("add", hidden=True)
def model_add(
    provider: str = typer.Argument(..., help="Provider name: gemini, openai, anthropic, openrouter"),
    model: str = typer.Option(..., "--model", help="Default model ID for this provider"),
    api_key_env: str = typer.Option("", "--api-key-env", help="Env var containing the provider API key"),
    base_url: str = typer.Option("", "--base-url", help="Optional custom base URL"),
) -> None:
    """Add or update a configured provider."""
    cfg = upsert_provider(
        provider,
        api_key_env=api_key_env or None,
        default_model=model,
        base_url=base_url or None,
    )
    typer.echo(f"Configured provider {provider.strip().lower()} -> {cfg.providers[provider.strip().lower()].default_model}")


@app.command("use")
def model_use(
    binding: str = typer.Argument(..., help="Provider/model binding in provider:model form."),
) -> None:
    """Set the default provider:model binding used across ProductiveBrain."""
    set_default_model_binding(binding)
    typer.echo(f"Default model binding set to {binding}")


@app.command("roles", hidden=True)
def model_roles(
    role: str = typer.Option("", "--role", help="Optional role to change: default, planner, reviewer, recall, fast, fast_inference, namer"),
    binding: str = typer.Option("", "--binding", help="Provider/model binding in provider:model form"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Show or update model-role bindings."""
    if role and binding:
        set_model_role(role, binding)
        typer.echo(f"Updated {role} -> {binding}")
        return

    config = get_config()
    payload = {
        "default": config.model_roles.default,
        "planner": config.model_roles.planner,
        "reviewer": config.model_roles.reviewer,
        "recall": config.model_roles.recall,
        "fast": config.model_roles.fast,
        "fast_inference": config.model_roles.fast_inference,
        "namer": config.model_roles.namer,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return
    console = get_console()
    console.rule("[header]Model Roles[/]")
    for key, value in payload.items():
        console.print(f"{key}: {value}")


@app.command("doctor", hidden=True)
def model_doctor(ctx: typer.Context) -> None:
    """Check provider credentials and current default model binding."""
    runtime = _runtime(ctx)
    health = runtime.health()
    console = get_console()
    console.rule("[header]Model Doctor[/]")
    console.print(f"Provider: {health.provider}")
    console.print(f"Default model: {health.default_model}")
    console.print(f"Credentials: {health.credential_source}")
    console.print(f"Request path configured: {'yes' if health.available else 'no'}")
    console.print(f"[dim]{health.message}[/]")
    if not health.available:
        raise typer.Exit(code=53)
