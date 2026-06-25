# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Configuration viewing and small updates."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import typer

from pb.cli.helpers import prompt_text
from pb.cli.pickers import pick_many_choices
from pb.core.agent_instruction_judge import (
    AgentInstructionPatchRecord,
    apply_agent_instruction_patch,
    list_agent_instruction_patches,
    revert_agent_instruction_patch,
    sweep_agent_instruction_judge,
    write_agent_instruction_digest,
)
from pb.core.agent_weights import list_agent_weights, set_weight_override
from pb.runtime import get_session_auto_yes, set_session_auto_yes
from pb.storage.config import get_config, get_config_path, set_config_value


app = typer.Typer(no_args_is_help=False, invoke_without_command=True)
session_app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context) -> None:
    """View the current resolved configuration."""
    if ctx.invoked_subcommand is not None:
        return
    config_show()


PRIMARY_CONFIG_KEYS = [
    "general.active_vault",
    "general.verbose",
    "interaction.mode",
    "ui.theme",
    "ui.plain_mode",
    "ui.content_width_ratio",
    "ui.max_content_width",
    "llm.provider",
    "llm.default_model",
    "llm.backend",
    "llm.long_model_timeout_seconds",
    "llm.auto_pro_fallback",
    "model_roles.default",
    "model_roles.fast_inference",
    "learning.promotion_threshold",
    "learning.decay_days_default",
    "learning.learnt_suggestion_threshold",
]


@app.command("show")
def config_show(json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON only")) -> None:
    """Show the current configuration and config path."""
    config = get_config()
    if json_out:
        typer.echo(json.dumps({
            "config_path": str(get_config_path()),
            "config": config.model_dump(mode="json"),
        }, indent=2))
        return
    typer.echo(f"Config path: {get_config_path()}")
    typer.echo(json.dumps(config.model_dump(mode="json"), indent=2))


def _split_config_key(key: str) -> tuple[str, str]:
    if "." not in key:
        raise ValueError("Use section.key format, for example llm.default_model")
    return key.split(".", 1)


def _flatten_scalar_config_fields(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in sorted(value.items()):
            child_prefix = f"{prefix}.{key}" if prefix else key
            rows.extend(_flatten_scalar_config_fields(item, child_prefix))
        return rows
    if isinstance(value, (str, bool, int, float)) or value is None:
        return [(prefix, value)]
    return []


def _display_config_value(value: Any) -> str:
    if value is None:
        return "(empty)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_config_value(raw: str, current: Any) -> Any:
    text = (raw or "").strip()
    if isinstance(current, bool):
        lowered = text.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError("Enter true/false, yes/no, on/off, or 1/0.")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if current is None and text.lower() == "none":
        return None
    return raw


def _interactive_config_updates(*, developer: bool = False) -> list[tuple[str, Any]]:
    config = get_config(force_reload=True)
    flattened = _flatten_scalar_config_fields(config.model_dump(mode="python", exclude_none=False))
    if not developer:
        allowed = set(PRIMARY_CONFIG_KEYS)
        flattened = [(key, value) for key, value in flattened if key in allowed]
    options = [
        (key, f"{key} = {_display_config_value(value)}")
        for key, value in flattened
        if key
    ]
    details = [
        f"Key: {key}\nCurrent value: {_display_config_value(value)}"
        for key, value in flattened
        if key
    ]
    selected = pick_many_choices(
        options,
        title="Config settings",
        text="Select one or more settings to update.",
        details=details,
    )
    if not selected:
        return []

    current_by_key = {key: value for key, value in flattened}
    updates: list[tuple[str, Any]] = []
    for key in selected:
        current = current_by_key.get(key)
        default_value = _display_config_value(current)
        raw = prompt_text(key, default="" if default_value == "(empty)" else default_value)
        try:
            parsed = _parse_config_value(raw, current)
        except ValueError as exc:
            typer.echo(f"{key}: {exc}", err=True)
            raise typer.Exit(code=40)
        updates.append((key, parsed))
    return updates


@app.command("set")
def config_set(
    key: str | None = typer.Argument(None, help="Config key in section.key form"),
    value: str | None = typer.Argument(None, help="New value"),
    developer: bool = typer.Option(False, "--developer", help="Show advanced/developer settings in the interactive picker."),
) -> None:
    """Set configuration values directly or choose them interactively in a TTY."""
    if key is None:
        if not sys.stdin.isatty():
            typer.echo(
                "Pass `section.key value`, or run `pb config set` in a terminal to choose settings interactively.",
                err=True,
            )
            raise typer.Exit(code=40)
        updates = _interactive_config_updates(developer=developer)
        if not updates:
            typer.echo("Cancelled.")
            return
        for selected_key, parsed in updates:
            section, field = _split_config_key(selected_key)
            try:
                set_config_value(section, field, parsed)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=40)
            typer.echo(f"Updated {section}.{field}")
        return

    try:
        section, field = _split_config_key(key)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=40)

    if value is None:
        if not sys.stdin.isatty():
            typer.echo("A value is required when not running interactively.", err=True)
            raise typer.Exit(code=40)
        current = get_config(force_reload=True).model_dump(mode="python", exclude_none=False)
        current_value = dict(_flatten_scalar_config_fields(current)).get(key)
        raw = prompt_text(key, default=_display_config_value(current_value))
        try:
            parsed = _parse_config_value(raw, current_value)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=40)
    else:
        current = dict(
            _flatten_scalar_config_fields(
                get_config(force_reload=True).model_dump(mode="python", exclude_none=False)
            )
        ).get(key)
        try:
            parsed = _parse_config_value(value, current)
        except ValueError:
            parsed = value

    try:
        set_config_value(section, field, parsed)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=40)
    typer.echo(f"Updated {section}.{field}")


def _render_agent_weights(*, json_out: bool) -> None:
    rows = list_agent_weights()
    if json_out:
        typer.echo(json.dumps({"agents": rows}, indent=2))
        return
    if not rows:
        typer.echo("No agent-weight data yet.")
        return
    for row in rows:
        score = f"{float(row.get('frecency_score', 0.0)):.3f}"
        override = row.get("override") or "-"
        updated_at = row.get("updated_at") or "-"
        typer.echo(f"{row['agent_id']}\t{score}\t{override}\t{updated_at}")


def _patch_record_to_dict(record: AgentInstructionPatchRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "agent_id": record.agent_id,
        "session_id": record.session_id,
        "status": record.status,
        "trigger_kind": record.trigger_kind,
        "confidence": record.confidence,
        "summary": record.summary,
        "instruction_patch": record.instruction_patch,
        "clarifying_question": record.clarifying_question,
        "evidence": list(record.evidence),
        "created_at": record.created_at,
        "applied_at": record.applied_at,
        "reverted_at": record.reverted_at,
    }


def _render_agent_instruction_patches(
    *,
    agent_id: str = "",
    json_out: bool,
) -> None:
    rows = list_agent_instruction_patches(agent_id=agent_id)
    if json_out:
        typer.echo(
            json.dumps(
                {"patches": [_patch_record_to_dict(record) for record in rows]},
                indent=2,
            )
        )
        return
    if not rows:
        typer.echo("No agent-instruction patches yet.")
        return
    for record in rows:
        summary = record.summary or record.clarifying_question or "-"
        typer.echo(
            f"{record.id}\t{record.agent_id}\t{record.status}\t"
            f"{record.confidence:.2f}\t{summary}"
        )


@app.command("agents")
def config_agents(
    action: str = typer.Argument(
        "list",
        help="Action: list, patches, sweep, pin, suppress, clear, apply, or revert",
    ),
    agent_id: str | None = typer.Argument(
        None,
        help="Agent id for pin/suppress/clear/patches/sweep, or patch id for apply/revert",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Inspect weights and manage reversible specialised-agent instruction patches."""
    normalized = (action or "list").strip().lower()
    if normalized == "list":
        _render_agent_weights(json_out=json_out)
        return
    if normalized == "patches":
        _render_agent_instruction_patches(agent_id=agent_id or "", json_out=json_out)
        return
    if normalized == "sweep":
        records = asyncio.run(sweep_agent_instruction_judge(agent_id=agent_id or ""))
        digest_path = write_agent_instruction_digest(
            Path(get_config().general.vault_path),
            records,
        )
        if json_out:
            typer.echo(
                json.dumps(
                    {
                        "digest_path": str(digest_path),
                        "patches": [_patch_record_to_dict(record) for record in records],
                    },
                    indent=2,
                )
            )
            return
        typer.echo(f"Agent judge digest: {digest_path}")
        if records:
            for record in records:
                typer.echo(
                    f"{record.id}\t{record.agent_id}\t{record.status}\t"
                    f"{record.confidence:.2f}\t{record.summary or '-'}"
                )
        else:
            typer.echo("No agent-instruction patches proposed.")
        return
    if normalized in {"apply", "revert"}:
        if not agent_id:
            typer.echo(f"`pb config agents {normalized}` needs a patch id.", err=True)
            raise typer.Exit(code=40)
        try:
            record = (
                apply_agent_instruction_patch(agent_id)
                if normalized == "apply"
                else revert_agent_instruction_patch(agent_id)
            )
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=40)
        if json_out:
            typer.echo(json.dumps(_patch_record_to_dict(record), indent=2))
            return
        typer.echo(f"{record.id}: {record.status} for {record.agent_id}")
        return
    if normalized not in {"pin", "suppress", "clear"}:
        typer.echo(
            "Action must be one of: list, patches, sweep, pin, suppress, clear, apply, revert.",
            err=True,
        )
        raise typer.Exit(code=40)
    if not agent_id:
        typer.echo(f"`pb config agents {normalized}` needs an agent id.", err=True)
        raise typer.Exit(code=40)

    override = None if normalized == "clear" else normalized
    stored = set_weight_override(agent_id, override)
    if json_out:
        typer.echo(json.dumps({"agent_id": agent_id, "override": stored}, indent=2))
        return
    label = stored or "cleared"
    typer.echo(f"{agent_id}: {label}")


@session_app.command("auto-yes")
def config_session_auto_yes(
    state: str = typer.Argument(..., help="Use 'on' to enable or 'off' to disable session-scoped auto-yes."),
) -> None:
    """Toggle session-scoped auto-yes for the current terminal session."""
    normalized = (state or "").strip().lower()
    if normalized not in {"on", "off"}:
        typer.echo("State must be 'on' or 'off'.", err=True)
        raise typer.Exit(code=40)
    enabled = normalized == "on"
    flag_path = set_session_auto_yes(enabled)
    status = "enabled" if enabled else "disabled"
    typer.echo(f"Session auto-yes {status}: {flag_path}")


@session_app.command("status")
def config_session_status() -> None:
    """Show current session-scoped CLI flags."""
    typer.echo(
        json.dumps(
            {
                "auto_yes": get_session_auto_yes(),
            },
            indent=2,
        )
    )


app.add_typer(session_app, name="session", help="Ephemeral per-terminal session settings")
