# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Initialization commands for first-time setup and LLM configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from pb.cli.helpers import _interactive_pick, confirm_choice, prompt_text
from pb.llm.gemini import FLASH_MODEL
from pb.storage.config import (
    create_default_config,
    ensure_config_dir,
    get_config,
    get_config_path,
    load_config,
    save_config,
    set_default_model_binding,
    set_config_value,
)
from pb.vault import VAULT_SCHEMA
from pb.core.anki_bootstrap import ANKI_AUTO_OPEN_PREF

_INIT_DEFAULT_MODEL = "gemini-3-flash-preview"

app = typer.Typer(no_args_is_help=False, invoke_without_command=True)

_DIR_ACTION_CURRENT = "__current__"
_DIR_ACTION_UP = "__up__"
_DIR_ACTION_MANUAL = "__manual__"
_DIR_ACTION_CANCEL = "__cancel__"
_DIR_ACTION_PARENT = "__parent__"
_VAULT_SOURCE_EXISTING = "existing"
_VAULT_SOURCE_CREATE = "create"
_VAULT_SOURCE_PATH = "path"


def _browse_root() -> Path:
    """Return the default interactive directory browser root."""
    return Path.home().expanduser().resolve()


def _windows_drive_roots() -> list[Path]:
    if os.name != "nt":
        return []
    roots: list[Path] = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = Path(f"{letter}:/")
        if drive.exists():
            roots.append(drive)
    return roots


def _is_obsidian_vault(path: Path) -> bool:
    """Best-effort check for an existing Obsidian vault root."""
    return path.is_dir() and (path / ".obsidian").exists()


def _sorted_subdirectories(directory: Path) -> list[Path]:
    """Return child directories with likely vaults first."""
    entries: list[Path] = []
    try:
        for entry in directory.iterdir():
            if entry.is_dir():
                entries.append(entry)
    except OSError:
        return []

    return sorted(entries, key=lambda path: (not _is_obsidian_vault(path), path.name.lower()))


def _pick_vault_source() -> Optional[str]:
    """Choose between connecting an existing vault and creating a new one."""
    labels = [
        "Enter a vault path",
        "Connect to an existing vault",
        "Create a new vault",
    ]
    result = _interactive_pick(labels, "How should pb init set up your vault?", multi=False)
    if not result:
        return None
    return [_VAULT_SOURCE_PATH, _VAULT_SOURCE_EXISTING, _VAULT_SOURCE_CREATE][result[0]]


def _browse_for_directory(
    *,
    start_path: Path,
    use_current_label: str,
    title: str,
    manual_prompt: str,
) -> Optional[Path]:
    """Interactive directory navigator.

    Each child selection descends into that folder and re-renders the listing,
    which matches the desired `cd` + `ls` feel.
    """
    current = start_path.expanduser().resolve()

    while True:
        choices: list[tuple[str, str]] = [
            (_DIR_ACTION_CURRENT, use_current_label),
        ]
        if os.name == "nt":
            for drive in _windows_drive_roots():
                if drive != current and current.drive != drive.drive:
                    choices.append((str(drive), f"{drive.drive}\\"))
        if current.parent != current:
            choices.append((_DIR_ACTION_PARENT, ".."))

        for child in _sorted_subdirectories(current):
            suffix = "  [vault]" if _is_obsidian_vault(child) else ""
            choices.append((str(child), f"{child.name}{suffix}"))

        choices.extend(
            [
                (_DIR_ACTION_MANUAL, "Enter path manually"),
                (_DIR_ACTION_CANCEL, "Cancel"),
            ]
        )

        header = f"{title}\n  Current path: {current}"
        result = _interactive_pick([label for _, label in choices], header, multi=False)
        if not result:
            return None

        selected = choices[result[0]][0]
        if selected == _DIR_ACTION_CURRENT:
            return current
        if selected == _DIR_ACTION_PARENT:
            current = current.parent
            continue
        if selected == _DIR_ACTION_CANCEL:
            return None
        if selected == _DIR_ACTION_MANUAL:
            typed = prompt_text(manual_prompt, default=str(current))
            if not typed:
                continue
            typed_path = Path(typed).expanduser()
            if not typed_path.exists() or not typed_path.is_dir():
                typer.echo(f"Directory not found: {typed_path}", err=True)
                continue
            current = typed_path.resolve()
            continue

        current = Path(selected)


def _prompt_vault_path(label: str, *, default: str = "") -> Optional[Path]:
    while True:
        typed = prompt_text(label, default=default)
        if not typed:
            return None
        typed_path = Path(typed).expanduser()
        if not typed_path.exists() or not typed_path.is_dir():
            typer.echo(f"Directory not found: {typed_path}", err=True)
            continue
        return typed_path.resolve()


def _choose_existing_vault_path(start_path: Path) -> Optional[Path]:
    """Browse to an existing vault directory."""
    return _browse_for_directory(
        start_path=start_path,
        use_current_label="Select this folder as the vault",
        title="Connect to an existing vault",
        manual_prompt="Existing vault path",
    )


def _choose_new_vault_path(start_path: Path, *, default_name: str) -> Optional[Path]:
    """Choose a parent directory, then name the new vault folder."""
    parent = _browse_for_directory(
        start_path=start_path,
        use_current_label="Select this parent folder",
        title="Choose where to create the new vault",
        manual_prompt="Parent directory",
    )
    if parent is None:
        return None

    suggested_name = (default_name or "main").strip() or "main"
    while True:
        folder_name = prompt_text("New vault folder name", default=suggested_name)
        if not folder_name:
            typer.echo("Vault folder name is required.", err=True)
            continue

        candidate = (parent / folder_name).expanduser()
        if candidate.exists() and not candidate.is_dir():
            typer.echo(f"Cannot use file path as a vault: {candidate}", err=True)
            continue
        return candidate


def _scaffold_preview(vault_path: Path, quarantine_folder: str) -> list[str]:
    entries = list(VAULT_SCHEMA)
    if quarantine_folder not in entries:
        entries.append(quarantine_folder)
    return entries


def _ensure_scaffold(vault_path: Path, quarantine_folder: str) -> None:
    for relative in _scaffold_preview(vault_path, quarantine_folder):
        (vault_path / relative).mkdir(parents=True, exist_ok=True)


def init_command(
    *,
    non_interactive: bool = False,
    vault_name: str = "main",
    vault_path: str = "",
    provider: str = "gemini",
    model: str = _INIT_DEFAULT_MODEL,
    interaction_mode: str = "guided",
    scaffold: bool = True,
    yes: bool = False,
    embeddings: bool = False,
    allow_anki_auto_open: bool = False,
) -> None:
    """Create or refresh a ProductiveBrain configuration."""
    if embeddings:
        _run_embeddings_init()
        return

    config_path = get_config_path()
    existing = config_path.exists()

    if existing and non_interactive and not yes:
        typer.echo(
            f"Config already exists at {config_path}. Re-run with `--yes` to replace it in non-interactive mode.",
            err=True,
        )
        raise typer.Exit(code=40)

    if existing and not non_interactive:
        typer.echo(f"Config exists: {config_path}")
        if not confirm_choice("Replace the current ProductiveBrain config?", default=False):
            typer.echo("No changes made.")
            raise typer.Exit(code=0)

    selected_vault_path = vault_path
    selected_vault_name = vault_name
    # Provider and interaction mode are not prompted interactively — always gemini/guided.
    selected_provider = (provider or "gemini").strip().lower()
    selected_model = model or _INIT_DEFAULT_MODEL
    selected_interaction = interaction_mode or "guided"
    selected_anki_auto_open = bool(allow_anki_auto_open)

    if not non_interactive:
        if not selected_vault_path:
            source = _pick_vault_source()
            if source is None:
                typer.echo("No changes made.")
                raise typer.Exit(code=0)

            if source == _VAULT_SOURCE_PATH:
                typed_path = _prompt_vault_path("Vault path")
                if typed_path is None:
                    typer.echo("No changes made.")
                    raise typer.Exit(code=0)
                selected_vault_path = str(typed_path)
            elif source == _VAULT_SOURCE_EXISTING:
                existing_path = _choose_existing_vault_path(_browse_root())
                if existing_path is None:
                    typer.echo("No changes made.")
                    raise typer.Exit(code=0)
                selected_vault_path = str(existing_path)
            else:
                new_path = _choose_new_vault_path(_browse_root(), default_name=vault_name or "main")
                if new_path is None:
                    typer.echo("No changes made.")
                    raise typer.Exit(code=0)
                selected_vault_path = str(new_path)

        # Prompt 2: model — pre-filled with the locked Flash endpoint; user input takes precedence
        selected_model = prompt_text("Default model", default=selected_model) or _INIT_DEFAULT_MODEL
        selected_anki_auto_open = confirm_choice(
            "Allow pb to automatically open Anki when needed for review sync?",
            default=False,
        )

    if not selected_vault_path:
        typer.echo("Vault path is required.", err=True)
        raise typer.Exit(code=40)

    vault_abs = Path(selected_vault_path).expanduser()
    vault_abs.mkdir(parents=True, exist_ok=True)

    content = create_default_config(
        str(vault_abs),
        vault_name=selected_vault_name,
        provider=selected_provider,
        model=selected_model,
        interaction_mode=selected_interaction,
    )

    ensure_config_dir()
    config_path.write_text(content)

    cfg = load_config(config_path, force_reload=True)
    prefs = dict(getattr(cfg, "preferences", {}) or {})
    prefs[ANKI_AUTO_OPEN_PREF] = selected_anki_auto_open
    cfg.preferences = prefs
    save_config(cfg, path=config_path)
    cfg = load_config(config_path, force_reload=True)
    quarantine_folder = cfg.vaults[selected_vault_name].quarantine_folder

    if scaffold:
        _ensure_scaffold(vault_abs, quarantine_folder)
        scaffold_created = True
    else:
        scaffold_created = False

    typer.echo(f"Config created: {config_path}")
    typer.echo(f"Active vault:   {selected_vault_name} -> {vault_abs}")
    typer.echo(f"Default model:  {selected_provider}:{selected_model}")
    if not scaffold_created:
        typer.echo("Use `pb vault scaffold` later to create the recommended folders.")
    if selected_anki_auto_open:
        typer.echo("Anki auto-open: approved")
    else:
        typer.echo("Anki auto-open: not approved")


@app.callback(invoke_without_command=True)
def init_root(
    ctx: typer.Context,
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Run without prompts."),
    vault_name: str = typer.Option("main", "--vault-name", help="Vault profile name."),
    vault_path: str = typer.Option("", "--vault-path", help="Path to the Obsidian vault."),
    provider: str = typer.Option("gemini", "--provider", help="Default provider to configure."),
    model: str = typer.Option(_INIT_DEFAULT_MODEL, "--model", help="Default model ID (default: gemini-3-flash-preview)."),
    interaction_mode: str = typer.Option("guided", "--interaction-mode", help="guided | batch | terse | advanced"),
    scaffold: bool = typer.Option(True, "--scaffold/--no-scaffold", help="Create the recommended vault folders."),
    yes: bool = typer.Option(False, "--yes", help="Replace existing config or accept scaffold creation."),
    embeddings: bool = typer.Option(
        False, "--embeddings",
        help="Build embedding index for all vault notes (requires sqlite-vec)",
    ),
    allow_anki_auto_open: bool = typer.Option(
        False,
        "--allow-anki-auto-open",
        help="Approve automatic Anki launch for non-interactive setup.",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    init_command(
        non_interactive=non_interactive,
        vault_name=vault_name,
        vault_path=vault_path,
        provider=provider,
        model=model,
        interaction_mode=interaction_mode,
        scaffold=scaffold,
        yes=yes,
        embeddings=embeddings,
        allow_anki_auto_open=allow_anki_auto_open,
    )


@app.command("llm")
def init_llm(
    provider: str = typer.Option("gemini", "--provider", help="Provider: gemini|openai|anthropic|openrouter"),
    backend: str = typer.Option("auto", "--backend", help="Legacy Gemini backend: auto|aistudio|vertex"),
    model: str = typer.Option(_INIT_DEFAULT_MODEL, "--model", help="Default model ID"),
    api_key_env: str = typer.Option("", "--api-key-env", help="Optional API key env var override"),
    base_url: str = typer.Option("", "--base-url", help="Optional custom base URL"),
) -> None:
    """Compatibility helper to update the configured LLM provider."""
    config_path = get_config_path()
    if not config_path.exists():
        typer.echo("Config not found. Run `pb init` first.", err=True)
        raise typer.Exit(code=53)

    provider_name = (provider or "gemini").strip().lower()
    set_config_value("llm", "provider", provider_name)
    set_config_value("llm", "backend", (backend or "auto").strip().lower())
    set_config_value("llm", "default_model", model.strip())
    set_config_value("llm", "prompt_template_version", "v3")
    set_config_value("llm", "require_llm_for_core_workflows", True)

    cfg = get_config(force_reload=True)
    payload = cfg.model_dump(mode="python", exclude_none=True)
    providers = payload.setdefault("providers", {})
    provider_payload = providers.setdefault(provider_name, {})
    provider_payload["default_model"] = model.strip()
    if api_key_env:
        provider_payload["api_key_env"] = api_key_env.strip()
    if base_url:
        provider_payload["base_url"] = base_url.strip()
    save_config(type(cfg)(**payload), path=config_path)
    set_default_model_binding(f"{provider_name}:{model.strip()}", path=config_path)

    typer.echo("LLM configuration saved.")
    typer.echo(f"Default provider: {provider_name}")
    typer.echo(f"Default model: {model.strip()}")


def _run_embeddings_init() -> None:
    """Build vec_note_embeddings for all existing vault notes."""
    from pb.vault import get_vault_path
    from pb.vault.embeddings import EmbeddingStore, EmbeddingUnavailableError
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    try:
        vault_path = get_vault_path()
        store = EmbeddingStore(vault_path)
        store.ensure_schema()
    except EmbeddingUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    knowledge_dir = vault_path / "knowledge"
    note_files = sorted(knowledge_dir.rglob("*.md")) if knowledge_dir.exists() else []
    note_files = [f for f in note_files if not f.name.startswith("_")]

    console = Console()
    embedded = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Embedding notes...", total=len(note_files))
        for note_file in note_files:
            slug = note_file.stem
            content = note_file.read_text(errors="replace")
            store.store_embedding(slug, content[:2048])
            embedded += 1
            progress.advance(task_id)

    console.print(f"[green]Embedded {embedded} notes.[/green]")
