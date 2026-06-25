"""File discovery across all pb-managed directories."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import typer

from pb.cli.console import get_console, get_err_console

app = typer.Typer(no_args_is_help=False, invoke_without_command=True)

PROTECTED_FILES = {"pb.db", "productivebrain.db", "config.toml"}


@dataclass
class FindQuery:
    mode: str  # "days", "since", "string", "interactive"
    value: Union[int, datetime, str, None]


def parse_query(raw: Optional[str]) -> FindQuery:
    if not raw:
        return FindQuery(mode="interactive", value=None)
    if raw.isdigit():
        return FindQuery(mode="days", value=int(raw))
    date_match = re.match(r"^(\d{2})-(\d{2})-(\d{2})$", raw)
    if date_match:
        day, month, year_short = date_match.groups()
        year = 2000 + int(year_short)
        return FindQuery(mode="since", value=datetime(year, int(month), int(day)))
    return FindQuery(mode="string", value=raw)


def collect_files(
    vault_path: Path, data_dir: Path,
    log_dir: Optional[Path] = None, config_path: Optional[Path] = None,
) -> list[Path]:
    files: list[Path] = []
    for root_dir in [vault_path, data_dir]:
        if root_dir and root_dir.exists():
            for path in root_dir.rglob("*"):
                if path.is_file() and not any(p.startswith(".") for p in path.relative_to(root_dir).parts):
                    files.append(path)
    if log_dir and log_dir.exists():
        for path in log_dir.rglob("*"):
            if path.is_file():
                files.append(path)
    if config_path and config_path.is_file():
        files.append(config_path)
    return files


def filter_by_days(files: list[Path], days: int) -> list[Path]:
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_ts = cutoff.timestamp()
    return [f for f in files if f.stat().st_mtime >= cutoff_ts]


def filter_by_since(files: list[Path], since: datetime) -> list[Path]:
    cutoff_ts = since.timestamp()
    return [f for f in files if f.stat().st_mtime >= cutoff_ts]


def filter_by_fzf(files: list[Path], query: str) -> list[Path]:
    if not shutil.which("fzf"):
        query_lower = query.lower()
        return [f for f in files if query_lower in f.name.lower() or query_lower in str(f).lower()]
    input_text = "\n".join(str(f) for f in files)
    try:
        result = subprocess.run(
            ["fzf", "--filter", query],
            input=input_text, capture_output=True, text=True, timeout=10,
        )
        return [Path(line) for line in result.stdout.strip().splitlines() if line]
    except (subprocess.TimeoutExpired, OSError):
        query_lower = query.lower()
        return [f for f in files if query_lower in str(f).lower()]


def interactive_fzf(files: list[Path]) -> list[Path]:
    if not shutil.which("fzf"):
        return files
    input_text = "\n".join(str(f) for f in files)
    try:
        result = subprocess.run(
            ["fzf", "--multi"],
            input=input_text, capture_output=False, text=True,
            timeout=120, stdout=subprocess.PIPE,
        )
        if result.returncode != 0:
            return []
        return [Path(line) for line in result.stdout.strip().splitlines() if line]
    except (subprocess.TimeoutExpired, OSError, KeyboardInterrupt):
        return []


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    for unit in ["K", "M", "G"]:
        size_bytes /= 1024
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
    return f"{size_bytes:.1f}T"


def display_files(
    files: list[Path], vault_path: Path, data_dir: Path,
    log_dir: Optional[Path] = None,
) -> None:
    console = get_console()
    if not files:
        console.print("[dim]No files found.[/]")
        return
    console.print(f"[dim]{'modified':<20} {'type':<8} {'size':>6}  path[/]")
    for f in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        ext = f.suffix.lstrip(".") or "file"
        size = format_size(stat.st_size)
        display_path = str(f)
        for label, base in [("vault", vault_path), ("data", data_dir), ("logs", log_dir)]:
            if base and f.is_relative_to(base):
                display_path = str(f.relative_to(base.parent))
                break
        console.print(f"{modified:<20} {ext:<8} {size:>6}  {display_path}")


def delete_files(files: list[Path]) -> tuple[int, int]:
    deleted = 0
    skipped = 0
    for f in files:
        if f.name in PROTECTED_FILES:
            skipped += 1
            continue
        try:
            f.unlink()
            deleted += 1
        except OSError:
            skipped += 1
    return deleted, skipped


@app.callback(invoke_without_command=True)
def find_command(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Days (number), date (DD-MM-YY), or search string"),
    delete: bool = typer.Option(False, "--delete", help="Delete all matched files (with confirmation)"),
):
    """Find files across all pb-managed directories."""
    runtime = ctx.obj.get("runtime")
    if runtime is None:
        get_err_console().print("[error]No active vault. Run `pb init` first.[/]")
        raise typer.Exit(code=1)

    from pb.storage.config import get_log_dir, get_config_path
    vault_path = runtime.vault_path
    data_dir = runtime.data_dir
    try:
        log_dir = get_log_dir(runtime.config)
    except Exception:
        log_dir = None
    try:
        config_path = get_config_path(None)
    except Exception:
        config_path = None

    all_files = collect_files(vault_path, data_dir, log_dir, config_path)
    parsed = parse_query(query)

    if parsed.mode == "days":
        matched = filter_by_days(all_files, parsed.value)
    elif parsed.mode == "since":
        matched = filter_by_since(all_files, parsed.value)
    elif parsed.mode == "string":
        matched = filter_by_fzf(all_files, parsed.value)
    elif parsed.mode == "interactive":
        matched = interactive_fzf(all_files)
        if not matched:
            return
    else:
        matched = all_files

    display_files(matched, vault_path, data_dir, log_dir)

    if not delete or not matched:
        return

    console = get_console()
    protected = [f for f in matched if f.name in PROTECTED_FILES]
    deletable = [f for f in matched if f.name not in PROTECTED_FILES]

    if protected:
        console.print(f"[dim]Skipping {len(protected)} protected file(s): {', '.join(f.name for f in protected)}[/]")
    if not deletable:
        console.print("[dim]No deletable files.[/]")
        return

    console.print(f"\n[bold]Delete {len(deletable)} file(s)?[/] [dim](protected files excluded)[/]")
    confirm = typer.confirm("Proceed?", default=False)
    if not confirm:
        console.print("[dim]Cancelled.[/]")
        return

    deleted, skipped = delete_files(deletable)
    console.print(f"[success]Deleted {deleted} file(s).[/]")
    if skipped:
        console.print(f"[dim]Skipped {skipped} file(s) (protected or locked).[/]")
