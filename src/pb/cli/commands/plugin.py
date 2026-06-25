# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Plugin management stub -- Phase 27 implementation."""
import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def plugin_callback(ctx: typer.Context):
    """Manage ProductiveBrain plugins."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo("pb plugin -- not yet implemented. Coming in Phase 27.")


@app.command("list")
def plugin_list():
    """List installed plugins."""
    typer.echo("Not yet implemented. Coming in Phase 27.")


@app.command("enable")
def plugin_enable(
    name: str = typer.Argument(..., help="Plugin name"),
):
    """Enable a plugin."""
    typer.echo("Not yet implemented. Coming in Phase 27.")
