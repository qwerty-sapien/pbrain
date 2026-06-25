# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Per-vault evidence metric commands."""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

import typer

from pb.cli.helpers import confirm_choice
from pb.storage.database import get_connection
from pb.storage.repository import Repository


app = typer.Typer(no_args_is_help=True)


def _resolve_metric(metric_ref: str) -> tuple[str, str] | None:
    lowered = metric_ref.strip().lower()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name FROM metric_definitions ORDER BY name"
        ).fetchall()
    for row in rows:
        if row["id"] == metric_ref or row["id"].startswith(metric_ref):
            return row["id"], row["name"]
        if row["name"].lower() == lowered or row["name"].lower().startswith(lowered):
            return row["id"], row["name"]
    return None


@app.command("list")
def metric_list(json_out: bool = typer.Option(False, "--json")) -> None:
    """List available metric definitions."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, description, unit, domain FROM metric_definitions ORDER BY name"
        ).fetchall()
    payload = [
        {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "unit": row["unit"],
            "domain": row["domain"],
        }
        for row in rows
    ]
    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return
    if not payload:
        typer.echo("No metrics defined.")
        return
    for row in payload:
        typer.echo(f"{row['name']} [{row['id'][:8]}]")
        typer.echo(f"  {row['description']} ({row['unit'] or 'count'})")


@app.command("add")
def metric_add(
    name: str = typer.Argument(..., help="Metric name"),
    description: str = typer.Option("", "--description"),
    unit: str = typer.Option("", "--unit"),
    domain: str = typer.Option("", "--domain"),
) -> None:
    """Create a metric definition."""
    now = datetime.utcnow().isoformat()
    metric_id = str(uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO metric_definitions (id, name, description, unit, domain, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (metric_id, name, description, unit, domain, now, now),
        )
        conn.commit()
    typer.echo(f"Added metric '{name}' [{metric_id[:8]}]")


@app.command("edit")
def metric_edit(
    metric: str = typer.Argument(..., help="Metric id or name"),
    name: str = typer.Option("", "--name"),
    description: str = typer.Option("", "--description"),
    unit: str = typer.Option("", "--unit"),
    domain: str = typer.Option("", "--domain"),
) -> None:
    """Edit a metric definition."""
    resolved = _resolve_metric(metric)
    if resolved is None:
        typer.echo(f"Metric not found: {metric}", err=True)
        raise typer.Exit(code=1)
    metric_id, metric_name = resolved
    with get_connection() as conn:
        row = conn.execute(
            "SELECT name, description, unit, domain FROM metric_definitions WHERE id = ?",
            (metric_id,),
        ).fetchone()
        conn.execute(
            """
            UPDATE metric_definitions
            SET name = ?, description = ?, unit = ?, domain = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                name or row["name"],
                description or row["description"],
                unit or row["unit"],
                domain or row["domain"],
                datetime.utcnow().isoformat(),
                metric_id,
            ),
        )
        conn.commit()
    typer.echo(f"Updated metric '{metric_name}'")


@app.command("delete")
def metric_delete(
    metric: str = typer.Argument(..., help="Metric id or name"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Delete a metric definition."""
    resolved = _resolve_metric(metric)
    if resolved is None:
        typer.echo(f"Metric not found: {metric}", err=True)
        raise typer.Exit(code=1)
    metric_id, metric_name = resolved
    if not yes and not confirm_choice(f"Delete metric '{metric_name}'?"):
        raise typer.Exit(code=0)
    with get_connection() as conn:
        conn.execute("DELETE FROM metric_assignments WHERE metric_id = ?", (metric_id,))
        conn.execute("DELETE FROM metric_definitions WHERE id = ?", (metric_id,))
        conn.commit()
    typer.echo(f"Deleted metric '{metric_name}'")


@app.command("assign")
def metric_assign(
    metric: str = typer.Argument(..., help="Metric id or name"),
    goal: str = typer.Option(..., "--goal", help="Goal title or id prefix"),
) -> None:
    """Assign a metric to a goal."""
    resolved = _resolve_metric(metric)
    if resolved is None:
        typer.echo(f"Metric not found: {metric}", err=True)
        raise typer.Exit(code=1)
    metric_id, metric_name = resolved

    repo = Repository()
    matched_goal = None
    goal_ref = goal.strip().lower()
    for candidate in repo.list_goal_arcs(status=None):
        if candidate.id == goal or candidate.id.startswith(goal):
            matched_goal = candidate
            break
        if candidate.title.lower() == goal_ref or candidate.title.lower().startswith(goal_ref):
            matched_goal = candidate
            break
    if matched_goal is None:
        typer.echo(f"Goal not found: {goal}", err=True)
        raise typer.Exit(code=1)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO metric_assignments (id, metric_id, goal_id, goal_title, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(uuid4()), metric_id, matched_goal.id, matched_goal.title, datetime.utcnow().isoformat()),
        )
        conn.commit()
    if not matched_goal.primary_metric:
        matched_goal.primary_metric = metric_name
        repo.update_goal_arc(matched_goal)
    typer.echo(f"Assigned metric '{metric_name}' to goal '{matched_goal.title}'")
