# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Skills management commands.

Bare 'pb skills' defaults to 'pb skills add' per D-25.
"""

import typer

app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def skills_default(ctx: typer.Context):
    """Manage skills. Bare 'pb skills' adds a skill (D-25)."""
    if ctx.invoked_subcommand is not None:
        return
    # D-25: bare command = add
    add_skill(
        name=typer.prompt("Skill name"),
        domain=typer.prompt("Domain (e.g., engineering, data-science)", default="", show_default=False),
        proficiency=typer.prompt(
            "Proficiency (beginner/intermediate/advanced/expert)",
            default="beginner",
            show_default=False,
        ),
    )


@app.command("add")
def add_skill(
    name: str = typer.Argument(..., help="Skill name (e.g., 'Python', 'Machine Learning')"),
    domain: str = typer.Option("", "--domain", "-d", help="Domain (e.g., engineering, data-science)"),
    proficiency: str = typer.Option(
        "beginner",
        "--level",
        "-l",
        help="Proficiency: beginner|intermediate|advanced|expert",
    ),
):
    """Add a new skill note to the vault."""
    from pb.core.skills import SkillManager

    mgr = SkillManager()
    path = mgr.create_skill(name, domain=domain, proficiency=proficiency)
    if path:
        typer.echo(f"Skill note: {path.name}")
    else:
        typer.echo("Failed to create skill note.", err=True)
        raise typer.Exit(code=1)


@app.command("show")
def show_skill(
    name: str = typer.Argument(..., help="Skill name to look up"),
):
    """Show a skill note's details."""
    from pb.core.skills import SkillManager

    mgr = SkillManager()
    path = mgr.get_skill(name)
    if path is None:
        typer.echo(f"Skill not found: {name}", err=True)
        raise typer.Exit(code=1)
    content = path.read_text()
    typer.echo(content)


@app.command("list")
def list_skills():
    """List all skills with active/done task counts (D-11, D-12)."""
    from pb.core.skill_links import list_skills_with_counts
    from pb.core.skills import SkillManager

    rows = list_skills_with_counts()
    if not rows:
        typer.echo("No skills tracked yet.")
        typer.echo("Tag skills with 'pb finish' or 'pb add --skill <name>'.")
        return

    _skill_mgr = SkillManager()
    for row in rows:
        name = row["skill_name"]
        active = row.get("active_count", 0)
        done = row.get("done_count", 0)
        # Read proficiency from vault note (optional — degrades to [?] if missing)
        data = _skill_mgr.get_skill_data(name)
        prof = data.get("proficiency", "?") if data else "?"
        typer.echo(f"  {name:<20} {active} active  {done:>3} done  [{prof}]")

    total = len(rows)
    typer.echo(f"  {'─' * 40}")
    typer.echo(f"  {total} skill{'s' if total != 1 else ''} tracked")


@app.command("evidence")
def add_evidence(
    name: str = typer.Argument(..., help="Skill name"),
    note: str = typer.Option(..., "--note", "-n", help="Evidence description"),
    session: str = typer.Option(
        "", "--session", "-s", help="Session log slug for wikilink"
    ),
):
    """Add an evidence entry to a skill note (D-18)."""
    from datetime import datetime

    from pb.core.skills import SkillManager

    mgr = SkillManager()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    session_slug = session or f"{date_str}-manual"
    path = mgr.add_evidence(name, date_str, note, session_slug)
    if path:
        typer.echo(f"Evidence added to {path.name}")
    else:
        typer.echo(f"Skill not found: {name}", err=True)
        raise typer.Exit(code=1)
