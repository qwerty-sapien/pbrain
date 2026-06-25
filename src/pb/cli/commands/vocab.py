# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Fast Obsidian-friendly quick capture for language vocabulary."""

import typer

app = typer.Typer(no_args_is_help=False)

@app.command("vocab")
def add_vocab(
    ctx: typer.Context,
    term: str = typer.Argument(..., help="The vocabulary word or phrase"),
    definition: str = typer.Argument(..., help="The definition or translation"),
    domain: str = typer.Option("german-a1-to-b1", "--domain", "-d", help="Knowledge domain"),
):
    """Fast Obsidian-friendly quick capture for language vocabulary."""
    from pb.vault.config import get_vault_path
    from pb.core.graph_writer import make_slug
    import yaml as _yaml
    from datetime import datetime

    vault = get_vault_path()
    domain_dir = vault / "knowledge" / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    slug = make_slug(term)
    note_path = domain_dir / f"{slug}.md"
    
    fm = {
        "type": "vocabulary",
        "term": term,
        "definition": definition,
        "learning_stage": "#new",
        "created": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    
    frontmatter = _yaml.dump(fm, default_flow_style=False, allow_unicode=True)
    body = f"# {term}\n\n- **Definition:** {definition}\n- **Notes:** \n"
    
    note_path.write_text(f"---\n{frontmatter}---\n\n{body}")
    
    from pb.cli.console import get_console
    console = get_console()
    console.print(f"[success]Vocab captured: {term} -> {note_path}[/]")
