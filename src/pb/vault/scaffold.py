# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Idempotent vault folder scaffolding.

Creates missing folders per D-07 without modifying existing content per D-08.
Also seeds LLM-agnostic guidance files (AGENTS.md, ISSUES.md) on first scaffold.
"""

from pathlib import Path
from typing import Optional

from pb.vault.config import get_vault_path, VAULT_SCHEMA


AGENTS_MD_TEMPLATE = """# AGENTS.md

LLM-agnostic guidance for any agent (Claude, Codex, Cline, custom MCP clients)
operating on this ProductiveBrain vault. Read this file before writing.

## Write protocol

- `pb` is the sole authoritative writer of vault state. MCP `vault_write`
  routes through pb internals; do not bypass with direct filesystem writes.
- All mutations are gated. Tier-1 captures (thought, todo) auto-execute;
  tier-2 mutations (goal commit, session finish, anki generate, notes
  organise, vault write/create) queue a pending action — surface the
  returned `pending_id` to the user verbatim.

## Vault conventions

- Flat structure, depth ≤ 2: `vault_root/<category>/<note>.md`.
- Categories follow the numbered prefix scheme already present in the vault.
- Every note has YAML frontmatter with `type:` and `updated:`.
- Bidirectional `[[wiki-links]]` for cross-references; pb enforces backlinks
  on write.

## Note lifecycle

- `learning_stage: new` → `learning` → `learnt` → `stale`. Don't skip stages.
- Use `vault_link_graph` before refactoring connected notes.

## What not to do

- Do not delete or rename notes via MCP tools — only via `pb` CLI with the
  user present.
- Do not synthesize content for `_state.md` or `_index.md` per-folder files;
  pb regenerates them.
- Do not embed long pre-existing files; use links.
"""


ISSUES_MD_TEMPLATE = """# ISSUES.md

Running log for issues observed by agents while working in this vault.
Append; do not rewrite history. Format:

```
## YYYY-MM-DD — short title
- **Where:** path or tool name
- **Symptom:** one sentence
- **Status:** open | resolved | wontfix
- **Notes:** (optional)
```

## Open

_(none yet)_
"""


def _seed_guidance_files(vault_path: Path) -> list[str]:
    """Write AGENTS.md and ISSUES.md at vault root if absent. Returns created files."""
    created: list[str] = []
    agents_path = vault_path / "AGENTS.md"
    if not agents_path.exists():
        agents_path.write_text(AGENTS_MD_TEMPLATE)
        created.append("AGENTS.md")
    issues_path = vault_path / "ISSUES.md"
    if not issues_path.exists():
        issues_path.write_text(ISSUES_MD_TEMPLATE)
        created.append("ISSUES.md")
    return created


def scaffold_vault(vault_path: Optional[Path] = None) -> list[str]:
    """Create missing vault folders + guidance files without touching existing content.

    Implements D-07 (auto-create on first tool use) and D-08 (merge approach).
    Seeds AGENTS.md and ISSUES.md if absent.

    Args:
        vault_path: Path to vault root. If None, reads from pb config.

    Returns:
        List of paths that were created (folders and guidance files).

    Raises:
        FileNotFoundError: If vault_path doesn't exist.
    """
    if vault_path is None:
        vault_path = get_vault_path()

    if not vault_path.exists():
        raise FileNotFoundError(f"Vault not found at {vault_path}")

    created: list[str] = []

    for folder in VAULT_SCHEMA:
        folder_path = vault_path / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            created.append(folder)

    created.extend(_seed_guidance_files(vault_path))
    return created


def ensure_vault_folder(folder: str, vault_path: Optional[Path] = None) -> Path:
    """Ensure a specific vault folder exists and return its path.

    Args:
        folder: Relative folder path within vault (e.g., "people")
        vault_path: Path to vault root. If None, reads from pb config.

    Returns:
        Absolute path to the folder.

    Raises:
        FileNotFoundError: If vault_path doesn't exist.
    """
    if vault_path is None:
        vault_path = get_vault_path()

    if not vault_path.exists():
        raise FileNotFoundError(f"Vault not found at {vault_path}")

    folder_path = vault_path / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path
