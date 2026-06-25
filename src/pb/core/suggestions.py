# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""AI-assisted suggestion engine for the learning CLI.

Provides:
- SuggestionEngine: assembles context and calls Flash Lite for learning-command suggestions
- tier2_confirm: mandatory Y/n confirmation for all AI-suggested actions (AISG-02, D-08)
- get_recent_commands: reads last N usage log entries for context (ULOG-04)

All functions degrade gracefully when Flash Lite is unavailable (D-13 pattern):
exceptions are caught, logged as warnings, and safe empty/None values returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import structlog

from pb.core.matching import MatchCandidate, resolve_strict_match
from pb.llm.gemini import get_client, FLASH_LITE_MODEL
from pb.storage.database import get_connection
from pb.storage.repository import Repository

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Static command list (D-01, AISG-03)
# ---------------------------------------------------------------------------

AVAILABLE_COMMANDS: list[str] = [
    "goal",
    "plan",
    "study",
    "practise",
    "learn",
    "start",
    "pause",
    "resume",
    "finish",
    "next",
    "do",
    "review",
    "anki",
    "note",
    "now",
]

COMMAND_DOCS = """\
goal add — Create a structured learning goal
goal refine <goal-id> — Re-draft a goal from current evidence
plan day — Materialize today's study and practise blocks
plan week — Shape next week's learning focus
study <topic> — Start conceptual/internalisation work
study recall <scope> — Generate a scoped recall set
study debrief [topic] — Consolidate a finished study block
practise <topic> — Start deliberate practice
learn <topic> — Auto-route a learning request to study or practise
start <task-id> — Start a planned learning block
pause — Pause the active session
resume <task-id> — Resume a prior learning task
finish — Finish the active session
finish --debrief — Study-only closeout with Socratic debrief
anki list --suggested — Review suggested Anki candidates
anki export — Export accepted or edited Anki cards
review day — Synthesize today's learning progress
review week — Synthesize this week's learning progress
next — Show the best next learning actions
do <intent> — Route a natural-language learning request
note — Capture a learning observation without leaving the terminal
now — Show the active learning session
"""


# ---------------------------------------------------------------------------
# get_recent_commands (ULOG-04, D-04)
# ---------------------------------------------------------------------------


def get_recent_commands(limit: int = 20) -> list[str]:
    """Return the last `limit` command strings from usage_log ordered by timestamp DESC.

    Returns an empty list if the database is unavailable or any error occurs.

    Args:
        limit: Maximum number of entries to return. Defaults to 20.

    Returns:
        List of command strings, most recent first. Empty list on error.
    """
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT command FROM usage_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["command"] for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# tier2_confirm (D-08, AISG-02)
# ---------------------------------------------------------------------------


def tier2_confirm(label: str, detail: str = "") -> bool:
    """Display an AI suggestion inline and prompt for Y/n confirmation.

    This is the mandatory confirmation gate for all AI-suggested actions.
    No AI action may be executed without calling this function first (AISG-02).

    Args:
        label: The suggested command string to show the user.
        detail: Optional one-sentence explanation to display below the label.

    Returns:
        True if user confirms (empty, 'y', or 'yes'), False otherwise.
        Also returns False on KeyboardInterrupt or EOFError.
    """
    import typer

    typer.echo(f"\n  Suggestion: {label}")
    if detail:
        typer.echo(f"  {detail}")
    
    try:
        raw = input("Execute? [Y/n] ").strip()
    except (KeyboardInterrupt, EOFError):
        return False
    normalized = " ".join(raw.lower().split())
    if normalized in ("", "y", "yes", "sure", "ok", "go", "go ahead"):
        return True
    if normalized in ("n", "no", "cancel", "skip", "later", "not now"):
        return False
    return False


# ---------------------------------------------------------------------------
# SuggestionEngine (D-01, D-02, D-03, AISG-01, AISG-03)
# ---------------------------------------------------------------------------


class SuggestionEngine:
    """Assembles context from active task, vault cwd, and usage history, then calls
    Flash Lite to suggest the single best pb command for a given intent (D-01, D-02).

    Degrades gracefully: returns None if Flash Lite is unavailable or returns empty output.
    """

    def __init__(
        self,
        repo=None,
        get_vault_cwd: Optional[Callable[[], Path]] = None,
        vault_root: Optional[Path] = None,
    ):
        self._repo = repo if repo is not None else Repository()
        self._get_vault_cwd = get_vault_cwd
        self._vault_root = vault_root

    def suggest(self, intent: str) -> Optional[tuple[str, str]]:
        """Return (command, explanation) for the given intent, or None if unavailable.

        Per D-02: single best command with brief one-sentence explanation.
        Result must be passed to tier2_confirm before execution.

        Args:
            intent: Natural-language description of what the user wants to do.

        Returns:
            (command, explanation) tuple, or None if Flash Lite unavailable/error.
        """
        client = get_client()
        if not client.is_available():
            return None

        context = self._build_context()
        prompt = (
            "You are a command assistant for the 'pb' CLI learning system.\n"
            f"Context:\n{context}\n\n"
            f"Available commands and their EXACT syntax:\n{COMMAND_DOCS}\n"
            "RULES:\n"
            "- Use ONLY the commands and flags shown above. Do NOT invent flags or options.\n"
            "- Do NOT prefix commands with 'pb'. Just give the bare command (e.g., 'start my-task' not 'pb start my-task').\n"
            "- Stay inside the study/practise learning product. Do not suggest shell navigation or generic productivity commands.\n\n"
            f"User intent: {intent}\n\n"
            "Respond with EXACTLY two lines:\n"
            "Line 1: The single best command (e.g., 'start abcd1234' or 'study german cases')\n"
            "Line 2: A brief explanation (one sentence)\n"
            "Do NOT include anything else."
        )

        result = client.generate_with_model(prompt, FLASH_LITE_MODEL)
        if not result or not result.strip():
            return None

        return self._parse_suggestion(result)

    def _build_context(self) -> str:
        """Assemble context string from active task, vault cwd, and last 20 usage entries.

        Per D-01, AISG-03: context includes task title, vault location, and recent commands.
        Each source is independently guarded; any failure is silently skipped.

        Returns:
            Multi-line context string, or 'No context available.' if all sources fail.
        """
        parts: list[str] = []

        # Active task
        try:
            task = self._repo.get_active_task()
            if task:
                parts.append(f"Active task: {task.title}")
        except Exception:
            pass

        # Vault cwd
        try:
            if self._get_vault_cwd is not None and self._vault_root is not None:
                cwd = self._get_vault_cwd()
                rel = str(cwd.relative_to(self._vault_root))
                parts.append(f"Vault location: {rel if rel != '.' else 'root'}")
        except Exception:
            pass

        # Last 20 usage log entries (ULOG-04)
        try:
            cmds = get_recent_commands(20)
            if cmds:
                parts.append(f"Recent commands: {', '.join(cmds)}")
        except Exception:
            pass

        return "\n".join(parts) if parts else "No context available."

    @staticmethod
    def _parse_suggestion(raw: str) -> Optional[tuple[str, str]]:
        """Parse Flash Lite two-line response into (command, explanation).

        Args:
            raw: Raw text from Flash Lite.

        Returns:
            (command, explanation) tuple, or None if response is empty/unparseable.
        """
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        if not lines:
            return None
        command = lines[0]
        explanation = lines[1] if len(lines) > 1 else ""
        return (command, explanation)


# ---------------------------------------------------------------------------
# MkMvEngine (D-05, D-06, D-07, D-10, AISG-05, AISG-06)
# ---------------------------------------------------------------------------


class MkMvEngine:
    """AI engine for the mkmv chimera command — find and move notes.

    Searches vault for notes matching a topic description, then suggests
    a target folder to move them into. Flash Lite identifies which notes
    are relevant from search results. User confirms before any moves.

    Folder ranking enforces the flat vault constraint (depth guard D-10).
    """

    def __init__(
        self,
        vault_root: Path,
    ):
        self._vault_root = vault_root

    def find_matching_notes(self, description: str) -> list[tuple[Path, str]]:
        """Search vault for notes matching the description.

        Uses FTS index per folder, falls back to filename + content grep.
        Returns list of (absolute_path, snippet) tuples.
        """
        from pb.vault.indexer import search_folder_index

        matches: list[tuple[Path, str]] = []
        keywords = description.lower().split()

        for folder in self._vault_root.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            fts_results = search_folder_index(folder, description)
            if fts_results:
                for rel_path, snippet in fts_results:
                    full = (self._vault_root / rel_path).resolve()
                    try:
                        full.relative_to(self._vault_root.resolve())
                    except ValueError:
                        continue
                    if full.is_file():
                        matches.append((full, snippet))
            else:
                for md_file in folder.glob("*.md"):
                    if md_file.name.startswith("."):
                        continue
                    name_lower = md_file.stem.lower().replace("-", " ").replace("_", " ")
                    if any(kw in name_lower for kw in keywords):
                        snippet = _extract_heading(md_file)
                        matches.append((md_file, snippet))

        return matches

    def ai_filter_notes(
        self, description: str, candidates: list[tuple[Path, str]]
    ) -> list[Path]:
        """Use Flash Lite to pick the most relevant notes from candidates."""
        if len(candidates) <= 1:
            return [p for p, _ in candidates]

        client = get_client()
        if client.is_available():
            note_list = "\n".join(
                f"{i+1}. {p.relative_to(self._vault_root)} — {snip[:80]}"
                for i, (p, snip) in enumerate(candidates)
            )
            prompt = (
                f"Topic: {description}\n\n"
                f"Notes found:\n{note_list}\n\n"
                "If you are highly confident, return ONLY the relevant line numbers separated by commas.\n"
                "If everything clearly belongs together, return: all\n"
                "If confidence is low, return: unknown"
            )
            result = client.generate_with_model(prompt, FLASH_LITE_MODEL)
            if result:
                text = result.strip().lower()
                if text == "all":
                    return [p for p, _ in candidates]
                if text not in {"unknown", "uncertain", "idk"}:
                    selected: list[Path] = []
                    for token in text.replace(",", " ").split():
                        try:
                            idx = int(token) - 1
                        except ValueError:
                            continue
                        if 0 <= idx < len(candidates):
                            selected.append(candidates[idx][0])
                    if selected:
                        return selected

        if not client.is_available():
            return [p for p, _ in candidates]

        match_candidates = [
            MatchCandidate(
                key=str(path.relative_to(self._vault_root)),
                label=path.stem.replace("-", " ").replace("_", " "),
                text=f"{path.stem} | {snippet}",
            )
            for path, snippet in candidates
        ]
        result = resolve_strict_match(description, match_candidates)
        if result.accepted and result.matched_index is not None:
            return [candidates[result.matched_index][0]]
        return []

    def rank_folder(self, description: str, folder_names: list[str]) -> Optional[str]:
        """Flash Lite picks the best top-level folder to move notes into (D-06, D-10)."""
        if not folder_names:
            return None

        client = get_client()
        if not client.is_available():
            return None

        prompt = (
            f"Topic: {description}\n"
            f"Available vault folders: {', '.join(folder_names)}\n"
            "Return ONLY the single best folder name from the list above, nothing else."
        )

        result = client.generate_with_model(prompt, FLASH_LITE_MODEL)
        if not result:
            return folder_names[0] if folder_names else None

        ranked = result.strip()

        if "/" in ranked:
            logger.warning("suggestions.mkmv_depth_guard", ranked=ranked)
            return folder_names[0]

        if ranked in folder_names:
            return ranked

        for name in folder_names:
            if name.lower() == ranked.lower():
                return name

        return folder_names[0]


def _extract_heading(md_file: Path) -> str:
    """Extract the first H1 heading from a markdown file, or return filename."""
    try:
        with open(md_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    return line[2:].strip()
                if line and not line.startswith("---"):
                    break
        return md_file.stem
    except OSError:
        return md_file.stem
