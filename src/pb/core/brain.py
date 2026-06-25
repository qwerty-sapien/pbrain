# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""BrainEngine -- graph-aware vault intelligence with Gemini function calling.

Sends the entire vault graph topology (nodes + edges) to the LLM,
which calls read_vault_note via AFC to fetch specific note contents
before answering.

Graph topology is persistent at {vault}/.pb-graph.yaml, updated
incrementally on every vault_write. Brain just reads it — no scanning.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import structlog

from pb.llm.gemini import (
    get_client as get_gemini_client,
    FLASH_LITE_MODEL,
    FLASH_MODEL,
    PRO_MODEL,
)
from pb.vault import get_vault_path
from pb.vault.graph import load_vault_graph, graph_to_adjacency_text

logger = structlog.get_logger()

GRAPH_PROMPT = (
    "You are a personal knowledge assistant. The user has an Obsidian vault "
    "with {node_count} notes and {edge_count} links.\n\n"
    "Below is the vault graph. Each line is a note path, optionally "
    "followed by -> and the notes it links to.\n\n"
    "{graph}\n\n"
    "Use the read_vault_note function to read notes relevant to the question. "
    "You can call it multiple times. Answer based on what you read.\n\n"
    "Question: {query}"
)

AUTO_SUFFIX = (
    "\n\nAfter reading relevant notes, assess whether you can give a thorough "
    "answer. If yes, answer directly. If this needs deeper reasoning or "
    "synthesis, respond with ONLY one line: ESCALATE: flash or ESCALATE: pro"
)

# ---------------------------------------------------------------------------
# Stage-aware interrogation prompts per D-07
# ---------------------------------------------------------------------------

STAGE_PROMPTS = {
    "explore": (
        "You are a firm but curious tutor. The student is encountering this material for the first time. "
        "Ask probing questions like 'what do you think X means?' and 'how does X relate to Y?' "
        "Test whether they have genuinely engaged with the material. Do NOT reveal answers. "
        "If they cannot answer, push back: 'think about [specific aspect] again.'"
    ),
    "consolidate": (
        "You are a strict, no-BS tutor. The student has been studying this material. "
        "Read their previous Socratic capture notes and use their own words against them: "
        "'you said X last time, explain why.' Test for deeper understanding and find gaps. "
        "Never accept vague answers. Push for specifics."
    ),
    "exploit": (
        "You are a demanding examiner testing mastery. Ask for edge cases, cross-domain connections, "
        "and rapid explanations: 'teach me this concept in 30 seconds.' "
        "Challenge assumptions and test transfer to new contexts."
    ),
    "re-engage": (
        "You are mildly impatient. The student has not touched this material recently. "
        "Start with: 'you haven't reviewed this in N days, let's see what you remember.' "
        "Ask targeted refresher probes. Flag weak recall areas."
    ),
}

# Constellation view prompt per D-08
CONSTELLATION_PROMPT = (
    "You have access to the note graph. I will give you a list of connected notes "
    "(1-hop and 2-hop neighbors). Do NOT show the graph to the user. Instead, "
    "ask the user to explain how specific notes connect to each other. "
    "Example: 'how does [note A] connect to [note B]?' "
    "The user must prove they understand the connections, not just see them."
)

# Gap detection prompt per D-09
GAP_DETECTION_PROMPT = (
    "You are analyzing a knowledge domain for missing prerequisite notes. "
    "Read the domain graph structure and existing notes carefully. "
    "Compare what exists vs what should exist for complete understanding. "
    "Identify missing concepts, then probe the user: "
    "'You have notes on X and Z but nothing on Y -- can you explain Y?' "
    "Test if the gap is real (missing knowledge) or just undocumented."
)

# Wrong answer handling per D-10
# The sentinel phrase "This needs more study" is the machine-readable trigger.
# When detected in the LLM response, the CLI handler (brain.py CLI) logs a weak_area
# interaction so the study planner can query these flagged notes.
PUSHBACK_SUFFIX = (
    "\n\nIMPORTANT: If the user gives a wrong or vague answer, push back ONCE: "
    "'That's not right, think again about [specific aspect].' "
    "If they still cannot answer, say EXACTLY 'This needs more study' and move on. "
    "NEVER passively reveal the answer."
)

# Sentinel string for D-10 weak-area detection in CLI response handler
WEAK_AREA_SENTINEL = "This needs more study"

# Map learning_stage frontmatter values to mode keys used in STAGE_PROMPTS
_STAGE_TAG_TO_MODE = {
    "#new": "explore",
    "#learning": "consolidate",
    "#learnt": "exploit",
    "#stale": "re-engage",
}


# ---------------------------------------------------------------------------
# AFC read tool
# ---------------------------------------------------------------------------


def _make_read_fn(vault_path: Path):
    """Create a vault note reader for Gemini AFC.

    Logs implicit 'query' interaction per D-02 after each successful read.
    Interaction logging is wrapped in try/except so it never breaks a brain query.
    """
    vault_resolved = vault_path.resolve()

    def read_vault_note(path: str) -> str:
        """Read the full content of a vault note.

        Args:
            path: Relative path within the vault (e.g. 'people/alice.md')

        Returns:
            Full text content of the note, or an error message.
        """
        target = (vault_path / path).resolve()
        if not target.is_relative_to(vault_resolved):
            return "Error: path outside vault boundary"
        if not target.exists():
            return f"Error: note not found at {path}"
        try:
            content = target.read_text()
            try:
                from pb.vault.lifecycle import log_interaction
                log_interaction(note_path=path, event_type="query")
            except Exception:
                pass
            return content
        except Exception as e:
            return f"Error reading {path}: {e}"

    return read_vault_note


# ---------------------------------------------------------------------------
# Auto-escalation helpers
# ---------------------------------------------------------------------------

_ESCALATE_RE = re.compile(r"^ESCALATE:\s*(flash|pro)\s*$", re.IGNORECASE)


def _parse_escalation(text: str) -> Optional[str]:
    """Check if LLM response is an escalation request.

    Returns target model constant, or None if this is a normal answer.
    """
    m = _ESCALATE_RE.match(text.strip())
    if not m:
        return None
    target = m.group(1).lower()
    return PRO_MODEL if target == "pro" else FLASH_MODEL


# ---------------------------------------------------------------------------
# BrainEngine
# ---------------------------------------------------------------------------


class BrainEngine:
    """Graph-aware vault query engine with Gemini AFC."""

    def __init__(self):
        self._gemini = get_gemini_client()
        self._graph_stats: Optional[str] = None
        self._model_used: Optional[str] = None

    def is_available(self) -> bool:
        return self._gemini.is_available()

    def query(
        self,
        question: str,
        show_prompt: bool = False,
        use_pro: bool = False,
        use_flash: bool = False,
        auto_escalate: bool = False,
        pre_ranked_candidates: Optional[list[str]] = None,  # Phase 19
        learning_stage: Optional[str] = None,               # Phase 19
        verbose: bool = False,                              # Phase 19
    ) -> str:
        """Execute a brain query: send vault graph, let LLM read notes via AFC.

        Phase 19 extensions:
            pre_ranked_candidates: When provided, only those note slugs appear in
                LLM context (filtered graph). Built from CompositeScorer results.
            learning_stage: When provided (frontmatter tag like '#new'), selects
                stage-appropriate interrogation tone via STAGE_PROMPTS.
            verbose: Signal breakdown is handled by the CLI caller; this param
                is accepted for forward compatibility.

        Model selection:
            Default: Flash Lite, no escalation.
            use_flash: Flash, no escalation.
            use_pro: Pro directly.
            auto_escalate: Flash Lite first — model self-assesses and may
                request escalation to Flash or Pro.
        """
        if not self.is_available():
            return "LLM unavailable -- set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT to enable brain queries."

        vault_path = get_vault_path()

        # Phase 19: Build filtered graph from pre-ranked slugs when provided (D-04)
        # Uses parameterized SQL — never f-string SQL with user-controlled input (T-19-09)
        if pre_ranked_candidates:
            try:
                from pb.vault.graph_store import open_vault_db
                conn = open_vault_db(vault_path)
                try:
                    placeholders = ",".join("?" * len(pre_ranked_candidates))
                    edges = conn.execute(
                        f"SELECT src, dst FROM links WHERE src IN ({placeholders}) AND dst IN ({placeholders})",
                        pre_ranked_candidates + pre_ranked_candidates,
                    ).fetchall()
                    graph_lines = [f"  {src} -> {dst}" for src, dst in edges]
                    graph_text = "\n".join(graph_lines) if graph_lines else "  (no connections)"
                    n_nodes = len(pre_ranked_candidates)
                    n_edges = len(edges)
                finally:
                    conn.close()
            except Exception:
                # Graceful degradation: fall back to full graph on any DB error
                edges_full = load_vault_graph(vault_path)
                graph_text, n_nodes, n_edges = graph_to_adjacency_text(edges_full)
        else:
            edges_full = load_vault_graph(vault_path)
            graph_text, n_nodes, n_edges = graph_to_adjacency_text(edges_full)

        self._graph_stats = f"Graph: {n_nodes} notes, {n_edges} links"

        if n_nodes == 0:
            return "Vault is empty — add notes first."

        read_fn = _make_read_fn(vault_path)
        base_prompt = GRAPH_PROMPT.format(
            node_count=n_nodes, edge_count=n_edges,
            graph=graph_text, query=question,
        )

        # Phase 19: Prepend stage-aware interrogation tone when learning_stage is given (D-07)
        if learning_stage is not None:
            mode = _STAGE_TAG_TO_MODE.get(learning_stage, learning_stage)
            stage_tone = STAGE_PROMPTS.get(mode)
            if stage_tone:
                base_prompt = stage_tone + PUSHBACK_SUFFIX + "\n\n" + base_prompt

        if show_prompt:
            import sys
            print("--- PROMPT ---", file=sys.stderr)
            print(base_prompt, file=sys.stderr)
            print("--- END PROMPT ---", file=sys.stderr)

        if use_pro:
            model = PRO_MODEL
        elif use_flash:
            model = FLASH_MODEL
        elif auto_escalate:
            return self._query_auto(base_prompt, read_fn)
        else:
            model = FLASH_LITE_MODEL

        result = self._gemini.generate_with_tools(base_prompt, model, [read_fn])
        if result is not None:
            self._model_used = model
            logger.debug("brain.query", model=model, answer_len=len(result))
            return result

        return "LLM request failed — check API key and quota, then try again."

    def query_constellation(self, slug: str) -> str:
        """Constellation view: ask user to prove connections in 1-hop + 2-hop neighborhood (D-08).

        Uses get_hop2_neighborhood() silently — the graph is NEVER shown to the user.
        Instead the LLM challenges the user to explain the connections.
        """
        if not self.is_available():
            return "LLM unavailable."
        vault_path = get_vault_path()
        from pb.vault.graph_store import get_hop2_neighborhood
        neighborhood = get_hop2_neighborhood(vault_path, slug)
        all_neighbors = set()
        for key in ("out1", "in1", "out2", "in2"):
            all_neighbors.update(neighborhood.get(key, []))
        if not all_neighbors:
            return f"No connections found for {slug}."

        read_fn = _make_read_fn(vault_path)
        prompt = (
            CONSTELLATION_PROMPT + "\n\n"
            f"Center note: {slug}\n"
            f"1-hop outgoing: {', '.join(neighborhood.get('out1', []))}\n"
            f"1-hop incoming: {', '.join(neighborhood.get('in1', []))}\n"
            f"2-hop outgoing: {', '.join(neighborhood.get('out2', []))}\n"
            f"2-hop incoming: {', '.join(neighborhood.get('in2', []))}\n\n"
            f"Now challenge the user to explain the connections."
        )
        result = self._gemini.generate_with_tools(prompt, FLASH_LITE_MODEL, [read_fn])
        if result:
            self._model_used = FLASH_LITE_MODEL
            return result
        return "Failed to generate constellation challenge."

    def _query_auto(self, base_prompt: str, read_fn) -> str:
        """Auto mode: Flash Lite self-assesses, escalates if needed."""
        prompt = base_prompt + AUTO_SUFFIX

        result = self._gemini.generate_with_tools(prompt, FLASH_LITE_MODEL, [read_fn])
        if result is None:
            return "LLM request failed — check API key and quota, then try again."

        escalate_to = _parse_escalation(result)
        if escalate_to is None:
            self._model_used = FLASH_LITE_MODEL
            logger.debug("brain.query", model=FLASH_LITE_MODEL, answer_len=len(result))
            return result

        logger.debug("brain.escalate", from_model=FLASH_LITE_MODEL, to_model=escalate_to)
        result = self._gemini.generate_with_tools(base_prompt, escalate_to, [read_fn])
        if result is not None:
            self._model_used = escalate_to
            logger.debug("brain.query", model=escalate_to, answer_len=len(result))
            return result

        return "LLM request failed — check API key and quota, then try again."

    def get_context_display(self) -> Optional[str]:
        """Return graph stats + model info for CLI display."""
        parts = []
        if self._graph_stats:
            parts.append(self._graph_stats)
        if self._model_used:
            parts.append(f"Model: {self._model_used}")
        return " | ".join(parts) if parts else None

    def detect_orphans(self) -> list[dict]:
        """Return list of orphan notes: notes with no inbound AND no outbound links (GRPH-02)."""
        import re as _re
        from pb.vault.graph import get_backlinks, load_vault_graph
        from pb.vault.lifecycle import read_frontmatter

        vault_path = get_vault_path()
        edges = load_vault_graph(vault_path)
        backlinks = get_backlinks(vault_path)
        orphans = []

        for rel_path, targets in edges.items():
            # Skip special files (underscore-prefixed)
            if Path(rel_path).name.startswith("_"):
                continue
            inbound = backlinks.get(rel_path, [])
            if targets or inbound:
                continue
            # This note is an orphan
            try:
                content = (vault_path / rel_path).read_text()
                fm, _ = read_frontmatter(content)
                words = len(content.split())
                h1 = _re.search(r"^#\s+(.+)", content, _re.MULTILINE)
                title = h1.group(1).strip() if h1 else Path(rel_path).stem
                orphans.append({
                    "path": rel_path,
                    "title": title,
                    "learning_stage": fm.get("learning_stage"),
                    "created": fm.get("created", "---"),
                    "words": words,
                    "folder": str(Path(rel_path).parent),
                })
            except Exception:
                pass

        return orphans
