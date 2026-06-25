# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Note creation commands — top-level Socratic verb + back-compat subcommands (Plan 24-03).

CLI entry points (D-01, D-02, D-03, D-04, D-06):
  pb note [text]          -- Socratic debrief (1-2 or 10-12 rounds depending on picker)
  pb note --concept       -- Route to QuestionTreeEngine concept flow (D-02)
  pb note --person        -- Route to QuestionTreeEngine person flow
  pb note --opp           -- Route to QuestionTreeEngine opportunity flow
  pb note --quick [text]  -- Skip debrief; write stub note from topic (SOCR-02)
  pb note concept         -- Back-compat subcommand alias (D-02)
  pb note person          -- Back-compat subcommand alias
  pb note opp             -- Back-compat subcommand alias

Removed (D-02, D-04):
  pb note book     -- Moved to plugin shelf
  pb note socratic -- Replaced by top-level verb

INV-5: This file contains NO direct calls to SocraticDebriefEngine, build_socratic_note,
infer_wikilinks, or extract_socratic_cards — only arg parsing, picker UI, and service delegation.
"""

from __future__ import annotations

from typing import Optional

import typer
import structlog

from pb.core.schemas import ensure_default_schemas, load_schema, list_schemas
from pb.core.question_tree import QuestionTreeEngine

logger = structlog.get_logger()

app = typer.Typer(
    name="note",
    help="Capture a Socratic insight or create a structured vault note",
    no_args_is_help=False,
    invoke_without_command=True,
)


def _is_interactive() -> bool:
    """Return True if stdin is a TTY. Extracted for testability."""
    import sys
    return sys.stdin.isatty()


# ---------------------------------------------------------------------------
# Top-level callback — replaces the old subcommand-only design (D-01)
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def note_capture(
    ctx: typer.Context,
    topic: Optional[list[str]] = typer.Argument(None, help="Topic text. Multi-word OK -- no quotes needed."),
    concept: bool = typer.Option(False, "--concept", help="Route to QuestionTreeEngine concept flow"),
    person: bool = typer.Option(False, "--person", help="Route to QuestionTreeEngine person flow"),
    opp: bool = typer.Option(False, "--opp", help="Route to QuestionTreeEngine opportunity flow"),
    quick: bool = typer.Option(False, "--quick", help="Skip Socratic debrief; capture topic as a stub note (SOCR-02)"),
    flash: bool = typer.Option(False, "--flash", help="Use Flash model for note structuring (default Flash Lite)"),
    sync: bool = typer.Option(False, "--sync", help="Bypass Vertex Batch; create note synchronously"),
    trust: bool = typer.Option(False, "--trust", help="Skip post-edit wikilink inference"),
):
    """Capture a Socratic insight. `pb note [text]` opens an interrogation; flags select tree flows."""
    # If a subcommand was invoked (concept/person/opp old-style), let it run
    if ctx.invoked_subcommand is not None:
        return

    # Flag routes -- existing QuestionTreeEngine flow per D-02
    if concept:
        _run_question_tree("concept")
        return
    if person:
        _run_question_tree("person")
        return
    if opp:
        _run_question_tree("opportunity")
        return

    # Socratic capture -- new top-level flow per D-01/D-03/D-06
    if not _is_interactive():
        from pb.cli.console import get_console
        console = get_console()
        console.print("[error]pb note requires an interactive terminal[/]")
        raise typer.Exit(code=1)

    from pb.cli.console import get_console
    from pb.vault import get_vault_path
    from pb.core.graph_writer import make_slug
    from pb.llm.gemini import FLASH_LITE_MODEL, FLASH_MODEL

    console = get_console()
    try:
        vault_path = get_vault_path()
    except Exception:
        console.print("[error]Vault not configured. Run `pb init` first.[/]")
        raise typer.Exit(code=1)

    socratic_service = ctx.obj['factory']['socratic_service']()

    # Domain detection per D-05
    knowledge_dir = vault_path / "knowledge"
    domain = socratic_service.detect_domain(knowledge_dir)
    if not domain:
        domain = _pick_domain_for_note(knowledge_dir, console)
    if not domain:
        console.print("[error]No domain selected; aborting.[/]")
        raise typer.Exit(code=1)

    topic_str = " ".join(topic or []).strip()

    # --quick path: stub note from topic alone, no debrief (SOCR-02)
    if quick:
        if not topic_str:
            console.print("[error]--quick requires a topic argument.[/]")
            raise typer.Exit(code=1)
        slug = make_slug(topic_str[:60])
        model = FLASH_MODEL if flash else FLASH_LITE_MODEL
        result = socratic_service.build_and_submit(
            qa_pairs=[],
            domain=domain,
            slug=slug,
            template="brief",
            sync=True,  # --quick implies sync (no LLM round-trip needed for stub)
            model=model,
            console=console,
        )
        if result:
            console.print(f"[success]Note stub created: {result}[/]")
        return

    # Short/Long picker per D-06
    max_rounds = _pick_session_depth(console)
    template = "brief" if max_rounds <= 3 else "deep"

    # Run debrief
    qa_pairs = socratic_service.run_note_debrief(
        topic=topic_str, domain=domain, max_rounds=max_rounds, console=console
    )
    if not qa_pairs:
        console.print("[warn]No Q&A captured; nothing to save.[/]")
        return

    # Slug from answers per Pitfall 5
    all_answers = " ".join(a for _, a in qa_pairs)
    slug = make_slug(all_answers[:60]) or make_slug(topic_str[:60]) or "note"

    model = FLASH_MODEL if flash else FLASH_LITE_MODEL
    result = socratic_service.build_and_submit(
        qa_pairs=qa_pairs,
        domain=domain,
        slug=slug,
        template=template,
        sync=sync,
        model=model,
        console=console,
    )
    if result:
        console.print(f"[success]{'Saved' if sync else 'Submitted'}: {result}[/]")

    # Bridge note suggestion (long mode only -- service enforces >=4 gate)
    if max_rounds > 3:
        socratic_service.suggest_bridge(qa_pairs, domain, console)


def _pick_session_depth(console) -> int:
    """Per D-06: short (2-3) vs long (10-12) picker. Returns max_rounds."""
    from pb.cli.pickers import pick_single_choice
    
    options = [
        ("3", "Short -- 2-3 questions, Flash Lite generates most of note"),
        ("12", "Long -- Flash Lite decides when satisfied, up to 10-12 rounds")
    ]
    
    choice = pick_single_choice(options, title="Session depth")
    return int(choice) if choice is not None else 3  # Default to short on cancel


def _pick_domain_for_note(knowledge_dir, console) -> Optional[str]:
    """Reuse the shared domain picker helper from learn.py."""
    try:
        from pb.cli.commands.learn import _pick_domain
        return _pick_domain(knowledge_dir, console)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# _run_question_tree and its helpers (unchanged from original note.py)
# ---------------------------------------------------------------------------

def _run_question_tree(schema_id: str) -> None:
    """Run the question tree interactive flow for a given schema.

    Implements the full D-41 to D-44 flow: required fields, optional selection,
    confirmation, and vault write.
    """
    ensure_default_schemas()

    try:
        schema = load_schema(schema_id)
    except FileNotFoundError:
        typer.echo(f"Schema not found: {schema_id}", err=True)
        available = list_schemas()
        if available:
            typer.echo(f"Available: {', '.join(available)}", err=True)
        raise typer.Exit(code=1)

    engine = QuestionTreeEngine(schema)
    typer.echo(f"\n-- New {schema.name} Note --")
    typer.echo("Commands: /skip (skip optional field), /done (finish early), /chat (ask LLM)\n")

    # ---- Phase 1: Required fields -----------------------------------
    while not engine.is_done:
        field = engine.current_field()
        if field is None:
            break  # No more required fields, move on

        progress = engine.progress_text()
        prompt_text = f"{progress} {field.prompt}"

        if field.field_type == "select" and field.options:
            prompt_text += f" ({'/'.join(field.options)})"
        if field.field_type == "multiline":
            prompt_text += " (empty line to finish)"

        try:
            if field.field_type == "multiline":
                typer.echo(f"{prompt_text}:")
                value = _read_multiline(engine)
                if value is None:
                    # /done or /chat was handled inside _read_multiline
                    pass
                # D-47: Concept probing trigger after multiline definition stored
                if (
                    not engine.is_done
                    and field.name == "definition"
                    and engine.schema.name.lower() == "concept"
                ):
                    stored_value = engine.values.get("definition", "")
                    if stored_value:
                        from pb.core.probing import should_probe
                        if should_probe(stored_value):
                            _run_probing(
                                concept_name=engine.values.get("title", "concept"),
                                definition=stored_value,
                                domain=engine.values.get("domain", ""),
                            )
            else:
                raw = typer.prompt(prompt_text)
                result = engine.process_input(raw)
                if result["action"] == "cancelled":
                    typer.echo("\nCancelled.")
                    return
                elif result["action"] == "error":
                    typer.echo(f"  ! {result['message']}")
                    continue
                elif result["action"] == "chat_activated":
                    _handle_chat(engine)
                elif result["action"] in ("done", "next_phase"):
                    break
                # D-47: Concept probing trigger after non-multiline definition stored
                elif (
                    result["action"] == "stored"
                    and field.name == "definition"
                    and engine.schema.name.lower() == "concept"
                ):
                    stored_value = engine.values.get("definition", "")
                    if stored_value:
                        from pb.core.probing import should_probe
                        if should_probe(stored_value):
                            _run_probing(
                                concept_name=engine.values.get("title", "concept"),
                                definition=stored_value,
                                domain=engine.values.get("domain", ""),
                            )

        except (EOFError, KeyboardInterrupt):
            typer.echo("\nAborted.")
            raise typer.Exit(code=0)

        # Check if we transitioned to optional_select after storing
        if engine._phase == "optional_select" or engine.is_done:
            break

    if engine.is_done:
        _finish(engine)
        return

    # ---- Phase 2: Optional field selection --------------------------
    optional = engine.get_optional_fields()
    if optional:
        typer.echo("\n-- Optional Fields --")
        typer.echo("Enter numbers to fill (comma-separated), or press Enter to skip all:\n")
        for i, f in enumerate(optional, 1):
            typer.echo(f"  {i}) {f.prompt}")

        try:
            raw = input("\nSelect optional fields: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""

        if raw:
            indices = []
            for part in raw.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(optional):
                        indices.append(idx)
            engine.set_optional_selections(indices)
        else:
            engine.set_optional_selections([])

        if engine.is_done:
            _finish(engine)
            return

        # ---- Phase 3: Fill selected optional fields -----------------
        while not engine.is_done:
            field = engine.current_field()
            if field is None:
                break

            progress = engine.progress_text()
            prompt_text = f"{progress} {field.prompt}"

            try:
                if field.field_type == "multiline":
                    typer.echo(f"{prompt_text} (empty line to finish):")
                    _read_multiline(engine)
                else:
                    raw = typer.prompt(prompt_text)
                    result = engine.process_input(raw)
                    if result["action"] == "error":
                        typer.echo(f"  ! {result['message']}")
                        continue
                    elif result["action"] == "chat_activated":
                        _handle_chat(engine)
                    elif result["action"] == "done":
                        break
            except (EOFError, KeyboardInterrupt):
                engine.process_input("/done")
                break

    _finish(engine)


def _read_multiline(engine: QuestionTreeEngine):
    """Read multiline input until empty line, handling inline commands.

    Returns the joined text, or None if /done or /chat consumed the input.
    """
    lines = []
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        cmd = line.strip().lower()
        if cmd == "":
            # Empty line = end of multiline input
            if lines:
                result = engine.process_input("\n".join(lines))
                if result["action"] == "error":
                    typer.echo(f"  ! {result['message']}")
                    lines = []
                    continue
            else:
                # Empty multiline = skip
                engine.process_input("/skip")
            return None

        if cmd in ("/done", "/skip", "/chat"):
            if cmd == "/chat":
                _handle_chat(engine)
            else:
                result = engine.process_input(cmd)
                if result["action"] == "error":
                    typer.echo(f"  ! {result['message']}")
                    continue
            return None

        lines.append(line)

    # EOF — treat buffered lines as value
    if lines:
        engine.process_input("\n".join(lines))
    else:
        engine.process_input("/skip")
    return None


def _handle_chat(engine: QuestionTreeEngine) -> None:
    """Handle /chat mode: Flash Lite with schema + filled-field context (D-42)."""
    from pb.llm.gemini import get_client

    client = get_client()
    if not client.is_available():
        typer.echo("  LLM unavailable (set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT to enable /chat)")
        engine.deactivate_chat()
        return

    context = engine.get_chat_context()
    field = engine.current_field()
    field_name = field.name if field else "the note"

    typer.echo(f"  [Chat mode — ask about {field_name}. Type /back to return]")

    while True:
        try:
            user = input("  chat> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user.lower() in ("/back", "/exit", ""):
            break

        prompt = (
            f"You are helping create a {engine.schema.name} note.\n\n"
            f"Context:\n{context}\n\n"
            f"Current field: {field_name}\n"
            f"User question: {user}\n\n"
            "Help them fill in this field. Be concise."
        )
        response = client.generate(prompt)
        if response:
            typer.echo(f"  {response}")
        else:
            typer.echo("  No response.")

    engine.deactivate_chat()


def _run_probing(concept_name: str, definition: str, domain: str) -> None:
    """Run Socratic probing session for a concept definition (D-47 to D-49)."""
    from pb.core.probing import ProbingEngine

    engine = ProbingEngine(concept_name, definition, domain)

    if not engine._client.is_available():
        typer.echo("\n  [Probing skipped — LLM unavailable]")
        return

    typer.echo("\n  Your definition is detailed. Let's probe your understanding...")
    typer.echo("  (Type /done to exit probing)\n")

    # First question
    question = engine.get_question()
    if not question:
        return

    while engine.should_continue():
        typer.echo(f"  Mentor: {question}")
        try:
            answer = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if answer.lower() in ("/done", "/exit", "/quit", ""):
            break

        question = engine.get_question(answer)
        if not question:
            break

    typer.echo(f"\n  [Probing complete — {engine.round_number} rounds]")


def _finish(engine: QuestionTreeEngine) -> None:
    """Show confirmation and write to vault if confirmed (D-44)."""
    if not engine.values:
        typer.echo("No fields filled. Note not created.")
        return

    typer.echo(f"\n{engine.format_confirmation()}")
    typer.echo(f"\nVault path: {engine.get_vault_path()}")

    try:
        from pb.cli.pickers import pick_boolean
        confirm = pick_boolean("Write to vault?")
    except (EOFError, KeyboardInterrupt):
        confirm = False

    if not confirm:
        typer.echo("Cancelled.")
        return

    # Phase 16 D-16-07: concept notes route through write_concept_note for D-16-07 YAML schema.
    # All other schema types fall through to the legacy vault_write path.
    if engine.schema.name.lower() == "concept":
        from pb.vault.concept_note import write_concept_note
        from pb.storage.repository import Repository
        from pb.vault import get_vault_path as _get_vault_path

        title = engine.values.get("title", "")
        domain = engine.values.get("domain", "")
        body = engine.values.get("definition", "")

        try:
            vault_root = _get_vault_path()
        except Exception:
            vault_root = None

        try:
            repo = Repository()
        except Exception:
            repo = None

        try:
            rel_path, qc_candidate = write_concept_note(
                title,
                domain,
                body,
                repo=repo,
                vault_root=str(vault_root) if vault_root else ".",
            )
            typer.echo(f"Created: {rel_path}")
            if qc_candidate:
                typer.echo(
                    f"[QC] Note body is {len(body)} chars (>1000). "
                    "Consider splitting into smaller atomic concepts."
                )
        except Exception as e:
            typer.echo(f"Failed to write concept note: {e}", err=True)
            raise typer.Exit(code=2)
        return

    # Import locally to avoid circular import issues at module level
    from pb.mcp.tools.vault import vault_write

    content = engine.generate_note_content()
    vault_path = engine.get_vault_path()

    try:
        vault_write(vault_path, content)
        typer.echo(f"Created: {vault_path}")
    except Exception as e:
        typer.echo(f"Failed to write note: {e}", err=True)
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Back-compat per-type subcommands (D-02: retained for existing usage)
# book subcommand removed (D-02: moved to plugin shelf)
# socratic subcommand removed (D-04: replaced by top-level verb)
# ---------------------------------------------------------------------------

@app.command("concept")
def concept_add():
    """Create a new Concept note (D-46)."""
    _run_question_tree("concept")


@app.command("person")
def person_add():
    """Create a new Person note (D-46)."""
    _run_question_tree("person")


@app.command("opp")
def opp_add():
    """Create a new Opportunity note (D-46)."""
    _run_question_tree("opportunity")
