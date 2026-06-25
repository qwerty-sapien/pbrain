#!/usr/bin/env python3
"""Scaffold Generator — bulk-create a research note graph for a knowledge domain.

Usage:
    python scripts/scaffold_generator.py [--sync | --async] [--retry MANIFEST]

Phases (implemented across plans 20-01 through 20-03):
    Phase 0a: Entry, domain detection, GCS setup wizard  <- Plan 20-01 (this file)
    Phase 0b: Flash intake conversation                  <- Plan 20-02
    Phase 1:  Note plan generation via Gemini Pro        <- Plan 20-02
    Phase 2:  Vertex Batch / sync note generation        <- Plan 20-02
    Phase 3:  Link resolution                            <- Plan 20-03
    Phase 4:  Validation + vault write                   <- Plan 20-03
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from pb.storage.config import get_config, get_vault_path, set_config_value
from pb.llm.gemini import get_client, FLASH_MODEL, PRO_MODEL
from pb.vault.graph import bulk_vault_write
from pb.vault.graph_store import open_vault_db, upsert_node, add_link
from pb.vault.embeddings import EmbeddingStore
from pb.vault.lifecycle import read_frontmatter, write_frontmatter
from pb.core.graph_writer import GraphWriter


# ---------------------------------------------------------------------------
# ScaffoldContext — shared state passed through all scaffold phases
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldContext:
    """Shared state passed through all scaffold phases."""

    vault_path: Path
    domain: str
    domain_dir: Path
    knowledge_dir: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    learning_profile: dict = field(default_factory=dict)
    note_plan: list[dict] = field(default_factory=list)
    generated_notes: list[tuple[str, str]] = field(default_factory=list)
    failed_notes: list[dict] = field(default_factory=list)
    sync_mode: bool = False
    retry_manifest_path: Optional[Path] = None
    gcs_bucket: str = ""
    gcp_project: str = ""
    location: str = "us-central1"


# ---------------------------------------------------------------------------
# Domain inference helpers (D-01)
# ---------------------------------------------------------------------------


def _detect_domain_from_cwd(knowledge_dir: Path) -> Optional[str]:
    """Detect domain from current working directory.

    Checks if the process cwd sits inside a valid domain subdirectory
    (i.e. a direct child of knowledge_dir that contains _state.md).
    Prevents path-traversal by using Path.relative_to().
    """
    try:
        cwd = Path(os.getcwd())
        relative = cwd.relative_to(knowledge_dir)
        parts = relative.parts
        if parts:
            candidate = knowledge_dir / parts[0]
            # T-20-01: validate against iterdir results — only accept dirs with _state.md
            if candidate.is_dir() and (candidate / "_state.md").exists():
                return parts[0]
    except (ValueError, OSError):
        pass
    return None


def _pick_domain(knowledge_dir: Path, console: Console) -> Optional[str]:
    """Show numbered domain picker as fallback (D-01).

    T-20-02: Index-based selection from enumerated valid dirs — user cannot
    inject arbitrary paths.
    """
    domains: list[str] = []
    try:
        for d in sorted(knowledge_dir.iterdir()):
            if d.is_dir() and (d / "_state.md").exists():
                domains.append(d.name)
    except OSError as exc:
        console.print(f"[red]Cannot list knowledge directory: {exc}[/]")
        return None

    if not domains:
        console.print("[dim]No domains with _state.md found in knowledge directory.[/]")
        return None

    if len(domains) == 1:
        console.print(f"[dim]Using only domain: {domains[0]}[/]")
        return domains[0]

    console.print("[bold]Select domain:[/]")
    for i, d in enumerate(domains, 1):
        console.print(f"  [dim]{i}[/]  {d}")

    try:
        choice = input("> ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(domains):
            return domains[idx]
        console.print("[red]Choice out of range.[/]")
    except (ValueError, EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/]")

    return None


# ---------------------------------------------------------------------------
# GCS setup wizard (D-15)
# ---------------------------------------------------------------------------


class GCSSetupWizard:
    """Interactive wizard to configure GCS bucket and GCP project for async mode.

    Prompts the user on first async use and persists values to config.toml via
    set_config_value(). T-20-03: bucket and project are non-secret identifiers.
    """

    def run(self, console: Console) -> tuple[str, str, str]:
        """Run the wizard and return (gcs_bucket, gcp_project, location).

        If config already populated, offers to reuse existing values.
        Saves accepted values to [scaffold] section of config.toml.
        """
        cfg = get_config().scaffold
        existing_bucket = cfg.gcs_bucket
        existing_project = cfg.gcp_project
        existing_location = cfg.location

        if existing_bucket and existing_project:
            console.print(
                f"\n[bold]Existing GCS config found:[/]\n"
                f"  project : [cyan]{existing_project}[/]\n"
                f"  bucket  : [cyan]{existing_bucket}[/]\n"
                f"  location: [cyan]{existing_location}[/]"
            )
            use_existing = Confirm.ask("Use existing config?", default=True)
            if use_existing:
                self._warn_if_no_auth(existing_project, console)
                return existing_bucket, existing_project, existing_location

        # Prompt for values
        console.print("\n[bold]GCS Setup — Scaffold Generator[/]")
        console.print("[dim]Values will be saved to ~/.config/productivebrain/config.toml[/]\n")

        default_project = existing_project or os.environ.get("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
        gcp_project = self._prompt("GCP project ID", default_project)

        default_bucket = existing_bucket or f"{gcp_project}-scaffold"
        gcs_bucket = self._prompt("GCS bucket name", default_bucket)

        default_location = existing_location or "us-central1"
        location = self._prompt("GCS location", default_location)

        # Persist to config.toml
        set_config_value("scaffold", "gcs_bucket", gcs_bucket)
        set_config_value("scaffold", "gcp_project", gcp_project)
        set_config_value("scaffold", "location", location)

        console.print("[green]GCS config saved.[/]\n")
        self._warn_if_no_auth(gcp_project, console)
        return gcs_bucket, gcp_project, location

    @staticmethod
    def _prompt(label: str, default: str) -> str:
        """Prompt user with a default value; return input or default on empty."""
        try:
            raw = input(f"{label} [{default}]: ").strip()
            return raw if raw else default
        except (EOFError, KeyboardInterrupt):
            return default

    @staticmethod
    def _warn_if_no_auth(gcp_project: str, console: Console) -> None:
        """Warn if GOOGLE_CLOUD_PROJECT is not set and project looks unconfigured."""
        env_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not env_project and not gcp_project:
            console.print(
                "[yellow]Warning: GOOGLE_CLOUD_PROJECT env var is not set. "
                "Run `gcloud auth application-default login` to authenticate.[/]"
            )


# ---------------------------------------------------------------------------
# IntakePhase — Flash-driven learning advisor interview (D-03, D-04, D-05)
# ---------------------------------------------------------------------------


class IntakePhase:
    """Flash-driven learning advisor interview."""

    _STOP_SIGNAL = "__INTAKE_COMPLETE__"

    def __init__(self, domain: str, state_content: str):
        self._domain = domain
        self._state_content = state_content
        self._history: list[dict[str, str]] = []
        self._client = get_client()

    def _build_question_prompt(self) -> str:
        history_text = "\n".join(
            f"  {'Advisor' if e['role'] == 'advisor' else 'Learner'}: {e['text']}"
            for e in self._history
        )
        return (
            f"You are a learning advisor helping someone build a deep study scaffold.\n"
            f"Domain: {self._domain}\n"
            f"Domain state context:\n{self._state_content[:1500]}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            "Your task:\n"
            "- If you have enough information (background level, goals, depth, application), "
            f"respond ONLY with: {self._STOP_SIGNAL}\n"
            "- Otherwise, ask ONE natural follow-up question.\n"
            "- Topics to cover: current understanding, learning goal, application context, "
            "desired depth (general / undergrad / PhD / professional).\n"
            "- Be conversational. Do not list all topics at once.\n"
            "Rules: One question max. No preamble. No numbering."
        )

    def run(self, console: Console) -> list[tuple[str, str]]:
        """Run intake conversation. Returns Q&A pairs."""
        console.print("\n[bold]Learning Advisor Interview[/]")
        console.print("[dim]Answer freely — no right answers. Type 'done' or Ctrl+C to end early.[/]\n")
        pairs: list[tuple[str, str]] = []
        while True:
            prompt = self._build_question_prompt()
            question = self._client.generate_with_model(prompt, FLASH_MODEL, timeout=20)
            if not question or self._STOP_SIGNAL in question:
                break
            question = question.strip()
            self._history.append({"role": "advisor", "text": question})
            console.print(f"[bold cyan]Advisor:[/] {question}")
            try:
                answer = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if answer.lower() in ("done", "/done", "/exit", ""):
                break
            self._history.append({"role": "learner", "text": answer})
            pairs.append((question, answer))
        return pairs

    def compress_to_profile(self, pairs: list[tuple[str, str]]) -> dict:
        """Compress Q&A into structured learning profile (D-05)."""
        if not pairs:
            return {
                "depth": "general",
                "background": "unknown",
                "goals": [],
                "application": "",
                "emphasis": [],
                "vocabulary_level": "accessible",
            }
        transcript = "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
        prompt = (
            f"Compress this learning intake transcript into a JSON profile.\n"
            f"Domain: {self._domain}\n"
            f"Transcript:\n{transcript}\n\n"
            "Return ONLY valid JSON with keys:\n"
            '  "depth": one of ["general", "undergrad", "phd", "professional"],\n'
            '  "background": brief string describing current knowledge,\n'
            '  "goals": list of strings (learning objectives),\n'
            '  "application": string (how they plan to apply this),\n'
            '  "emphasis": list of strings (cross-domain connections to emphasize),\n'
            '  "vocabulary_level": one of ["accessible", "technical", "rigorous"]\n'
            "No explanation, just JSON."
        )
        result = self._client.generate_with_model(prompt, FLASH_MODEL, timeout=30)
        try:
            return json.loads(result.strip()) if result else {}
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if result and "```" in result:
                json_block = result.split("```")[1]
                if json_block.startswith("json"):
                    json_block = json_block[4:]
                try:
                    return json.loads(json_block.strip())
                except json.JSONDecodeError:
                    pass
            return {
                "depth": "general",
                "background": "unknown",
                "goals": [],
                "application": "",
                "emphasis": [],
                "vocabulary_level": "accessible",
            }


# ---------------------------------------------------------------------------
# PlanPhase — Generate note plan and get user approval (D-06, D-07, D-09, D-12, SCAF-02)
# ---------------------------------------------------------------------------


class PlanPhase:
    """Generate note plan and get user approval."""

    def __init__(self, ctx: ScaffoldContext):
        self._ctx = ctx
        self._client = get_client()

    def _build_plan_prompt(self) -> str:
        profile_json = json.dumps(self._ctx.learning_profile, indent=2)
        return (
            f"You are planning a research note scaffold for domain: {self._ctx.domain}\n"
            f"Learning profile:\n{profile_json}\n\n"
            f"Generate a note graph plan as a JSON array. Each note entry:\n"
            '{{"path": "<domain>/<slug>.md", "type": "concept|technique|example|overview", '
            '"title": "<note title>", "intra_links": ["<other-slug>", ...], '
            '"prompt_hint": "<what this note should cover>"}}\n\n'
            "Rules:\n"
            f"- All paths must be under {self._ctx.domain}/\n"
            "- Each note is atomic: ONE concept per note\n"
            "- Include 10-30 notes depending on domain breadth and depth\n"
            "- intra_links reference other slugs in THIS plan (Pro will generate wikilinks)\n"
            "- Include overview notes that connect sub-topics\n"
            "- Consider cross-domain connections mentioned in emphasis\n"
            "- Return ONLY valid JSON array, no explanation"
        )

    def _dedup_against_vault(self, plan: list[dict]) -> tuple[list[dict], list[str]]:
        """Check plan against existing vault embeddings (D-12).

        Returns (kept, skipped) where skipped is a list of human-readable strings.
        Gracefully degrades if EmbeddingStore is unavailable.
        """
        store = EmbeddingStore(self._ctx.vault_path)
        if not store.available:
            return plan, []  # graceful degradation

        kept: list[dict] = []
        skipped: list[str] = []
        for note in plan:
            candidates = store.query_similarity(
                note.get("title", "") + " " + note.get("prompt_hint", ""),
                k=5,
            )
            # If a very similar note already exists (>0.85 similarity), skip
            high_match = [(slug, sim) for slug, sim in candidates if sim > 0.85]
            if high_match:
                skipped.append(
                    f"{note['path']} (similar to {high_match[0][0]}, sim={high_match[0][1]:.2f})"
                )
            else:
                kept.append(note)
        return kept, skipped

    def run(self, console: Console) -> list[dict]:
        """Generate plan and get user approval (SCAF-02). Returns approved plan."""
        console.print("\n[bold]Generating note plan...[/]")
        prompt = self._build_plan_prompt()
        result = self._client.generate_with_model(prompt, FLASH_MODEL, timeout=60)
        if not result:
            console.print("[red]Failed to generate note plan[/]")
            sys.exit(1)

        # Parse JSON
        try:
            plan = json.loads(result.strip())
        except json.JSONDecodeError:
            # Try extracting from code block
            if "```" in result:
                json_block = result.split("```")[1]
                if json_block.startswith("json"):
                    json_block = json_block[4:]
                try:
                    plan = json.loads(json_block.strip())
                except json.JSONDecodeError:
                    console.print("[red]Failed to parse note plan JSON[/]")
                    sys.exit(1)
            else:
                console.print("[red]Failed to parse note plan JSON[/]")
                sys.exit(1)

        if not isinstance(plan, list) or not plan:
            console.print("[red]Note plan is empty or invalid[/]")
            sys.exit(1)

        # Dedup (D-12)
        kept, skipped = self._dedup_against_vault(plan)
        if skipped:
            console.print(f"\n[yellow]Deduplication: {len(skipped)} notes skipped (similar exists):[/]")
            for s in skipped:
                console.print(f"  [dim]{s}[/]")

        if not kept:
            console.print("[yellow]All planned notes already exist in vault (dedup). Nothing to generate.[/]")
            sys.exit(0)

        # Display plan as Rich table
        table = Table(title=f"Note Plan ({len(kept)} notes)", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Path", style="cyan")
        table.add_column("Type", width=10)
        table.add_column("Title")
        table.add_column("Links", width=20)
        for i, note in enumerate(kept, 1):
            links_str = ", ".join(note.get("intra_links", [])[:3])
            if len(note.get("intra_links", [])) > 3:
                links_str += "..."
            table.add_row(
                str(i),
                note["path"],
                note.get("type", "?"),
                note.get("title", "?"),
                links_str,
            )
        console.print(table)

        # Explicit approval gate — Enter alone does NOT proceed (Pitfall 4)
        if not Confirm.ask("\nApprove this note plan?", default=False):
            console.print("[dim]Aborted.[/]")
            sys.exit(0)

        return kept


# ---------------------------------------------------------------------------
# GeneratePhase helpers and constants (D-08, D-13, D-14, SCAF-03, SCAF-04)
# ---------------------------------------------------------------------------

TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
RETRY_MANIFEST_DIR = Path(".pb")


def _build_system_instruction(ctx: ScaffoldContext) -> str:
    """Build the system instruction for Pro note generation from learning profile (D-05, D-07)."""
    profile = ctx.learning_profile
    return (
        f"You are generating an atomic research note for domain: {ctx.domain}.\n"
        f"Learner profile:\n"
        f"- Depth: {profile.get('depth', 'general')}\n"
        f"- Background: {profile.get('background', 'unknown')}\n"
        f"- Goals: {', '.join(profile.get('goals', []))}\n"
        f"- Application: {profile.get('application', '')}\n"
        f"- Cross-domain emphasis: {', '.join(profile.get('emphasis', []))}\n"
        f"- Vocabulary: {profile.get('vocabulary_level', 'accessible')}\n\n"
        "Output format rules:\n"
        "- Start with YAML frontmatter (--- delimited) containing: title, tags (list), learning_stage: new\n"
        "- Body uses bullet points with MAXIMUM 30 words per bullet\n"
        "- Note is ATOMIC: covers ONE concept only\n"
        "- Total body must be UNDER 2200 characters (excluding frontmatter and wikilinks section)\n"
        "- Include [[wikilinks]] to related notes (provided in the prompt)\n"
        "- Include citations where applicable (author, year) as inline references\n"
        "- End with a ## Links section listing all wikilinks\n"
        "- Use tags from: #new (always include), plus domain-relevant tags\n"
        "Do NOT add explanation outside the note content. Output the note ONLY."
    )


def _build_batch_line(key: str, note_prompt: str, system_instruction: str) -> str:
    """Build one JSONL line for Vertex Batch (D-08, SCAF-03)."""
    request = {
        "key": key,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": note_prompt}]}],
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024},
        },
    }
    return json.dumps(request)


def _build_note_prompt(note_spec: dict, all_notes: list[dict]) -> str:
    """Build the user prompt for a single note generation request."""
    intra_links = note_spec.get("intra_links", [])
    link_context = ""
    if intra_links:
        related = [n for n in all_notes if any(slug in n.get("path", "") for slug in intra_links)]
        link_context = "\nRelated notes in this scaffold (use [[wikilinks]] to reference):\n"
        for r in related[:10]:
            slug = Path(r["path"]).stem
            link_context += f"  - [[{slug}]]: {r.get('title', '')} ({r.get('type', '')})\n"

    return (
        f"Generate a note: {note_spec.get('title', 'Untitled')}\n"
        f"Type: {note_spec.get('type', 'concept')}\n"
        f"Slug: {Path(note_spec['path']).stem}\n"
        f"Hint: {note_spec.get('prompt_hint', '')}\n"
        f"{link_context}"
    )


def _cost_gate(genai_client, prompts: list[str], console: Console) -> bool:
    """Estimate batch cost and require user confirmation (D-14)."""
    total_tokens = 0
    console.print("\n[dim]Counting tokens for cost estimate...[/]")
    for prompt in prompts:
        try:
            resp = genai_client.models.count_tokens(
                model=PRO_MODEL,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            total_tokens += resp.total_tokens
        except Exception:
            # Fallback: ~1 token per 4 characters
            total_tokens += len(prompt) // 4

    console.print("\n[bold]Cost estimate:[/]")
    console.print(f"  Input tokens: ~{total_tokens:,}")
    console.print(f"  Notes to generate: {len(prompts)}")
    console.print("  [dim]Batch pricing = 50% off on-demand. Final cost depends on output length.[/]")
    console.print("  [dim]See: cloud.google.com/vertex-ai/pricing[/]")
    console.print()
    return Confirm.ask("Proceed with Vertex Batch submission?", default=False)


def _parse_batch_output(jsonl_text: str) -> tuple[list[dict], list[dict]]:
    """Parse batch output JSONL. Returns (successes, failures) (D-13).

    Successes have status="" (empty string or absent); failures have non-empty status.
    Handles unparseable lines gracefully (T-20-09).
    """
    successes, failures = [], []
    for line in jsonl_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            failures.append({"key": "unknown", "error": "unparseable line"})
            continue
        status = obj.get("status", "")
        if status == "":
            successes.append(obj)
        else:
            failures.append({"key": obj.get("key"), "error": status, "raw": obj})
    return successes, failures


def _save_retry_manifest(vault_path: Path, failed: list[dict], run_id: str, note_plan: list[dict]) -> Path:
    """Save failed note specs for --retry resubmission (D-13)."""
    manifest_dir = vault_path / RETRY_MANIFEST_DIR
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "scaffold-retry.json"
    # Include original note_plan entries that match failed keys
    failed_keys = {f.get("key") for f in failed}
    failed_specs = [n for n in note_plan if Path(n["path"]).stem in failed_keys]
    manifest = {
        "run_id": run_id,
        "failed_keys": list(failed_keys),
        "failed_specs": failed_specs,
        "errors": failed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _load_retry_manifest(path: Path) -> tuple[str, list[dict]]:
    """Load retry manifest. Returns (original_run_id, failed_note_specs).

    T-20-07: Validates expected keys before use.
    """
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read retry manifest {path}: {exc}") from exc
    # T-20-07: validate expected structure
    if "run_id" not in data or "failed_specs" not in data:
        raise ValueError(f"Retry manifest missing required keys (run_id, failed_specs): {path}")
    return data["run_id"], data["failed_specs"]


# ---------------------------------------------------------------------------
# GeneratePhase — Generate notes via Vertex Batch (async) or direct API (sync)
# ---------------------------------------------------------------------------


class GeneratePhase:
    """Generate notes via Vertex Batch (async) or direct API (sync)."""

    def __init__(self, ctx: ScaffoldContext):
        self._ctx = ctx

    def _generate_sync(self, console: Console) -> tuple[list[tuple[str, str]], list[dict]]:
        """Sync mode: generate notes one at a time via generate_with_model (--sync)."""
        client = get_client()
        system_instruction = _build_system_instruction(self._ctx)
        generated: list[tuple[str, str]] = []
        failed: list[dict] = []

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Generating notes...", total=len(self._ctx.note_plan))
            for note_spec in self._ctx.note_plan:
                slug = Path(note_spec["path"]).stem
                note_prompt = _build_note_prompt(note_spec, self._ctx.note_plan)
                full_prompt = f"{system_instruction}\n\n{note_prompt}"
                result = client.generate_with_model(full_prompt, PRO_MODEL, timeout=120)
                if result and result.strip():
                    generated.append((note_spec["path"], result.strip()))
                else:
                    failed.append({"key": slug, "error": "empty response"})
                progress.advance(task)
        return generated, failed

    def _generate_async(self, console: Console) -> tuple[list[tuple[str, str]], list[dict]]:
        """Async mode: Vertex Batch via GCS (D-08, SCAF-04)."""
        from google import genai as genai_sdk
        from google.genai.types import CreateBatchJobConfig
        from google.cloud import storage as gcs_sdk

        system_instruction = _build_system_instruction(self._ctx)

        # Build JSONL lines
        jsonl_lines: list[str] = []
        prompts_for_cost: list[str] = []
        for note_spec in self._ctx.note_plan:
            note_prompt = _build_note_prompt(note_spec, self._ctx.note_plan)
            prompts_for_cost.append(f"{system_instruction}\n\n{note_prompt}")
            slug = Path(note_spec["path"]).stem
            line = _build_batch_line(slug, note_prompt, system_instruction)
            jsonl_lines.append(line)

        # Create genai client for Vertex
        genai_client = genai_sdk.Client(
            vertexai=True,
            project=self._ctx.gcp_project,
            location=self._ctx.location,
        )

        # Cost gate (D-14)
        if not _cost_gate(genai_client, prompts_for_cost, console):
            console.print("[dim]Batch submission cancelled.[/]")
            sys.exit(0)

        # Upload to GCS
        console.print("\n[dim]Uploading JSONL to GCS...[/]")
        gcs_client = gcs_sdk.Client(project=self._ctx.gcp_project)
        bucket = gcs_client.bucket(self._ctx.gcs_bucket)
        input_blob = bucket.blob(f"scaffold/{self._ctx.run_id}/input.jsonl")
        input_blob.upload_from_string(
            "\n".join(jsonl_lines),
            content_type="application/jsonl",
        )
        input_uri = f"gs://{self._ctx.gcs_bucket}/scaffold/{self._ctx.run_id}/input.jsonl"
        output_uri_prefix = f"gs://{self._ctx.gcs_bucket}/scaffold/{self._ctx.run_id}/output/"

        # Submit batch job
        console.print(f"[dim]Submitting batch job ({len(jsonl_lines)} notes)...[/]")
        job = genai_client.batches.create(
            model=PRO_MODEL,
            src=input_uri,
            config=CreateBatchJobConfig(dest=output_uri_prefix),
        )
        console.print(f"[dim]Job: {job.name}[/]")

        # Poll every 90s (SCAF-04)
        cfg = get_config()
        poll_interval = cfg.scaffold.poll_interval_seconds or 90
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            poll_task = progress.add_task("Waiting for Vertex Batch...", total=None)
            state = ""
            while True:
                job = genai_client.batches.get(name=job.name)
                state = job.state.name
                progress.update(poll_task, description=f"Batch: {state}")
                if state in TERMINAL_STATES:
                    break
                time.sleep(poll_interval)

        if state != "JOB_STATE_SUCCEEDED":
            console.print(f"[red]Batch job ended with state: {state}[/]")
            if state == "JOB_STATE_FAILED":
                console.print("[yellow]Check GCS output prefix for error details[/]")
            return [], [{"key": "batch", "error": f"Job state: {state}"}]

        # Download output JSONL from GCS
        console.print("[dim]Downloading results from GCS...[/]")
        output_blobs = list(bucket.list_blobs(prefix=f"scaffold/{self._ctx.run_id}/output/"))
        all_output = ""
        for blob in output_blobs:
            if blob.name.endswith(".jsonl"):
                all_output += blob.download_as_text() + "\n"

        # Parse output (D-13 — handle partial failures, T-20-09)
        successes, failures = _parse_batch_output(all_output)

        # Convert successes to (path, content) tuples
        generated: list[tuple[str, str]] = []
        key_to_path = {Path(n["path"]).stem: n["path"] for n in self._ctx.note_plan}
        for obj in successes:
            key = obj.get("key", "")
            path = key_to_path.get(key, f"{self._ctx.domain}/{key}.md")
            # Extract generated text from response (T-20-09: handle malformed structure)
            try:
                parts = obj["response"]["candidates"][0]["content"]["parts"]
                content = "".join(p.get("text", "") for p in parts)
                generated.append((path, content.strip()))
            except (KeyError, IndexError):
                failures.append({"key": key, "error": "malformed response structure"})

        return generated, failures

    def run(self, console: Console) -> None:
        """Run generation phase. Updates ctx.generated_notes and ctx.failed_notes."""
        if self._ctx.sync_mode:
            generated, failed = self._generate_sync(console)
        else:
            generated, failed = self._generate_async(console)

        self._ctx.generated_notes = generated
        self._ctx.failed_notes = failed

        console.print(f"\n[bold]Generation complete:[/] {len(generated)} succeeded, {len(failed)} failed")

        # Save retry manifest if there were failures (D-13)
        if failed:
            manifest_path = _save_retry_manifest(
                self._ctx.vault_path, failed, self._ctx.run_id, self._ctx.note_plan
            )
            console.print(f"[yellow]Retry manifest saved: {manifest_path}[/]")
            console.print(
                f"[dim]Re-run with: python scripts/scaffold_generator.py --retry {manifest_path}[/]"
            )


# ---------------------------------------------------------------------------
# Validation helpers (D-16, D-17)
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^[-*]\s+(.+)$", re.MULTILINE)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


@dataclass
class ValidationResult:
    """Result of validating a single generated note."""

    is_broken: bool = False
    warnings: list[str] = field(default_factory=list)


def _validate_note(rel_path: str, content: str) -> ValidationResult:
    """Lenient validation (D-16). Broken = unparseable/empty -> quarantine (D-17).

    Checks: frontmatter present, body <=2200 chars, has wikilinks, has tags, bullets <=30 words.
    Violations produce warnings only — do NOT block write.
    """
    result = ValidationResult()

    # Truly broken: empty or no content
    if not content or not content.strip():
        result.is_broken = True
        return result

    # Parse frontmatter
    fm, body = read_frontmatter(content)

    # Broken: no body at all after frontmatter
    if not body.strip():
        result.is_broken = True
        return result

    # Warnings (lenient — don't block write)
    if not fm:
        result.warnings.append(f"{rel_path}: no frontmatter")

    # Body length check (exclude links section from count)
    body_for_length = body
    links_idx = body.find("## Links")
    if links_idx > 0:
        body_for_length = body[:links_idx]
    body_len = len(body_for_length.strip())
    if body_len > 2200:
        result.warnings.append(f"{rel_path}: body {body_len} chars > 2200 limit")

    # Wikilinks check
    if not _WIKILINK_RE.search(content):
        result.warnings.append(f"{rel_path}: no wikilinks found")

    # Tags check
    tags = fm.get("tags") if fm else None
    if not tags:
        result.warnings.append(f"{rel_path}: no tags in frontmatter")

    # Bullet length check
    for m in _BULLET_RE.finditer(body):
        words = len(m.group(1).split())
        if words > 30:
            result.warnings.append(f"{rel_path}: bullet > 30 words: '{m.group(1)[:50]}...'")
            break  # Only report first violation per note

    return result


# ---------------------------------------------------------------------------
# LinkResolutionPhase — cross-domain link resolution (D-10, D-11)
# ---------------------------------------------------------------------------


class LinkResolutionPhase:
    """Cross-domain link resolution: embeddings (2a) + Flash validation (2b)."""

    def __init__(self, ctx: ScaffoldContext):
        self._ctx = ctx
        self._client = get_client()
        self._threshold = get_config().scaffold.similarity_threshold or 0.6

    def run(self, console: Console) -> None:
        """Resolve cross-domain links for all generated notes. Mutates ctx.generated_notes in place."""
        store = EmbeddingStore(self._ctx.vault_path)

        if not store.available:
            console.print(
                "[dim]sqlite-vec unavailable — skipping embedding pass, Flash-only link resolution[/]"
            )
            self._flash_only_pass(console)
            return

        console.print("\n[bold]Resolving cross-domain links...[/]")
        updated_notes: list[tuple[str, str]] = []

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Processing...", total=len(self._ctx.generated_notes))
            for rel_path, content in self._ctx.generated_notes:
                # Pass 2a: embedding similarity scan
                candidates = store.query_similarity(content[:500], k=20)
                filtered = [(slug, sim) for slug, sim in candidates if sim > self._threshold]

                if filtered:
                    # Pass 2b: Flash validates and adds analogical links
                    new_content = self._flash_validate_links(content, rel_path, filtered)
                    updated_notes.append((rel_path, new_content))
                else:
                    updated_notes.append((rel_path, content))
                progress.advance(task)

        self._ctx.generated_notes = updated_notes

    def _flash_validate_links(
        self, content: str, rel_path: str, candidates: list[tuple[str, float]]
    ) -> str:
        """Flash validates embedding candidates and adds valid cross-domain links (D-11)."""
        candidate_text = "\n".join(
            f"  - [[{slug}]] (similarity={sim:.2f})" for slug, sim in candidates[:15]
        )
        prompt = (
            f"You are linking research notes across knowledge domains.\n"
            f"New note ({rel_path}):\n{content[:1200]}\n\n"
            f"Semantic neighbor candidates from existing vault:\n{candidate_text}\n\n"
            "Task:\n"
            "1. From the candidates, list slugs that have a REAL conceptual connection to this note.\n"
            "2. Also suggest any analogical connections the embeddings missed "
            "(e.g., 'Euler method ↔ gradient descent step — both iterative numerical approximation').\n\n"
            'Return ONLY valid JSON: {"valid": ["slug1", "slug2"], "analogical": ["slug3"]}\n'
            'If no valid connections, return: {"valid": [], "analogical": []}'
        )
        result = self._client.generate_with_model(prompt, FLASH_MODEL, timeout=20)
        if not result:
            return content

        try:
            links_data = json.loads(result.strip())
        except json.JSONDecodeError:
            # Try extracting from code block
            if "```" in result:
                json_block = result.split("```")[1]
                if json_block.startswith("json"):
                    json_block = json_block[4:]
                try:
                    links_data = json.loads(json_block.strip())
                except json.JSONDecodeError:
                    return content
            else:
                return content

        # Append new cross-domain links to content
        valid_links = links_data.get("valid", [])
        analogical_links = links_data.get("analogical", [])
        all_new_links = valid_links + analogical_links

        if not all_new_links:
            return content

        # Check which links are NOT already in the content
        existing_links = set(_WIKILINK_RE.findall(content))
        new_links = [lnk for lnk in all_new_links if lnk not in existing_links]

        if not new_links:
            return content

        # Append to ## Links section or create one
        links_section = "\n## Cross-Domain Links\n"
        links_section += "\n".join(f"- [[{lnk}]]" for lnk in new_links)

        if "## Links" in content:
            content = content.rstrip() + "\n" + links_section + "\n"
        elif "## Cross-Domain Links" in content:
            content = content.rstrip() + "\n" + "\n".join(f"- [[{lnk}]]" for lnk in new_links) + "\n"
        else:
            content = content.rstrip() + "\n\n" + links_section + "\n"

        return content

    def _flash_only_pass(self, console: Console) -> None:
        """Fallback: no-op when sqlite-vec is unavailable."""
        updated_notes: list[tuple[str, str]] = []
        for rel_path, content in self._ctx.generated_notes:
            updated_notes.append((rel_path, content))
        self._ctx.generated_notes = updated_notes
        console.print(
            "[dim]Flash-only fallback: no cross-domain links added "
            "(install sqlite-vec for full resolution)[/]"
        )


# ---------------------------------------------------------------------------
# ValidateWritePhase — validate, quarantine broken notes, write vault (SCAF-05, SCAF-06)
# ---------------------------------------------------------------------------


class ValidateWritePhase:
    """Validate notes and write to vault with state updates."""

    def __init__(self, ctx: ScaffoldContext):
        self._ctx = ctx

    def run(self, console: Console) -> int:
        """Validate, quarantine broken notes, write valid ones, update state. Returns count written."""
        console.print("\n[bold]Validating generated notes...[/]")

        valid_notes: list[tuple[str, str]] = []
        quarantined: list[tuple[str, str]] = []
        all_warnings: list[str] = []

        for rel_path, content in self._ctx.generated_notes:
            result = _validate_note(rel_path, content)
            if result.is_broken:
                quarantined.append((rel_path, content))
            else:
                valid_notes.append((rel_path, content))
                all_warnings.extend(result.warnings)

        # Report validation results
        if all_warnings:
            console.print(f"\n[yellow]Validation warnings ({len(all_warnings)}):[/]")
            for w in all_warnings[:20]:
                console.print(f"  [dim]{w}[/]")
            if len(all_warnings) > 20:
                console.print(f"  [dim]... and {len(all_warnings) - 20} more[/]")

        # Quarantine broken notes (D-17)
        if quarantined:
            quarantine_dir = self._ctx.vault_path / ".pb" / "scaffold-quarantine"
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            for rel_path, content in quarantined:
                q_path = quarantine_dir / Path(rel_path).name
                q_path.write_text(content if content else "# EMPTY NOTE\n")
            console.print(f"[red]Quarantined {len(quarantined)} broken notes to .pb/scaffold-quarantine/[/]")

        if not valid_notes:
            console.print("[red]No valid notes to write.[/]")
            return 0

        # Write to vault via bulk_vault_write — single call only (SCAF-05, GRPH-04)
        console.print(f"\n[dim]Writing {len(valid_notes)} notes to vault...[/]")
        written = bulk_vault_write(self._ctx.vault_path, valid_notes)

        # Register nodes + links in vault.db only (D-18)
        # IMPORTANT: Use open_vault_db — NOT EmbeddingStore connection (Pitfall 2)
        console.print("[dim]Updating vault.db graph...[/]")
        gs_conn = open_vault_db(self._ctx.vault_path)
        try:
            for rel_path, content in valid_notes:
                slug = Path(rel_path).stem
                subfolder = str(Path(rel_path).parent)
                upsert_node(gs_conn, slug, subfolder)
                # Extract wikilinks and register as edges
                for linked_slug in _WIKILINK_RE.findall(content):
                    try:
                        add_link(gs_conn, slug, linked_slug)
                    except (KeyError, ValueError):
                        pass  # Non-fatal: target node may not exist yet
                    except Exception as exc:
                        console.print(f"[dim]Warning: link {slug}->{linked_slug} failed: {exc}[/]")
            gs_conn.commit()
        finally:
            gs_conn.close()

        # Store embeddings for new notes (separate EmbeddingStore connection — Pitfall 2)
        store = EmbeddingStore(self._ctx.vault_path)
        if store.available:
            console.print("[dim]Storing embeddings for new notes...[/]")
            embed_failures = 0
            for rel_path, content in valid_notes:
                slug = Path(rel_path).stem
                try:
                    store.store_embedding(slug, content)
                except Exception as exc:
                    embed_failures += 1
                    if embed_failures == 1:
                        console.print(f"[yellow]Embedding store error: {exc}[/]")
            if embed_failures:
                console.print(f"[yellow]{embed_failures} embedding(s) failed to store[/]")

        # Update domain _state.md (SCAF-06)
        console.print("[dim]Updating domain _state.md...[/]")
        gw = GraphWriter(self._ctx.vault_path)
        gw.update_state_md(
            self._ctx.domain_dir,
            f"Scaffold: {written} notes generated (run {self._ctx.run_id})",
            self._ctx.vault_path,
        )

        # Update goal note links: frontmatter (SCAF-06)
        self._update_goal_note_links(valid_notes, console)

        # Final summary
        console.print("\n[bold green]Scaffold complete![/]")
        console.print(f"  Notes written: {written}")
        console.print(f"  Warnings: {len(all_warnings)}")
        console.print(f"  Quarantined: {len(quarantined)}")
        console.print(f"  Domain: {self._ctx.domain}")
        console.print(f"  Run ID: {self._ctx.run_id}")

        return written

    def _update_goal_note_links(self, valid_notes: list[tuple[str, str]], console: Console) -> None:
        """Update the goal note's links: frontmatter with new note slugs (SCAF-06)."""
        # Goal note is the domain's _index.md
        index_path = self._ctx.domain_dir / "_index.md"
        if not index_path.exists():
            console.print("[dim]No _index.md found — skipping goal note links update[/]")
            return

        try:
            content = index_path.read_text()
            fm, body = read_frontmatter(content)
            if fm is None:
                fm = {}

            # Add/extend links list in frontmatter
            existing_links = fm.get("links", [])
            if not isinstance(existing_links, list):
                existing_links = []

            new_slugs = [Path(rel_path).stem for rel_path, _ in valid_notes]
            for slug in new_slugs:
                if slug not in existing_links:
                    existing_links.append(slug)

            fm["links"] = existing_links
            updated_content = write_frontmatter(fm, body)
            index_path.write_text(updated_content)
            console.print(f"[dim]Updated _index.md links: +{len(new_slugs)} slugs[/]")
        except Exception as e:
            console.print(f"[yellow]Warning: could not update _index.md links: {e}[/]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Scaffold generator entry point."""
    parser = argparse.ArgumentParser(
        description="Scaffold Generator - bulk-create a research note graph for a knowledge domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/scaffold_generator.py            # async mode (default)\n"
            "  python scripts/scaffold_generator.py --sync     # synchronous generation\n"
            "  python scripts/scaffold_generator.py --retry manifest.json\n"
        ),
    )

    # D-02: --sync, --async (default), --retry flags
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sync",
        action="store_true",
        default=False,
        help="Run generation synchronously via direct Gemini API calls",
    )
    mode_group.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        default=False,
        help="Run generation asynchronously via Vertex Batch (default)",
    )
    parser.add_argument(
        "--retry",
        metavar="MANIFEST",
        help="Path to retry manifest JSON from a previous failed run",
    )

    args = parser.parse_args()

    # Async is the default when neither flag is given
    sync_mode = args.sync
    # (async_mode flag is the explicit form; default is also async)

    console = Console()

    # D-03: Require interactive terminal for Socratic intake
    if not sys.stdin.isatty():
        console.print("[red]Error: scaffold_generator requires an interactive terminal (stdin must be a TTY).[/]")
        sys.exit(1)

    # Check Gemini availability
    client = get_client()
    if not client.is_available():
        console.print(
            "[red]Error: Gemini client is not available. "
            "Ensure GEMINI_API_KEY is set or a valid Gemini config exists.[/]"
        )
        sys.exit(1)

    # Resolve vault and knowledge directory
    vault_path = get_vault_path()
    knowledge_dir = vault_path / "knowledge"

    if not knowledge_dir.exists():
        console.print(
            f"[red]Error: Knowledge directory not found: {knowledge_dir}[/]\n"
            "[dim]Create a knowledge/ folder in your vault with at least one domain subdirectory.[/]"
        )
        sys.exit(1)

    # D-01: Domain inference — cwd first, picker fallback
    domain = _detect_domain_from_cwd(knowledge_dir)
    if domain is None:
        domain = _pick_domain(knowledge_dir, console)

    if domain is None:
        console.print("[red]No domain selected. Aborting.[/]")
        sys.exit(1)

    domain_dir = knowledge_dir / domain

    # Handle --retry flag
    retry_manifest_path: Optional[Path] = None
    if args.retry:
        retry_path = Path(args.retry)
        # T-20-04: Validate retry manifest path exists and is a .json file
        if not retry_path.exists():
            console.print(f"[red]Error: Retry manifest not found: {retry_path}[/]")
            sys.exit(1)
        if retry_path.suffix.lower() != ".json":
            console.print(f"[red]Error: Retry manifest must be a .json file: {retry_path}[/]")
            sys.exit(1)
        retry_manifest_path = retry_path

    # GCS wizard — runs for async mode only (first use or missing config)
    gcs_bucket = ""
    gcp_project = ""
    location = "us-central1"

    if not sync_mode:
        wizard = GCSSetupWizard()
        gcs_bucket, gcp_project, location = wizard.run(console)

    # Build shared scaffold context
    ctx = ScaffoldContext(
        vault_path=vault_path,
        domain=domain,
        domain_dir=domain_dir,
        knowledge_dir=knowledge_dir,
        sync_mode=sync_mode,
        retry_manifest_path=retry_manifest_path,
        gcs_bucket=gcs_bucket,
        gcp_project=gcp_project,
        location=location,
    )

    # Startup banner
    mode_label = "sync" if sync_mode else "async (Vertex Batch)"
    console.print(
        f"\n[bold]Scaffold Generator[/] — run [cyan]{ctx.run_id}[/]\n"
        f"  domain  : [cyan]{ctx.domain}[/]\n"
        f"  mode    : [cyan]{mode_label}[/]\n"
        f"  vault   : [dim]{ctx.vault_path}[/]\n"
    )

    if retry_manifest_path:
        console.print(f"  retry   : [yellow]{retry_manifest_path}[/]\n")
        # --retry path: load manifest, skip intake+plan, go straight to generation
        orig_run_id, failed_specs = _load_retry_manifest(retry_manifest_path)
        ctx.note_plan = failed_specs
        ctx.run_id = f"{orig_run_id}-retry"
        ctx.learning_profile = {}  # profile not needed for retry (prompts already built)
        console.print(f"[bold]Retrying {len(failed_specs)} failed notes from run {orig_run_id}[/]")
        # Phase 2 (retry): Note generation
        generator = GeneratePhase(ctx)
        generator.run(console)
    else:
        # Phase 0b: Flash intake interview (D-03, D-04)
        state_path = ctx.domain_dir / "_state.md"
        state_content = state_path.read_text() if state_path.exists() else ""
        intake = IntakePhase(ctx.domain, state_content)
        pairs = intake.run(console)
        ctx.learning_profile = intake.compress_to_profile(pairs)
        console.print(f"\n[dim]Learning profile: {json.dumps(ctx.learning_profile, indent=2)}[/]")

        # Phase 1: Note plan + dedup + approval (SCAF-02, D-12)
        planner = PlanPhase(ctx)
        ctx.note_plan = planner.run(console)

        # Phase 2: Note generation (SCAF-03, SCAF-04)
        generator = GeneratePhase(ctx)
        generator.run(console)

        if not ctx.generated_notes:
            console.print("[red]No notes generated. Exiting.[/]")
            sys.exit(1)

    # Phase 3: Cross-domain link resolution (D-10, D-11)
    if ctx.generated_notes:
        linker = LinkResolutionPhase(ctx)
        linker.run(console)

    # Phase 4: Validation + vault write (SCAF-05, SCAF-06)
    if ctx.generated_notes:
        writer = ValidateWritePhase(ctx)
        written = writer.run(console)
        if written == 0:
            sys.exit(1)
    else:
        console.print("[red]No notes to write. Exiting.[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
