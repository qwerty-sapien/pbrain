# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""SocraticService — orchestrates Socratic capture across `pb note`, `pb study debrief`, and `pb finish --debrief`.

Wraps reusable functions from pb.vault.socratic; does not re-implement them.
INV-4: this module never imports rich or typer. Console is passed by CLI callers.
"""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

import structlog

from pb.core.base import BaseService, LoggableMixin
from pb.storage.yaml_io import extract_structured_yaml


class SocraticService(BaseService, LoggableMixin):
    """Service-layer orchestrator for the Socratic capture pipeline.

    Args:
        vault_path: Vault root path.
        ai: Optional AI client (deferred — current implementation reuses pb.vault.socratic helpers).
    """

    def __init__(self, vault_path: Path, ai: Any = None):
        super().__init__()
        self.vault_path = vault_path
        self.ai = ai
        self._log = structlog.get_logger()
        self.last_exit_reason: str = ""
        self.last_completion_summary: str = ""

    # ----- Private helpers -----

    def _read_state(self, domain: str) -> str:
        """Read knowledge/<domain>/_state.md content; truncated to 2000 chars."""
        state_path = self.vault_path / "knowledge" / domain / "_state.md"
        try:
            return state_path.read_text()[:2000]
        except OSError:
            return ""

    def _infer_domain_from_session(self, session: Any, task: Any) -> Optional[str]:
        """Best-effort domain inference from session/task."""
        try:
            from pb.vault.socratic import infer_domain_from_session  # if available
            return infer_domain_from_session(session, task)
        except (ImportError, AttributeError):
            pass
        # Fallback: read task.domain attribute or session.domain
        for obj in (task, session):
            d = getattr(obj, "domain", None)
            if d:
                return d
        return None

    # ----- Domain detection -----

    def detect_domain(self, knowledge_dir: Path) -> Optional[str]:
        """Per D-05/D-13: PB_SHELL_VAULT_CWD env -> cwd -> None (CLI handles picker fallback).

        Returns the first vault-relative path component that has a _state.md.
        Returns None if no domain can be inferred -- CLI handles the picker fallback.
        """
        try:
            vault_cwd_str = os.environ.get("PB_SHELL_VAULT_CWD")
            cwd = Path(vault_cwd_str) if vault_cwd_str else Path(os.getcwd())
            relative = cwd.relative_to(knowledge_dir)
            parts = relative.parts
            if parts:
                candidate = knowledge_dir / parts[0]
                if candidate.is_dir() and (candidate / "_state.md").exists():
                    return parts[0]
        except (ValueError, OSError):
            pass
        return None

    # ----- Debrief loops -----

    def run_note_debrief(
        self, topic: str, domain: str, max_rounds: int, console: Any
    ) -> list[tuple[str, str]]:
        """`pb note` flow. max_rounds: 2-3 (short) or 10-12 (long). Returns Q&A pairs."""
        from pb.vault.socratic import SocraticDebriefEngine, run_debrief_loop
        from pb.core.feedback_profile import load_feedback_guidance
        state_content = self._read_state(domain)
        guidance = load_feedback_guidance(self.vault_path, "study")
        # Topic is provided as initial seed by augmenting state_md_content (engine has no
        # explicit topic param — it opens with an LLM-generated question per D-03).
        if topic:
            engine = SocraticDebriefEngine(
                domain=domain,
                state_md_content=(state_content + f"\n\n## User topic\n{topic}")[:2200],
                max_rounds=max_rounds,
                prompt_guidance=guidance,
            )
        else:
            engine = SocraticDebriefEngine(
                domain=domain,
                state_md_content=state_content,
                max_rounds=max_rounds,
                prompt_guidance=guidance,
            )
        self._log.info(
            "socratic.note_debrief_started",
            domain=domain,
            max_rounds=max_rounds,
            has_topic=bool(topic),
        )
        pairs = run_debrief_loop(engine, console)
        self.last_exit_reason = getattr(engine, "exit_reason", "")
        self.last_completion_summary = getattr(engine, "build_completion_summary", lambda: "")() or ""
        return pairs

    def run_study_debrief(self, domain: str, console: Any, topic: str = "") -> list[tuple[str, str]]:
        """Deep study debrief flow. 5-question deep debrief per D-12."""
        from pb.vault.socratic import SocraticDebriefEngine, run_debrief_loop
        from pb.core.feedback_profile import load_feedback_guidance
        state_content = self._read_state(domain)
        engine = SocraticDebriefEngine(
            domain=domain,
            state_md_content=state_content,
            max_rounds=5,
            topic=topic,
            prompt_guidance=load_feedback_guidance(self.vault_path, "study"),
        )
        self._log.info("socratic.study_debrief_started", domain=domain, max_rounds=5, topic=topic)
        pairs = run_debrief_loop(engine, console)
        self.last_exit_reason = getattr(engine, "exit_reason", "")
        self.last_completion_summary = getattr(engine, "build_completion_summary", lambda: "")() or ""
        return pairs

    def run_teach_session(
        self,
        domain: str,
        console: Any,
        *,
        topic: str = "",
        bloom_level: str = "apply",
        max_rounds: int = 6,
    ) -> list[tuple[str, str]]:
        """Run a supportive Socratic teaching loop for a new concept."""
        from pb.vault.socratic import SocraticDebriefEngine, run_debrief_loop
        from pb.core.feedback_profile import load_feedback_guidance

        state_content = self._read_state(domain)
        engine = SocraticDebriefEngine(
            domain=domain,
            state_md_content=state_content,
            max_rounds=max_rounds,
            topic=topic,
            teaching=True,
            bloom_level=bloom_level,
            prompt_guidance=load_feedback_guidance(self.vault_path, "teach"),
        )
        self._log.info(
            "socratic.teach_session_started",
            domain=domain,
            topic=topic,
            bloom_level=bloom_level,
            max_rounds=max_rounds,
        )
        pairs = run_debrief_loop(engine, console)
        self.last_exit_reason = getattr(engine, "exit_reason", "")
        self.last_completion_summary = getattr(engine, "build_completion_summary", lambda: "")() or ""
        return pairs

    def run_adaptive_diagnostic(
        self,
        domain: str,
        console: Any,
        *,
        topic: str = "",
        difficulty_start: str = "",
        difficulty_limit: str = "",
        max_rounds: int = 30,
        soft_cap_rounds: int = 24,
        model: Optional[str] = None,
        time_limit_minutes: int = 10,
    ) -> list[tuple[str, str]]:
        """Run a longer strict diagnostic interview for downstream Anki generation."""
        from pb.vault.socratic import SocraticDebriefEngine, run_debrief_loop
        from pb.core.feedback_profile import load_feedback_guidance

        state_content = self._read_state(domain)
        engine = SocraticDebriefEngine(
            domain=domain,
            state_md_content=state_content,
            max_rounds=max_rounds,
            topic=topic,
            strict=True,
            adaptive=True,
            difficulty_start=difficulty_start,
            difficulty_limit=difficulty_limit,
            soft_cap_rounds=soft_cap_rounds,
            model=model,
            time_limit_minutes=time_limit_minutes,
            prompt_guidance=load_feedback_guidance(self.vault_path, "diagnostic"),
        )
        self._log.info(
            "socratic.adaptive_diagnostic_started",
            domain=domain,
            topic=topic,
            difficulty_start=difficulty_start,
            difficulty_limit=difficulty_limit,
            max_rounds=max_rounds,
            soft_cap_rounds=soft_cap_rounds,
            model=model,
            time_limit_minutes=time_limit_minutes,
        )
        pairs = run_debrief_loop(engine, console)
        self.last_exit_reason = getattr(engine, "exit_reason", "")
        self.last_completion_summary = getattr(engine, "build_completion_summary", lambda: "")() or ""
        return pairs

    def run_finish_debrief(
        self, session: Any, task: Any, console: Any
    ) -> list[tuple[str, str]]:
        """`pb finish --debrief` flow. 3-round brief debrief on a domain session per D-16/D-19.

        Note: is_domain_session(task, vault_path) — takes task and vault_path (not session+task).
        """
        from pb.vault.socratic import (
            SocraticDebriefEngine,
            run_debrief_loop,
            is_domain_session,
        )
        from pb.core.feedback_profile import load_feedback_guidance
        if not is_domain_session(task, self.vault_path):
            self._log.info("socratic.finish_debrief_skipped_non_domain")
            return []
        domain = self._infer_domain_from_session(session, task)
        if not domain:
            self._log.info("socratic.finish_debrief_skipped_no_domain")
            return []
        state_content = self._read_state(domain)
        engine = SocraticDebriefEngine(
            domain=domain,
            state_md_content=state_content,
            max_rounds=3,
            prompt_guidance=load_feedback_guidance(self.vault_path, "study"),
        )
        self._log.info("socratic.finish_debrief_started", domain=domain, max_rounds=3)
        pairs = run_debrief_loop(engine, console)
        self.last_exit_reason = getattr(engine, "exit_reason", "")
        self.last_completion_summary = getattr(engine, "build_completion_summary", lambda: "")() or ""
        return pairs

    def build_diagnostic_report(
        self,
        qa_pairs: list[tuple[str, str]],
        domain: str,
        *,
        topic: str = "",
        difficulty_start: str = "",
        difficulty_limit: str = "",
        note_types: Optional[list[str]] = None,
        model: Optional[str] = None,
    ) -> dict:
        """Summarize a Socratic diagnostic transcript into YAML-ready guidance."""
        from pb.vault.socratic import build_diagnostic_report

        return build_diagnostic_report(
            qa_pairs,
            domain,
            self._read_state(domain),
            topic=topic,
            difficulty_start=difficulty_start,
            difficulty_limit=difficulty_limit,
            note_types=note_types,
            model=model,
        )

    def _persist_note_artifact(
        self,
        *,
        note_rel_path: str,
        note_content: str,
        slug: str,
        domain: str,
        wikilinks: list[str],
    ) -> str:
        """Write a durable note and best-effort register it for later inference."""
        from pb.vault.graph_store import open_vault_db, upsert_node, add_link
        from pb.vault.lifecycle import log_interaction

        note_abs_path = self.vault_path / note_rel_path
        note_abs_path.parent.mkdir(parents=True, exist_ok=True)
        note_abs_path.write_text(note_content, encoding="utf-8")

        try:
            gs_conn = open_vault_db(self.vault_path)
            try:
                upsert_node(gs_conn, slug, str(Path(note_rel_path).parent))
                for wl in wikilinks:
                    try:
                        add_link(gs_conn, slug, wl)
                    except Exception:
                        pass
            finally:
                gs_conn.close()
        except Exception as exc:
            self._log.warning("socratic.graph_write_failed", error=str(exc))

        try:
            log_interaction(note_rel_path, "socratic", domain)
        except Exception as exc:
            self._log.warning("socratic.log_interaction_failed", error=str(exc))
        return note_rel_path

    def _draft_teach_lesson_sections(
        self,
        *,
        qa_pairs: list[tuple[str, str]],
        domain: str,
        topic: str,
        wikilinks: list[str],
    ) -> dict[str, object]:
        """Generate note sections for teach sessions, with deterministic fallback."""
        fallback_summary = (
            f"Guided teaching session on {topic or domain} across {len(qa_pairs)} exchanges."
        )
        fallback_key_insight = qa_pairs[0][1] if qa_pairs else f"No key insight captured for {topic or domain}."
        fallback_downstream = [
            f"Connect {topic or domain} to [[{link}]]." for link in wikilinks[:3]
        ] or [f"Trace one prerequisite and one follow-on concept related to {topic or domain}."]
        fallback_attempts = [
            f"Explain {topic or domain} again from memory without notes.",
            f"Work one fresh example or application involving {topic or domain}.",
            f"Turn the least stable point in {topic or domain} into a recall prompt.",
        ]
        payload: dict[str, object] = {
            "summary": fallback_summary,
            "key_insight": fallback_key_insight,
            "downstream_concepts": fallback_downstream,
            "next_attempts": fallback_attempts,
        }

        try:
            from pb.llm.gemini import FLASH_LITE_MODEL, get_client

            client = get_client()
            if client.is_available() and qa_pairs:
                transcript = "\n\n".join(
                    f"Q{i}: {question}\nA{i}: {answer}"
                    for i, (question, answer) in enumerate(qa_pairs, start=1)
                )
                prompt = (
                    "Turn this guided teaching transcript into a durable lesson-note scaffold.\n"
                    f"Domain: {domain or 'general'}\n"
                    f"Topic: {topic or domain or 'general'}\n"
                    "Return YAML only with keys:\n"
                    "summary: string\n"
                    "key_insight: string\n"
                    "downstream_concepts:\n"
                    "  - string\n"
                    "next_attempts:\n"
                    "  - string\n"
                    "Focus on conceptual connections, future learning paths, and the next concrete attempts.\n\n"
                    f"Transcript:\n{transcript[:7000]}"
                )
                structured = extract_structured_yaml(
                    client.generate_with_model(prompt, FLASH_LITE_MODEL, timeout=30, max_output_tokens=4500) or "",
                    {},
                )
                if isinstance(structured, dict):
                    for key in ("summary", "key_insight"):
                        value = structured.get(key)
                        if isinstance(value, str) and value.strip():
                            payload[key] = value.strip()
                    for key in ("downstream_concepts", "next_attempts"):
                        values = structured.get(key)
                        if isinstance(values, list):
                            clean = [str(item).strip() for item in values if str(item).strip()]
                            if clean:
                                payload[key] = clean
        except Exception as exc:
            self._log.debug("socratic.teach_sections_fallback", error=str(exc))

        return payload

    def save_teach_lesson(
        self,
        *,
        qa_pairs: list[tuple[str, str]],
        domain: str,
        topic: str = "",
        completed: bool = True,
    ) -> str:
        """Persist a teach transcript as a durable lesson note."""
        from pb.core.graph_writer import make_slug
        from pb.vault.socratic import build_teach_lesson_note, infer_wikilinks

        state_content = self._read_state(domain) if domain else ""
        all_answers = " ".join(answer for _, answer in qa_pairs).strip()
        slug = make_slug(topic or all_answers[:80] or f"{domain}-teach-lesson") or "teach-lesson"
        wikilinks = infer_wikilinks(all_answers, domain or "general", state_content, self.vault_path) if qa_pairs else []
        lesson_sections = self._draft_teach_lesson_sections(
            qa_pairs=qa_pairs,
            domain=domain or "general",
            topic=topic,
            wikilinks=wikilinks,
        )
        note_content = build_teach_lesson_note(
            domain=domain or "general",
            slug=slug,
            topic=topic or domain or slug,
            qa_pairs=qa_pairs,
            wikilinks=wikilinks,
            summary=str(lesson_sections.get("summary", "")),
            key_insight=str(lesson_sections.get("key_insight", "")),
            downstream_concepts=list(lesson_sections.get("downstream_concepts", []) or []),
            next_attempts=list(lesson_sections.get("next_attempts", []) or []),
            completed=completed,
        )
        note_rel_path = (
            f"knowledge/{domain}/{slug}.md"
            if domain
            else f"Learning/Inbox/pb/conversations/{slug}.md"
        )
        return self._persist_note_artifact(
            note_rel_path=note_rel_path,
            note_content=note_content,
            slug=slug,
            domain=domain or "general",
            wikilinks=wikilinks,
        )

    def cache_diagnostic_transcript(
        self,
        *,
        qa_pairs: list[tuple[str, str]],
        domain: str,
        topic: str = "",
    ) -> str:
        """Persist a diagnostic transcript into the vault for later inference."""
        from pb.core.graph_writer import make_slug
        from pb.vault.lifecycle import read_frontmatter, write_frontmatter
        from pb.vault.socratic import build_socratic_note, infer_wikilinks

        state_content = self._read_state(domain) if domain else ""
        all_answers = " ".join(answer for _, answer in qa_pairs).strip()
        slug = make_slug(f"{topic or domain}-diagnostic") or "diagnostic"
        wikilinks = infer_wikilinks(all_answers, domain or "general", state_content, self.vault_path) if qa_pairs else []
        note_content = build_socratic_note(
            qa_pairs=qa_pairs,
            domain=domain or "general",
            slug=slug,
            wikilinks=wikilinks,
            template="deep",
        )
        fm, body = read_frontmatter(note_content)
        fm["conversation_kind"] = "diagnostic"
        if topic:
            fm["topic"] = topic
        note_content = write_frontmatter(fm, body)
        note_rel_path = (
            f"knowledge/{domain}/{slug}.md"
            if domain
            else f"Learning/Inbox/pb/conversations/{slug}.md"
        )
        return self._persist_note_artifact(
            note_rel_path=note_rel_path,
            note_content=note_content,
            slug=slug,
            domain=domain or "general",
            wikilinks=wikilinks,
        )

    # ----- Note creation -----

    def _write_note_sync(
        self,
        qa_pairs: list[tuple[str, str]],
        domain: str,
        slug: str,
        template: str,
        console: Any,
    ) -> str:
        """Synchronous note write path. Returns vault-relative note path."""
        from pb.vault.socratic import (
            build_socratic_note,
            infer_wikilinks,
            show_note_preview_and_confirm,
            extract_socratic_cards,
        )
        from pb.vault.graph_store import open_vault_db, upsert_node, add_link
        from pb.vault.anki_client import insert_cards_to_db
        from pb.vault.lifecycle import log_interaction
        from pb.mcp.tools.vault import vault_write

        state_content = self._read_state(domain)
        all_answers = " ".join(a for _, a in qa_pairs)
        wikilinks = infer_wikilinks(all_answers, domain, state_content, self.vault_path)
        note_content = build_socratic_note(qa_pairs, domain, slug, wikilinks, template=template)

        final = show_note_preview_and_confirm(
            console,
            note_content,
            slug,
            domain,
            trust=False,
            vault_path=self.vault_path,
            state_md_content=state_content,
        )
        if not final:
            self._log.info("socratic.note_cancelled", domain=domain, slug=slug)
            return ""

        note_rel_path = f"knowledge/{domain}/{slug}.md"
        vault_write(note_rel_path, final)

        try:
            gs_conn = open_vault_db(self.vault_path)
            try:
                upsert_node(gs_conn, slug, f"knowledge/{domain}")
                for wl in wikilinks:
                    try:
                        add_link(gs_conn, slug, wl)
                    except Exception:
                        pass
            finally:
                gs_conn.close()
        except Exception as exc:
            self._log.warning("socratic.graph_write_failed", error=str(exc))

        cards = extract_socratic_cards(qa_pairs, slug, domain, domain)
        if cards:
            insert_cards_to_db(cards)

        log_interaction(note_rel_path, "socratic", domain)
        self._log.info(
            "socratic.note_written_sync",
            note_rel_path=note_rel_path,
            domain=domain,
            slug=slug,
            wikilink_count=len(wikilinks),
            card_count=len(cards) if cards else 0,
        )
        return note_rel_path

    def _finalise_async_note(
        self,
        qa_pairs: list[tuple[str, str]],
        domain: str,
        slug: str,
        template: str,
        llm_body: str,
    ) -> None:
        """Write note from async batch result. No console preview (D-10: notification on completion)."""
        from pb.vault.socratic import (
            build_socratic_note,
            infer_wikilinks,
            extract_socratic_cards,
        )
        from pb.vault.graph_store import open_vault_db, upsert_node, add_link
        from pb.vault.anki_client import insert_cards_to_db
        from pb.vault.lifecycle import log_interaction
        from pb.mcp.tools.vault import vault_write

        state_content = self._read_state(domain)
        all_answers = " ".join(a for _, a in qa_pairs)
        wikilinks = infer_wikilinks(all_answers, domain, state_content, self.vault_path)
        # Use build_socratic_note for frontmatter scaffolding; replace body with LLM output.
        base_note = build_socratic_note(qa_pairs, domain, slug, wikilinks, template=template)
        fm_end = base_note.find("---", 4)
        if fm_end > 0:
            fm_block = base_note[: fm_end + 3]
            final = f"{fm_block}\n\n{llm_body.strip()}\n"
        else:
            final = base_note  # fallback: programmatic note

        note_rel_path = f"knowledge/{domain}/{slug}.md"
        vault_write(note_rel_path, final)

        try:
            gs_conn = open_vault_db(self.vault_path)
            try:
                upsert_node(gs_conn, slug, f"knowledge/{domain}")
                for wl in wikilinks:
                    try:
                        add_link(gs_conn, slug, wl)
                    except Exception:
                        pass
            finally:
                gs_conn.close()
        except Exception:
            pass

        cards = extract_socratic_cards(qa_pairs, slug, domain, domain)
        if cards:
            insert_cards_to_db(cards)
        log_interaction(note_rel_path, "socratic", domain)

    def _notify_macos(self, message: str, title: str) -> None:
        """Per D-10: macOS notification via osascript. Silent skip on non-Darwin.

        T-24-09 mitigation: message and title are caller-controlled values only (slug, static
        strings). No user-controlled topic text is passed in. Slug is already sanitised by
        make_slug (strips non-alphanumeric) before reaching this method.
        """
        if platform.system() != "Darwin":
            return
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "{title}"',
                ],
                check=False,
                timeout=5,
            )
        except Exception:
            pass

    def build_and_submit(
        self,
        qa_pairs: list[tuple[str, str]],
        domain: str,
        slug: str,
        template: str,  # "brief" | "deep"
        sync: bool,
        model: str,
        console: Any,
    ) -> Optional[str]:
        """Per D-08/D-09/D-10:
        - sync=True: write note synchronously, return rel path.
        - sync=False: submit Vertex Batch (single-request), spawn polling thread, return job_name.
          On completion, the thread writes the note and fires a macOS notification.
        """
        if sync:
            return self._write_note_sync(qa_pairs, domain, slug, template, console)

        # Async path: validate GCS config; fall back to sync if missing
        from pb.storage.config import get_config
        cfg = get_config()
        if not cfg.scaffold.gcs_bucket or not cfg.scaffold.gcp_project:
            try:
                console.print(
                    "[warn]GCS not configured; falling back to --sync. "
                    "Run scripts/scaffold_generator.py once to enable async batch.[/]"
                )
            except Exception:
                pass
            return self._write_note_sync(qa_pairs, domain, slug, template, console)

        # Build a single batch request from the Q&A pairs
        from pb.vault.batch_client import VertexBatchClient, build_batch_line

        state_content = self._read_state(domain)
        qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
        system_instruction = (
            f"You are restructuring a Socratic capture into an atomic note for domain '{domain}'. "
            f"Preserve the user's exact words. Do not paraphrase. Domain context:\n{state_content[:1500]}"
        )
        prompt = f"Generate the body of a markdown note from these Q&A pairs:\n{qa_text}"
        request = build_batch_line(
            key=slug,
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.4,
            max_output_tokens=4000,
        )
        run_id = uuid.uuid4().hex[:8]
        client = VertexBatchClient(
            gcp_project=cfg.scaffold.gcp_project,
            gcs_bucket=cfg.scaffold.gcs_bucket,
            location=cfg.scaffold.location,
        )
        job_name = client.submit(
            requests=[request], model=model, run_id=run_id, prefix="socratic"
        )

        def _poll_and_finalise():
            try:
                successes, failures = client.poll_and_download(job_name)
                if successes:
                    self._finalise_async_note(
                        qa_pairs, domain, slug, template, successes[0]["content"]
                    )
                    self._notify_macos(f"Note saved: {slug}.md", "pb note complete")
                else:
                    self._log.warning(
                        "socratic.batch_failed",
                        job_name=job_name,
                        failures=failures,
                    )
                    self._notify_macos(f"Batch failed for {slug}", "pb note error")
            except Exception as exc:
                self._log.error("socratic.poll_thread_error", error=str(exc))

        threading.Thread(target=_poll_and_finalise, daemon=True).start()
        self._log.info("socratic.batch_submitted", job_name=job_name, slug=slug, domain=domain)
        return job_name

    # ----- Bridge note -----

    def suggest_bridge(
        self, qa_pairs: list[tuple[str, str]], domain: str, console: Any
    ) -> None:
        """Per D-11/D-15: cross-domain bridge note suggestion when qa_pairs >= 4.

        T-24-11 mitigation: _suggest_bridge_note body is inlined here to avoid importing
        from learn.py (which imports typer at module level, violating INV-4).
        """
        if len(qa_pairs) < 4:
            return
        try:
            self._suggest_bridge_inline(qa_pairs, domain, console)
        except Exception as exc:
            # Non-fatal: bridge suggestion failures must not block the main capture
            self._log.warning("socratic.suggest_bridge_failed", error=str(exc))

    def _suggest_bridge_inline(
        self,
        qa_pairs: list[tuple[str, str]],
        domain: str,
        console: Any,
    ) -> None:
        """Inlined port of learn._suggest_bridge_note — avoids importing from CLI module (INV-4).

        SOCR-05, D-10: Suggest bridge note if cross-domain answer detected in Q4+.
        """
        from pb.llm.gemini import get_client, FLASH_LITE_MODEL
        client = get_client()
        if not client.is_available():
            return

        # Check Q4+ answers for cross-domain signals
        later_answers = " ".join(a for _, a in qa_pairs[3:])
        prompt = (
            f"Does this text reference concepts from a domain OTHER than '{domain}'?\n"
            f"Text: {later_answers[:1000]}\n\n"
            "If yes, respond with YAML:\n"
            "cross_domain: true\n"
            "bridge_slug: suggested-bridge-note-name\n"
            "domains:\n"
            "  - domain1\n"
            "  - domain2\n"
            "If no, respond with:\n"
            "cross_domain: false\n"
            "Return ONLY YAML."
        )
        result = client.generate_with_model(prompt, FLASH_LITE_MODEL, timeout=10)
        if not result:
            return

        data = extract_structured_yaml(result.strip(), {})
        if not isinstance(data, dict):
            return

        if data.get("cross_domain"):
            bridge_slug = data.get("bridge_slug", "cross-domain-connection")
            console.print()
            console.print("[warn]Cross-domain answer detected.[/]")
            console.print("[dim]Flash Lite found a connection to concepts/ -- suggest bridge note?[/]")
            console.print(f"[dim]Bridge: '{bridge_slug}'[/]")
            console.print()
            console.print("[dim]y[/] create bridge  [dim]n[/] skip")
            try:
                choice = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if choice in ("y", "yes"):
                from pb.vault.socratic import build_socratic_frontmatter
                from pb.vault.lifecycle import write_frontmatter
                fm = build_socratic_frontmatter("concepts", bridge_slug, [domain])
                body = f"## Bridge: {domain}\n\n{later_answers}\n"
                bridge_content = write_frontmatter(fm, body)
                bridge_rel = f"knowledge/concepts/{bridge_slug}.md"
                from pb.mcp.tools.vault import vault_write
                vault_write(bridge_rel, bridge_content)
                console.print(f"[success]Bridge note created: {bridge_rel}[/]")
