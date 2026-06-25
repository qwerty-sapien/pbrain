# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Evidence note writer for the learning evidence system.

Writes Markdown evidence notes with YAML frontmatter to vault/evidence/{domain}/.
Indexes written notes in SQLite (write-through cache).

Per Phase 2 decisions:
  D-01: Replaces SessionLogWriter for evidence tracking.
  D-02: Vault path is vault/evidence/{domain}/.
  D-03: YAML frontmatter + Markdown body.
  D-04: Bare evidence note (no assessment) when --skip or non-TTY.

Non-fatal: vault write failures log warning and return None.
Security (T-02-01): make_slug() sanitizes domain and slug before path construction.
Security (T-02-03): yaml.safe_dump() only -- never yaml.dump().
Security (T-02-04): exceptions logged via structlog, not shown to user.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Optional, TYPE_CHECKING

import structlog
import yaml

from pb.core.durations import elapsed_minutes_and_label
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.resources import read_template_text, template_exists
from pb.core.session_blueprints import blueprint_from_payload
from pb.core.graph_writer import make_slug

if TYPE_CHECKING:
    from pb.domain.models import Session, Task

logger = structlog.get_logger()


class EvidenceWriter:
    """Writes evidence notes to vault/evidence/{domain}/{date}-{slug}.md.

    Per D-01: replaces SessionLogWriter.
    Per D-02: vault/evidence/{domain}/ path.
    Per D-03: YAML frontmatter + Markdown body.
    Non-fatal: vault write failures log warning and return None.
    """

    def __init__(self, vault_path: Optional[Path] = None):
        if vault_path is None:
            from pb.vault.config import get_vault_path
            vault_path = get_vault_path()
        self.evidence_root = vault_path / "evidence"

    def _unique_path(self, base_path: Path) -> Path:
        """Collision-safe: append -2, -3, etc. (same as SessionLogWriter._unique_path)."""
        if not base_path.exists():
            return base_path
        stem, suffix, parent = base_path.stem, base_path.suffix, base_path.parent
        counter = 2
        while True:
            candidate = parent / f"{stem}-{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def write_evidence(
        self,
        session: "Session",
        task: "Task",
        assessment: Optional[object],  # AssessmentResult or None
        domain: str,
    ) -> Optional[Path]:
        """Write evidence note. Returns path or None on non-fatal failure.

        If assessment is None (--skip or non-TTY), still writes the captured
        session signal, but skips AI-generated assessment fields (per D-04).
        """
        try:
            # T-02-01: sanitize domain via make_slug before path construction
            domain_slug = make_slug(domain or "general")
            domain_dir = self.evidence_root / domain_slug
            domain_dir.mkdir(parents=True, exist_ok=True)

            date_str = (session.end_at or datetime.utcnow()).strftime("%Y-%m-%d")
            slug = make_slug(task.title)
            path = self._unique_path(domain_dir / f"{date_str}-{slug}.md")

            duration_min, duration_display = elapsed_minutes_and_label(session.start_at, session.end_at)

            frontmatter = self._build_frontmatter(session, task, assessment, domain, date_str, duration_min)
            body = self._render_body(session, task, assessment, domain, date_str, duration_min, duration_display)

            # T-02-03: yaml.safe_dump only -- never yaml.dump
            content = "---\n" + yaml.safe_dump(frontmatter, allow_unicode=True, default_flow_style=False) + "---\n\n" + body
            path.write_text(content)
            logger.info("evidence_writer.written", path=str(path))
            return path
        except Exception as e:
            # T-02-04: log to structlog, not console
            logger.warning("evidence_writer.failed", error=str(e))
            return None

    def _build_frontmatter(self, session, task, assessment, domain, date_str, duration_min) -> dict:
        """Build YAML frontmatter dict per D-03."""
        from pb.core.domain_templates import get_template
        template = get_template(domain, branch=getattr(session, "branch", "") or "study", session=session, task=task)
        meta = parse_learning_task_metadata(task)
        generated = dict(getattr(session, "generated_names", {}) or {})
        blueprint = blueprint_from_payload(
            generated.get("session_blueprint") if isinstance(generated.get("session_blueprint"), dict) else meta.session_blueprint
        )

        fm = {
            "type": "evidence",
            "domain": domain,
            "date": date_str,
            "session_id": session.id,
            "duration_min": duration_min,
            "planned_duration_min": getattr(session, "duration_minutes", None),
            "outcome": getattr(session, "actual_outcome", None) or "done",
            "template": template.name,
            "assessment_skipped": assessment is None,
        }
        if blueprint is not None:
            fm["skill_kind"] = blueprint.skill_kind.value
            fm["primary_frame"] = blueprint.primary_frame.value
            fm["subskills"] = list(blueprint.subskills)

        if assessment is not None:
            sub_skills = []
            for ss in getattr(assessment, "sub_skill_scores", []):
                sub_skills.append({
                    "name": ss.name,
                    "score": ss.score,
                    "weak": ss.is_weak,
                })
            fm["sub_skills_assessed"] = sub_skills
            fm["retry_items_generated"] = len(getattr(assessment, "retry_items", []))
        else:
            fm["sub_skills_assessed"] = []
            fm["retry_items_generated"] = 0

        return fm

    @staticmethod
    def _first_nonempty(*values: object, default: str = "_Not recorded_") -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return default

    @staticmethod
    def _field_text(value: object, default: str = "_Not recorded_") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    @staticmethod
    def _finish_checkin_section(session) -> str:
        generated_names = getattr(session, "generated_names", {}) or {}
        qa_pairs = generated_names.get("finish_checkin_qa")
        if not isinstance(qa_pairs, list):
            return ""

        lines = []
        for item in qa_pairs:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            lines.append(f"- **{question}** {answer}")
        if not lines:
            return ""
        return "## Finish Check-In\n" + "\n".join(lines)

    @staticmethod
    def _retry_items(session, assessment) -> list[str]:
        items: list[str] = []
        if assessment is not None:
            items.extend(str(item).strip() for item in getattr(assessment, "retry_items", []) if str(item).strip())

        generated_names = getattr(session, "generated_names", {}) or {}
        finish_assessment = generated_names.get("finish_assessment")
        if isinstance(finish_assessment, dict):
            items.extend(
                str(item).strip()
                for item in finish_assessment.get("retry_items", [])
                if str(item).strip()
            )
        partner_closeout = generated_names.get("learning_partner_closeout")
        if isinstance(partner_closeout, dict):
            next_drill = str(partner_closeout.get("next_drill", "")).strip()
            if next_drill:
                items.append(next_drill)

        next_adjustment = str(getattr(session, "next_adjustment", "") or "").strip()
        if next_adjustment:
            items.append(next_adjustment)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _render_body(self, session, task, assessment, domain, date_str, duration_min, duration_display) -> str:
        """Render Markdown body via string.Template.safe_substitute()."""
        from pb.core.domain_templates import get_template
        template_def = get_template(domain, branch=getattr(session, "branch", "") or "study", session=session, task=task)
        meta = parse_learning_task_metadata(task)
        generated_names = getattr(session, "generated_names", {}) or {}
        blueprint = blueprint_from_payload(
            generated_names.get("session_blueprint") if isinstance(generated_names.get("session_blueprint"), dict) else meta.session_blueprint
        )
        template_name = template_def.markdown_template_file
        if not template_exists(template_name):
            template_name = "evidence_generic.md"

        template = Template(read_template_text(template_name))

        # Build sub-skills section
        actual_outcome = str(getattr(session, "actual_outcome", "") or "").strip()
        observed_errors = str(getattr(session, "observed_errors", "") or "").strip()
        next_adjustment = str(getattr(session, "next_adjustment", "") or "").strip()
        sub_skills_section = "_No assessment signals captured._"
        critique = "AI assessment was skipped, so this note uses lightweight session signals instead."
        retry_items = self._retry_items(session, assessment)
        retry_items_section = "\n".join(f"- [ ] {item}" for item in retry_items) if retry_items else "_None_"
        weak_skill_names = ""

        if assessment is not None:
            lines = []
            for ss in getattr(assessment, "sub_skill_scores", []):
                weak_marker = " -- weak" if ss.is_weak else ""
                lines.append(f"- {ss.name} (score: {ss.score}/5{weak_marker})")
            sub_skills_section = "\n".join(lines) if lines else "_No sub-skills assessed_"
            critique = getattr(assessment, "critique", "_No critique_") or "_No critique_"
            weak_skill_names = ", ".join(
                ss.name for ss in getattr(assessment, "sub_skill_scores", []) if getattr(ss, "is_weak", False)
            ).strip()
        else:
            signal_lines = []
            if actual_outcome:
                signal_lines.append(f"- Progress signal: {actual_outcome}")
            if observed_errors:
                signal_lines.append(f"- Shaky area: {observed_errors}")
            if next_adjustment:
                signal_lines.append(f"- Retry focus: {next_adjustment}")
            if signal_lines:
                sub_skills_section = "\n".join(signal_lines)

        what_you_learned = self._first_nonempty(
            actual_outcome,
            getattr(session, "intended_outcome", None),
            getattr(task, "title", None),
        )
        what_is_shaky = self._first_nonempty(
            observed_errors,
            weak_skill_names,
            default="_Nothing explicit was captured._",
        )
        what_to_do_next = self._first_nonempty(
            next_adjustment,
            retry_items[0] if retry_items else None,
            default="_No concrete next step was captured._",
        )

        body = template.safe_substitute(
            title=task.title,
            date=date_str,
            domain=domain,
            duration_min=str(duration_min),
            duration_display=duration_display,
            sub_skills_section=sub_skills_section,
            critique=critique,
            retry_items_section=retry_items_section,
            # Generic fields are populated from session/task signal when available.
            session_goal=self._first_nonempty(
                getattr(session, "expectation", None),
                getattr(session, "intended_outcome", None),
                getattr(task, "title", None),
            ),
            what_practiced=self._first_nonempty(
                getattr(session, "subject_scope", None),
                getattr(task, "title", None),
                getattr(session, "actual_outcome", None),
            ),
            difficulties=self._field_text(getattr(session, "observed_errors", None)),
            self_assessment=self._first_nonempty(
                getattr(session, "actual_outcome", None),
                getattr(session, "next_adjustment", None),
            ),
            what_you_learned=what_you_learned,
            what_is_shaky=what_is_shaky,
            what_to_do_next=what_to_do_next,
            problem_set=self._first_nonempty(
                getattr(session, "subject_scope", None),
                getattr(task, "title", None),
            ),
            mistakes_log=self._field_text(getattr(session, "observed_errors", None)),
            concepts_applied=self._first_nonempty(
                getattr(session, "actual_outcome", None),
                getattr(session, "subject_scope", None),
                getattr(task, "title", None),
            ),
            compiler_errors=self._field_text(getattr(session, "observed_errors", None)),
            phrases_attempted=self._first_nonempty(
                getattr(session, "actual_outcome", None),
                getattr(session, "subject_scope", None),
                getattr(task, "title", None),
            ),
            corrections=self._first_nonempty(
                getattr(session, "observed_errors", None),
                getattr(session, "next_adjustment", None),
            ),
        )

        extras: list[str] = []
        if blueprint is not None:
            extras.append(
                "## Session Blueprint\n"
                f"- Skill kind: {blueprint.skill_kind.value}\n"
                f"- Primary frame: {blueprint.primary_frame.value}\n"
                f"- Subskills: {', '.join(blueprint.subskills) or '_None_'}"
            )
            evidence_items = generated_names.get("learning_partner_evidence")
            if isinstance(evidence_items, list) and evidence_items:
                lines: list[str] = []
                for item in evidence_items:
                    if not isinstance(item, dict):
                        continue
                    subskill = str(item.get("subskill", "")).strip()
                    note = str(item.get("note", "") or item.get("evidence", "") or "").strip()
                    if not note:
                        continue
                    prefix = f"{subskill}: " if subskill else ""
                    lines.append(f"- {prefix}{note}")
                if lines:
                    extras.append("## Evidence Observed\n" + "\n".join(lines))
        if template_def.name != "_generic":
            summary_lines = []
            if actual_outcome:
                summary_lines.append(f"- Outcome: {actual_outcome}")
            if observed_errors:
                summary_lines.append(f"- Observed errors: {observed_errors}")
            if summary_lines:
                extras.append("## Session Summary\n" + "\n".join(summary_lines))
            extras.extend(
                [
                    "## What You Learned\n" + what_you_learned,
                    "## What Is Still Shaky\n" + what_is_shaky,
                    "## What To Do Next\n" + what_to_do_next,
                ]
            )

        finish_checkin = self._finish_checkin_section(session)
        if finish_checkin:
            extras.append(finish_checkin)

        if extras:
            body = body.rstrip() + "\n\n" + "\n\n".join(extras) + "\n"
        return body


def index_evidence_note(
    session: "Session",
    task: "Task",
    assessment: Optional[object],
    evidence_path: Path,
    domain: str,
) -> None:
    """Insert a record into evidence_notes SQLite table (write-through cache).

    Non-fatal: logs warning on failure.
    T-02-02: parameterized queries -- never f-string SQL.
    """
    try:
        from pb.storage.database import get_connection
        date_str = (session.end_at or datetime.utcnow()).strftime("%Y-%m-%d")
        slug = make_slug(task.title)
        duration_min, _ = elapsed_minutes_and_label(session.start_at, session.end_at)

        sub_skills = []
        retry_count = 0
        if assessment is not None:
            sub_skills = [{"name": ss.name, "score": ss.score, "weak": ss.is_weak}
                          for ss in getattr(assessment, "sub_skill_scores", [])]
            retry_count = len(getattr(assessment, "retry_items", []))

        with get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO evidence_notes
                   (id, path, domain, date, slug, duration_min, outcome, sub_skills, retry_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.id,
                    str(evidence_path),
                    domain,
                    date_str,
                    slug,
                    duration_min,
                    getattr(session, "actual_outcome", None) or "done",
                    json.dumps(sub_skills),
                    retry_count,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        logger.info("evidence_writer.indexed", session_id=session.id, domain=domain)
    except Exception as e:
        logger.warning("evidence_writer.index_failed", error=str(e))
