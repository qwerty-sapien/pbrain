# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
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
from typing import Optional, TYPE_CHECKING

import structlog
import yaml

from pb.core.durations import elapsed_minutes_and_label
from pb.core.graph_writer import make_slug
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.renderables import renderable_markdown_text
from pb.core.session_blueprints import blueprint_from_payload

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
            body = self._render_body(session, task, assessment, domain, date_str, duration_display)

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
    def _clean_visible_text(value: object) -> str:
        text = renderable_markdown_text(str(value or "")).strip()
        text = text.strip()
        lowered = text.strip(" _-*`").lower()
        if lowered in {"", "none", "no", "n/a", "na", "not recorded", "nothing"}:
            return ""
        if lowered.startswith("no concrete ") or lowered.startswith("nothing explicit "):
            return ""
        return text

    @classmethod
    def _append_unique(cls, items: list[str], value: object) -> None:
        text = cls._clean_visible_text(value)
        if not text:
            return
        key = text.lower()
        if key not in {item.lower() for item in items}:
            items.append(text)

    @staticmethod
    def _bullet_section(title: str, lines: list[str]) -> str:
        clean = [line.strip() for line in lines if line.strip()]
        if not clean:
            return ""
        return f"## {title}\n" + "\n".join(f"- {line}" for line in clean)

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

    def _render_body(self, session, task, assessment, domain, date_str, duration_display) -> str:
        """Render the compact learner-facing finish evidence body."""
        generated_names = getattr(session, "generated_names", {}) or {}
        actual_outcome = str(getattr(session, "actual_outcome", "") or "").strip()
        observed_errors = str(getattr(session, "observed_errors", "") or "").strip()
        next_adjustment = str(getattr(session, "next_adjustment", "") or "").strip()
        retry_items = self._retry_items(session, assessment)

        scope = self._clean_visible_text(getattr(session, "subject_scope", None)) or self._clean_visible_text(getattr(task, "title", None))
        intended = self._clean_visible_text(getattr(session, "expectation", None)) or self._clean_visible_text(getattr(session, "intended_outcome", None))

        proficient: list[str] = []
        mistakes: list[str] = []
        related: list[str] = []
        scores = list(getattr(assessment, "sub_skill_scores", []) if assessment is not None else [])
        for ss in scores:
            target = self._clean_visible_text(getattr(ss, "name", ""))
            if not target:
                continue
            if getattr(ss, "score", 0) >= 4 and not getattr(ss, "is_weak", False):
                self._append_unique(proficient, f"{target} ({getattr(ss, 'score', 0)}/5)")
            else:
                self._append_unique(mistakes, f"{target} ({getattr(ss, 'score', 0)}/5)")
        self._append_unique(proficient, actual_outcome if actual_outcome.lower() not in {"done", "q", "quit", "exit"} else scope)
        self._append_unique(mistakes, observed_errors)
        for item in retry_items:
            self._append_unique(mistakes, item)

        compact = generated_names.get("learning_partner_compact")
        if isinstance(compact, dict):
            for key in ("detected_gaps", "unknowns"):
                for item in compact.get(key, []) or []:
                    self._append_unique(mistakes, item)
            for item in compact.get("corrections", []) or []:
                self._append_unique(proficient, item)
            for key in ("connections", "related_topics", "interrelated_topics"):
                for item in compact.get(key, []) or []:
                    self._append_unique(related, item)

        learnt: list[str] = []
        if mistakes and next_adjustment:
            self._append_unique(learnt, f"Corrected focus: {mistakes[0]}; next adjustment is {next_adjustment}.")
        elif mistakes:
            self._append_unique(learnt, f"Identified the unstable point: {mistakes[0]}.")
        self._append_unique(learnt, actual_outcome if actual_outcome.lower() not in {"done", "q", "quit", "exit"} else "")
        if not learnt:
            self._append_unique(learnt, f"Stabilised the current work on {scope or domain}.")

        proficiency_lines: list[str] = []
        if scores:
            average = sum(int(getattr(ss, "score", 0) or 0) for ss in scores) / max(1, len(scores))
            proficiency_lines.append(f"Overall: {average:.1f}/5 across {len(scores)} assessed area(s).")
            if mistakes:
                proficiency_lines.append("Proficient with corrections needed.")
            else:
                proficiency_lines.append("Proficient on the captured evidence.")
        else:
            proficiency_lines.append(
                "Assessment was lightweight: proficiency is inferred from the session note because AI assessment was skipped."
            )

        body_sections = [
            f"# Evidence: {renderable_markdown_text(task.title)} -- {date_str}",
            f"**Domain:** {renderable_markdown_text(domain)}",
            f"**Duration:** {duration_display}",
        ]
        if intended:
            body_sections.append(self._bullet_section("Session Goal", [intended]))
        practised_lines = []
        if scope:
            practised_lines.append(f"Scope: {scope}")
        practised_lines.extend(f"Proved proficient in: {item}" for item in proficient[:4])
        practised_lines.extend(f"Mistake corrected: {item}" for item in mistakes[:4])
        body_sections.append(self._bullet_section("What Was Practised", practised_lines))
        body_sections.append(self._bullet_section("What You Learnt", learnt[:5]))
        body_sections.append(self._bullet_section("Assessment of Proficiency", proficiency_lines))
        if related:
            body_sections.append(self._bullet_section("Interrelated Topics", related[:6]))

        body = "\n\n".join(section for section in body_sections if section.strip()) + "\n"

        finish_checkin = self._finish_checkin_section(session)
        if finish_checkin:
            body = body.rstrip() + "\n\n" + finish_checkin + "\n"
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
