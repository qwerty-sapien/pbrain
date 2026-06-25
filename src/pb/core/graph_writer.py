# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Graph note writer for Obsidian-compatible knowledge graph.

Writes task notes and project notes to the vault on every pb finish.
Per Phase 8 D-01 through D-10 decisions.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Optional

import structlog

from pb.core.durations import elapsed_minutes_and_label
from pb.core.resources import read_template_text, resource, template_exists
from pb.domain.models import Project, Session, Task
from pb.storage.config import get_vault_path

logger = structlog.get_logger()


def suggest_learnt_promotions(vault_path: Path) -> list[str]:
    """Check all #learning notes across vault for promotion to #learnt (D-13, LIFE-04).

    Returns list of note paths (relative to vault_path) that meet the
    learnt_suggestion_threshold.  Does NOT promote — caller must present to
    user for tier-2 confirmation.
    """
    from pb.storage.config import get_config
    from pb.vault.lifecycle import read_frontmatter, get_weighted_total

    cfg = get_config()
    threshold = cfg.learning.learnt_suggestion_threshold  # default 10.0
    candidates: list[str] = []

    try:
        for md in vault_path.rglob("*.md"):
            parts = md.relative_to(vault_path).parts
            # Skip hidden dirs and underscore files
            if any(p.startswith(".") for p in parts):
                continue
            if md.name.startswith("_"):
                continue
            try:
                fm, _ = read_frontmatter(md.read_text())
                if fm.get("learning_stage") != "#learning":
                    continue
                rel_path = str(md.relative_to(vault_path))
                total = get_weighted_total(rel_path)
                if total >= threshold:
                    candidates.append(rel_path)
            except Exception:
                continue
    except Exception:
        pass

    return candidates


def get_templates_dir():
    """Return the installed template resource directory."""
    return resource("templates")


def make_slug(title: str) -> str:
    """Generate a URL-safe slug from a task title.

    Lowercases, replaces non-alphanumeric chars with hyphens,
    strips leading/trailing hyphens, truncates at 50 chars.
    """
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:50]


class GraphWriter:
    """Writes Obsidian-compatible task and project graph notes to the vault.

    On each pb finish call, writes:
    1. A per-task note: {vault}/projects/{project-name}/{YYYY-MM-DD}-{task-slug}.md
    2. A per-project note (create or append): {vault}/projects/{project-name}.md

    If project_id is None, writes task note to: {vault}/tasks/{YYYY-MM-DD}-{slug}.md
    (project note upsert is skipped for unassigned tasks).

    Vault failures are non-fatal: log a warning and return None.
    """

    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = vault_path or get_vault_path()
        self.projects_dir = self.vault_path / "projects"
        self.tasks_dir = self.vault_path / "tasks"  # fallback for unassigned tasks

    def _ensure_dir(self, path: Path) -> bool:
        """Ensure directory exists. Returns False and logs warning on failure."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            logger.warning("graph_writer.mkdir_failed", path=str(path), error=str(e))
            return False

    def _unique_path(self, base_path: Path) -> Path:
        """Return base_path if it does not exist, else append -2, -3, etc."""
        if not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        counter = 2
        while True:
            candidate = parent / f"{stem}-{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _render_task_note(
        self,
        session: Session,
        task: Task,
        project: Optional[Project],
        next_steps: list,
        date_str: str,
        duration_min: int,
    ) -> str:
        """Render task note content using task_note.md template or domain-specific template."""
        domain = getattr(task, 'domain', None)
        if not domain and task.title.startswith("Practice: "):
            domain = task.title.split("Practice: ", 1)[1].strip()
            
        template_name = "task_note.md"
        
        if domain:
            domain_slug = domain.lower().replace("-", "_").replace("/", "_")
            if domain_slug == "rust_c++":
                domain_slug = "rust_cpp"
            possible_template = f"{domain_slug}_session.md"
            if template_exists(possible_template):
                template_name = possible_template
                
        template = Template(read_template_text(template_name))

        project_name = project.name if project else "_unassigned"
        project_tag = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")

        # Derive track tag: first linked track or "untracked"
        track_tag = "untracked"
        if task.linked_track_ids:
            track_tag = re.sub(r"[^a-z0-9]+", "-", task.linked_track_ids[0].lower()).strip("-")

        # Build next_steps wikilinks section
        if next_steps:
            links = "\n".join(f"- [[{s}]]" for s in next_steps)
            next_steps_section = links
        else:
            next_steps_section = "_None_"

        _, duration_display = elapsed_minutes_and_label(session.start_at, session.end_at)

        # Fallback values for basic templates
        kwargs = {
            "project_name": project_name,
            "project_tag": project_tag,
            "track_tag": track_tag,
            "date": date_str,
            "task_title": task.title,
            "topic": task.title,
            "duration_min": str(duration_min),
            "duration": str(duration_min),
            "duration_display": duration_display,
            "expectation": session.expectation or "done",
            "completion_pct": str(session.completion_pct if session.completion_pct is not None else ""),
            "distraction": str(session.distraction if session.distraction is not None else ""),
            "next_steps_section": next_steps_section,
        }

        return template.safe_substitute(**kwargs)

    def _render_project_note(
        self,
        project: Project,
        first_task_link: str,
        date_created: str,
    ) -> str:
        """Render initial project note content using project_note.md template."""
        template = Template(read_template_text("project_note.md"))

        track_tag = "untracked"

        return template.safe_substitute(
            project_name=project.name,
            date_created=date_created,
            status=project.status.value,
            track_tag=track_tag,
            first_task_link=first_task_link,
        )

    def write_task_note(
        self,
        session: Session,
        task: Task,
        project: Optional[Project],
        next_steps: list,
    ) -> Optional[Path]:
        """Write task note to vault. Returns written path or None on failure.

        Path: {vault}/projects/{project-name}/{YYYY-MM-DD}-{task-slug}.md
        Fallback (no project): {vault}/tasks/{YYYY-MM-DD}-{task-slug}.md
        """
        try:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            slug = make_slug(task.title)

            # Determine target directory
            project_name = project.name if project else None
            if project_name:
                target_dir = self.projects_dir / project_name
            else:
                target_dir = self.tasks_dir

            if not self._ensure_dir(target_dir):
                return None

            duration_min, _ = elapsed_minutes_and_label(session.start_at, session.end_at)

            base_path = target_dir / f"{date_str}-{slug}.md"
            note_path = self._unique_path(base_path)

            content = self._render_task_note(
                session=session,
                task=task,
                project=project,
                next_steps=next_steps,
                date_str=date_str,
                duration_min=duration_min,
            )
            note_path.write_text(content)
            logger.info("graph_writer.task_note_written", path=str(note_path))
            return note_path

        except Exception as e:
            logger.warning("graph_writer.write_task_note_failed", error=str(e))
            return None

    def update_state_md(
        self,
        domain_path: Path,
        session_summary: str,
        vault_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Overwrite _state.md with refreshed learning metrics after pb finish (D-06, GRPH-05).

        Enforces 30-line cap (D-05): YAML frontmatter with stage counts + last 3 session
        summaries.  Non-fatal: log warning on failure, return None.
        """
        try:
            import yaml as _yaml
            from pb.vault.lifecycle import read_frontmatter

            state_path = domain_path / "_state.md"

            # Load existing session summaries if present
            existing_summaries: list[str] = []
            if state_path.exists():
                content = state_path.read_text()
                fm_old, _ = read_frontmatter(content)
                existing_summaries = fm_old.get("session_summaries", [])

            # Keep last 3 summaries (D-05)
            summaries = (existing_summaries + [session_summary])[-3:]

            # Compute stage counts from .md files in domain
            stage_counts: dict[str, int] = {
                "new": 0,
                "learning": 0,
                "learnt": 0,
                "stale": 0,
                "archive": 0,
            }
            for md in domain_path.glob("*.md"):
                if md.name.startswith("_"):
                    continue
                try:
                    fm_note, _ = read_frontmatter(md.read_text())
                    stage = fm_note.get("learning_stage", "#new")
                    key = stage.lstrip("#") if isinstance(stage, str) else "new"
                    if key in stage_counts:
                        stage_counts[key] += 1
                    else:
                        stage_counts["new"] += 1
                except Exception:
                    stage_counts["new"] += 1

            fm_new = {
                "type": "domain_state",
                "updated": datetime.now().strftime("%Y-%m-%d"),
                "stage_counts": stage_counts,
                "session_summaries": summaries,
            }
            domain_name = domain_path.name
            body = f"# {domain_name} -- Learning State\n\nUpdated by `pb finish`.\n"

            state_path.write_text(
                f"---\n{_yaml.dump(fm_new, default_flow_style=False, allow_unicode=True)}---\n\n{body}"
            )
            logger.info("graph_writer.state_md_updated", path=str(state_path))
            return state_path
        except Exception as e:
            logger.warning("graph_writer.update_state_md_failed", error=str(e))
            return None

    def upsert_project_note(
        self,
        project: Project,
        task_slug: str,
        date_str: str,
    ) -> Optional[Path]:
        """Create or append to project note. Returns path or None on failure.

        Creates: {vault}/projects/{project-name}.md
        Appends: - [[{date_str}-{task_slug}]] to the ## Completed tasks section.
        """
        try:
            if not self._ensure_dir(self.projects_dir):
                return None

            note_path = self.projects_dir / f"{project.name}.md"
            link = f"{date_str}-{task_slug}"

            if not note_path.exists():
                # Create new project note with frontmatter (D-10)
                content = self._render_project_note(
                    project=project,
                    first_task_link=link,
                    date_created=datetime.utcnow().strftime("%Y-%m-%d"),
                )
                note_path.write_text(content)
            else:
                # Append task link (ensure trailing newline before append)
                existing = note_path.read_text()
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                note_path.write_text(existing + f"- [[{link}]]\n")

            logger.info("graph_writer.project_note_upserted", path=str(note_path))
            return note_path

        except Exception as e:
            logger.warning("graph_writer.upsert_project_note_failed", error=str(e))
            return None
