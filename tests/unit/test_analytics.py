"""Unit tests for analytics helpers in review.py — Phase 19 ANLT-01/02/03.

TDD RED phase: these tests are written before implementation.
Covers snapshot_domain_stats, build_growth_table, get_zero_activity_domains.
"""

import datetime
from pathlib import Path

import pytest

from pb.storage.database import get_connection, init_db, set_db_path


# ---------------------------------------------------------------------------
# Helpers to build a minimal vault structure in tmp_path
# ---------------------------------------------------------------------------

def _make_note(domain_dir: Path, name: str, stage: str = "#new") -> None:
    """Create a minimal note with learning_stage frontmatter."""
    (domain_dir / f"{name}.md").write_text(
        f"---\nlearning_stage: \"{stage}\"\n---\n\nBody text.\n"
    )


def _make_domain(knowledge_dir: Path, domain: str, notes: list[tuple[str, str]] | None = None) -> Path:
    """Create a domain directory with _state.md and optional notes."""
    domain_dir = knowledge_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / "_state.md").write_text("# State\n")
    for name, stage in (notes or []):
        _make_note(domain_dir, name, stage)
    return domain_dir


# ---------------------------------------------------------------------------
# Test 1: snapshot_domain_stats creates rows for each domain
# ---------------------------------------------------------------------------

class TestSnapshotDomainStats:

    def test_creates_rows_for_each_domain(self, temp_db, tmp_path):
        """snapshot_domain_stats inserts one row per domain."""
        from pb.cli.commands.review import snapshot_domain_stats

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        _make_domain(knowledge_dir, "piano", [("note1", "#new"), ("note2", "#learning")])
        _make_domain(knowledge_dir, "ml", [("note-a", "#learnt")])

        snapshot_domain_stats(vault)

        with get_connection() as conn:
            rows = conn.execute("SELECT domain FROM domain_weekly_stats ORDER BY domain").fetchall()
        domains = [r[0] for r in rows]
        assert "ml" in domains
        assert "piano" in domains

    def test_idempotent_upsert_same_week(self, temp_db, tmp_path):
        """Calling snapshot_domain_stats twice in the same week produces exactly one row per domain."""
        from pb.cli.commands.review import snapshot_domain_stats

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        _make_domain(knowledge_dir, "math", [("note1", "#new")])

        snapshot_domain_stats(vault)
        snapshot_domain_stats(vault)  # second call same week

        with get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM domain_weekly_stats WHERE domain = 'math'"
            ).fetchone()[0]
        assert count == 1

    def test_counts_notes_in_domain(self, temp_db, tmp_path):
        """snapshot_domain_stats counts total notes in notes_created column."""
        from pb.cli.commands.review import snapshot_domain_stats

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        _make_domain(knowledge_dir, "piano", [
            ("note1", "#new"),
            ("note2", "#learning"),
            ("note3", "#learnt"),
        ])

        snapshot_domain_stats(vault)

        with get_connection() as conn:
            row = conn.execute(
                "SELECT notes_created FROM domain_weekly_stats WHERE domain = 'piano'"
            ).fetchone()
        assert row is not None
        assert row[0] == 3

    def test_counts_socratic_sessions(self, temp_db, tmp_path):
        """snapshot_domain_stats counts socratic interactions from pb.db."""
        from pb.cli.commands.review import snapshot_domain_stats

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        _make_domain(knowledge_dir, "languages", [("hello", "#learning")])

        today = datetime.date.today()
        week_start = (today - datetime.timedelta(days=today.weekday())).isoformat()
        week_start_ts = f"{week_start}T00:00:00"

        with get_connection() as conn:
            conn.execute(
                "INSERT INTO interactions (note_path, event_type, weight, ts, domain) VALUES (?, ?, ?, ?, ?)",
                ("languages/hello.md", "socratic", 5.0, week_start_ts, "languages"),
            )
            conn.execute(
                "INSERT INTO interactions (note_path, event_type, weight, ts, domain) VALUES (?, ?, ?, ?, ?)",
                ("languages/hello.md", "socratic", 5.0, week_start_ts, "languages"),
            )
            conn.commit()

        snapshot_domain_stats(vault)

        with get_connection() as conn:
            row = conn.execute(
                "SELECT socratic_sessions FROM domain_weekly_stats WHERE domain = 'languages'"
            ).fetchone()
        assert row is not None
        assert row[0] == 2

    def test_skips_dotfiles_and_non_dirs(self, temp_db, tmp_path):
        """snapshot_domain_stats skips dotfiles and files in knowledge."""
        from pb.cli.commands.review import snapshot_domain_stats

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        # Hidden dir should be skipped
        hidden = knowledge_dir / ".cache"
        hidden.mkdir()
        (hidden / "_state.md").write_text("# State\n")

        # Valid domain
        _make_domain(knowledge_dir, "math", [("theorem1", "#new")])

        snapshot_domain_stats(vault)

        with get_connection() as conn:
            rows = conn.execute("SELECT domain FROM domain_weekly_stats").fetchall()
        domains = [r[0] for r in rows]
        assert ".cache" not in domains
        assert "math" in domains


# ---------------------------------------------------------------------------
# Test 5-7: build_growth_table
# ---------------------------------------------------------------------------

class TestBuildGrowthTable:

    def _insert_stats(self, conn, domain: str, week_start: str, notes: int = 5,
                      links: int = 2, anki: int = 0, sessions: int = 1,
                      stage_new: int = 3, stage_learning: int = 1, stage_learnt: int = 1, stage_stale: int = 0):
        conn.execute(
            """INSERT OR REPLACE INTO domain_weekly_stats
               (domain, week_start, notes_created, links_added, anki_exported,
                socratic_sessions, stage_new, stage_learning, stage_learnt, stage_stale, snapshot_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (domain, week_start, notes, links, anki, sessions,
             stage_new, stage_learning, stage_learnt, stage_stale,
             datetime.date.today().isoformat()),
        )
        conn.commit()

    def test_returns_rich_table_with_required_columns(self, temp_db, tmp_path):
        """build_growth_table returns a Rich Table with DOMAIN, NOTES, LINKS, ANKI, SESSIONS, STAGES columns."""
        from pb.cli.commands.review import build_growth_table

        today = datetime.date.today()
        week_start = (today - datetime.timedelta(days=today.weekday())).isoformat()

        with get_connection() as conn:
            self._insert_stats(conn, "piano", week_start)

        vault = tmp_path / "vault"
        (vault / "knowledge").mkdir(parents=True, exist_ok=True)

        table = build_growth_table(vault)
        assert table is not None

        col_names = [col.header for col in table.columns]
        assert "DOMAIN" in col_names
        assert "NOTES" in col_names
        assert "LINKS" in col_names
        assert "ANKI" in col_names
        assert "SESSIONS" in col_names
        assert "STAGES" in col_names

    def test_shows_delta_arrows_for_increase(self, temp_db, tmp_path):
        """build_growth_table includes +N delta markup when notes_created increased vs previous week."""
        from pb.cli.commands.review import build_growth_table
        from rich.console import Console
        from io import StringIO

        today = datetime.date.today()
        week_start = (today - datetime.timedelta(days=today.weekday())).isoformat()
        prev_week_start = (today - datetime.timedelta(days=today.weekday() + 7)).isoformat()

        with get_connection() as conn:
            # Previous: 3 notes; Current: 7 notes → delta +4
            self._insert_stats(conn, "ml", prev_week_start, notes=3)
            self._insert_stats(conn, "ml", week_start, notes=7)

        vault = tmp_path / "vault"
        (vault / "knowledge").mkdir(parents=True, exist_ok=True)

        table = build_growth_table(vault)
        assert table is not None

        # Render to plain text to inspect delta
        buf = StringIO()
        c = Console(file=buf, highlight=False, markup=True, width=120)
        c.print(table)
        output = buf.getvalue()
        assert "+4" in output

    def test_shows_delta_arrows_for_decrease(self, temp_db, tmp_path):
        """build_growth_table includes -N delta markup when notes_created decreased vs previous week."""
        from pb.cli.commands.review import build_growth_table
        from rich.console import Console
        from io import StringIO

        today = datetime.date.today()
        week_start = (today - datetime.timedelta(days=today.weekday())).isoformat()
        prev_week_start = (today - datetime.timedelta(days=today.weekday() + 7)).isoformat()

        with get_connection() as conn:
            # Previous: 10; Current: 6 → delta -4
            self._insert_stats(conn, "languages", prev_week_start, notes=10)
            self._insert_stats(conn, "languages", week_start, notes=6)

        vault = tmp_path / "vault"
        (vault / "knowledge").mkdir(parents=True, exist_ok=True)

        table = build_growth_table(vault)
        assert table is not None

        buf = StringIO()
        c = Console(file=buf, highlight=False, markup=True, width=120)
        c.print(table)
        output = buf.getvalue()
        assert "-4" in output

    def test_shows_warning_for_zero_activity_domains(self, temp_db, tmp_path):
        """build_growth_table shows 'no activity' for domains with 0 interactions in past 7 days."""
        from pb.cli.commands.review import build_growth_table
        from rich.console import Console
        from io import StringIO

        today = datetime.date.today()
        week_start = (today - datetime.timedelta(days=today.weekday())).isoformat()

        with get_connection() as conn:
            self._insert_stats(conn, "piano", week_start, sessions=0)

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        _make_domain(knowledge_dir, "piano", [("sonata", "#learning")])

        # No interactions in pb.db → zero activity
        table = build_growth_table(vault)
        assert table is not None

        buf = StringIO()
        c = Console(file=buf, highlight=False, markup=False, width=160)
        c.print(table)
        output = buf.getvalue()
        assert "no activity" in output

    def test_returns_none_when_no_current_week_data(self, temp_db, tmp_path):
        """build_growth_table returns None if domain_weekly_stats has no rows for this week."""
        from pb.cli.commands.review import build_growth_table

        vault = tmp_path / "vault"
        (vault / "knowledge").mkdir(parents=True, exist_ok=True)

        result = build_growth_table(vault)
        assert result is None


# ---------------------------------------------------------------------------
# Test 8: get_zero_activity_domains
# ---------------------------------------------------------------------------

class TestGetZeroActivityDomains:

    def test_returns_domains_with_notes_but_no_recent_interactions(self, temp_db, tmp_path):
        """get_zero_activity_domains returns domains that have notes but 0 interactions in past 7 days."""
        from pb.cli.commands.review import get_zero_activity_domains

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        _make_domain(knowledge_dir, "neglected", [("old-note", "#learning")])

        # No interactions at all → should appear as zero-activity
        result = get_zero_activity_domains(vault)
        assert "neglected" in result

    def test_excludes_active_domains(self, temp_db, tmp_path):
        """get_zero_activity_domains excludes domains that have recent interactions."""
        from pb.cli.commands.review import get_zero_activity_domains

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        _make_domain(knowledge_dir, "active-domain", [("note1", "#learning")])

        now_ts = datetime.datetime.utcnow().isoformat()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO interactions (note_path, event_type, weight, ts, domain) VALUES (?, ?, ?, ?, ?)",
                ("active-domain/note1.md", "read", 1.0, now_ts, "active-domain"),
            )
            conn.commit()

        result = get_zero_activity_domains(vault)
        assert "active-domain" not in result

    def test_excludes_empty_domains(self, temp_db, tmp_path):
        """get_zero_activity_domains excludes domains that have no notes at all."""
        from pb.cli.commands.review import get_zero_activity_domains

        vault = tmp_path / "vault"
        knowledge_dir = vault / "knowledge"
        # Domain with no notes (only _state.md)
        _make_domain(knowledge_dir, "empty-domain", notes=[])

        result = get_zero_activity_domains(vault)
        assert "empty-domain" not in result
