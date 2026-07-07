"""Tests for ReviewLogWriter — daily and weekly review log notes.

Per Phase 3 D-02, D-03, D-04, D-06, D-07, I-09, I-10.
"""

from datetime import date


class TestDailyReviewLog:
    """Test write_daily_log() creates correct files."""

    def test_daily_log_created_at_correct_path(self, temp_dir):
        """D-02, D-06: Learning/Inbox/pb/reviews/daily/{date}-daily.md"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_daily_log("## Review\nContent here", date(2026, 4, 25))
        assert path is not None
        assert path.exists()
        assert path.name == "2026-04-25-daily.md"
        assert "Learning/Inbox/pb/reviews/daily" in str(path)

    def test_daily_log_frontmatter_contains_type(self, temp_dir):
        """D-04: type: daily_log in frontmatter"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_daily_log("content", date(2026, 4, 25))
        content = path.read_text()
        assert "type: daily_log" in content

    def test_daily_log_frontmatter_contains_date(self, temp_dir):
        """Frontmatter date field matches the review date"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_daily_log("content", date(2026, 4, 25))
        content = path.read_text()
        assert "date: 2026-04-25" in content

    def test_daily_log_body_contains_review_content(self, temp_dir):
        """Content round-trip: body included in output"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_daily_log("## Review\nSpecial content here", date(2026, 4, 25))
        content = path.read_text()
        assert "Special content here" in content

    def test_daily_log_collision_appends_suffix(self, temp_dir):
        """D-07: second write for same day gets -2 suffix"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path1 = writer.write_daily_log("first review", date(2026, 4, 25))
        path2 = writer.write_daily_log("second review", date(2026, 4, 25))
        assert path1 != path2
        assert path1.name == "2026-04-25-daily.md"
        assert path2.name == "2026-04-25-daily-2.md"

    def test_daily_log_unwritable_vault_returns_none(self, tmp_path):
        """I-09: unwritable vault path returns None, no exception"""
        from pb.core.review_log_writer import ReviewLogWriter

        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file, not a directory")
        writer = ReviewLogWriter(vault_path=blocker)
        result = writer.write_daily_log("content", date(2026, 4, 25))
        assert result is None

    def test_daily_log_creates_missing_directory(self, temp_dir):
        """mkdir parents=True: auto-create missing vault directory"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        # vault/80-logs/daily/ does not exist yet
        assert not (vault / "80-logs" / "daily").exists()
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_daily_log("content", date(2026, 4, 25))
        assert path is not None
        assert path.exists()


class TestWeeklyReviewLog:
    """Test write_weekly_log() creates correct files."""

    def test_weekly_log_created_at_correct_path(self, temp_dir):
        """D-03, D-06: Learning/Inbox/pb/reviews/weekly/{YYYY}-W{WW}-weekly.md"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_weekly_log("## Weekly\nContent", date(2026, 4, 25))
        assert path is not None
        assert path.exists()
        assert "Learning/Inbox/pb/reviews/weekly" in str(path)

    def test_weekly_log_uses_iso_week_numbering(self, temp_dir):
        """I-10, Pitfall 3: uses isocalendar(), not strftime('%W')
        2026-01-06 is Tuesday of ISO week 2."""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_weekly_log("content", date(2026, 1, 6))
        assert path.name == "2026-W02-weekly.md"

    def test_weekly_log_iso_week_format(self, temp_dir):
        """D-06: weekly log named {YYYY}-W{WW}-weekly.md with zero-padded week"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        # 2026-04-25 is in ISO week 17
        path = writer.write_weekly_log("content", date(2026, 4, 25))
        assert path.name == "2026-W17-weekly.md"

    def test_weekly_log_frontmatter_contains_type(self, temp_dir):
        """D-04: type: weekly_log in frontmatter"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_weekly_log("content", date(2026, 4, 25))
        content = path.read_text()
        assert "type: weekly_log" in content

    def test_weekly_log_body_contains_review_content(self, temp_dir):
        """Content round-trip: body included in output"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_weekly_log("## Weekly\nSpecial weekly content", date(2026, 4, 25))
        content = path.read_text()
        assert "Special weekly content" in content

    def test_weekly_log_unwritable_vault_returns_none(self, tmp_path):
        """I-09: non-fatal on failure"""
        from pb.core.review_log_writer import ReviewLogWriter

        blocker = tmp_path / "blocker"
        blocker.write_text("file")
        writer = ReviewLogWriter(vault_path=blocker)
        result = writer.write_weekly_log("content", date(2026, 4, 25))
        assert result is None

    def test_weekly_log_creates_missing_directory(self, temp_dir):
        """mkdir parents=True: auto-create missing vault directory"""
        from pb.core.review_log_writer import ReviewLogWriter

        vault = temp_dir / "vault"
        assert not (vault / "80-logs" / "weekly").exists()
        writer = ReviewLogWriter(vault_path=vault)
        path = writer.write_weekly_log("content", date(2026, 4, 25))
        assert path is not None
        assert path.exists()
