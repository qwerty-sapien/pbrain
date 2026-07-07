"""Tests for unified inbox -- vault scanning, frontmatter parsing,
grouped display, and type conversion.

Covers _parse_frontmatter, _scan_vault_inbox, _display_grouped,
_launch_conversion (Skip path), and TYPE_OPTIONS.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pb.cli.commands.inbox import (
    TYPE_OPTIONS,
    _display_grouped,
    _launch_conversion,
    _parse_frontmatter,
    _scan_vault_inbox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEED_FRONTMATTER = """\
---
type: feed_item
title: "How to Build a Second Brain"
source: "HackerNews"
url: "https://example.com/article"
snippet: "First 200 chars of the article..."
ingested: "2026-04-27"
tags: [ai, productivity]
---

# How to Build a Second Brain

Content here.
"""

GMAIL_FRONTMATTER = """\
---
type: gmail_item
subject: "Re: Project update"
sender: "alice@example.com"
date: "2026-04-27"
snippet: "Hey, just wanted to follow up..."
message_id: "18f4a2b3c4d5e6f7"
---

# Re: Project update

Email body.
"""


def _make_task(title: str, archived_at=None):
    """Create a mock task object."""
    return SimpleNamespace(
        id="abc12345-1234-5678-9abc-def012345678",
        title=title,
        state=SimpleNamespace(value="inbox"),
        archived_at=archived_at,
    )


# ---------------------------------------------------------------------------
# _parse_frontmatter tests
# ---------------------------------------------------------------------------


def test_parse_frontmatter_valid():
    """Valid markdown with YAML frontmatter returns correct dict."""
    result = _parse_frontmatter(FEED_FRONTMATTER)
    assert result["type"] == "feed_item"
    assert result["title"] == "How to Build a Second Brain"
    assert result["source"] == "HackerNews"
    assert result["url"] == "https://example.com/article"


def test_parse_frontmatter_empty():
    """Empty string returns empty dict."""
    result = _parse_frontmatter("")
    assert result == {}


def test_parse_frontmatter_no_delimiters():
    """Text without --- delimiters returns empty dict."""
    result = _parse_frontmatter("Just some text without frontmatter.")
    assert result == {}


def test_parse_frontmatter_single_delimiter():
    """Only one --- returns empty dict."""
    result = _parse_frontmatter("---\ntitle: test\n")
    assert result == {}


def test_parse_frontmatter_invalid_yaml():
    """Malformed YAML between delimiters returns empty dict."""
    content = "---\n: [\ninvalid yaml\n---\n\nBody."
    result = _parse_frontmatter(content)
    assert result == {}


def test_parse_frontmatter_gmail():
    """Gmail frontmatter parses correctly."""
    result = _parse_frontmatter(GMAIL_FRONTMATTER)
    assert result["type"] == "gmail_item"
    assert result["subject"] == "Re: Project update"
    assert result["sender"] == "alice@example.com"
    assert result["message_id"] == "18f4a2b3c4d5e6f7"


# ---------------------------------------------------------------------------
# _scan_vault_inbox tests
# ---------------------------------------------------------------------------


def test_scan_vault_inbox_empty(tmp_path):
    """Empty 00-inbox/feeds/ and 00-inbox/gmail/ returns empty list."""
    (tmp_path / "00-inbox" / "feeds").mkdir(parents=True)
    (tmp_path / "00-inbox" / "gmail").mkdir(parents=True)
    result = _scan_vault_inbox(tmp_path)
    assert result == []


def test_scan_vault_inbox_finds_feed_items(tmp_path):
    """Feed .md file with valid frontmatter is found with _source='feeds'."""
    feeds_dir = tmp_path / "00-inbox" / "feeds"
    feeds_dir.mkdir(parents=True)
    (feeds_dir / "article-1.md").write_text(FEED_FRONTMATTER)

    result = _scan_vault_inbox(tmp_path)
    assert len(result) == 1
    assert result[0]["_source"] == "feeds"
    assert result[0]["title"] == "How to Build a Second Brain"
    assert isinstance(result[0]["_path"], Path)
    assert result[0]["_path"].name == "article-1.md"


def test_scan_vault_inbox_finds_gmail_items(tmp_path):
    """Gmail .md file with valid frontmatter is found with _source='gmail'."""
    gmail_dir = tmp_path / "00-inbox" / "gmail"
    gmail_dir.mkdir(parents=True)
    (gmail_dir / "email-1.md").write_text(GMAIL_FRONTMATTER)

    result = _scan_vault_inbox(tmp_path)
    assert len(result) == 1
    assert result[0]["_source"] == "gmail"
    assert result[0]["subject"] == "Re: Project update"


def test_scan_vault_inbox_creates_dirs(tmp_path):
    """If 00-inbox/ doesn't exist, _scan_vault_inbox creates dirs without error."""
    # tmp_path is empty -- no 00-inbox/
    result = _scan_vault_inbox(tmp_path)
    assert result == []
    assert (tmp_path / "00-inbox" / "feeds").is_dir()
    assert (tmp_path / "00-inbox" / "gmail").is_dir()


def test_scan_vault_inbox_skips_bad_frontmatter(tmp_path):
    """Files with empty/invalid frontmatter are skipped."""
    feeds_dir = tmp_path / "00-inbox" / "feeds"
    feeds_dir.mkdir(parents=True)
    (feeds_dir / "bad.md").write_text("No frontmatter here.")
    (feeds_dir / "good.md").write_text(FEED_FRONTMATTER)

    result = _scan_vault_inbox(tmp_path)
    assert len(result) == 1
    assert result[0]["title"] == "How to Build a Second Brain"


def test_scan_vault_inbox_sorts_newest_first(tmp_path):
    """Items are sorted by ingested field descending (newest first)."""
    feeds_dir = tmp_path / "00-inbox" / "feeds"
    feeds_dir.mkdir(parents=True)

    old_content = "---\ntype: feed_item\ntitle: Old Article\ningested: '2026-04-25'\n---\n\n# Old"
    new_content = "---\ntype: feed_item\ntitle: New Article\ningested: '2026-04-27'\n---\n\n# New"

    (feeds_dir / "old.md").write_text(old_content)
    (feeds_dir / "new.md").write_text(new_content)

    result = _scan_vault_inbox(tmp_path)
    assert len(result) == 2
    assert result[0]["title"] == "New Article"
    assert result[1]["title"] == "Old Article"


def test_scan_vault_inbox_mixed_sources(tmp_path):
    """Both feeds and gmail items are returned together."""
    feeds_dir = tmp_path / "00-inbox" / "feeds"
    gmail_dir = tmp_path / "00-inbox" / "gmail"
    feeds_dir.mkdir(parents=True)
    gmail_dir.mkdir(parents=True)

    (feeds_dir / "article.md").write_text(FEED_FRONTMATTER)
    (gmail_dir / "email.md").write_text(GMAIL_FRONTMATTER)

    result = _scan_vault_inbox(tmp_path)
    assert len(result) == 2
    sources = {item["_source"] for item in result}
    assert sources == {"feeds", "gmail"}


# ---------------------------------------------------------------------------
# TYPE_OPTIONS test
# ---------------------------------------------------------------------------


def test_type_options_has_six_entries():
    """TYPE_OPTIONS contains exactly 6 conversion types."""
    assert TYPE_OPTIONS == ["Event", "Opportunity", "Concept", "Task", "Person", "Skip"]
    assert len(TYPE_OPTIONS) == 6


# ---------------------------------------------------------------------------
# _display_grouped tests
# ---------------------------------------------------------------------------


def test_display_grouped_output(capsys):
    """Grouped display shows Gmail and Feeds sections."""
    vault_items = [
        {"_source": "gmail", "subject": "Test Email", "sender": "bob@test.com", "date": "2026-04-27"},
        {"_source": "feeds", "title": "Test Article", "source": "HN", "ingested": "2026-04-27"},
    ]

    _display_grouped(vault_items)

    output = capsys.readouterr().out
    assert "Gmail (1)" in output
    assert "Test Email" in output
    assert "bob@test.com" in output
    assert "Feeds (1)" in output
    assert "Test Article" in output
    assert "HN" in output


def test_display_grouped_empty_sections(capsys):
    """Sections with zero items are not displayed."""
    vault_items = [
        {"_source": "feeds", "title": "Only Feed", "source": "RSS", "ingested": "2026-04-27"},
    ]

    _display_grouped(vault_items)

    output = capsys.readouterr().out
    assert "Feeds (1)" in output
    assert "Gmail" not in output


def test_display_grouped_no_vault_items(capsys):
    """Empty vault items shows nothing."""
    _display_grouped([])

    output = capsys.readouterr().out
    assert "Gmail" not in output
    assert "Feeds" not in output


# ---------------------------------------------------------------------------
# _launch_conversion tests
# ---------------------------------------------------------------------------


def test_launch_conversion_skip(tmp_path):
    """Skip returns True and does NOT delete the file."""
    feeds_dir = tmp_path / "00-inbox" / "feeds"
    feeds_dir.mkdir(parents=True)
    md_file = feeds_dir / "keep-me.md"
    md_file.write_text(FEED_FRONTMATTER)

    item = _parse_frontmatter(FEED_FRONTMATTER)
    item["_path"] = md_file
    item["_source"] = "feeds"

    result = _launch_conversion(item, "Skip", tmp_path)
    assert result is True
    assert md_file.exists(), "File should NOT be deleted on Skip"
