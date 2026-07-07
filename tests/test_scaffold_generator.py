"""Tests for scaffold_generator.py — Plans 20-01 through 20-03."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# The script is standalone — import via sys.path manipulation
SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


class TestScaffoldConfig:
    """Test ScaffoldConfig model integration."""

    def test_scaffold_config_defaults(self):
        from pb.storage.config import get_config
        cfg = get_config()
        assert hasattr(cfg, "scaffold")
        assert cfg.scaffold.gcs_bucket == ""
        assert cfg.scaffold.gcp_project == ""
        assert cfg.scaffold.location == "us-central1"
        assert cfg.scaffold.poll_interval_seconds == 90
        assert cfg.scaffold.similarity_threshold == 0.6


class TestDomainInference:
    """Test domain inference from cwd."""

    def test_detect_domain_from_cwd_valid(self, tmp_path: Path):
        """Detects domain when cwd is inside a domain folder with _state.md."""
        knowledge_dir = tmp_path / "knowledge"
        domain_dir = knowledge_dir / "math"
        domain_dir.mkdir(parents=True)
        (domain_dir / "_state.md").write_text("---\ntitle: Math\n---\n")

        import scaffold_generator as sg
        with patch("os.getcwd", return_value=str(domain_dir)):
            result = sg._detect_domain_from_cwd(knowledge_dir)
        assert result == "math"

    def test_detect_domain_from_cwd_no_state(self, tmp_path: Path):
        """Returns None when folder has no _state.md."""
        knowledge_dir = tmp_path / "knowledge"
        domain_dir = knowledge_dir / "orphan"
        domain_dir.mkdir(parents=True)

        import scaffold_generator as sg
        with patch("os.getcwd", return_value=str(domain_dir)):
            result = sg._detect_domain_from_cwd(knowledge_dir)
        assert result is None

    def test_detect_domain_from_cwd_outside_vault(self, tmp_path: Path):
        """Returns None when cwd is not under knowledge_dir."""
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir(parents=True)

        import scaffold_generator as sg
        with patch("os.getcwd", return_value="/tmp"):
            result = sg._detect_domain_from_cwd(knowledge_dir)
        assert result is None


class TestArgparse:
    """Test CLI argument parsing."""

    def test_help_output(self):
        """--help runs without error and mentions key flags."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "scaffold_generator.py"), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--sync" in result.stdout
        assert "--retry" in result.stdout

    def test_sync_and_async_mutually_exclusive(self):
        """Cannot pass --sync and --async together."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "scaffold_generator.py"), "--sync", "--async"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0


class TestValidateNote:
    """Test _validate_note lenient validation logic."""

    def test_empty_content_is_broken(self):
        import scaffold_generator as sg
        result = sg._validate_note("test.md", "")
        assert result.is_broken is True

    def test_whitespace_only_is_broken(self):
        import scaffold_generator as sg
        result = sg._validate_note("test.md", "   \n  \n  ")
        assert result.is_broken is True

    def test_no_body_after_frontmatter_is_broken(self):
        import scaffold_generator as sg
        result = sg._validate_note("test.md", "---\ntitle: Test\n---\n")
        assert result.is_broken is True

    def test_valid_note_no_warnings(self):
        import scaffold_generator as sg
        content = (
            "---\ntitle: Test Note\ntags:\n  - new\n  - math\n---\n"
            "- Short bullet point [[related-concept]]\n"
            "- Another bullet [[other-note]]\n"
        )
        result = sg._validate_note("math/test.md", content)
        assert result.is_broken is False
        assert result.warnings == []

    def test_missing_frontmatter_warning(self):
        import scaffold_generator as sg
        content = "Some content without frontmatter [[link]]\n- Bullet\n"
        result = sg._validate_note("test.md", content)
        assert result.is_broken is False
        assert any("no frontmatter" in w for w in result.warnings)

    def test_missing_wikilinks_warning(self):
        import scaffold_generator as sg
        content = "---\ntitle: Test\ntags:\n  - new\n---\n- Bullet without links\n"
        result = sg._validate_note("test.md", content)
        assert result.is_broken is False
        assert any("no wikilinks" in w for w in result.warnings)

    def test_missing_tags_warning(self):
        import scaffold_generator as sg
        content = "---\ntitle: Test\n---\n- Bullet [[link]]\n"
        result = sg._validate_note("test.md", content)
        assert result.is_broken is False
        assert any("no tags" in w for w in result.warnings)

    def test_body_over_2200_chars_warning(self):
        import scaffold_generator as sg
        long_body = "- " + "word " * 500 + "[[link]]\n"  # > 2200 chars
        content = f"---\ntitle: Test\ntags:\n  - new\n---\n{long_body}"
        result = sg._validate_note("test.md", content)
        assert result.is_broken is False
        assert any("2200" in w for w in result.warnings)

    def test_bullet_over_30_words_warning(self):
        import scaffold_generator as sg
        long_bullet = "- " + " ".join(f"word{i}" for i in range(35)) + "\n"
        content = f"---\ntitle: Test\ntags:\n  - new\n---\n{long_bullet}[[link]]\n"
        result = sg._validate_note("test.md", content)
        assert result.is_broken is False
        assert any("30 words" in w for w in result.warnings)


class TestBuildBatchLine:
    """Test JSONL line construction."""

    def test_produces_valid_json(self):
        import scaffold_generator as sg
        line = sg._build_batch_line("test-note", "Generate a note about X", "You are a helper")
        parsed = json.loads(line)
        assert parsed["key"] == "test-note"
        assert "request" in parsed
        assert "contents" in parsed["request"]
        assert "system_instruction" in parsed["request"]
        assert parsed["request"]["generationConfig"]["temperature"] == 0.4
        assert parsed["request"]["generationConfig"]["maxOutputTokens"] == 1024

    def test_key_matches_slug(self):
        import scaffold_generator as sg
        line = sg._build_batch_line("my-slug", "prompt", "system")
        parsed = json.loads(line)
        assert parsed["key"] == "my-slug"

    def test_system_instruction_included(self):
        import scaffold_generator as sg
        line = sg._build_batch_line("k", "p", "my system instruction")
        parsed = json.loads(line)
        parts = parsed["request"]["system_instruction"]["parts"]
        assert parts[0]["text"] == "my system instruction"


class TestParseBatchOutput:
    """Test batch output JSONL parsing with partial failures."""

    def test_all_success(self):
        import scaffold_generator as sg
        jsonl = (
            '{"key": "note1", "status": "", "response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}}\n'
            '{"key": "note2", "status": "", "response": {"candidates": [{"content": {"parts": [{"text": "bye"}]}}]}}\n'
        )
        successes, failures = sg._parse_batch_output(jsonl)
        assert len(successes) == 2
        assert len(failures) == 0

    def test_partial_failure(self):
        import scaffold_generator as sg
        jsonl = (
            '{"key": "note1", "status": "", "response": {}}\n'
            '{"key": "note2", "status": "RESOURCE_EXHAUSTED", "response": {}}\n'
        )
        successes, failures = sg._parse_batch_output(jsonl)
        assert len(successes) == 1
        assert len(failures) == 1
        assert failures[0]["key"] == "note2"
        assert failures[0]["error"] == "RESOURCE_EXHAUSTED"

    def test_unparseable_line(self):
        import scaffold_generator as sg
        jsonl = 'not valid json\n{"key": "note1", "status": ""}\n'
        successes, failures = sg._parse_batch_output(jsonl)
        assert len(successes) == 1
        assert len(failures) == 1
        assert failures[0]["key"] == "unknown"

    def test_empty_input(self):
        import scaffold_generator as sg
        successes, failures = sg._parse_batch_output("")
        assert len(successes) == 0
        assert len(failures) == 0


class TestRetryManifest:
    """Test retry manifest save/load."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        import scaffold_generator as sg
        failed = [{"key": "note1", "error": "timeout"}]
        note_plan = [
            {"path": "math/note1.md", "type": "concept", "title": "Test", "intra_links": []},
            {"path": "math/note2.md", "type": "concept", "title": "Other", "intra_links": []},
        ]
        manifest_path = sg._save_retry_manifest(tmp_path, failed, "abc123", note_plan)
        assert manifest_path.exists()

        run_id, specs = sg._load_retry_manifest(manifest_path)
        assert run_id == "abc123"
        assert len(specs) == 1
        assert specs[0]["path"] == "math/note1.md"


class TestBuildSystemInstruction:
    """Test system instruction construction from learning profile."""

    def test_includes_profile_fields(self):
        import scaffold_generator as sg
        ctx = sg.ScaffoldContext(
            vault_path=Path("/tmp"),
            domain="math",
            domain_dir=Path("/tmp/math"),
            knowledge_dir=Path("/tmp"),
        )
        ctx.learning_profile = {
            "depth": "phd",
            "background": "MSc in CS",
            "goals": ["understand topology", "apply to ML"],
            "application": "research",
            "emphasis": ["connections to physics"],
            "vocabulary_level": "rigorous",
        }
        instruction = sg._build_system_instruction(ctx)
        assert "phd" in instruction
        assert "MSc in CS" in instruction
        assert "understand topology" in instruction
        assert "connections to physics" in instruction
        assert "rigorous" in instruction
        assert "2200" in instruction
        assert "30 words" in instruction
        assert "atomic" in instruction.lower() or "ATOMIC" in instruction
