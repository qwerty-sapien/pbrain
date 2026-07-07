"""RED tests for SocraticService detect_domain + debrief methods (Plan 02 Task 1).

Tests:
    1. detect_domain returns domain when PB_SHELL_VAULT_CWD points inside knowledge_dir with _state.md
    2. detect_domain returns None when PB_SHELL_VAULT_CWD points outside knowledge_dir
    3. detect_domain returns None when no _state.md in candidate dir
    4. run_note_debrief instantiates SocraticDebriefEngine with correct args and returns pairs
    5. run_study_debrief calls SocraticDebriefEngine with max_rounds=5
    6. run_finish_debrief returns [] when is_domain_session returns False
    7. run_finish_debrief calls SocraticDebriefEngine with max_rounds=3 on domain session
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def vault_path(tmp_path):
    """Create a tmp vault with knowledge/piano/_state.md."""
    knowledge_dir = tmp_path / "knowledge"
    piano_dir = knowledge_dir / "piano"
    piano_dir.mkdir(parents=True)
    (piano_dir / "_state.md").write_text("# Piano state\nCurrent topic: scales")
    return tmp_path


@pytest.fixture
def knowledge_dir(vault_path):
    return vault_path / "knowledge"


@pytest.fixture
def service(vault_path):
    from pb.vault.socratic_service import SocraticService
    return SocraticService(vault_path=vault_path)


# ---------------------------------------------------------------------------
# Test 1: detect_domain returns domain when PB_SHELL_VAULT_CWD inside knowledge_dir
# ---------------------------------------------------------------------------

def test_detect_domain_from_env_var(service, knowledge_dir, monkeypatch):
    """detect_domain returns 'piano' when PB_SHELL_VAULT_CWD=knowledge_dir/piano."""
    monkeypatch.setenv("PB_SHELL_VAULT_CWD", str(knowledge_dir / "piano"))
    result = service.detect_domain(knowledge_dir)
    assert result == "piano"


# ---------------------------------------------------------------------------
# Test 2: detect_domain returns None when PB_SHELL_VAULT_CWD points outside knowledge_dir
# ---------------------------------------------------------------------------

def test_detect_domain_outside_knowledge_dir(service, knowledge_dir, tmp_path, monkeypatch):
    """detect_domain returns None when env var points outside knowledge_dir."""
    outside = tmp_path / "some-other-dir"
    outside.mkdir()
    monkeypatch.setenv("PB_SHELL_VAULT_CWD", str(outside))
    result = service.detect_domain(knowledge_dir)
    assert result is None


# ---------------------------------------------------------------------------
# Test 3: detect_domain returns None when no _state.md in candidate dir
# ---------------------------------------------------------------------------

def test_detect_domain_no_state_md(service, knowledge_dir, monkeypatch):
    """detect_domain returns None when domain dir exists but has no _state.md."""
    no_state_dir = knowledge_dir / "guitar"
    no_state_dir.mkdir()
    monkeypatch.setenv("PB_SHELL_VAULT_CWD", str(no_state_dir))
    result = service.detect_domain(knowledge_dir)
    assert result is None


# ---------------------------------------------------------------------------
# Test 4: run_note_debrief instantiates SocraticDebriefEngine with correct args
# ---------------------------------------------------------------------------

def test_run_note_debrief_instantiates_engine(service, vault_path, monkeypatch):
    """run_note_debrief calls SocraticDebriefEngine(domain=..., max_rounds=3) and returns Q&A pairs."""
    captured_kwargs = {}
    fake_pairs = [("Q1", "A1"), ("Q2", "A2")]

    class FakeEngine:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    def fake_run_debrief_loop(engine, console):
        return fake_pairs

    monkeypatch.setattr("pb.vault.socratic.SocraticDebriefEngine", FakeEngine)
    monkeypatch.setattr("pb.vault.socratic.run_debrief_loop", fake_run_debrief_loop)

    console = MagicMock()
    result = service.run_note_debrief(topic="VAE", domain="piano", max_rounds=3, console=console)

    assert result == fake_pairs
    assert captured_kwargs.get("domain") == "piano"
    assert captured_kwargs.get("max_rounds") == 3
    # state_md_content should contain the piano _state.md content
    assert "Piano state" in captured_kwargs.get("state_md_content", "")


# ---------------------------------------------------------------------------
# Test 5: run_study_debrief calls SocraticDebriefEngine with max_rounds=5
# ---------------------------------------------------------------------------

def test_run_study_debrief_uses_five_rounds(service, vault_path, monkeypatch):
    """run_study_debrief always uses max_rounds=5 per D-12."""
    captured_kwargs = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    def fake_run_debrief_loop(engine, console):
        return [("Q1", "A1")]

    monkeypatch.setattr("pb.vault.socratic.SocraticDebriefEngine", FakeEngine)
    monkeypatch.setattr("pb.vault.socratic.run_debrief_loop", fake_run_debrief_loop)

    console = MagicMock()
    result = service.run_study_debrief(domain="piano", console=console)

    assert captured_kwargs.get("max_rounds") == 5
    assert captured_kwargs.get("domain") == "piano"


def test_run_adaptive_diagnostic_sets_strict_adaptive_flags(service, monkeypatch):
    """run_adaptive_diagnostic should configure the engine for long-form strict probing."""
    captured_kwargs = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    def fake_run_debrief_loop(engine, console):
        return [("Q1", "A1")]

    monkeypatch.setattr("pb.vault.socratic.SocraticDebriefEngine", FakeEngine)
    monkeypatch.setattr("pb.vault.socratic.run_debrief_loop", fake_run_debrief_loop)

    console = MagicMock()
    result = service.run_adaptive_diagnostic(
        domain="piano",
        console=console,
        topic="scales",
        difficulty_start="undergrad",
        difficulty_limit="first year phd",
        max_rounds=30,
        soft_cap_rounds=24,
        model="flash",
    )

    assert result == [("Q1", "A1")]
    assert captured_kwargs["adaptive"] is True
    assert captured_kwargs["strict"] is True
    assert captured_kwargs["difficulty_start"] == "undergrad"
    assert captured_kwargs["difficulty_limit"] == "first year phd"
    assert captured_kwargs["soft_cap_rounds"] == 24
    assert captured_kwargs["model"] == "flash"


def test_run_adaptive_diagnostic_passes_time_limit(service, monkeypatch):
    """Adaptive diagnostics should carry the configured time limit into the engine."""
    captured_kwargs = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    def fake_run_debrief_loop(engine, console):
        return [("Q1", "A1")]

    monkeypatch.setattr("pb.vault.socratic.SocraticDebriefEngine", FakeEngine)
    monkeypatch.setattr("pb.vault.socratic.run_debrief_loop", fake_run_debrief_loop)

    console = MagicMock()
    service.run_adaptive_diagnostic(
        domain="piano",
        console=console,
        time_limit_minutes=12,
    )

    assert captured_kwargs["time_limit_minutes"] == 12


# ---------------------------------------------------------------------------
# Test 6: run_finish_debrief returns [] when is_domain_session returns False
# ---------------------------------------------------------------------------

def test_run_finish_debrief_skips_non_domain(service, monkeypatch):
    """run_finish_debrief returns [] when is_domain_session returns False."""
    monkeypatch.setattr("pb.vault.socratic.is_domain_session", lambda task, vault_path: False)

    fake_session = MagicMock()
    fake_task = MagicMock()
    console = MagicMock()

    result = service.run_finish_debrief(session=fake_session, task=fake_task, console=console)
    assert result == []


# ---------------------------------------------------------------------------
# Test 7: run_finish_debrief calls SocraticDebriefEngine with max_rounds=3
# ---------------------------------------------------------------------------

def test_run_finish_debrief_uses_three_rounds(service, vault_path, monkeypatch):
    """run_finish_debrief uses max_rounds=3 and returns Q&A pairs on domain session."""
    captured_kwargs = {}
    fake_pairs = [("Q1", "A1"), ("Q2", "A2")]

    class FakeEngine:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    def fake_run_debrief_loop(engine, console):
        return fake_pairs

    # is_domain_session returns True — domain session
    monkeypatch.setattr("pb.vault.socratic.is_domain_session", lambda task, vault_path: True)
    monkeypatch.setattr("pb.vault.socratic.SocraticDebriefEngine", FakeEngine)
    monkeypatch.setattr("pb.vault.socratic.run_debrief_loop", fake_run_debrief_loop)

    # Task has domain attribute to infer from
    fake_task = MagicMock()
    fake_task.domain = "piano"
    fake_session = MagicMock()
    console = MagicMock()

    result = service.run_finish_debrief(session=fake_session, task=fake_task, console=console)

    assert result == fake_pairs
    assert captured_kwargs.get("max_rounds") == 3
    assert captured_kwargs.get("domain") == "piano"


def test_build_diagnostic_report_reads_state_and_returns_dict(service, monkeypatch):
    """build_diagnostic_report should delegate with the current domain state."""
    captured = {}

    def fake_build_report(qa_pairs, domain, state_md_content, **kwargs):
        captured["qa_pairs"] = qa_pairs
        captured["domain"] = domain
        captured["state"] = state_md_content
        captured["kwargs"] = kwargs
        return {"summary": "ok", "knowledge_gaps": []}

    monkeypatch.setattr("pb.vault.socratic.build_diagnostic_report", fake_build_report)

    report = service.build_diagnostic_report(
        [("Q1", "A1")],
        "piano",
        topic="scales",
        difficulty_start="foundational basics",
        difficulty_limit="undergrad",
        note_types=["Basic", "Cloze"],
        model="flash",
    )

    assert report["summary"] == "ok"
    assert captured["domain"] == "piano"
    assert "Piano state" in captured["state"]
    assert captured["kwargs"]["note_types"] == ["Basic", "Cloze"]


def test_save_teach_lesson_writes_domain_note(service, vault_path, monkeypatch):
    """Teach sessions should persist a durable lesson note directly into the vault."""
    fake_conn = MagicMock()
    fake_conn.close = MagicMock()

    monkeypatch.setattr(
        service,
        "_draft_teach_lesson_sections",
        lambda **kwargs: {
            "summary": "Summary",
            "key_insight": "Insight",
            "downstream_concepts": ["Next concept"],
            "next_attempts": ["Try again from memory"],
        },
    )
    monkeypatch.setattr("pb.vault.socratic.infer_wikilinks", lambda *a, **kw: ["scales"])
    monkeypatch.setattr(
        "pb.vault.socratic.build_teach_lesson_note",
        lambda **kwargs: "# Teach note\n\nBody",
    )
    monkeypatch.setattr("pb.vault.graph_store.open_vault_db", lambda *a, **kw: fake_conn)
    monkeypatch.setattr("pb.vault.graph_store.upsert_node", lambda *a, **kw: None)
    monkeypatch.setattr("pb.vault.graph_store.add_link", lambda *a, **kw: None)
    monkeypatch.setattr("pb.vault.lifecycle.log_interaction", lambda *a, **kw: None)

    rel_path = service.save_teach_lesson(
        qa_pairs=[("Q1", "A1")],
        domain="piano",
        topic="major scales",
        completed=True,
    )

    assert rel_path.startswith("knowledge/piano/")
    saved = vault_path / rel_path
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == "# Teach note\n\nBody"


def test_cache_diagnostic_transcript_writes_note(service, vault_path, monkeypatch):
    """Adaptive diagnostics should always be cached into the vault for later inference."""
    fake_conn = MagicMock()
    fake_conn.close = MagicMock()

    monkeypatch.setattr("pb.vault.socratic.infer_wikilinks", lambda *a, **kw: [])
    monkeypatch.setattr(
        "pb.vault.socratic.build_socratic_note",
        lambda **kwargs: "---\nsource: socratic\n---\n\nTranscript body",
    )
    monkeypatch.setattr("pb.vault.graph_store.open_vault_db", lambda *a, **kw: fake_conn)
    monkeypatch.setattr("pb.vault.graph_store.upsert_node", lambda *a, **kw: None)
    monkeypatch.setattr("pb.vault.graph_store.add_link", lambda *a, **kw: None)
    monkeypatch.setattr("pb.vault.lifecycle.log_interaction", lambda *a, **kw: None)

    rel_path = service.cache_diagnostic_transcript(
        qa_pairs=[("Q1", "A1")],
        domain="piano",
        topic="scales",
    )

    assert rel_path == "knowledge/piano/scales-diagnostic.md"
    saved = vault_path / rel_path
    assert saved.exists()
    contents = saved.read_text(encoding="utf-8")
    assert "conversation_kind: diagnostic" in contents
    assert "Transcript body" in contents
