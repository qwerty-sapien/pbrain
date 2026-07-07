"""RED tests for SocraticService.build_and_submit + suggest_bridge (Plan 02 Task 2).

Tests:
    1. build_and_submit sync=True writes note to vault, calls graph/anki/lifecycle, returns rel path
    2. build_and_submit sync=False with GCS configured returns job_name without writing immediately
    3. build_and_submit sync=False with GCS NOT configured falls back to sync + console warning
    4. Async completion handler fires osascript notification (subprocess.run called)
    5. suggest_bridge returns immediately when qa_pairs < 4 (no Flash Lite call)
    6. suggest_bridge invokes bridge logic when qa_pairs >= 4
    7. template param forwarded verbatim to build_socratic_note
"""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault_path(tmp_path):
    """Create a tmp vault with knowledge/ml/_state.md."""
    knowledge_dir = tmp_path / "knowledge"
    ml_dir = knowledge_dir / "ml"
    ml_dir.mkdir(parents=True)
    (ml_dir / "_state.md").write_text("# ML state\nCurrent topic: VAE")
    return tmp_path


@pytest.fixture
def service(vault_path):
    from pb.vault.socratic_service import SocraticService
    return SocraticService(vault_path=vault_path)


@pytest.fixture
def console():
    return MagicMock(spec=["print", "rule"])


@pytest.fixture
def simple_qa():
    return [("What is a VAE?", "A variational autoencoder.")]


@pytest.fixture
def four_qa():
    return [
        ("Q1", "A1 about machine learning"),
        ("Q2", "A2 about encoders"),
        ("Q3", "A3 about latent spaces"),
        ("Q4", "A4 referencing concepts from physics"),
    ]


# ---------------------------------------------------------------------------
# Helper: patch common sync dependencies
# ---------------------------------------------------------------------------

def _patch_sync_deps(monkeypatch, note_content="---\nsource: socratic\n---\n\nbody"):
    """Patch all sync-path external calls."""
    fake_conn = MagicMock()
    fake_conn.__enter__ = lambda s: s
    fake_conn.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("pb.vault.socratic.infer_wikilinks", lambda *a, **kw: [])
    monkeypatch.setattr(
        "pb.vault.socratic.build_socratic_note",
        lambda *a, **kw: note_content,
    )
    monkeypatch.setattr(
        "pb.vault.socratic.show_note_preview_and_confirm",
        lambda console, content, *a, **kw: content,  # return as-is (user "saved")
    )
    monkeypatch.setattr(
        "pb.vault.socratic.extract_socratic_cards",
        lambda *a, **kw: [{"id": "card1"}],
    )
    monkeypatch.setattr("pb.mcp.tools.vault.vault_write", lambda path, content, **kw: path)
    monkeypatch.setattr("pb.vault.graph_store.open_vault_db", lambda *a: fake_conn)
    monkeypatch.setattr("pb.vault.graph_store.upsert_node", lambda *a: None)
    monkeypatch.setattr("pb.vault.graph_store.add_link", lambda *a: None)
    monkeypatch.setattr("pb.vault.anki_client.insert_cards_to_db", lambda cards: len(cards))
    monkeypatch.setattr("pb.vault.lifecycle.log_interaction", lambda *a, **kw: None)
    return fake_conn


# ---------------------------------------------------------------------------
# Test 1: sync=True writes note, calls dependencies, returns rel path
# ---------------------------------------------------------------------------

def test_build_and_submit_sync_returns_rel_path(service, simple_qa, console, monkeypatch):
    """build_and_submit(sync=True) writes note and returns vault-relative path."""
    written_paths = []
    _patch_sync_deps(monkeypatch)
    monkeypatch.setattr(
        "pb.mcp.tools.vault.vault_write",
        lambda path, content, **kw: written_paths.append(path) or path,
    )
    log_calls = []
    monkeypatch.setattr(
        "pb.vault.lifecycle.log_interaction",
        lambda path, event, domain="": log_calls.append((path, event, domain)),
    )

    result = service.build_and_submit(
        qa_pairs=simple_qa,
        domain="ml",
        slug="vae",
        template="brief",
        sync=True,
        model="gemini-3.1-flash-lite-preview",
        console=console,
    )

    assert result == "knowledge/ml/vae.md"
    assert written_paths == ["knowledge/ml/vae.md"]
    assert log_calls == [("knowledge/ml/vae.md", "socratic", "ml")]


def test_build_and_submit_sync_calls_anki(service, simple_qa, console, monkeypatch):
    """build_and_submit(sync=True) calls insert_cards_to_db with extracted cards."""
    _patch_sync_deps(monkeypatch)
    anki_calls = []
    monkeypatch.setattr(
        "pb.vault.anki_client.insert_cards_to_db",
        lambda cards: anki_calls.append(cards) or len(cards),
    )

    service.build_and_submit(
        qa_pairs=simple_qa,
        domain="ml",
        slug="vae",
        template="brief",
        sync=True,
        model="gemini-3.1-flash-lite-preview",
        console=console,
    )

    assert len(anki_calls) == 1


def test_build_and_submit_sync_calls_upsert_node(service, simple_qa, console, monkeypatch):
    """build_and_submit(sync=True) calls upsert_node with slug and subfolder."""
    _patch_sync_deps(monkeypatch)
    upsert_calls = []
    monkeypatch.setattr(
        "pb.vault.graph_store.upsert_node",
        lambda conn, slug, subfolder: upsert_calls.append((slug, subfolder)),
    )

    service.build_and_submit(
        qa_pairs=simple_qa,
        domain="ml",
        slug="vae",
        template="brief",
        sync=True,
        model="gemini-3.1-flash-lite-preview",
        console=console,
    )

    assert ("vae", "knowledge/ml") in upsert_calls


# ---------------------------------------------------------------------------
# Test 2: sync=False with GCS configured returns job_name (no immediate write)
# ---------------------------------------------------------------------------

def test_build_and_submit_async_returns_job_name(service, simple_qa, console, monkeypatch):
    """build_and_submit(sync=False) with GCS configured returns job_name, no immediate write."""
    written_paths = []

    # Patch config with GCS configured
    fake_cfg = MagicMock()
    fake_cfg.scaffold.gcs_bucket = "my-bucket"
    fake_cfg.scaffold.gcp_project = "my-project"
    fake_cfg.scaffold.location = "us-central1"
    monkeypatch.setattr("pb.storage.config.get_config", lambda: fake_cfg)

    # Patch VertexBatchClient
    fake_client = MagicMock()
    fake_client.submit.return_value = "projects/my-project/jobs/batch-123"
    fake_client.poll_and_download.return_value = ([], [])

    monkeypatch.setattr(
        "pb.vault.batch_client.VertexBatchClient",
        lambda **kw: fake_client,
    )
    monkeypatch.setattr(
        "pb.mcp.tools.vault.vault_write",
        lambda path, content, **kw: written_paths.append(path) or path,
    )
    monkeypatch.setattr("pb.vault.socratic.infer_wikilinks", lambda *a, **kw: [])

    result = service.build_and_submit(
        qa_pairs=simple_qa,
        domain="ml",
        slug="vae-async",
        template="brief",
        sync=False,
        model="gemini-3.1-flash-lite-preview",
        console=console,
    )

    assert result == "projects/my-project/jobs/batch-123"
    # Note is NOT written immediately
    assert "knowledge/ml/vae-async.md" not in written_paths


# ---------------------------------------------------------------------------
# Test 3: sync=False with GCS NOT configured falls back to sync with warning
# ---------------------------------------------------------------------------

def test_build_and_submit_async_no_gcs_falls_back(service, simple_qa, console, monkeypatch):
    """build_and_submit(sync=False) without GCS falls back to sync and warns via console."""
    _patch_sync_deps(monkeypatch)

    # Patch config with NO GCS
    fake_cfg = MagicMock()
    fake_cfg.scaffold.gcs_bucket = ""
    fake_cfg.scaffold.gcp_project = ""
    monkeypatch.setattr("pb.storage.config.get_config", lambda: fake_cfg)

    result = service.build_and_submit(
        qa_pairs=simple_qa,
        domain="ml",
        slug="vae-fallback",
        template="brief",
        sync=False,
        model="gemini-3.1-flash-lite-preview",
        console=console,
    )

    # Falls back to sync path
    assert result == "knowledge/ml/vae-fallback.md"
    # Console warned
    console.print.assert_called()
    warn_args = [str(c) for c in console.print.call_args_list]
    assert any("GCS not configured" in a or "warn" in a.lower() for a in warn_args)


# ---------------------------------------------------------------------------
# Test 4: Async completion thread calls osascript
# ---------------------------------------------------------------------------

def test_build_and_submit_async_fires_notification(service, simple_qa, console, monkeypatch):
    """Async polling thread calls subprocess.run with osascript on completion."""
    done_event = threading.Event()
    osascript_calls = []

    fake_cfg = MagicMock()
    fake_cfg.scaffold.gcs_bucket = "bucket"
    fake_cfg.scaffold.gcp_project = "proj"
    fake_cfg.scaffold.location = "us-central1"
    monkeypatch.setattr("pb.storage.config.get_config", lambda: fake_cfg)

    fake_client = MagicMock()
    fake_client.submit.return_value = "projects/proj/jobs/batch-notify"
    fake_client.poll_and_download.return_value = (
        [{"key": "vae-notify", "content": "LLM body text"}],
        [],
    )
    monkeypatch.setattr("pb.vault.batch_client.VertexBatchClient", lambda **kw: fake_client)

    # Patch _finalise_async_note to avoid heavy deps
    monkeypatch.setattr(
        service,
        "_finalise_async_note",
        lambda *a, **kw: None,
    )

    orig_subprocess_run = __import__("subprocess").run

    def fake_subprocess_run(cmd, **kw):
        if cmd and len(cmd) > 1 and cmd[0] == "osascript":
            osascript_calls.append(cmd)
            done_event.set()
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)
    monkeypatch.setattr("pb.vault.socratic.infer_wikilinks", lambda *a, **kw: [])

    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")

    service.build_and_submit(
        qa_pairs=simple_qa,
        domain="ml",
        slug="vae-notify",
        template="brief",
        sync=False,
        model="gemini-3.1-flash-lite-preview",
        console=console,
    )

    # Wait for daemon thread to complete notification
    done_event.wait(timeout=5)
    assert len(osascript_calls) > 0
    assert any("osascript" in c[0] for c in osascript_calls)


# ---------------------------------------------------------------------------
# Test 5: suggest_bridge returns immediately when qa_pairs < 4
# ---------------------------------------------------------------------------

def test_suggest_bridge_below_threshold_does_nothing(service, console, monkeypatch):
    """suggest_bridge returns without calling Flash Lite when qa_pairs < 4."""
    flash_calls = []

    def fake_get_client():
        client = MagicMock()
        client.is_available.return_value = True
        client.generate_with_model.side_effect = lambda *a, **kw: flash_calls.append(a) or "{}"
        return client

    monkeypatch.setattr("pb.llm.gemini.get_client", fake_get_client)

    # Only 1 Q&A pair — below threshold
    service.suggest_bridge(
        qa_pairs=[("Q1", "A1")],
        domain="ml",
        console=console,
    )

    # Flash Lite must NOT have been called
    assert len(flash_calls) == 0


# ---------------------------------------------------------------------------
# Test 6: suggest_bridge invokes bridge logic when qa_pairs >= 4
# ---------------------------------------------------------------------------

def test_suggest_bridge_above_threshold_calls_flash_lite(service, four_qa, console, monkeypatch):
    """suggest_bridge calls Flash Lite when qa_pairs >= 4."""
    flash_calls = []

    fake_client = MagicMock()
    fake_client.is_available.return_value = True
    fake_client.generate_with_model.side_effect = lambda *a, **kw: (
        flash_calls.append(a) or '{"cross_domain": false}'
    )
    monkeypatch.setattr("pb.llm.gemini.get_client", lambda: fake_client)

    service.suggest_bridge(
        qa_pairs=four_qa,
        domain="ml",
        console=console,
    )

    # Flash Lite was called
    assert len(flash_calls) > 0


# ---------------------------------------------------------------------------
# Test 7: template param is forwarded verbatim to build_socratic_note
# ---------------------------------------------------------------------------

def test_build_and_submit_template_param_forwarded(service, simple_qa, console, monkeypatch):
    """template param is forwarded verbatim to build_socratic_note."""
    captured_templates = []

    def fake_build_note(qa_pairs, domain, slug, wikilinks, template="brief"):
        captured_templates.append(template)
        return "---\nsource: socratic\n---\n\nbody"

    _patch_sync_deps(monkeypatch)
    monkeypatch.setattr("pb.vault.socratic.build_socratic_note", fake_build_note)

    service.build_and_submit(
        qa_pairs=simple_qa,
        domain="ml",
        slug="vae-deep",
        template="deep",
        sync=True,
        model="gemini-3.1-flash-lite-preview",
        console=console,
    )

    assert "deep" in captured_templates

    captured_templates.clear()

    service.build_and_submit(
        qa_pairs=simple_qa,
        domain="ml",
        slug="vae-brief",
        template="brief",
        sync=True,
        model="gemini-3.1-flash-lite-preview",
        console=console,
    )

    assert "brief" in captured_templates
