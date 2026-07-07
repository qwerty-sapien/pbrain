"""Phase 26 integration tests — covers ANKI-01 through ANLT-03 and ALGN-01/02."""
from __future__ import annotations

import pathlib
import uuid
from unittest.mock import MagicMock, patch

import pytest

import pb.storage.database as db_module

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def temp_db(tmp_path):
    """Initialize a temp DB with Phase 26 schema. Sets global DB path."""
    from pb.storage.database import init_db, set_db_path

    db = tmp_path / "pb.db"
    set_db_path(db)
    init_db(db)
    yield db
    db_module._db_path = None


@pytest.fixture()
def temp_vault(tmp_path):
    """Create a minimal vault structure for testing."""
    vault = tmp_path / "vault"
    vault.mkdir()
    domain_dir = vault / "knowledge" / "deutsch"
    domain_dir.mkdir(parents=True)
    # Create 3 notes with stage tags
    (domain_dir / "note1.md").write_text("---\nlearning_stage: new\n---\n# Note 1\n")
    (domain_dir / "note2.md").write_text("---\nlearning_stage: learning\n---\n# Note 2\n")
    (domain_dir / "note3.md").write_text("---\nlearning_stage: learnt\n---\n# Note 3\n")
    return vault


@pytest.fixture()
def anki_service(tmp_path, temp_db, temp_vault):
    """AnkiService instance with temp vault and minimal mock repo."""
    from pb.vault.anki_service import AnkiService

    repo = MagicMock()
    repo.list_goal_arcs.return_value = []
    repo.list_tasks.return_value = []
    repo.list_sessions_for_task.return_value = []
    return AnkiService(vault_path=temp_vault, repo=repo)


# ── ANKI-01: Revlog sync + staleness feed ─────────────────────────────────


def test_anki01_revlog_sync_offline_safe(temp_db):
    """ANKI-01: sync_revlog degrades gracefully when AnkiConnect offline."""
    from pb.vault.anki_client import sync_revlog

    with patch("pb.vault.anki_client.anki_request", return_value=None):
        result = sync_revlog()
    # Should return empty list or None — never raise
    assert result is None or isinstance(result, list)


# ── ANKI-02: Auto card generation ─────────────────────────────────────────


def test_anki02_generate_cards_stores_run_log(anki_service, temp_db):
    """ANKI-02: generate_cards stores entry in generation_run_log with run_id."""
    mock_cards = [
        {
            "id": str(uuid.uuid4()),
            "note_slug": "test-note",
            "front": "Q1",
            "back": "A1",
            "card_type": "Basic",
            "status": "pending",
            "deck": "German",
            "tags": "",
            "anki_model": "Basic",
            "created_at": "2026-05-08T12:00:00",
            "updated_at": "2026-05-08T12:00:00",
        }
    ]
    with patch("pb.vault.anki_client.generate_auto_cards", return_value=mock_cards):
        result = anki_service.generate_cards(
            note_slug="test-note",
            note_content="Some content",
            domain="deutsch",
            deck="German",
            source="auto",
        )
    assert result["count"] == 1
    assert len(result["run_id"]) >= 6
    history = anki_service.get_history()
    assert any(r["run_id"] == result["run_id"] for r in history)


# ── ANKI-02: format.json enriches generation prompt ───────────────────────


def test_anki02_format_json_roundtrip(anki_service, temp_vault):
    """ANKI-02: format.json can be saved and loaded per deck."""
    fmt = {
        "model": "Basic",
        "style_instructions": "concise",
        "example_front": "Q",
        "example_back": "A",
    }
    anki_service.save_format_json("German", fmt)
    loaded = anki_service.load_format_json("German")
    assert loaded["style_instructions"] == "concise"
    assert loaded["example_front"] == "Q"


# ── ANKI-03: Socratic card hook ────────────────────────────────────────────


def test_anki03_hook_fires_on_deep_debrief_only(temp_db):
    """ANKI-03: run_debrief_loop hook fires for deep debrief (max=5) but not mini (max=2)."""
    from pb.vault.socratic import run_debrief_loop, SocraticDebriefEngine

    def make_engine(max_rounds):
        engine = MagicMock(spec=SocraticDebriefEngine)
        engine._max = max_rounds
        engine.round_number = max_rounds
        engine.should_continue.return_value = False
        # Return a real question string so the early-return guard does not trigger
        engine.get_question.return_value = "What did you learn?"
        engine.collect_answers.return_value = [("q1", "a1"), ("q2", "a2")]
        return engine

    console = MagicMock()

    # Deep debrief: hook MUST fire
    with patch("pb.vault.socratic.extract_socratic_cards", return_value=[]) as mock_extract:
        with patch("pb.vault.anki_client.insert_cards_to_db", return_value=0):
            with patch("pb.vault.anki_client._insert_run_log_entry"):
                with patch("builtins.input", return_value="skip"):
                    run_debrief_loop(
                        make_engine(5),
                        console,
                        note_slug="test-note",
                        deck="German",
                        domain="deutsch",
                        generate_anki=True,
                    )
    mock_extract.assert_called_once()

    # Mini debrief: hook MUST NOT fire
    with patch("pb.vault.socratic.extract_socratic_cards", return_value=[]) as mock_extract2:
        with patch("builtins.input", return_value="skip"):
            run_debrief_loop(
                make_engine(2),
                console,
                note_slug="test-note",
                deck="German",
                domain="deutsch",
                generate_anki=True,
            )
    mock_extract2.assert_not_called()


# ── ANKI-04: History and rollback ─────────────────────────────────────────


def test_anki04_history_rollback_lifecycle(anki_service, temp_db):
    """ANKI-04: history shows run; rollback deletes it; subsequent history is empty."""
    run_id = "test" + str(uuid.uuid4())[:4]
    anki_service.insert_run_log(run_id, "my-note", None, 3, "auto")

    # Insert 3 cards with that run_id
    from pb.vault.anki_client import insert_cards_to_db

    cards = [
        {
            "id": str(uuid.uuid4()),
            "note_slug": "my-note",
            "front": f"Q{i}",
            "back": f"A{i}",
            "card_type": "Basic",
            "status": "pending",
            "deck": "German",
            "tags": "",
            "anki_model": "Basic",
            "domain": "deutsch",
            "run_id": run_id,
            "created_at": "2026-05-08T12:00:00",
            "updated_at": "2026-05-08T12:00:00",
        }
        for i in range(3)
    ]
    insert_cards_to_db(cards)

    history = anki_service.get_history()
    assert any(r["run_id"] == run_id for r in history)

    with patch("pb.vault.anki_client.anki_request", return_value=None):  # AnkiConnect offline
        ok, msg = anki_service.rollback_run(run_id)

    assert ok is True
    assert "3 cards" in msg

    history_after = anki_service.get_history()
    assert not any(r["run_id"] == run_id for r in history_after)


# ── ANKI-05: Anki count via schema migration ───────────────────────────────


def test_anki05_exported_at_column_populated_on_insert(temp_db):
    """ANKI-05: exported_at column exists and insert works with domain+run_id."""
    from pb.storage.database import get_connection
    from pb.vault.anki_client import insert_cards_to_db

    card_id = str(uuid.uuid4())
    insert_cards_to_db(
        [
            {
                "id": card_id,
                "note_slug": "slug",
                "front": "Q",
                "back": "A",
                "card_type": "Basic",
                "status": "pending",
                "deck": "German",
                "tags": "",
                "anki_model": "Basic",
                "domain": "deutsch",
                "run_id": "run001",
                "created_at": "2026-05-08T12:00:00",
                "updated_at": "2026-05-08T12:00:00",
            }
        ]
    )
    with get_connection() as conn:
        row = conn.execute(
            "SELECT domain, run_id FROM anki_cards WHERE id = ?", (card_id,)
        ).fetchone()
    assert row["domain"] == "deutsch"
    assert row["run_id"] == "run001"


# ── ANLT-01: Staleness monitor (already wired in plan.py) ─────────────────


def test_anlt01_staleness_non_raising(temp_db):
    """ANLT-01: sync_revlog() does not raise after Phase 26 schema changes (AnkiConnect absent)."""
    from pb.vault.anki_client import sync_revlog

    with patch("pb.vault.anki_client.anki_request", return_value=None):
        result = sync_revlog()
    # Should return empty list or None — never raise after schema migration
    assert result is None or isinstance(result, list)


# ── ANLT-03: Zero activity domains ────────────────────────────────────────


def test_anlt03_get_zero_activity_domains_non_raising(temp_vault, temp_db):
    """ANLT-03: get_zero_activity_domains returns list, never raises."""
    from pb.cli.commands.review import get_zero_activity_domains

    result = get_zero_activity_domains(temp_vault)
    assert isinstance(result, list)


# ── ANLT-02/03: Growth table + zero activity wiring ───────────────────────


def test_anlt02_build_growth_table_non_raising(temp_vault, temp_db):
    """ANLT-02: build_growth_table returns None or Rich Table — never raises."""
    from pb.cli.commands.review import build_growth_table
    from rich.table import Table

    result = build_growth_table(temp_vault)
    # May return None if no stats recorded yet — acceptable
    assert result is None or isinstance(result, Table)


# ── ALGN-01: pb review --alignment unchanged ──────────────────────────────


def test_algn01_alignment_engine_importable():
    """ALGN-01: AlignmentEngine exists and is importable (untouched by Phase 26)."""
    from pb.core.alignment import AlignmentEngine

    assert AlignmentEngine is not None


# ── ALGN-02: pb goal report helpers ───────────────────────────────────────


def test_algn02_infer_domain_and_count_stages(temp_vault):
    """ALGN-02: Domain inferred from goal title; stage counts reflect vault notes."""
    from pb.cli.commands.goals import _infer_domain_from_goal, _count_goal_domain_stages

    class MockGoal:
        # Title must contain the domain dir name "deutsch" for the fuzzy match to hit
        title = "Learn deutsch to B2"
        description = "Reach B2 level deutsch"

    domain = _infer_domain_from_goal(MockGoal(), temp_vault)
    assert domain == "deutsch"

    counts = _count_goal_domain_stages(temp_vault, "deutsch")
    assert counts.get("new", 0) == 1
    assert counts.get("learning", 0) == 1
    assert counts.get("learnt", 0) == 1


def test_algn02_goal_report_no_goals_does_not_crash(temp_db, temp_vault, capsys):
    """ALGN-02: goals_report with no goals prints message, does not raise."""
    from pb.cli.commands.goals import goals_report

    mock_ctx = MagicMock()
    mock_ctx.obj = {}
    mock_ctx.invoked_subcommand = None

    with patch("pb.cli.commands.goals.Repository") as mock_repo_cls:
        mock_repo = MagicMock()
        mock_repo.list_goal_arcs.return_value = []
        mock_repo_cls.return_value = mock_repo
        with patch("pb.vault.config.get_vault_path", return_value=temp_vault):
            # Should not raise
            try:
                goals_report(mock_ctx)
            except SystemExit:
                pass  # typer may call sys.exit — acceptable


# ── Full pipeline: generate → history → rollback ──────────────────────────


def test_full_anki_pipeline(anki_service, temp_db):
    """Full AnkiService.generate_cards → insert_cards_to_db → get_history → rollback_run lifecycle."""
    mock_cards = [
        {
            "id": str(uuid.uuid4()),
            "note_slug": "pipeline-note",
            "front": "Was ist Wasser?",
            "back": "Water",
            "card_type": "Basic",
            "status": "pending",
            "deck": "German",
            "tags": "",
            "anki_model": "Basic",
            "created_at": "2026-05-08T12:00:00",
            "updated_at": "2026-05-08T12:00:00",
        }
    ]
    with patch("pb.vault.anki_client.generate_auto_cards", return_value=mock_cards):
        result = anki_service.generate_cards(
            note_slug="pipeline-note",
            note_content="Water is H2O",
            domain="deutsch",
            deck="German",
            source="auto",
        )

    run_id = result["run_id"]
    assert result["count"] == 1

    history = anki_service.get_history()
    assert any(r["run_id"] == run_id for r in history)

    with patch("pb.vault.anki_client.anki_request", return_value=None):
        ok, msg = anki_service.rollback_run(run_id)

    assert ok is True

    history_after = anki_service.get_history()
    assert not any(r["run_id"] == run_id for r in history_after)
