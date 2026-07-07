# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

import pb.storage.database as db_module
import pb.cli.commands.context as context_mod
from pb.cli.commands.context import app as context_app
from pb.cli.commands.study import app as study_app
from pb.core.context_file_intake import SourceBundle, SourceBundleItem, active_context_from_sources
from pb.domain.enums import SessionMode, TaskState
from pb.domain.models import Session, Task
from pb.storage.config import Config, GeneralConfig, LLMConfig, ModelRolesConfig, ProviderConfig, VaultProfileConfig
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository


def setup_function() -> None:
    db_module._db_path = None


def teardown_function() -> None:
    db_module._db_path = None


def _config(vault_path: Path, data_dir: Path) -> Config:
    return Config(
        general=GeneralConfig(active_vault="main"),
        vaults={
            "main": VaultProfileConfig(
                path=str(vault_path),
                data_dir=str(data_dir),
            )
        },
        providers={
            "gemini": ProviderConfig(
                api_key_env="GEMINI_API_KEY",
                default_model="gemini-3-flash-preview",
            )
        },
        model_roles=ModelRolesConfig(default="gemini:gemini-3-flash-preview"),
        llm=LLMConfig(provider="gemini", default_model="gemini-3-flash-preview"),
    )


def _ctx_obj(tmp_path: Path) -> dict[str, object]:
    vault_path = tmp_path / "vault"
    data_dir = tmp_path / "data"
    vault_path.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    db_path = data_dir / "productivebrain.db"
    set_db_path(db_path)
    init_db(db_path)
    config = _config(vault_path, data_dir)
    runtime = SimpleNamespace(
        vault_path=vault_path,
        data_dir=data_dir,
        db_path=db_path,
        config=config,
    )
    return {
        "repo": Repository(),
        "runtime": runtime,
        "config": config,
        "factory": {},
        "yes": False,
        "verbose": False,
        "dryrun": False,
    }


def test_context_add_bundle_lock_and_status(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)
    notes = tmp_path / "lecture_notes.md"
    notes.write_text("# Lecture notes\nStokes theorem.", encoding="utf-8")

    runner = CliRunner()

    added = runner.invoke(context_app, ["add", str(notes)], obj=obj)
    assert added.exit_code == 0, added.output
    assert "Stored source:" in added.stdout
    assert (obj["runtime"].vault_path / "source" / "lecture_notes.md").exists()
    assert "vault://source/lecture_notes.md" in added.stdout

    listed = runner.invoke(context_app, ["list"], obj=obj)
    assert listed.exit_code == 0
    assert "lecture_notes.md" in listed.stdout
    assert "vault://source/lecture_notes.md" in listed.stdout

    created = runner.invoke(context_app, ["bundle", "create", "MA1002", str(notes)], obj=obj)
    assert created.exit_code == 0, created.output
    assert "Created bundle `MA1002`" in created.stdout

    locked = runner.invoke(context_app, ["lock", "MA1002"], obj=obj)
    assert locked.exit_code == 0, locked.output
    assert "Locked context:" in locked.stdout

    status = runner.invoke(context_app, ["status"], obj=obj)
    assert status.exit_code == 0
    assert "Mode: bundle" in status.stdout
    assert "Scope mode:" in status.stdout


def test_context_defaults_lock_guard_and_remove(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First\nGreen functions.", encoding="utf-8")
    second.write_text("# Second\nBoundary conditions.", encoding="utf-8")

    runner = CliRunner()

    added = runner.invoke(context_app, ["add", str(first)], obj=obj)
    assert added.exit_code == 0, added.output

    shown = runner.invoke(context_app, ["show"], obj=obj)
    assert shown.exit_code == 0, shown.output
    assert "Stored context sources:" in shown.stdout
    assert "first.md" in shown.stdout

    locked = runner.invoke(context_app, ["lock", "first"], obj=obj)
    assert locked.exit_code == 0, locked.output
    assert "Locked context:" in locked.stdout

    blocked = runner.invoke(context_app, ["add", str(second)], obj=obj)
    assert blocked.exit_code == 1
    assert "Context is locked:" in blocked.stdout
    assert "pb context unlock" in blocked.stdout

    forced = runner.invoke(context_app, ["add", str(second), "--force"], obj=obj)
    assert forced.exit_code == 0, forced.output
    assert "Stored source:" in forced.stdout

    removed = runner.invoke(context_app, ["remove", "first.md"], obj=obj)
    assert removed.exit_code == 0, removed.output
    assert "Context lock cleared" in removed.stdout
    assert "Removed source: first.md" in removed.stdout


def test_context_unlock_pauses_active_session_before_clearing_lock(tmp_path: Path, monkeypatch) -> None:
    obj = _ctx_obj(tmp_path)
    notes = tmp_path / "active.md"
    notes.write_text("# Active\nContext scope.", encoding="utf-8")

    runner = CliRunner()
    added = runner.invoke(context_app, ["add", str(notes)], obj=obj)
    assert added.exit_code == 0, added.output
    locked = runner.invoke(context_app, ["lock", "active.md"], obj=obj)
    assert locked.exit_code == 0, locked.output

    repo = obj["repo"]
    task = repo.create_task(Task(title="Study active scope", state=TaskState.ACTIVE))
    repo.create_session(Session(task_id=task.id, mode=SessionMode.FOCUS))

    monkeypatch.setattr(context_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(context_mod, "pick_single_choice", lambda *args, **kwargs: "pause")

    result = runner.invoke(context_app, ["unlock"], obj=obj)

    assert result.exit_code == 0, result.output
    assert "Paused session: Study active scope" in result.stdout
    assert "Context unlocked." in result.stdout
    assert repo.get_locked_context() is None
    assert repo.get_active_session() is None


def test_context_unlock_can_finish_active_session_before_clearing_lock(tmp_path: Path, monkeypatch) -> None:
    obj = _ctx_obj(tmp_path)
    notes = tmp_path / "finish.md"
    notes.write_text("# Finish\nContext scope.", encoding="utf-8")

    runner = CliRunner()
    added = runner.invoke(context_app, ["add", str(notes)], obj=obj)
    assert added.exit_code == 0, added.output
    locked = runner.invoke(context_app, ["lock", "finish.md"], obj=obj)
    assert locked.exit_code == 0, locked.output

    repo = obj["repo"]
    task = repo.create_task(Task(title="Finish active scope", state=TaskState.ACTIVE))
    repo.create_session(Session(task_id=task.id, mode=SessionMode.FOCUS))

    monkeypatch.setattr(context_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(context_mod, "pick_single_choice", lambda *args, **kwargs: "finish")

    result = runner.invoke(context_app, ["unlock"], obj=obj)

    assert result.exit_code == 0, result.output
    assert "Finished session: Finish active scope" in result.stdout
    assert "Context unlocked." in result.stdout
    assert repo.get_locked_context() is None
    assert repo.get_active_session() is None
    assert repo.get_task(task.id).completion == 100


def test_context_empty_bundle_absolute_remove_and_doctor_status(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)
    notes = tmp_path / "module.md"
    notes.write_text("# Module\nEigenvalues.", encoding="utf-8")

    runner = CliRunner()

    created = runner.invoke(context_app, ["bundle", "create", "MA1002"], obj=obj)
    assert created.exit_code == 0, created.output
    assert "Created bundle `MA1002` with 0 source(s)." in created.stdout

    added = runner.invoke(context_app, ["bundle", "add", "MA1002", str(notes)], obj=obj)
    assert added.exit_code == 0, added.output
    assert "Added 1 source(s)" in added.stdout

    removed = runner.invoke(context_app, ["bundle", "remove", "MA1002", str(notes.resolve())], obj=obj)
    assert removed.exit_code == 0, removed.output
    assert "Removed 1 source(s)" in removed.stdout

    doctor = runner.invoke(context_app, ["doctor"], obj=obj)
    assert doctor.exit_code == 0, doctor.output
    assert "Context doctor" in doctor.stdout
    assert "Bundles: 1" in doctor.stdout


def test_context_infer_guides_to_learning_command(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)
    notes = tmp_path / "module.md"
    notes.write_text("# Module\nEigenvalues.", encoding="utf-8")

    runner = CliRunner()
    added = runner.invoke(context_app, ["add", str(notes)], obj=obj)
    assert added.exit_code == 0, added.output

    inferred = runner.invoke(context_app, ["infer", "explain", "the", "core", "idea"], obj=obj)

    assert inferred.exit_code == 0, inferred.output
    assert "Use context through a learning command" in inferred.stdout
    assert "pb learn" in inferred.stdout
    assert "pb context lock" in inferred.stdout
    assert "--context" not in inferred.stdout


def test_study_context_parser_strips_files_from_topic_and_inherits_locked_scope(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)
    notes = tmp_path / "worksheet.md"
    notes.write_text("# Worksheet\nManifold smoothness.", encoding="utf-8")

    runner = CliRunner()
    created = runner.invoke(context_app, ["bundle", "create", "MA1002", str(notes)], obj=obj)
    assert created.exit_code == 0, created.output
    locked = runner.invoke(context_app, ["lock", "MA1002"], obj=obj)
    assert locked.exit_code == 0, locked.output

    with patch("pb.cli.commands.study.launch_study_session") as mocked:
        result = runner.invoke(
            study_app,
            ["manifold", "smoothness", "--context", str(notes)],
            obj=obj,
        )
    assert result.exit_code == 0, result.output
    assert mocked.call_args.kwargs["topic"] == "manifold smoothness"
    prepared = obj.get("_prepared_context_scope")
    assert prepared is not None
    assert prepared.scope is not None
    assert prepared.scope.mode == "direct_files"

    obj.pop("_prepared_context_scope", None)
    with patch("pb.cli.commands.study.launch_study_session") as mocked_locked:
        result_locked = runner.invoke(
            study_app,
            ["manifold", "smoothness"],
            obj=obj,
        )
    assert result_locked.exit_code == 0, result_locked.output
    prepared_locked = obj.get("_prepared_context_scope")
    assert prepared_locked is not None
    assert prepared_locked.scope is not None
    assert prepared_locked.scope.locked is True
    assert prepared_locked.scope.mode == "bundle"
    assert mocked_locked.call_args.kwargs["topic"] == "manifold smoothness"


def test_context_add_accepts_multiple_files(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First\nGradient descent.", encoding="utf-8")
    second.write_text("# Second\nLine search.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(context_app, ["add", str(first), str(second)], obj=obj)

    assert result.exit_code == 0, result.output
    assert result.stdout.count("Stored source:") == 2
    rows = obj["repo"].list_context_sources()
    assert {row["filename"] for row in rows} == {"first.md", "second.md"}
    assert all(str(row["source_ref"]).startswith("vault://source/") for row in rows)


def test_context_add_no_args_uses_picker_when_tty(tmp_path: Path, monkeypatch) -> None:
    obj = _ctx_obj(tmp_path)
    picked = tmp_path / "picked.md"
    picked.write_text("# Picked\nStokes theorem.", encoding="utf-8")

    monkeypatch.setattr(context_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(context_mod, "pick_context_files", lambda start_dir: [picked])

    runner = CliRunner()
    result = runner.invoke(context_app, ["add"], obj=obj)

    assert result.exit_code == 0, result.output
    assert "Stored source: picked.md -> vault://source/picked.md" in result.stdout


def test_context_add_invalid_path_uses_picker_when_tty(tmp_path: Path, monkeypatch) -> None:
    obj = _ctx_obj(tmp_path)
    picked = tmp_path / "picked.md"
    picked.write_text("# Picked\nEigenvalues.", encoding="utf-8")

    monkeypatch.setattr(context_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(context_mod, "pick_context_files", lambda start_dir: [picked])

    runner = CliRunner()
    result = runner.invoke(context_app, ["add", str(tmp_path / "missing.pdf")], obj=obj)

    assert result.exit_code == 0, result.output
    assert "Stored source: picked.md -> vault://source/picked.md" in result.stdout
    assert "Error" not in result.stdout


def test_context_add_no_args_non_tty_guides_without_error(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)

    runner = CliRunner()
    result = runner.invoke(context_app, ["add"], obj=obj)

    assert result.exit_code == 0, result.output
    assert "No usable context file was selected." in result.stdout
    assert "Run `pb context add <file>`" in result.stdout


def test_context_add_persists_readable_source_paths(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)
    notes = tmp_path / "readable source.md"
    notes.write_text("# Readable\nBoundary conditions.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(context_app, ["add", str(notes)], obj=obj)

    assert result.exit_code == 0, result.output
    row = obj["repo"].list_context_sources()[0]
    assert row["source_ref"] == "vault://source/readable source.md"
    assert Path(str(row["stored_path"])) == obj["runtime"].vault_path / "source" / "readable source.md"
    assert "/sources/" not in str(row["stored_path"])
    assert "vault://sources/" not in str(row["source_ref"])


def test_context_storage_normalizes_old_source_refs_bundles_and_lock(tmp_path: Path) -> None:
    obj = _ctx_obj(tmp_path)
    runtime = obj["runtime"]
    repo = obj["repo"]
    old_ref = "vault://sources/old-internal-id/Robust_agents_learn_causa.pdf"
    old_dir = runtime.vault_path / "sources" / "old-internal-id"
    old_dir.mkdir(parents=True)
    old_file = old_dir / "original.pdf"
    old_file.write_bytes(b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n%%EOF\n")
    old_meta = old_dir / "ingest-result.json"
    old_meta.write_text("{}", encoding="utf-8")
    repo.create_context_source(
        {
            "id": "old-internal-id",
            "filename": "Robust_agents_learn_causa.pdf",
            "original_path": str(tmp_path / "Robust_agents_learn_causa.pdf"),
            "stored_path": str(old_file),
            "normalized_path": str(old_meta),
            "mime_type": "application/pdf",
            "canonical_class": "document.pdf",
            "source_utility": "unknown",
            "scope_mode": "corpus_first",
            "domain_id": None,
            "domain_name": "general learning",
            "scope_boundary": "Use only the uploaded source material.",
            "source_ref": old_ref,
            "ingest_result": {
                "current_provider": "gemini",
                "current_model": "gemini-3-flash-preview",
                "dryrun": False,
                "status": "failed",
                "scope_mode": "corpus_first",
                "source_utility": "unknown",
                "parsed_files": [],
                "failed_files": [
                    {
                        "filename": "Robust_agents_learn_causa.pdf",
                        "extension": "pdf",
                        "mime_type": "application/pdf",
                        "size_mb": 0.001,
                        "canonical_class": "document.pdf",
                        "failure_stage": "model_could_not_read",
                        "failure_reason_user_safe": "Old failure.",
                    }
                ],
                "domain_resolution": {
                    "status": "proposed_new",
                    "domain_id": None,
                    "domain_name": "general learning",
                    "domain_granularity": "broad",
                    "matched_existing_domains": [],
                    "new_domain_name": "general learning",
                    "new_domain_basis": "uploaded_files",
                    "source_bundle_id": None,
                    "source_bundle_name": None,
                    "scope_boundary": "Use only the uploaded source material.",
                    "requires_user_confirmation": False,
                },
                "scope_clarification": {
                    "needed": False,
                    "reason": "none",
                    "suggested_question": None,
                    "allowed_answers": [],
                },
                "recommended_targets": [],
                "fallback_conversions": [],
            },
        }
    )
    bundle = SourceBundle(
        name="old-bundle",
        domain_name="general learning",
        scope_mode="corpus_first",
        scope_boundary="Use only the uploaded source material.",
        source_refs=[old_ref],
        items=[
            SourceBundleItem(
                bundle_id="",
                source_id="old-internal-id",
                position=0,
                source_ref=old_ref,
                filename="Robust_agents_learn_causa.pdf",
            )
        ],
    )
    bundle.items = [item.model_copy(update={"bundle_id": bundle.id}) for item in bundle.items]
    repo.create_source_bundle(bundle)
    repo.set_locked_context(
        active_context_from_sources(
            [old_ref],
            label="general learning",
            scope_mode="corpus_first",
            scope_boundary="Use only the uploaded source material.",
            locked=True,
        )
    )

    runner = CliRunner()
    result = runner.invoke(context_app, ["list"], obj=obj)

    assert result.exit_code == 0, result.output
    row = repo.get_context_source("old-internal-id")
    assert row is not None
    assert row["source_ref"] == "vault://source/Robust_agents_learn_causa.pdf"
    assert Path(str(row["stored_path"])) == runtime.vault_path / "source" / "Robust_agents_learn_causa.pdf"
    assert row["ingest_result"]["status"] == "ok"
    assert row["ingest_result"]["parsed_files"][0]["isSearchable"] is True
    assert (runtime.vault_path / "source" / "Robust_agents_learn_causa.pdf").exists()
    assert not old_file.exists()
    updated_bundle = repo.get_source_bundle_by_name("old-bundle")
    assert updated_bundle is not None
    assert updated_bundle.source_refs == ["vault://source/Robust_agents_learn_causa.pdf"]
    assert updated_bundle.items[0].source_ref == "vault://source/Robust_agents_learn_causa.pdf"
    locked = repo.get_locked_context()
    assert locked is not None
    assert locked.source_refs == ["vault://source/Robust_agents_learn_causa.pdf"]
