# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pb.storage.database as db_module
from pb.core.context_file_intake import SourceBundle, SourceBundleItem
from pb.mcp.tools.productivebrain import (
    context_file_ingest,
    context_file_status,
    context_lock,
    context_status,
    context_unlock,
    learn_with_context,
    source_bundle_list,
    source_bundle_show,
    tool_catalog,
)
from pb.storage.config import Config, GeneralConfig, LLMConfig, ModelRolesConfig, ProviderConfig, VaultProfileConfig
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository


def setup_function() -> None:
    db_module._db_path = None


def teardown_function() -> None:
    db_module._db_path = None


def _runtime(tmp_path: Path):
    vault_path = tmp_path / "vault"
    data_dir = tmp_path / "data"
    db_path = data_dir / "productivebrain.db"
    vault_path.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    quarantine_path = vault_path / "quarantine"
    quarantine_path.mkdir(parents=True)
    config = Config(
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
    return SimpleNamespace(
        config=config,
        vault_name="main",
        vault_path=vault_path,
        data_dir=data_dir,
        db_path=db_path,
        quarantine_path=quarantine_path,
    )


def test_mcp_context_ingest_status_and_lock_cycle(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    set_db_path(runtime.db_path)
    init_db(runtime.db_path)
    note_path = tmp_path / "lecture_notes.md"
    note_path.write_text("# Lecture notes\nStokes theorem.\n", encoding="utf-8")

    with patch("pb.mcp.tools.productivebrain.get_runtime_context", return_value=runtime), patch(
        "pb.mcp.tools.productivebrain.get_mcp_context",
        return_value=SimpleNamespace(allow_writes=True),
    ):
        ingest = context_file_ingest([str(note_path)])
        assert ingest["sources"]
        source_id = ingest["sources"][0]["id"]

        listed = context_file_status()
        assert listed["sources"]
        assert listed["sources"][0]["filename"] == "lecture_notes.md"

        locked = context_lock(source_id)
        assert locked["locked"] is True
        assert locked["scope"]["mode"] == "direct_files"

        status = context_status()
        assert status["locked"] is True
        assert status["scope"]["mode"] == "direct_files"

        unlocked = context_unlock()
        assert unlocked["locked"] is False


def test_mcp_source_bundle_reads_and_learn_with_context(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    set_db_path(runtime.db_path)
    init_db(runtime.db_path)
    repo = Repository()
    note_path = tmp_path / "worksheet.md"
    note_path.write_text("# Worksheet\nManifold smoothness.\n", encoding="utf-8")

    with patch("pb.mcp.tools.productivebrain.get_runtime_context", return_value=runtime), patch(
        "pb.mcp.tools.productivebrain.get_mcp_context",
        return_value=SimpleNamespace(allow_writes=True),
    ):
        ingest = context_file_ingest([str(note_path)])
        source = ingest["sources"][0]
        bundle = SourceBundle(
            name="MA1002",
            domain_name=str(source.get("domain_name") or ""),
            scope_mode=str(source.get("scope_mode") or "unclear"),
            scope_boundary=str(source.get("scope_boundary") or ""),
            source_refs=[str(source["source_ref"])],
            items=[
                SourceBundleItem(
                    bundle_id="",
                    source_id=str(source["id"]),
                    position=0,
                    source_ref=str(source["source_ref"]),
                    filename=str(source["filename"]),
                )
            ],
        )
        bundle.items = [item.model_copy(update={"bundle_id": bundle.id}) for item in bundle.items]
        repo.create_source_bundle(bundle)

        bundles = source_bundle_list()
        shown = source_bundle_show("MA1002")
        assert bundles["bundles"]
        assert shown["bundle"]["name"] == "MA1002"

        started = learn_with_context("manifold smoothness", [str(note_path)], branch="study")
        assert started["started"] is True
        assert started["context_scope"]["mode"] == "direct_files"
        session = repo.get_session(started["session_id"])
        assert session is not None
        assert "active_context_scope" in dict(getattr(session, "generated_names", {}) or {})


def test_tool_catalog_exposes_context_and_learning_surface() -> None:
    catalog = tool_catalog()
    names = {item["name"]: item["classification"] for item in catalog["tools"]}

    assert names["context_file_ingest"] == "tier_1_write"
    assert names["learn_with_context"] == "tier_1_write"
    assert names["source_bundle_show"] == "read_only"
    assert names["pb_command"] == "debug_escape_hatch"
