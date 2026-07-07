# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pb.core.enums import SkillKind
from pb.core.session_blueprints import (
    pack_display_label,
    resolve_learning_session_blueprint,
    user_pack_dir,
)
from pb.storage.config import Config, GeneralConfig, LLMConfig, ModelRolesConfig, ProviderConfig


def test_resolve_math_problem_solving_pack():
    resolution = resolve_learning_session_blueprint(
        branch="study",
        domain="mathematics",
        topic="surface parameterization",
    )

    assert resolution.pack_id == "math.problem_solving"
    assert resolution.source == "exact"
    assert resolution.blueprint.skill_kind == SkillKind.PROCEDURAL_COGNITIVE


def test_resolve_cardistry_grips_pack():
    resolution = resolve_learning_session_blueprint(
        branch="practise",
        domain="cardistry",
        topic="Biddle grip",
        drill="static hold",
    )

    assert resolution.pack_id == "cardistry.grips"
    assert resolution.source == "exact"
    assert resolution.blueprint.skill_kind == SkillKind.PROCEDURAL_MOTOR


def test_resolve_programming_debugging_pack():
    resolution = resolve_learning_session_blueprint(
        branch="practise",
        domain="Rust",
        topic="debug a failing borrow-checker test",
        drill="capture the compiler trace",
    )

    assert resolution.pack_id == "programming.debugging"
    assert resolution.source == "exact"
    assert resolution.blueprint.skill_kind == SkillKind.ENGINEERING_BUILD_DEBUG


def test_unknown_motor_skill_uses_generic_pack_without_vault():
    resolution = resolve_learning_session_blueprint(
        branch="practise",
        domain="snowboarding",
        topic="toe-side carve pressure distribution",
        drill="slow edge hold",
        allow_custom_init=False,
    )

    assert resolution.pack_id == "generic.procedural_motor"
    assert resolution.source == "generic"
    assert resolution.blueprint.skill_kind == SkillKind.PROCEDURAL_MOTOR


def test_unknown_motor_skill_auto_creates_custom_pack(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    resolution = resolve_learning_session_blueprint(
        branch="practise",
        domain="snowboarding",
        topic="toe-side carve pressure distribution",
        drill="slow edge hold",
        vault_path=vault,
        allow_custom_init=True,
    )

    custom_dir = user_pack_dir(vault)

    assert resolution.source == "custom"
    assert resolution.pack_id.startswith("custom.")
    assert resolution.blueprint.skill_kind == SkillKind.PROCEDURAL_MOTOR
    assert custom_dir is not None
    assert (custom_dir / f"{resolution.pack_id}.yaml").exists()


def test_gemini_fast_roles_default_to_flash_lite_when_unset():
    config = Config(
        general=GeneralConfig(active_vault="main", vault_path="/tmp/vault"),
        vaults={"main": {"path": "/tmp/vault"}},
        providers={
            "gemini": ProviderConfig(
                api_key_env="GEMINI_API_KEY",
                default_model="gemini-3-flash-preview",
            )
        },
        llm=LLMConfig(provider="gemini", default_model="gemini-3-flash-preview"),
        model_roles=ModelRolesConfig(default="gemini:gemini-3-flash-preview"),
    )

    assert config.model_roles.fast == "gemini:gemini-3.1-flash-lite-preview"
    assert config.model_roles.fast_inference == "gemini:gemini-3.1-flash-lite-preview"


def test_pack_display_label_is_human_friendly():
    assert pack_display_label("generic.foo_bar.baz") == "Generic Foo Bar - Baz"
