# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pathlib import Path

from pb.cli.markdown import resolve_glow_style_path
from pb.core.resources import iter_domain_pack_resources, read_template_text, resource_path
from pb.core.session_blueprints import load_domain_packs


def test_bundled_templates_are_readable_without_repo_relative_paths():
    assert "# Evidence:" in read_template_text("evidence_generic.md")
    assert "# Review:" in read_template_text("review_packet.md")


def test_bundled_domain_packs_are_readable_without_repo_relative_paths():
    pack_names = {item.name for item in iter_domain_pack_resources()}
    assert "generic.conceptual.yaml" in pack_names
    assert "programming.debugging.yaml" in pack_names
    assert "generic.conceptual" in load_domain_packs()


def test_bundled_glow_style_has_real_path():
    style = resolve_glow_style_path()
    assert style is not None
    assert Path(style).name == "glow_style.json"


def test_resource_path_context_manager_exposes_existing_file():
    with resource_path("templates", "session_log.md") as path:
        assert path.is_file()
