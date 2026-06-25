# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Access bundled package resources without relying on checkout-relative paths."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Iterator


def package_root() -> Traversable:
    """Return the installed ``pb`` package root."""

    return resources.files("pb")


def resource(*parts: str) -> Traversable:
    """Return one resource under the installed ``pb`` package."""

    item = package_root()
    for part in parts:
        item = item.joinpath(part)
    return item


def template_resource(name: str) -> Traversable:
    """Return a bundled Markdown template resource."""

    return resource("templates", name)


def template_exists(name: str) -> bool:
    """Return whether a bundled Markdown template exists."""

    return template_resource(name).is_file()


def read_template_text(name: str) -> str:
    """Read a bundled Markdown template as UTF-8 text."""

    return template_resource(name).read_text(encoding="utf-8")


def iter_domain_pack_resources() -> Iterator[Traversable]:
    """Yield bundled domain-pack YAML resources."""

    packs = resource("domain_packs")
    if not packs.is_dir():
        return
    for item in sorted(packs.iterdir(), key=lambda value: value.name):
        if item.is_file() and item.name.endswith(".yaml"):
            yield item


@contextmanager
def resource_path(*parts: str) -> Iterator[Path]:
    """Yield a real filesystem path for a bundled resource."""

    with resources.as_file(resource(*parts)) as path:
        yield path
