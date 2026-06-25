# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Helpers for invoking pb commands from other pb commands."""

from __future__ import annotations

import shlex


def run_internal_command(ctx, command: str):
    """Invoke another pb command through the root Click command."""
    args = shlex.split(command)
    root = ctx.find_root()
    return root.command.main(
        args=args,
        prog_name=root.info_name or "pb",
        obj=root.obj,
        standalone_mode=False,
    )
