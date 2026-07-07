# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

def test_goal_delete_command_registered():
    from pb.cli.commands.goals import app
    names = [cmd.name for cmd in app.registered_commands]
    assert "delete" in names
