# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

def test_tasks_app_invokes_without_command():
    from pb.cli.commands.tasks import app
    assert app.info.invoke_without_command is True
