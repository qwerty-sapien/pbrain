# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Backward-compatibility shim. Import from pb.core.rules instead."""
from pb.core.rules import *  # noqa: F401, F403
from pb.core.rules import (
    RuleViolation,
    validate_no_learning_without_socratic,
    validate_project_has_packet,
)
