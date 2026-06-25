# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Backward-compatibility shim. Import from pb.core.exceptions instead."""
from pb.core.exceptions import *  # noqa: F401, F403
from pb.core.exceptions import (
    ConfigError,
    ConflictError,
    DatabaseError,
    ExitCode,
    NotFoundError,
    PbSystemError,
    UserError,
    ValidationError,
)
