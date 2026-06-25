# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Exception hierarchy with exit codes per D-07, D-08.

Exit codes follow HTTP-style categories:
- 0 = success (2xx equivalent)
- 4x = user/input errors (4xx equivalent)
- 5x = system/internal errors (5xx equivalent)

Avoids reserved codes 1-2 (Typer/Click) and 126-255 (shell).
"""


class ExitCode:
    """
    Exit codes following HTTP-style categories per D-08.

    0 = success (2xx)
    4x = user/input errors (4xx)
    5x = system/internal errors (5xx)
    """

    SUCCESS = 0

    # User errors (4x) - caller can fix by changing input
    BAD_INPUT = 40  # Invalid argument format
    VALIDATION = 42  # Business rule violation
    NOT_FOUND = 44  # Entity not found
    CONFLICT = 49  # State conflict (e.g., task already active)

    # System errors (5x) - internal failure, caller cannot fix
    INTERNAL = 50  # Generic internal error
    IO_ERROR = 51  # File system error
    DB_ERROR = 52  # Database error
    CONFIG_ERROR = 53  # Configuration error


class UserError(Exception):
    """Base class for user-caused errors (exit code 4x)."""

    exit_code: int = ExitCode.BAD_INPUT


class NotFoundError(UserError):
    """Entity not found."""

    exit_code = ExitCode.NOT_FOUND


class ValidationError(UserError):
    """Business rule violation."""

    exit_code = ExitCode.VALIDATION


class ConflictError(UserError):
    """State conflict."""

    exit_code = ExitCode.CONFLICT


class PbSystemError(Exception):
    """Base class for system errors (exit code 5x)."""

    exit_code: int = ExitCode.INTERNAL


class DatabaseError(PbSystemError):
    """Database operation failed."""

    exit_code = ExitCode.DB_ERROR


class ConfigError(PbSystemError):
    """Configuration error."""

    exit_code = ExitCode.CONFIG_ERROR
