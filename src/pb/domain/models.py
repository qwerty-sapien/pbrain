# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Backward-compatibility shim. Import from pb.core.models instead."""
from pb.core.models import *  # noqa: F401, F403
from pb.core.models import (
    Clip,
    DailyDebrief,
    DailyReviewResponse,
    Domain,
    Goal,
    GoalArc,
    Note,
    Packet,
    Project,
    Session,
    Task,
    TimeBlock,
    Track,
    generate_internal_id,
    generate_slug,
    utc_now,
)
