# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Review and reporting service.

Stub service for Phase 21 -- method bodies raise NotImplementedError.
Real implementations migrated from pb.core.review_engine,
pb.core.review_log_writer, pb.core.reports, pb.core.insights,
pb.core.screen_time in later phases.
"""
from __future__ import annotations
from datetime import date
from typing import Optional
from pb.core.models import DailyDebrief, DailyReviewResponse, Session, Task
from pb.core.base import BaseService


class ReviewService(BaseService):
    """Generates daily/weekly reviews comparing plan vs actual.

    Composes with SessionService for session data and GoalsService
    for alignment checks.
    """
    def __init__(self, session_service: Optional[object] = None,
                 goals_service: Optional[object] = None,
                 repo: Optional[object] = None):
        super().__init__()
        self.session_service = session_service
        self.goals_service = goals_service
        self.repo = repo

    def daily_review(self, review_date: date | None = None) -> DailyDebrief:
        raise NotImplementedError

    def weekly_review(self, week_offset: int = 0) -> dict:
        raise NotImplementedError

    def plan_vs_actual(self, review_date: date | None = None) -> dict:
        raise NotImplementedError

    def get_sessions(self, review_date: date | None = None) -> list[Session]:
        raise NotImplementedError

    def get_insights(self, days: int = 7) -> list[str]:
        raise NotImplementedError

    def generate_report(self, review_date: date | None = None) -> DailyReviewResponse:
        raise NotImplementedError
