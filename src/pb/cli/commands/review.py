# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Review commands - daily and weekly review."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.markup import escape
from rich.table import Table

from pb.cli.console import get_console, get_err_console
from pb.cli.llm_guard import runtime_for_ctx
from pb.cli.preview import confirm_preview
from pb.core.alignment import AlignmentEngine
from pb.core.models import utc_now
from pb.core.packet_engine import PacketEngine
from pb.core.reports import ReportEngine
from pb.core.review_engine import ReviewEngine
from pb.domain.models import DailyReviewResponse
from pb.llm.drafts import DailyReviewDraft, WeeklyReviewDraft
from pb.llm.runtime import DraftGenerationError
from pb.llm import GeminiClient, score_text_response, generate_followup
from pb.storage.repository import Repository


def _show_people_prompts():
    """Show proactive relationship prompts. Non-fatal: vault errors silently skipped."""
    try:
        from pb.core.prompts import ProactivePromptsEngine

        console = get_console()
        engine = ProactivePromptsEngine()
        prompts = engine.get_prompts()
        if prompts:
            console.rule("[subheader]Relationship Reminders[/]")
            for p in prompts[:5]:
                icons = {"overdue_commitment": "!", "birthday": "*",
                         "gift_reminder": "~", "decay_warning": "?"}
                icon = icons.get(p.prompt_type, "-")
                console.print(f"  [{escape(icon)}] [header]{escape(p.person_name)}[/]: {escape(p.message)}")
            if len(prompts) > 5:
                console.print(f"  [dim]... and {len(prompts) - 5} more (run 'pb prompts' for all)[/]")
            console.print("")
    except Exception:
        pass  # Non-fatal


app = typer.Typer()


def _session_minutes(session, now: datetime) -> int:
    end_at = session.end_at or now
    elapsed = int((end_at - session.start_at).total_seconds() / 60)
    return max(1, elapsed)


def _collect_review_metrics(repo: Repository, *, days: int) -> dict:
    now = utc_now()
    window_end = now
    window_start = now.replace(hour=0, minute=0, second=0, microsecond=0) if days == 1 else now - timedelta(days=days)
    sessions = repo.list_sessions_in_range(window_start, window_end)

    goal_titles: dict[str, str] = {}
    goal_minutes = 0
    study_minutes = 0
    practise_minutes = 0
    practice_evidence = 0
    practice_evidence_debt = 0
    friction_notes: list[str] = []
    observed_errors: list[str] = []
    next_adjustments: list[str] = []
    interruptions = 0
    track_minutes: dict[str, int] = {}

    for session in sessions:
        minutes = _session_minutes(session, now)
        branch = (session.branch or "study").lower()
        if branch == "practice":
            branch = "practise"
        interruptions += int(session.interruption_count or 0)
        if branch == "practise":
            practise_minutes += minutes
            if session.observed_errors or session.next_adjustment or session.evidence_target:
                practice_evidence += 1
            else:
                practice_evidence_debt += 1
            if session.observed_errors:
                observed_errors.append(session.observed_errors)
            if session.next_adjustment:
                next_adjustments.append(session.next_adjustment)
        else:
            study_minutes += minutes

        task = repo.get_task(session.task_id)
        goal_id = session.goal_id
        if not goal_id and task and getattr(task, "linked_goal_arc_ids", None):
            goal_id = task.linked_goal_arc_ids[0]
        if goal_id:
            goal = repo.get_goal_arc(goal_id)
            if goal is not None:
                goal_titles[goal.id] = goal.title
                goal_minutes += minutes
        if task and getattr(task, "linked_track_ids", None):
            for track_id in task.linked_track_ids:
                track = repo.get_track(track_id)
                track_name = track.name if track is not None else track_id
                track_minutes[track_name] = track_minutes.get(track_name, 0) + minutes

        if session.actual_outcome and branch == "study":
            friction_notes.append(session.actual_outcome)

    active_goals = repo.list_goal_arcs(status=None)
    study_stage_gaps = []
    practise_stage_gaps = []
    for goal in active_goals:
        if goal.execution_mode in {"study", "mixed"} and goal.target_bloom_stage:
            current = goal.current_bloom_stage.value if goal.current_bloom_stage else "unset"
            target = goal.target_bloom_stage.value
            if current != target:
                study_stage_gaps.append(f"{goal.title}: {current} -> {target}")
        if goal.execution_mode in {"practise", "practice", "mixed"} and goal.target_practice_stage:
            current = goal.current_practice_stage.value if goal.current_practice_stage else "unset"
            target = goal.target_practice_stage.value
            if current != target:
                practise_stage_gaps.append(f"{goal.title}: {current} -> {target}")

    from pb.vault.anki_client import get_cards_by_status, get_pending_card_count

    suggested_cards = len(get_cards_by_status("suggested"))
    export_ready_cards = get_pending_card_count()
    slippage_titles: list[str] = []
    if days == 1:
        for block in repo.list_time_blocks_for_date(now):
            task = repo.get_task(block.task_id)
            if task is None or task.completion >= 100 or task.archived_at is not None:
                continue
            if task.title not in slippage_titles:
                slippage_titles.append(task.title)

    # Phase 2: Evidence notes and retry queue data
    evidence_notes_today = []
    pending_retries = []
    evidence_trend_summary = ""
    try:
        from pb.storage.database import get_connection
        with get_connection() as conn:
            # Evidence notes for the review window
            if days == 1:
                # "day" review: today's notes
                today_str = now.strftime("%Y-%m-%d")
                rows = conn.execute(
                    "SELECT domain, duration_min, outcome, sub_skills, path "
                    "FROM evidence_notes WHERE date = ? ORDER BY created_at DESC",
                    (today_str,),
                ).fetchall()
            else:
                # "week"/"month" review: window range
                start_str = window_start.strftime("%Y-%m-%d")
                end_str = window_end.strftime("%Y-%m-%d")
                rows = conn.execute(
                    "SELECT domain, duration_min, outcome, sub_skills, path "
                    "FROM evidence_notes WHERE date >= ? AND date <= ? ORDER BY created_at DESC",
                    (start_str, end_str),
                ).fetchall()
            evidence_notes_today = [dict(r) for r in rows]

            # Pending retry items
            today_str = now.strftime("%Y-%m-%d")
            retry_rows = conn.execute(
                "SELECT domain, item_text, priority FROM retry_queue "
                "WHERE status = 'pending' AND (cooldown_until IS NULL OR cooldown_until <= ?) "
                "ORDER BY priority ASC LIMIT 10",
                (today_str,),
            ).fetchall()
            pending_retries = [dict(r) for r in retry_rows]

            # Trend summary: evidence counts by domain over last 7 days (per D-19)
            if days == 1:
                trend_rows = conn.execute(
                    "SELECT domain, COUNT(*) as cnt FROM evidence_notes "
                    "WHERE date >= date(?, '-7 days') GROUP BY domain ORDER BY cnt DESC",
                    (today_str,),
                ).fetchall()
                if trend_rows:
                    parts = [f"{r['domain']}: {r['cnt']} session(s)" for r in trend_rows]
                    evidence_trend_summary = "Last 7 days: " + ", ".join(parts)
    except Exception:
        pass  # Non-fatal

    return {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "study_sessions": sum(1 for session in sessions if (session.branch or "study") == "study"),
        "practise_sessions": sum(
            1 for session in sessions if (session.branch or "study") in {"practise", "practice"}
        ),
        "study_minutes": study_minutes,
        "practise_minutes": practise_minutes,
        "goal_aligned_minutes": goal_minutes,
        "goal_coverage": list(goal_titles.values()),
        "active_goals": [goal.title for goal in active_goals],
        "suggested_anki_cards": suggested_cards,
        "export_ready_anki_cards": export_ready_cards,
        "practice_evidence_captured": practice_evidence,
        "practice_evidence_debt": practice_evidence_debt,
        "interruptions": interruptions,
        "friction_notes": friction_notes[:10],
        "observed_errors": observed_errors[:10],
        "next_adjustments": next_adjustments[:10],
        "study_stage_gaps": study_stage_gaps[:10],
        "practise_stage_gaps": practise_stage_gaps[:10],
        "track_breakdown": track_minutes,
        "slippage_titles": slippage_titles,
        "evidence_notes": evidence_notes_today,
        "pending_retries": pending_retries,
        "evidence_trend_summary": evidence_trend_summary,
    }


def _thought_items(thought_context) -> list[dict]:
    if not thought_context:
        return []
    if isinstance(thought_context, dict):
        return list(thought_context.get("thoughts", []))
    return list(thought_context)


def _build_daily_review_prompt(metrics: dict, thought_context: dict | list[dict] | None = None) -> str:
    parts = [
        "Synthesize a daily learning review from deterministic CLI state.\n"
        "Answer whether today's work moved goals forward.\n"
        "Use only the facts provided. Keep the tone concrete and operational.\n"
        "Highlight study progress, practise evidence, recall/Anki debt, friction, and the next adjustment.\n",
    ]
    thought_items = _thought_items(thought_context)
    if thought_items:
        parts.append(
            "\nThe user captured these thoughts today. Weave them into the reflection — "
            "ask what they meant, connect them to goals, surface patterns:\n"
        )
        for t in thought_items:
            summary = t.get("summary") or t.get("text", "")
            links = ", ".join(t.get("links", []))
            parts.append(f'  - "{summary}"' + (f" (related: {links})" if links else ""))
        parts.append("")
    if isinstance(thought_context, dict):
        threads = thought_context.get("thought_threads", [])
        if threads:
            parts.append("Recurring thought threads:\n")
            for thread in threads:
                parts.append(
                    f"  - {thread.get('label', 'thread')} "
                    f"(members={thread.get('member_count', 0)}, mass={thread.get('weighted_mass', 0)})"
                )
        for key, heading in (
            ("open_questions", "Open questions"),
            ("weak_areas", "Weak areas"),
            ("candidate_focuses", "Candidate focuses"),
        ):
            values = thought_context.get(key, [])
            if values:
                parts.append(f"{heading}:\n")
                parts.extend(f"  - {value}" for value in values)
                parts.append("")
    parts.append(f"{metrics}")
    return "\n".join(parts)


def _build_weekly_review_prompt(metrics: dict, thought_context: dict | list[dict] | None = None) -> str:
    parts = [
        "Synthesize a weekly learning review from deterministic CLI state.\n"
        "Answer what improved, what stalled, what evidence accumulated, and what should change next week.\n"
        "Use only the facts provided. Highlight Bloom/practice-stage gaps, evidence debt, recurring friction, and next-week focus.\n",
    ]
    thought_items = _thought_items(thought_context)
    if thought_items:
        parts.append(
            "\nThe user captured these thoughts this week. Weave them into the reflection — "
            "surface recurring themes, connect to goals, prompt deeper thinking:\n"
        )
        for t in thought_items:
            summary = t.get("summary") or t.get("text", "")
            links = ", ".join(t.get("links", []))
            parts.append(f'  - "{summary}"' + (f" (related: {links})" if links else ""))
        parts.append("")
    if isinstance(thought_context, dict):
        threads = thought_context.get("thought_threads", [])
        if threads:
            parts.append("Recurring thought threads:\n")
            for thread in threads:
                parts.append(
                    f"  - {thread.get('label', 'thread')} "
                    f"(members={thread.get('member_count', 0)}, mass={thread.get('weighted_mass', 0)})"
                )
        for key, heading in (
            ("emergent_interests", "Emergent interests"),
            ("open_questions", "Open questions"),
            ("weak_areas", "Weak areas"),
            ("candidate_focuses", "Candidate focuses"),
        ):
            values = thought_context.get(key, [])
            if values:
                parts.append(f"{heading}:\n")
                parts.extend(f"  - {value}" for value in values)
                parts.append("")
    parts.append(f"{metrics}")
    return "\n".join(parts)


def _append_thought_sections(lines: list[str], thought_context: dict | list[dict] | None) -> None:
    if not isinstance(thought_context, dict):
        return
    threads = thought_context.get("thought_threads", [])
    if threads:
        lines.extend(["", "## Thought Threads"])
        for thread in threads:
            support = ", ".join(thread.get("supporting_thoughts", [])[:2])
            suffix = f" — {support}" if support else ""
            lines.append(
                f"- {thread.get('label', 'thread')} ({thread.get('member_count', 0)} thoughts, "
                f"mass {thread.get('weighted_mass', 0):.2f}){suffix}"
            )
    for key, heading in (
        ("open_questions", "## Open Questions"),
        ("weak_areas", "## Weak Areas"),
        ("emergent_interests", "## Emergent Interests"),
        ("candidate_focuses", "## Candidate Focuses"),
    ):
        values = thought_context.get(key, [])
        if values:
            lines.extend(["", heading])
            lines.extend(f"- {value}" for value in values)


def _render_daily_review(
    metrics: dict,
    draft: DailyReviewDraft,
    thought_context: dict | list[dict] | None = None,
) -> str:
    lines = [
        "# Daily Review",
        "",
        draft.summary,
        "",
        "## Deterministic Facts",
        f"- Goal-aligned time: {metrics['goal_aligned_minutes']} min",
        f"- Study sessions: {metrics['study_sessions']} ({metrics['study_minutes']} min)",
        f"- Practise sessions: {metrics['practise_sessions']} ({metrics['practise_minutes']} min)",
        f"- Suggested Anki candidates: {metrics['suggested_anki_cards']}",
        f"- Accepted/edited Anki candidates awaiting export: {metrics['export_ready_anki_cards']}",
        f"- Practise evidence captured: {metrics['practice_evidence_captured']}",
        f"- Practise evidence missing: {metrics['practice_evidence_debt']}",
        f"- Interruptions: {metrics['interruptions']}",
    ]
    if metrics["track_breakdown"]:
        lines.extend(["", "## Track Breakdown"])
        lines.extend(
            f"- {track}: {minutes} min" for track, minutes in sorted(metrics["track_breakdown"].items())
        )
    if metrics["slippage_titles"]:
        lines.extend(["", "## Slippage"])
        lines.extend(f"- {title}" for title in metrics["slippage_titles"])
    if draft.progress_signals:
        lines.extend(["", "## Progress Signals", *[f"- {item}" for item in draft.progress_signals]])
    if draft.evidence_captured:
        lines.extend(["", "## Evidence", *[f"- {item}" for item in draft.evidence_captured]])
    if draft.friction_patterns:
        lines.extend(["", "## Friction", *[f"- {item}" for item in draft.friction_patterns]])
    if draft.next_adjustments:
        lines.extend(["", "## Next Adjustments", *[f"- {item}" for item in draft.next_adjustments]])

    # Evidence notes section (per D-19)
    if metrics.get("evidence_notes"):
        lines.append("")
        lines.append("## Evidence Notes")
        for note in metrics["evidence_notes"]:
            domain = note.get("domain", "general")
            duration = note.get("duration_min", 0)
            outcome = note.get("outcome", "")
            lines.append(f"- {domain}: {duration} min ({outcome})")

    # Pending retries section (per D-19)
    if metrics.get("pending_retries"):
        lines.append("")
        lines.append("## Pending Retry Items")
        for item in metrics["pending_retries"]:
            domain = item.get("domain", "")
            text = item.get("item_text", "")
            lines.append(f"- [{domain}] {text}")

    # Trend summary (per D-19)
    if metrics.get("evidence_trend_summary"):
        lines.append("")
        lines.append(f"**Trends:** {metrics['evidence_trend_summary']}")

    _append_thought_sections(lines, thought_context)
    return "\n".join(lines)


def _render_weekly_review(
    metrics: dict,
    draft: WeeklyReviewDraft,
    thought_context: dict | list[dict] | None = None,
) -> str:
    lines = [
        "# Weekly Reflection",
        "",
        draft.summary,
        "",
        "## This Week's Numbers",
        f"- Goal-aligned time: {metrics['goal_aligned_minutes']} min",
        f"- Study sessions: {metrics['study_sessions']} ({metrics['study_minutes']} min)",
        f"- Practise sessions: {metrics['practise_sessions']} ({metrics['practise_minutes']} min)",
        f"- Suggested Anki candidates: {metrics['suggested_anki_cards']}",
        f"- Accepted/edited Anki candidates awaiting export: {metrics['export_ready_anki_cards']}",
        f"- Practise evidence captured: {metrics['practice_evidence_captured']}",
        f"- Practise evidence missing: {metrics['practice_evidence_debt']}",
    ]
    if draft.wins:
        lines.extend(["", "## Wins", *[f"- {item}" for item in draft.wins]])
    if draft.stalls:
        lines.extend(["", "## Stalls", *[f"- {item}" for item in draft.stalls]])
    if draft.evidence_progress:
        lines.extend(["", "## Evidence Progress", *[f"- {item}" for item in draft.evidence_progress]])
    if draft.friction_patterns:
        lines.extend(["", "## Friction Patterns", *[f"- {item}" for item in draft.friction_patterns]])
    if draft.next_week_focus:
        lines.extend(["", "## Next Week Focus", *[f"- {item}" for item in draft.next_week_focus]])

    # Evidence notes section (per D-19)
    if metrics.get("evidence_notes"):
        lines.append("")
        lines.append("## Evidence Notes")
        for note in metrics["evidence_notes"]:
            domain = note.get("domain", "general")
            duration = note.get("duration_min", 0)
            outcome = note.get("outcome", "")
            lines.append(f"- {domain}: {duration} min ({outcome})")

    # Pending retries section (per D-19)
    if metrics.get("pending_retries"):
        lines.append("")
        lines.append("## Pending Retry Items")
        for item in metrics["pending_retries"]:
            domain = item.get("domain", "")
            text = item.get("item_text", "")
            lines.append(f"- [{domain}] {text}")

    # Trend summary (per D-19)
    if metrics.get("evidence_trend_summary"):
        lines.append("")
        lines.append(f"**Trends:** {metrics['evidence_trend_summary']}")

    _append_thought_sections(lines, thought_context)
    return "\n".join(lines)


def _fallback_daily_review(metrics: dict) -> DailyReviewDraft:
    summary = (
        f"Study: {metrics['study_minutes']} min. Practise: {metrics['practise_minutes']} min. "
        f"Goal-aligned work: {metrics['goal_aligned_minutes']} min."
    )
    progress = []
    if metrics["goal_coverage"]:
        progress.append(f"Covered goals: {', '.join(metrics['goal_coverage'][:3])}")
    evidence = [f"Practise evidence captured: {metrics['practice_evidence_captured']}"]
    friction = []
    if metrics["practice_evidence_debt"]:
        friction.append(f"Practice evidence missing in {metrics['practice_evidence_debt']} session(s).")
    if metrics["study_stage_gaps"]:
        friction.append(metrics["study_stage_gaps"][0])
    adjustments = ["Run the next concrete block rather than broadening scope."]
    return DailyReviewDraft(
        summary=summary,
        progress_signals=progress,
        evidence_captured=evidence,
        friction_patterns=friction,
        next_adjustments=adjustments,
    )


def _fallback_weekly_review(metrics: dict) -> WeeklyReviewDraft:
    summary = (
        f"This week included {metrics['study_sessions']} study session(s) and "
        f"{metrics['practise_sessions']} practise session(s), totaling {metrics['goal_aligned_minutes']} goal-aligned minutes."
    )
    wins = [f"Captured practice evidence in {metrics['practice_evidence_captured']} session(s)."]
    stalls = metrics["study_stage_gaps"][:2] + metrics["practise_stage_gaps"][:2]
    evidence_progress = [f"{metrics['export_ready_anki_cards']} recall items are ready for export."]
    friction = []
    if metrics["interruptions"]:
        friction.append(f"Interruptions recorded: {metrics['interruptions']}.")
    next_week = ["Reduce ambiguity by choosing one explicit study or practise block first."]
    return WeeklyReviewDraft(
        summary=summary,
        wins=wins,
        stalls=stalls,
        evidence_progress=evidence_progress,
        friction_patterns=friction,
        next_week_focus=next_week,
    )


def _record_review_provenance(
    repo: Repository,
    runtime,
    generated_draft,
    *,
    artifact_kind: str,
    artifact_id: str,
    accepted: bool,
) -> None:
    try:
        repo.create_generation_provenance(
            runtime.build_provenance(
                artifact_kind=artifact_kind,
                artifact_id=artifact_id,
                generated_draft=generated_draft,
                accepted_by_user=accepted,
            )
        )
    except Exception:
        pass


def _check_anki_gate(console) -> None:
    """Show Anki context as optional follow-up information, never a gate."""
    try:
        from pb.storage.database import get_connection
        from pb.vault.anki_client import get_pending_card_count, is_anki_available

        # Compute avg distraction from today's completed sessions
        from datetime import date
        today_str = date.today().isoformat()

        with get_connection() as conn:
            row = conn.execute(
                "SELECT AVG(distraction) FROM sessions "
                "WHERE DATE(start_at) = ? AND distraction IS NOT NULL",
                (today_str,),
            ).fetchone()
            avg_distraction = float(row[0]) if row and row[0] is not None else 1.0

        pending_count = get_pending_card_count()

        # Get outstanding Anki reviews (if Anki available)
        outstanding_reviews = 0
        if is_anki_available():
            try:
                from pb.vault.anki_client import anki_request
                deck_stats = anki_request("getDeckStats", decks=["Default"])
                if deck_stats and isinstance(deck_stats, dict):
                    for deck_id, stats in deck_stats.items():
                        if isinstance(stats, dict):
                            outstanding_reviews += stats.get("due_count", 0) + stats.get("new_count", 0)
            except Exception:
                pass

        if pending_count == 0 and outstanding_reviews == 0:
            return  # Nothing to show

        if avg_distraction >= 3:
            console.print(
                f"[dim]Anki follow-up: {pending_count} accepted/edited cards awaiting export"
                f" · {outstanding_reviews} outstanding reviews[/]"
            )
            console.print("[dim]Use `pb study recall` or `pb anki` when you want to process them.[/]")
        else:
            console.print(
                f"[dim]Anki: {pending_count} accepted/edited cards awaiting export"
                f" · {outstanding_reviews} outstanding reviews[/]"
            )

    except Exception:
        pass  # Non-fatal: Anki gate must never break day review


# Review questions per D-07 (all 5 required) — kept for legacy mode
REVIEW_QUESTIONS = [
    {
        "id": "energy",
        "text": "How was your energy level today?",
        "context": "Consider physical and mental energy throughout the day",
    },
    {
        "id": "presence",
        "text": "How present were you during work sessions?",
        "context": "1=constantly distracted, 10=fully focused",
    },
    {
        "id": "best_window",
        "text": "Rate your best productive window today",
        "context": "Think of your peak performance period",
    },
    {
        "id": "blockers",
        "text": "How much did friction/distractions impact you?",
        "context": "1=minimal friction, 10=severely disrupted (higher=worse)",
    },
    {
        "id": "alignment",
        "text": "How aligned was your work with your goals?",
        "context": "Consider whether effort matched priorities",
    },
]

# Anchor labels per D-09 (at every other number)
ANCHOR_LABELS = {
    1: "Very low",
    3: "Low",
    5: "Moderate",
    7: "High",
    9: "Very high",
}


def _get_question_response(
    question: dict,
    gemini_available: bool,
) -> tuple[int, Optional[str], Optional[str]]:
    """
    Get response for a single review question (per D-06).

    Args:
        question: Question dict with id, text, context
        gemini_available: Whether Gemini API is available for chat mode

    Returns:
        (score, text_response, llm_rationale) tuple
    """
    console = get_console()
    console.print("")
    console.print(f"[header][{question['id'].upper()}] {question['text']}[/]")
    console.print(f"[dim]  {question['context']}[/]")
    console.print("")

    # Show anchor labels per D-09
    for num in [1, 3, 5, 7, 9]:
        label = ANCHOR_LABELS.get(num, "")
        console.print(f"  [dim]{num}[/] = {label}")
    console.print("")
    console.print("[dim]Enter 1-10 (or 0 for text mode)[/]")

    while True:
        try:
            raw = input("> ").strip()

            if raw.lower() in ("esc", "exit", "quit"):
                raise typer.Abort()

            value = int(raw)

            if value == 0:
                # Chat mode per D-06
                if not gemini_available:
                    console.print("[warn]Text mode unavailable (no API key). Please enter 1-10.[/]")
                    continue

                console.print("[dim]Describe in your own words:[/]")
                text_response = input("> ").strip()

                if not text_response:
                    console.print("[warn]Response cannot be empty. Please enter 1-10 or try text mode again.[/]")
                    continue

                # Get LLM scoring per D-12
                result = score_text_response(question["text"], text_response)
                if result is None:
                    console.print("[warn]Could not score response. Please enter 1-10 instead.[/]")
                    continue

                score, rationale = result
                console.print(f"  [success]Scored: {score}/10[/] - {escape(rationale)}")

                # Follow-up per D-10, D-11 (max 1 round)
                followup = generate_followup(question["text"], text_response)
                if followup:
                    console.print(f"\n  [dim]Follow-up:[/] {escape(followup)}")
                    followup_response = input("  > ").strip()
                    if followup_response:
                        # Re-score with combined context
                        combined = f"{text_response}\n\nFollow-up answer: {followup_response}"
                        result2 = score_text_response(question["text"], combined)
                        if result2:
                            score, rationale = result2
                            console.print(f"  [success]Updated score: {score}/10[/] - {escape(rationale)}")
                            text_response = combined

                return score, text_response, rationale

            if 1 <= value <= 10:
                return value, None, None

            console.print("[warn]Please enter a number between 1 and 10 (or 0 for text mode)[/]")

        except ValueError:
            console.print("[warn]Please enter a number between 1 and 10 (or 0 for text mode)[/]")


def _run_legacy_questions(repo: Repository, engine: ReviewEngine) -> None:
    """Run the old 5-question format (preserved for --legacy / --skip flag)."""
    console = get_console()
    console.rule("[header]Daily Reflection[/]")
    console.print("[dim]Answer 5 questions to track your progress.[/]")
    console.print("[dim]Press Ctrl+C to skip.[/]")

    # Check Gemini availability for chat mode per D-13
    gemini_client = GeminiClient()
    gemini_available = gemini_client.is_available()
    if not gemini_available:
        console.print("\n[dim](Text mode disabled - set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT)[/]")

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    responses: list[DailyReviewResponse] = []

    try:
        for question in REVIEW_QUESTIONS:
            score, text_response, llm_rationale = _get_question_response(
                question, gemini_available
            )

            response = DailyReviewResponse(
                review_date=today_str,
                question_id=question["id"],
                numeric_score=score,
                text_response=text_response,
                llm_rationale=llm_rationale,
            )
            responses.append(response)
            repo.create_review_response(response)

        # Show friction score with trend after questions per D-16
        blocker_score, trend_arrow = engine.get_blocker_trend(datetime.utcnow())
        if blocker_score is not None:
            console.print("")
            console.print(f"[dim]Friction Score:[/] {blocker_score:.1f}{escape(trend_arrow)}")

        console.rule("[header]Reflection Complete[/]")
        console.print("[success]Responses saved.[/]")

    except (KeyboardInterrupt, typer.Abort):
        console.print("\n\n[dim]Reflection skipped.[/]")
        # Save any responses collected so far
        if responses:
            console.print(f"[dim]({len(responses)} of 5 responses saved)[/]")


@app.command("day")
def review_day(
    ctx: typer.Context,
    save: bool = typer.Option(False, "--save", "-s", help="Save review to packet"),
    legacy: bool = typer.Option(False, "--legacy", help="Use old 5-question format"),
    skip_questions: bool = typer.Option(
        False, "--skip", help="Skip interactive questions (shows legacy report only)"
    ),
    yes: bool = typer.Option(False, "--yes", help="Persist the review without prompting"),
):
    """Run the daily learning review and compact debrief."""
    console = get_console()
    repo = ctx.obj["repo"] if ctx.obj and ctx.obj.get("repo") is not None else Repository()
    runtime = runtime_for_ctx(ctx)
    today = utc_now()

    metrics = _collect_review_metrics(repo, days=1)

    thought_context = None

    generated = None
    if legacy or skip_questions:
        console.print("[dim]Using deterministic daily review; no LLM request will be made.[/]")
        draft = _fallback_daily_review(metrics)
    else:
        try:
            generated = runtime.generate_draft(
                DailyReviewDraft,
                _build_daily_review_prompt(metrics, thought_context=thought_context),
                source_scope="review.day",
            )
            draft = generated.payload
        except DraftGenerationError as exc:
            console.print(f"[warn]{exc.to_user_message()}[/]")
            draft = _fallback_daily_review(metrics)
    report = _render_daily_review(metrics, draft, thought_context=thought_context)
    console.print(report, highlight=False)

    config = ctx.obj.get("config") if ctx.obj else None
    persist_review = save or (
        config is not None
        and getattr(getattr(config, "commit_policy", None), "daily_reviews", "") == "auto_to_quarantine"
    )

    if persist_review:
        accepted = confirm_preview(yes=yes, action_label="Save this daily review")
        if generated is not None:
            _record_review_provenance(
                repo,
                runtime,
                generated,
                artifact_kind="daily_review",
                artifact_id=today.strftime("%Y-%m-%d"),
                accepted=accepted,
            )
        if not accepted:
            console.print("[dim]Preview only. Daily review was not written.[/]")
            return

        try:
            from pb.core.review_log_writer import ReviewLogWriter
            from pb.vault.config import get_vault_path

            path = ReviewLogWriter(get_vault_path()).write_daily_log(report, today.date())
            if path is not None:
                console.print(f"[success]Saved daily review:[/] {escape(str(path))}")
        except Exception as exc:
            get_err_console().print(f"[error]Could not write daily review: {escape(str(exc))}[/]")
            raise typer.Exit(code=1)

    _check_anki_gate(console)


@app.command("week")
def review_week(
    ctx: typer.Context,
    save: bool = typer.Option(False, "--save", "-s", help="Save review to packet"),
    legacy: bool = typer.Option(False, "--legacy", help="Use old table-only format"),
    yes: bool = typer.Option(False, "--yes", help="Persist the review without prompting"),
):
    """Run the weekly structured reflection for your learning system."""
    console = get_console()
    repo = ctx.obj["repo"] if ctx.obj and ctx.obj.get("repo") is not None else Repository()
    runtime = runtime_for_ctx(ctx)
    today = utc_now()

    if legacy:
        console.print("[dim]`--legacy` is deprecated; using the LLM-backed weekly review.[/]")

    metrics = _collect_review_metrics(repo, days=7)

    thought_context = None

    generated = None
    try:
        generated = runtime.generate_draft(
            WeeklyReviewDraft,
            _build_weekly_review_prompt(metrics, thought_context=thought_context),
            source_scope="review.week",
        )
        draft = generated.payload
    except DraftGenerationError as exc:
        console.print(f"[warn]{exc.to_user_message()}[/]")
        draft = _fallback_weekly_review(metrics)
    report = _render_weekly_review(metrics, draft, thought_context=thought_context)
    console.print(report, highlight=False)

    try:
        from pb.vault import get_vault_path as _gvp
        _vault = _gvp()
        snapshot_domain_stats(_vault)
        growth_table = build_growth_table(_vault)
        if growth_table:
            console.print()
            console.rule("[header]Learning Growth[/]")
            console.print(growth_table)
    except Exception:
        pass

    config = ctx.obj.get("config") if ctx.obj else None
    persist_review = save or (
        config is not None
        and getattr(getattr(config, "commit_policy", None), "weekly_reviews", "") == "auto_to_quarantine"
    )

    if persist_review:
        week_id = f"{today.isocalendar()[0]}-W{today.isocalendar()[1]:02d}"
        accepted = confirm_preview(yes=yes, action_label="Save this weekly review")
        if generated is not None:
            _record_review_provenance(
                repo,
                runtime,
                generated,
                artifact_kind="weekly_review",
                artifact_id=week_id,
                accepted=accepted,
            )
        if not accepted:
            console.print("[dim]Preview only. Weekly review was not written.[/]")
            return

        try:
            from pb.core.review_log_writer import ReviewLogWriter
            from pb.vault.config import get_vault_path

            path = ReviewLogWriter(get_vault_path()).write_weekly_log(report, today.date())
            if path is not None:
                console.print(f"[success]Saved weekly review:[/] {escape(str(path))}")
        except Exception as exc:
            get_err_console().print(f"[error]Could not write weekly review: {escape(str(exc))}[/]")
            raise typer.Exit(code=1)


@app.command("alignment", hidden=True)
def review_alignment(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to analyze"),
    save: bool = typer.Option(False, "--save", "-s", help="Save report to packet"),
):
    """
    Show effort distribution across goal arcs.

    Displays how time has been allocated to different goals
    over the specified period (default: last 7 days).
    """
    # Clamp days to reasonable range per T-04-05 mitigation
    days = max(1, min(days, 365))

    console = get_console()
    repo = Repository()
    engine = AlignmentEngine(repo)

    breakdown = engine.get_alignment(days=days)
    report = engine.format_alignment_report(breakdown, days=days)

    console.print(report, highlight=False)

    if save:
        packet_engine = PacketEngine()
        period = f"alignment_{datetime.utcnow().strftime('%Y-%m-%d')}"
        path = packet_engine.write_review_packet(period)
        console.print(f"\n[dim]Saved to: {escape(str(path))}[/]")


@app.command("month", hidden=True)
def review_month():
    """Show 30-day aggregate with sparkline trends and MoM comparison (D-38)."""
    from pb.core.reports import ReportEngine
    console = get_console()
    repo = Repository()
    engine = ReportEngine(repo)
    report = engine.generate_month_report()
    console.print(report, highlight=False)


# ---------------------------------------------------------------------------
# Phase 19 ANLT-01/02/03: Domain growth analytics helpers
# ---------------------------------------------------------------------------


def snapshot_domain_stats(vault_path: Path) -> None:
    """Snapshot per-domain stats into domain_weekly_stats (D-18). Idempotent via upsert."""
    import datetime as _dt

    from pb.storage.database import get_connection
    from pb.vault.lifecycle import read_frontmatter

    today = _dt.date.today()
    week_start = (today - _dt.timedelta(days=today.weekday())).isoformat()
    knowledge_dir = vault_path / "knowledge"
    if not knowledge_dir.exists():
        return

    with get_connection() as conn:
        for domain_dir in knowledge_dir.iterdir():
            if not domain_dir.is_dir() or domain_dir.name.startswith("."):
                continue
            if not (domain_dir / "_state.md").exists():
                continue

            domain = domain_dir.name

            # Count notes by stage
            stage_counts = {"#new": 0, "#learning": 0, "#learnt": 0, "#stale": 0}
            notes_total = 0
            for md in domain_dir.glob("*.md"):
                if md.name.startswith("_"):
                    continue
                try:
                    fm, _ = read_frontmatter(md.read_text())
                    stage = fm.get("learning_stage", "#new")
                    if stage in stage_counts:
                        stage_counts[stage] += 1
                    notes_total += 1
                except Exception:
                    continue

            # Count socratic interactions this week from pb.db
            week_start_ts = f"{week_start}T00:00:00"
            socratic_count = 0
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM interactions "
                    "WHERE domain = ? AND event_type = 'socratic' AND ts >= ?",
                    (domain, week_start_ts),
                ).fetchone()
                socratic_count = row[0] if row else 0
            except Exception:
                pass

            # Count links in vault.db for this domain
            links_count = 0
            try:
                from pb.vault.graph_store import open_vault_db
                vconn = open_vault_db(vault_path)
                try:
                    row = vconn.execute(
                        "SELECT COUNT(*) FROM links "
                        "WHERE src IN (SELECT slug FROM nodes WHERE subfolder = ?)",
                        (domain,),
                    ).fetchone()
                    links_count = row[0] if row else 0
                finally:
                    vconn.close()
            except Exception:
                pass

            # Count Anki cards exported this week
            anki_count = 0
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM anki_cards WHERE domain = ? AND exported_at >= ?",
                    (domain, week_start_ts),
                ).fetchone()
                anki_count = row[0] if row else 0
            except Exception:
                pass

            # Upsert snapshot (ANLT-02: idempotent via INSERT OR REPLACE)
            conn.execute(
                """INSERT OR REPLACE INTO domain_weekly_stats
                   (domain, week_start, notes_created, links_added, anki_exported,
                    socratic_sessions, stage_new, stage_learning, stage_learnt, stage_stale,
                    snapshot_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    domain, week_start, notes_total, links_count, anki_count,
                    socratic_count,
                    stage_counts["#new"], stage_counts["#learning"],
                    stage_counts["#learnt"], stage_counts["#stale"],
                    today.isoformat(),
                ),
            )
        conn.commit()


def get_zero_activity_domains(vault_path: Path) -> list:
    """Return domain names with notes but 0 interactions in past 7 days (ANLT-03)."""
    import datetime as _dt

    from pb.storage.database import get_connection

    cutoff = (_dt.date.today() - _dt.timedelta(days=7)).isoformat() + "T00:00:00"
    knowledge_dir = vault_path / "knowledge"
    if not knowledge_dir.exists():
        return []

    zero_domains = []
    with get_connection() as conn:
        for domain_dir in knowledge_dir.iterdir():
            if not domain_dir.is_dir() or domain_dir.name.startswith("."):
                continue
            # Check if domain has any notes (exclude _state.md and other _-prefixed files)
            has_notes = any(
                f.suffix == ".md" and not f.name.startswith("_")
                for f in domain_dir.iterdir()
            )
            if not has_notes:
                continue

            # Check for any interactions in past 7 days
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM interactions WHERE domain = ? AND ts >= ?",
                    (domain_dir.name, cutoff),
                ).fetchone()
                if row and row[0] == 0:
                    zero_domains.append(domain_dir.name)
            except Exception:
                pass

    return zero_domains


def build_growth_table(vault_path: Path) -> Optional[Table]:
    """Build Rich table showing per-domain growth with delta arrows (ANLT-02).

    Compares current week snapshot vs previous week. Shows warning for zero-activity domains.
    Returns None if no current-week data is available.
    """
    import datetime as _dt

    from pb.storage.database import get_connection

    today = _dt.date.today()
    week_start = (today - _dt.timedelta(days=today.weekday())).isoformat()
    prev_week_start = (today - _dt.timedelta(days=today.weekday() + 7)).isoformat()

    with get_connection() as conn:
        # Current week stats
        current = {}
        for row in conn.execute(
            "SELECT * FROM domain_weekly_stats WHERE week_start = ?", (week_start,)
        ).fetchall():
            current[row["domain"]] = dict(row)

        # Previous week stats
        previous = {}
        for row in conn.execute(
            "SELECT * FROM domain_weekly_stats WHERE week_start = ?", (prev_week_start,)
        ).fetchall():
            previous[row["domain"]] = dict(row)

    if not current:
        return None

    zero_activity = set(get_zero_activity_domains(vault_path))

    t = Table(
        title="Learning Growth (this week)",
        show_header=True,
        header_style="bold",
        show_edge=False,
        show_lines=False,
        pad_edge=False,
        box=None,
    )
    t.add_column("DOMAIN", style="cyan")
    t.add_column("NOTES", justify="right")
    t.add_column("LINKS", justify="right")
    t.add_column("ANKI", justify="right")
    t.add_column("SESSIONS", justify="right")
    t.add_column("STAGES", no_wrap=True)
    t.add_column("", no_wrap=True)  # Warning column

    def _delta(curr_val, prev_val):
        """Return Rich markup delta string. Empty string if no previous data."""
        if prev_val is None:
            return ""
        diff = curr_val - prev_val
        if diff > 0:
            return f" [green]+{diff}[/]"
        elif diff < 0:
            return f" [red]{diff}[/]"
        return ""

    for domain, stats in sorted(current.items()):
        prev = previous.get(domain, {})
        stages = (
            f"N:{stats['stage_new']} L:{stats['stage_learning']} "
            f"Lt:{stats['stage_learnt']} S:{stats['stage_stale']}"
        )
        warning = "no activity" if domain in zero_activity else ""
        t.add_row(
            domain,
            f"{stats['notes_created']}{_delta(stats['notes_created'], prev.get('notes_created'))}",
            f"{stats['links_added']}{_delta(stats['links_added'], prev.get('links_added'))}",
            f"{stats['anki_exported']}{_delta(stats['anki_exported'], prev.get('anki_exported'))}",
            f"{stats['socratic_sessions']}{_delta(stats['socratic_sessions'], prev.get('socratic_sessions'))}",
            stages,
            warning,
        )

    return t


@app.command("track", hidden=True)
def review_track(
    track_name: str = typer.Argument(..., help="Track name"),
):
    """Show per-track report: hours, tasks, completion rate, trend (D-38)."""
    from pb.core.reports import ReportEngine
    console = get_console()
    repo = Repository()
    engine = ReportEngine(repo)
    report = engine.generate_track_report(track_name)
    console.print(report, highlight=False)


@app.command("energy", hidden=True)
def review_energy(
    ctx: typer.Context,
    days: int = typer.Option(7, "--days", "-d", help="Number of days to include"),
):
    """Energy trends with sparklines."""
    repo = ctx.obj["repo"]
    engine = ReportEngine(repo)
    typer.echo(engine.generate_energy_report(days=days))


@app.command("friction", hidden=True)
def review_friction(
    ctx: typer.Context,
    days: int = typer.Option(7, "--days", "-d", help="Number of days to include"),
):
    """Friction pattern analysis."""
    repo = ctx.obj["repo"]
    engine = ReportEngine(repo)
    typer.echo(engine.generate_friction_report(days=days))


@app.command("blockers", hidden=True)
def review_blockers(
    ctx: typer.Context,
    days: int = typer.Option(7, "--days", "-d", help="Number of days to include"),
):
    """Backward-compatible alias for `pb review friction`."""
    review_friction(ctx, days=days)


@app.command("priority", hidden=True)
def review_priority(ctx: typer.Context):
    """Priority and Eisenhower distribution."""
    repo = ctx.obj["repo"]
    engine = ReportEngine(repo)
    typer.echo(engine.generate_priority_report())
