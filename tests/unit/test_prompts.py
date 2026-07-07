"""Tests for ProactivePromptsEngine -- deterministic prompt generation (D-17, D-18).

Prompt types:
  - overdue_commitment: past cadence window or due date
  - birthday: within 14 days (D-17)
  - gift_reminder: idea exists + occasion approaching
  - decay_warning: no interaction in 90+ days or 2x cadence
  - goal_deadline: goals with target_date within 14 days (D-13)
  - event_prep: events within 48 hours (D-13)
  - skill_gap: opportunities with weak skill match and deadline within 30 days (D-13)
  - stale_inbox: inbox items older than 7 days (D-13)

All computation is date math only -- no LLM dependency (D-18).
"""

import pytest
import yaml
from datetime import date
from pathlib import Path
from typing import Optional

from pb.core.prompts import ProactivePromptsEngine, Prompt

_people_available = True
try:
    from pb.core.people import PeopleManager  # noqa: F401
except ImportError:
    _people_available = False

pytestmark = pytest.mark.skipif(
    not _people_available,
    reason="pb.core.people quarantined — prompts engine requires it",
)


# -- Helpers ----------------------------------------------------------------


def _create_person_note(
    vault_dir: Path,
    name: str,
    frontmatter_overrides: Optional[dict] = None,
    sections: Optional[dict] = None,
) -> Path:
    """Write a person note to vault_dir/30-people/{name.lower()}.md.

    Default frontmatter: type: person, name: {name}, relationship_type: friend,
    status: active, contact_cadence: none, tags: [].

    sections: dict mapping section name to content body text.
    """
    people_dir = vault_dir / "30-people"
    people_dir.mkdir(parents=True, exist_ok=True)

    fm = {
        "type": "person",
        "name": name,
        "relationship_type": "friend",
        "status": "active",
        "contact_cadence": "none",
        "tags": [],
    }
    if frontmatter_overrides:
        fm.update(frontmatter_overrides)

    fm_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True)
    body_parts = [f"---\n{fm_str}---\n\n# {name}\n"]

    if sections:
        for section_name, content in sections.items():
            body_parts.append(f"\n## {section_name}\n\n{content}\n")
    else:
        # Default empty sections
        for sec in ("Context", "Profile", "Commitments", "Gifts", "Interaction Log", "Notes"):
            body_parts.append(f"\n## {sec}\n\n")

    path = people_dir / f"{name.lower()}.md"
    path.write_text("".join(body_parts))
    return path


# -- TestPromptDataclass ----------------------------------------------------


class TestPromptDataclass:
    """Test Prompt dataclass fields."""

    def test_prompt_has_required_fields(self):
        """Prompt has prompt_type, person_name, message, urgency."""
        p = Prompt(
            prompt_type="birthday",
            person_name="Alice",
            message="Birthday in 5 days",
            urgency=9,
        )
        assert p.prompt_type == "birthday"
        assert p.person_name == "Alice"
        assert p.message == "Birthday in 5 days"
        assert p.urgency == 9


# -- TestOverdueCommitments -------------------------------------------------


class TestOverdueCommitments:
    """Test overdue commitment prompt generation."""

    def test_overdue_commitment_detected(self, tmp_path):
        """Cadence-based commitment past window generates overdue_commitment prompt."""
        _create_person_note(
            tmp_path, "Alice",
            sections={
                "Commitments": "- [ ] Call monthly (cadence: monthly, last: 2026-03-01)",
                "Interaction Log": "- 2026-03-01: Called",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        overdue = [p for p in prompts if p.prompt_type == "overdue_commitment"]
        assert len(overdue) >= 1
        assert overdue[0].person_name == "Alice"
        assert overdue[0].urgency > 0

    def test_commitment_within_window_no_prompt(self, tmp_path):
        """Commitment within cadence window generates no prompt."""
        _create_person_note(
            tmp_path, "Alice",
            sections={
                "Commitments": "- [ ] Call monthly (cadence: monthly, last: 2026-04-20)",
                "Interaction Log": "- 2026-04-20: Called",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        overdue = [p for p in prompts if p.prompt_type == "overdue_commitment"]
        assert len(overdue) == 0

    def test_completed_commitment_ignored(self, tmp_path):
        """Completed (checked) commitment generates no prompt."""
        _create_person_note(
            tmp_path, "Alice",
            sections={
                "Commitments": "- [x] Done thing (done: 2026-04-01)",
                "Interaction Log": "- 2026-04-01: Did the thing",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        overdue = [p for p in prompts if p.prompt_type == "overdue_commitment"]
        assert len(overdue) == 0

    def test_due_date_commitment_overdue(self, tmp_path):
        """Due-date commitment past due generates overdue prompt."""
        _create_person_note(
            tmp_path, "Alice",
            sections={
                "Commitments": "- [ ] Send report (due: 2026-04-20)",
                "Interaction Log": "- 2026-04-15: Chatted",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        overdue = [p for p in prompts if p.prompt_type == "overdue_commitment"]
        assert len(overdue) >= 1
        assert overdue[0].urgency == 6  # 2026-04-26 - 2026-04-20 = 6 days overdue


# -- TestBirthdayPrompts ----------------------------------------------------


class TestBirthdayPrompts:
    """Test birthday prompt generation (D-17: within 14 days)."""

    def test_birthday_within_14_days(self, tmp_path):
        """Birthday 9 days away generates birthday prompt."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"birthday": "1990-05-05"},
            sections={"Interaction Log": "- 2026-04-20: Chatted"},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        bday = [p for p in prompts if p.prompt_type == "birthday"]
        assert len(bday) == 1
        assert bday[0].person_name == "Alice"
        assert bday[0].urgency == 5  # 14 - 9 = 5

    def test_birthday_beyond_14_days(self, tmp_path):
        """Birthday more than 14 days away generates NO prompt."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"birthday": "1990-06-15"},
            sections={"Interaction Log": "- 2026-04-20: Chatted"},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        bday = [p for p in prompts if p.prompt_type == "birthday"]
        assert len(bday) == 0

    def test_birthday_today(self, tmp_path):
        """Birthday matching today generates prompt with urgency=14."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"birthday": "1990-04-26"},
            sections={"Interaction Log": "- 2026-04-20: Chatted"},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        bday = [p for p in prompts if p.prompt_type == "birthday"]
        assert len(bday) == 1
        assert bday[0].urgency == 14

    def test_no_birthday_no_prompt(self, tmp_path):
        """Person with no birthday field generates no birthday prompt."""
        _create_person_note(
            tmp_path, "Alice",
            sections={"Interaction Log": "- 2026-04-20: Chatted"},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        bday = [p for p in prompts if p.prompt_type == "birthday"]
        assert len(bday) == 0

    def test_birthday_already_passed_this_year(self, tmp_path):
        """Birthday already passed this year generates no prompt."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"birthday": "1990-03-01"},
            sections={"Interaction Log": "- 2026-04-20: Chatted"},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        bday = [p for p in prompts if p.prompt_type == "birthday"]
        assert len(bday) == 0


# -- TestGiftReminders ------------------------------------------------------


class TestGiftReminders:
    """Test gift reminder prompt generation."""

    def test_gift_idea_before_occasion(self, tmp_path):
        """Gift idea with approaching occasion generates gift_reminder."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"birthday": "1990-05-05"},
            sections={
                "Gifts": "- \U0001f4a1 Watch (occasion: birthday 2026-05)",
                "Interaction Log": "- 2026-04-20: Chatted",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        gifts = [p for p in prompts if p.prompt_type == "gift_reminder"]
        assert len(gifts) == 1
        assert gifts[0].person_name == "Alice"
        assert "Watch" in gifts[0].message

    def test_no_gift_ideas_no_reminder(self, tmp_path):
        """Person with no gift ideas generates no gift_reminder."""
        _create_person_note(
            tmp_path, "Alice",
            sections={"Interaction Log": "- 2026-04-20: Chatted"},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        gifts = [p for p in prompts if p.prompt_type == "gift_reminder"]
        assert len(gifts) == 0

    def test_gift_already_given_no_reminder(self, tmp_path):
        """Gift already given (checkmark) generates no gift_reminder."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"birthday": "1990-05-05"},
            sections={
                "Gifts": "- ✅ Watch (occasion: birthday 2026-05)",
                "Interaction Log": "- 2026-04-20: Chatted",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        gifts = [p for p in prompts if p.prompt_type == "gift_reminder"]
        assert len(gifts) == 0


# -- TestDecayWarning -------------------------------------------------------


class TestDecayWarning:
    """Test relationship decay warning prompt generation (D-17)."""

    def test_decay_no_contact_90_days(self, tmp_path):
        """Person with last interaction 100 days ago and cadence=none generates decay_warning."""
        _create_person_note(
            tmp_path, "Alice",
            sections={
                "Interaction Log": "- 2026-01-16: Last contact",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        decay = [p for p in prompts if p.prompt_type == "decay_warning"]
        assert len(decay) >= 1
        assert decay[0].person_name == "Alice"

    def test_decay_2x_cadence(self, tmp_path):
        """Person with cadence=monthly and last contact 65 days ago (>2x30=60) generates decay_warning."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"contact_cadence": "monthly"},
            sections={
                "Interaction Log": "- 2026-02-20: Called",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        decay = [p for p in prompts if p.prompt_type == "decay_warning"]
        assert len(decay) >= 1

    def test_no_decay_within_threshold(self, tmp_path):
        """Person with last contact 20 days ago and cadence=monthly generates NO decay_warning."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"contact_cadence": "monthly"},
            sections={
                "Interaction Log": "- 2026-04-06: Called",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        decay = [p for p in prompts if p.prompt_type == "decay_warning"]
        assert len(decay) == 0

    def test_decay_no_log_entries_warns(self, tmp_path):
        """Person with no interaction log generates decay_warning (never contacted)."""
        _create_person_note(
            tmp_path, "Alice",
            sections={},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        decay = [p for p in prompts if p.prompt_type == "decay_warning"]
        assert len(decay) >= 1
        assert "Never contacted" in decay[0].message


# -- TestGetPrompts ---------------------------------------------------------


class TestGetPrompts:
    """Test get_prompts() integration: scans all people, sorts, handles edge cases."""

    def test_get_prompts_scans_all_people(self, tmp_path):
        """Three person notes with various states produce combined prompts."""
        # Alice: overdue commitment
        _create_person_note(
            tmp_path, "Alice",
            sections={
                "Commitments": "- [ ] Call (cadence: monthly, last: 2026-03-01)",
                "Interaction Log": "- 2026-03-01: Called",
            },
        )
        # Bob: upcoming birthday
        _create_person_note(
            tmp_path, "Bob",
            frontmatter_overrides={"birthday": "1985-05-05"},
            sections={"Interaction Log": "- 2026-04-20: Met"},
        )
        # Carol: decay warning (no recent interaction)
        _create_person_note(
            tmp_path, "Carol",
            sections={
                "Interaction Log": "- 2026-01-01: Last contact",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        types = {p.prompt_type for p in prompts}
        assert "overdue_commitment" in types
        assert "birthday" in types
        assert "decay_warning" in types

    def test_get_prompts_sorted_by_urgency_desc(self, tmp_path):
        """Most urgent prompt is first in returned list."""
        # Low urgency: commitment overdue by 1 day
        _create_person_note(
            tmp_path, "Alice",
            sections={
                "Commitments": "- [ ] Call (cadence: monthly, last: 2026-03-25)",
                "Interaction Log": "- 2026-03-25: Called",
            },
        )
        # High urgency: never contacted (urgency=999)
        _create_person_note(
            tmp_path, "Bob",
            sections={},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        if len(prompts) >= 2:
            assert prompts[0].urgency >= prompts[1].urgency

    def test_get_prompts_skips_archived(self, tmp_path):
        """Archived person generates no prompts."""
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"status": "archived"},
            sections={
                "Commitments": "- [ ] Call (cadence: monthly, last: 2026-01-01)",
                "Interaction Log": "- 2026-01-01: Called",
            },
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        assert len(prompts) == 0

    def test_get_prompts_empty_vault_returns_empty(self, tmp_path):
        """No person notes returns empty list."""
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        assert prompts == []

    def test_get_prompts_malformed_data_skipped(self, tmp_path):
        """Person note with bad frontmatter is skipped (non-fatal)."""
        people_dir = tmp_path / "30-people"
        people_dir.mkdir(parents=True)
        # Write a note with unparseable frontmatter
        bad_note = people_dir / "bad.md"
        bad_note.write_text("---\n: invalid yaml [[\n---\n\n# Bad")
        # Write a good note
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"birthday": "1990-05-05"},
            sections={"Interaction Log": "- 2026-04-20: Chatted"},
        )
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        # Should still get prompts from Alice despite bad note
        assert any(p.person_name == "Alice" for p in prompts)


# -- Cross-domain vault note helpers ----------------------------------------


def _create_vault_note(
    vault_dir: Path,
    subfolder: str,
    filename: str,
    frontmatter: dict,
) -> Path:
    """Write a vault note with YAML frontmatter to vault_dir/{subfolder}/{filename}.

    Used for goals, events, opportunities, and inbox items.
    """
    target_dir = vault_dir / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    fm_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    title = frontmatter.get("title", filename.replace(".md", ""))
    content = f"---\n{fm_str}---\n\n# {title}\n\nSome body text.\n"

    path = target_dir / filename
    path.write_text(content)
    return path


# -- TestGoalDeadlinePrompts ------------------------------------------------


class TestGoalDeadlinePrompts:
    """Test goal deadline prompt generation (D-13: within 14 days)."""

    def test_goal_deadline_within_14_days(self, tmp_path):
        """Goal with target_date 7 days away generates goal_deadline prompt with urgency=7."""
        _create_vault_note(tmp_path, "direction/goals", "learn-rust.md", {
            "type": "goal",
            "title": "Learn Rust",
            "target_date": "2026-05-03",
            "status": "active",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        goals = [p for p in prompts if p.prompt_type == "goal_deadline"]
        assert len(goals) == 1
        assert goals[0].person_name == "Learn Rust"
        assert goals[0].urgency == 7  # 14 - 7 = 7

    def test_goal_deadline_outside_14_day_window(self, tmp_path):
        """Goal with target_date 20 days away generates NO prompt."""
        _create_vault_note(tmp_path, "direction/goals", "learn-rust.md", {
            "type": "goal",
            "title": "Learn Rust",
            "target_date": "2026-05-16",
            "status": "active",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        goals = [p for p in prompts if p.prompt_type == "goal_deadline"]
        assert len(goals) == 0

    def test_goal_complete_no_prompt(self, tmp_path):
        """Completed goal generates no prompt even if deadline is near."""
        _create_vault_note(tmp_path, "direction/goals", "done-goal.md", {
            "type": "goal",
            "title": "Done Goal",
            "target_date": "2026-04-28",
            "status": "complete",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        goals = [p for p in prompts if p.prompt_type == "goal_deadline"]
        assert len(goals) == 0

    def test_goal_deadline_today_max_urgency(self, tmp_path):
        """Goal with target_date today has urgency=14 (most urgent)."""
        _create_vault_note(tmp_path, "direction/goals", "urgent-goal.md", {
            "type": "goal",
            "title": "Urgent Goal",
            "target_date": "2026-04-26",
            "status": "active",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        goals = [p for p in prompts if p.prompt_type == "goal_deadline"]
        assert len(goals) == 1
        assert goals[0].urgency == 14  # 14 - 0 = 14


# -- TestEventPrepPrompts --------------------------------------------------


class TestEventPrepPrompts:
    """Test event preparation prompt generation (D-13: within 48 hours)."""

    def test_event_within_48_hours(self, tmp_path):
        """Event tomorrow generates event_prep prompt."""
        _create_vault_note(tmp_path, "40-events", "conference.md", {
            "type": "event",
            "title": "Tech Conference",
            "date": "2026-04-27",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        events = [p for p in prompts if p.prompt_type == "event_prep"]
        assert len(events) == 1
        assert events[0].person_name == "Tech Conference"

    def test_event_5_days_away_no_prompt(self, tmp_path):
        """Event 5 days away generates NO prompt."""
        _create_vault_note(tmp_path, "40-events", "conference.md", {
            "type": "event",
            "title": "Tech Conference",
            "date": "2026-05-01",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        events = [p for p in prompts if p.prompt_type == "event_prep"]
        assert len(events) == 0

    def test_event_includes_linked_people(self, tmp_path):
        """Event with linked_people includes them in the message."""
        _create_vault_note(tmp_path, "40-events", "dinner.md", {
            "type": "event",
            "title": "Team Dinner",
            "date": "2026-04-27",
            "linked_people": ["Alice", "Bob"],
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        events = [p for p in prompts if p.prompt_type == "event_prep"]
        assert len(events) == 1
        assert "Alice" in events[0].message
        assert "Bob" in events[0].message


# -- TestSkillGapPrompts ---------------------------------------------------


class TestSkillGapPrompts:
    """Test skill gap nudge prompt generation (D-13)."""

    def test_skill_gap_low_score_near_deadline(self, tmp_path):
        """Opportunity with low skill_match_score and deadline within 30 days generates prompt."""
        _create_vault_note(tmp_path, "50-opportunities", "job-posting.md", {
            "type": "opportunity",
            "title": "Senior Dev Role",
            "deadline": "2026-05-10",
            "skill_match_score": 0.4,
            "status": "active",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        skills = [p for p in prompts if p.prompt_type == "skill_gap"]
        assert len(skills) == 1
        assert skills[0].person_name == "Senior Dev Role"
        assert "40%" in skills[0].message

    def test_skill_gap_high_score_no_prompt(self, tmp_path):
        """Opportunity with high skill_match_score generates NO prompt."""
        _create_vault_note(tmp_path, "50-opportunities", "good-fit.md", {
            "type": "opportunity",
            "title": "Perfect Fit",
            "deadline": "2026-05-10",
            "skill_match_score": 0.9,
            "status": "active",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        skills = [p for p in prompts if p.prompt_type == "skill_gap"]
        assert len(skills) == 0

    def test_skill_gap_archived_no_prompt(self, tmp_path):
        """Archived opportunity generates NO prompt even with low score."""
        _create_vault_note(tmp_path, "50-opportunities", "old-opp.md", {
            "type": "opportunity",
            "title": "Old Opportunity",
            "deadline": "2026-05-10",
            "skill_match_score": 0.3,
            "status": "archived",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        skills = [p for p in prompts if p.prompt_type == "skill_gap"]
        assert len(skills) == 0


# -- TestStaleInboxPrompts -------------------------------------------------


class TestStaleInboxPrompts:
    """Test stale inbox prompt generation (D-13: older than 7 days)."""

    def test_stale_inbox_old_item(self, tmp_path):
        """Inbox item older than 7 days generates stale_inbox prompt."""
        _create_vault_note(tmp_path, "00-inbox", "old-idea.md", {
            "type": "inbox",
            "title": "Old Idea",
            "created": "2026-04-15",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        stale = [p for p in prompts if p.prompt_type == "stale_inbox"]
        assert len(stale) == 1
        assert stale[0].person_name == "Old Idea"
        assert stale[0].urgency == 11  # 26 - 15 = 11 days old

    def test_stale_inbox_recent_item_no_prompt(self, tmp_path):
        """Inbox item younger than 7 days generates NO prompt."""
        _create_vault_note(tmp_path, "00-inbox", "fresh-idea.md", {
            "type": "inbox",
            "title": "Fresh Idea",
            "created": "2026-04-22",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        stale = [p for p in prompts if p.prompt_type == "stale_inbox"]
        assert len(stale) == 0


# -- TestCrossDomainGetPrompts ---------------------------------------------


class TestCrossDomainGetPrompts:
    """Test get_prompts() returns prompts from all 8 types sorted by urgency."""

    def test_get_prompts_all_8_types_sorted(self, tmp_path):
        """get_prompts returns prompts from people AND cross-domain sources, sorted by urgency."""
        # Person: birthday in 5 days (urgency=9)
        _create_person_note(
            tmp_path, "Alice",
            frontmatter_overrides={"birthday": "1990-05-01"},
            sections={"Interaction Log": "- 2026-04-20: Chatted"},
        )
        # Goal: deadline in 3 days (urgency=11)
        _create_vault_note(tmp_path, "direction/goals", "ship-v1.md", {
            "type": "goal",
            "title": "Ship v1",
            "target_date": "2026-04-29",
            "status": "active",
        })
        # Event: tomorrow (urgency=2)
        _create_vault_note(tmp_path, "40-events", "standup.md", {
            "type": "event",
            "title": "Standup",
            "date": "2026-04-27",
        })
        # Stale inbox: 10 days old (urgency=10)
        _create_vault_note(tmp_path, "00-inbox", "stale.md", {
            "type": "inbox",
            "title": "Stale Item",
            "created": "2026-04-16",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))

        types = {p.prompt_type for p in prompts}
        assert "birthday" in types
        assert "goal_deadline" in types
        assert "event_prep" in types
        assert "stale_inbox" in types

        # Verify sorted by urgency descending
        for i in range(len(prompts) - 1):
            assert prompts[i].urgency >= prompts[i + 1].urgency

    def test_malformed_goal_frontmatter_skipped(self, tmp_path):
        """Malformed frontmatter in goal/event/opp files does not crash."""
        goals_dir = tmp_path / "direction" / "goals"
        goals_dir.mkdir(parents=True)
        bad_note = goals_dir / "bad-goal.md"
        bad_note.write_text("---\n: invalid yaml [[\n---\n\n# Bad")
        # Write a good goal
        _create_vault_note(tmp_path, "direction/goals", "good-goal.md", {
            "type": "goal",
            "title": "Good Goal",
            "target_date": "2026-04-29",
            "status": "active",
        })
        engine = ProactivePromptsEngine(vault_path=tmp_path)
        prompts = engine.get_prompts(today=date(2026, 4, 26))
        goals = [p for p in prompts if p.prompt_type == "goal_deadline"]
        assert len(goals) == 1
        assert goals[0].person_name == "Good Goal"
