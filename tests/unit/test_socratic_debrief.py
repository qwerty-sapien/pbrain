"""Unit tests for pb.vault.socratic (Phase 18, Plan 02).

Tests cover:
- is_domain_session: True/False domain detection
- SocraticDebriefEngine: LLM call, max rounds, collect_answers
- build_socratic_frontmatter: source enforcement, fields
- build_socratic_note: brief/deep templates, verbatim preservation
- extract_socratic_cards: no LLM, verbatim Q&A, card_type=socratic
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_task_with_project(tmp_path):
    """A minimal task-like object with a project name."""
    project = MagicMock()
    project.name = "deutsch"
    task = MagicMock()
    task.project = project
    return task


@pytest.fixture
def fake_task_no_project():
    """A minimal task-like object with no project."""
    task = MagicMock()
    task.project = None
    return task


@pytest.fixture
def domain_vault(tmp_path):
    """Vault with a knowledge/deutsch/_state.md file."""
    vault = tmp_path / "vault"
    vault.mkdir()
    knowledge_dir = vault / "knowledge" / "deutsch"
    knowledge_dir.mkdir(parents=True)
    state_md = knowledge_dir / "_state.md"
    state_md.write_text("# Domain State\n\nSession summaries here.\n")
    return vault


@pytest.fixture
def fake_client():
    """A mock GeminiClient that is available and returns a canned question."""
    client = MagicMock()
    client.is_available.return_value = True
    client.generate_with_model.return_value = "What was the most surprising thing you learned today?"
    return client


# ---------------------------------------------------------------------------
# is_domain_session
# ---------------------------------------------------------------------------


class TestIsDomainSession:
    def test_returns_true_when_project_maps_to_knowledge_folder_with_state(
        self, fake_task_with_project, domain_vault
    ):
        from pb.vault.socratic import is_domain_session

        assert is_domain_session(fake_task_with_project, domain_vault) is True

    def test_returns_false_when_task_has_no_project(self, fake_task_no_project, domain_vault):
        from pb.vault.socratic import is_domain_session

        assert is_domain_session(fake_task_no_project, domain_vault) is False

    def test_returns_false_when_project_folder_has_no_state_md(self, tmp_path):
        from pb.vault.socratic import is_domain_session

        vault = tmp_path / "vault"
        vault.mkdir()
        knowledge_dir = vault / "knowledge" / "math"
        knowledge_dir.mkdir(parents=True)
        # No _state.md created

        project = MagicMock()
        project.name = "math"
        task = MagicMock()
        task.project = project

        assert is_domain_session(task, vault) is False

    def test_returns_false_when_no_20_knowledge_dir(self, tmp_path):
        from pb.vault.socratic import is_domain_session

        vault = tmp_path / "vault"
        vault.mkdir()
        # No knowledge directory

        project = MagicMock()
        project.name = "deutsch"
        task = MagicMock()
        task.project = project

        assert is_domain_session(task, vault) is False


# ---------------------------------------------------------------------------
# SocraticDebriefEngine
# ---------------------------------------------------------------------------


class TestSocraticDebriefEngine:
    def test_get_question_calls_generate_with_model_with_flash_lite(
        self, fake_client
    ):
        from pb.vault.socratic import SocraticDebriefEngine
        from pb.llm.gemini import FLASH_LITE_MODEL

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client", return_value=fake_client):
            engine = SocraticDebriefEngine("deutsch", "State content.", max_rounds=3)
            engine._client = fake_client
            question = engine.get_question()

        assert question == "What was the most surprising thing you learned today?"
        fake_client.generate_with_model.assert_called_once()
        args = fake_client.generate_with_model.call_args
        # Second positional arg is model
        assert args[0][1] == FLASH_LITE_MODEL

    def test_stops_after_max_rounds_reached(self, fake_client):
        from pb.vault.socratic import SocraticDebriefEngine

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client", return_value=fake_client):
            engine = SocraticDebriefEngine("deutsch", "State.", max_rounds=2)
            engine._client = fake_client

            q1 = engine.get_question()
            assert q1 is not None
            assert engine.should_continue() is True

            q2 = engine.get_question("First answer")
            assert q2 is not None
            assert engine.should_continue() is False

            q3 = engine.get_question("Second answer")
            assert q3 is None

    def test_collect_answers_returns_qa_pairs(self, fake_client):
        from pb.vault.socratic import SocraticDebriefEngine

        call_count = 0
        questions = ["What clicked today?", "Can you elaborate?"]

        def side_effect(prompt, model, timeout=30):
            nonlocal call_count
            q = questions[call_count] if call_count < len(questions) else None
            call_count += 1
            return q

        fake_client.generate_with_model.side_effect = side_effect

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client", return_value=fake_client):
            engine = SocraticDebriefEngine("deutsch", "State.", max_rounds=3)
            engine._client = fake_client

            engine.get_question()                   # Q1 = "What clicked today?"
            engine.get_question("My first answer")  # records answer, gets Q2
            engine.get_question("My second answer") # records answer, gets Q3

        pairs = engine.collect_answers()
        assert len(pairs) == 2
        assert pairs[0][0] == "What clicked today?"
        assert pairs[0][1] == "My first answer"
        assert pairs[1][0] == "Can you elaborate?"
        assert pairs[1][1] == "My second answer"

    def test_get_question_uses_extended_timeout_on_first_round(self, fake_client):
        """Round 0 must use timeout=30 to survive Vertex AI ADC cold-start delays."""
        from pb.vault.socratic import SocraticDebriefEngine

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client", return_value=fake_client):
            engine = SocraticDebriefEngine("deutsch", "State.", max_rounds=3)
            engine._client = fake_client
            engine.get_question()

        assert fake_client.generate_with_model.call_args.kwargs["timeout"] == 30

    def test_get_question_uses_normal_timeout_on_subsequent_rounds(self, fake_client):
        """Rounds after the first use the normal 15s timeout (connection is warm)."""
        from pb.vault.socratic import SocraticDebriefEngine

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client", return_value=fake_client):
            engine = SocraticDebriefEngine("deutsch", "State.", max_rounds=3)
            engine._client = fake_client
            engine.get_question()
            fake_client.generate_with_model.reset_mock()
            engine.get_question("First answer")

        assert fake_client.generate_with_model.call_args.kwargs["timeout"] == 15

    def test_adaptive_mode_rejects_early_done_until_minimum_rounds(self, fake_client):
        from pb.vault.socratic import SocraticDebriefEngine
        from pb.llm.gemini import PRO_MODEL

        fake_client.generate_with_model.return_value = "DONE"

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client", return_value=fake_client):
            engine = SocraticDebriefEngine(
                "deutsch",
                "State.",
                max_rounds=30,
                adaptive=True,
                strict=True,
                model="pro",
                soft_cap_rounds=24,
            )
            engine._client = fake_client
            question = engine.get_question()

        assert question is not None
        assert "not enough to finish yet" in question.lower()
        assert engine.should_continue() is True
        args = fake_client.generate_with_model.call_args
        assert args[0][1] == PRO_MODEL

    def test_adaptive_prompt_mentions_time_remaining(self, fake_client):
        from pb.vault.socratic import SocraticDebriefEngine

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client", return_value=fake_client):
            engine = SocraticDebriefEngine(
                "deutsch",
                "State.",
                max_rounds=12,
                adaptive=True,
                time_limit_minutes=10,
            )
            engine._client = fake_client
            engine._started_at -= 60
            prompt = engine._build_adaptive_prompt("")

        assert "Time remaining:" in prompt
        assert "There is still useful time left" in prompt

    def test_get_question_stops_when_time_limit_has_elapsed(self, fake_client):
        from pb.vault.socratic import SocraticDebriefEngine

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client", return_value=fake_client):
            engine = SocraticDebriefEngine(
                "deutsch",
                "State.",
                max_rounds=12,
                adaptive=True,
                time_limit_minutes=3,
            )
            engine._client = fake_client
            engine._started_at -= 181
            question = engine.get_question("Late answer")

        assert question is None
        assert engine.exit_reason == "time_limit"
        fake_client.generate_with_model.assert_not_called()


# ---------------------------------------------------------------------------
# build_socratic_frontmatter
# ---------------------------------------------------------------------------


class TestBuildSocraticFrontmatter:
    def test_always_includes_source_socratic(self):
        from pb.vault.socratic import build_socratic_frontmatter

        fm = build_socratic_frontmatter("deutsch", "separable-verbs", [])
        assert fm["source"] == "socratic"

    def test_includes_learning_stage_new(self):
        from pb.vault.socratic import build_socratic_frontmatter

        fm = build_socratic_frontmatter("deutsch", "separable-verbs", [])
        assert fm["learning_stage"] == "#new"

    def test_includes_domain(self):
        from pb.vault.socratic import build_socratic_frontmatter

        fm = build_socratic_frontmatter("deutsch", "separable-verbs", [])
        assert fm["domain"] == "deutsch"

    def test_includes_links(self):
        from pb.vault.socratic import build_socratic_frontmatter

        wikilinks = ["german-grammar", "verb-prefixes"]
        fm = build_socratic_frontmatter("deutsch", "separable-verbs", wikilinks)
        assert fm["links"] == wikilinks

    def test_includes_created_date(self):
        from pb.vault.socratic import build_socratic_frontmatter
        import datetime

        fm = build_socratic_frontmatter("deutsch", "separable-verbs", [])
        assert fm["created"] == datetime.date.today().isoformat()


# ---------------------------------------------------------------------------
# build_socratic_note
# ---------------------------------------------------------------------------


class TestBuildSocraticNote:
    def _qa(self):
        return [
            ("What clicked today?", "I finally understood separable verbs."),
            ("Can you give an example?", "aufmachen splits to mach...auf in main clause."),
            ("Why does position matter?", "German verb-second rule forces prefix to end."),
            ("What is still unclear?", "Indirect speech verb ordering confuses me."),
            ("Cross-domain connection?", "Reminds me of Python decorators wrapping functions."),
        ]

    def test_brief_template_has_insight_and_links_sections(self):
        from pb.vault.socratic import build_socratic_note

        note = build_socratic_note(self._qa(), "deutsch", "sep-verbs", [], template="brief")
        assert "## Insight" in note
        assert "## Links" in note

    def test_brief_template_does_not_have_deep_sections(self):
        from pb.vault.socratic import build_socratic_note

        note = build_socratic_note(self._qa(), "deutsch", "sep-verbs", [], template="brief")
        assert "## Key Insight" not in note
        assert "## Connections" not in note

    def test_deep_template_has_all_four_sections(self):
        from pb.vault.socratic import build_socratic_note

        note = build_socratic_note(self._qa(), "deutsch", "sep-verbs", [], template="deep")
        assert "## Key Insight" in note
        assert "## Connections" in note
        assert "## Open Questions" in note
        assert "## Cross-Domain" in note

    def test_deep_template_does_not_have_brief_insight_section(self):
        from pb.vault.socratic import build_socratic_note

        note = build_socratic_note(self._qa(), "deutsch", "sep-verbs", [], template="deep")
        # "## Insight" should not appear as a standalone section (Key Insight is ok)
        lines = note.splitlines()
        assert not any(line.strip() == "## Insight" for line in lines)

    def test_preserves_users_exact_words_in_body(self):
        from pb.vault.socratic import build_socratic_note

        qa = [("Question?", "My verbatim answer goes here exactly.")]
        note = build_socratic_note(qa, "deutsch", "test-note", [], template="brief")
        assert "My verbatim answer goes here exactly." in note

    def test_includes_source_socratic_in_frontmatter(self):
        from pb.vault.socratic import build_socratic_note

        note = build_socratic_note(self._qa(), "deutsch", "sep-verbs", [], template="brief")
        assert "source: socratic" in note

    def test_wikilinks_appear_in_output(self):
        from pb.vault.socratic import build_socratic_note

        note = build_socratic_note(
            [("Q?", "A.")], "deutsch", "slug", ["german-grammar", "verb-prefixes"], template="brief"
        )
        assert "[[german-grammar]]" in note
        assert "[[verb-prefixes]]" in note


# ---------------------------------------------------------------------------
# extract_socratic_cards
# ---------------------------------------------------------------------------


class TestExtractSocraticCards:
    def test_returns_list_of_dicts_with_required_fields(self):
        from pb.vault.socratic import extract_socratic_cards

        qa = [("What is X?", "X is the thing.")]
        cards = extract_socratic_cards(qa, "my-slug", "German", "deutsch")
        assert len(cards) == 1
        card = cards[0]
        assert card["front"] == "What is X?"
        assert card["back"] == "X is the thing."
        assert card["card_type"] == "socratic"
        assert card["note_slug"] == "my-slug"
        assert card["deck"] == "German"

    def test_card_type_is_always_socratic(self):
        from pb.vault.socratic import extract_socratic_cards

        qa = [("Q1?", "A1."), ("Q2?", "A2.")]
        cards = extract_socratic_cards(qa, "slug", "Deck", "domain")
        for card in cards:
            assert card["card_type"] == "socratic"

    def test_produces_no_llm_calls(self):
        """extract_socratic_cards must never call any LLM."""
        from pb.vault.socratic import extract_socratic_cards

        with patch("pb.vault.socratic.SocraticDebriefEngine._get_client") as mock_get:
            qa = [("Q?", "A.")]
            cards = extract_socratic_cards(qa, "slug", "Deck", "domain")
            mock_get.assert_not_called()

        assert len(cards) == 1

    def test_verbatim_question_as_front(self):
        from pb.vault.socratic import extract_socratic_cards

        qa = [("Explain the Konjunktiv II usage.", "It signals hypothetical mood.")]
        cards = extract_socratic_cards(qa, "slug", "Deck", "deutsch")
        assert cards[0]["front"] == "Explain the Konjunktiv II usage."

    def test_verbatim_answer_as_back(self):
        from pb.vault.socratic import extract_socratic_cards

        qa = [("Q?", "The answer is preserved verbatim here.")]
        cards = extract_socratic_cards(qa, "slug", "Deck", "deutsch")
        assert cards[0]["back"] == "The answer is preserved verbatim here."

    def test_multiple_pairs_produce_multiple_cards(self):
        from pb.vault.socratic import extract_socratic_cards

        qa = [("Q1?", "A1."), ("Q2?", "A2."), ("Q3?", "A3.")]
        cards = extract_socratic_cards(qa, "slug", "Deck", "domain")
        assert len(cards) == 3

    def test_card_has_status_pending(self):
        from pb.vault.socratic import extract_socratic_cards

        cards = extract_socratic_cards([("Q?", "A.")], "slug", "Deck", "domain")
        assert cards[0]["status"] == "pending"

    def test_tags_are_serialized_as_yaml_list(self):
        from pb.storage.yaml_io import load_yaml_text
        from pb.vault.socratic import extract_socratic_cards

        cards = extract_socratic_cards([("Q?", "A.")], "slug", "Deck", "deutsch")
        assert load_yaml_text(cards[0]["tags"], []) == ["deutsch", "socratic"]
