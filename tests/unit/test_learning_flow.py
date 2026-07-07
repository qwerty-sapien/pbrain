from __future__ import annotations

from unittest.mock import MagicMock

from pb.cli.learning_flow import (
    ResourceFetchResult,
    build_learning_session_markdown,
    fetch_grounded_learning_resources,
)


def test_build_learning_session_markdown_includes_resources():
    resources = ResourceFetchResult(
        bundle={
            "summary": "Use one reference and one walkthrough.",
            "resources": [
                {
                    "title": "Charisma walkthrough",
                    "url": "https://example.com/charisma",
                    "resource_type": "video",
                    "why": "Shows pacing and delivery in context.",
                }
            ],
            "search_terms": ['YouTube: "charisma" walkthrough -shorts'],
        }
    )

    rendered = build_learning_session_markdown(
        task_title="Study: communication charisma",
        steps=[{"title": "Record one take", "instruction": "Speak for 2 minutes.", "success_check": "You have one sample."}],
        resources=resources,
    )

    assert rendered is not None
    assert "## Resources" in rendered
    assert "https://example.com/charisma" in rendered
    assert "## Search help" in rendered


def test_fetch_grounded_learning_resources_returns_search_help_when_grounding_fails(monkeypatch):
    client = MagicMock()
    client.is_available.return_value = True
    client.generate_with_grounding.return_value = None
    monkeypatch.setattr("pb.cli.learning_flow.get_client", lambda: client)

    result = fetch_grounded_learning_resources(
        topic="communication charisma",
        branch="study",
        block_payload={"subject_scope": "communication charisma", "success_check": "Explain three pacing moves."},
    )

    assert result.bundle is not None
    assert result.bundle.search_terms
    assert "another model" in result.warning.lower()
