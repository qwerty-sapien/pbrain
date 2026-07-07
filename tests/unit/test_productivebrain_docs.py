"""Documentation guardrails for the ProductiveBrain focus and examples."""

from pathlib import Path


def test_readme_centers_the_next_step_learning_loop() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "ProductiveBrain" in readme
    assert "next concrete session" in readme
    assert "pb learn" in readme
    assert "pb next" in readme
    assert "pb do" in readme
    assert "pb finish" in readme
    assert "pb review week" in readme
    assert "[Examples]" in readme


def test_examples_doc_uses_real_user_pb_transcripts() -> None:
    examples = Path("docs/examples.md").read_text(encoding="utf-8")

    assert "> pb learn" in examples
    assert "> pb next" in examples
    assert "> pb do" in examples
    assert "> pb finish" in examples
    assert "> pb study recall" in examples


def test_readme_keeps_advanced_setup_available() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "pb init" in readme
    assert "productivebrain-mcp" in readme


def test_readme_drops_stale_product_claims() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert "llm being optional" not in lowered
    assert "taskwarrior" not in lowered
    assert "timewarrior" not in lowered
