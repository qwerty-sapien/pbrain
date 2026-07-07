from __future__ import annotations

from datetime import datetime, timedelta

from pb.core.context_scope import ContextScopeFilter
from pb.core.interest_hierarchy import InterestHierarchyService, InterestSignal
from pb.core.models import Session, Task


def _session(repo, title: str, scope: str, start_at: datetime) -> None:
    task = repo.create_task(Task(title=title, work_type="study"))
    repo.create_session(
        Session(
            task_id=task.id,
            branch="study",
            subject_scope=scope,
            start_at=start_at,
            end_at=start_at + timedelta(minutes=30),
        )
    )


def test_interest_hierarchy_keeps_fine_grained_labels_when_activity_is_narrow(repo):
    now = datetime.utcnow()
    for scope in ["Rust async", "Ruby objects", "Python decorators", "C pointers", "Cybersecurity threat models"]:
        _session(repo, scope, scope, now - timedelta(days=2))

    result = InterestHierarchyService(repo, now=now).build(limit=6)
    labels = [node.label for node in result.nodes]

    assert set(labels) == {"Rust", "Ruby", "Python", "C", "Cybersecurity"}
    assert len(labels) == 5


def test_interest_hierarchy_preserves_breadth_despite_recent_streak(repo):
    now = datetime.utcnow()
    for index in range(4):
        _session(repo, f"ML {index}", "Transformer attention in machine learning", now - timedelta(hours=index))
    _session(repo, "Financial modeling", "Financial Modeling DCF", now - timedelta(days=10))
    _session(repo, "Music theory", "Music Theory harmony", now - timedelta(days=12))
    _session(repo, "Group theory", "Group Theory cosets", now - timedelta(days=14))

    result = InterestHierarchyService(repo, now=now).build(limit=6)
    labels = [node.label for node in result.nodes]

    assert "AI & ML" in labels
    assert "Financial Modeling" in labels
    assert "Music Theory" in labels
    assert "Group Theory" in labels
    assert labels.count("AI & ML") == 1


def test_interest_hierarchy_compresses_when_breadth_exceeds_visible_budget(repo):
    now = datetime.utcnow()
    for scope in [
        "Rust async",
        "Python decorators",
        "Cybersecurity threat models",
        "Financial Modeling DCF",
        "Music Theory harmony",
        "Group Theory cosets",
        "History sources",
        "Literature close reading",
        "Philosophy epistemology",
    ]:
        _session(repo, scope, scope, now - timedelta(days=3))

    result = InterestHierarchyService(repo, now=now).build(limit=6)
    labels = [node.label for node in result.nodes]

    assert len(labels) <= 6
    assert "Technology" in labels
    assert "Humanities" in labels


def test_interest_hierarchy_archive_shows_only_inactive_older_topics(repo):
    now = datetime.utcnow()
    _session(repo, "Python", "Python decorators", now - timedelta(days=2))
    _session(repo, "Rust", "Rust async", now - timedelta(days=45))

    active = InterestHierarchyService(repo, now=now).build(limit=6)
    archived = InterestHierarchyService(repo, now=now).build(archive=True, limit=6)

    assert "Python" in [node.label for node in active.nodes]
    assert "Rust" not in [node.label for node in active.nodes]
    assert "Rust" in [node.label for node in archived.nodes]


def test_interest_hierarchy_filters_general_learning_scaffold_and_uses_subjects(repo):
    now = datetime.utcnow()
    service = InterestHierarchyService(repo, now=now)
    service._collect_signals = lambda: [  # type: ignore[method-assign]
        InterestSignal("general learning", "source", now - timedelta(days=1), evidence={"reason": "source bundle"}),
        InterestSignal("cs pointers", "concept_confidence", now - timedelta(hours=1), evidence={"reason": "weak concept cluster"}),
        InterestSignal("cs recursion", "concept_confidence", now - timedelta(hours=1), evidence={"reason": "weak concept cluster"}),
        InterestSignal("Mastering Zeta Function Regularization", "goal", now - timedelta(days=2), evidence={"reason": "active goal"}),
        InterestSignal("Differentiable manifolds, tangent spaces, and vector fields.", "session", now - timedelta(days=3)),
        InterestSignal("Robust agents learn causal representations", "source", now - timedelta(days=1), evidence={"reason": "source bundle"}),
    ]

    result = service.build(limit=6, context_filter=ContextScopeFilter())
    labels = [node.label for node in result.nodes]

    assert "General Learning" not in labels
    assert "Computer Science" in labels or {"Pointers", "Recursion"}.issubset(labels)
    assert "Zeta Function Regularization" in labels
    assert "Differential Geometry" in labels
    assert "AI & ML" in labels
