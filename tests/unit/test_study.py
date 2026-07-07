"""Unit tests for the pb study command -- StudyPlanner allocation algorithm.

Tests cover: block allocation, stale-first ordering, budget splits,
domain filtering, stage skipping, mode fields, pb_command generation,
domain status, and threshold resolution.

Per plan: Do NOT test Rich output rendering (visual). Test logic only.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pb.core.learning_metadata import build_learning_task_description
from pb.core.models import GoalArc, Session, Task, utc_now


# ---------------------------------------------------------------------------
# Helpers to build mock DomainStatus objects
# ---------------------------------------------------------------------------


class _RepoStub:
    def __init__(self, *, goals=None, tracks=None, tasks=None, sessions_by_task=None):
        self._goals = goals or []
        self._tracks = tracks or []
        self._tasks = tasks or []
        self._sessions_by_task = sessions_by_task or {}

    def list_goal_arcs(self, status=None):
        return list(self._goals)

    def list_tracks(self, active_only=True):
        return list(self._tracks)

    def list_tasks(self):
        return list(self._tasks)

    def list_sessions_for_task(self, task_id):
        return list(self._sessions_by_task.get(task_id, []))


def _make_domain(
    name: str,
    stage_new: int = 0,
    stage_learning: int = 0,
    stage_learnt: int = 0,
    stage_stale: int = 0,
    decay_pressure: float = 0.0,
    is_stale: bool = False,
):
    """Build a minimal DomainStatus for testing without vault I/O."""
    from pb.study_service import DomainStatus

    d = DomainStatus(name=name, path=Path("/fake/vault/knowledge") / name)
    d.stage_new = stage_new
    d.stage_learning = stage_learning
    d.stage_learnt = stage_learnt
    d.stage_stale = stage_stale
    d.decay_pressure = decay_pressure
    d.is_stale = is_stale
    return d


def test_collect_broad_categories_keeps_topics_within_their_domain(monkeypatch):
    from pb.cli.commands import study as study_cmd

    ml_goal = GoalArc(
        id="goal-ml",
        title="Machine Learning (general)",
        domain="Machine Learning",
        execution_mode="study",
    )
    german_goal = GoalArc(
        id="goal-de",
        title="Get my German from A1 to B1",
        domain="German",
        execution_mode="study",
    )
    ml_task = Task(
        id="task-ml",
        title="Study: Gradient descent",
        description=build_learning_task_description(
            branch="study",
            scope="Gradient descent and loss surfaces",
            domain="Machine Learning",
        ),
        linked_goal_arc_ids=[ml_goal.id],
        work_type="study",
    )
    german_task = Task(
        id="task-de",
        title="Study: German cases",
        description=build_learning_task_description(
            branch="study",
            scope="German accusative and dative cases",
            domain="German",
        ),
        linked_goal_arc_ids=[german_goal.id],
        work_type="study",
    )
    repo = _RepoStub(
        goals=[ml_goal, german_goal],
        tasks=[ml_task, german_task],
        sessions_by_task={
            ml_task.id: [
                Session(task_id=ml_task.id, branch="study", subject_scope="ML optimization intuition", start_at=utc_now())
            ]
        },
    )

    monkeypatch.setattr(study_cmd, "_list_knowledge_domains", lambda: [])

    categories = study_cmd._collect_broad_categories(repo)

    assert "Machine Learning" in categories
    assert "Machine Learning (general)" in categories["Machine Learning"]
    assert "Gradient descent and loss surfaces" in categories["Machine Learning"]
    assert "ML optimization intuition" in categories["Machine Learning"]
    assert "Get my German from A1 to B1" not in categories["Machine Learning"]
    assert "German accusative and dative cases" not in categories["Machine Learning"]


def test_scope_bullets_render_bare_latex_before_markdown_preview():
    from pb.cli.commands import study as study_cmd

    lines = study_cmd._scope_bullets_with_freshness(
        r"homeomorphisms to open subsets of (\mathbb{R}^n)",
        vault_path=None,
    )

    assert lines == ["- **homeomorphisms to open subsets of (ℝⁿ)**"]
    assert r"\mathbb" not in lines[0]


# ---------------------------------------------------------------------------
# Test 1: stale domain gets first 15-min slot
# ---------------------------------------------------------------------------


def test_allocate_blocks_stale_domain_gets_first_slot():
    """Test 1: 60 min, 1 stale domain -> stale domain gets first 15-min slot."""
    from pb.study_service import allocate_blocks

    stale = _make_domain("deutsch", stage_stale=3, decay_pressure=5.0, is_stale=True)
    fresh = _make_domain("ml", stage_learning=2)

    blocks = allocate_blocks(60, [stale, fresh])

    assert len(blocks) > 0
    assert blocks[0].domain == "deutsch"
    assert blocks[0].minutes == 15
    assert blocks[0].stage == "#stale"


# ---------------------------------------------------------------------------
# Test 2: 40/35/25 split when no stale domains
# ---------------------------------------------------------------------------


def test_allocate_blocks_budget_split_no_stale():
    """Test 2: 60 min, no stale -> 40/35/25 split across learning/new/learnt."""
    from pb.study_service import allocate_blocks

    domains = [
        _make_domain("ml", stage_learning=3),
        _make_domain("piano", stage_new=2),
        _make_domain("spanish", stage_learnt=4),
    ]

    blocks = allocate_blocks(60, domains)

    learning_mins = sum(b.minutes for b in blocks if b.stage == "#learning")
    new_mins = sum(b.minutes for b in blocks if b.stage == "#new")
    learnt_mins = sum(b.minutes for b in blocks if b.stage == "#learnt")

    total = learning_mins + new_mins + learnt_mins
    assert total == 60

    # 40% learning = 24, 35% new = 21, 25% learnt = 15
    assert learning_mins == 24
    assert new_mins == 21
    assert learnt_mins == 15


# ---------------------------------------------------------------------------
# Test 3: proportionally smaller blocks for 30-min budget, stale-first
# ---------------------------------------------------------------------------


def test_allocate_blocks_30_min_budget_stale_first():
    """Test 3: 30 min -> proportionally smaller blocks, still stale-first."""
    from pb.study_service import allocate_blocks

    stale = _make_domain("languages", stage_stale=1, decay_pressure=3.0, is_stale=True)
    fresh = _make_domain("math", stage_learning=2)

    blocks = allocate_blocks(30, [stale, fresh])

    assert blocks[0].domain == "languages"
    assert blocks[0].stage == "#stale"
    # Stale block is 15 min max, leaving 15 for remaining
    assert blocks[0].minutes == 15
    # Total should not exceed budget
    assert sum(b.minutes for b in blocks) <= 30


# ---------------------------------------------------------------------------
# Test 4: 2 stale domains -> both get slots, highest decay pressure first
# ---------------------------------------------------------------------------


def test_allocate_blocks_two_stale_highest_decay_first():
    """Test 4: 2 stale domains -> both get slots, highest decay_pressure first."""
    from pb.study_service import allocate_blocks

    low_stale = _make_domain("piano", stage_stale=1, decay_pressure=2.0, is_stale=True)
    high_stale = _make_domain("japanese", stage_stale=2, decay_pressure=7.0, is_stale=True)

    blocks = allocate_blocks(60, [low_stale, high_stale])

    stale_blocks = [b for b in blocks if b.stage == "#stale"]
    assert len(stale_blocks) == 2
    assert stale_blocks[0].domain == "japanese"  # highest decay pressure first
    assert stale_blocks[1].domain == "piano"


# ---------------------------------------------------------------------------
# Test 5: domain filter -> only that domain's notes in plan
# ---------------------------------------------------------------------------


def test_allocate_blocks_domain_filter():
    """Test 5: domain filter -> only that domain's notes in plan."""
    from pb.study_service import allocate_blocks

    domains = [
        _make_domain("ml", stage_learning=3),
        _make_domain("piano", stage_new=2),
    ]

    blocks = allocate_blocks(60, domains, domain_filter="ml")

    assert all(b.domain == "ml" for b in blocks)


# ---------------------------------------------------------------------------
# Test 6: 0 notes in any stage -> skips that stage
# ---------------------------------------------------------------------------


def test_allocate_blocks_skips_empty_stages():
    """Test 6: 0 notes in any stage -> skips that stage's allocation."""
    from pb.study_service import allocate_blocks

    # Only learning notes; no new, no learnt
    domains = [_make_domain("math", stage_learning=5)]

    blocks = allocate_blocks(60, domains)

    stages = {b.stage for b in blocks}
    assert "#new" not in stages
    assert "#learnt" not in stages
    assert "#learning" in stages


# ---------------------------------------------------------------------------
# Test 7: Each block has mode field from expected set
# ---------------------------------------------------------------------------


def test_allocate_blocks_mode_field_valid():
    """Test 7: Each block has mode from: re-engage, consolidate, explore, review."""
    from pb.study_service import allocate_blocks

    domains = [
        _make_domain("ml", stage_learning=2, stage_new=3, stage_learnt=1,
                     stage_stale=1, decay_pressure=4.0, is_stale=True),
    ]

    blocks = allocate_blocks(60, domains)

    valid_modes = {"re-engage", "consolidate", "explore", "review"}
    for block in blocks:
        assert block.mode in valid_modes, f"Invalid mode: {block.mode!r}"


# ---------------------------------------------------------------------------
# Test 8: Each block has a pb_command field with valid command string
# ---------------------------------------------------------------------------


def test_allocate_blocks_pb_command_valid():
    """Test 8: Each block has a pb_command field containing valid command string."""
    from pb.study_service import allocate_blocks

    domains = [
        _make_domain("ml", stage_learning=2, stage_new=3, stage_learnt=1,
                     stage_stale=1, decay_pressure=4.0, is_stale=True),
    ]

    blocks = allocate_blocks(60, domains)

    for block in blocks:
        assert block.pb_command, f"Empty pb_command for block {block}"
        assert "pb " in block.pb_command, f"pb_command does not start with 'pb ': {block.pb_command!r}"


# ---------------------------------------------------------------------------
# Test 9: No block has passive mode or passive commands
# ---------------------------------------------------------------------------


def test_allocate_blocks_no_passive_commands():
    """Test 9: No block has mode 'passive' or contains 'read'/'cat' in pb_command."""
    from pb.study_service import allocate_blocks

    domains = [
        _make_domain("spanish", stage_new=5, stage_learning=3, stage_learnt=2,
                     stage_stale=1, decay_pressure=2.0, is_stale=True),
    ]

    blocks = allocate_blocks(120, domains)

    for block in blocks:
        assert block.mode != "passive", f"Found passive mode in block {block}"
        assert "read" not in block.pb_command.lower().split(), (
            f"'read' found as word in pb_command: {block.pb_command!r}"
        )
        assert "cat" not in block.pb_command.lower().split(), (
            f"'cat' found in pb_command: {block.pb_command!r}"
        )


# ---------------------------------------------------------------------------
# Test 10: get_domain_statuses returns correct stage counts per domain
# ---------------------------------------------------------------------------


def test_get_domain_statuses_stage_counts(tmp_path):
    """Test 10: get_domain_statuses returns correct stage counts per domain."""
    from pb.study_service import get_domain_statuses

    # Build minimal vault structure
    vault = tmp_path / "vault"
    knowledge = vault / "knowledge"
    domain_dir = knowledge / "python"
    domain_dir.mkdir(parents=True)

    # Create _state.md
    state_md = domain_dir / "_state.md"
    state_md.write_text("---\nlast_activity: 2026-04-01\n---\n\nState file.\n")

    # Create notes with different stages
    notes = [
        ("note-new.md", "#new"),
        ("note-learning.md", "#learning"),
        ("note-learnt.md", "#learnt"),
        ("note-learnt2.md", "#learnt"),
    ]
    for filename, stage in notes:
        (domain_dir / filename).write_text(
            f"---\nlearning_stage: \"{stage}\"\n---\n\nContent.\n"
        )

    # Mock check_staleness to return empty list (no stale for this test)
    # patch at the source module since study.py imports it inside the function
    with patch("pb.vault.lifecycle.check_staleness", return_value=[]):
        statuses = get_domain_statuses(vault, {"_default": 5})

    assert len(statuses) == 1
    s = statuses[0]
    assert s.name == "python"
    assert s.stage_new == 1
    assert s.stage_learning == 1
    assert s.stage_learnt == 2


# ---------------------------------------------------------------------------
# Test 11: _resolve_threshold exact, prefix, fallback
# ---------------------------------------------------------------------------


def test_resolve_threshold_exact_match():
    """Test 11a: exact match returns exact value."""
    from pb.study_service import _resolve_threshold

    thresholds = {"piano": 2, "languages": 3, "_default": 5}
    assert _resolve_threshold("piano", thresholds) == 2


def test_resolve_threshold_prefix_match():
    """Test 11b: prefix match (e.g., 'ml-notes' matches 'ml')."""
    from pb.study_service import _resolve_threshold

    thresholds = {"ml": 7, "_default": 5}
    assert _resolve_threshold("ml-notes", thresholds) == 7


def test_resolve_threshold_fallback():
    """Test 11c: no match -> _default fallback."""
    from pb.study_service import _resolve_threshold

    thresholds = {"piano": 2, "_default": 5}
    assert _resolve_threshold("deutsch", thresholds) == 5


def test_resolve_threshold_no_default():
    """Test 11d: no match and no _default -> hardcoded fallback of 5."""
    from pb.study_service import _resolve_threshold

    thresholds = {"piano": 2}
    assert _resolve_threshold("unknown", thresholds) == 5


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


def test_allocate_blocks_empty_domains():
    """Empty domain list returns empty blocks."""
    from pb.study_service import allocate_blocks

    blocks = allocate_blocks(60, [])
    assert blocks == []


def test_allocate_blocks_filter_no_match():
    """Domain filter with no matching domain returns empty."""
    from pb.study_service import allocate_blocks

    domains = [_make_domain("ml", stage_learning=3)]
    blocks = allocate_blocks(60, domains, domain_filter="nonexistent")
    assert blocks == []


def test_allocate_blocks_stale_blocks_have_re_engage_mode():
    """Stale blocks always have mode 're-engage'."""
    from pb.study_service import allocate_blocks

    stale = _make_domain("piano", stage_stale=2, decay_pressure=5.0, is_stale=True)
    blocks = allocate_blocks(60, [stale])

    stale_blocks = [b for b in blocks if b.stage == "#stale"]
    for b in stale_blocks:
        assert b.mode == "re-engage"


def test_allocate_blocks_learning_blocks_have_consolidate_mode():
    """#learning blocks use 'consolidate' mode."""
    from pb.study_service import allocate_blocks

    domains = [_make_domain("math", stage_learning=3)]
    blocks = allocate_blocks(60, domains)

    learning_blocks = [b for b in blocks if b.stage == "#learning"]
    assert len(learning_blocks) > 0
    for b in learning_blocks:
        assert b.mode == "consolidate"


def test_allocate_blocks_new_blocks_have_explore_mode():
    """#new blocks use 'explore' mode."""
    from pb.study_service import allocate_blocks

    domains = [_make_domain("physics", stage_new=4)]
    blocks = allocate_blocks(60, domains)

    new_blocks = [b for b in blocks if b.stage == "#new"]
    assert len(new_blocks) > 0
    for b in new_blocks:
        assert b.mode == "explore"


def test_allocate_blocks_learnt_blocks_have_review_mode():
    """#learnt blocks use 'review' mode."""
    from pb.study_service import allocate_blocks

    domains = [_make_domain("chemistry", stage_learnt=3)]
    blocks = allocate_blocks(60, domains)

    learnt_blocks = [b for b in blocks if b.stage == "#learnt"]
    assert len(learnt_blocks) > 0
    for b in learnt_blocks:
        assert b.mode == "review"


def test_get_pb_command_no_passive():
    """_get_pb_command never returns passive reading commands."""
    from pb.study_service import _get_pb_command

    modes = ["re-engage", "consolidate", "explore", "review"]
    for mode in modes:
        cmd = _get_pb_command(mode, "test-domain")
        assert "read" not in cmd.lower().split()
        assert "cat" not in cmd.lower().split()
        assert "pb " in cmd
