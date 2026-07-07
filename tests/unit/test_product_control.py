from __future__ import annotations

from types import SimpleNamespace

from pb.core.product_control import ProductControlEngine


class _RepoStub:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.snapshots: dict[str, dict[str, object]] = {}

    def append_feedback_event(self, event: dict[str, object]) -> None:
        self.events.append(event)

    def list_feedback_events(self, scope_key=None, artifact_kind=None, artifact_id=None, kind=None, limit=100):
        rows = list(self.events)
        if scope_key is not None:
            rows = [row for row in rows if row.get("scope_key") == scope_key]
        if artifact_kind is not None:
            rows = [row for row in rows if row.get("artifact_kind") == artifact_kind]
        if artifact_id is not None:
            rows = [row for row in rows if row.get("artifact_id") == artifact_id]
        if kind is not None:
            rows = [row for row in rows if row.get("kind") == kind]
        return rows[-limit:]

    def get_control_state_snapshot(self, scope_key: str):
        return self.snapshots.get(scope_key)

    def save_control_state_snapshot(self, scope_key: str, scope: str, state: dict[str, object]) -> None:
        self.snapshots[scope_key] = state


def _runtime_available():
    return SimpleNamespace(
        health=lambda: SimpleNamespace(available=False),
    )


def _engine():
    return ProductControlEngine(repo=_RepoStub(), runtime=_runtime_available())


def test_foundational_feedback_escalates_local_then_rebase_then_rebuild():
    engine = _engine()

    _, state, decision_one = engine.record_feedback(
        artifact_kind="goal_roadmap",
        artifact_id="gr",
        label="Make this more foundational",
        free_text="make this more foundational",
        current_artifact="Palatini variation",
        domain="general relativity",
        target="Palatini variation",
    )
    assert decision_one.action == "local_refine"

    _, state, decision_two = engine.record_feedback(
        artifact_kind="goal_roadmap",
        artifact_id="gr",
        label="Still too advanced",
        free_text="start earlier and do prerequisites first",
        current_artifact="Palatini variation",
        domain="general relativity",
        target="Palatini variation",
    )
    assert decision_two.action == "branch_rebase"
    assert any("Linear algebra" in option.label or "Multivariable calculus" in option.label for option in decision_two.adaptive_options)

    _, state, decision_three = engine.record_feedback(
        artifact_kind="goal_roadmap",
        artifact_id="gr",
        label="I don't know the basics",
        free_text="I know Gaussian curvature but don't know what Riemann tensors are; can we work on the basics?",
        current_artifact="Palatini variation",
        domain="general relativity",
        target="Palatini variation",
    )
    assert decision_three.action == "global_rebuild"
    assert decision_three.requires_global_rebuild is True
    assert "Palatini variation" in decision_three.instruction
    assert "Riemann tensors" in " ".join(state.unknowns)


def test_natural_language_signals_drive_domain_specific_restart_options():
    engine = _engine()

    _, _, decision = engine.record_feedback(
        artifact_kind="goal_roadmap",
        artifact_id="gr2",
        label="Custom learner feedback",
        free_text="I know Gaussian curvature but don't know what Riemann tensors are; can we work on the basics?",
        current_artifact="Derive the Einstein field equations via the Palatini variation.",
        domain="differential geometry / general relativity",
        target="Palatini variation",
    )

    labels = [option.label for option in decision.adaptive_options]
    assert decision.action == "local_refine"
    assert any("Multivariable calculus" in label for label in labels)
    assert any("Linear algebra" in label for label in labels)
    assert any("Tensor" in label or "Manifolds" in label or "Covariant" in label for label in labels)


def test_repeated_concrete_and_advance_signals_change_strategy():
    engine = _engine()

    _, _, first = engine.record_feedback(
        artifact_kind="study_block",
        artifact_id="study",
        label="Too abstract",
        free_text="this is too abstract, give me drills",
        current_artifact="Abstract explanation",
        domain="geometry",
        target="curvature",
    )
    assert first.action == "concretize"

    _, _, second = engine.record_feedback(
        artifact_kind="study_block",
        artifact_id="study",
        label="Still abstract",
        free_text="give me drills",
        current_artifact="Abstract explanation",
        domain="geometry",
        target="curvature",
    )
    assert second.action == "drillify"

    _, _, third = engine.record_feedback(
        artifact_kind="teach_block",
        artifact_id="teach",
        label="Already know this",
        free_text="I already know this, skip ahead",
        current_artifact="basic explanation",
        domain="geometry",
        target="metric tensor",
    )
    assert third.action == "diagnostic_skip"


def test_default_revision_options_use_application_theory_and_time_labels():
    engine = _engine()

    labels = [option.label for option in engine.default_options(mode="study")]

    assert "Set time" in labels
    assert "Make this more application-based" in labels
    assert "Make this more theoretical" in labels
    assert "Make this more concrete" not in labels
    assert "Make this more conceptual" not in labels
