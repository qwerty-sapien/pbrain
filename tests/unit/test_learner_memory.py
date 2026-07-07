from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pb.core.learner_memory import append_partner_session_memory
from pb.llm.runtime import GeneratedDraft
from pb.vault.lifecycle import read_frontmatter


class _RepoStub:
    def __init__(self) -> None:
        self.updated_sessions = []

    def update_session(self, session) -> None:
        self.updated_sessions.append(session)

    def list_feedback_events(self, **kwargs):
        return []


class _RuntimeStub:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.config = SimpleNamespace(model_roles=SimpleNamespace(fast_inference="gemini:test", default="gemini:test"))

    def health(self):
        return SimpleNamespace(available=True)

    def generate_draft(self, schema_cls, prompt, **kwargs):
        return GeneratedDraft(
            payload=self.payload,
            model="gemini:test",
            source_scope=kwargs.get("source_scope", "test"),
            prompt_template_version="test",
            raw_response="{}",
        )


def _write_transcript(data_dir: Path, session_id: str) -> None:
    transcript_dir = data_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{session_id}.json").write_text(
        json.dumps(
            [
                {"role": "assistant", "content": "State the metric tensor."},
                {"role": "user", "content": "I know Gaussian curvature but not the Riemann tensor."},
            ]
        ),
        encoding="utf-8",
    )


def _compact_payload():
    from pb.core.learner_memory import PartnerSessionCompactDraft

    return PartnerSessionCompactDraft(
        summary="Needs a restart from tensors before curvature tensors.",
        knowns=["Gaussian curvature"],
        unknowns=["Riemann tensor"],
        detected_gaps=["Tensor prerequisites"],
        recall_candidates=["Explain the Riemann tensor in words."],
        corrections=["Do not jump straight to Palatini variation."],
        next_drill="Work one tensor-index manipulation example.",
        next_action="Restart from tensors as multilinear maps.",
        control_signals=["needs_prerequisite", "foundational"],
        escalation_level=2,
    )


def test_append_partner_session_memory_updates_one_dossier_and_generated_names(tmp_path):
    repo = _RepoStub()
    runtime_ctx = SimpleNamespace(vault_path=tmp_path / "vault", data_dir=tmp_path / "data")
    runtime_ctx.vault_path.mkdir(parents=True, exist_ok=True)
    runtime_ctx.data_dir.mkdir(parents=True, exist_ok=True)
    runtime = _RuntimeStub(_compact_payload())
    task = SimpleNamespace(id="task-1", title="GR fundamentals")

    session_one = SimpleNamespace(
        id="session-1",
        branch="study",
        subject_scope="Riemann tensor",
        intended_outcome="Understand the Riemann tensor",
        observed_errors="Tensor prerequisites",
        next_adjustment="Restart from tensors",
        end_at=None,
        generated_names={
            "learning_partner_used": True,
            "learning_partner_note_path": str(runtime_ctx.vault_path / "partner" / "gr.md"),
            "control_state_snapshot": {"signal_counts": {"foundational": 2}},
            "learning_partner_closeout": {"summary": "Needs restart"},
        },
    )
    _write_transcript(Path(runtime_ctx.data_dir), session_one.id)

    path_one = append_partner_session_memory(
        runtime=runtime,
        runtime_ctx=runtime_ctx,
        repo=repo,
        session=session_one,
        task=task,
    )
    assert path_one is not None
    assert path_one.exists()
    frontmatter_one, body_one = read_frontmatter(path_one.read_text(encoding="utf-8"))
    assert frontmatter_one["type"] == "learning_dossier"
    assert "Riemann tensor" in frontmatter_one["aliases"]
    assert "Gaussian curvature" in frontmatter_one["strengths"]
    assert "Riemann tensor" in frontmatter_one["weaknesses"]
    assert "Work one tensor-index manipulation example." in frontmatter_one["next_drills"]
    assert "session-1" in body_one
    assert session_one.generated_names["learning_partner_dossier_path"] == str(path_one)
    assert session_one.generated_names["learning_partner_compact"]["unknowns"] == ["Riemann tensor"]

    session_two = SimpleNamespace(
        id="session-2",
        branch="study",
        subject_scope="Riemann tensor",
        intended_outcome="Understand the Riemann tensor",
        observed_errors="Tensor prerequisites",
        next_adjustment="Restart from tensors",
        end_at=None,
        generated_names={
            "learning_partner_used": True,
            "learning_partner_note_path": str(runtime_ctx.vault_path / "partner" / "gr.md"),
            "control_state_snapshot": {"signal_counts": {"foundational": 3}},
            "learning_partner_closeout": {"summary": "Still too advanced"},
        },
    )
    _write_transcript(Path(runtime_ctx.data_dir), session_two.id)

    path_two = append_partner_session_memory(
        runtime=runtime,
        runtime_ctx=runtime_ctx,
        repo=repo,
        session=session_two,
        task=task,
    )
    assert path_two == path_one
    frontmatter_two, body_two = read_frontmatter(path_two.read_text(encoding="utf-8"))
    session_ids = [item["id"] for item in frontmatter_two["session_refs"]]
    task_ids = [item["id"] for item in frontmatter_two["task_refs"]]
    assert session_ids == ["session-2", "session-1"]
    assert task_ids == ["task-1"]
    assert len(list((runtime_ctx.vault_path / "knowledge").rglob("*.md"))) == 1
    assert "session-2" in body_two


def test_append_partner_session_memory_skips_sessions_without_partner_usage(tmp_path):
    repo = _RepoStub()
    runtime_ctx = SimpleNamespace(vault_path=tmp_path / "vault", data_dir=tmp_path / "data")
    runtime_ctx.vault_path.mkdir(parents=True, exist_ok=True)
    runtime_ctx.data_dir.mkdir(parents=True, exist_ok=True)
    runtime = _RuntimeStub(_compact_payload())
    task = SimpleNamespace(id="task-1", title="GR fundamentals")
    session = SimpleNamespace(
        id="session-1",
        branch="study",
        subject_scope="Riemann tensor",
        intended_outcome="Understand the Riemann tensor",
        observed_errors="Tensor prerequisites",
        next_adjustment="Restart from tensors",
        end_at=None,
        generated_names={},
    )

    assert append_partner_session_memory(
        runtime=runtime,
        runtime_ctx=runtime_ctx,
        repo=repo,
        session=session,
        task=task,
    ) is None
    assert not list((runtime_ctx.vault_path / "knowledge").rglob("*.md"))
