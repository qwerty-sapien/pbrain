# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Learning-scoped preference and workflow patch proposals."""

from __future__ import annotations

import re
from pathlib import Path

from pb.core.clock import utc_now
from pb.core.feedback_profile import feedback_profile_dir, normalize_feedback_scope
from pb.core.graph_writer import make_slug
from pb.llm.drafts import FeedbackProposalDraft
from pb.storage.config import update_preferences


class FeedbackProposalService:
    """Turn blunt product feedback into previewable learning-scoped patches."""

    def looks_like_feedback(self, text: str) -> bool:
        lowered = (text or "").lower()
        cues = (
            "useless",
            "broken",
            "mechanical",
            "generic",
            "flatter",
            "flattery",
            "gpt oss",
            "120b",
            "study partner",
            "tool",
            "question",
        )
        return any(cue in lowered for cue in cues)

    def generate_proposal(self, text: str, *, scope: str = "general") -> FeedbackProposalDraft:
        lowered = (text or "").lower()
        preference_patches: dict[str, object] = {}
        workflow_patches: list[str] = []

        if any(token in lowered for token in ("mechanical", "generic", "hardcoded")):
            preference_patches["prefer_contextual_questions"] = True
            preference_patches["avoid_generic_clarifiers"] = True
            workflow_patches.append("clarify-learning-intent")
        if any(token in lowered for token in ("never flatter", "flatter", "flattery", "empty praise")):
            preference_patches["coach_tone"] = "frank_no_flattery"
            workflow_patches.extend(["run-study-session", "run-practise-session"])
        if any(token in lowered for token in ("gpt oss", "120b", "oss_api_key", "simple naming")):
            preference_patches["simple_inference_model_role"] = "fast_inference"
            preference_patches["naming_strategy"] = "llm_generated"
            workflow_patches.append("generate-learning-names")
        if any(token in lowered for token in ("study partner", "proper study partner", "agentic", "chat")):
            preference_patches["study_session_mode"] = "agentic_partner"
            workflow_patches.extend(["run-study-session", "run-practise-session"])
        if "questions" in lowered and "context" in lowered:
            preference_patches["prefer_contextual_questions"] = True
        if "recover" in lowered or "discipline" in lowered:
            workflow_patches.append("finish-session")

        if not preference_patches and not workflow_patches:
            workflow_patches.append(f"review-{normalize_feedback_scope(scope)}-workflow")

        summary = (
            "Proposed learning-system patches from feedback: "
            + ", ".join(sorted(preference_patches) + workflow_patches[:2])
        )
        return FeedbackProposalDraft(
            summary=summary,
            preference_patches=preference_patches,
            workflow_patches=sorted(dict.fromkeys(workflow_patches)),
        )

    def write_proposal(
        self,
        vault_path: Path,
        text: str,
        proposal: FeedbackProposalDraft,
        *,
        scope: str = "general",
    ) -> Path:
        proposal_dir = feedback_profile_dir(vault_path) / "proposals"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        slug = make_slug(text[:60] or "feedback-proposal")
        path = proposal_dir / f"{utc_now().strftime('%Y-%m-%d')}-{slug}.md"
        lines = [
            "---",
            "type: feedback_proposal",
            f"scope: {normalize_feedback_scope(scope)}",
            f"updated: {utc_now().strftime('%Y-%m-%d')}",
            "---",
            "",
            "# Feedback Proposal",
            "",
            "## Feedback",
            "",
            text.strip() or "_No feedback text captured._",
            "",
            "## Summary",
            "",
            proposal.summary or "_No proposal summary._",
            "",
            "## Preference Patches",
            "",
        ]
        if proposal.preference_patches:
            for key, value in proposal.preference_patches.items():
                lines.append(f"- `{key}` = `{value}`")
        else:
            lines.append("- _No preference patch proposed._")
        lines.extend(["", "## Workflow Patches", ""])
        if proposal.workflow_patches:
            for item in proposal.workflow_patches:
                lines.append(f"- `{item}`")
        else:
            lines.append("- _No workflow patch proposed._")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def apply_proposal(self, proposal: FeedbackProposalDraft, *, config_path: Path | None = None) -> None:
        if proposal.preference_patches:
            update_preferences(proposal.preference_patches, path=config_path)
