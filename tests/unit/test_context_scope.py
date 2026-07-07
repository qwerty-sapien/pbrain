from __future__ import annotations

from types import SimpleNamespace

from pb.core.context_scope import ContextScopeFilter


def test_context_scope_filter_excludes_unrelated_candidates_when_locked():
    scope = SimpleNamespace(
        locked=True,
        label="PC3213 rotational mechanics",
        source_bundle_id="bundle-1",
        source_refs=["vault://source/pc3213.pdf"],
        domain_id="physics",
        scope_boundary="Stay inside PC3213 rotations.",
    )
    filter_ = ContextScopeFilter.from_scope(scope)

    assert filter_.allows(text="Rotational inertia from PC3213", domain="physics")
    assert filter_.allows(text="anything", source_ref="vault://source/pc3213.pdf")
    assert not filter_.allows(text="Ruby metaprogramming", domain="technology")


def test_context_scope_filter_returns_exclusion_diagnostics():
    scope = SimpleNamespace(
        locked=True,
        label="music theory harmony",
        source_bundle_id="",
        source_refs=[],
        domain_id="music",
        scope_boundary="Stay inside harmony.",
    )
    filter_ = ContextScopeFilter.from_scope(scope)

    allowed, excluded = filter_.filter_items(
        [{"label": "Harmony cadence", "domain": "music"}, {"label": "Python asyncio", "domain": "tech"}],
        text=lambda item: item["label"],
        domain=lambda item: item["domain"],
    )

    assert [item["label"] for item in allowed] == ["Harmony cadence"]
    assert excluded == [{"reason": "excluded by context lock", "label": "Python asyncio"}]


def test_context_scope_filter_does_not_allow_unrelated_items_by_generic_words():
    scope = SimpleNamespace(
        locked=True,
        label="general learning",
        source_bundle_id="",
        source_refs=["vault://source/Robust_agents_learn_causa.pdf"],
        domain_id="",
        scope_boundary="Use only the uploaded source material from Robust_agents_learn_causa.pdf.",
    )
    filter_ = ContextScopeFilter.from_scope(scope)

    assert filter_.allows(text="Robust agents learn causal representations", source_ref="vault://source/Robust_agents_learn_causa.pdf")
    assert not filter_.allows(text="Definition of the Riemann Zeta function and the functional equation")
    assert not filter_.allows(text="general learning")
