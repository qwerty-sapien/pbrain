"""Tests for agent color assignment."""
from pb.agents.colors import color_for_agent


def test_different_agents_get_different_colors():
    c1 = color_for_agent("accountability")
    c2 = color_for_agent("review")
    assert c1 != c2


def test_same_agent_gets_same_color():
    c1 = color_for_agent("study")
    c2 = color_for_agent("study")
    assert c1 == c2


def test_color_is_valid_rich_color():
    c = color_for_agent("todo")
    assert isinstance(c, str)
    assert len(c) > 0


def test_domain_agents_get_colors():
    c1 = color_for_agent("domain_german")
    c2 = color_for_agent("domain_coding")
    assert c1 != c2


def test_more_than_24_agents_recycles_oldest():
    from pb.agents.colors import _agent_last_seen, _agent_color_map
    _agent_color_map.clear()
    _agent_last_seen.clear()
    agents = [f"agent_{i}" for i in range(30)]
    colors = [color_for_agent(a) for a in agents]
    assert all(isinstance(c, str) for c in colors)
    assert len(set(colors)) <= 24
