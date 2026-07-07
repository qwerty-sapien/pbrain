"""Unique quaternary colors for agent labels in terminal output.

Each registered agent gets a distinct Rich color name from a 24-color
quaternary palette. When more than 24 agents exist, the color of the
agent inactive for the longest period is recycled.
"""
from __future__ import annotations

import time

_QUATERNARY_PALETTE: list[str] = [
    "bright_cyan",
    "bright_magenta",
    "bright_yellow",
    "bright_green",
    "dodger_blue2",
    "dark_orange",
    "medium_purple1",
    "spring_green2",
    "deep_pink2",
    "chartreuse2",
    "cornflower_blue",
    "indian_red1",
    "turquoise2",
    "orchid1",
    "gold1",
    "pale_green1",
    "hot_pink",
    "sky_blue1",
    "salmon1",
    "medium_spring_green",
    "plum2",
    "light_goldenrod2",
    "aquamarine1",
    "light_coral",
]

_PALETTE_SIZE = len(_QUATERNARY_PALETTE)

_agent_color_map: dict[str, str] = {}
_agent_last_seen: dict[str, float] = {}


def color_for_agent(agent_id: str) -> str:
    """Return a unique Rich color name for the given agent ID.

    Deterministic for known agents. If the palette is full, evicts the
    agent with the oldest last-seen timestamp and reassigns its color.
    """
    now = time.monotonic()
    _agent_last_seen[agent_id] = now

    if agent_id in _agent_color_map:
        return _agent_color_map[agent_id]

    used_colors = set(_agent_color_map.values())
    for color in _QUATERNARY_PALETTE:
        if color not in used_colors:
            _agent_color_map[agent_id] = color
            return color

    # Palette exhausted — evict the longest-inactive agent
    oldest_agent = min(
        (aid for aid in _agent_last_seen if aid != agent_id and aid in _agent_color_map),
        key=lambda aid: _agent_last_seen[aid],
    )
    recycled_color = _agent_color_map.pop(oldest_agent)
    _agent_last_seen.pop(oldest_agent, None)
    _agent_color_map[agent_id] = recycled_color
    return recycled_color
