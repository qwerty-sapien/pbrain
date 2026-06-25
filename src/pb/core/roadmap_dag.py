# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Compact dependency DAG helpers for roadmap-style learner artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import textwrap
from typing import Sequence

from pb.core.renderables import renderable_cli_text


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class SymbolicDagNode:
    """One roadmap node normalized for DAG rendering."""

    node_id: str
    symbol: str
    title: str
    prerequisites: tuple[str, ...]
    level: int
    topo_index: int


@dataclass(frozen=True)
class SymbolicDag:
    """A stable symbol map plus dependency metadata for rendering."""

    nodes: tuple[SymbolicDagNode, ...]
    children: dict[str, tuple[str, ...]]

    @property
    def symbol_by_id(self) -> dict[str, str]:
        return {node.node_id: node.symbol for node in self.nodes}

    @property
    def title_by_id(self) -> dict[str, str]:
        return {node.node_id: node.title for node in self.nodes}

    @property
    def nodes_by_id(self) -> dict[str, SymbolicDagNode]:
        return {node.node_id: node for node in self.nodes}


def build_symbolic_dag(nodes: Sequence[object]) -> SymbolicDag:
    """Return a stable symbol-based DAG view for roadmap-like nodes."""

    normalized: list[tuple[str, str, tuple[str, ...]]] = []
    known_ids = {
        str(getattr(node, "node_id", "")).strip()
        for node in nodes
        if str(getattr(node, "node_id", "")).strip()
    }
    for node in nodes:
        node_id = str(getattr(node, "node_id", "")).strip()
        if not node_id:
            continue
        title = renderable_cli_text(getattr(node, "title", "")).strip() or node_id
        prerequisites = tuple(
            dep
            for dep in getattr(node, "prerequisites", []) or []
            if dep in known_ids
        )
        normalized.append((node_id, title, prerequisites))

    order_index = {node_id: index for index, (node_id, _, _) in enumerate(normalized)}
    children: dict[str, list[str]] = {node_id: [] for node_id, _, _ in normalized}
    indegree: dict[str, int] = {}
    prerequisites_by_id: dict[str, tuple[str, ...]] = {}
    for node_id, _, prerequisites in normalized:
        prerequisites_by_id[node_id] = prerequisites
        indegree[node_id] = len(prerequisites)
        for dep in prerequisites:
            children.setdefault(dep, []).append(node_id)

    ready = sorted(
        [node_id for node_id, degree in indegree.items() if degree == 0],
        key=lambda item: order_index[item],
    )
    topo_order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        topo_order.append(node_id)
        for child in sorted(children.get(node_id, []), key=lambda item: order_index[item]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=lambda item: order_index[item])

    for node_id, _, _ in normalized:
        if node_id not in topo_order:
            topo_order.append(node_id)

    levels: dict[str, int] = {}
    for node_id in topo_order:
        prerequisites = prerequisites_by_id.get(node_id, ())
        levels[node_id] = 0 if not prerequisites else max(levels.get(dep, 0) + 1 for dep in prerequisites)

    title_by_id = {node_id: title for node_id, title, _ in normalized}
    symbolic_nodes = tuple(
        SymbolicDagNode(
            node_id=node_id,
            symbol=_symbol_for_index(index),
            title=title_by_id.get(node_id, node_id),
            prerequisites=prerequisites_by_id.get(node_id, ()),
            level=levels.get(node_id, 0),
            topo_index=index,
        )
        for index, node_id in enumerate(topo_order)
    )
    child_map = {
        node_id: tuple(sorted(children.get(node_id, []), key=lambda item: order_index[item]))
        for node_id in topo_order
    }
    return SymbolicDag(nodes=symbolic_nodes, children=child_map)


def render_unicode_dependency_lines(dag: SymbolicDag) -> list[str]:
    """Render a compact left-to-right Unicode dependency summary."""

    if not dag.nodes:
        return ["(no tasks)"]

    node_map = dag.nodes_by_id
    symbols = dag.symbol_by_id
    single_parent_groups: dict[str, list[str]] = {}
    isolated_nodes: list[str] = []
    lines: list[str] = []

    for node in dag.nodes:
        if len(node.prerequisites) == 1:
            single_parent_groups.setdefault(node.prerequisites[0], []).append(node.node_id)
        elif not node.prerequisites and not dag.children.get(node.node_id):
            isolated_nodes.append(node.node_id)

    emitted_isolated: set[str] = set()
    for node in dag.nodes:
        grouped_children = single_parent_groups.get(node.node_id, [])
        if grouped_children:
            child_symbols = " + ".join(symbols[child_id] for child_id in grouped_children)
            lines.append(f"{symbols[node.node_id]} ─▶ {child_symbols}")
        if len(node.prerequisites) > 1:
            parent_symbols = " + ".join(symbols[parent_id] for parent_id in node.prerequisites)
            lines.append(f"{parent_symbols} ─▶ {symbols[node.node_id]}")
        if node.node_id in isolated_nodes and node.node_id not in emitted_isolated:
            lines.append(symbols[node.node_id])
            emitted_isolated.add(node.node_id)

    return lines or [symbols[node.node_id] for node in dag.nodes]


def render_mermaid_flowchart_lines(dag: SymbolicDag) -> list[str]:
    """Render a Mermaid LR flowchart using short symbol labels only."""

    if not dag.nodes:
        return ["flowchart LR"]

    lines = ["flowchart LR"]
    for node in dag.nodes:
        lines.append(f'    {node.symbol}["{node.symbol}"]')
    for node in dag.nodes:
        for prereq in node.prerequisites:
            parent_symbol = dag.symbol_by_id.get(prereq)
            if parent_symbol:
                lines.append(f"    {parent_symbol} --> {node.symbol}")
    return lines


def render_legend_lines(dag: SymbolicDag, *, width: int) -> list[str]:
    """Render wrapped legend lines with hanging indentation."""

    lines: list[str] = ["Legend"]
    for node in dag.nodes:
        prefix = f"{node.symbol}  "
        available = max(12, width - _visible_width(prefix))
        wrapped = textwrap.wrap(
            node.title,
            width=available,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [node.title]
        lines.append(f"{prefix}{wrapped[0]}")
        indent = " " * _visible_width(prefix)
        for continuation in wrapped[1:]:
            lines.append(f"{indent}{continuation}")
    return lines


def render_symbolic_node_lines(dag: SymbolicDag) -> list[tuple[str, str, str]]:
    """Return `(node_id, symbol, title)` rows in stable topological order."""

    return [(node.node_id, node.symbol, node.title) for node in dag.nodes]


def _symbol_for_index(index: int) -> str:
    value = index + 1
    chars: list[str] = []
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _visible_width(text: str) -> int:
    return len(_ANSI_RE.sub("", text))
