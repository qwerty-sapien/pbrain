# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Unit tests for compact symbolic roadmap DAG rendering."""

from __future__ import annotations

from pb.core.roadmap_dag import (
    build_symbolic_dag,
    render_legend_lines,
    render_mermaid_flowchart_lines,
    render_symbolic_node_lines,
    render_unicode_dependency_lines,
)
from pb.llm.drafts import GoalRoadmapNodeDraft


def _sample_nodes() -> list[GoalRoadmapNodeDraft]:
    return [
        GoalRoadmapNodeDraft(
            node_id="final-export",
            title="Finalize Mesh Export",
            prerequisites=["surface", "seam-check"],
        ),
        GoalRoadmapNodeDraft(
            node_id="surface",
            title="Surface Parameterization and Orientation Setup for Complex Meshes",
        ),
        GoalRoadmapNodeDraft(
            node_id="uv",
            title="Generate UV Atlas and Boundary Constraints",
            prerequisites=["surface"],
        ),
        GoalRoadmapNodeDraft(
            node_id="normals",
            title="Validate Normals, Seam Continuity, and Degenerate Faces",
            prerequisites=["surface"],
        ),
        GoalRoadmapNodeDraft(
            node_id="constraint-graph",
            title="Build Final Constraint Graph",
            prerequisites=["uv", "normals"],
        ),
        GoalRoadmapNodeDraft(
            node_id="seam-check",
            title="Run Quality Checks",
            prerequisites=["normals"],
        ),
    ]


def test_symbolic_dag_assigns_short_symbols_in_topological_order() -> None:
    dag = build_symbolic_dag(_sample_nodes())

    assert render_symbolic_node_lines(dag) == [
        ("surface", "A", "Surface Parameterization and Orientation Setup for Complex Meshes"),
        ("uv", "B", "Generate UV Atlas and Boundary Constraints"),
        ("normals", "C", "Validate Normals, Seam Continuity, and Degenerate Faces"),
        ("constraint-graph", "D", "Build Final Constraint Graph"),
        ("seam-check", "E", "Run Quality Checks"),
        ("final-export", "F", "Finalize Mesh Export"),
    ]


def test_unicode_dependency_lines_support_fan_out_and_fan_in() -> None:
    dag = build_symbolic_dag(_sample_nodes())
    lines = render_unicode_dependency_lines(dag)

    assert "A ─▶ B + C" in lines
    assert "B + C ─▶ D" in lines
    assert "C ─▶ E" in lines
    assert "A + E ─▶ F" in lines


def test_mermaid_flowchart_keeps_graph_nodes_symbol_only() -> None:
    dag = build_symbolic_dag(_sample_nodes())
    rendered = "\n".join(render_mermaid_flowchart_lines(dag))

    assert 'A["A"]' in rendered
    assert "A --> B" in rendered
    assert "Surface Parameterization" not in rendered
    assert "Finalize Mesh Export" not in rendered


def test_legend_wraps_with_hanging_indentation_and_no_ellipsis() -> None:
    dag = build_symbolic_dag(_sample_nodes())
    lines = render_legend_lines(dag, width=36)

    assert lines[0] == "Legend"
    assert "..." not in "\n".join(lines)
    assert any(line.startswith("A  Surface Parameterization") for line in lines)
    wrapped = [line for line in lines if line.startswith("   ") and "Meshes" in line]
    assert wrapped, lines
