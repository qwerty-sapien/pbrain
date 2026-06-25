# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Catppuccin Mocha theme -- default preset.

11 semantic roles mapped to Rich style strings.
Dot-notation names (value.high, table.border) work as both
Rich Theme keys and markup tags: [value.high]text[/value.high].
"""

CATPPUCCIN: dict[str, str] = {
    "header":       "bold bright_blue",
    "subheader":    "bold",
    "dim":          "bright_black",
    "info":         "bright_cyan",
    "success":      "green",
    "warn":         "yellow",
    "error":        "bold red",
    "command":      "bold bright_cyan",
    "path":         "cyan",
    "duration":     "bold yellow",
    "math":         "bold magenta",
    "branch.study": "bold bright_blue",
    "branch.practise": "bold green",
    "value.high":   "red",
    "value.med":    "yellow",
    "value.low":    "green",
    "table.header": "bold",
    "table.border": "bright_black",
    "panel.border": "bright_black",
    "section.rule": "bright_black",
    "graph.edge":   "bright_black",
    "legend.heading": "bold white",
    "legend.text":  "white",
    "roadmap.bracket": "bright_black",
    "roadmap.title": "bold white",
    "roadmap.bullet": "bright_black",
    "roadmap.label": "bold bright_cyan",
    "roadmap.meta":  "white",
    "roadmap.check": "bold bright_white",
    "plan.bracket": "bright_black",
    "plan.title": "bold white",
    "plan.meta": "yellow",
    "plan.bullet": "bright_black",
    "plan.label": "bold bright_cyan",
    "plan.detail": "white",
    "step.bracket": "bright_black",
    "step.title": "bold white",
    "step.bullet": "bright_black",
    "step.label": "bold bright_cyan",
    "step.detail": "white",
    "step.check": "bold bright_white",
}
