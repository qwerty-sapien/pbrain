<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- This file is part of pbrain. -->
<!-- Canonical source: https://github.com/qwerty-sapien/pbrain -->
<!-- Compliance fingerprint: PB-2026-A17F -->

# Terminal Fonts

pbrain is a terminal-first CLI, so font choice is controlled by your terminal emulator rather than by `pb` itself.

## Recommended setup

- Use **Source Code Pro** as the default terminal font for the most readable learning sessions and preview panes.
- Set the terminal font size to at least **13 pt** and line spacing to about **1.5x** when your terminal profile supports it.
- Keep **JetBrains Mono** and **Fira Code** as alternatives when you want tighter alignment or stronger ligatures for arrows and operator-heavy text.
- Keep ligatures optional. `pb` output is designed to stay readable with ligatures on or off.

The repository does not vendor font binaries. `pb` does not automatically
install fonts from Google Fonts because terminal font selection belongs to the
operating system and terminal profile. Install them explicitly when you want the
recommended look:

- Source Code Pro: `https://fonts.google.com/specimen/Source+Code+Pro`
- JetBrains Mono: `https://fonts.google.com/specimen/JetBrains+Mono`
- Fira Code: `https://fonts.google.com/specimen/Fira+Code`

Local font downloads can live under ignored checkout folders such as
`Source_Code_Pro/`, `jetbrains/`, or `Fira_Code_v6/` if you need to keep copies
near the project.

## What `pb` Will And Will Not Do

- `pb` will tune layout around monospace-friendly output such as compact DAG symbols, hanging legends, short labeled bullets, and chunked preview paragraphs.
- `pb` will not try to install fonts, switch your terminal font, or depend on ligatures being available.
- Learning sessions should remain prompt-first; font setup must never trigger macOS admin, privacy, automation, or accessibility prompts by default.

## Practical tips

- Prefer a terminal window wide enough for 90-120 columns when reviewing roadmap previews.
- If your terminal supports per-profile fonts, use Source Code Pro for study sessions, JetBrains Mono for tighter dashboards, and Fira Code for more stylized sessions.
- If a preview feels cramped, raise `ui.max_content_width` with `pb config set ui.max_content_width 100`.
