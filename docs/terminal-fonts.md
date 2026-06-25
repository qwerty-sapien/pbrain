<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- This file is part of ProductiveBrain. -->
<!-- Canonical source: https://github.com/qwerty-sapien/pbrain -->
<!-- Compliance fingerprint: PB-2026-A17F -->

# Terminal Fonts

ProductiveBrain is a terminal-first CLI, so font choice is controlled by your terminal emulator rather than by `pb` itself.

## Recommended setup

- Use **JetBrains Mono** as the default terminal font for the cleanest roadmap legends, aligned symbols, and compact preview blocks.
- Use **Fira Code** when you want a more stylized look with stronger ligatures for arrows and operator-heavy text.
- Keep ligatures optional. `pb` output is designed to stay readable with ligatures on or off.

The repository already includes local font files:

- `jetbrains/ttf/JetBrainsMono-Regular.ttf`
- `Fira_Code_v6/`

## What `pb` Will And Will Not Do

- `pb` will tune layout around monospace-friendly output such as compact DAG symbols, hanging legends, and short labeled bullets.
- `pb` will not try to install fonts, switch your terminal font, or depend on ligatures being available.
- Learning sessions should remain prompt-first; font setup must never trigger macOS admin, privacy, automation, or accessibility prompts by default.

## Practical tips

- Prefer a terminal window wide enough for 90-120 columns when reviewing roadmap previews.
- If your terminal supports per-profile fonts, use JetBrains Mono for day-to-day work and keep a Fira Code profile for more stylized sessions.
- If a preview feels cramped, raise `ui.max_content_width` with `pb config set ui.max_content_width 100`.
