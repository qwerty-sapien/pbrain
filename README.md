# ProductiveBrain

ProductiveBrain is a terminal-based system for turning learning intent into study, deliberate practice, evidence, and recall.

The key long-running workflows are:

```text
goal -> plan -> study
goal -> plan -> practise
```

or you can just start a session with: `pb teach`, `pb study`, or `pb practice` (better for adhoc work)

`pb do` routes to the most suitable command, helps when unclear which command would be the best. `pb next` chooses the next concrete step from local state. `pb review` allows you to reflect and report for improving context -> giving you a better learning UX.

## What It Does Now

- Creates structured learning goals and short roadmaps with `pb goal add`.
- Drafts daily and weekly learning plans with `pb plan day` and `pb plan week`.
- Starts study, practise, and teach blocks from a topic or skill.
- Keeps session/evidence notes in Markdown and fast runtime state in SQLite.
- Generates scoped Anki candidates with `genanki` export support.
- Stores generated notes in a quarantine inbox before any merge into the vault.
- Manages named vault profiles for Obsidian-style Markdown vaults.
- Exposes an MCP server for agent clients while keeping the CLI as the primary interface.

## Install

Requires Python 3.11+.

From a checkout:

```bash
git clone https://github.com/qwerty-sapien/pbrain.git
cd pbrain
python3 -m pip install -e .
pb init
pb doctor
```

With `uv`:

```bash
uv tool install git+https://github.com/qwerty-sapien/pbrain.git
pb init
```

For an isolated first run or scripted smoke test, use an explicit config path:

```bash
tmp="$(mktemp -d)"
pb --config "$tmp/config.toml" init --non-interactive --vault-name demo --vault-path "$tmp/vault" --provider gemini --model gemini-3-flash-preview
pb --config "$tmp/config.toml" doctor
```

`pb init` is conservative when a config already exists. Use a separate `--config` file when testing replacement behavior.

## First Session

```bash
pb goal add "Understand Bayesian reasoning well enough to solve medical-test base-rate examples" --yes
pb plan day --quick --budget 90m --yes
pb do "I want to practise Bayes rule word problems with base rates"
pb practise "Bayes word problems" --duration 5m --drill "posterior odds" --cues "prior, likelihood" --steps --yes
pb finish "still mixing up P(A|B) and P(B|A)" --skip --yes
pb review day --skip
pb next
```

Use `pb study "Rust async cancellation" --duration 10m --understand --steps --yes` for conceptual work, and `pb teach "Bayes theorem" --understand --steps --yes` when you want the system to make you explain.

## Stable CLI Surface

- `pb` / `brain`: open the interactive learning shell.
- `pb init`, `pb doctor`, `pb update`, `pb reset`, `pb set`: setup and preferences.
- `pb goal`, `pb plan`: direction before execution.
- `pb learn`, `pb study`, `pb practise`, `pb practice`, `pb teach`: learning blocks.
- `pb do`, `pb next`: routing and next action.
- `pb pause`, `pb resume`, `pb finish`: session lifecycle.
- `pb thought`, `pb todo`, `pb notes`, `pb feedback`: capture and workflow guidance.
- `pb anki`: scoped recall candidates and export.
- `pb context`, `pb vault`, `pb mcp`: source context, vault profiles, and agent integration.

`pb model` and `pb config` still exist as compatibility/diagnostic surfaces. New preference docs use `pb set`.

## State Model

```text
Markdown vault
  goals, evidence notes, generated notes, reviews, feedback proposals

SQLite data dir
  sessions, tasks, plan blocks, reminders, Anki candidate state, indexes, caches

Quarantine inbox
  Learning/Inbox/pb/ by default; generated Markdown lands here before merging
```

This split is deliberate: learner-facing durable material stays readable and auditable in Markdown; SQLite handles fast local state.

## LLMs And Offline Behavior

The default provider is Gemini. The working Gemini model IDs are:

- `gemini-3.1-flash-lite-preview`
- `gemini-3-flash-preview`
- `gemini-3.1-pro-preview`

Many commands have deterministic fallbacks or useful read-only behavior without a live model. Use `pb doctor` for local setup and `pb doctor --llm` only when you want a live provider probe.

## Anki

Anki is optional. `pb anki generate` creates suggested cards, `pb anki accept` promotes candidates, and `pb anki export` packages accepted cards to `.apkg` or CSV. If AnkiConnect is offline, export still works through the package path.

## Additional Docs

- [Quickstart](docs/quickstart.md)
- [Examples](docs/examples.md)
- [Concepts](docs/concepts.md)
- [Command Contract](docs/COMMAND_CONTRACT.md)
- [Architecture](docs/architecture.md)
- [Vaults](docs/vaults.md)
- [Notes](docs/notes.md)
- [Anki](docs/anki.md)
- [LLM Providers](docs/llm-providers.md)
- [MCP](docs/mcp.md)
- [Product Control](docs/product-control.md)
- [Evidence Base](docs/evidence.md)
- [Development](docs/development.md)

## License

Copyright 2026 qwerty-sapien

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) for the full text.

**AGPL Section 13 note:** If this software is modified and run as a network service, you may be required to offer Corresponding Source to users interacting with it over the network.

Canonical source: https://github.com/qwerty-sapien/pbrain
