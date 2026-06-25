# Architecture

ProductiveBrain is CLI-first and local-first.

```text
CLI
  pb goal / plan / learn / study / practise / teach / do / next / finish / review

Core services
  routing, planning, learning block drafts, session lifecycle, recall, feedback

Markdown vault
  goals, evidence notes, generated notes, reviews, feedback proposals

SQLite data dir
  sessions, tasks, plan blocks, reminders, Anki candidates, indexes, caches

MCP server
  semantic agent tools over the same local services
```

## Durable State

Markdown/frontmatter is canonical for learner-facing artifacts:

- goal notes and domain indexes
- generated session/evidence notes
- quarantine notes in `Learning/Inbox/pb/`
- daily/weekly review packets when saved
- feedback proposals and workflow guidance

SQLite is canonical for fast runtime state:

- active and historical sessions
- learning tasks and plan blocks
- reminders
- Anki candidate status
- indexes, caches, and routing metadata

This is not perfect symmetry. The current repo still has SQLite-only task/plan state, while durable learning evidence is written as Markdown at finish/review boundaries.

## Runtime Principles

- Help should not require config, vault, network, migrations, or a model.
- Study, practise, teach, and diagnostics should not trigger macOS admin/privacy/automation prompts by default.
- Starting a learning session should return to a usable terminal prompt instead of trapping the user in a blocking clock UI.
- Conversational learning flows should compact useful reasoning into the vault so later planning can use it.

## Boundaries

The public product surface is learning-first. Legacy modules for broader capture, schema, people/events/opportunities, or ingestion may still exist in-tree, but they are not the center of the stable CLI contract.
