# Historical Audit: v0.2

This is a historical audit snapshot, not the current command reference. Use [COMMAND_CONTRACT.md](COMMAND_CONTRACT.md) and `pb --help` for the live surface.

## What Still Matters

- The public product should stay learning-first: `goal`, `plan`, `study`, `practise`, `teach`, `do`, `next`, `finish`, optional `review`.
- `pb do` and `pb next` should remain distinct. `do` routes the user's stated intent; `next` ranks the next locally grounded forward action.
- Markdown remains the durable learner-facing artifact where practical.
- SQLite remains the runtime/index/cache layer.
- Generated notes should be quarantined before vault merge.
- `brain` is only a packaging alias for `pb`.

## Current Status Since This Audit

The live top-level help now exposes the learning loop commands directly. `pb set` is the preferred model/language preference surface, while `pb model` remains a compatibility diagnostic command.

The audit's broad removal concerns are still directionally valid: legacy task, schema, feed, people, event, opportunity, and plugin-adjacent code may exist in-tree, but those areas are not the stable product story.

## Known Asymmetries

- Goals are mirrored to Markdown and SQLite.
- Thoughts and generated notes are Markdown-first.
- Sessions and plan blocks are primarily SQLite until finish/review writes evidence.
- Anki candidate state is SQLite-first, with `.apkg` or CSV output on export.
- Todos and plan blocks are still not fully Markdown-canonical.

## Follow-Up Themes

1. Keep domain memory and context packets auditable.
2. Make finish evidence richer without adding ceremony.
3. Keep MCP semantic tools ahead of raw `pb_command`.
4. Avoid documenting legacy modules as core product.
5. Prefer deterministic, prompt-first session behavior over blocking timers or hidden automation.
