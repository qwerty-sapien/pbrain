# MCP

ProductiveBrain exposes a stdio MCP server for local agent clients. The CLI remains the primary human interface.

## Start

```bash
pb mcp status
pb mcp doctor
pb mcp print-config --client claude-desktop --vault main
productivebrain-mcp --vault main
productivebrain-mcp --vault main --allow-writes
```

Read-only is the default. Write tools require `--allow-writes` or the pending confirmation flow exposed by `pb mcp pending`, `pb mcp confirm`, and `pb mcp reject`.

## Preferred Tool Shape

Agents should use semantic ProductiveBrain tools rather than shelling out through `pb_command`:

- goals
- plan day
- next action
- thought and todo capture
- feedback capture
- study, practise, and teach session start
- session pause, resume, finish, and status
- review day and week
- notes inbox and organization
- context packet build
- Anki candidate generation

`pb_command` remains available as a debug escape hatch.

## Current Boundaries

The MCP surface is broader than the stable human CLI because it includes agent-only utilities and compatibility paths. Public product docs should still teach `goal -> plan -> study/practise`, `pb do`, `pb next`, and optional `pb review` first.
