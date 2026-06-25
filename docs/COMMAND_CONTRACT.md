# Command Contract

This is the current public CLI contract for ProductiveBrain. It is intentionally narrower than every module in the repository.

## Rules

- The product loop is `goal -> plan -> study/practise -> finish -> recall/evidence -> next`.
- `pb do` routes messy direct intent.
- `pb next` ranks the next forward action from local context.
- `pb review` is optional reflection/reporting and should not gate execution.
- Durable learner-facing state is Markdown where practical.
- SQLite holds fast runtime state, indexes, scheduling, and candidates.
- `brain` is a packaging alias for `pb`.
- `practise` is canonical; `practice` is an alias.

## Exit Codes

- `0`: success
- `40-49`: user/config/input/state conflict
- `50+`: system/database/runtime/provider problem

## Stable Commands

| Command | Purpose | Writes |
|---|---|---|
| `pb` | Open the interactive learning shell in a TTY | routed child commands only |
| `brain` | Alias for `pb` | same as `pb` |
| `pb init` | Create config and vault scaffold | config, vault folders |
| `pb doctor` | Check setup; `--llm` does a live provider probe | none |
| `pb set` | Set model tiers and language preferences | config |
| `pb goal add/list/refine/delete` | Create and manage learning goals | Markdown goals, SQLite goal rows |
| `pb plan day/week` | Draft executable learning plans | SQLite plan/task rows when accepted |
| `pb learn` | Route a topic to study or practise | usually task/session rows |
| `pb study` | Start conceptual study or scoped recall | task/session rows; recall notes when accepted |
| `pb practise` | Start deliberate practice | task/session rows |
| `pb practice` | Alias for `pb practise` | same as `practise` |
| `pb teach` | Start guided teach/explain mode | task/session rows and lesson state |
| `pb do` | Route free text to commands | none unless a child command runs |
| `pb next` | Rank the next forward action; `--schedule` can queue a reminder | optional reminder rows |
| `pb pause` | Pause active session or postpone a task | session/task rows |
| `pb resume` | Resume a paused task | session/task rows |
| `pb finish` | End active session and create evidence | session/task rows, Markdown evidence |
| `pb thought` | Capture a thought | Markdown inbox note |
| `pb todo` | Capture a follow-up task | SQLite task row |
| `pb notes inbox/organise` | Inspect or merge quarantined generated notes | Markdown moves when applied |
| `pb feedback` | Capture scoped workflow guidance or wrong-router feedback | Markdown proposals, feedback rows |
| `pb anki` | Generate, review, accept/reject, and export recall cards | SQLite candidates, `.apkg`/CSV exports |
| `pb context` | Inspect/add/list/lock source context and bundles | copied source records and context metadata |
| `pb vault` | Manage vault profiles and inspect graph health | config or graph cache depending on subcommand |
| `pb mcp` | Print config, status, diagnostics, and pending write confirmations | pending MCP confirmations when used |

## Important Options

- Root `--config FILE`: run against a specific config.
- Root `--vault NAME`: select a configured vault profile for the invocation.
- Root `--dryrun`: redirect writes to a temporary output area.
- Root `--yes`: accept confirmations for commands that read the root flag.
- `pb study --duration 10m --understand --steps --yes TOPIC`: steer a study block.
- `pb practise --duration 5m --drill DRILL --cues CUES --steps --yes SKILL`: steer a practice block.
- `pb finish --skip --yes`: finish with lightweight evidence and no LLM assessment.
- `pb finish --debrief`: opt into Socratic debrief.
- `pb review day --skip`: deterministic daily review, no LLM request.
- `pb plan day --quick --budget 90m --yes`: low-ceremony day plan.

## Compatibility Commands

These work but are not the preferred public teaching surface:

- `pb model status/list/use`: compatibility diagnostics; new docs prefer `pb set`.
- `pb config show/set/session`: direct configuration inspection and editing.
- `pb study start` and `pb practise start`: compatibility aliases for top-level `study` and `practise`.

Avoid documenting hidden or legacy task-manager surfaces as the core product.

## MCP Contract

The MCP server should prefer semantic tools over raw CLI execution:

- goal drafting/commit/list
- plan day
- next action
- study/practise/teach start
- session pause/resume/finish/status
- thought/todo/feedback capture
- review day/week
- notes inbox/organise
- context build and scoped packets
- Anki candidate generation

`pb_command` is a debug escape hatch, not the normal agent interface.
