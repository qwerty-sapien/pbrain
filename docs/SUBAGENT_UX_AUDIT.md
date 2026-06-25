# Historical Subagent UX Audit

This document preserves an older simulated UX audit. It is not the live CLI contract.

## Durable Findings

The product is strongest when an agent or user stays inside:

```text
goal -> plan -> study/practise/teach -> finish -> recall/review -> next
```

The best-fit domains in the audit were conceptual and skill-building workflows:

- Rust/project learning
- math problem sets
- German speaking
- piano practice
- communication rehearsal
- memory resurfacing
- messy capture into day planning

## Gaps That Still Matter

- Domain scoping should become less heuristic.
- Finish evidence should feed retry/recall more directly.
- Teach mode must feel distinct from study by asking the learner to explain.
- Movement and embodied-practice domains need richer evidence than text alone.
- Sparse-state `pb next` should be helpful without pretending to know too much.

## MCP Lesson

Agent clients should prefer semantic tools such as goal, plan, next action, study/practise/teach start, session finish, review, notes, feedback, and context build. Raw `pb_command` should stay a fallback.

## Current Reference

Use:

- [COMMAND_CONTRACT.md](COMMAND_CONTRACT.md)
- [examples.md](examples.md)
- `pb --help`
