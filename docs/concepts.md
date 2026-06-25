# Concepts

ProductiveBrain is built around learning loops, not task lists.

## Canonical Loops

```text
goal -> plan -> study
goal -> plan -> practise
```

`study` is for conceptual understanding, Bloom-stage movement, and explanations. `practise` is for reps, constraints, drills, artifacts, and performance. `teach` makes you explain a concept so gaps become visible.

## Routing Surfaces

- `pb learn`: turn a topic into study or practise, with optional `--study` or `--practise`.
- `pb do`: route a free-text request to the best command.
- `pb next`: rank the next concrete action from local state.
- `pb review`: optional reflection and reporting.

## State

Markdown is the durable learner-facing record. SQLite is the fast runtime/index/cache layer. Generated notes are quarantined before merge so the vault stays auditable.

## Recall Scope

Recall should be scoped to a term, domain, note, session, goal, or due set. ProductiveBrain intentionally avoids vault-wide card spam.
