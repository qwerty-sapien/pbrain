# Memory And Attenuation Review

This is a current design review of how ProductiveBrain remembers learning context.

## What Exists

- Thoughts can be captured into Markdown under `Learning/Inbox/pb/thoughts`.
- Runtime metadata is mirrored into SQLite for retrieval and routing.
- Context files can be inspected, added, locked, unlocked, and used to route learning commands.
- Reviews and `pb next` read sessions, goals, tasks, Anki state, and context signals.
- Feedback proposals can shape later workflow behavior.

## Strengths

- Markdown remains inspectable.
- Capture is low ceremony.
- SQLite can rank and cache without hiding all durable learning evidence.
- Domain-aware context is possible today.
- `pb review day --skip` can stay deterministic and model-free.

## Weaknesses

- Domain scoping is still heuristic.
- Old but important misconceptions may be underweighted.
- Recent chatter can dominate durable evidence.
- Review due-ness and repeated failure are not yet first-class ranking signals everywhere.
- Todos and plan blocks are less auditable than Markdown evidence.

## Desired Retrieval Score

```text
memory_score =
  scope_gate
  * domain_relevance
  * active_goal_relevance
  * semantic_similarity
  * time_weight
  * salience
  * evidence_strength
  * recurring_error_boost
  * review_due_boost
  * user_feedback_weight
  * diversity_penalty
```

Several of these factors are still aspirational or partial.

## Policy

- Scope before ranking.
- Prefer active goals over generic recency.
- Resurface repeated confusion faster than generic notes.
- Carry durable preferences until contradicted.
- Keep generated memory inspectable.
- Do not let private memory silently override explicit commands.

## Useful Tests

- Piano context does not leak into a Rust packet.
- Repeated confusion grows retrieval strength.
- Old high-salience mistakes can still resurface.
- Feedback changes ranking in an auditable way.
- Deterministic context generation works without a model.
- Markdown and SQLite stay aligned after capture, finish, and review flows.
