# Domain Context Architecture

`domain agents` means scoped learning contexts such as math, German, piano, Rust, biology, or communication. It does not mean git worktrees.

## Current Implementation

Domain scoping exists, but it is still heuristic:

- goals carry domains
- session/task text contributes to domain matching
- thought/context caches can be domain-aware
- MCP context packets can be built by domain
- feedback proposals can target workflow surfaces

Domain execution is not yet a strict isolation boundary across every command.

## Intended Split

```text
global planner/reviewer
  -> choose direction, time allocation, and cross-domain next actions

domain context
  -> goals, recent sessions, weak areas, recall items, notes, feedback

private agent profile
  -> future auditable teaching preferences and recurring friction
```

Private domain profiles are a design direction, not a finished user-facing feature.

## Context Packet Shape

A useful domain packet should include:

- active goals
- recent study/practise/teach sessions
- high-salience weak areas
- recurring mistakes
- pending recall/review items
- active todos that clearly belong to the domain
- useful notes and source context
- workflow preferences and feedback proposals
- omitted-context summary

## Safety Rules

- Domain context must not override the user's explicit intent.
- Private/profile memory must be inspectable and resettable before it becomes influential.
- Feedback should be accepted/rejected explicitly where possible.
- Old assumptions should decay or be reviewable.
- Domain packets should prefer evidence over vibes.

## Near-Term Work

1. Strengthen domain links from finish evidence into recall and retry items.
2. Make domain context packets less thought-centric and more session/evidence-aware.
3. Add clearer export/reset controls before richer private profiles.
4. Keep user-facing artifacts in Markdown wherever they matter for trust.
