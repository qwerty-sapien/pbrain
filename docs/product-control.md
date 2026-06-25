# Product Control

ProductiveBrain treats feedback as auditable product-control state, not as invisible prompt drift.

## Commands

```bash
pb feedback study "Use concrete examples before abstractions."
pb feedback practise "Start with reps faster; keep setup short."
pb feedback general "Explanations were too abstract and used undefined jargon."
pb feedback wrong "This should have routed to practise, not study."
pb feedback level "Bayes denominator setup" --level 2 --confidence 2 --evidence "I still mix up P(B|A) and P(A|B)"
```

`wrong` is for an active dispatch session. If there is no active dispatch session, `pb` tells you to use scoped feedback instead.

## Feedback Surfaces

Supported surfaces are learning-oriented:

- `learn`
- `study`
- `practise`
- `teach`
- `diagnostic`
- `anki`
- `goal`
- `plan`
- `review`
- `general`

Feedback is stored as proposals/workflow guidance instead of being silently treated as permanent truth.

## Adaptation Signals

The code understands signals such as:

- too advanced or too basic
- too abstract or too applied
- wrong scope
- needs prerequisite
- custom revision
- chat/accept/cancel

Repeated signals should change strategy: shift prerequisites, narrow scope, add concrete drills, or rebuild the block sequence. The intended behavior is not scalar difficulty tuning; it is learning-loop control.
