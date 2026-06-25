# Examples

These are realistic command patterns from current `pb` behavior and the `Desktop/pb_runs` evaluation logs. Outputs are shortened, but the command shapes are live.

## Route Messy Intent

```text
> pb do "I want to practise Bayes rule word problems with base rates"
Do
1. Practise Bayes rule word problems with base rates
   Run: pb practise 'Bayes rule word problems with base rates'
   Why: The request asks for reps or drills, so practise is the right first move.
2. Study vocab
   Run: pb study vocab
   Why: Work through vocabulary in the study flow.
```

Use `pb do` when the sentence in your head is not yet a command.

## Start Conceptual Study

```text
> pb study "Rust async cancellation" --duration 10m --understand --steps --yes
Study Block Draft
Title: Understanding Rust Async Cancellation
Duration: 10 min
Scope:
- Future dropping
- poll state cleanup
- cancellation impact on async tasks
Steps:
[1] The Drop Trait and Futures
[2] Resource Cleanup
[3] Cancellation Safety
[4] The Select Macro Trap
```

Use `--understand`, `--apply`, `--evaluate`, or `--create` when you want to steer Bloom stage explicitly.

## Drill A Skill

```text
> pb practise "Bayes word problems" --duration 5m --drill "posterior odds" --cues "prior, likelihood" --steps --yes
No study history for 'Bayes word problems' (confidence: none). Consider `pb study Bayes word problems` first.
Practise Block Draft
Title: Bayes Posterior Odds Drill
Scope: Calculating posterior odds using prior odds and likelihood ratio
Duration: 5 min
Drill: posterior odds
Success: The calculated posterior odds match the expected ratio.
Started: Calculate Posterior Odds (5m)
```

The warning is intentional: practise is allowed, but `pb` reminds you when reps may be premature.

## Close A Session Without A Debrief

```text
> pb finish "done one odds-form drill; still shaky on denominator setup" --skip --yes
Finished: Calculate Posterior Odds
Evidence created:
  - evidence/bayes-word-problems/YYYY-MM-DD-calculate-posterior-odds.md
```

`--skip` writes lightweight evidence without asking the LLM to assess the session. `--debrief` is opt-in.

## Run A Deterministic Review

```text
> pb review day --skip
Using deterministic daily review; no LLM request will be made.
# Daily Review
- Study sessions: 1
- Practise sessions: 1
- Suggested Anki candidates: 0
- Accepted/edited Anki candidates awaiting export: 0
```

Review is for reflection and reporting. It is not required before `pb next`.

## Generate Recall

```text
> pb anki generate "Bayes theorem" --model flash-lite
Generating... (gemini-3.1-flash-lite-preview)
Generated 5 cards
Review with 'pb anki list --suggested' or export with 'pb anki export'

> pb anki pending
Suggested: 5
Export ready: 0

> pb anki accept
Accepted Bayes theorem-auto-0.

> pb anki export
Packaged 1 cards into export-YYYYMMDD-HHMMSS.apkg
```

## Work With Source Context

```text
> pb context inspect notes/bayes.md
Status: ok
Scope boundary: Use only the uploaded source material from bayes.md.

> pb context add notes/bayes.md
Stored source: ... -> vault://sources/.../bayes.md

> pb context lock
Locked context: general learning

> pb context status
Locked: general learning
Source refs: 1
```

When context is locked, `pb` blocks accidental source drift:

```text
Context is locked: general learning
Run `pb context unlock` before adding another source, or pass --force.
```

## Ask For The Next Action

```text
> pb next
Today's best next action

  1. Review suggested recall cards
     because You have suggested cards waiting.
  2. Clarify your learning goal
     because Direction should come before more study blocks.
```

`pb next` reads current state. It may recommend finishing an active session, reviewing suggested cards, clarifying a goal, shaping a plan, or starting a study/practise block.

## Capture Preferences

```bash
pb feedback study "Use concrete reps first; avoid generic motivational copy."
pb feedback general "Explanations were too abstract and used undefined jargon."
```

Feedback becomes auditable preference/proposal state instead of disappearing into a prompt.
