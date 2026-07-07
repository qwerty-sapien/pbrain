# Historical UX Smoke Report

This file replaces a long dated smoke report with the parts that still help readers understand the repo. For live usage, start with [quickstart.md](quickstart.md) and [examples.md](examples.md).

## What The Later Runs Showed

The June 25, 2026 adversarial rerun under `Desktop/pb_runs/vertex_adversarial_loop_final_20260625_1405` completed three focused agents with:

- no timeouts
- no unexpected non-zero exits
- no serious or critical findings

Covered areas:

- learning UX: `do`, `study`, `practise`, `teach`, `finish`, `review`, `next`
- goals/feedback/doctor: `goal`, `plan`, `learn`, `practice`, `feedback`, `doctor`, `model`
- context/Anki/vault: `context`, `vault`, `anki`, `next`

## Good Examples From The Logs

`pb do` routed explicit practice intent correctly:

```text
pb do "I want to practise Bayes rule word problems with base rates"
1. Practise Bayes rule word problems with base rates
   Run: pb practise 'Bayes rule word problems with base rates'
```

`pb study` accepted trailing options and produced a scoped Rust cancellation block:

```bash
pb study "Rust async cancellation" --duration 10m --understand --steps --yes
```

`pb practise` produced a constrained Bayes drill:

```bash
pb practise "Bayes word problems" --duration 5m --drill "posterior odds" --cues "prior, likelihood" --steps --yes
```

`pb finish --skip --yes` created evidence without interactive debrief, and `pb review day --skip` produced deterministic review output.

`pb anki generate "Bayes theorem" --model flash-lite` generated suggested cards; `pb anki accept` and `pb anki export` packaged accepted cards.

## Minor Findings To Keep In Mind

- Some rich output can truncate in narrow/non-TTY captures.
- `finish --skip` evidence may show `Duration: 0 min` in short automated probes.
- `plan day` has shown inconsistent total-time display in a report footer.
- AnkiConnect warnings are accurate but can be more helpful.
- Context inference can be verbose when it routes back through learning commands.

These are polish issues, not changes to the core loop.
