# Quickstart

ProductiveBrain works best when you give it a learning target, let it shape one block, then close the loop with a finish note.

## Install And Initialize

```bash
git clone https://github.com/qwerty-sapien/pbrain.git
cd pbrain
python3 -m pip install -e .
pb init
pb doctor
```

For a disposable smoke test:

```bash
tmp="$(mktemp -d)"
pb --config "$tmp/config.toml" init --non-interactive --vault-name demo --vault-path "$tmp/vault" --provider gemini --model gemini-3-flash-preview
pb --config "$tmp/config.toml" doctor
```

## The Small Loop

```bash
pb goal add "Understand Bayesian reasoning well enough to solve medical-test base-rate examples" --yes
pb plan day --quick --budget 90m --yes
pb do "I want to practise Bayes rule word problems with base rates"
pb practise "Bayes word problems" --duration 5m --drill "posterior odds" --cues "prior, likelihood" --steps --yes
pb finish "I still confuse P(A|B) and P(B|A)" --skip --yes
pb review day --skip
pb next
```

Use `study` for concepts:

```bash
pb study "Rust async cancellation" --duration 10m --understand --steps --yes
```

Use `teach` when you want to test whether you can explain:

```bash
pb teach "Bayes theorem" --understand --steps --yes
```

Use `anki` when you want scoped recall:

```bash
pb anki generate "Bayes theorem" --model flash-lite
pb anki pending
pb anki accept
pb anki export
```

Generated Markdown goes to `Learning/Inbox/pb/` by default. Check it with `pb notes inbox`.

For more grounded transcripts, see [Examples](examples.md).
