# Development

Run tests with:

```bash
python3 -m pytest -q
```

Run the CLI from a checkout with:

```bash
uv run pb --help
uv run pb doctor
```

Use an isolated config/vault for manual probes:

```bash
tmp="$(mktemp -d)"
uv run pb --config "$tmp/config.toml" init --non-interactive --vault-name demo --vault-path "$tmp/vault" --provider gemini --model gemini-3-flash-preview
uv run pb --config "$tmp/config.toml" doctor
```

Keep the product learning-first:

- preserve the canonical loops `goal -> plan -> study` and `goal -> plan -> practise`
- keep `pb do` and `pb next` as routing surfaces
- keep `pb review` optional
- prefer active recall and scoped practice over passive capture
- keep Markdown learner-facing and SQLite runtime-oriented
- do not expand the stable surface into generic project management
- do not reintroduce voice, STT, or TTS as product features

When changing docs, verify examples against `uv run pb --help` and targeted subcommand help.
