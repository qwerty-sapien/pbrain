# UX Harness Notes

This file describes how to run isolated ProductiveBrain UX probes. It is for development, not end-user setup.

## Isolation Strategy

Each probe should use:

- a temporary vault directory
- a temporary `config.toml`
- a temporary SQLite/data directory
- isolated XDG paths when subprocesses are involved
- explicit `--config /path/to/config.toml`
- `PRODUCTIVEBRAIN_AUTO_YES=1` only when the scenario should auto-accept

Useful first-run command:

```bash
tmp="$(mktemp -d)"
uv run pb --config "$tmp/config.toml" init --non-interactive --vault-name main --vault-path "$tmp/vault" --provider gemini --model gemini-3-flash-preview
```

## High-Value Smoke Commands

```bash
uv run pb --help
uv run pb --config "$tmp/config.toml" doctor --json
uv run pb --config "$tmp/config.toml" goal add "Understand Bayes rule base-rate examples" --yes
uv run pb --config "$tmp/config.toml" plan day --quick --budget 90m --yes
uv run pb --config "$tmp/config.toml" do "I want to practise Bayes rule word problems with base rates"
uv run pb --config "$tmp/config.toml" study "Rust async cancellation" --duration 10m --understand --steps --yes
uv run pb --config "$tmp/config.toml" finish --skip --yes
uv run pb --config "$tmp/config.toml" review day --skip
uv run pb --config "$tmp/config.toml" anki generate "Bayes theorem" --model flash-lite
uv run pb --config "$tmp/config.toml" anki pending
```

## Interactive Testing

Use a PTY for:

- bare `pb` shell
- `brain` alias shell
- active study/practise/teach sessions
- pickers such as resume without an ID

`PRODUCTIVEBRAIN_SHELL_TEST_MODE=1` can simplify shell tests, but it should not be treated as the production UX.

## What To Watch

- no traceback or raw JSON in normal CLI output
- no writes outside the isolated vault/data/config paths
- no macOS admin/privacy/automation prompts
- help works without bootstrap
- `pb do` respects explicit study vs practise intent
- `pb finish --skip --yes` creates evidence
- `pb review day --skip` stays deterministic
- MCP starts and lists semantic tools
