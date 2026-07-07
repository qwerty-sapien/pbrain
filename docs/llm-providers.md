# LLM Providers

LLM-backed commands draft goals, plans, study blocks, practice blocks, teach sessions, reviews, feedback patches, and recall cards. Many read-only or deterministic paths still work without a live model.

## Provider Model

pbrain stores model choices as `provider:model` bindings. The same runtime path
handles Gemini, OpenAI-compatible providers, OpenRouter, and Claude through
Anthropic. A bare model name uses the configured default provider.

`pb init` still defaults to Gemini because those model IDs are the locked known
working defaults:

```bash
pb init --non-interactive --vault-name main --vault-path ~/brain --provider gemini --model gemini-3-flash-preview
```

OpenAI and Claude are first-class provider choices:

```bash
pb init --non-interactive --vault-name main --vault-path ~/brain --provider openai --model gpt-5-mini
pb init --non-interactive --vault-name main --vault-path ~/brain --provider claude --model claude-sonnet-4-5
```

Use `pb doctor` for local setup and `pb doctor --llm` for a live request.

## Locked Gemini Model IDs

These model IDs are known working and must not be guessed or renamed:

- Flash Lite: `gemini-3.1-flash-lite-preview`
- Flash: `gemini-3-flash-preview`
- Pro: `gemini-3.1-pro-preview`

## Configuration Surfaces

Fast inference roles default to the lightweight model for the active provider:

- Gemini: `gemini-3.1-flash-lite-preview`
- OpenAI: `gpt-5.4-nano`
- Anthropic/Claude: `claude-haiku-4-5-20251001`

Preferred:

```bash
pb set status
pb set model fast <provider:model>
pb set model balanced <provider:model>
pb set model pro <provider:model>
pb set language auto
```

Compatibility/diagnostic:

```bash
pb model status
pb model list
pb model use gemini:gemini-3-flash-preview
pb model use openai:gpt-5-mini
pb model use claude:claude-sonnet-4-5
pb init llm --provider openrouter --model openai/gpt-5 --api-key-env OPENROUTER_API_KEY --base-url https://openrouter.ai/api/v1
pb config show
```

Provider environment variables follow the config defaults:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`

Gemini support uses the optional `gemini` extra. OpenAI-compatible and Claude
requests use the standard library HTTP path and do not add SDK dependencies.
Vertex AI can also be used through Google Cloud credentials when configured in
the local environment.
