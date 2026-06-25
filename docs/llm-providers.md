# LLM Providers

LLM-backed commands draft goals, plans, study blocks, practice blocks, teach sessions, reviews, feedback patches, and recall cards. Many read-only or deterministic paths still work without a live model.

## Default Provider

`pb init` defaults to Gemini:

```bash
pb init --non-interactive --vault-name main --vault-path ~/brain --provider gemini --model gemini-3-flash-preview
```

Use `pb doctor` for local setup and `pb doctor --llm` for a live request.

## Locked Gemini Model IDs

These model IDs are known working and must not be guessed or renamed:

- Flash Lite: `gemini-3.1-flash-lite-preview`
- Flash: `gemini-3-flash-preview`
- Pro: `gemini-3.1-pro-preview`

## Configuration Surfaces

Preferred:

```bash
pb set status
pb set model fast gemini-3.1-flash-lite-preview
pb set model balanced gemini-3-flash-preview
pb set model pro gemini-3.1-pro-preview
pb set language auto
```

Compatibility/diagnostic:

```bash
pb model status
pb model list
pb model use gemini:gemini-3-flash-preview
pb init llm --provider openrouter --model openai/gpt-5 --api-key-env OPENROUTER_API_KEY --base-url https://openrouter.ai/api/v1
pb config show
```

Provider environment variables follow the config defaults:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`

Vertex AI can also be used through Google Cloud credentials when configured in the local environment.
