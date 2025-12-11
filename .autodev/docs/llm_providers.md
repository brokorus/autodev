# LLM Provider Abstraction Layer

AutoDev routes planning, coding, and troubleshooting requests through a provider-agnostic layer. Providers are tried in priority order and fall back gracefully.

## Supported Providers (priority)
1) Codex CLI (preferred)
2) Gemini CLI
3) LM Studio local server

## Configuration
### Codex CLI
Ensure `codex`/`codex.cmd` is on PATH. `start-autodev.ps1` will install via `npm install -g codex` unless `-SkipCliInstall` is passed.

### Gemini CLI
Ensure `gemini`/`gemini.cmd` is on PATH. `start-autodev.ps1` will install via `npm install -g gemini` unless `-SkipCliInstall` is passed.

### LM Studio
1) Install LM Studio (https://lmstudio.ai) and download a compatible model.
2) Start the local server in the LM Studio UI (Server tab -> Start Server) and keep it running at `http://localhost:1234` (override with `LMSTUDIO_BASE_URL`).
3) Optional tuning env vars:
   - `LMSTUDIO_CONTEXT_WINDOW` (default 4096)
   - `LMSTUDIO_MAX_TOKENS` (default 1024)
   - `LMSTUDIO_TIMEOUT_SECONDS` (default 30)
   - `LMSTUDIO_HEALTH_TIMEOUT` (default 5) for the /v1/models health check
   - `LMSTUDIO_RETRIES` (default 2) to retry transient failures with backoff
   - Model overrides: `LMSTUDIO_MODEL` (global) or task-specific `LMSTUDIO_MODEL_PLAN`, `LMSTUDIO_MODEL_CODE`, `LMSTUDIO_MODEL_TROUBLESHOOT`

What AutoDev does for LM Studio:
- Health checks `/v1/models` (cached briefly) and intersects desired models with those the server reports.
- Trims prompts to stay within the configured context window; retries with a smaller window on context errors.
- Retries transient failures with backoff and logs actionable hints if the server is unreachable.

Default model preferences (filtered by available VRAM):
- Planning: `rnj-1`, `Hermes 2 Pro 7B`, `Mistral 7B`, `Nous-Hermes 2 7B`
- Coding: `rnj-1`, `Qwen2.5 Coder 7B`, `StarCoder 7B`, `CodeGemma 7B`
- Troubleshooting: `rnj-1`, `MixtralInstruct 8x7B`
