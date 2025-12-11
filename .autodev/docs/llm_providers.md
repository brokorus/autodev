
# LLM Provider Abstraction Layer

The AutoDev system uses a provider-agnostic LLM routing layer to interact with different Large Language Models (LLMs). This layer provides a standardized interface for planning, coding, and troubleshooting tasks, while intelligently routing requests to the most suitable provider based on availability and configuration.

## Supported Providers

The LLM provider layer supports the following providers in order of priority:

1.  **Codex CLI**: The primary provider for all tasks. AutoDev will first attempt to use the Codex CLI for planning, coding, and troubleshooting. If the Codex CLI is not available or fails, it will fall back to the next provider.
2.  **Gemini CLI**: The fallback provider. If the Codex CLI is unavailable, AutoDev will use the Gemini CLI. The Gemini CLI is used in headless mode with JSON output.
3.  **LM Studio**: The final fallback provider. If both the Codex and Gemini CLIs are unavailable, AutoDev will attempt to use a local LM Studio instance.

## Provider Configuration

### Codex CLI

To use the Codex CLI, ensure that it is installed and available on your system's `PATH`. The `start-autodev.ps1` script will check for the presence of the `codex` or `codex.cmd` executable.

### Gemini CLI

To use the Gemini CLI, ensure that it is installed and available on your system's `PATH`. The `start-autodev.ps1` script will check for the presence of the `gemini` or `gemini.cmd` executable.

### LM Studio

To use LM Studio:

1. Install the desktop app from [https://lmstudio.ai](https://lmstudio.ai) and download a compatible model (the table below lists defaults AutoDev will try).
2. Start the local server from the LM Studio UI (`Server` tab → `Start Server`) and leave it running at `http://localhost:1234` (or set `LMSTUDIO_BASE_URL`).
3. Optional tuning: set `LMSTUDIO_CONTEXT_WINDOW` to match the context you loaded the model with, `LMSTUDIO_MAX_TOKENS` for generation length, and `LMSTUDIO_TIMEOUT_SECONDS` for slow hardware.

AutoDev now trims long prompts to fit the configured context window and retries once with an even smaller prompt if LM Studio reports a context overflow (the “Trying to keep the first X tokens when context overflows” error). Requests are capped with `max_tokens` to avoid running past the model’s context.

## Model Selection for LM Studio

AutoDev will automatically select the best model for each task based on the available VRAM on your system. The following models are supported:

*   **Planning**: `rnj-1`, `Hermes 2 Pro 7B`, `Mistral 7B`, `Nous-Hermes 2 7B`
*   **Coding**: `rnj-1`, `Qwen2.5 Coder 7B`, `StarCoder 7B`, `CodeGemma 7B`
*   **Troubleshooting**: `rnj-1`, `MixtralInstruct 8x7B`

If a model requires more VRAM than is available on your system, it will be automatically rejected.

## Usage

The `LLMProvider` class provides the following methods:

*   `plan(prompt: str) -> str`: Generates a development plan based on the provided prompt.
*   `code(prompt: str) -> str`: Generates code based on the provided prompt.
*   `troubleshoot(prompt: str) -> str`: Troubleshoots a failed deployment based on the provided prompt.

The `planner.py` and `orchestrator.py` scripts use the `LLMProvider` class to interact with the LLM providers.
