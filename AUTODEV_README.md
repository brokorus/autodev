# AutoDev System Map

This file is injected into all LLM prompts (planner, coder, troubleshooter). Keep it up to date when behavior changes.

## Entry Points
- `start-autodev.ps1` (repo root): Bootstrap. Ensures Codex/Gemini CLIs are on `PATH` (uses `.autodev/codex_path.txt` hint, optionally installs via `npm install -g codex|gemini` or env overrides), creates/uses `.autodev/venv`, installs `.autodev/requirements-autodev.txt`, then runs `.autodev/orchestrator.py`. Flags: `-SkipVenvCreate`, `-SkipCliInstall`.
- `.autodev/start-autodev.ps1`: Thin shim so running from `.autodev` still works; forwards to root script.

## Core Python Modules
- `.autodev/orchestrator.py`
  - `snapshot(base=BASE, max_chars=60_000)`: Text-only repo snapshot for planner. Skips noisy dirs (`.git`, `.autodev`, `node_modules`, venvs, caches, build artifacts), skips files >100KB, truncates at `max_chars`. Always injects `AUTODEV_README.md` first if present.
  - `load_autodev_readme()`: Reads `AUTODEV_README.md` (sanitized) for prompt inclusion.
  - `build_codex_cmd(last_message_path, sandbox="workspace-write", ask_for_approval="on-request")`: Builds Codex CLI invocation for coder path.
  - `codex(prompt, sandbox="workspace-write")`: Calls Codex CLI or returns fake output when `AUTODEV_FAKE_CODEX_OUTPUT` is set; logs to `.autodev/logs/codex_orchestrator_last.log` and `..._last_message.txt`.
  - `implement(task)`: Builds coder prompt (includes rules, AUTODEV_README, task/tests) and routes to `LLMProvider.code`.
  - `verify_deployment()`: Fetches latest GitHub Actions run via `gh.last_workflow_status`; requires `GITHUB_REPO` + token env vars.
  - `troubleshoot_deployment(deploy_result, task)`: Builds troubleshooting prompt (includes AUTODEV_README) and routes to `LLMProvider.troubleshoot`.
  - `main()`: Endless loop: gather snapshot, call planner (`create_backlog`), pick highest-priority actionable task, call `implement`, log outputs under `.autodev/logs/run_*.log`, optionally troubleshoot failed deployments, sleep between iterations. Timeouts configurable via env (`AUTODEV_CODER_TIMEOUT`).
- `.autodev/planner.py`
  - `build_planner_cmd(last_message_path)`: Codex CLI command builder for read-only planning with `--ask-for-approval never --sandbox read-only`.
  - `codex(prompt)`: Invoke Codex CLI or fake output for planning; logs to `.autodev/logs/codex_planner_last.log` and `..._last_message.txt`.
  - `load_rules()`: Reads `.autodev/prompts/rules.md`.
  - `load_autodev_readme()`: Reads `AUTODEV_README.md` for prompt context.
  - `read_ideas()`: Reads `ideas.md` from repo root if present.
  - `create_backlog(snapshot)`: Builds strict JSON-only planning prompt (includes rules, AUTODEV_README, ideas, snapshot), sends to `LLMProvider.plan`, returns parsed JSON or raises on failure. Env `AUTODEV_PLANNER_TIMEOUT` sets timeout for orchestrator loop (handled in orchestrator).
- `.autodev/llm_provider.py` (`LLMProvider`):
  - Public: `plan(prompt)`, `code(prompt)`, `troubleshoot(prompt)`; tries Gemini CLI first, then LM Studio HTTP (`http://localhost:1234/v1/chat/completions`) with task-type-specific model selection constrained by `max_vram_gb` (default 8).
  - Gemini detection: scans PATH for `gemini[.cmd/.exe]`; `_try_gemini` shells out with `--prompt "<prompt>" --output-format json`.
  - LM Studio selection: chooses from per-task model lists; simple VRAM filter.
- `.autodev/gh.py`: `last_workflow_status(repo: str)` calls GitHub API `/actions/runs?per_page=1`, requires token from `GITHUB_TOKEN|GH_TOKEN|GITHUB_PAT`.
- `.autodev/smoke.py`: CLI smoke checks. `planner` runs snapshot -> `create_backlog` and prints JSON. `orchestrator` calls Codex in read-only mode with a trivial JSON echo prompt.

## Tests
- `.autodev/tests/test_autodev.py`: Verifies planner command includes read-only/never approval flags and fake Codex output logging; checks `snapshot` ignores `.autodev` contents.
- `.autodev/tests/test_llm_provider.py`: (if present) covers provider behavior.

## Logs & Support Files
- Logs land in `.autodev/logs/` (`codex_*`, `run_*.log`, `iteration_*_error.log`).
- `codex_path.txt`: Optional hint for Codex CLI location added to PATH at startup.
- `requirements-autodev.txt`: Python deps for venv bootstrap (`requests`, `PyYAML`).

## Conventions
- Planner operates read-only; coder runs with workspace-write sandbox. All prompts instruct LLMs to emit complete files, not patches. Snapshots exclude large/noisy artifacts to keep context tight.
