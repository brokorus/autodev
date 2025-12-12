# AutoDev System Map

This file is injected into all LLM prompts (planner, coder, troubleshooter). Keep it up to date when behavior changes.

## Entry Points
- `autodev.ps1` (repo root): Bootstrap. Ensures Codex/Gemini CLIs are on `PATH` (uses `.autodev/codex_path.txt` hint, optionally installs via `npm install -g codex|gemini` or env overrides), creates/uses `.autodev/venv`, installs `.autodev/requirements-autodev.txt`, then runs `.autodev/orchestrator.py`. Flags: `-SkipVenvCreate`, `-SkipCliInstall`, `-NoInstall` (skip installing missing CLIs), `-InstallOnly` (ensure deps then exit), `-LocalOnly`/`-l` (force local providers / LM Studio only).

### Running it
- Run from repo root: `./autodev.ps1 -LocalOnly -NoInstall` (Windows PowerShell). Do **not** `cd autodev` before running; the script expects repo root paths.
- Ensure LM Studio has at least one small model pre-downloaded (e.g., `TinyLlama-1.1B`) to avoid first-run downloads when VRAM is tight.
- To cap VRAM use when detection overestimates, set `LMSTUDIO_VRAM_CAP_GB=<int>`; otherwise AutoDev now prefers the largest detected adapter VRAM to avoid under-reporting iGPU RAM.

## Product Story (do not prune)
- Directory: `autodev/story/` (copied alongside this file). Contains append-only `story.log.jsonl`, per-entry Markdown, and `index.md` digest for quick LLM consumption.
- Auto updates when planner runs, coder runs/tests, deployment troubleshooting happens, or the orchestrator throws exceptions. Entries carry tags/status/files when known.
- Goal: persistent narrative memory so LLMs know what was tried, what worked, and what to avoid repeating. Add new entries; never delete existing ones.
- Story IDs: each coder iteration gets a `STORY:<id>` marker—add a small inline code comment containing that marker near significant changes so later agents can find the rationale quickly.
- Context management: prompts use a concise HEADLINES + BIG PICTURE view from `StoryLog.context_view()` to keep context budgets small while retaining intent.
- Story Q&A: run `autodev.ps1 -Ask "How did the model management come to be?"` to get a concise narrative and improvement ideas for a specific feature/question.

## Core Python Modules
- `.autodev/orchestrator.py`
  - `LiveInputListener` + `interpret_live_commands`: background stdin listener; commands `replan`/`wrong direction` -> replan, `skip` -> next task, `stop|quit` -> halt; all live notes get injected into planner/coder prompts.
  - `snapshot(base=BASE, max_chars=60_000)`: Text-only repo snapshot for planner. Skips noisy dirs (`.git`, `.autodev`, `node_modules`, venvs, caches, build artifacts), skips files >100KB, truncates at `max_chars`. Always injects `AUTODEV_README.md` first if present.
  - `load_autodev_readme()`: Reads `AUTODEV_README.md` (sanitized) for prompt inclusion.
  - `build_codex_cmd(last_message_path, sandbox="workspace-write", ask_for_approval="on-request")`: Builds Codex CLI invocation for coder path.
  - `codex(prompt, sandbox="workspace-write")`: Calls Codex CLI or returns fake output when `AUTODEV_FAKE_CODEX_OUTPUT` is set; logs to `.autodev/logs/codex_orchestrator_last.log` and `..._last_message.txt`.
  - `implement(task)`: Builds coder prompt (includes rules, AUTODEV_README, live user notes, task/tests) and routes to `LLMProvider.code`.
  - `verify_deployment()`: Fetches latest GitHub Actions run via `gh.last_workflow_status`; requires `GITHUB_REPO` + token env vars.
  - `troubleshoot_deployment(deploy_result, task)`: Builds troubleshooting prompt (includes AUTODEV_README) and routes to `LLMProvider.troubleshoot`.
  - `main()`: Endless loop with live chat: gather snapshot, call planner (`create_backlog`), pick highest-priority actionable task, fold in live guidance, call `implement`, log outputs under `.autodev/logs/run_*.log`, optionally troubleshoot failed deployments, sleep between iterations. Timeouts configurable via env (`AUTODEV_CODER_TIMEOUT`).
- `.autodev/planner.py`
  - `build_planner_cmd(last_message_path)`: Codex CLI command builder for read-only planning with `--ask-for-approval never --sandbox read-only`.
  - `codex(prompt)`: Invoke Codex CLI or fake output for planning; logs to `.autodev/logs/codex_planner_last.log` and `..._last_message.txt`.
  - `load_rules()`: Reads `.autodev/prompts/rules.md`.
  - `load_autodev_readme()`: Reads `AUTODEV_README.md` for prompt context.
  - `read_ideas()`: Reads `ideas.md` from repo root if present.
  - `create_backlog(snapshot, live_notes=None)`: Builds strict JSON-only planning prompt (includes rules, AUTODEV_README, ideas, live notes, snapshot); self-seeds 3–5 high-value, accessible ideas if none exist. Env `AUTODEV_PLANNER_TIMEOUT` sets timeout for orchestrator loop (handled in orchestrator).
- `.autodev/llm_provider.py` (`LLMProvider`):
  - Public: `plan(prompt)`, `code(prompt)`, `troubleshoot(prompt)`; tries Gemini CLI first, then LM Studio HTTP (`http://localhost:1234/v1/chat/completions`) with task-type-specific model selection constrained by `max_vram_gb` (default 8).
  - Gemini detection: scans PATH for `gemini[.cmd/.exe]`; `_try_gemini` shells out with `--prompt "<prompt>" --output-format json`.
  - LM Studio selection: chooses from per-task model lists; simple VRAM filter. Set `AUTODEV_LOCAL_ONLY=1` (or run `autodev.ps1 -l`) to skip Gemini and use local providers only.

## Packaging / distribution
- Release artifact: `git archive --format=tar.gz -o autodev.tar.gz main` (run in this repo). The install one-liners consume `main`, so shipping a fresh archive is optional but remains compatible.
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
