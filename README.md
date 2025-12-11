# AutoDev

AutoDev is a portable automation loop (planner + coder + troubleshooter) you can drop into any repo. The bootstrap scripts create a Python venv, install dependencies, and locate a working LLM CLI so you can start iterating immediately.

## One‑liner install (run in the target repo)
- Bash / macOS / Linux  
  `bash <(curl -fsSL https://raw.githubusercontent.com/brokorus/autodev/main/install.sh)`
- PowerShell / Windows  
  `irm https://raw.githubusercontent.com/brokorus/autodev/main/install.ps1 | iex`

What happens: we clone `brokorus/autodev` to a temp dir, copy `.autodev`, `autodev.ps1`, and `AUTODEV_README.md` into the current repo, then clean up. Pass `AUTODEV_FORCE=1` (bash) or `-Force` (PowerShell) to overwrite without a prompt.

## Requirements
- Git (used by the installer)
- Python 3 with `venv`
- Node + npm (to install Codex/Gemini CLI if missing)
- PowerShell (`powershell` on Windows or `pwsh` cross‑platform) to run `autodev.ps1`

## Run AutoDev
From the repo root after install:
1) `./autodev.ps1`
   - Export-only: `./autodev.ps1 -ExportPm linear|jira|taskwarrior|all`
   - Local-only: `./autodev.ps1 -l` to skip hosted LLMs and stick to local (LM Studio/offline)
2) Follow the prompts. AutoDev will:
   - Ensure Codex or Gemini CLI exists (installs via npm unless `-SkipCliInstall`)
   - Create `.autodev/venv` (skip with `-SkipVenvCreate`)
   - Install Python deps from `.autodev/requirements-autodev.txt`
   - Launch the orchestrator loop

## Packaging / release
- Create a distributable archive from `main`: `git archive --format=tar.gz -o autodev.tar.gz main`
- To verify contents: `tar tzf autodev.tar.gz | head`
- The install one‑liners work directly against the `main` branch of `brokorus/autodev`, so keeping `main` up to date is sufficient for consumers.

## Customization
- Env/flags: `AUTODEV_REPO`, `AUTODEV_BRANCH`, `AUTODEV_TARGET`, `AUTODEV_FORCE` (bash) or `-Repo`, `-Branch`, `-Force` (PowerShell)
- Edit `.autodev/prompts/rules.md` or `AUTODEV_README.md` to tune behavior; these are injected into every prompt AutoDev sends.
- Re-run the installer anytime to refresh to the latest AutoDev bits.
