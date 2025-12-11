# AutoDev

AutoDev is a portable automation loop (planner + coder + troubleshooter) you can drop into any repo. The bootstrap script handles Python venv creation, dependency installs, and finding the Codex/Gemini CLI so you can start iterating immediately.

## Quick install (run from your project root)
- Bash: `bash <(curl -fsSL https://raw.githubusercontent.com/brokorus/autodev/main/install.sh)`
- PowerShell: `irm https://raw.githubusercontent.com/brokorus/autodev/main/install.ps1 | iex`

What this does: clones `brokorus/autodev` to a temp directory, copies `.autodev`, `start-autodev.ps1`, and `AUTODEV_README.md` into your current repo, then cleans up. It will prompt before overwriting; set `AUTODEV_FORCE=1` (bash) or pass `-Force` (PowerShell) to skip the prompt.

## Requirements
- Git (installer clones the repo)
- Python 3 with `venv` available
- Node + npm (to install the Codex/Gemini CLI if they are not on PATH)
- PowerShell (Windows `powershell` or cross-platform `pwsh`) to run `start-autodev.ps1`

## Run AutoDev
After installation, from your project root:
1) `./start-autodev.ps1` (or `pwsh -File start-autodev.ps1` on macOS/Linux)
2) Follow the prompts. The script will:
   - Ensure Codex or Gemini CLI is available (it can install via npm unless `-SkipCliInstall` is passed).
   - Create `.autodev/venv` unless `-SkipVenvCreate` is passed.
   - Install Python deps from `.autodev/requirements-autodev.txt`.
   - Launch the orchestrator loop.

## Customization
- `AUTODEV_REPO`, `AUTODEV_BRANCH`, `AUTODEV_TARGET`, `AUTODEV_FORCE` env vars tweak the bash installer; `-Repo`, `-Branch`, and `-Force` parameters do the same in PowerShell.
- Edit `.autodev/prompts/rules.md` or `AUTODEV_README.md` to tune behavior; these are injected into every prompt AutoDev sends.
- Re-run the installer (with `-Force`/`AUTODEV_FORCE=1`) anytime to refresh to the latest AutoDev bits.
