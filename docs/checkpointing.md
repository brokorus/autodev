# AutoDev Checkpointing & Test Discipline

AutoDev now maintains explicit git checkpoints around every coding iteration and runs tests rigorously to protect against regressions.

## What happens each iteration
- A **pre-change checkpoint** is recorded before the CODER applies edits. This captures the current workspace so you can return to it.
- After coding, AutoDev **runs tests** (defaults: `npm test`, `npx playwright test`, `python -m pytest` when applicable, or overrides from `AUTODEV_TEST_COMMANDS`).
- A **post-change checkpoint** is recorded with the test results annotated in the commit message and in a JSONL audit log.

## Logs and metadata
- Checkpoints are logged to `.autodev/logs/checkpoints.jsonl` with the commit hash, label, timestamp, and paths staged.
- Test run details are written to `.autodev/logs/tests_iteration_<N>.log`.

## Customizing tests
- Set `AUTODEV_TEST_COMMANDS` to a semicolon-separated list to control exactly what runs. Example:  
  `AUTODEV_TEST_COMMANDS="python -m pytest;npm run lint"`
- Set `AUTODEV_TEST_TIMEOUT` (seconds) to extend per-command timeouts if needed.

## Restoring a checkpoint
- AutoDev will not perform destructive rollbacks automatically. To revert to a checkpoint:  
  `git restore --source <checkpoint-hash> -- .`  
  or inspect the log entry in `.autodev/logs/checkpoints.jsonl` and `git checkout <hash>` in a detached head to review changes.

The workflow prefers correctness and traceability over speed: every change is captured, and tests gate each iteration. If tests fail, the checkpoint log shows both the failing state and the prior safe hash so you can switch back confidently.
