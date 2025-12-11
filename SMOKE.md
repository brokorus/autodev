# AutoDev Smoke Checks

Prereqs:
- `codex` CLI available on `PATH`
- `.autodev\venv` created (as expected by `start-autodev.ps1`)

Run planner-only smoke (verifies backlog JSON and non-interactive planner call):
```
.autodev\venv\Scripts\python.exe .autodev\smoke.py planner
```

Run orchestrator Codex call smoke (verifies non-interactive exec path):
```
.autodev\venv\Scripts\python.exe .autodev\smoke.py orchestrator
```

Start full AutoDev loop:
```
.\start-autodev.ps1
```

Logs:
- Planner: `.autodev\logs\codex_planner_last.log` / `codex_planner_last_message.txt`
- Orchestrator: `.autodev\logs\codex_orchestrator_last.log` / `codex_orchestrator_last_message.txt`
