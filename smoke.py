"""
Lightweight smoke checks for AutoDev.

Usage (from repo root):
    .autodev\\venv\\Scripts\\python.exe .autodev\\smoke.py planner
    .autodev\\venv\\Scripts\\python.exe .autodev\\smoke.py orchestrator
"""
import argparse
import json
from pathlib import Path

from orchestrator import snapshot, codex  # reuse orchestrator helpers
from planner import create_backlog


def run_planner() -> None:
    snap = snapshot(max_chars=2000)
    backlog = create_backlog(snap)
    print(json.dumps(backlog, indent=2))


def run_orchestrator() -> None:
    prompt = """
You are assisting with a smoke test of the AutoDev orchestrator Codex call.
Reply ONLY with the JSON object:
{"status": "orchestrator-ok"}
Do not run any commands.
""".strip()
    result = codex(prompt, sandbox="read-only")
    print(result.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoDev smoke checks")
    parser.add_argument(
        "which",
        choices=["planner", "orchestrator"],
        help="Which smoke check to run",
    )
    args = parser.parse_args()

    if args.which == "planner":
        run_planner()
    else:
        run_orchestrator()


if __name__ == "__main__":
    main()
