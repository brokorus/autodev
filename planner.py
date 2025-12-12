from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Ensure the .autodev directory is on sys.path before local imports
AUTODEV = Path(__file__).parent
if str(AUTODEV) not in sys.path:
    sys.path.append(str(AUTODEV))

from llm_provider import LLMProvider
from codex_cli import resolve_codex_command
from story import StoryLog


BASE = AUTODEV.parent
LOGS = AUTODEV / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
DEFAULT_TIMEOUT = int(os.getenv("AUTODEV_PLANNER_TIMEOUT", "240"))
STORY = StoryLog(AUTODEV / "story")


def sanitize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\x00", "")


def load_story_digest(limit: int = 12) -> str:
    try:
        return STORY.context_view(head=limit, big_picture=4, max_chars=3800)
    except Exception:  # noqa: BLE001
        return "No story entries yet."


def build_planner_cmd(last_message_path: Path) -> list[str]:
    """
    Build the Codex CLI command for planning in read-only mode.
    The planner never requests approvals and always runs sandboxed.
    """
    base_cmd = resolve_codex_command(AUTODEV)
    return [
        *base_cmd,
        "--ask-for-approval",
        "never",
        "exec",
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(last_message_path),
        "-",
    ]


def codex(prompt: str) -> str:
    """
    Invoke the Codex CLI for planning. If AUTODEV_FAKE_CODEX_OUTPUT is set,
    return it while still writing the expected log files for tests.
    """
    log_file = LOGS / "codex_planner_last.log"
    last_msg_file = LOGS / "codex_planner_last_message.txt"
    fake_output = os.getenv("AUTODEV_FAKE_CODEX_OUTPUT")

    if fake_output is not None:
        log_file.write_text(
            f"FAKE MODE\\nPROMPT:\\n{prompt}\\nOUTPUT:\\n{fake_output}\\n",
            encoding="utf-8",
            errors="ignore",
        )
        last_msg_file.write_text(fake_output, encoding="utf-8", errors="ignore")
        return fake_output

    cmd = build_planner_cmd(last_msg_file)
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        log_file.write_text(
            f"Codex CLI not found: {exc}\\nPROMPT:\\n{prompt}\\n",
            encoding="utf-8",
            errors="ignore",
        )
        raise RuntimeError("Codex CLI is not installed or not on PATH") from exc

    log_file.write_text(
        "CMD: {}\\nRETURN CODE: {}\\nSTDOUT:\\n{}\\nSTDERR:\\n{}".format(
            " ".join(cmd),
            proc.returncode,
            sanitize(proc.stdout),
            sanitize(proc.stderr),
        ),
        encoding="utf-8",
        errors="ignore",
    )
    last_msg_file.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    return proc.stdout


def load_rules() -> str:
    rules_path = AUTODEV / "prompts" / "rules.md"
    try:
        return rules_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


RULES = load_rules()

def load_autodev_readme() -> str:
    readme_path = BASE / "AUTODEV_README.md"
    if not readme_path.exists():
        return ""
    try:
        return sanitize(readme_path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return ""


def read_ideas() -> str:
    ideas_path = BASE / "ideas.md"
    if not ideas_path.exists():
        return ""
    return ideas_path.read_text(encoding="utf-8", errors="ignore")



def create_backlog(snapshot: str) -> dict:
    llm_provider = LLMProvider()
    autodev_readme = load_autodev_readme()
    story_digest = load_story_digest()
    prompt = f"""
{RULES}

AUTODEV_README (system overview):
{autodev_readme or "No AUTODEV_README.md provided."}

PRODUCT STORY (recent narrative; avoid repeating failed ideas):
{story_digest or "No story entries yet."}

You are the PLANNER (seasoned software architect) for AutoDev.

CRITICAL INSTRUCTIONS:
- NEVER execute shell commands or code; planning is read-only and based solely on the provided snapshot.
- KEEP CONTEXT TIGHT: use only what appears in AUTODEV_README + snapshot; do not invent files/APIs. If unsure, ask for prerequisites rather than guessing.
- OUTPUT AT MOST 3 TASKS. If work should pause, set halt_reason instead of emitting tasks.
- Apply 15-factor/12-factor-inspired cloud-native principles (config-as-code, backing services as attached resources, disposability, parity, observability, security, governance, secrets hygiene, logs/metrics, admin processes).
- For EACH idea, verify prerequisites and call them out explicitly. Example: "build a user map" requires identity/user management + location permissions; if missing, create prerequisite tasks first and mark the feature as not yet met.
- Inventory infrastructure dependencies. Prefer GitHub Actions + Cloudflare free-tier (Workers, KV, D1, Pages, Queues, R2 within free limits). Provide a YAML/CLI snippet the coder can drop into the repo/GitHub Actions to create missing infra.
- Avoid paid-only services. If any idea cannot be implemented without paid resources, set "halt_reason" explaining the blocker and leave the backlog empty.
- TAKE THE HARD THINKING OUT OF CODING: for every task, include an "impl_spec" field with:
  - "functions": list of function signatures/responsibilities (name, params, returns/raises, side effects).
  - "pseudo": terse bullet pseudocode for happy path + error handling.
  - "tests_needed": concrete test cases (inputs -> expected outputs).
  - "edge_cases": explicit limit/empty/error conditions to handle.
- Prefer single-iteration, well-scoped tasks (no vague epics).
- Always emit JSON ONLY using the format below. Do not add narrative text.

Planner output format:
{{
  "halt_reason": "<string, optional; present only when work must stop>",
  "backlog": [
    {{
      "idea": "...",
      "task": "...",
      "tests": ["...", "..."],
      "priority": 1,
      "prerequisites": {{
        "met": true,
        "missing": ["..."],
        "blocked_due_to_paid_service": false,
        "fifteen_factor_gates": ["config-as-code", "backing-services", "logs-metrics", "security"],
        "infra": [
          {{
            "resource": "cloudflare_kv",
            "needed": true,
            "free_tier_only": true,
            "gh_action": "name: Provision KV\\nuses: cloudflare/wrangler-action@v3\\nwith:\\n  apiToken: ${{{{ secrets.CLOUDFLARE_API_TOKEN }}}}\\n  command: wrangler kv:namespace create app-cache",
            "notes": "What the resource is for and how it supports the task"
          }}
        ]
      }},
      "impl_spec": {{
        "functions": [
          {{"name": "function_name", "params": ["arg1: type", "arg2: type"], "returns": "type", "raises": ["..."], "side_effects": ["filesystem writes ..."] }}
        ],
        "pseudo": ["step 1", "step 2"],
        "tests_needed": ["input -> expected", "..."],
        "edge_cases": ["empty list", "timeout", "..."]
      }}
    }}
  ]
}}

Prioritization rules:
- Pre-req tasks must precede dependent feature tasks (lower priority number = earlier).
- A blocked feature (missing prereqs) MUST NOT outrank its enabling tasks.
- Always include at least one concrete test idea per task.

IDEAS:
----------------
{read_ideas()}

REPO SNAPSHOT:
----------------
{snapshot}
"""

    response = llm_provider.plan(prompt)
    if response:
        try:
            STORY.add_entry(
                kind="plan",
                title="Planner backlog refresh",
                summary="Planner produced a backlog based on latest snapshot.",
                tags=["plan"],
                details=response,
            )
        except Exception:
            pass
        return response
    raise RuntimeError(
        "Planner failed to return a valid plan. See .autodev/logs/ for details."
    )
