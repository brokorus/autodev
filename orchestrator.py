from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

AUTODEV = Path(__file__).parent
BASE = AUTODEV.parent
sys.path.append(str(AUTODEV))

from llm_provider import LLMProvider
from codex_cli import resolve_codex_command
from state import AutoDevState
from story import StoryLog


LOGS = AUTODEV / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
PM_DIR = AUTODEV / "pm"
PM_DIR.mkdir(parents=True, exist_ok=True)
STORY_DIR = AUTODEV / "story"
STORY = StoryLog(STORY_DIR)
DEFAULT_TIMEOUT = int(os.getenv("AUTODEV_CODER_TIMEOUT", "900"))
DEFAULT_TEST_TIMEOUT = int(os.getenv("AUTODEV_TEST_TIMEOUT", "900"))
CHECKPOINT_LOG = LOGS / "checkpoints.jsonl"
SKIP_CHECKPOINT_PREFIXES = {
    ".autodev/logs",
    ".autodev/venv",
    ".autodev/.venv",
    "venv",
    ".venv",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    "__pycache__",
    ".svelte-kit",
    "coverage",
}

from planner import create_backlog  # noqa: E402
import gh  # noqa: E402


def sanitize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\x00", "")


def story_safe_add(**kwargs) -> str | None:
    """Never let story logging break the loop. Returns entry_id when possible."""
    try:
        return STORY.add_entry(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Story logging failed: %s", exc)
        return None


class GitCheckpointManager:
    """
    Lightweight git checkpointing helper.
    Creates labeled commits (optionally empty) to capture the workspace state
    without performing destructive rollbacks.
    """

    def __init__(self, repo_root: Path, log_file: Path):
        self.repo_root = repo_root
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _git(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=check,
        )

    def is_repo(self) -> bool:
        try:
            result = self._git(["rev-parse", "--is-inside-work-tree"])
            return result.stdout.strip() == "true"
        except subprocess.CalledProcessError:
            return False

    def _changed_paths(self) -> list[str]:
        """
        Return a filtered list of changed/untracked paths we should checkpoint.
        Skips bulky/generated directories to keep checkpoints useful and fast.
        """
        try:
            result = self._git(["status", "--porcelain"], check=True)
        except subprocess.CalledProcessError:
            return []

        paths: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            # Format: XY path (rename shows "R  old -> new")
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if any(path.startswith(prefix) for prefix in SKIP_CHECKPOINT_PREFIXES):
                continue
            paths.append(path)
        return paths

    def create_checkpoint(
        self,
        label: str,
        metadata: dict | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        """
        Stage filtered changes and commit them with a descriptive message.
        Writes a JSONL audit entry with the commit hash and context.
        """
        if not self.is_repo():
            return None

        paths = self._changed_paths()
        if paths:
            try:
                self._git(["add", "--", *paths])
            except subprocess.CalledProcessError as exc:
                logging.error("Failed to stage checkpoint paths: %s", exc)
                return None

        commit_args = ["commit", "--no-verify", "-m", self._format_message(label, metadata)]
        if allow_empty:
            commit_args.insert(1, "--allow-empty")

        try:
            self._git(commit_args)
            commit_hash = self._git(["rev-parse", "HEAD"]).stdout.strip()
        except subprocess.CalledProcessError as exc:
            logging.error("Failed to create checkpoint '%s': %s", label, exc)
            return None

        self._log_checkpoint(commit_hash, label, metadata or {}, paths)
        return commit_hash

    def _format_message(self, label: str, metadata: dict | None) -> str:
        parts = [f"AutoDev checkpoint: {label}"]
        if metadata:
            # keep message succinct; detailed metadata goes to JSONL log
            summary_bits = []
            if "iteration" in metadata:
                summary_bits.append(f"iter {metadata['iteration']}")
            if "phase" in metadata:
                summary_bits.append(metadata["phase"])
            if "tests_passed" in metadata:
                summary_bits.append("tests=pass" if metadata["tests_passed"] else "tests=fail")
            if summary_bits:
                parts.append(f"({' | '.join(summary_bits)})")
        return " ".join(parts)

    def _log_checkpoint(self, commit_hash: str, label: str, metadata: dict, paths: list[str]) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "commit": commit_hash,
            "label": label,
            "paths": paths,
            "metadata": metadata,
        }
        try:
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logging.error("Failed to record checkpoint log: %s", exc)


RULES = (AUTODEV / "prompts" / "rules.md").read_text(
    encoding="utf-8",
    errors="ignore",
)

def load_autodev_readme() -> str:
    """
    Provide repository context to downstream LLMs.
    """
    readme_path = BASE / "AUTODEV_README.md"
    if not readme_path.exists():
        return ""
    try:
        return sanitize(readme_path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return ""


def load_story_digest(limit: int = 12) -> str:
    """
    Provide a short, recent narrative to LLMs.
    """
    try:
        return STORY.context_view(head=limit, big_picture=4, max_chars=3800)
    except Exception:  # noqa: BLE001
        return "No story digest available yet."

def snapshot(base: Path = BASE, max_chars: int = 40_000) -> str:
    """
    Capture a text-only snapshot of the repo for the planner.
    Skips noisy/irrelevant directories (including .autodev) and truncates output.
    """
    exclude_dirs = {
        ".git",
        ".autodev",
        "node_modules",
        "venv",
        ".venv",
        "__pycache__",
        ".idea",
        ".vscode",
        ".turbo",
        ".next",
        "dist",
        "build",
    }
    chunks: list[str] = []
    readme_path = BASE / "AUTODEV_README.md"
    relative_readme = None
    try:
        relative_readme = readme_path.relative_to(base)
    except ValueError:
        relative_readme = None
    if readme_path.exists() and relative_readme:
        try:
            content = sanitize(readme_path.read_text(encoding="utf-8", errors="ignore"))
            entry = f"# File: {relative_readme}\n{content}\n"
            if len(entry) < max_chars:
                chunks.append(entry)
        except OSError:
            pass
    total = 0

    for path in sorted(base.rglob("*")):
        rel_parts = path.relative_to(base).parts
        if any(part in exclude_dirs for part in rel_parts):
            continue

        if path.is_dir():
            continue

        try:
            if relative_readme and path.resolve() == readme_path.resolve():
                continue
        except OSError:
            pass

        try:
            if path.stat().st_size > 100_000:
                continue
            content = sanitize(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, UnicodeDecodeError):
            continue

        entry = f"# File: {path.relative_to(base)}\n{content}\n"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                chunks.append(entry[:remaining])
            break

        chunks.append(entry)
        total += len(entry)

    return "\n".join(chunks)


def derive_expertise(task: dict) -> str:
    """
    Heuristic to steer the coder persona toward the right discipline.
    """
    text = (task.get("task") or "").lower()
    tests = " ".join(task.get("tests") or []).lower()
    combined = f"{text} {tests}"
    if any(word in combined for word in ["playwright", "svelte", "frontend", "ui", "ux"]):
        return "Staff frontend+UX engineer pairing with QA"
    if any(word in combined for word in ["infra", "cloudflare", "deployment", "actions", "ci"]):
        return "Staff reliability + DevOps engineer"
    if "api" in combined or "backend" in combined:
        return "Staff backend engineer with API hardening focus"
    return "Principal full-stack engineer"


def summarize_board(state: AutoDevState, max_chars: int = 2000) -> str:
    """
    Provide a compact PM snapshot to the coder to avoid repeated attempts.
    """
    try:
        raw = state.board_file.read_text(encoding="utf-8")
    except OSError:
        return ""
    return raw[:max_chars]


def _detect_playwright_config(base: Path) -> bool:
    for name in ("playwright.config.ts", "playwright.config.js", "playwright.config.mjs"):
        if (base / name).exists():
            return True
    return False


def discover_test_commands(base: Path = BASE) -> list[str]:
    """
    Determine which test commands to run. Users can override via AUTODEV_TEST_COMMANDS
    (semicolon-separated). Defaults favor accuracy over speed.
    """
    env_cmds = os.getenv("AUTODEV_TEST_COMMANDS")
    if env_cmds:
        return [cmd.strip() for cmd in env_cmds.split(";") if cmd.strip()]

    commands: list[str] = []
    if (base / "package.json").exists():
        commands.append("npm test")
        if _detect_playwright_config(base) or (base / "tests").exists():
            commands.append("npx playwright test")
    if (base / "tests").exists():
        commands.append("python -m pytest")

    # preserve order while removing duplicates
    seen = set()
    unique_commands: list[str] = []
    for cmd in commands:
        if cmd in seen:
            continue
        seen.add(cmd)
        unique_commands.append(cmd)
    return unique_commands


def run_rigorous_tests(iteration: int, task_label: str, base: Path = BASE) -> dict:
    commands = discover_test_commands(base=base)
    if not commands:
        return {"ran": False, "all_passed": True, "results": [], "log_file": None}

    log_file = LOGS / f"tests_iteration_{iteration}.log"
    log_lines: list[str] = []
    results: list[dict] = []

    for cmd in commands:
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=base,
                text=True,
                capture_output=True,
                timeout=DEFAULT_TEST_TIMEOUT,
            )
            returncode = proc.returncode
            stdout = sanitize(proc.stdout)
            stderr = sanitize(proc.stderr)
        except FileNotFoundError as exc:
            returncode = 127
            stdout = ""
            stderr = str(exc)
        except subprocess.TimeoutExpired:
            returncode = 124
            stdout = ""
            stderr = f"Timed out after {DEFAULT_TEST_TIMEOUT}s"

        duration = time.time() - start
        result = {
            "command": cmd,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration_seconds": round(duration, 2),
        }
        results.append(result)
        log_lines.append(
            "=== {} ===\ncmd: {}\nreturncode: {}\nduration: {}s\nstdout:\n{}\nstderr:\n{}\n".format(
                task_label,
                cmd,
                returncode,
                round(duration, 2),
                stdout,
                stderr,
            )
        )

    try:
        log_file.write_text("\n".join(log_lines), encoding="utf-8", errors="ignore")
    except OSError:
        pass

    all_passed = all(item["returncode"] == 0 for item in results)
    return {"ran": True, "all_passed": all_passed, "results": results, "log_file": str(log_file)}


def build_codex_cmd(
    last_message_path: Path,
    sandbox: str = "workspace-write",
    ask_for_approval: str = "on-request",
) -> list[str]:
    base_cmd = resolve_codex_command(AUTODEV)
    return [
        *base_cmd,
        "--ask-for-approval",
        ask_for_approval,
        "exec",
        "--sandbox",
        sandbox,
        "--output-last-message",
        str(last_message_path),
        "-",
    ]


def codex(prompt: str, sandbox: str = "workspace-write") -> str:
    """
    Invoke the Codex CLI (or a fake output when AUTODEV_FAKE_CODEX_OUTPUT is set).
    Used by smoke tests and manual debugging.
    """
    log_file = LOGS / "codex_orchestrator_last.log"
    last_msg = LOGS / "codex_orchestrator_last_message.txt"
    fake = os.getenv("AUTODEV_FAKE_CODEX_OUTPUT")

    if fake is not None:
        log_file.write_text(
            f"FAKE MODE\nPROMPT:\n{prompt}\nOUTPUT:\n{fake}\n",
            encoding="utf-8",
            errors="ignore",
        )
        last_msg.write_text(fake, encoding="utf-8", errors="ignore")
        return fake

    cmd = build_codex_cmd(last_msg, sandbox=sandbox)
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
            f"Codex CLI not found: {exc}\nPROMPT:\n{prompt}\n",
            encoding="utf-8",
            errors="ignore",
        )
        raise RuntimeError("Codex CLI is not installed or not on PATH") from exc

    log_file.write_text(
        "CMD: {}\nRETURN CODE: {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
            " ".join(cmd), proc.returncode, sanitize(proc.stdout), sanitize(proc.stderr)
        ),
        encoding="utf-8",
        errors="ignore",
    )
    last_msg.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    return proc.stdout


def implement(task: dict, board_snapshot: str, expertise: str, story_ref: Optional[str]) -> str:
    with LLMProvider() as llm_provider:
        autodev_readme = load_autodev_readme()
        story_digest = load_story_digest()
        impl_spec = task.get("impl_spec") or {}
        impl_spec_text = json.dumps(impl_spec, indent=2) if impl_spec else "None provided."
        prompt = """
{rules}

AUTODEV_README (system overview):
{autodev_readme}

PRODUCT STORY (recent narrative):
{story_digest}

STORY REFERENCE FOR THIS ITERATION:
{story_ref}

You are the CODER.
Operate as: {expertise}

TASK:
{task}

Implementation spec from planner (follow exactly, extend only when necessary):
{impl_spec}

TESTS:
{tests}

Project board snapshot (recent tickets, failures):
{board_snapshot}

Work in the project root: {base}

Requirements:
- Write COMPLETE files (no partial patches).
- Add or update tests as needed.
- Run: npm test && npx playwright test
- Fix any test failures and rerun until everything passes.
- When you add or change code, include a nearby inline comment with `STORY:{story_ref}` (use the provided Story ID) to link the narrative to the code. Keep these comments concise.
- Assume GitHub Actions will run on push; the orchestrator will check deployment status separately.
- Keep context tight: prefer the referenced files/functions; do not invent APIs or files that are not in the snapshot.
- Be verbose in your final response:
  - Start with a clear plan/approach describing the concrete steps you will take.
  - List each action performed (commands run, files created/updated, rationale).
  - Call out the status of each listed test and any additional tests you ran.
- Finish with next steps or remaining risks, even if none.
""".format(
            rules=RULES,
            autodev_readme=autodev_readme or "No AUTODEV_README.md provided.",
            task=task["task"],
            impl_spec=impl_spec_text,
            tests=json.dumps(task["tests"], indent=2),
            board_snapshot=board_snapshot or "No tickets logged yet.",
            expertise=expertise,
            base=BASE,
            story_digest=story_digest or "No story digest available.",
            story_ref=story_ref or "STORY:missing",
        )
        return llm_provider.code(prompt)


def verify_deployment() -> dict:
    """
    Ask GitHub Actions for the last workflow run result.
    Requires env var GITHUB_REPO (e.g. "owner/repo") and a GitHub token.
    """
    repo = os.getenv("GITHUB_REPO")
    if not repo:
        return {"error": "GITHUB_REPO env var missing"}
    return gh.last_workflow_status(repo)


def troubleshoot_deployment(deploy_result: dict, task: dict) -> str:
    with LLMProvider() as llm_provider:
        autodev_readme = load_autodev_readme()
        prompt = """
{rules}

AUTODEV_README (system overview):
{autodev_readme}

Deployment appears to have FAILED or is not successful.

GitHub Actions last workflow result:
{deploy}

TASK BEING DEPLOYED:
{task}

Instructions:
- Diagnose the likely root cause based on the workflow result.
- Propose code or configuration changes.
- Modify files as needed (complete files only).
- Ensure tests still pass.
- Aim for a successful deployment on the next GitHub Actions run.
""".format(
            rules=RULES,
            autodev_readme=autodev_readme or "No AUTODEV_README.md provided.",
            deploy=json.dumps(deploy_result, indent=2),
            task=json.dumps(task, indent=2),
        )
        return llm_provider.troubleshoot(prompt)



def main() -> None:
    print("=== AutoDev Orchestrator Running ===")

    git_cp = GitCheckpointManager(BASE, CHECKPOINT_LOG)
    if not git_cp.is_repo():
        print("Warning: not inside a git repository; checkpoints will be skipped.")

    iteration = 0
    state = AutoDevState(PM_DIR)
    while True:
        iteration += 1
        print("\n=== Iteration {} ===".format(iteration))
        try:
            snap = snapshot(base=BASE)
            print(
                "Snapshot collected ({} chars) to provide context to the planner.".format(
                    len(snap)
                )
            )
            print("Planner running in ask/read-only mode with enforced timeout.")
            backlog_data = create_backlog(snap)
            story_safe_add(
                kind="plan",
                title=f"Planner iteration {iteration}",
                summary="Planner generated backlog items from snapshot.",
                iteration=iteration,
                tags=["plan"],
                details={"backlog_raw": backlog_data},
            )
            halt_reason = (backlog_data.get("halt_reason") or "").strip()
            if halt_reason:
                print(
                    "Planner halted work due to non-free/prereq constraint: {}".format(
                        halt_reason
                    )
                )
                story_safe_add(
                    kind="plan",
                    title=f"Planner halted iteration {iteration}",
                    summary="Planner stopped due to halt_reason.",
                    iteration=iteration,
                    status="halted",
                    tags=["plan", "halted"],
                    details={"halt_reason": halt_reason},
                )
                break

            backlog = backlog_data.get("backlog", [])
            if not backlog:
                print("No backlog items produced from ideas.md yet. Sleeping...")
                time.sleep(30)
                continue

            print("Planner produced {} backlog item(s).".format(len(backlog)))
            actionable = []
            blocked = []
            for idx, item in enumerate(backlog, start=1):
                prereq = item.get("prerequisites") or {}
                missing = prereq.get("missing") or []
                blocked_paid = prereq.get("blocked_due_to_paid_service")
                met = prereq.get("met")
                if blocked_paid:
                    blocked.append((item, "requires paid service"))
                elif missing and met is not True:
                    blocked.append(
                        (item, "missing prerequisites: {}".format(", ".join(missing)))
                    )
                else:
                    actionable.append(item)
                print(
                    "- [{}] Priority {}: {}".format(
                        idx, item.get("priority", "N/A"), item.get("task", "").strip()
                    )
                )
                if item.get("idea"):
                    print("  Idea: {}".format(item["idea"].strip()))
                tests = item.get("tests") or []
                if tests:
                    print("  Expected tests: {}".format(", ".join(tests)))
                if missing:
                    print("  Missing prereqs: {}".format(", ".join(missing)))
                if blocked_paid:
                    print("  BLOCKED: requires paid service (free tier unavailable).")
                infra = prereq.get("infra") or []
                if infra:
                    resources = [r.get("resource", "resource") for r in infra if r.get("needed")]
                    if resources:
                        print("  Infra needs: {}".format(", ".join(resources)))

            state.sync_backlog(backlog, blocked)
            state.render_board()
            if not actionable:
                print(
                    "Planner did not produce an actionable task (all blocked by prerequisites or cost). Sleeping..."
                )
                paid_blockers = [
                    reason
                    for _, reason in blocked
                    if "paid service" in reason.lower()
                ]
                if paid_blockers and len(paid_blockers) == len(blocked):
                    print(
                        "All backlog items require paid services; stopping work per policy."
                    )
                    break
                time.sleep(60)
                continue

            next_item = state.next_actionable()
            if not next_item:
                print("No actionable tasks in the board after syncing backlog. Sleeping...")
                time.sleep(30)
                continue

            task_id, task = next_item
            print("\nSelected task (highest priority):")
            print("  Task:", task.get("task"))
            if task.get("idea"):
                print("  Idea:", task.get("idea"))
            print("  Priority:", task.get("priority"))
            tests = task.get("tests") or []
            print("  Tests to consider:", tests if tests else "none provided")
            print(
                "Invoking CODER in exec mode with workspace-write sandbox (timeout {}s)...".format(
                    DEFAULT_TIMEOUT
                )
            )

            task_label = (task.get("task") or "task").strip()
            state.mark_in_progress(task_id, iteration)
            pre_meta = {
                "iteration": iteration,
                "phase": "pre-change",
                "task": task_label,
                "id": task_id,
            }
            pre_checkpoint = git_cp.create_checkpoint(
                f"iteration {iteration} pre-change: {task_label}",
                metadata=pre_meta,
                allow_empty=True,
            )
            if pre_checkpoint:
                print(f"Checkpoint captured before changes: {pre_checkpoint}")

            board_snapshot = summarize_board(state)
            code_story_id = story_safe_add(
                kind="code",
                title=f"Coder start for {task_id}",
                summary="Coder began implementation for task.",
                iteration=iteration,
                task_id=task_id,
                task=task_label,
                status="in_progress",
                tests=task.get("tests", []),
                tags=["code", task_id],
            )
            result = implement(task, board_snapshot, derive_expertise(task), code_story_id)
            print(result)
            story_safe_add(
                kind="code",
                title=f"Coder work on {task_id}",
                summary="Coder produced implementation output.",
                iteration=iteration,
                task_id=task_id,
                task=task_label,
                status="in_progress",
                tests=task.get("tests", []),
                tags=["code", task_id],
                details={"output_log": str((LOGS / f"run_{iteration}.log").as_posix()), "story_ref": code_story_id},
            )

            log_path = LOGS / "run_{}.log".format(iteration)
            log_path.write_text(result, encoding="utf-8", errors="ignore")

            changed_paths = []
            try:
                changed_paths = git_cp._changed_paths()
            except Exception:
                changed_paths = []

            test_outcome = run_rigorous_tests(iteration, task_label)
            if test_outcome.get("ran"):
                print("Test suite executed; all passed:", test_outcome["all_passed"])
                for res in test_outcome["results"]:
                    print(
                        "  - {} -> rc {} ({}s)".format(
                            res["command"],
                            res["returncode"],
                            res["duration_seconds"],
                        )
                    )
                if test_outcome.get("log_file"):
                    print("Test log:", test_outcome["log_file"])
            else:
                print("No test commands detected/configured; skipping test run.")

            if test_outcome.get("all_passed", True):
                state.mark_done(task_id, iteration, test_outcome)
                story_safe_add(
                    kind="tests",
                    title=f"Tests passed for {task_id}",
                    summary="All configured test commands passed after coder changes.",
                    iteration=iteration,
                    task_id=task_id,
                    task=task_label,
                    status="done",
                    tests=[r["command"] for r in test_outcome.get("results", [])],
                    files=changed_paths,
                    tags=["tests", "pass", task_id],
                    details=test_outcome,
                )
            else:
                state.mark_failed(task_id, iteration, test_outcome)
                fix_tasks = state.create_fix_tasks(task_id, task_label, test_outcome)
                if fix_tasks:
                    state.add_tasks(fix_tasks)
                    print("Added {} remediation task(s) based on failing tests.".format(len(fix_tasks)))
                story_safe_add(
                    kind="tests",
                    title=f"Tests failed for {task_id}",
                    summary="At least one test command failed; remediation tasks may be added.",
                    iteration=iteration,
                    task_id=task_id,
                    task=task_label,
                    status="needs_fix",
                    tests=[r["command"] for r in test_outcome.get("results", [])],
                    files=changed_paths,
                    tags=["tests", "fail", task_id],
                    details=test_outcome,
                )

            post_meta = {
                "iteration": iteration,
                "phase": "post-change",
                "task": task_label,
                "id": task_id,
                "tests_passed": test_outcome.get("all_passed", True),
                "test_commands": [r["command"] for r in test_outcome.get("results", [])],
                "test_log": test_outcome.get("log_file"),
            }
            post_checkpoint = git_cp.create_checkpoint(
                f"iteration {iteration} post-change: {task_label}",
                metadata=post_meta,
                allow_empty=True,
            )
            if post_checkpoint:
                status_note = "tests-pass" if post_meta["tests_passed"] else "tests-fail"
                print(f"Checkpoint captured after changes ({status_note}): {post_checkpoint}")

            deploy = verify_deployment()
            if deploy.get("conclusion") not in ("success", "neutral") and not deploy.get("error"):
                print("Deployment not successful; attempting troubleshooting...")
                fix_output = troubleshoot_deployment(deploy, task)
                print(fix_output)
                fix_log_path = LOGS / "deploy_fix_{}.log".format(iteration)
                fix_log_path.write_text(fix_output, encoding="utf-8", errors="ignore")
                story_safe_add(
                    kind="deployment",
                    title=f"Deployment troubleshooting iteration {iteration}",
                    summary="GitHub Actions deployment was not successful; troubleshooting run recorded.",
                    iteration=iteration,
                    task_id=task_id,
                    task=task_label,
                    status="deploy_fix",
                    details=deploy,
                )

            time.sleep(10)
        except Exception as exc:  # noqa: BLE001
            err_path = LOGS / "iteration_{}_error.log".format(iteration)
            err_path.write_text(
                "Iteration {} failed: {}\n".format(iteration, repr(exc)),
                encoding="utf-8",
                errors="ignore",
            )
            print(
                "Iteration {} hit an error: {}. Logged to {}. Sleeping before retry...".format(
                    iteration, exc, err_path
                )
            )
            story_safe_add(
                kind="error",
                title=f"Iteration {iteration} exception",
                summary="Orchestrator loop hit an exception; see error log.",
                iteration=iteration,
                status="error",
                details={"exception": repr(exc), "error_log": str(err_path)},
            )
            time.sleep(30)
            continue


if __name__ == "__main__":
    main()
