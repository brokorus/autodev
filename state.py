from __future__ import annotations

import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_STATE: Dict = {"tasks": {}}


def _slugify(text: str, length: int = 80) -> str:
    """
    Create a filesystem-safe identifier for a task.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        slug = "task"
    return slug[:length] or "task"


class AutoDevState:
    """
    Persistent, lightweight project board for AutoDev.
    Tracks tasks, attempts, and remediation tickets to avoid repeating work
    and to surface granular follow-ups after failures.
    """

    def __init__(self, pm_dir: Path):
        self.pm_dir = pm_dir
        self.pm_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.pm_dir / "state.json"
        self.tickets_file = self.pm_dir / "tickets.jsonl"
        self.board_file = self.pm_dir / "board.md"
        self.export_dir = self.pm_dir / "export"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict = self._load()

    # ---------------- State persistence ----------------
    def _load(self) -> Dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return json.loads(json.dumps(DEFAULT_STATE))

    def save(self) -> None:
        try:
            self.state_file.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
        self._export_formats()

    # ---------------- Task helpers ----------------
    def _ensure_task(self, task_obj: Dict) -> Tuple[str, Dict]:
        task_id = task_obj.get("id") or _slugify(task_obj.get("task", "task"))
        current = self.data["tasks"].get(task_id, {})
        merged = {
            "id": task_id,
            "task": task_obj.get("task", ""),
            "idea": task_obj.get("idea"),
            "tests": task_obj.get("tests", []),
            "priority": task_obj.get("priority", current.get("priority", 999)),
            "status": current.get("status", task_obj.get("status", "todo")),
            "attempts": current.get("attempts", 0),
            "last_result": current.get("last_result"),
            "last_iteration": current.get("last_iteration"),
            "parent": task_obj.get("parent", current.get("parent")),
            "prerequisites": task_obj.get("prerequisites", current.get("prerequisites", {})),
            "context": task_obj.get("context", current.get("context", {})),
        }
        self.data["tasks"][task_id] = merged
        return task_id, merged

    def sync_backlog(self, backlog: List[Dict], blocked: List[Tuple[Dict, str]]) -> None:
        """
        Merge planner backlog into state. Blocked tasks are recorded but not selected.
        """
        for item in backlog:
            self._ensure_task(item)
        for item, reason in blocked:
            item_copy = dict(item)
            item_copy["status"] = "blocked"
            item_copy["context"] = {"blocked_reason": reason}
            self._ensure_task(item_copy)
        self.save()

    def next_actionable(self) -> Optional[Tuple[str, Dict]]:
        """
        Pick the next task, prioritizing remediation items and avoiding blocked/done work.
        """
        def sort_key(entry: Tuple[str, Dict]):
            _, task = entry
            status_weight = {
                "needs_fix": 0,
                "todo": 1,
                "in_progress": 2,
                "cooldown": 3,
                "blocked": 98,
                "done": 99,
            }.get(task.get("status", "todo"), 50)
            if task.get("parent"):
                status_weight = -1  # remediation subtasks always take precedence
            return (status_weight, task.get("priority", 999), task.get("attempts", 0))

        candidates = [
            (task_id, task)
            for task_id, task in self.data["tasks"].items()
            if task.get("status") not in {"done", "blocked"}
        ]
        if not candidates:
            return None
        candidates.sort(key=sort_key)
        chosen_id, chosen = candidates[0]
        return chosen_id, chosen

    def mark_in_progress(self, task_id: str, iteration: int) -> None:
        task = self.data["tasks"].get(task_id)
        if not task:
            return
        task["status"] = "in_progress"
        task["last_iteration"] = iteration
        self.save()

    def mark_done(self, task_id: str, iteration: int, test_outcome: Dict) -> None:
        task = self.data["tasks"].get(task_id)
        if not task:
            return
        task["status"] = "done"
        task["attempts"] = task.get("attempts", 0) + 1
        task["last_iteration"] = iteration
        task["last_result"] = {"tests_passed": True, "details": test_outcome}
        self._log_ticket(task, "done", "All tests passed")
        self.save()
        self.render_board()

    def mark_failed(self, task_id: str, iteration: int, test_outcome: Dict) -> None:
        task = self.data["tasks"].get(task_id)
        if not task:
            return
        task["status"] = "needs_fix"
        task["attempts"] = task.get("attempts", 0) + 1
        task["last_iteration"] = iteration
        task["last_result"] = {"tests_passed": False, "details": test_outcome}
        self._log_ticket(task, "needs_fix", "Tests failed")
        self.save()
        self.render_board()

    def add_tasks(self, tasks: List[Dict]) -> None:
        for item in tasks:
            self._ensure_task(item)
        self.save()
        self.render_board()

    # ---------------- Remediation helpers ----------------
    def create_fix_tasks(self, parent_id: str, task_label: str, test_outcome: Dict) -> List[Dict]:
        """
        Generate granular remediation tasks from failed test runs to avoid
        re-attempting the same broken feature blindly.
        """
        results = test_outcome.get("results", []) if isinstance(test_outcome, dict) else []
        failures = [r for r in results if r.get("returncode") not in (0, None)]
        if not failures:
            return []

        fix_tasks = []
        for idx, failure in enumerate(failures, start=1):
            stderr = (failure.get("stderr") or "").strip()
            stdout = (failure.get("stdout") or "").strip()
            evidence = (stderr or stdout).splitlines()[:3]
            evidence_text = "; ".join(evidence) if evidence else "No output captured."
            fix_tasks.append(
                {
                    "id": f"{parent_id}-fix-{idx}",
                    "task": f"Stabilize failing command `{failure.get('command')}` for '{task_label}'",
                    "tests": [failure.get("command", "tests")],
                    "priority": 0,  # always outrank the original task until fixed
                    "status": "todo",
                    "parent": parent_id,
                    "context": {"evidence": evidence_text},
                }
            )
        return fix_tasks

    # ---------------- Reporting ----------------
    def _log_ticket(self, task: Dict, status: str, note: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "id": task.get("id"),
            "task": task.get("task"),
            "status": status,
            "note": note,
            "attempts": task.get("attempts"),
        }
        try:
            with self.tickets_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def render_board(self) -> None:
        """
        Write a lightweight Kanban snapshot to pm/board.md for human review.
        """
        sections = {"todo": [], "needs_fix": [], "in_progress": [], "blocked": [], "done": []}
        for task in self.data["tasks"].values():
            sections.setdefault(task.get("status", "todo"), []).append(task)

        lines = ["# AutoDev Tickets\n"]
        for status in ["needs_fix", "in_progress", "todo", "blocked", "done"]:
            bucket = sections.get(status, [])
            if not bucket:
                continue
            lines.append(f"## {status.replace('_', ' ').title()} ({len(bucket)})")
            lines.append("| ID | Task | Priority | Attempts | Last Result |")
            lines.append("| --- | --- | --- | --- | --- |")
            for task in sorted(bucket, key=lambda t: t.get("priority", 999)):
                last = task.get("last_result")
                summary = "—"
                if last:
                    if last.get("tests_passed"):
                        summary = "tests ✅"
                    else:
                        summary = "tests ❌"
                lines.append(
                    f"| {task.get('id')} | {task.get('task')} | {task.get('priority', '')} | "
                    f"{task.get('attempts', 0)} | {summary} |"
                )
            lines.append("")  # blank line between sections

        try:
            self.board_file.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            pass
        self._export_formats()

    # ---------------- External export formats ----------------
    def _status_for_tool(self, status: str, tool: str) -> str:
        if tool in {"linear", "jira"}:
            mapping = {
                "todo": "todo",
                "needs_fix": "todo",
                "in_progress": "in-progress",
                "cooldown": "in-progress",
                "blocked": "blocked",
                "done": "done",
            }
            return mapping.get(status, "todo")
        if tool == "taskwarrior":
            return "completed" if status == "done" else "pending"
        return status

    def export(self, formats: Optional[List[str]] = None) -> List[Path]:
        """
        Public export entry point. Returns list of written paths.
        """
        return self._export_formats(formats)

    def _export_formats(self, only: Optional[List[str]] = None) -> List[Path]:
        """
        Emit lightweight export files for Linear/Jira CSV import and Taskwarrior JSONL.
        """
        tasks = list(self.data.get("tasks", {}).values())
        if not tasks:
            return []

        allowed = {fmt.lower() for fmt in only} if only else {"linear", "jira", "taskwarrior"}
        written: List[Path] = []

        # Linear-like CSV (can also suit Jira's basic CSV import)
        if "linear" in allowed:
            linear_path = self.export_dir / "linear.csv"
            fields = ["Title", "Description", "Status", "Priority", "Parent", "Attempts", "Tests"]
            try:
                with linear_path.open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fields)
                    writer.writeheader()
                    for task in tasks:
                        writer.writerow(
                            {
                                "Title": task.get("task", ""),
                                "Description": task.get("idea", "") or task.get("task", ""),
                                "Status": self._status_for_tool(task.get("status", "todo"), "linear"),
                                "Priority": task.get("priority", ""),
                                "Parent": task.get("parent", ""),
                                "Attempts": task.get("attempts", 0),
                                "Tests": "; ".join(task.get("tests", [])),
                            }
                        )
                written.append(linear_path)
            except OSError:
                pass

        # Jira-specific CSV (slightly different headers)
        if "jira" in allowed:
            jira_path = self.export_dir / "jira.csv"
            jira_fields = ["Summary", "Description", "Issue Type", "Status", "Priority", "Parent"]
            try:
                with jira_path.open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=jira_fields)
                    writer.writeheader()
                    for task in tasks:
                        writer.writerow(
                            {
                                "Summary": task.get("task", ""),
                                "Description": task.get("idea", "") or task.get("task", ""),
                                "Issue Type": "Task",
                                "Status": self._status_for_tool(task.get("status", "todo"), "jira"),
                                "Priority": task.get("priority", ""),
                                "Parent": task.get("parent", ""),
                            }
                        )
                written.append(jira_path)
            except OSError:
                pass

        # Taskwarrior JSONL
        if "taskwarrior" in allowed:
            tw_path = self.export_dir / "taskwarrior.jsonl"
            try:
                with tw_path.open("w", encoding="utf-8") as fh:
                    for task in tasks:
                        tw = {
                            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, task.get("id", task.get("task", "")))),
                            "description": task.get("task", ""),
                            "project": "autodev",
                            "tags": [task.get("status", "todo")],
                            "status": self._status_for_tool(task.get("status", "todo"), "taskwarrior"),
                            "priority": task.get("priority", ""),
                            "annotations": [task.get("idea", "")] if task.get("idea") else [],
                        }
                        fh.write(json.dumps(tw, ensure_ascii=False) + "\n")
                written.append(tw_path)
            except OSError:
                pass

        return written
