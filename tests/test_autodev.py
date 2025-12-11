import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = load_module("planner", BASE / "planner.py")
orchestrator = load_module("orchestrator", BASE / "orchestrator.py")
state_mod = load_module("state", BASE / "state.py")
AutoDevState = state_mod.AutoDevState


class PlannerTests(unittest.TestCase):
    def test_build_planner_cmd_includes_exec_and_read_only(self):
        tmp = Path(tempfile.gettempdir()) / "codex_last_msg.txt"
        cmd = planner.build_planner_cmd(tmp)
        flattened = " ".join(cmd)
        self.assertIn("codex", flattened)
        self.assertIn("--ask-for-approval", cmd)
        self.assertIn("never", cmd)
        self.assertIn("exec", cmd)
        self.assertIn("--sandbox", cmd)
        self.assertIn("read-only", cmd)
        self.assertIn("--output-last-message", cmd)
        self.assertIn(str(tmp), cmd)

    def test_codex_returns_fake_output_and_logs(self):
        fake_output = '{"ok": true}'
        prev_env = os.environ.get("AUTODEV_FAKE_CODEX_OUTPUT")
        os.environ["AUTODEV_FAKE_CODEX_OUTPUT"] = fake_output
        try:
            # isolate logs for the test
            tmp_logs = Path(tempfile.mkdtemp())
            tmp_logs.mkdir(parents=True, exist_ok=True)
            prev_logs = planner.LOGS
            planner.LOGS = tmp_logs  # type: ignore
            try:
                result = planner.codex("ignored prompt")
            finally:
                planner.LOGS = prev_logs  # type: ignore
        finally:
            if prev_env is None:
                os.environ.pop("AUTODEV_FAKE_CODEX_OUTPUT", None)
            else:
                os.environ["AUTODEV_FAKE_CODEX_OUTPUT"] = prev_env

        self.assertEqual(result, fake_output)
        log_file = tmp_logs / "codex_planner_last.log"
        last_msg_file = tmp_logs / "codex_planner_last_message.txt"
        self.assertTrue(log_file.exists())
        self.assertTrue(last_msg_file.exists())
        self.assertEqual(last_msg_file.read_text().strip(), fake_output)


class OrchestratorTests(unittest.TestCase):
    def test_snapshot_ignores_autodev_files(self):
        base = Path(tempfile.mkdtemp())
        keep = base / "keep.txt"
        ignore_dir = base / ".autodev"
        ignore_dir.mkdir()
        secret = ignore_dir / "secret.json"
        keep.write_text("hello world", encoding="utf-8")
        secret.write_text("secret", encoding="utf-8")

        snap = orchestrator.snapshot(base=base, max_chars=1000)
        self.assertIn("keep.txt", snap)
        self.assertIn("hello world", snap)
        self.assertNotIn(".autodev", snap)

    def test_git_checkpoint_manager_creates_commit(self):
        repo = Path(tempfile.mkdtemp())
        logs_dir = repo / "logs"
        subprocess.run(["git", "init"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "autodev@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "AutoDev"], cwd=repo, check=True)
        target = repo / "file.txt"
        target.write_text("initial", encoding="utf-8")
        subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)

        target.write_text("updated", encoding="utf-8")
        manager = orchestrator.GitCheckpointManager(repo, logs_dir / "checkpoints.jsonl")
        commit_hash = manager.create_checkpoint("unit-test checkpoint", metadata={"iteration": 1}, allow_empty=False)

        self.assertIsNotNone(commit_hash)
        log_file = logs_dir / "checkpoints.jsonl"
        self.assertTrue(log_file.exists())
        log_content = log_file.read_text()
        self.assertIn("unit-test checkpoint", log_content)
        self.assertIn(commit_hash, subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout)

    def test_run_rigorous_tests_handles_missing_command(self):
        prev_env = os.environ.get("AUTODEV_TEST_COMMANDS")
        os.environ["AUTODEV_TEST_COMMANDS"] = "nonexistent-autodev-test-command"
        tmp_logs = Path(tempfile.mkdtemp())
        prev_logs = orchestrator.LOGS
        orchestrator.LOGS = tmp_logs  # type: ignore
        try:
            outcome = orchestrator.run_rigorous_tests(1, "dummy-task", base=tmp_logs)
        finally:
            orchestrator.LOGS = prev_logs  # type: ignore
            if prev_env is None:
                os.environ.pop("AUTODEV_TEST_COMMANDS", None)
            else:
                os.environ["AUTODEV_TEST_COMMANDS"] = prev_env

        self.assertTrue(outcome["ran"])
        self.assertFalse(outcome["all_passed"])
        self.assertNotEqual(outcome["results"][0]["returncode"], 0)


class StateTests(unittest.TestCase):
    def test_state_tracks_fix_tasks_and_board(self):
        tmp = Path(tempfile.mkdtemp())
        pm_dir = tmp / "pm"
        state = AutoDevState(pm_dir)

        backlog = [
            {
                "task": "Ship feature X",
                "tests": ["npm test"],
                "priority": 1,
                "prerequisites": {"met": True, "missing": []},
            }
        ]
        state.sync_backlog(backlog, [])
        state.render_board()
        board_path = pm_dir / "board.md"
        self.assertTrue(board_path.exists())

        task_id, task = state.next_actionable()
        fake_outcome = {
            "ran": True,
            "all_passed": False,
            "results": [
                {"command": "npm test", "returncode": 1, "stdout": "", "stderr": "failed test", "duration_seconds": 1}
            ],
        }
        state.mark_failed(task_id, 1, fake_outcome)
        fixes = state.create_fix_tasks(task_id, task["task"], fake_outcome)
        state.add_tasks(fixes)

        self.assertGreaterEqual(len(fixes), 1)
        next_id, next_task = state.next_actionable()
        self.assertTrue(next_id.startswith(task_id))
        self.assertEqual(next_task["status"], "todo")

        # Export files should exist and be non-empty
        linear = pm_dir / "export" / "linear.csv"
        jira = pm_dir / "export" / "jira.csv"
        tw = pm_dir / "export" / "taskwarrior.jsonl"
        self.assertTrue(linear.exists())
        self.assertTrue(jira.exists())
        self.assertTrue(tw.exists())
        self.assertGreater(len(linear.read_text()), 0)
        self.assertGreater(len(tw.read_text()), 0)

    def test_export_can_filter_formats(self):
        tmp = Path(tempfile.mkdtemp())
        pm_dir = tmp / "pm"
        state = AutoDevState(pm_dir)
        state.add_tasks(
            [
                {
                    "task": "Filter export test",
                    "tests": [],
                    "priority": 1,
                    "prerequisites": {"met": True, "missing": []},
                }
            ]
        )
        # Clear any auto-generated exports from previous saves
        export_dir = pm_dir / "export"
        if export_dir.exists():
            for f in export_dir.glob("*"):
                f.unlink()
        written = state.export(["linear"])
        self.assertTrue(any(str(p).endswith("linear.csv") for p in written))
        self.assertFalse((pm_dir / "export" / "jira.csv").exists())
        self.assertFalse((pm_dir / "export" / "taskwarrior.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
