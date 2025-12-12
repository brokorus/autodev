import json
import tempfile
import unittest
from pathlib import Path

from story import StoryLog


class StoryLogTests(unittest.TestCase):
    def test_add_entry_writes_jsonl_and_markdown(self):
        tmp = Path(tempfile.mkdtemp())
        log = StoryLog(tmp)

        entry_id = log.add_entry(
            kind="plan",
            title="Planner seed",
            summary="Created initial backlog.",
            iteration=1,
            task_id="task-1",
            task="Do something",
            status="todo",
            tests=["npm test"],
            tags=["plan", "seed"],
            files=["app.py"],
            details={"note": "backlog seeded"},
        )

        jsonl = tmp / "story.log.jsonl"
        self.assertTrue(jsonl.exists())
        md_files = list(tmp.glob("*--planner-seed.md"))
        self.assertTrue(md_files, "expected a markdown entry file")
        self.assertTrue(entry_id.startswith("20"), "entry id should include timestamp")

        # Ensure JSONL content is parseable and contains expected fields
        parsed = json.loads(jsonl.read_text().splitlines()[-1])
        self.assertEqual(parsed["kind"], "plan")
        self.assertEqual(parsed["task_id"], "task-1")
        self.assertIn("plan", parsed["tags"])
        self.assertIn("app.py", parsed["files"])

    def test_render_index_orders_recent_entries(self):
        tmp = Path(tempfile.mkdtemp())
        log = StoryLog(tmp)
        log.add_entry(kind="plan", title="Old", summary="old entry")
        log.add_entry(kind="tests", title="New", summary="new entry")

        digest = log.read_digest(limit=5)
        self.assertIn("Product Story", digest)
        # Newest entry should be present
        self.assertIn("New (tests)", digest)
        self.assertIn("Story ID", digest)

    def test_context_view_includes_headlines_and_big_picture(self):
        tmp = Path(tempfile.mkdtemp())
        log = StoryLog(tmp)
        for i in range(8):
            log.add_entry(kind="plan", title=f"Item {i}", summary=f"summary {i}", tags=["plan"])

        view = log.context_view(head=3, big_picture=2, max_chars=2000)
        self.assertIn("HEADLINES", view)
        self.assertIn("BIG PICTURE", view)
        self.assertIn("Item 7", view)  # newest
        self.assertIn("Item 0", view)  # oldest


if __name__ == "__main__":
    unittest.main()
