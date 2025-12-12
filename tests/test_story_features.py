import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

import story_query
from story import StoryLog
import planner
import orchestrator


def test_story_entry_captures_tags_files_and_id():
    tmp = Path(tempfile.mkdtemp())
    log = StoryLog(tmp)

    entry_id = log.add_entry(
        kind="code",
        title="Added model manager",
        summary="Implemented model manager with caching.",
        tags=["code", "model"],
        files=["llm_provider.py", "model_catalog.json"],
    )

    parsed = json.loads((tmp / "story.log.jsonl").read_text().splitlines()[-1])
    assert parsed["id"] == entry_id
    assert "model" in parsed["tags"]
    assert "llm_provider.py" in parsed["files"]


def test_context_view_balances_headlines_and_big_picture():
    tmp = Path(tempfile.mkdtemp())
    log = StoryLog(tmp)
    for i in range(6):
        log.add_entry(kind="plan", title=f"plan {i}", summary=f"sum {i}", tags=["plan"])
    view = log.context_view(head=2, big_picture=2, max_chars=500)
    assert "HEADLINES" in view
    assert "BIG PICTURE" in view
    assert "plan 5" in view  # newest
    assert "plan 0" in view  # oldest
    assert len(view) <= 500


def test_story_query_summarize_includes_question_and_opinions():
    tmp = Path(tempfile.mkdtemp())
    log = StoryLog(tmp)
    log.add_entry(kind="tests", title="Model tests failed", summary="fix model", tags=["fail", "model"])
    entries = story_query.load_entries(tmp, limit=10)
    scored = sorted(entries, key=lambda e: story_query.score(e, "model"), reverse=True)
    output = story_query.summarize(scored[:5], "model")
    assert "model" in output.lower()
    assert "Opinions" in output


def test_planner_includes_story_digest_in_prompt(monkeypatch):
    captured = {}

    def fake_plan(prompt: str):
        captured["prompt"] = prompt
        return json.dumps({"backlog": []})

    tmp_story = Path(tempfile.mkdtemp())
    planner.STORY = StoryLog(tmp_story)
    planner.STORY.add_entry(kind="plan", title="Init", summary="seed", tags=["init"])

    with mock.patch.object(planner.LLMProvider, "plan", side_effect=fake_plan):
        planner.create_backlog("snapshot text")

    assert "PRODUCT STORY" in captured["prompt"]
    assert "HEADLINES" in captured["prompt"]


def test_implement_includes_story_ref_in_prompt(monkeypatch):
    captured = {}

    def fake_code(prompt: str):
        captured["prompt"] = prompt
        return "ok"

    with mock.patch.object(orchestrator.LLMProvider, "code", side_effect=fake_code):
        orchestrator.implement(
            {"task": "Do X", "tests": []},
            board_snapshot="board",
            expertise="expert",
            story_ref="STORY:123",
        )

    assert "STORY:123" in captured["prompt"]
    assert "inline comment" in captured["prompt"]


if __name__ == "__main__":
    pytest.main([__file__])
