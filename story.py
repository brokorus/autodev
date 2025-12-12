from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _slugify(text: str, length: int = 60) -> str:
    """Create a filesystem-safe slug for filenames."""
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    if not slug:
        slug = "entry"
    return slug[:length] or "entry"


class StoryLog:
    """
    Append-only narrative log for AutoDev.
    Each entry writes to JSONL plus a timestamped Markdown note to keep LLMs aligned
    on what happened, why, and what to avoid repeating.
    """

    def __init__(self, story_dir: Path, index_limit: int = 40):
        self.story_dir = story_dir
        self.story_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.story_dir / "story.log.jsonl"
        self.index_file = self.story_dir / "index.md"
        self.index_limit = index_limit

    # --------------- public API ---------------
    def add_entry(
        self,
        kind: str,
        title: str,
        summary: str,
        *,
        iteration: Optional[int] = None,
        task_id: Optional[str] = None,
        task: Optional[str] = None,
        status: Optional[str] = None,
        tests: Optional[Iterable[str]] = None,
        details: Optional[Any] = None,
        tags: Optional[Iterable[str]] = None,
        files: Optional[Iterable[str]] = None,
        related: Optional[str] = None,
    ) -> str:
        """
        Persist a new story entry and refresh the digest.
        Returns the unique entry_id for cross-referencing in code comments.
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        entry_id = f"{timestamp.replace(':', '').replace('-', '').replace('T', '_').replace('Z', 'Z')}--{_slugify(title, 30)}"
        entry: Dict[str, Any] = {
            "id": entry_id,
            "timestamp": timestamp,
            "kind": kind,
            "title": title,
            "summary": summary,
        }
        if iteration is not None:
            entry["iteration"] = iteration
        if task_id:
            entry["task_id"] = task_id
        if task:
            entry["task"] = task
        if status:
            entry["status"] = status
        if tests:
            entry["tests"] = list(tests)
        if details is not None:
            entry["details"] = details
        if tags:
            entry["tags"] = sorted(set(tags))
        if files:
            entry["files"] = sorted(set(files))
        if related:
            entry["related"] = related

        self._append_jsonl(entry)
        self._write_markdown(entry)
        self.render_index(limit=self.index_limit)
        return entry_id

    def render_index(self, limit: int = 20) -> Path:
        """Refresh the human/LLM-friendly digest of recent entries."""
        entries = self._read_entries(limit=limit)
        lines = [
            "# Product Story (AutoDev)",
            "",
            "Append-only log for LLM agents. Read newest-to-oldest. Never delete; add new files instead.",
            "",
        ]
        for entry in entries:
            lines.append(f"## {entry.get('title', 'Untitled')} ({entry.get('kind', 'event')})")
            lines.append(f"- Story ID: {entry.get('id', '')}")
            lines.append(f"- When: {entry.get('timestamp', '')}")
            if entry.get("iteration") is not None:
                lines.append(f"- Iteration: {entry['iteration']}")
            if entry.get("task"):
                lines.append(f"- Task: {entry['task']}")
            if entry.get("task_id"):
                lines.append(f"- Task ID: {entry['task_id']}")
            if entry.get("status"):
                lines.append(f"- Status: {entry['status']}")
            if entry.get("tests"):
                lines.append(f"- Tests: {', '.join(entry['tests'])}")
            if entry.get("tags"):
                lines.append(f"- Tags: {', '.join(entry['tags'])}")
            if entry.get("files"):
                lines.append(f"- Files: {', '.join(entry['files'])}")
            if entry.get("related"):
                lines.append(f"- Related: {entry['related']}")
            lines.append(f"- Summary: {entry.get('summary', '').strip()}")
            details = entry.get("details")
            if details:
                if isinstance(details, str):
                    lines.append(f"- Details: {details.strip()}")
                else:
                    details_json = json.dumps(details, indent=2, ensure_ascii=False)
                    lines.append("- Details:")
                    lines.append(f"```\n{details_json}\n```")
            lines.append("")

        self.index_file.write_text("\n".join(lines), encoding="utf-8")
        return self.index_file

    def read_digest(self, limit: int = 10) -> str:
        """Return the digest contents (refreshing it if missing)."""
        if not self.index_file.exists():
            self.render_index(limit=limit)
        try:
            return self.index_file.read_text(encoding="utf-8")
        except OSError:
            return ""

    def context_view(
        self,
        *,
        head: int = 6,
        big_picture: int = 4,
        tags: Optional[Iterable[str]] = None,
        max_chars: int = 4000,
    ) -> str:
        """
        Provide a concise but big-picture-aware view:
        - HEADLINES: newest `head` entries (optionally tag-filtered)
        - BIG PICTURE: oldest `big_picture` entries to retain long-term intent
        Trims to `max_chars` to keep prompt budgets sane.
        """
        tag_set = {t.lower() for t in tags} if tags else None
        entries_all = self._read_entries(limit=200)

        def matches(entry: Dict[str, Any]) -> bool:
            if not tag_set:
                return True
            entry_tags = {t.lower() for t in entry.get("tags", [])}
            return bool(entry_tags & tag_set)

        filtered = [e for e in entries_all if matches(e)]
        headlines = filtered[:head]
        big = list(reversed(filtered[-big_picture:])) if filtered else []

        def fmt(entry: Dict[str, Any]) -> str:
            parts = [
                f"{entry.get('timestamp', '')}",
                entry.get("title", "Untitled"),
                f"id={entry.get('id', '')}",
            ]
            if entry.get("status"):
                parts.append(f"status={entry['status']}")
            if entry.get("tags"):
                parts.append("tags=" + ",".join(entry["tags"]))
            if entry.get("files"):
                parts.append("files=" + ",".join(entry["files"]))
            parts.append("summary=" + entry.get("summary", "").strip())
            return " | ".join(parts)

        lines = ["HEADLINES (newest first):"]
        lines += [f"- {fmt(e)}" for e in headlines] or ["- (none)"]
        lines.append("")
        lines.append("BIG PICTURE (oldest key intents):")
        lines += [f"- {fmt(e)}" for e in big] or ["- (none)"]
        text = "\n".join(lines)
        return text[:max_chars]

    # --------------- internals ---------------
    def _append_jsonl(self, entry: Dict[str, Any]) -> None:
        try:
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # Story logging must never break the main loop; swallow quietly.
            pass

    def _write_markdown(self, entry: Dict[str, Any]) -> None:
        title = entry.get("title") or "entry"
        timestamp = entry.get("timestamp") or datetime.utcnow().isoformat() + "Z"
        entry_id = entry.get("id") or f"{timestamp.replace(':', '').replace('-', '').replace('T', '_').replace('Z', 'Z')}--{_slugify(title)}"
        name = f"{entry_id}.md"
        path = self.story_dir / name
        lines = [
            f"# {title}",
            "",
            f"- When: {timestamp}",
            f"- Kind: {entry.get('kind', 'event')}",
            f"- Story ID: {entry_id}",
        ]
        if entry.get("iteration") is not None:
            lines.append(f"- Iteration: {entry['iteration']}")
        if entry.get("task"):
            lines.append(f"- Task: {entry['task']}")
        if entry.get("task_id"):
            lines.append(f"- Task ID: {entry['task_id']}")
        if entry.get("status"):
            lines.append(f"- Status: {entry.get('status')}")
        if entry.get("tests"):
            lines.append(f"- Tests: {', '.join(entry['tests'])}")
        if entry.get("tags"):
            lines.append(f"- Tags: {', '.join(entry['tags'])}")
        if entry.get("files"):
            lines.append(f"- Files: {', '.join(entry['files'])}")
        if entry.get("related"):
            lines.append(f"- Related: {entry.get('related')}")
        lines.append("")
        lines.append("## Summary")
        lines.append(entry.get("summary", "").strip())
        lines.append("")
        details = entry.get("details")
        if details:
            lines.append("## Details")
            if isinstance(details, str):
                lines.append(details.strip())
            else:
                lines.append("```")
                lines.append(json.dumps(details, indent=2, ensure_ascii=False))
                lines.append("```")
        try:
            path.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            pass

    def _read_entries(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        try:
            lines = self.log_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        entries: List[Dict[str, Any]] = []
        for line in reversed(lines):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(entries) >= limit:
                break
        return entries

    # --------------- helpers ---------------
    @staticmethod
    def format_story_comment(entry_id: str) -> str:
        """
        Canonical marker for inline code comments to link code to narrative.
        Example usage in code: `// STORY:20251211_1530...`
        """
        return f"STORY:{entry_id}"
