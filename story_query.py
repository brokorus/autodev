from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from story import StoryLog


def load_entries(story_dir: Path, limit: int) -> List[Dict]:
    log = StoryLog(story_dir)
    return log._read_entries(limit=limit)  # type: ignore[attr-defined]


def score(entry: Dict, q: str) -> int:
    ql = q.lower()
    fields = [
        entry.get("title", ""),
        entry.get("summary", ""),
        entry.get("task", ""),
        entry.get("task_id", ""),
        " ".join(entry.get("tags", [])),
        " ".join(entry.get("files", [])),
    ]
    return sum(field.lower().count(ql) for field in fields)


def summarize(entries: List[Dict], question: str) -> str:
    if not entries:
        return f"No story entries mention '{question}'. Try running more AutoDev iterations first."

    lines: List[str] = []
    lines.append(f"# Story summary for: {question}")
    lines.append("")
    lines.append("## Timeline (newest -> oldest)")
    for e in entries:
        parts = [
            f"{e.get('timestamp','')}",
            e.get("title", "Untitled"),
            f"[{e.get('status','')}]".strip("[]"),
        ]
        if e.get("tags"):
            parts.append("tags=" + ",".join(e["tags"]))
        if e.get("files"):
            parts.append("files=" + ",".join(e["files"]))
        lines.append("- " + " | ".join(p for p in parts if p))
        lines.append(f"  summary: {e.get('summary','').strip()}")
        details = e.get("details")
        if details:
            preview = details if isinstance(details, str) else json.dumps(details)[:240]
            lines.append(f"  details: {preview}")
    lines.append("")

    # Heuristic opinions
    statuses = {e.get("status") for e in entries if e.get("status")}
    tags = set().union(*(set(e.get("tags", [])) for e in entries))
    opinions: List[str] = []
    if any("fail" in (t or "").lower() for t in tags) or "needs_fix" in statuses:
        opinions.append("Stabilize failing areas: revisit failing test entries and close remediation tasks.")
    if not any("tests" in (t or "").lower() for t in tags):
        opinions.append("Add explicit test coverage references for this feature to reduce future regressions.")
    opinions.append("Re-read the earliest (big picture) entries to ensure current direction still matches the original intent.")
    lines.append("## Opinions / next steps")
    for op in opinions:
        lines.append(f"- {op}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AutoDev product story for a feature/question.")
    parser.add_argument("--query", required=True, help="Question or feature to summarize (e.g., 'model management').")
    parser.add_argument("--limit", type=int, default=30, help="Max entries to inspect.")
    args = parser.parse_args()

    story_dir = Path(__file__).parent / "story"
    entries = load_entries(story_dir, limit=args.limit)
    scored = sorted(entries, key=lambda e: score(e, args.query), reverse=True)
    # take top matches but keep order newest->oldest among them
    top = scored[:10]
    top_sorted = sorted(top, key=lambda e: e.get("timestamp", ""), reverse=True)
    output = summarize(top_sorted, args.query)
    print(output)


if __name__ == "__main__":
    main()
