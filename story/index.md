# Product Story (AutoDev)

Append-only log for LLM agents. Read newest-to-oldest. Never delete; add new files instead.

## AutoDev Origin and Vision (backstory)
- Story ID: 20251212_001225.110123Z--autodev-origin-and-vision
- When: 2025-12-12T00:12:25.110123Z
- Iteration: 0
- Status: done
- Tags: backstory, origin, vision
- Summary: AutoDev automates planner/coder loops with persistent narrative to prevent rework and preserve intent.
- Details:
```
{
  "problem": "LLMs repeat work without durable memory; context is limited.",
  "solution": "Append-only story log + context_view digests + STORY:<id> code markers.",
  "principles": [
    "append-only story",
    "context-efficient digests",
    "inline STORY markers",
    "free-tier friendly automation"
  ],
  "scope": "portable automation loop dropped into any repo"
}
```
