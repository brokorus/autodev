from __future__ import annotations

import argparse
import sys
from pathlib import Path

from state import AutoDevState


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export AutoDev PM board to Linear/Jira CSV or Taskwarrior JSONL."
    )
    parser.add_argument(
        "--format",
        choices=["linear", "jira", "taskwarrior", "all"],
        default="all",
        help="Which export format to generate.",
    )
    args = parser.parse_args(argv)

    pm_dir = Path(__file__).parent / "pm"
    state = AutoDevState(pm_dir)
    formats = None if args.format == "all" else [args.format]
    paths = state.export(formats)

    if not paths:
        print("No tasks to export.", file=sys.stderr)
        return 0

    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
