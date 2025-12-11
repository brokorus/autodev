from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List


def _read_codex_path(base_dir: Path) -> str | None:
    """
    Read a user-specified Codex CLI path from codex_path.txt if present.
    """
    path_file = base_dir / "codex_path.txt"
    try:
        value = path_file.read_text(encoding="utf-8", errors="ignore").strip()
        return value or None
    except OSError:
        return None


def resolve_codex_command(base_dir: Path) -> List[str]:
    """
    Resolve the Codex CLI executable to something subprocess can launch.
    Prefers an explicit path (env var or codex_path.txt), otherwise falls
    back to PATH lookups. Handles PowerShell scripts on Windows.
    """
    candidates: list[str] = []
    for value in (
        os.getenv("CODEX_CLI_PATH"),
        os.getenv("AUTODEV_CODEX_PATH"),
        _read_codex_path(base_dir),
        shutil.which("codex"),
        shutil.which("codex.cmd"),
    ):
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.exists():
            continue
        # Prefer the .cmd shim when the user pointed at the PowerShell script
        if path.suffix.lower() == ".ps1":
            cmd_sibling = path.with_suffix(".cmd")
            if cmd_sibling.exists():
                path = cmd_sibling
        candidates.append(str(path))

    for path in candidates:
        # PowerShell script shim
        if path.lower().endswith(".ps1"):
            return [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                path,
                "--%",
            ]
        return [path]

    raise FileNotFoundError("Codex CLI is not installed or not on PATH")
