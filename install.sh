#!/usr/bin/env bash
set -euo pipefail

REPO="${AUTODEV_REPO:-https://github.com/brokorus/autodev.git}"
BRANCH="${AUTODEV_BRANCH:-main}"
TARGET="${AUTODEV_TARGET:-$(pwd)}"
FORCE="${AUTODEV_FORCE:-0}"

info() {
  printf "[AutoDev Install] %s\n" "$*"
}

confirm_overwrite() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    return
  fi

  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$path"
    return
  fi

  read -r -p "Found existing $path. Overwrite? [y/N] " reply
  case "$reply" in
    [Yy]* ) rm -rf "$path" ;;
    * ) echo "Aborted at user request; leaving $path unchanged."; exit 1 ;;
  esac
}

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required but was not found on PATH. Please install Git and rerun." >&2
  exit 1
fi

tmp="$(mktemp -d 2>/dev/null || mktemp -d -t autodev)"
trap 'rm -rf "$tmp"' EXIT

info "Installing AutoDev into $TARGET"
info "Using repo $REPO (branch: $BRANCH)"

git clone --depth 1 --branch "$BRANCH" "$REPO" "$tmp" >/dev/null

if [[ ! -d "$tmp/.autodev" ]]; then
  echo "Cloned repo missing .autodev folder; aborting." >&2
  exit 1
fi

confirm_overwrite "$TARGET/.autodev"
mkdir -p "$TARGET/.autodev"
cp -R "$tmp/.autodev/." "$TARGET/.autodev"

for file in autodev.ps1 AUTODEV_README.md; do
  if [[ -f "$tmp/$file" ]]; then
    confirm_overwrite "$TARGET/$file"
    cp "$tmp/$file" "$TARGET/$file"
  fi
done

info "AutoDev files copied."
echo
echo "Next steps:"
echo "1) Review .autodev/requirements-autodev.txt if you want to pin/adjust packages."
echo "2) Run ./autodev.ps1 with PowerShell (pwsh or powershell) to bootstrap AutoDev."
echo
echo "Tip: set AUTODEV_FORCE=1 to overwrite without prompting."
