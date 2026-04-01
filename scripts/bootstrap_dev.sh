#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf "Expected virtualenv interpreter at %s\n" "$PYTHON_BIN" >&2
  printf "Create it first, e.g. python3.12 -m venv .venv\n" >&2
  exit 1
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" -m pip install -r requirements-dev.txt
git config core.hooksPath .githooks
"$PYTHON_BIN" -m pre_commit install-hooks

printf "Development bootstrap complete.\n"
printf "Git hooks path: %s\n" "$(git config --get core.hooksPath)"
