#!/usr/bin/env bash
# Install the Hinaing backend environment so `hinaing-eval run-fixtures` can
# actually execute the LangGraph pipeline (otherwise imports fail with
# `ModuleNotFoundError: No module named 'langgraph'`).
#
# Usage: scripts/setup_backend_env.sh [path-to-hinaing-repo]
set -euo pipefail

HINAING_ROOT="${1:-../Hinaing}"
BACKEND="$HINAING_ROOT/backend"

if [ ! -d "$BACKEND" ]; then
  echo "Hinaing backend not found at $BACKEND" >&2
  echo "Pass the Hinaing repo path as the first argument." >&2
  exit 1
fi

if ! command -v poetry >/dev/null 2>&1; then
  echo "poetry is required. Install from https://python-poetry.org/docs/#installation" >&2
  exit 1
fi

echo "Installing Hinaing backend deps in $BACKEND ..."
( cd "$BACKEND" && poetry install )

echo
echo "Verifying the env imports generate_snapshot ..."
( cd "$BACKEND" && poetry run python -c "from app.services.insights.graph import generate_snapshot; print('ok')" )

echo
echo "Run fixtures with:"
echo "  uv run hinaing-eval run-fixtures \\"
echo "    --hinaing-root \"$HINAING_ROOT\" \\"
echo "    --python-cmd \"poetry -C $BACKEND run python\""
