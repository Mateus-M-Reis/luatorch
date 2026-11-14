#!/usr/bin/env bash
# Compare luatorch against PyTorch on this machine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv-bench/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON" ]]; then
  echo "error: no Python interpreter found" >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/bench/compare.py" "$@"
