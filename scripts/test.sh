#!/usr/bin/env bash
# Run the luatorch test suite against the locally built shim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LUA_PATH="$ROOT/lua/?.lua;$ROOT/lua/?/init.lua;;"
exec luajit "$ROOT/tests/run.lua" "$@"
