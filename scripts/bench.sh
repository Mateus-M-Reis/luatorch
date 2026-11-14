#!/usr/bin/env bash
# Run the micro-benchmarks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LUA_PATH="$ROOT/lua/?.lua;$ROOT/lua/?/init.lua;;"
luajit "$ROOT/bench/bench.lua" "$@"
luajit "$ROOT/bench/compare_moses.lua"
