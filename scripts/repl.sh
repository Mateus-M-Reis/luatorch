#!/usr/bin/env bash
# Open an interactive LuaJIT REPL with `torch` preloaded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LUA_PATH="$ROOT/lua/?.lua;$ROOT/lua/?/init.lua;;"
exec luajit -e 'torch = require("torch"); fun = require("fun")' -i
