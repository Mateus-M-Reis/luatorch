#!/bin/sh
set -eu
ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
PREFIX=${PREFIX:-"$HOME/.local"}
cd "$ROOT"
./scripts/build.sh
cmake --install build --prefix "$PREFIX"
printf 'installed luatorch to %s\n' "$PREFIX"
printf 'LUA_PATH="%s/share/lua/5.1/?.lua;%s/share/lua/5.1/?/init.lua;;"\n' "$PREFIX" "$PREFIX"
