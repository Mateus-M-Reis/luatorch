#!/usr/bin/env bash
# Build the luatorch native shim against the bundled libtorch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
JOBS="${JOBS:-2}"

if [[ ! -d "$ROOT/third_party/libtorch" ]]; then
  echo "error: third_party/libtorch not found." >&2
  echo "Download it with scripts/fetch_libtorch.sh (or set Torch_DIR)." >&2
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "built: $BUILD_DIR/libluatorch.so"
