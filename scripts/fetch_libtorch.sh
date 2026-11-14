#!/usr/bin/env bash
# Fetch the official CPU libtorch distribution used by the native shim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${LIBTORCH_VERSION:-2.5.1}"
VARIANT="${LIBTORCH_VARIANT:-cpu}"
ABI="${LIBTORCH_ABI:-cxx11-abi-shared-with-deps}"
NAME="libtorch-${ABI}-${VERSION}%2B${VARIANT}.zip"
URL="https://download.pytorch.org/libtorch/${VARIANT}/${NAME}"
DEST="$ROOT/third_party"

mkdir -p "$DEST"
if [[ -d "$DEST/libtorch" ]]; then
  echo "third_party/libtorch already exists; delete it to re-download."
  exit 0
fi

echo "downloading $URL"
curl -L --fail --progress-bar -o "$DEST/libtorch.zip" "$URL"
unzip -q "$DEST/libtorch.zip" -d "$DEST"
rm -f "$DEST/libtorch.zip"
echo "libtorch extracted to $DEST/libtorch"
