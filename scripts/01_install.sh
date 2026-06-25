#!/usr/bin/env bash
# 01_install.sh -- install Python deps + build C++ port.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[01/install] Python dependencies"
python3 -m pip install --user -r requirements.txt

echo "[01/install] Cloning parlaylib for C++ work-stealing primitives"
PARLAY_DIR="${PARLAYANN_ROOT:-/tmp/parlaylib}"
if [ ! -d "$PARLAY_DIR" ]; then
  git clone --depth 1 https://github.com/cmuparlay/parlaylib.git "$PARLAY_DIR"
fi
export PARLAYANN_ROOT="$PARLAY_DIR"

echo "[01/install] Building C++ port"
mkdir -p cpp/build
cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPARLAYANN_ROOT="$PARLAY_DIR"
make -j"$(nproc)"

echo "[01/install] Done. Binaries in $ROOT/cpp/build/."
