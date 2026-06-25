#!/usr/bin/env bash
# 13_reproduce_sift1b.sh -- SIFT1B scale-out (Table 2).
# Reproduces: 0.995 recall@10 at 5.7 ms.
# Disk requirement: ~250 GB after decompression.
#
# hamcg_sift1b_clean.py uses module-level constants and runs top-down
# (no argparse). Edit constants in the script directly to change config.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOLEANN_ROOT="${BOOLEANN_ROOT:-$ROOT}"
cd "$ROOT"

python3 python/method/hamcg_sift1b_clean.py

echo "[13/sift1b] results under $BOOLEANN_ROOT/03_experiment_bridge/results/raw/"
