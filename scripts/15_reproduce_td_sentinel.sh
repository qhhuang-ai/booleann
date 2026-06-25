#!/usr/bin/env bash
# 15_reproduce_td_sentinel.sh -- T-D group-testing sentinel validation (Sec. 6.3).
# Reproduces: SIFT1M, C=16384 k-means cells, R=32 candidate cells,
# t in {8, 16, 24} (R >= t+1 holds), 100% no-omission + 100% exact-recovery.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOLEANN_ROOT="${BOOLEANN_ROOT:-$ROOT}"
cd "$ROOT"

python3 python/method/sanity_check_td_sentinel.py \
  --C-list 16384 \
  --R-list 32 \
  --t-list 8 16 24 \
  --n-queries 1000 \
  --selectivities 1e-4 5e-4 1e-3 5e-3 1e-2 \
  --delta-sentinel 0.05

echo "[15/td] results under $BOOLEANN_ROOT/03_experiment_bridge/results/raw/"
