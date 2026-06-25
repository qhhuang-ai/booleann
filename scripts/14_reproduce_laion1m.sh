#!/usr/bin/env bash
# 14_reproduce_laion1m.sh -- LAION1M conjunctions.
# Reproduces: BCI on LAION1M with CLIP features + top-200 tags.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOLEANN_ROOT="${BOOLEANN_ROOT:-$ROOT}"
cd "$ROOT"

# 1. Prepare LAION1M shards + richer filters (top-~400 tags).
echo "[14/laion1m] step 1/2: prepare LAION1M shards + richer filters"
python3 python/data_prep/laion_prepare.py
python3 python/data_prep/laion_richer_filters.py

# 2. BCI-HAMCG on LAION1M (richer filter mode).
echo "[14/laion1m] step 2/2: BCI-HAMCG bench (mode=bci)"
python3 python/method/bci_hamcg_laion1m_richer.py \
  --mode bci \
  --threads 8 \
  --inner-threads 1 \
  --build-threads 16 \
  --pair-budget-mult 2.0

echo "[14/laion1m] results under $BOOLEANN_ROOT/03_experiment_bridge/results/raw/"
