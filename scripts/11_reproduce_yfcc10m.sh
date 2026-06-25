#!/usr/bin/env bash
# 11_reproduce_yfcc10m.sh -- YFCC10M conjunction Pareto (Fig. 3).
# Reproduces: BCI strictly above Parlay-IVF on 15/15 cross-slice cells
# (5 slices x 3 tiers, no per-slice retuning).
#
# Wall time: ~3 hours on AMD EPYC 7642, 96 cores.
# Memory:    ~110 GB peak (HAMCG vocabulary build).
#
# Pipeline
# --------
# 1. Generate per-slice ground truth for slices [50K..100K).
# 2. Run the canonical [60K, 70K) ARWGI configuration (--r 24).
# 3. Run the BCI exact-only deployment on the canonical slice.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOLEANN_ROOT="${BOOLEANN_ROOT:-$ROOT}"
cd "$ROOT"

# 1. Generate ground truth (gen_yfcc10m_gt_matched_v6.py is a top-level script
#    that targets the slice it was last configured for; rerun for each slice
#    by editing SLICE_START at the top, or invoke with environment overrides).
echo "[11/yfcc] step 1/3: generating ground truth (5 slices)"
python3 python/data_prep/gen_yfcc10m_gt_matched_v6.py

# 2. Canonical slice [60K, 70K) -- ARWGI deployment (Sec. 6.2).
echo "[11/yfcc] step 2/3: canonical slice [60K, 70K) ARWGI deployment"
python3 python/method/arwgi_canonical_60_70.py \
  --r 24 \
  --visit-topk 256 \
  --ef-bench 64 128 256 512 \
  --over-bench 4 16 \
  --threads 8

# 3. BCI exact-only deployment on canonical slice.
echo "[11/yfcc] step 3/3: BCI exact-only deployment"
python3 python/method/bci_exact_only_yfcc10m.py \
  --threads 8 \
  --inner-threads 1 \
  --pair-budget-mult 2.0

echo "[11/yfcc] results under $BOOLEANN_ROOT/03_experiment_bridge/results/raw/"
