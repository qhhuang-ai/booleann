#!/usr/bin/env bash
# 12_reproduce_sift100m.sh -- SIFT100M scale-out (Table 2).
# Reproduces: 0.999 recall@10 at 3.6 ms, 8 conjunction pairs, single-host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOLEANN_ROOT="${BOOLEANN_ROOT:-$ROOT}"
cd "$ROOT"

# 1. Prepare SIFT100M shards + conjunction data + GT.
echo "[12/sift100m] step 1/4: prepare base float32 + 12-predicate manifest"
python3 python/data_prep/prepare_sift100m_unified.py

echo "[12/sift100m] step 2/4: build conjunction pairs"
python3 python/data_prep/sift100m_build_conjunction_data.py

echo "[12/sift100m] step 3/4: generate conjunction GT (GPU brute-force)"
python3 python/data_prep/sift100m_gen_conjunction_gt.py

# 2. ARWGI deployment + HAMCG inner HNSW.
echo "[12/sift100m] step 4/4: ARWGI deployment (build_hnsw -> build_wv -> bench)"
python3 python/method/arwgi_sift100m_deploy.py --stage all --r 24

echo "[12/sift100m] results under $BOOLEANN_ROOT/03_experiment_bridge/results/raw/"
