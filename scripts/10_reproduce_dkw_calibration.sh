#!/usr/bin/env bash
# 10_reproduce_dkw_calibration.sh -- DKW calibration of epsilon_corr (Lemma 3).
# Reproduces: Sec. 6.4, epsilon_corr <= 0.075 at m = 594.
#
# Notes
# -----
# - dkw_calibration_yfcc10m.py uses module-level constants (NB, DIM, NQ, M,
#   EFC, EFS, SLICE_START) -- no CLI flags. Edit those constants directly
#   to change the calibration parameters.
# - Output is written to $BOOLEANN_ROOT/03_experiment_bridge/results/raw/
#   dkw_calibration_yfcc10m/summary.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOLEANN_ROOT="${BOOLEANN_ROOT:-$ROOT}"
cd "$ROOT"

python3 python/method/dkw_calibration_yfcc10m.py

OUT="$BOOLEANN_ROOT/03_experiment_bridge/results/raw/dkw_calibration_yfcc10m/summary.json"
echo "[10/dkw] summary: $OUT"
[ -f "$OUT" ] && python3 -m json.tool < "$OUT"
