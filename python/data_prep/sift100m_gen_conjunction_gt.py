#!/usr/bin/env python3
"""Generate exact conjunction GT for SIFT100M 2-tag BCI bench.

For each of 200 active queries (8 pairs × 25 each):
  - Look up which 2 tags this query carries
  - Compute filtered subset = {points where both tags match}
  - Exact top-K via faiss IndexFlatL2 over subset (in float32)
  - Map local back to global IDs

Output: <BOOLEANN_ROOT>/data/raw/sift100m/bci/sift100m_2tag_conj_gt.bin
         (uint32 N + uint32 K + N*K uint32 global ids) — LAION-style; loader will
         look for sift100m_2tag_conj_gt.dist.bin sidecar for TIE_AWARE distances.
"""
import numpy as np
import struct
import time
import os
from pathlib import Path
import faiss

ROOT = Path(os.environ.get('BOOLEANN_ROOT', Path(__file__).resolve().parents[2]))
SIFT_DIR = ROOT / 'data/raw/sift100m/bci'
DA_DIR = ROOT / 'baselines/MS_DiskANN/datasets/sift100m'
SIFT_F32_RAW = ROOT / 'data/raw/sift100m/sift100m_base_f32.raw'
SIFT_QUERY = ROOT / 'data/raw/sift/sift_query.fvecs'

N = 100_000_000
DIM = 128
NQ_TOTAL = 10_000
NQ_ACTIVE = 200
K = 10
THREADS = 96
faiss.omp_set_num_threads(THREADS)

# -- Step 1: read predicates manifest + load matching-row indices per predicate --
preds = []
with open(DA_DIR / 'predicates_manifest_gt.txt') as f:
    f.readline()
    for line in f:
        s_target, pred_i, target_label, actual_s, label_file, _ = line.strip().split(',')
        preds.append(dict(target_label=int(target_label),
                           label_file=label_file,
                           pred_idx=len(preds)))

print(f"[manifest] {len(preds)} predicates", flush=True)
# Cache valid_idx per predicate
print("[masks] loading all 12 predicate valid-idx sets...", flush=True)
valid_idx_cache = []
for p in preds:
    t0 = time.time()
    arr = np.loadtxt(p['label_file'], dtype=np.int32)
    idx = np.where(arr == p['target_label'])[0].astype(np.int64)
    print(f"  pred_idx={p['pred_idx']}: |valid|={len(idx)} ({time.time()-t0:.1f}s)", flush=True)
    valid_idx_cache.append(idx)
    del arr

# -- Step 2: parse conjunction_pairs.txt and figure out which 2 tags each query has --
pairs = []   # list of (a_pred_idx, b_pred_idx, joint_count)
with open(SIFT_DIR / 'conjunction_pairs.txt') as f:
    f.readline()
    for line in f:
        toks = line.strip().split(',')
        pair = int(toks[0]); a = int(toks[1]); b = int(toks[2]); jc = int(toks[7])
        pairs.append((a, b, jc))
print(f"[pairs] {len(pairs)} conjunctions; sizes: {[p[2] for p in pairs]}", flush=True)

# Per-query: 25 queries per pair, total 200
n_per_pair = NQ_ACTIVE // len(pairs)
assert n_per_pair == 25

# -- Step 3: load base via memmap (NO copy into RAM; 51GB) and queries --
print("[base] memmap base + queries...", flush=True)
base = np.memmap(SIFT_F32_RAW, dtype=np.float32, mode='r', shape=(N, DIM))
def read_fvecs(p, max_n):
    with open(p, 'rb') as f:
        data = f.read()
    d = struct.unpack('<i', data[:4])[0]
    n = min(len(data) // (4 + d*4), max_n)
    return np.frombuffer(data, dtype=np.float32).reshape(-1, d+1)[:n, 1:].copy()
queries = read_fvecs(SIFT_QUERY, NQ_TOTAL).astype(np.float32, copy=False)
print(f"  base shape={base.shape}, queries={queries.shape}", flush=True)

# -- Step 4: for each pair, materialize conjunction subset and run flat search on 25 queries --
gt_indices = np.zeros((NQ_TOTAL, K), dtype=np.uint32)
gt_dists   = np.zeros((NQ_TOTAL, K), dtype=np.float32)

t_all = time.time()
for pair_id, (a, b, jc) in enumerate(pairs):
    t0 = time.time()
    idx_a = valid_idx_cache[a]
    idx_b = valid_idx_cache[b]
    # set intersection (idx arrays are sorted because np.where returns sorted)
    inter = np.intersect1d(idx_a, idx_b, assume_unique=True)
    n_a = len(inter)
    print(f"\n[pair {pair_id}] |A∩B|={n_a} (manifest says {jc})", flush=True)
    assert n_a == jc, f"intersection mismatch: got {n_a}, expected {jc}"

    # Materialize D_a = base[inter] as contiguous float32 (n_a × 128).
    # 12.8GB for the largest pair (1M rows), 0.5GB for small pairs.
    t1 = time.time()
    D_a = np.ascontiguousarray(base[inter], dtype=np.float32)
    print(f"  loaded D_a={D_a.shape}, dtype={D_a.dtype} ({time.time()-t1:.1f}s)", flush=True)

    # Build flat index + search 25 queries
    flat = faiss.IndexFlatL2(DIM)
    flat.add(D_a)
    q_lo = pair_id * n_per_pair
    q_hi = (pair_id+1) * n_per_pair
    qs = np.ascontiguousarray(queries[q_lo:q_hi], dtype=np.float32)
    D, I_local = flat.search(qs, K)
    I_global = inter[I_local].astype(np.uint32)
    gt_indices[q_lo:q_hi] = I_global
    gt_dists[q_lo:q_hi] = D.astype(np.float32)
    print(f"  searched {q_hi-q_lo} queries in {time.time()-t0:.1f}s; sample row[0][:5]={I_global[0,:5]}", flush=True)
    del D_a, flat

# -- Step 5: write GT files (main + .dist.bin sidecar) --
gt_path = SIFT_DIR / 'sift100m_2tag_conj_gt.bin'
gt_dist_path = SIFT_DIR / 'sift100m_2tag_conj_gt.dist.bin'

with open(gt_path, 'wb') as f:
    f.write(struct.pack('<II', NQ_TOTAL, K))
    gt_indices.tofile(f)
with open(gt_dist_path, 'wb') as f:
    f.write(struct.pack('<II', NQ_TOTAL, K))
    gt_dists.tofile(f)

print(f"\n[done] wrote {gt_path} ({gt_path.stat().st_size}B) + {gt_dist_path} ({gt_dist_path.stat().st_size}B) in {time.time()-t_all:.1f}s")
print(f"sample GT[0]={gt_indices[0]}, dists[0][:3]={gt_dists[0,:3]}")
print(f"sample GT[200] (inactive)={gt_indices[200]}, dists[200][:3]={gt_dists[200,:3]}")
