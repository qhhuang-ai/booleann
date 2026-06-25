#!/usr/bin/env python3
"""HAMCG SIFT1B: per-atom faiss HNSW (M=64 efc=100) on the full BIGANN.

Extends `hamcg_prototype_sift100m.py` to 10^9 scale. Base is read from the
uint8 raw file (120 GB) via memmap (kernel-evictable, hard-RSS budget below
251 GB host); per-atom D_a is materialized as float32 (the heaviest atom is
|D_a|=100M at sel=0.10 → 51 GB) and the HNSW (~100 GB at M=64) lives entirely
in RAM. No disk persistence of indices.

Same protocol as SIFT1M / 10M / 100M:
- 12 predicates: 4 per sel × 3 sels = (0.005, 0.05, 0.10)
- 200 queries
- K=10
- ef sweep [10, 20, 50, 100, 200, 400]
- threads=96 (build), single-threaded search timing

Estimated total wall: ~50-60 h (sel=0.10 atom build dominates at ~10 h/atom).
Estimated peak RSS: ~155 GB (sel=0.10 atom: 51 GB D_a + ~100 GB HNSW + 5 GB
faiss/python overhead).

Output: hamcg_sift1b_r100.parquet
"""
import gc
import numpy as np
import struct
import time
import os
from pathlib import Path
import faiss
import pandas as pd

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
SIFT1B_U8 = ROOT / 'data/raw/bigann/sift1b_base.u8raw'
SIFT_QUERY = ROOT / 'data/raw/sift/sift_query.fvecs'
OUT_PARQUET = ROOT / '03_experiment_bridge/results/raw/real_recall/hamcg_sift1b_r100.parquet'

N = 1_000_000_000
DIM = 128
NQ = 200
K = 10
SEED = 42
SELS = [0.005, 0.05, 0.10]
N_PRED_PER_SEL = 4
HNSW_M = 64
HNSW_EFC = 100
EF_LIST = [10, 20, 50, 100, 200, 400]
THREADS = 96

faiss.omp_set_num_threads(THREADS)

def read_fvecs(p, max_n=None):
    with open(p, 'rb') as f: data = f.read()
    d = struct.unpack('<i', data[:4])[0]
    n = len(data) // (4 + d*4)
    if max_n: n = min(n, max_n)
    return np.frombuffer(data, dtype=np.float32).reshape(-1, d+1)[:n, 1:].copy()

print(f"[load] SIFT1B base (uint8 memmap, 120 GB) + {NQ} queries...", flush=True)
t0 = time.time()
base_u8 = np.memmap(SIFT1B_U8, dtype=np.uint8, mode='r', shape=(N, DIM))
queries = read_fvecs(SIFT_QUERY)[:NQ]
print(f"  base {base_u8.shape} (uint8 memmap), queries {queries.shape}, load={time.time()-t0:.1f}s", flush=True)

rows = []
for s_target in SELS:
    n_classes = max(2, int(round(1.0 / s_target)))
    for pred_i in range(N_PRED_PER_SEL):
        rng = np.random.RandomState(SEED + int(s_target * 1000) + pred_i)
        labels = rng.randint(0, n_classes, size=N).astype(np.int32)
        target_class = rng.randint(0, n_classes)
        valid_idx = np.where(labels == target_class)[0]
        actual_s = len(valid_idx) / N
        n_a = len(valid_idx)
        print(f"\n[hot-atom] s={s_target:.3f} p{pred_i} target={target_class} |D_a|={n_a:,}", flush=True)

        # Materialize D_a as a contiguous float32 array (from uint8 memmap)
        t0 = time.time()
        # Read u8 then cast — done in one shot to avoid stride issues
        D_a_u8 = base_u8[valid_idx]
        D_a = D_a_u8.astype(np.float32, order='C', copy=True)
        del D_a_u8
        t_load = time.time() - t0
        print(f"  loaded D_a ({n_a*DIM*4/1e9:.2f} GB float32) in {t_load:.1f}s", flush=True)

        # Build per-atom HNSW
        t0 = time.time()
        idx_a = faiss.IndexHNSWFlat(DIM, HNSW_M)
        idx_a.hnsw.efConstruction = HNSW_EFC
        idx_a.add(D_a)
        t_build = time.time() - t0
        print(f"  built G_a in {t_build:.1f}s", flush=True)

        # GT: exact filtered top-K via faiss IndexFlat over D_a
        t0 = time.time()
        flat = faiss.IndexFlatL2(DIM)
        flat.add(D_a)
        _, gt_local = flat.search(queries, K)
        gt_ids = valid_idx[gt_local]
        t_gt = time.time() - t0
        print(f"  computed GT in {t_gt:.1f}s", flush=True)

        for ef in EF_LIST:
            idx_a.hnsw.efSearch = ef
            faiss.omp_set_num_threads(1)
            t1 = time.perf_counter()
            D, I = idx_a.search(queries, K)
            t2 = time.perf_counter()
            faiss.omp_set_num_threads(THREADS)
            total_ms = (t2 - t1) * 1000
            avg_us = total_ms / NQ * 1000
            pred_ids = valid_idx[I.clip(0, n_a-1)]
            recalls = []
            for q_i in range(NQ):
                gt_set = set(int(x) for x in gt_ids[q_i])
                pred_set = set(int(x) for x in pred_ids[q_i])
                recalls.append(len(gt_set & pred_set) / K)
            mean_recall = float(np.mean(recalls))
            rows.append({
                'method': 'HAMCG (hot-atom subgraph)',
                'predicate_type': 'equality',
                's_target': s_target,
                'actual_s': float(actual_s),
                'pred_i': pred_i,
                'ef': ef,
                'k': K,
                'recall_at_k': mean_recall,
                'avg_lat_us': avg_us,
                'build_time_s': t_build,
                'load_time_s': t_load,
                'gt_time_s': t_gt,
                'n_a': n_a,
                'hnsw_M': HNSW_M,
                'hnsw_efc': HNSW_EFC,
            })
            print(f"  ef={ef:4d}: recall={mean_recall:.4f} lat={avg_us:.1f}μs", flush=True)

        # Snapshot partial results after every atom — crash-resilient
        df_partial = pd.DataFrame(rows)
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df_partial.to_parquet(OUT_PARQUET, index=False)

        del D_a, idx_a, flat, gt_ids, labels, valid_idx
        gc.collect()

df = pd.DataFrame(rows)
df.to_parquet(OUT_PARQUET, index=False)
print(f"\n[done] {OUT_PARQUET}, {len(df)} rows", flush=True)

print("\n=== HAMCG SIFT1B equality summary ===", flush=True)
print(df.groupby(['s_target','ef']).agg(
    recall=('recall_at_k','mean'),
    lat_us=('avg_lat_us','mean'),
    build_s=('build_time_s','mean'),
).round(3), flush=True)
