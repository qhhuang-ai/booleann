#!/usr/bin/env python3
"""Generate ground truth for matched-qids benchmark of Attack 1 v6.

Used by:
- bci_hamcg_yfcc10m_r115v6.py (BCI bench)
- run_sieve_yfcc10m.py (SIEVE rerun)

Both will use the SAME qids slice [60K, 70K) (sequential, 10K queries) for
fair recall@10 head-to-head. GT computed via base HAMCG conjunction (rare-label-
first inverted-list intersection + exact L2 scan over intersection).

Output: data/raw/yfcc100m/yfcc-10M-gt-60-70-AND.bin in SIEVE driver's expected
format: int32 nrow + int32 k + (int32 ids)[nrow*k] + (float32 distances)[nrow*k]
"""
import numpy as np
import time
import os
from pathlib import Path

ROOT = Path(os.environ.get('BOOLEANN_ROOT', Path(__file__).resolve().parents[2]))
BASE_VECTORS = ROOT / 'data/raw/yfcc100m/base.10M.u8bin'
QUERY_VECTORS = ROOT / 'data/raw/yfcc100m/query.public.100K.u8bin'
BASE_SPMAT = ROOT / 'data/raw/yfcc100m/base.metadata.10M.spmat'
QUERY_SPMAT = ROOT / 'data/raw/yfcc100m/query.metadata.public.100K.spmat'
OUT_GT = ROOT / 'data/raw/yfcc100m/yfcc-10M-gt-60-70-AND.bin'
OUT_NPZ = ROOT / '03_experiment_bridge/results/raw/real_recall/bci_yfcc10m/v6_gt_60_70_cache.npz'

NB = 10_000_000
DIM = 192
QUERY_START = 60_000
QUERY_COUNT = 10_000
K = 10
K_STORE = 10  # Store top-K (we want top-10 GT only for matching SIEVE driver)


def read_u8bin(path, n=None):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(8), dtype=np.uint32)
        total_n, d = int(hdr[0]), int(hdr[1])
        if n is None or n > total_n: n = total_n
        return np.frombuffer(f.read(n * d), dtype=np.uint8).reshape(n, d), d


def read_spmat(path):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(24), dtype=np.int64)
        nrows = int(hdr[0]); ncols = int(hdr[1]); nnz = int(hdr[2])
        indptr = np.frombuffer(f.read((nrows+1)*8), dtype=np.int64).copy()
        indices = np.frombuffer(f.read(nnz*4), dtype=np.int32).copy()
    return nrows, ncols, nnz, indptr, indices


def main():
    t_total = time.time()
    print(f"[load]", flush=True)
    base_u8, _ = read_u8bin(BASE_VECTORS, n=NB)
    queries_u8, _ = read_u8bin(QUERY_VECTORS, n=100_000)
    base_f32 = base_u8.astype(np.float32)
    queries_f32 = queries_u8.astype(np.float32)
    nrows, _, _, base_indptr, base_indices = read_spmat(BASE_SPMAT)
    qn, _, _, q_indptr, q_indices = read_spmat(QUERY_SPMAT)
    print(f"  base {base_f32.shape}, queries {queries_f32.shape}, in {time.time()-t_total:.0f}s")

    # label_to_rs (sorted)
    t_inv = time.time()
    row_id_per_nnz = np.repeat(np.arange(nrows, dtype=np.int32), np.diff(base_indptr).astype(np.int64))
    sort_idx = np.argsort(base_indices, kind='stable')
    sorted_labels = base_indices[sort_idx]
    sorted_row_ids = row_id_per_nnz[sort_idx]
    boundaries = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
    label_to_rs = {}
    for i in range(len(boundaries) - 1):
        l = int(sorted_labels[boundaries[i]])
        g = sorted_row_ids[boundaries[i]:boundaries[i+1]].astype(np.int64); g.sort()
        label_to_rs[l] = g
    print(f"  label_to_rs in {time.time()-t_inv:.0f}s")

    # GT compute for sequential [60K, 70K)
    print(f"\n[GT] computing for {QUERY_COUNT:,} queries [{QUERY_START}, {QUERY_START+QUERY_COUNT})")
    t_gt = time.time()
    gt_ids = np.full((QUERY_COUNT, K_STORE), -1, dtype=np.int32)
    gt_dists = np.full((QUERY_COUNT, K_STORE), np.inf, dtype=np.float32)
    lats = np.zeros(QUERY_COUNT, dtype=np.float64)
    for ti in range(QUERY_COUNT):
        qi = QUERY_START + ti
        q_labels = list(q_indices[q_indptr[qi]:q_indptr[qi+1]])
        t0 = time.perf_counter()
        if len(q_labels) == 0:
            lats[ti] = (time.perf_counter()-t0)*1e6; continue
        if len(q_labels) == 1:
            x_phi = label_to_rs[int(q_labels[0])]
        else:
            # rare-label first intersection
            ll = sorted([int(l) for l in q_labels], key=lambda x: len(label_to_rs[x]))
            x_phi = label_to_rs[ll[0]]
            for l in ll[1:]:
                x_phi = np.intersect1d(x_phi, label_to_rs[l], assume_unique=False)
                if len(x_phi) == 0: break
        if len(x_phi) == 0:
            lats[ti] = (time.perf_counter()-t0)*1e6; continue
        # exact L2 top-K
        diffs = base_f32[x_phi] - queries_f32[qi]
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        if len(x_phi) <= K_STORE:
            order = np.argsort(d2)
            top = x_phi[order]
            top_d = d2[order]
            gt_ids[ti, :len(top)] = top.astype(np.int32)
            gt_dists[ti, :len(top)] = top_d.astype(np.float32)
        else:
            top_idx = np.argpartition(d2, K_STORE)[:K_STORE]
            order = np.argsort(d2[top_idx])
            top = x_phi[top_idx[order]]
            top_d = d2[top_idx[order]]
            gt_ids[ti] = top.astype(np.int32)
            gt_dists[ti] = top_d.astype(np.float32)
        lats[ti] = (time.perf_counter()-t0)*1e6
        if ti % 500 == 0 and ti > 0:
            rate = ti / (time.time() - t_gt)
            print(f"  ti={ti:,}, rate={rate:.1f} q/s, mean lat last 500 = {np.mean(lats[max(0,ti-500):ti]):.0f}us", flush=True)
    print(f"\n[done] GT for {QUERY_COUNT:,} queries in {time.time()-t_gt:.0f}s")
    print(f"  base HAMCG conj: mean lat = {lats[lats>0].mean():.0f}us, median = {np.median(lats[lats>0]):.0f}us")
    print(f"  QPS single-thread (this script, not optimized): {1e6/lats[lats>0].mean():.1f}")

    # Save as ibin (id-only or id+dist, SIEVE expects id+dist format with int32 + float32)
    print(f"\n[save] {OUT_GT}")
    OUT_GT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GT, 'wb') as f:
        hdr = np.array([QUERY_COUNT, K_STORE], dtype=np.uint32)
        f.write(hdr.tobytes())
        f.write(gt_ids.tobytes())
        f.write(gt_dists.tobytes())
    print(f"  wrote {OUT_GT.stat().st_size / 2**20:.2f} MB")

    # Also save npz for BCI bench
    print(f"\n[save npz] {OUT_NPZ}")
    np.savez(OUT_NPZ, gt_ids=gt_ids.astype(np.int64), gt_dists=gt_dists, lat=lats,
             qi=np.arange(QUERY_START, QUERY_START+QUERY_COUNT, dtype=np.int64))
    print(f"  wrote {OUT_NPZ.stat().st_size / 2**20:.2f} MB")
    print(f"\n[total elapsed] {time.time()-t_total:.0f}s")


if __name__ == '__main__':
    main()
