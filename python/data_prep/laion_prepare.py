#!/usr/bin/env python3
"""laion_prepare.py — Stage LAION-1M data for BCI multi-dataset replication.

Inputs (under data/raw/laion1m/raw/):
  img_emb_0.npy   ~1.024 GB, (1000448, 512) float16 LAION CLIP image embeddings

Outputs (under data/raw/laion1m/):
  base.1M.f32bin                 (n=1000000, d=512, float32, with int32+int32 hdr)
  query.10K.f32bin               (n=10000,   d=512, float32, last 10K of the 1M)
  base.metadata.laion1m.spmat    (synthetic sparse-random base filters; matches YFCC10M format)
  query.metadata.laion1m.spmat   (synthetic 1-2 random labels per query)
  gt.10K.bin                     (filtered top-10 GT via brute-force; (10K, 10) int32 ids)
  gt.10K.dist.bin                (10K, 10) float32 distances

Synthetic filter design:
  * N_LABELS = 200 unique labels  (BCI cares about itemset structure, so 200 is enough)
  * avg labels per base row ≈ 5  (sparse-random, like YFCC10M's ~11 but smaller alphabet)
  * Per-label support ≈ 5 * 1M / 200 = 25K  → all labels exceed MIN_SUPPORT_BASE=500
  * Per-pair support ≈ 25K * 25K / 1M = 625  → many pairs exceed MIN_SUPPORT_BASE
  * Each query: 1 or 2 random labels (matching YFCC10M ~1.4 avg)
  * Seed deterministic for reproducibility.

NOTE: GT is computed by brute-force L2 over the labeled base subset, per-query.
With 10K queries and per-query candidate set ~25K-200K (depending on label freq),
total work is ~10K * 100K * 512 ≈ 5e11 flops — ~1h on 8 threads with vectorized numpy.
"""
import json, time, sys, os
import numpy as np
import os
from pathlib import Path

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / 'data/raw/laion1m/raw'
OUT = ROOT / 'data/raw/laion1m'

DIM = 512
N_TARGET = 1_000_000
N_QUERY = 10_000
N_LABELS = 200
AVG_LABELS_PER_ROW = 5
SEED = 42


def f32bin_write(path, x):
    """big-ann-benchmarks fbin format: int32 n, int32 d, then n*d float32."""
    n, d = x.shape
    assert x.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(np.array([n, d], dtype=np.int32).tobytes())
        x.tofile(f)


def spmat_write(path, indptr, indices, ncols):
    """YFCC10M spmat format: int64 nrow, int64 ncol, int64 nnz; int64 indptr[n+1]; int32 indices[nnz]."""
    nrows = len(indptr) - 1
    nnz = len(indices)
    assert indptr.dtype == np.int64
    assert indices.dtype == np.int32
    with open(path, 'wb') as f:
        f.write(np.array([nrows, ncols, nnz], dtype=np.int64).tobytes())
        indptr.tofile(f)
        indices.tofile(f)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # 1. Load 1 shard, slice to 1M+10K vectors total.
    shard0 = RAW / 'img_emb_0.npy'
    assert shard0.exists(), f"missing {shard0}"
    print(f"[load] {shard0} ({shard0.stat().st_size/1e9:.2f} GB)")
    arr = np.load(shard0, mmap_mode='r')  # float16
    print(f"  shape={arr.shape}, dtype={arr.dtype}")
    n_avail = arr.shape[0]
    assert n_avail >= N_TARGET, f"shard0 has only {n_avail} < {N_TARGET}"

    # Take first 1M for base; last 10K of those 1M serve as queries (per plan).
    # In other words: base = arr[:N_TARGET], queries = arr[N_TARGET-N_QUERY:N_TARGET].
    # GT is filtered top-10 of queries against the labeled (full) base.
    print(f"[convert fp16 → fp32] {N_TARGET} base vectors")
    base = np.asarray(arr[:N_TARGET]).astype(np.float32, copy=False)
    print(f"  base.shape={base.shape}, dtype={base.dtype}, "
          f"mean_norm={np.linalg.norm(base[:1000], axis=1).mean():.3f}")
    queries = base[N_TARGET - N_QUERY:N_TARGET].copy()
    print(f"  queries.shape={queries.shape} (last {N_QUERY} of base)")

    # 2. Save base + query f32bin
    base_path = OUT / 'base.1M.f32bin'
    f32bin_write(base_path, base)
    print(f"[write] {base_path} ({base_path.stat().st_size/1e9:.2f} GB)")
    q_path = OUT / 'query.10K.f32bin'
    f32bin_write(q_path, queries)
    print(f"[write] {q_path} ({q_path.stat().st_size/1e6:.2f} MB)")

    # 3. Synthesize sparse-random base filters
    print(f"[synthesize base filters] N_LABELS={N_LABELS}, avg/row={AVG_LABELS_PER_ROW}")
    # For each row, draw Poisson(AVG_LABELS_PER_ROW) labels uniformly from [0, N_LABELS)
    n_labels_per_row = rng.poisson(AVG_LABELS_PER_ROW, size=N_TARGET).clip(min=1, max=20)
    total_nnz = int(n_labels_per_row.sum())
    print(f"  total nnz = {total_nnz:,}, avg = {total_nnz/N_TARGET:.2f}")
    raw_indptr = np.zeros(N_TARGET + 1, dtype=np.int64)
    raw_indptr[1:] = np.cumsum(n_labels_per_row)
    flat = rng.integers(0, N_LABELS, size=total_nnz, dtype=np.int32)
    # dedup per row + sorted; collect into new compact arrays
    deduped_per_row = [None] * N_TARGET
    valid_per_row = np.zeros(N_TARGET, dtype=np.int64)
    for ri in range(N_TARGET):
        s, e = int(raw_indptr[ri]), int(raw_indptr[ri+1])
        if e > s:
            uniq = np.unique(flat[s:e])
            deduped_per_row[ri] = uniq
            valid_per_row[ri] = len(uniq)
        else:
            deduped_per_row[ri] = np.empty(0, dtype=np.int32)
    base_indptr = np.zeros(N_TARGET + 1, dtype=np.int64)
    base_indptr[1:] = np.cumsum(valid_per_row)
    total_nnz_final = int(base_indptr[-1])
    base_indices = np.zeros(total_nnz_final, dtype=np.int32)
    for ri in range(N_TARGET):
        s, e = int(base_indptr[ri]), int(base_indptr[ri+1])
        if e > s:
            base_indices[s:e] = deduped_per_row[ri]
    del deduped_per_row, flat, raw_indptr
    print(f"  after dedup: nnz = {len(base_indices):,}, avg = {len(base_indices)/N_TARGET:.2f}")

    # Sanity: label-frequency distribution
    label_freq = np.bincount(base_indices, minlength=N_LABELS)
    print(f"  per-label support: min={label_freq.min()}, "
          f"max={label_freq.max()}, mean={label_freq.mean():.0f}, "
          f">=500: {int((label_freq >= 500).sum())}, "
          f">=5000: {int((label_freq >= 5000).sum())}")
    sparsity = label_freq / N_TARGET
    print(f"  sparsity per label: min={sparsity.min():.4f}, max={sparsity.max():.4f}, "
          f"mean={sparsity.mean():.4f}")

    base_spmat_path = OUT / 'base.metadata.laion1m.spmat'
    spmat_write(base_spmat_path, base_indptr, base_indices, N_LABELS)
    print(f"[write] {base_spmat_path} ({base_spmat_path.stat().st_size/1e6:.2f} MB)")

    # 4. Synthesize query filters: 1 or 2 random labels per query (mimic YFCC10M ~1.4 avg)
    print(f"[synthesize query filters]")
    n_q_labels = rng.choice([1, 2], size=N_QUERY, p=[0.6, 0.4])
    total_q_nnz = int(n_q_labels.sum())
    q_indptr = np.zeros(N_QUERY + 1, dtype=np.int64)
    q_indptr[1:] = np.cumsum(n_q_labels)
    q_indices_list = []
    for qi in range(N_QUERY):
        nl = n_q_labels[qi]
        chosen = rng.choice(N_LABELS, size=nl, replace=False)
        q_indices_list.append(np.sort(chosen).astype(np.int32))
    q_indices = np.concatenate(q_indices_list)
    print(f"  query nnz total = {total_q_nnz}, avg = {total_q_nnz/N_QUERY:.2f}")
    q_spmat_path = OUT / 'query.metadata.laion1m.spmat'
    spmat_write(q_spmat_path, q_indptr, q_indices, N_LABELS)
    print(f"[write] {q_spmat_path} ({q_spmat_path.stat().st_size/1e3:.0f} KB)")

    # 5. Compute filtered ground truth: per query, filter base by query labels (AND-of-labels),
    #    then brute-force top-K L2.
    print(f"[GT] computing filtered top-10 brute-force GT over {N_QUERY} queries...")
    K = 10
    # Build label -> row list inverted index
    print("  building label inverted index...")
    row_id_per_nnz = np.repeat(np.arange(N_TARGET, dtype=np.int32),
                                np.diff(base_indptr).astype(np.int64))
    sort_idx = np.argsort(base_indices, kind='stable')
    sorted_labels = base_indices[sort_idx]
    sorted_rows = row_id_per_nnz[sort_idx]
    boundaries = np.concatenate([[0],
                                  np.where(np.diff(sorted_labels) != 0)[0] + 1,
                                  [len(sorted_labels)]])
    label_to_rs = {}
    for i in range(len(boundaries) - 1):
        l = int(sorted_labels[boundaries[i]])
        g = sorted_rows[boundaries[i]:boundaries[i+1]].astype(np.int64); g.sort()
        label_to_rs[l] = g

    gt_ids = np.zeros((N_QUERY, K), dtype=np.int32)
    gt_dists = np.zeros((N_QUERY, K), dtype=np.float32)
    t_gt = time.time()
    for qi in range(N_QUERY):
        qls = q_indices[q_indptr[qi]:q_indptr[qi+1]]
        # AND-of-labels: intersect candidate rows
        if len(qls) == 1:
            cand = label_to_rs[int(qls[0])]
        else:
            a = label_to_rs[int(qls[0])]
            for l in qls[1:]:
                a = a[np.isin(a, label_to_rs[int(l)], assume_unique=True)]
            cand = a
        if len(cand) == 0:
            gt_ids[qi] = -1
            gt_dists[qi] = np.inf
            continue
        # Skip self (query is base[N_TARGET-N_QUERY + qi])
        self_id = (N_TARGET - N_QUERY) + qi
        cand = cand[cand != self_id]
        if len(cand) == 0:
            gt_ids[qi] = -1; gt_dists[qi] = np.inf; continue
        q_vec = queries[qi:qi+1]
        diffs = base[cand] - q_vec
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        if len(cand) <= K:
            top_idx = np.argsort(d2)
            gt_ids[qi, :len(cand)] = cand[top_idx]
            gt_ids[qi, len(cand):] = -1
            gt_dists[qi, :len(cand)] = d2[top_idx]
            gt_dists[qi, len(cand):] = np.inf
        else:
            top_idx = np.argpartition(d2, K)[:K]
            order = np.argsort(d2[top_idx])
            top_idx = top_idx[order]
            gt_ids[qi] = cand[top_idx]
            gt_dists[qi] = d2[top_idx]
        if (qi+1) % 1000 == 0:
            print(f"    {qi+1:,}/{N_QUERY:,} elapsed={time.time()-t_gt:.0f}s "
                  f"avg_cand={len(cand):,}")
    print(f"  GT done in {time.time()-t_gt:.0f}s")

    gt_path = OUT / 'gt.10K.bin'
    gt_dist_path = OUT / 'gt.10K.dist.bin'
    with open(gt_path, 'wb') as f:
        f.write(np.array([N_QUERY, K], dtype=np.int32).tobytes())
        gt_ids.tofile(f)
    with open(gt_dist_path, 'wb') as f:
        f.write(np.array([N_QUERY, K], dtype=np.int32).tobytes())
        gt_dists.tofile(f)
    print(f"[write] {gt_path} ({gt_path.stat().st_size/1e6:.2f} MB)")
    print(f"[write] {gt_dist_path} ({gt_dist_path.stat().st_size/1e6:.2f} MB)")

    summary = {
        'n_base': N_TARGET, 'n_query': N_QUERY, 'dim': DIM,
        'n_labels': N_LABELS, 'avg_labels_per_base_row': float(len(base_indices)/N_TARGET),
        'avg_labels_per_query': float(total_q_nnz/N_QUERY),
        'min_per_label_support': int(label_freq.min()),
        'max_per_label_support': int(label_freq.max()),
        'mean_per_label_support': float(label_freq.mean()),
        'labels_above_500': int((label_freq >= 500).sum()),
        'mean_sparsity': float(sparsity.mean()),
        'seed': SEED,
        'total_time_s': time.time() - t0,
    }
    (OUT / 'prepare_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\n[done] total time {time.time()-t0:.0f}s")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
