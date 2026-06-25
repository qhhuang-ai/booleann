#!/usr/bin/env python3
"""ARWGI on SIFT100M: structural-gap closure deployment.

Goal
----
Deploy ARWGI as a *real query-time system* on SIFT100M (100M base, dim=128) so
that Theorem T-1A applies to a deployed-scale system (not just a YFCC10M audit).

Reviewers' critique closed by this experiment: "ARWGI not deployed at scale;
T-1A doesn't certify any winning deployed system."

What this script does
---------------------
  1. Build (or reuse) a full-base SIFT100M HNSW (M=32, efc=200). This is the
     HNSW that T-1A's "spine" assumption is about.
  2. Discover the union of nodes visited by 200 queries at ef=128, top-512
     (visited_set ≈ a few hundred K nodes).
  3. For each visited node v, compute U_v = closed 2-hop layer-0 neighbourhood,
     sample r witnesses W_v ⊂ U_v \ {v} uniformly without replacement (paper-
     faithful T-1A construction). Persist (Uv_indptr/indices, Wv, r_eff).
  4. Generate the SAME 12 predicates as the HAMCG SIFT100M r097 run
     (3 selectivities {0.005, 0.05, 0.10} × 4 atoms; equality on a synthetic
     integer Bernoulli label; seed= SEED + int(s*1000) + pred_i).
  5. For each predicate:
       - Compute exact filtered top-K=10 GT via faiss IndexFlat over D_a
         (same protocol as HAMCG r097).
       - Run two filtered systems on the FULL HNSW:
            (a) Unmodified HNSW + post-filter  (top-(over*k), keep phi-passers)
            (b) ARWGI:    same descent, then expand each visited candidate's
                          W_v witnesses, keep phi-passers, re-rank by L2.
       - Report recall@10 and avg latency per query for ef in {50, 100, 200}.
  6. Save parquet alongside hamcg_sift100m_r097.parquet for direct comparison.

Outputs
-------
  results/raw/arwgi_sift100m/
    arwgi_sift100m_hnsw_M32_efc200.faiss        (full HNSW; ~60 GB)
    arwgi_sift100m_wv_r24.npz                    (U_v + W_v for visited nodes)
    arwgi_sift100m_bench.parquet                 (recall + lat)
    arwgi_sift100m_build_meta.json               (timings, sizes, env)

Run
---
  proxy + nohup + run_in_background recommended. Each stage idempotent.
  python arwgi_sift100m_deploy.py --stage build_hnsw
  python arwgi_sift100m_deploy.py --stage build_wv
  python arwgi_sift100m_deploy.py --stage bench
  python arwgi_sift100m_deploy.py --stage all
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
import time
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
SIFT_F32_RAW = ROOT / 'data/raw/sift100m/sift100m_base_f32.raw'   # bare f32 (no per-row header)
SIFT_QUERY = ROOT / 'data/raw/sift/sift_query.fvecs'
HAMCG_REF_PARQUET = ROOT / '03_experiment_bridge/results/raw/real_recall/hamcg_sift100m_r097.parquet'
OUT_DIR = ROOT / '03_experiment_bridge/results/raw/arwgi_sift100m'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Index path — kept in OUT_DIR so artifact is grouped with the rest
HNSW_PATH = OUT_DIR / 'arwgi_sift100m_hnsw_M32_efc200.faiss'
WV_PATH_TMPL = OUT_DIR / 'arwgi_sift100m_wv_r{r}.npz'
BENCH_PARQUET = OUT_DIR / 'arwgi_sift100m_bench.parquet'
META_JSON = OUT_DIR / 'arwgi_sift100m_build_meta.json'

N = 100_000_000
DIM = 128
NQ = 200
K = 10
SEED = 42
SELS = [0.005, 0.05, 0.10]
N_PRED_PER_SEL = 4
HNSW_M = 32
HNSW_EFC = 200
EF_BENCH = [50, 100, 200]
OVER_BENCH = [16]    # candidate pool top-(over*K) before filter; matches arwgi_proper standard
DEFAULT_R = 24       # witnesses per node (matches arwgi_proper YFCC10M r=24 baseline)
# W_v node-set discovery must cover every node the bench's HNSW search could return
# as a candidate. Bench worst case = ef * over_k = 200 * 16 = 3200, so set
# VISIT_TOPK >= that.  EF_VISIT >= max bench ef so HNSW reaches the same nodes.
VISIT_TOPK = 3200
EF_VISIT = 200
THREADS = 96


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------

def read_fvecs(p, max_n=None):
    with open(p, 'rb') as f:
        data = f.read()
    d = struct.unpack('<i', data[:4])[0]
    n = len(data) // (4 + d * 4)
    if max_n:
        n = min(n, max_n)
    return np.frombuffer(data, dtype=np.float32).reshape(-1, d + 1)[:n, 1:].copy()


def open_base_memmap():
    return np.memmap(SIFT_F32_RAW, dtype=np.float32, mode='r', shape=(N, DIM))


# --------------------------------------------------------------------------------------
# HNSW view (re-used from arwgi_proper_deploy.py with minor cleanup)
# --------------------------------------------------------------------------------------

class HNSWView:
    def __init__(self, idx: faiss.IndexHNSWFlat):
        self.idx = idx
        h = idx.hnsw
        self.entry_point = int(h.entry_point)
        self.neighbors = faiss.vector_to_array(h.neighbors).astype(np.int32, copy=False)
        self.offsets = faiss.vector_to_array(h.offsets).astype(np.int64, copy=False)
        self.cnn = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64, copy=False)
        self.layer0_width = int(self.cnn[1])
        self.N = idx.ntotal
        self.d = idx.d

    def layer0_neighbors(self, v: int) -> np.ndarray:
        base = int(self.offsets[v])
        sl = self.neighbors[base: base + self.layer0_width]
        return sl[sl >= 0].astype(np.int64)


# --------------------------------------------------------------------------------------
# Stage 1: build full-base HNSW
# --------------------------------------------------------------------------------------

def stage_build_hnsw(args):
    """Build a single full-base SIFT100M HNSW (M=32, efc=200).

    This is the index T-1A's theorem is about. We materialise the base in
    contiguous float32 in RAM (49 GB) and call faiss.IndexHNSWFlat.add. Build
    is expected to take 6-18h depending on threads and memory bandwidth.
    """
    if HNSW_PATH.exists() and not args.force:
        sz_gb = HNSW_PATH.stat().st_size / 2**30
        print(f"[skip] HNSW exists at {HNSW_PATH} ({sz_gb:.1f} GB)", flush=True)
        return
    print(f"=== STAGE: build SIFT100M HNSW (M={HNSW_M}, efc={HNSW_EFC}) ===", flush=True)
    print(f"  threads = {THREADS}", flush=True)
    faiss.omp_set_num_threads(THREADS)

    t0 = time.time()
    print("  open base mmap (no in-RAM copy, faiss reads on-demand)", flush=True)
    base_mm = open_base_memmap()
    t_load = time.time() - t0
    print(f"  base mmap {base_mm.shape} in {t_load:.1f}s", flush=True)

    t1 = time.time()
    idx = faiss.IndexHNSWFlat(DIM, HNSW_M)
    idx.hnsw.efConstruction = HNSW_EFC
    BATCH = 2_000_000   # 2M rows per add, ~1 GB per batch
    print(f"  start add in batches of {BATCH:,}", flush=True)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        chunk = np.ascontiguousarray(base_mm[start:end])
        idx.add(chunk)
        del chunk
        if start // BATCH % 5 == 0:
            print(f"    added {end:,}/{N:,}, elapsed {time.time()-t1:.0f}s", flush=True)
    t_build = time.time() - t1
    print(f"  HNSW built in {t_build:.1f}s "
          f"({t_build/3600:.2f} h)", flush=True)

    t2 = time.time()
    faiss.write_index(idx, str(HNSW_PATH))
    t_write = time.time() - t2
    sz_gb = HNSW_PATH.stat().st_size / 2**30
    print(f"  wrote {HNSW_PATH} ({sz_gb:.1f} GB) in {t_write:.1f}s", flush=True)

    meta = {
        'stage': 'build_hnsw',
        'N': N, 'dim': DIM, 'M': HNSW_M, 'efc': HNSW_EFC,
        'threads': THREADS,
        'load_s': t_load, 'build_s': t_build, 'write_s': t_write,
        'index_size_gb': sz_gb,
        'index_path': str(HNSW_PATH),
    }
    META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"  meta -> {META_JSON}", flush=True)


# --------------------------------------------------------------------------------------
# Stage 2: U_v + W_v build for visited nodes
# --------------------------------------------------------------------------------------

def build_uv_wv(view: HNSWView, node_ids: np.ndarray, r: int, seed: int = 0xA17,
                log_every: int = 10_000, uv_cap: int = 4096):
    """For each v in node_ids: U_v = closed 2-hop layer-0 nbhd; W_v = r uniform
    samples from U_v \\ {v} without replacement. Memory-bounded by uv_cap.
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(node_ids)
    Wv = np.full((n_nodes, r), -1, dtype=np.int32)
    r_eff = np.zeros(n_nodes, dtype=np.int32)
    Uv_sizes = np.zeros(n_nodes, dtype=np.int32)
    Uv_parts = []
    Uv_indptr = np.zeros(n_nodes + 1, dtype=np.int64)

    t0 = time.time()
    for i, v in enumerate(node_ids):
        v = int(v)
        n1 = view.layer0_neighbors(v)
        if len(n1) == 0:
            uv = np.array([v], dtype=np.int64)
        else:
            parts = [np.array([v], dtype=np.int64), n1]
            for u in n1.tolist():
                n2 = view.layer0_neighbors(int(u))
                if len(n2):
                    parts.append(n2)
            uv = np.unique(np.concatenate(parts))
        uv_minus_v = uv[uv != v]
        if len(uv_minus_v) > uv_cap:
            sub_rng = np.random.default_rng((seed * 0x9E3779B97F4A7C15 + v) & ((1 << 64) - 1))
            sel = sub_rng.choice(len(uv_minus_v), size=uv_cap, replace=False)
            uv_minus_v = uv_minus_v[sel]
        Uv_sizes[i] = len(uv_minus_v)
        Uv_parts.append(uv_minus_v.astype(np.int32, copy=False))
        Uv_indptr[i + 1] = Uv_indptr[i] + len(uv_minus_v)

        if len(uv_minus_v) >= r:
            sel = rng.choice(len(uv_minus_v), size=r, replace=False)
            Wv[i] = uv_minus_v[sel]
            r_eff[i] = r
        elif len(uv_minus_v) > 0:
            Wv[i, :len(uv_minus_v)] = uv_minus_v
            r_eff[i] = len(uv_minus_v)
        else:
            r_eff[i] = 0

        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n_nodes - i - 1) / max(rate, 1e-6)
            print(f"    [W_v] {i+1:,}/{n_nodes:,} rate={rate:.0f} nodes/s "
                  f"ETA={eta/60:.1f} min", flush=True)

    Uv_indices = np.concatenate(Uv_parts).astype(np.int32, copy=False)
    return Uv_indptr, Uv_indices, Uv_sizes, Wv, r_eff


def stage_build_wv(args):
    """Discover nodes visited by 200 queries (top-VISIT_TOPK at ef=EF_VISIT),
    then build U_v + W_v on the union. Persist to NPZ."""
    out_npz = WV_PATH_TMPL.with_name(WV_PATH_TMPL.name.format(r=args.r))
    if out_npz.exists() and not args.force:
        sz_mb = out_npz.stat().st_size / 2**20
        print(f"[skip] W_v exists at {out_npz} ({sz_mb:.1f} MB)", flush=True)
        return out_npz
    print(f"=== STAGE: build W_v (r={args.r}) on SIFT100M visited subset ===", flush=True)
    faiss.omp_set_num_threads(THREADS)

    print(f"  load HNSW from {HNSW_PATH}", flush=True)
    t0 = time.time()
    idx = faiss.read_index(str(HNSW_PATH))
    view = HNSWView(idx)
    print(f"    loaded in {time.time()-t0:.1f}s; N={view.N:,} "
          f"layer0_width={view.layer0_width} ep={view.entry_point}", flush=True)

    print(f"  load {NQ} queries", flush=True)
    queries = read_fvecs(SIFT_QUERY)[:NQ]
    print(f"    queries {queries.shape}", flush=True)

    print(f"  discover visited nodes (ef={EF_VISIT}, top-{VISIT_TOPK})", flush=True)
    idx.hnsw.efSearch = EF_VISIT
    t1 = time.time()
    _, I = idx.search(queries, VISIT_TOPK)
    print(f"    search done in {time.time()-t1:.1f}s", flush=True)
    visited_set = set(int(x) for x in I.flatten() if x >= 0)
    visited_set.add(view.entry_point)
    visited_arr = np.array(sorted(visited_set), dtype=np.int64)
    print(f"    visited node set: {len(visited_arr):,}", flush=True)

    print(f"  build U_v (2-hop) + W_v (r={args.r}) for {len(visited_arr):,} nodes", flush=True)
    t2 = time.time()
    Uv_indptr, Uv_indices, Uv_sizes, Wv, r_eff = build_uv_wv(
        view, visited_arr, r=args.r, seed=args.seed, log_every=20_000)
    t_wv = time.time() - t2
    print(f"    built in {t_wv:.1f}s; mean |U_v|={Uv_sizes.mean():.1f} "
          f"mean r_eff={r_eff.mean():.1f}", flush=True)

    np.savez(out_npz,
             node_ids=visited_arr.astype(np.int64),
             Uv_indptr=Uv_indptr,
             Uv_indices=Uv_indices,
             Uv_sizes=Uv_sizes,
             Wv=Wv,
             r_eff=r_eff,
             r_requested=np.int32(args.r),
             seed=np.int64(args.seed))
    sz_mb = out_npz.stat().st_size / 2**20
    print(f"  wrote {out_npz} ({sz_mb:.1f} MB)", flush=True)

    # update meta
    meta = {}
    if META_JSON.exists():
        meta = json.loads(META_JSON.read_text())
    meta.update({
        'wv_path': str(out_npz),
        'wv_size_mb': sz_mb,
        'visited_nodes': int(len(visited_arr)),
        'wv_build_s': t_wv,
        'mean_Uv_size': float(Uv_sizes.mean()),
        'mean_r_eff': float(r_eff.mean()),
        'r_requested': args.r,
        'visit_topk': VISIT_TOPK,
        'ef_visit': EF_VISIT,
    })
    META_JSON.write_text(json.dumps(meta, indent=2))
    return out_npz


# --------------------------------------------------------------------------------------
# Stage 3: bench (unmod HNSW + post-filter) vs ARWGI on SIFT100M
# --------------------------------------------------------------------------------------

def gen_predicate_labels(s_target: float, pred_i: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Reproduce HAMCG r097 predicate generation EXACTLY.

    Returns (labels, valid_idx, target_class, actual_s).
    labels: int32 [N], values in [0, n_classes).
    valid_idx: int64 indices of records with labels==target_class.
    """
    n_classes = max(2, int(round(1.0 / s_target)))
    rng = np.random.RandomState(SEED + int(s_target * 1000) + pred_i)
    labels = rng.randint(0, n_classes, size=N).astype(np.int32)
    target_class = rng.randint(0, n_classes)
    valid_idx = np.where(labels == target_class)[0].astype(np.int64)
    actual_s = len(valid_idx) / N
    return labels, valid_idx, int(target_class), float(actual_s)


def compute_filtered_gt(queries: np.ndarray, valid_idx: np.ndarray,
                        base_mm: np.memmap) -> np.ndarray:
    """Exact filtered top-K GT: faiss IndexFlatL2 over D_a (the predicate subset).
    Returns gt_ids (NQ, K) in GLOBAL space.

    Uses memmap fancy-indexing once to materialise D_a (predicate subset only)
    in contiguous f32 RAM. For s=0.10 this is 10M*128*4 = 5.1 GB.
    """
    n_a = len(valid_idx)
    t0 = time.time()
    # fancy-index from memmap into contiguous RAM (one big read)
    D_a = np.array(base_mm[valid_idx], dtype=np.float32, copy=True)
    t_load_da = time.time() - t0
    t1 = time.time()
    flat = faiss.IndexFlatL2(DIM)
    flat.add(D_a)
    _, gt_local = flat.search(queries, K)
    t_gt = time.time() - t1
    gt_ids_global = valid_idx[gt_local]
    print(f"    GT: n_a={n_a:,} D_a-load {t_load_da:.1f}s search {t_gt:.1f}s", flush=True)
    del D_a, flat
    gc.collect()
    return gt_ids_global


def faiss_search_unmodified_lat(idx, queries, ef, over, k, labels, target_class):
    """Unmodified HNSW + post-filter, single-threaded for latency."""
    idx.hnsw.efSearch = ef
    nq = queries.shape[0]
    faiss.omp_set_num_threads(1)
    t0 = time.perf_counter()
    D, I = idx.search(queries, over * k)
    t1 = time.perf_counter()
    faiss.omp_set_num_threads(THREADS)
    avg_us = (t1 - t0) / nq * 1e6

    filtered_ids = np.full((nq, k), -1, dtype=np.int64)
    for i in range(nq):
        kept = []
        for x in I[i]:
            if x >= 0 and labels[x] == target_class:
                kept.append(int(x))
                if len(kept) >= k:
                    break
        if kept:
            filtered_ids[i, :len(kept)] = kept[:k]
    return filtered_ids, avg_us


def arwgi_search_lat(idx, view, queries, Wv_map, ef, over, k, labels, target_class,
                    base_mm):
    """ARWGI: HNSW top-(over*k) candidates + W_v expansion at each candidate,
    filter by phi, re-rank by L2.

    Latency measured single-threaded over the WHOLE pipeline (search + expand +
    re-rank). Re-rank reads from base_mm — this is what a real deployment would
    do (vectors loaded on demand; we don't keep full 49 GB in RAM during bench).
    """
    idx.hnsw.efSearch = ef
    nq = queries.shape[0]
    filtered_ids = np.full((nq, k), -1, dtype=np.int64)
    extra_used = np.zeros(nq, dtype=np.int32)

    faiss.omp_set_num_threads(1)
    t0 = time.perf_counter()
    D, I = idx.search(queries, over * k)

    for i in range(nq):
        seed = set()
        # 1) HNSW candidates passing phi
        for x in I[i]:
            if x >= 0 and labels[x] == target_class:
                seed.add(int(x))
        n_hnsw_phi = len(seed)
        # 2) W_v expansion from every HNSW candidate that has a W_v entry
        for x in I[i]:
            if x < 0:
                continue
            x_i = int(x)
            if x_i not in Wv_map:
                continue
            r_eff_v, Wrow = Wv_map[x_i]
            for w in Wrow[:r_eff_v]:
                w_i = int(w)
                if labels[w_i] == target_class:
                    seed.add(w_i)
        if not seed:
            continue
        ids_arr = np.array(sorted(seed), dtype=np.int64)
        # Fetch vectors from memmap (deployment-faithful) and rerank
        vecs = np.array(base_mm[ids_arr], dtype=np.float32, copy=True)
        q = queries[i]
        diffs = vecs - q
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        order = np.argsort(d2)[:k]
        top = ids_arr[order]
        filtered_ids[i, :len(top)] = top
        extra_used[i] = len(seed) - n_hnsw_phi

    t1 = time.perf_counter()
    faiss.omp_set_num_threads(THREADS)
    avg_us = (t1 - t0) / nq * 1e6
    return filtered_ids, avg_us, extra_used


def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    nq = pred_ids.shape[0]
    rec = []
    for i in range(nq):
        gt_set = set(int(x) for x in gt_ids[i, :k] if x >= 0)
        pr_set = set(int(x) for x in pred_ids[i, :k] if x >= 0)
        if gt_set:
            rec.append(len(gt_set & pr_set) / k)
    return float(np.mean(rec)) if rec else 0.0


def stage_bench(args):
    """Run filtered recall@10 bench on SIFT100M for 12 predicates × ef list."""
    wv_path = WV_PATH_TMPL.with_name(WV_PATH_TMPL.name.format(r=args.r))
    assert HNSW_PATH.exists(), f"missing HNSW {HNSW_PATH}; run --stage build_hnsw first"
    assert wv_path.exists(),   f"missing W_v {wv_path}; run --stage build_wv first"

    print(f"=== STAGE: bench ARWGI vs unmod HNSW on SIFT100M ===", flush=True)
    faiss.omp_set_num_threads(THREADS)

    print(f"  load HNSW {HNSW_PATH}", flush=True)
    t0 = time.time()
    idx = faiss.read_index(str(HNSW_PATH))
    view = HNSWView(idx)
    print(f"    loaded in {time.time()-t0:.1f}s; N={view.N:,}", flush=True)

    print(f"  load W_v from {wv_path}", flush=True)
    t1 = time.time()
    d = np.load(wv_path)
    Wv_node_ids = d['node_ids']
    Wv = d['Wv']; r_eff = d['r_eff']; r_req = int(d['r_requested'])
    Wv_map = {int(n): (int(r_eff[i]), Wv[i]) for i, n in enumerate(Wv_node_ids.tolist())}
    print(f"    W_v loaded in {time.time()-t1:.1f}s; "
          f"{len(Wv_node_ids):,} nodes, r={r_req}, mean r_eff={r_eff.mean():.1f}", flush=True)

    print(f"  load {NQ} queries", flush=True)
    queries = read_fvecs(SIFT_QUERY)[:NQ]
    print(f"    queries {queries.shape}", flush=True)

    print(f"  open base memmap (no full RAM)", flush=True)
    base_mm = open_base_memmap()

    # Build labels storage: 12 predicates use the SAME synthetic family but
    # different (s, pred_i) seeds. We hold the labels for ONE predicate at a
    # time to keep memory bounded (each labels is 400 MB int32).
    rows = []
    for s in SELS:
        for pi in range(N_PRED_PER_SEL):
            tag = f"s={s:.3f} p{pi}"
            print(f"\n  --- predicate {tag} ---", flush=True)
            labels, valid_idx, target_class, actual_s = gen_predicate_labels(s, pi)
            n_a = len(valid_idx)
            print(f"    target_class={target_class} n_a={n_a:,} actual_s={actual_s:.6f}", flush=True)

            # GT
            gt_ids_global = compute_filtered_gt(queries, valid_idx, base_mm)

            for ef in EF_BENCH:
                for over in OVER_BENCH:
                    # (a) Unmodified HNSW + post-filter
                    ids_u, lat_u = faiss_search_unmodified_lat(
                        idx, queries, ef, over, K, labels, target_class)
                    rec_u = recall_at_k(ids_u, gt_ids_global, K)

                    # (b) ARWGI
                    ids_a, lat_a, extra = arwgi_search_lat(
                        idx, view, queries, Wv_map, ef, over, K,
                        labels, target_class, base_mm)
                    rec_a = recall_at_k(ids_a, gt_ids_global, K)

                    row = {
                        'method_a': 'HNSW unmod + post-filter',
                        'method_b': 'ARWGI (W_v expansion)',
                        's_target': s, 'actual_s': actual_s,
                        'pred_i': pi, 'target_class': target_class,
                        'n_a': n_a,
                        'ef': ef, 'over_k': over, 'k': K,
                        'recall_unmod': rec_u,
                        'recall_arwgi': rec_a,
                        'lat_unmod_us': lat_u,
                        'lat_arwgi_us': lat_a,
                        'mean_arwgi_extras': float(extra.mean()),
                        'hnsw_M': HNSW_M, 'hnsw_efc': HNSW_EFC,
                        'r_witnesses': r_req,
                    }
                    rows.append(row)
                    print(f"    ef={ef:>3} over={over}  unmod rec={rec_u:.4f} lat={lat_u:.0f}us "
                          f"|  arwgi rec={rec_a:.4f} lat={lat_a:.0f}us  extras={extra.mean():.1f}",
                          flush=True)
            # Free
            del labels, valid_idx, gt_ids_global
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_parquet(BENCH_PARQUET, index=False)
    print(f"\n  wrote {BENCH_PARQUET} ({len(df)} rows)", flush=True)

    # Summary: agg by s + ef
    print(f"\n=== SUMMARY (mean over 4 predicates per s) ===", flush=True)
    agg = df.groupby(['s_target', 'ef']).agg(
        recall_unmod=('recall_unmod', 'mean'),
        recall_arwgi=('recall_arwgi', 'mean'),
        lat_unmod_us=('lat_unmod_us', 'mean'),
        lat_arwgi_us=('lat_arwgi_us', 'mean'),
    ).round(4)
    print(agg.to_string(), flush=True)

    # Merge in HAMCG r097 reference numbers (recall at same ef, same s)
    if HAMCG_REF_PARQUET.exists():
        href = pd.read_parquet(HAMCG_REF_PARQUET)
        # HAMCG used ef in [10,20,50,100,200,400]; pick ef in EF_BENCH
        href = href[href['ef'].isin(EF_BENCH)].groupby(['s_target', 'ef']).agg(
            recall_hamcg=('recall_at_k', 'mean'),
            lat_hamcg_us=('avg_lat_us', 'mean'),
        ).round(4)
        joined = agg.join(href)
        print(f"\n=== JOINED WITH HAMCG r097 ===", flush=True)
        print(joined.to_string(), flush=True)
        joined.to_csv(OUT_DIR / 'arwgi_sift100m_summary_with_hamcg.csv')

    # update meta
    meta = {}
    if META_JSON.exists():
        meta = json.loads(META_JSON.read_text())
    meta['bench_done'] = True
    meta['bench_n_rows'] = len(df)
    META_JSON.write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['build_hnsw', 'build_wv', 'bench', 'all'],
                    default='all')
    ap.add_argument('--r', type=int, default=DEFAULT_R)
    ap.add_argument('--seed', type=int, default=0xA17)
    ap.add_argument('--force', action='store_true', help='rebuild even if cached')
    args = ap.parse_args()

    if args.stage in ('build_hnsw', 'all'):
        stage_build_hnsw(args)
    if args.stage in ('build_wv', 'all'):
        stage_build_wv(args)
    if args.stage in ('bench', 'all'):
        stage_bench(args)


if __name__ == '__main__':
    main()
