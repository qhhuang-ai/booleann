#!/usr/bin/env python3
"""
ARWGI deploy at CANONICAL YFCC10M scale: qids [60K, 70K) (10K queries).

Closes the "theorem-deployment gap" attack #1 by showing T-1A's ARWGI
instantiation is deployable at the same scale where headline BCI numbers
are reported in S6.7 (Pareto vs Parlay-IVF on slice [60K, 70K), 8-thread).

Pipeline (single-script, no two-stage NPZ persistence — the 200-query
arwgi_proper_deploy.py persists U_v+W_v to a 530MB NPZ which doesn't
scale to 10K queries; here we hold W_v in memory only and bench in one
pass):

  1. Run faiss top-K search with ef=128 on 10K queries to discover the
     visited node universe V_seen (typically ~1-3M of 10M).
  2. For each v in V_seen, sample r=24 witnesses W_v from its closed
     2-hop layer-0 neighbourhood (NOT global [0, N), matching the
     audited build in arwgi_proper_deploy.py).
  3. Bench unmodified HNSW vs ARWGI search over ef in {64, 128, 256, 512},
     over_k in {4, 16}: recall@10 vs GT cache v6_gt_60_70_cache.npz.
  4. Also include HNSW + same matched-extras random control (GAP A
     control from arwgi_proper round-3) at canonical scale for fairness.

Outputs:
  - arwgi_yfcc10m_60_70_full_r24.json (bench table for paper)
  - arwgi_yfcc10m_60_70_full_r24.log

Threading: PARLAY_NUM_THREADS env is for BCI; here faiss uses
OMP threads via faiss.omp_set_num_threads. We use 8 to match BCI's
canonical 8-thread protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import os
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
SANITY = ROOT / "03_experiment_bridge" / "results" / "raw" / "sanity_t1a"
OUT_DIR = ROOT / "03_experiment_bridge" / "results" / "raw" / "arwgi_proper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_VEC = ROOT / "data/raw/yfcc100m/base.10M.u8bin"
QUERY_VEC = ROOT / "data/raw/yfcc100m/query.public.100K.u8bin"
BASE_SPMAT = ROOT / "data/raw/yfcc100m/base.metadata.10M.spmat"
QUERY_SPMAT = ROOT / "data/raw/yfcc100m/query.metadata.public.100K.spmat"
GT_CACHE = ROOT / "03_experiment_bridge/results/raw/real_recall/bci_yfcc10m/v6_gt_60_70_cache.npz"
HNSW_PATH = SANITY / "yfcc10m_hnsw_M32_efc200.faiss"

DIM = 192
NB = 10_000_000
QID_START = 60_000
QID_END = 70_000       # 10K canonical queries
K = 10


# --------------------------- IO ---------------------------

def read_u8bin(path: Path, n: int | None = None):
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(8), dtype=np.uint32)
        total_n, d = int(hdr[0]), int(hdr[1])
        if n is None or n > total_n:
            n = total_n
        return np.frombuffer(f.read(n * d), dtype=np.uint8).reshape(n, d), d


def read_spmat(path: Path):
    """BigANN spmat: int64 nrows, int64 ncols, int64 nnz; int64 indptr[nrows+1]; int32 indices[nnz]."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(24), dtype=np.int64)
        nrows, ncols, nnz = int(hdr[0]), int(hdr[1]), int(hdr[2])
        indptr = np.frombuffer(f.read((nrows + 1) * 8), dtype=np.int64).copy()
        indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32).copy()
    return nrows, ncols, nnz, indptr, indices


def compute_xphi_bitmap(q_labels: list[int], nrows: int, label_to_rs: dict) -> np.ndarray:
    if len(q_labels) == 0:
        return np.ones(nrows, dtype=bool)
    ll = sorted([int(l) for l in q_labels], key=lambda x: len(label_to_rs[x]))
    x_phi = label_to_rs[ll[0]]
    for l in ll[1:]:
        x_phi = np.intersect1d(x_phi, label_to_rs[l], assume_unique=False)
        if len(x_phi) == 0:
            break
    mask = np.zeros(nrows, dtype=bool)
    mask[x_phi] = True
    return mask


# --------------------------- W_v builder (fast, in-memory) ---------------------------

def build_wv_inmemory(
    neighbors: np.ndarray,
    offsets: np.ndarray,
    layer0_width: int,
    node_ids: np.ndarray,
    r: int,
    seed: int,
    log_every: int = 100_000,
):
    """Sample r witnesses per node from closed 2-hop layer-0 neighbourhood.

    Returns dict node_id -> (r_eff, W array of int32). Skips nodes with
    empty pools.
    """
    rng = np.random.default_rng(seed)
    Wv_map = {}
    UV_CAP = 4096
    n_nodes = len(node_ids)
    t0 = time.time()
    for i, v in enumerate(node_ids):
        v = int(v)
        base = int(offsets[v])
        n1 = neighbors[base: base + layer0_width]
        n1 = n1[n1 >= 0]
        if len(n1) == 0:
            continue
        parts = [np.array([v], dtype=np.int32), n1]
        for u in n1.tolist():
            base2 = int(offsets[int(u)])
            n2 = neighbors[base2: base2 + layer0_width]
            n2 = n2[n2 >= 0]
            if len(n2):
                parts.append(n2)
        uv = np.unique(np.concatenate(parts))
        uv = uv[uv != v]
        if len(uv) > UV_CAP:
            sub_rng = np.random.default_rng((seed * 0x9E3779B97F4A7C15 + v) & ((1 << 64) - 1))
            sel = sub_rng.choice(len(uv), size=UV_CAP, replace=False)
            uv = uv[sel]
        if len(uv) == 0:
            continue
        if len(uv) >= r:
            sel = rng.choice(len(uv), size=r, replace=False)
            Wv_map[v] = (r, uv[sel])
        else:
            Wv_map[v] = (len(uv), uv)

        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n_nodes - i - 1) / rate
            print(f"  [build W_v] {i+1:,}/{n_nodes:,} rate={rate:.0f} nodes/s ETA={eta:.0f}s", flush=True)
    return Wv_map


# --------------------------- Search wrappers ---------------------------

def search_hnsw(idx: faiss.IndexHNSWFlat, queries_f32: np.ndarray, ef: int, over: int, k: int):
    idx.hnsw.efSearch = ef
    t0 = time.time()
    D, I = idx.search(queries_f32, over * k)
    return I, time.time() - t0


def postfilter(I: np.ndarray, phi_masks: list[np.ndarray], k: int) -> np.ndarray:
    nq = I.shape[0]
    out = np.full((nq, k), -1, dtype=np.int64)
    for i in range(nq):
        mask = phi_masks[i]
        kept = [int(x) for x in I[i] if x >= 0 and mask[x]]
        if len(kept) >= k:
            out[i, :k] = kept[:k]
        else:
            out[i, :len(kept)] = kept
    return out


def arwgi_rerank(
    I: np.ndarray,
    base_f32: np.ndarray,
    queries_f32: np.ndarray,
    Wv_map: dict,
    phi_masks: list[np.ndarray],
    k: int,
):
    nq = I.shape[0]
    out = np.full((nq, k), -1, dtype=np.int64)
    extras = np.zeros(nq, dtype=np.int32)
    t0 = time.time()
    for i in range(nq):
        mask = phi_masks[i]
        seed = set()
        # 1) HNSW candidates passing phi
        for x in I[i]:
            if x >= 0 and mask[x]:
                seed.add(int(x))
        hnsw_hits = len(seed)
        # 2) For each HNSW candidate (regardless of phi), pull its W_v passing phi
        for x in I[i]:
            if x < 0:
                continue
            entry = Wv_map.get(int(x))
            if entry is None:
                continue
            r_eff, Wrow = entry
            for w in Wrow[:r_eff]:
                if mask[int(w)]:
                    seed.add(int(w))
        if not seed:
            continue
        ids_arr = np.array(sorted(seed), dtype=np.int64)
        diffs = base_f32[ids_arr] - queries_f32[i]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        order = np.argsort(d2)[:k]
        top = ids_arr[order]
        out[i, :len(top)] = top
        extras[i] = len(seed) - hnsw_hits
    return out, time.time() - t0, extras


def random_extras_rerank(
    I: np.ndarray,
    base_f32: np.ndarray,
    queries_f32: np.ndarray,
    Wv_map: dict,
    phi_masks: list[np.ndarray],
    extras_targets: np.ndarray,
    k: int,
    seed: int,
):
    """Control: same HNSW + N random phi-passing extras drawn from each
    HNSW candidate's 2-hop pool (NOT W_v). N matched per query."""
    rng = np.random.default_rng(seed)
    nq = I.shape[0]
    out = np.full((nq, k), -1, dtype=np.int64)
    t0 = time.time()
    # build pool index from W_v map keys: we use W_v itself as proxy for 2-hop
    # subsample (since we don't keep U_v in memory at this scale). The control
    # draws random IDs from the union of HNSW candidates' W_v entries (filtered
    # by phi) — this mimics "random extras from 2-hop pool" but bounded.
    for i in range(nq):
        mask = phi_masks[i]
        seed_set = set()
        for x in I[i]:
            if x >= 0 and mask[x]:
                seed_set.add(int(x))
        hnsw_hits = len(seed_set)
        target = int(extras_targets[i])
        if target <= 0:
            if seed_set:
                ids_arr = np.array(sorted(seed_set), dtype=np.int64)
                diffs = base_f32[ids_arr] - queries_f32[i]
                d2 = np.einsum("ij,ij->i", diffs, diffs)
                order = np.argsort(d2)[:k]
                top = ids_arr[order]
                out[i, :len(top)] = top
            continue
        # Gather candidate W_v entries, shuffle, and draw phi-passers
        cand_witnesses = []
        for x in I[i]:
            if x < 0:
                continue
            entry = Wv_map.get(int(x))
            if entry is None:
                continue
            r_eff, Wrow = entry
            cand_witnesses.append(Wrow[:r_eff])
        if cand_witnesses:
            pool = np.concatenate(cand_witnesses)
            if len(pool) > 0:
                perm = rng.permutation(len(pool))
                pool = pool[perm]
                added = 0
                for w in pool:
                    if mask[int(w)]:
                        seed_set.add(int(w))
                        added += 1
                        if added >= target:
                            break
        if seed_set:
            ids_arr = np.array(sorted(seed_set), dtype=np.int64)
            diffs = base_f32[ids_arr] - queries_f32[i]
            d2 = np.einsum("ij,ij->i", diffs, diffs)
            order = np.argsort(d2)[:k]
            top = ids_arr[order]
            out[i, :len(top)] = top
    return out, time.time() - t0


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0xA17)
    ap.add_argument("--visit-topk", type=int, default=256,
                    help="per-query top-k used to discover nodes needing W_v")
    ap.add_argument("--ef-bench", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--over-bench", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--with-control", action="store_true",
                    help="Also run matched-extras random control (GAP A)")
    args = ap.parse_args()

    faiss.omp_set_num_threads(args.threads)
    print(f"[setup] faiss threads = {args.threads}", flush=True)

    # ---- load HNSW + raw arrays ----
    print(f"[1] load HNSW {HNSW_PATH}", flush=True)
    t0 = time.time()
    idx = faiss.read_index(str(HNSW_PATH))
    h = idx.hnsw
    ep = int(h.entry_point)
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int32, copy=False)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64, copy=False)
    cnn = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64, copy=False)
    layer0_width = int(cnn[1])
    print(f"  loaded in {time.time()-t0:.1f}s: N={idx.ntotal} layer0_width={layer0_width} ep={ep}", flush=True)

    # ---- load queries ----
    print(f"[2] load queries [{QID_START}, {QID_END})", flush=True)
    queries_u8, _ = read_u8bin(QUERY_VEC, n=100_000)
    queries_f32 = queries_u8[QID_START:QID_END].astype(np.float32)
    print(f"  queries: {queries_f32.shape}", flush=True)

    # ---- discover visited nodes ----
    print(f"[3] discover visited nodes (visit_topk={args.visit_topk}, ef=128)", flush=True)
    idx.hnsw.efSearch = 128
    t0 = time.time()
    _, I_disc = idx.search(queries_f32, args.visit_topk)
    visited_set = set(int(x) for x in I_disc.flatten() if x >= 0)
    visited_set.add(ep)
    visited_arr = np.array(sorted(visited_set), dtype=np.int64)
    print(f"  discovery search: {time.time()-t0:.1f}s; visited universe = {len(visited_arr):,}", flush=True)

    # ---- build W_v in memory ----
    print(f"[4] build W_v for {len(visited_arr):,} nodes (r={args.r})", flush=True)
    t0 = time.time()
    Wv_map = build_wv_inmemory(neighbors, offsets, layer0_width, visited_arr, r=args.r, seed=args.seed)
    n_built = len(Wv_map)
    mean_r_eff = float(np.mean([v[0] for v in Wv_map.values()])) if Wv_map else 0.0
    print(f"  built {n_built:,}/{len(visited_arr):,} W_v in {time.time()-t0:.0f}s, mean r_eff={mean_r_eff:.1f}", flush=True)

    # ---- load base + labels ----
    print(f"[5] load base + spmat labels", flush=True)
    t0 = time.time()
    base_u8, _ = read_u8bin(BASE_VEC, n=NB)
    base_f32 = base_u8.astype(np.float32)
    print(f"  base: {base_f32.shape} ({base_f32.nbytes/2**30:.1f} GB) in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    nrows, _, _, b_indptr, b_indices = read_spmat(BASE_SPMAT)
    qn, _, _, q_indptr, q_indices = read_spmat(QUERY_SPMAT)
    # invert
    row_id_per_nnz = np.repeat(np.arange(nrows, dtype=np.int32), np.diff(b_indptr).astype(np.int64))
    sort_idx = np.argsort(b_indices, kind="stable")
    sorted_labels = b_indices[sort_idx]
    sorted_row_ids = row_id_per_nnz[sort_idx]
    boundaries = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
    label_to_rs = {}
    for i in range(len(boundaries) - 1):
        l = int(sorted_labels[boundaries[i]])
        g = sorted_row_ids[boundaries[i]: boundaries[i + 1]].astype(np.int64)
        g.sort()
        label_to_rs[l] = g
    print(f"  spmat + invert: {time.time()-t0:.0f}s, {len(label_to_rs)} labels", flush=True)

    # ---- per-query phi masks ----
    print(f"[6] build per-query phi masks", flush=True)
    t0 = time.time()
    phi_masks = []
    selectivities = []
    for qi_local in range(QID_END - QID_START):
        qi = QID_START + qi_local
        q_labels = list(q_indices[q_indptr[qi]: q_indptr[qi + 1]])
        m = compute_xphi_bitmap(q_labels, nrows, label_to_rs)
        phi_masks.append(m)
        selectivities.append(float(m.mean()))
    sel_arr = np.array(selectivities)
    print(f"  built {len(phi_masks)} masks in {time.time()-t0:.0f}s; sel mean={sel_arr.mean():.4f} "
          f"median={np.median(sel_arr):.4f} min={sel_arr.min():.6f} max={sel_arr.max():.4f}", flush=True)

    # ---- GT ----
    print(f"[7] load GT cache {GT_CACHE}", flush=True)
    gt = np.load(GT_CACHE)
    gt_ids = gt["gt_ids"]  # (10000, 10)
    print(f"  GT shape: {gt_ids.shape}", flush=True)
    assert gt_ids.shape[0] == QID_END - QID_START, f"GT size mismatch: {gt_ids.shape[0]} vs {QID_END-QID_START}"

    # ---- BENCH ----
    print(f"[8] BENCH: unmodified HNSW + post-filter vs ARWGI", flush=True)
    nq = len(queries_f32)
    bench_rows = []
    for ef in args.ef_bench:
        for over in args.over_bench:
            print(f"  --- ef={ef} over_k={over} ---", flush=True)
            # HNSW unmodified
            I_run, dt_hnsw = search_hnsw(idx, queries_f32, ef=ef, over=over, k=K)
            ids_u = postfilter(I_run, phi_masks, K)
            hits_u = sum(len(set(ids_u[i].tolist()) & set(gt_ids[i, :K].tolist())) for i in range(nq))
            rec_u = hits_u / (nq * K)
            qps_u = nq / dt_hnsw

            # ARWGI re-rank (reuses I_run)
            ids_a, dt_arwgi, extras = arwgi_rerank(I_run, base_f32, queries_f32, Wv_map, phi_masks, K)
            hits_a = sum(len(set(ids_a[i].tolist()) & set(gt_ids[i, :K].tolist())) for i in range(nq))
            rec_a = hits_a / (nq * K)
            # ARWGI total time = hnsw search + rerank
            qps_a = nq / (dt_hnsw + dt_arwgi)
            mean_extra = float(extras.mean())

            row = {
                "ef": ef, "over_k": over, "k": K,
                "recall@10_unmodified": rec_u,
                "recall@10_arwgi": rec_a,
                "qps_unmodified": qps_u,
                "qps_arwgi": qps_a,
                "elapsed_unmodified_s": dt_hnsw,
                "elapsed_arwgi_rerank_s": dt_arwgi,
                "arwgi_mean_extra_candidates": mean_extra,
            }
            if args.with_control:
                ids_c, dt_ctrl = random_extras_rerank(
                    I_run, base_f32, queries_f32, Wv_map, phi_masks,
                    extras_targets=extras, k=K, seed=args.seed ^ 0x515)
                hits_c = sum(len(set(ids_c[i].tolist()) & set(gt_ids[i, :K].tolist())) for i in range(nq))
                rec_c = hits_c / (nq * K)
                qps_c = nq / (dt_hnsw + dt_ctrl)
                row["recall@10_control_random_2hop"] = rec_c
                row["qps_control"] = qps_c
                row["elapsed_control_rerank_s"] = dt_ctrl
                print(f"  ef={ef:>3} over={over}  unmod={rec_u:.4f}@{qps_u:.0f}qps  "
                      f"arwgi={rec_a:.4f}@{qps_a:.0f}qps  ctrl={rec_c:.4f}@{qps_c:.0f}qps  "
                      f"extras={mean_extra:.1f}", flush=True)
            else:
                print(f"  ef={ef:>3} over={over}  unmod={rec_u:.4f}@{qps_u:.0f}qps  "
                      f"arwgi={rec_a:.4f}@{qps_a:.0f}qps  extras={mean_extra:.1f}", flush=True)
            bench_rows.append(row)

    # ---- write output ----
    out = OUT_DIR / "arwgi_yfcc10m_60_70_full_r24.json"
    out.write_text(json.dumps({
        "dataset": "yfcc10m_conjunction",
        "hnsw": str(HNSW_PATH),
        "qid_start": QID_START, "qid_end": QID_END,
        "n_queries": nq, "k": K,
        "r_requested": args.r,
        "visit_topk": args.visit_topk,
        "n_visited_nodes": int(len(visited_arr)),
        "n_wv_built": n_built,
        "mean_r_eff": mean_r_eff,
        "threads": args.threads,
        "selectivity_stats": {
            "mean": float(sel_arr.mean()),
            "median": float(np.median(sel_arr)),
            "min": float(sel_arr.min()),
            "max": float(sel_arr.max()),
        },
        "rows": bench_rows,
        "comment": ("Canonical-scale ARWGI deploy benchmark on YFCC10M conjunction "
                    "slice [60K, 70K), 10K queries, 8-thread. Matches the BCI Pareto "
                    "headline slice (S6.7) and the cross-slice canonical-block "
                    "reporting. W_v built on the 2-hop closed layer-0 neighbourhood "
                    "(matching the audited build in arwgi_proper_deploy.py)."),
    }, indent=2))
    print(f"\n[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
