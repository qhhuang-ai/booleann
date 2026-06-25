#!/usr/bin/env python3
"""
ARWGI proper deploy + concentration audit (T-1A faithful).

Per paper sections/03_above_threshold.tex Theorem T-1A (lines 20-40):
  Build-time: for each v in V, sample r witnesses W_v subset U_v
              uniformly WITHOUT REPLACEMENT.
  U_v       : local oracle pool used to choose N(v) at construction.
              We use 2-hop layer-0 neighbourhood union (closed) as surrogate
              for the build-time candidate pool (the paper says U_v
              construction is implementation-defined).
  Claim     : |W_v cap X_phi| concentrates around r * p_v(phi)
              within (1+/-eps) for all phi simultaneously, where
              p_v(phi) = |U_v cap X_phi| / |U_v|.

This script (fixes the 4 critical adversarial gaps in arwgi_deploy_bench.py):

  G1 [WRONG POOL]   : witnesses sampled from local U_v (2-hop), not global [0,N)
  G2 [WRONG SPACE]  : W_v fixed at build time, persisted to NPZ; query just reads
  G3 [NO FILTER]    : measure |W_v cap X_phi| with REAL YFCC predicates phi
  G4 [WRONG CTRL]   : compare ARWGI search (full hierarchical descent) vs
                      unmodified faiss HNSW (full descent), not kw=0 strawman

Dataset: YFCC10M (10M base, 192d uint8, sparse labels), held-out qids
         [90000, 90200) (200 queries) using pre-computed AND-conjunction GT.

Stages:
  build : compute U_v (2-hop layer-0) and sample W_v; persist to NPZ
  audit : for 200 queries, compute X_phi, then for visited descent nodes,
          measure |W_v cap X_phi|/r_v and compare to p_v(phi) and theorem
          envelope (1+/-eps) p_v(phi).
  bench : recall@10 of ARWGI search (HNSW descent + W_v re-rank inside
          accepted set) vs unmodified HNSW + post-filter.

Output: 03_experiment_bridge/results/raw/arwgi_proper/
"""

from __future__ import annotations

import argparse
import heapq
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
GT_CACHE = ROOT / "03_experiment_bridge/results/raw/real_recall/bci_yfcc10m/v6_gt_90_100_cache.npz"
HNSW_PATH = SANITY / "yfcc10m_hnsw_M32_efc200.faiss"

DIM = 192
NB = 10_000_000
QID_START = int(os.environ.get("QID_START", "90000"))
QID_END = int(os.environ.get("QID_END", "90200"))       # 200 held-out queries (env-override)
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
    """BigANN spmat: int64 nrows, int64 ncols, int64 nnz; int64 indptr[nrows+1]; int32 indices[nnz].

    Verified format from prior code (gen_yfcc10m_gt_holdout_90_100.py).
    """
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(24), dtype=np.int64)
        nrows, ncols, nnz = int(hdr[0]), int(hdr[1]), int(hdr[2])
        indptr = np.frombuffer(f.read((nrows + 1) * 8), dtype=np.int64).copy()
        indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32).copy()
    return nrows, ncols, nnz, indptr, indices


# --------------------------- HNSW view ---------------------------

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


# --------------------------- U_v + W_v build ---------------------------

def build_uv_wv(
    view: HNSWView,
    node_ids: np.ndarray,
    r: int,
    seed: int = 0xA17,
    log_every: int = 5_000,
):
    """For each v in node_ids, compute closed 2-hop layer-0 neighbourhood
    U_v = {v} cup N(v) cup union_{u in N(v)} N(u), then sample r witnesses
    uniformly without replacement from U_v \ {v} (so v itself is excluded).

    Returns:
      Uv_indptr (int64 [len(node_ids)+1]),
      Uv_indices (int32 flat),
      Uv_sizes (int32 [len(node_ids)]),
      Wv (int32 [len(node_ids), r])  -- -1 padded if |U_v| < r+1 (then we sample
                                         with replacement to fill, marked).
      r_eff (int32 [len(node_ids)])   -- effective r used per node.
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(node_ids)
    Wv = np.full((n_nodes, r), -1, dtype=np.int32)
    r_eff = np.zeros(n_nodes, dtype=np.int32)
    Uv_sizes = np.zeros(n_nodes, dtype=np.int32)
    Uv_parts = []           # list of int32 arrays (one per node, only first 64*M slot)
    Uv_indptr = np.zeros(n_nodes + 1, dtype=np.int64)

    # cap U_v size cap to keep memory bounded; 2-hop closed nbhd at M=32 is
    # roughly 64 * 65 = ~4160 unique IDs in worst case; we keep all.
    UV_CAP = 4096

    t0 = time.time()
    for i, v in enumerate(node_ids):
        v = int(v)
        n1 = view.layer0_neighbors(v)
        # gather 2-hop
        if len(n1) == 0:
            uv = np.array([v], dtype=np.int64)
        else:
            # union of n1 plus 2-hop expansion
            parts = [np.array([v], dtype=np.int64), n1]
            for u in n1.tolist():
                n2 = view.layer0_neighbors(int(u))
                if len(n2):
                    parts.append(n2)
            uv = np.unique(np.concatenate(parts))
        uv_minus_v = uv[uv != v]
        if len(uv_minus_v) > UV_CAP:
            # bounded; subsample uniformly to keep within cap (deterministic per node)
            sub_rng = np.random.default_rng((seed * 0x9E3779B97F4A7C15 + v) & ((1 << 64) - 1))
            sel = sub_rng.choice(len(uv_minus_v), size=UV_CAP, replace=False)
            uv_minus_v = uv_minus_v[sel]
        Uv_sizes[i] = len(uv_minus_v)
        Uv_parts.append(uv_minus_v.astype(np.int32, copy=False))
        Uv_indptr[i + 1] = Uv_indptr[i] + len(uv_minus_v)

        if len(uv_minus_v) >= r:
            # paper's "without replacement" condition r <= |U_v|/2 may not hold for sparse pools
            # but we still sample without replacement when |U_v| >= r.
            sel = rng.choice(len(uv_minus_v), size=r, replace=False)
            Wv[i] = uv_minus_v[sel]
            r_eff[i] = r
        elif len(uv_minus_v) > 0:
            # use all available, mark r_eff < r
            Wv[i, :len(uv_minus_v)] = uv_minus_v
            r_eff[i] = len(uv_minus_v)
        else:
            r_eff[i] = 0

        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n_nodes - i - 1) / rate
            print(f"  [build U_v/W_v] {i+1:,}/{n_nodes:,} rate={rate:.0f} nodes/s ETA={eta:.0f}s", flush=True)

    Uv_indices = np.concatenate(Uv_parts).astype(np.int32, copy=False)
    return Uv_indptr, Uv_indices, Uv_sizes, Wv, r_eff


# --------------------------- p_v(phi) and |W_v cap X_phi| ---------------------------

def compute_xphi_bitmap(
    base_indptr: np.ndarray,
    base_indices: np.ndarray,
    q_labels: list[int],
    nrows: int,
    label_to_rs: dict,
) -> np.ndarray:
    """Return boolean mask of length nrows, True iff record satisfies AND of q_labels."""
    if len(q_labels) == 0:
        return np.ones(nrows, dtype=bool)
    # rare-label first
    ll = sorted([int(l) for l in q_labels], key=lambda x: len(label_to_rs[x]))
    x_phi = label_to_rs[ll[0]]
    for l in ll[1:]:
        x_phi = np.intersect1d(x_phi, label_to_rs[l], assume_unique=False)
        if len(x_phi) == 0:
            break
    mask = np.zeros(nrows, dtype=bool)
    mask[x_phi] = True
    return mask


# --------------------------- Search wrappers ---------------------------

def faiss_search_unmodified(
    idx: faiss.IndexHNSWFlat,
    queries_f32: np.ndarray,
    ef: int,
    over: int,
    k: int,
    phi_masks: list[np.ndarray],
):
    """Unmodified HNSW search + post-filter: search top-(over*k), then keep top-k matching phi."""
    idx.hnsw.efSearch = ef
    nq = queries_f32.shape[0]
    t0 = time.time()
    D, I = idx.search(queries_f32, over * k)
    filtered_ids = np.full((nq, k), -1, dtype=np.int64)
    for i in range(nq):
        mask = phi_masks[i]
        kept = [int(x) for x in I[i] if x >= 0 and mask[x]]
        if len(kept) >= k:
            filtered_ids[i, :k] = kept[:k]
        else:
            filtered_ids[i, :len(kept)] = kept
    t1 = time.time()
    return filtered_ids, t1 - t0


def arwgi_search(
    view: HNSWView,
    base_f32: np.ndarray,
    queries_f32: np.ndarray,
    Wv_map: dict,          # node_id -> (r_eff, W array)
    ef: int,
    over: int,
    k: int,
    phi_masks: list[np.ndarray],
):
    """ARWGI search: faiss returns top-(over*k) candidates from full hierarchical
    descent; we then expand each candidate's W_v as additional re-ranking
    candidates that pass phi, and pick top-k by distance.

    This is the natural deployment: descent uses HNSW edges; at the leaf, W_v
    provides extra randomized re-ranking candidates that the theorem guarantees
    are dense in X_phi.
    """
    view_idx = view.idx
    view_idx.hnsw.efSearch = ef
    nq = queries_f32.shape[0]
    t0 = time.time()
    D, I = view_idx.search(queries_f32, over * k)
    filtered_ids = np.full((nq, k), -1, dtype=np.int64)
    extra_used = np.zeros(nq, dtype=np.int32)
    for i in range(nq):
        mask = phi_masks[i]
        seed = set()
        # 1) HNSW candidates that pass phi
        for x in I[i]:
            if x >= 0 and mask[x]:
                seed.add(int(x))
        # 2) For each HNSW candidate (regardless of phi), pull its W_v passing phi
        for x in I[i]:
            if x < 0 or int(x) not in Wv_map:
                continue
            r_eff, Wrow = Wv_map[int(x)]
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
        filtered_ids[i, :len(top)] = top
        extra_used[i] = len(seed) - sum(1 for x in I[i] if x >= 0 and mask[x])
    t1 = time.time()
    return filtered_ids, t1 - t0, extra_used


# --------------------------- Concentration audit ---------------------------

def concentration_audit(
    view: HNSWView,
    base_f32: np.ndarray,
    queries_f32: np.ndarray,
    Wv_node_ids: np.ndarray,         # nodes for which W_v was built (sorted)
    Wv_array: np.ndarray,            # [len(Wv_node_ids), r]
    r_eff_array: np.ndarray,
    Uv_indptr: np.ndarray,
    Uv_indices: np.ndarray,
    Uv_sizes: np.ndarray,
    phi_masks: list[np.ndarray],
    ef_for_visit: int = 64,
    p_min: float = 0.01,
    max_visited_per_q: int = 200,
):
    """For each query, find the visited descent nodes (via faiss search), then
    for each visited node v that has a W_v entry compute:
      p_v(phi) = |U_v cap X_phi| / |U_v|       (build-time ground truth on U_v)
      Wv_rate  = |W_v cap X_phi| / r_eff       (empirical concentration)
      gap      = Wv_rate - p_v(phi)

    Theorem envelope: (1-eps)*p_v <= Wv_rate <= (1+eps)*p_v  (with epsilon from r).
    """
    view_idx = view.idx
    view_idx.hnsw.efSearch = ef_for_visit
    nq = queries_f32.shape[0]
    # map node_id -> index into Wv arrays
    node_pos = {int(n): i for i, n in enumerate(Wv_node_ids.tolist())}

    # Trigger faiss search to use as a proxy for visited cells. faiss doesn't
    # expose visited node list directly; we use the returned top-(ef) candidates
    # as the visited set surrogate. The paper's concentration claim is
    # simultaneous over ALL v in V, so any subset is fair game.
    D, I = view_idx.search(queries_f32, ef_for_visit)

    rows = []
    for qi in range(nq):
        mask = phi_masks[qi]
        # visited = top-ef faiss candidates restricted to W_v-built nodes
        visited_nodes = [int(x) for x in I[qi] if x >= 0 and int(x) in node_pos]
        if len(visited_nodes) > max_visited_per_q:
            visited_nodes = visited_nodes[:max_visited_per_q]
        for v in visited_nodes:
            idx_v = node_pos[v]
            r_eff = int(r_eff_array[idx_v])
            if r_eff == 0:
                continue
            # build-time p_v(phi) on U_v
            start = int(Uv_indptr[idx_v]); end = int(Uv_indptr[idx_v + 1])
            uv_ids = Uv_indices[start:end].astype(np.int64)
            if len(uv_ids) == 0:
                continue
            pv = float(mask[uv_ids].sum()) / float(len(uv_ids))
            if pv < p_min:
                continue
            # W_v cap X_phi (only first r_eff entries are real)
            wrow = Wv_array[idx_v, :r_eff].astype(np.int64)
            wv_hits = int(mask[wrow].sum())
            wv_rate = wv_hits / r_eff
            rows.append({
                "qid": int(QID_START + qi),
                "node": v,
                "r_eff": r_eff,
                "Uv_size": int(Uv_sizes[idx_v]),
                "p_v_phi": pv,
                "wv_hits": wv_hits,
                "wv_rate": wv_rate,
                "abs_gap": float(abs(wv_rate - pv)),
                "rel_gap": float(abs(wv_rate - pv) / pv) if pv > 0 else 0.0,
            })
    return rows


# --------------------------- Main stages ---------------------------

def stage_build(args):
    print("=== STAGE: build U_v + W_v (subset) ===", flush=True)
    print(f"  load HNSW {HNSW_PATH}", flush=True)
    idx = faiss.read_index(str(HNSW_PATH))
    view = HNSWView(idx)
    print(f"  N={view.N} layer0_width={view.layer0_width} ep={view.entry_point}", flush=True)

    # Determine which nodes to build W_v for. Full N=10M is feasible
    # (~10M * (2-hop ~2K)) but disk-heavy. We instead build for the
    # UNION of nodes that the 200 held-out queries will visit, with
    # extra padding to also cover descent. We use ef=128 search to
    # collect candidate node IDs first.
    print("  load queries + base for visit prediction", flush=True)
    queries_u8, _ = read_u8bin(QUERY_VEC, n=100_000)
    queries_f32 = queries_u8[QID_START:QID_END].astype(np.float32)
    # discover visited node set via faiss top-256
    idx.hnsw.efSearch = max(args.ef_visit, 128)
    _, I = idx.search(queries_f32, args.visit_topk)
    visited_set = set(int(x) for x in I.flatten() if x >= 0)
    # also include the entry point and its 2-hop closure for completeness
    visited_set.add(view.entry_point)
    visited_arr = np.array(sorted(visited_set), dtype=np.int64)
    print(f"  visited node set size = {len(visited_arr):,} (from {len(queries_f32)} queries x top-{args.visit_topk})", flush=True)

    print(f"  building U_v (2-hop) + W_v (r={args.r}) for {len(visited_arr):,} nodes", flush=True)
    t0 = time.time()
    Uv_indptr, Uv_indices, Uv_sizes, Wv, r_eff = build_uv_wv(
        view, visited_arr, r=args.r, seed=args.seed)
    print(f"  built in {time.time()-t0:.1f}s; mean |U_v|={Uv_sizes.mean():.1f} mean r_eff={r_eff.mean():.1f}", flush=True)

    out_npz = OUT_DIR / f"arwgi_yfcc10m_r{args.r}_wv.npz"
    np.savez(out_npz,
             node_ids=visited_arr.astype(np.int64),
             Uv_indptr=Uv_indptr,
             Uv_indices=Uv_indices,
             Uv_sizes=Uv_sizes,
             Wv=Wv,
             r_eff=r_eff,
             r_requested=np.int32(args.r),
             seed=np.int64(args.seed))
    print(f"  wrote {out_npz} ({out_npz.stat().st_size/2**20:.1f} MB)", flush=True)
    return out_npz


def stage_audit_and_bench(args, wv_path: Path):
    print(f"=== STAGE: audit + bench on YFCC10M held-out qids [{QID_START}, {QID_END}) ===", flush=True)
    print(f"  load W_v from {wv_path}", flush=True)
    d = np.load(wv_path)
    Wv_node_ids = d["node_ids"]
    Uv_indptr = d["Uv_indptr"]; Uv_indices = d["Uv_indices"]; Uv_sizes = d["Uv_sizes"]
    Wv = d["Wv"]; r_eff = d["r_eff"]; r_req = int(d["r_requested"])
    print(f"  W_v: {len(Wv_node_ids):,} nodes, r={r_req}, mean r_eff={r_eff.mean():.1f}", flush=True)

    print("  load YFCC base + queries + spmat", flush=True)
    t0 = time.time()
    # Need full base for L2 re-rank and X_phi reasoning. 10M*192 = 1.92GB uint8.
    base_u8, _ = read_u8bin(BASE_VEC, n=NB)
    base_f32 = base_u8.astype(np.float32)
    queries_u8, _ = read_u8bin(QUERY_VEC, n=100_000)
    queries_f32 = queries_u8[QID_START:QID_END].astype(np.float32)
    nrows, _, _, b_indptr, b_indices = read_spmat(BASE_SPMAT)
    qn, _, _, q_indptr, q_indices = read_spmat(QUERY_SPMAT)
    print(f"  loaded in {time.time()-t0:.0f}s; base {base_f32.shape}, queries {queries_f32.shape}", flush=True)

    # invert
    print("  invert base labels -> label_to_rs", flush=True)
    t1 = time.time()
    row_id_per_nnz = np.repeat(np.arange(nrows, dtype=np.int32), np.diff(b_indptr).astype(np.int64))
    sort_idx = np.argsort(b_indices, kind="stable")
    sorted_labels = b_indices[sort_idx]; sorted_row_ids = row_id_per_nnz[sort_idx]
    boundaries = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
    label_to_rs = {}
    for i in range(len(boundaries) - 1):
        l = int(sorted_labels[boundaries[i]])
        g = sorted_row_ids[boundaries[i]: boundaries[i + 1]].astype(np.int64); g.sort()
        label_to_rs[l] = g
    print(f"  inverted in {time.time()-t1:.0f}s", flush=True)

    # per-query predicate X_phi masks
    print("  build per-query phi masks", flush=True)
    t2 = time.time()
    phi_masks = []
    selectivities = []
    for qi_local in range(QID_END - QID_START):
        qi = QID_START + qi_local
        q_labels = list(q_indices[q_indptr[qi]: q_indptr[qi + 1]])
        m = compute_xphi_bitmap(b_indptr, b_indices, q_labels, nrows, label_to_rs)
        phi_masks.append(m)
        selectivities.append(float(m.mean()))
    sel_arr = np.array(selectivities)
    print(f"  phi masks: mean sel={sel_arr.mean():.4f} median={np.median(sel_arr):.4f} "
          f"min={sel_arr.min():.4f} max={sel_arr.max():.4f}  in {time.time()-t2:.0f}s", flush=True)

    # Load HNSW
    print(f"  load HNSW {HNSW_PATH}", flush=True)
    idx = faiss.read_index(str(HNSW_PATH))
    view = HNSWView(idx)

    # --------- AUDIT ---------
    print("\n--- audit: concentration |W_v cap X_phi|/r vs p_v(phi) ---", flush=True)
    audit_rows = concentration_audit(
        view, base_f32, queries_f32,
        Wv_node_ids, Wv, r_eff,
        Uv_indptr, Uv_indices, Uv_sizes,
        phi_masks,
        ef_for_visit=args.ef_audit,
        p_min=args.p_min,
        max_visited_per_q=args.max_visited_per_q)
    audit_path = OUT_DIR / f"arwgi_concentration_yfcc10m_r{r_req}.json"
    audit_path.write_text(json.dumps({"rows": audit_rows[:50_000]}, indent=2) if len(audit_rows) <= 50_000
                          else json.dumps({"rows": audit_rows[:50_000], "truncated_at": 50_000,
                                           "total_rows": len(audit_rows)}, indent=2))
    print(f"  audit: {len(audit_rows):,} (node, query) measurements", flush=True)
    if audit_rows:
        # theorem epsilon implied: we report empirical 1/2/median bounds
        rel_gaps = np.array([r["rel_gap"] for r in audit_rows])
        pv_arr = np.array([r["p_v_phi"] for r in audit_rows])
        rate_arr = np.array([r["wv_rate"] for r in audit_rows])
        print(f"  rel_gap: mean={rel_gaps.mean():.3f} median={np.median(rel_gaps):.3f} "
              f"p90={np.percentile(rel_gaps, 90):.3f} p99={np.percentile(rel_gaps, 99):.3f}", flush=True)
        # bucketize by p_v(phi) and report mean wv_rate per bucket
        bins = np.linspace(0, 1, 21)
        digit = np.digitize(pv_arr, bins) - 1
        print(f"  concentration by p_v(phi) bucket:")
        bucket_summary = []
        for b in range(20):
            mask = digit == b
            if mask.sum() < 5:
                continue
            print(f"    p_v in [{bins[b]:.2f},{bins[b+1]:.2f}): n={mask.sum():>5}  "
                  f"mean wv_rate={rate_arr[mask].mean():.4f}  "
                  f"mean p_v={pv_arr[mask].mean():.4f}  "
                  f"mean rel_gap={rel_gaps[mask].mean():.3f}")
            bucket_summary.append({
                "pv_lo": float(bins[b]), "pv_hi": float(bins[b+1]),
                "n": int(mask.sum()),
                "mean_wv_rate": float(rate_arr[mask].mean()),
                "mean_pv": float(pv_arr[mask].mean()),
                "mean_rel_gap": float(rel_gaps[mask].mean()),
            })
        # theorem envelope evidence: with r=24, expect rel_gap = O(1/sqrt(r*pv))
        # so for pv=0.1, r=24 -> ~0.65; for pv=0.5 -> ~0.29; for pv=0.9 -> ~0.22
        summary = {
            "n_obs": len(audit_rows),
            "rel_gap_mean": float(rel_gaps.mean()),
            "rel_gap_median": float(np.median(rel_gaps)),
            "rel_gap_p90": float(np.percentile(rel_gaps, 90)),
            "rel_gap_p99": float(np.percentile(rel_gaps, 99)),
            "by_pv_bucket": bucket_summary,
        }
        (OUT_DIR / f"arwgi_concentration_summary_yfcc10m_r{r_req}.json").write_text(
            json.dumps(summary, indent=2))
        print(f"  wrote concentration summary", flush=True)

    # --------- BENCH ---------
    print("\n--- bench: recall@10 ARWGI vs unmodified HNSW + post-filter ---", flush=True)
    # build per-node Wv_map for fast lookup in arwgi_search
    Wv_map = {int(n): (int(r_eff[i]), Wv[i]) for i, n in enumerate(Wv_node_ids.tolist())}

    # Load GT
    gt = np.load(GT_CACHE)
    gt_ids_full = gt["gt_ids"]  # (10000, 10), int64 ids in [0, 10M)
    # held-out [0, 200) of cache (which covers qids 90000..99999)
    gt_ids = gt_ids_full[: (QID_END - QID_START)]
    print(f"  GT shape {gt_ids.shape}", flush=True)

    bench_rows = []
    for ef in args.ef_bench:
        for over in args.over_bench:
            # unmodified baseline
            ids_u, dt_u = faiss_search_unmodified(idx, queries_f32, ef=ef, over=over, k=K, phi_masks=phi_masks)
            hits_u = sum(len(set(ids_u[i].tolist()) & set(gt_ids[i, :K].tolist())) for i in range(len(queries_f32)))
            rec_u = hits_u / (len(queries_f32) * K)
            # ARWGI
            ids_a, dt_a, extra = arwgi_search(view, base_f32, queries_f32, Wv_map, ef=ef, over=over, k=K, phi_masks=phi_masks)
            hits_a = sum(len(set(ids_a[i].tolist()) & set(gt_ids[i, :K].tolist())) for i in range(len(queries_f32)))
            rec_a = hits_a / (len(queries_f32) * K)
            mean_extra = float(extra.mean())
            row = {
                "ef": ef, "over_k": over, "k": K,
                "recall@10_unmodified": rec_u,
                "recall@10_arwgi": rec_a,
                "qps_unmodified": len(queries_f32) / dt_u,
                "qps_arwgi": len(queries_f32) / dt_a,
                "elapsed_unmodified_s": dt_u,
                "elapsed_arwgi_s": dt_a,
                "arwgi_mean_extra_candidates": mean_extra,
            }
            print(f"  ef={ef:>3} over={over}  unmod recall@10={rec_u:.4f} qps={row['qps_unmodified']:.1f}  | "
                  f"arwgi recall@10={rec_a:.4f} qps={row['qps_arwgi']:.1f}  extra={mean_extra:.1f}", flush=True)
            bench_rows.append(row)

    bench_path = OUT_DIR / f"arwgi_bench_yfcc10m_r{r_req}.json"
    bench_path.write_text(json.dumps({"rows": bench_rows, "qid_start": QID_START, "qid_end": QID_END,
                                       "r_requested": r_req}, indent=2))
    print(f"  wrote {bench_path}", flush=True)

    return audit_path, bench_path


# --------------------------- ROUND-3 GAP FIXES ---------------------------
# These functions plug the three Round-2 gaps:
#   GAP A (control_bench): does the +recall come from extra L2 work (any random
#          extras), or from W_v structure specifically? Run a matched control.
#   GAP B (uncond_audit) : Round-2 audited only faiss top-(ef) nodes (query-
#          conditioned, high-similarity). Re-audit on uniformly random nodes
#          from Wv_node_ids; report metrics for p_v >= 0.10 only.
#   GAP C (sift_audit)   : second dataset. Build W_v subset on SIFT10M, audit
#          concentration with synthetic Bernoulli labels at s in {0.05,0.10,0.20}.


def _load_yfcc_for_audit():
    """Shared loader: returns base_f32, queries_f32, phi_masks list."""
    base_u8, _ = read_u8bin(BASE_VEC, n=NB)
    base_f32 = base_u8.astype(np.float32)
    queries_u8, _ = read_u8bin(QUERY_VEC, n=100_000)
    queries_f32 = queries_u8[QID_START:QID_END].astype(np.float32)
    nrows, _, _, b_indptr, b_indices = read_spmat(BASE_SPMAT)
    qn, _, _, q_indptr, q_indices = read_spmat(QUERY_SPMAT)

    # invert
    row_id_per_nnz = np.repeat(np.arange(nrows, dtype=np.int32), np.diff(b_indptr).astype(np.int64))
    sort_idx = np.argsort(b_indices, kind="stable")
    sorted_labels = b_indices[sort_idx]; sorted_row_ids = row_id_per_nnz[sort_idx]
    boundaries = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
    label_to_rs = {}
    for i in range(len(boundaries) - 1):
        l = int(sorted_labels[boundaries[i]])
        g = sorted_row_ids[boundaries[i]: boundaries[i + 1]].astype(np.int64); g.sort()
        label_to_rs[l] = g

    phi_masks = []
    for qi_local in range(QID_END - QID_START):
        qi = QID_START + qi_local
        q_labels = list(q_indices[q_indptr[qi]: q_indptr[qi + 1]])
        m = compute_xphi_bitmap(b_indptr, b_indices, q_labels, nrows, label_to_rs)
        phi_masks.append(m)
    return base_f32, queries_f32, phi_masks


def stage_control_bench(args, wv_path: Path):
    """GAP A: control bench. Compare:
      (a) HNSW unmodified + post-filter
      (b) HNSW + random extras drawn from each visited node's 2-hop pool U_v
          (NOT the persisted W_v witnesses), re-ranked by L2. Same per-query
          extra count as ARWGI.
      (c) ARWGI (W_v-based extras), already in bench JSON.

    If (b) >= ARWGI -> +lift is just from extra L2 work, not W_v structure.
    """
    print(f"=== STAGE: control_bench on YFCC10M [{QID_START}, {QID_END}) ===", flush=True)
    print(f"  load W_v from {wv_path}", flush=True)
    d = np.load(wv_path)
    Wv_node_ids = d["node_ids"]
    Uv_indptr = d["Uv_indptr"]; Uv_indices = d["Uv_indices"]; Uv_sizes = d["Uv_sizes"]
    Wv = d["Wv"]; r_eff = d["r_eff"]; r_req = int(d["r_requested"])
    node_pos = {int(n): i for i, n in enumerate(Wv_node_ids.tolist())}

    print("  load YFCC base + queries + labels (this takes ~30s)", flush=True)
    t0 = time.time()
    base_f32, queries_f32, phi_masks = _load_yfcc_for_audit()
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    print(f"  load HNSW {HNSW_PATH}", flush=True)
    idx = faiss.read_index(str(HNSW_PATH))
    view = HNSWView(idx)

    # Build per-node Wv_map for arwgi_search
    Wv_map = {int(n): (int(r_eff[i]), Wv[i]) for i, n in enumerate(Wv_node_ids.tolist())}

    # Load GT
    gt = np.load(GT_CACHE)
    gt_ids_full = gt["gt_ids"]
    gt_ids = gt_ids_full[: (QID_END - QID_START)]
    K_local = K
    nq = len(queries_f32)

    def random_extras_from_uv(qi, hnsw_cands, extras_target, rng):
        """Control draw for GAP A. Match the NUMBER OF PHI-PASSING extras to
        ARWGI's extras_target. We pull random IDs (without replacement, then
        with-replacement fallback if pool is small) from each HNSW candidate's
        2-hop pool U_v, filter by phi, accumulate phi-passing IDs into a set
        until size >= extras_target or pools exhausted.

        Note (round-3 fix): the round-3 v1 implementation drew exactly
        extras_target IDs total then filtered, which collapsed to ~p_v * target
        phi-passers (~9 vs ARWGI's 40). That under-counts compute. Here we draw
        in batches per candidate and accept only phi-passers, so the control
        gets the SAME NUMBER OF PHI-PASSING extras as ARWGI -- but from
        uniformly-random U_v draws instead of from W_v witnesses.
        """
        mask = phi_masks[qi]
        extras = set()
        cand_with_uv = [int(x) for x in hnsw_cands if int(x) in node_pos]
        if not cand_with_uv:
            return extras
        # Round-robin across candidates, drawing batches of 24 (matches r) and
        # cycling until we have enough phi-passers or all pools are exhausted.
        BATCH = 24
        per_cand_drawn = {c: 0 for c in cand_with_uv}
        # Pre-fetch pools and shuffle each
        pools = {}
        for c in cand_with_uv:
            idx_v = node_pos[c]
            start = int(Uv_indptr[idx_v]); end = int(Uv_indptr[idx_v + 1])
            pool = Uv_indices[start:end]
            if len(pool) > 0:
                perm = rng.permutation(len(pool))
                pools[c] = (pool, perm)
        active = [c for c in cand_with_uv if c in pools]
        while active and len(extras) < extras_target:
            for c in list(active):
                pool, perm = pools[c]
                start_off = per_cand_drawn[c]
                end_off = min(start_off + BATCH, len(pool))
                if start_off >= len(pool):
                    active.remove(c)
                    continue
                idxs = perm[start_off:end_off]
                for w in pool[idxs]:
                    w = int(w)
                    if mask[w]:
                        extras.add(w)
                        if len(extras) >= extras_target:
                            break
                per_cand_drawn[c] = end_off
                if len(extras) >= extras_target:
                    break
        return extras

    bench_rows = []
    rng = np.random.default_rng(args.seed ^ 0x515)
    for ef in args.ef_bench:
        for over in args.over_bench:
            idx.hnsw.efSearch = ef
            t0 = time.time()
            D, I = idx.search(queries_f32, over * K_local)
            t_search = time.time() - t0

            # (a) unmodified + post-filter
            ids_u = np.full((nq, K_local), -1, dtype=np.int64)
            for i in range(nq):
                mask = phi_masks[i]
                kept = [int(x) for x in I[i] if x >= 0 and mask[x]]
                if len(kept) >= K_local:
                    ids_u[i, :K_local] = kept[:K_local]
                else:
                    ids_u[i, :len(kept)] = kept
            hits_u = sum(len(set(ids_u[i].tolist()) & set(gt_ids[i, :K_local].tolist())) for i in range(nq))
            rec_u = hits_u / (nq * K_local)

            # (b) control: HNSW + random extras from 2-hop pool (NOT W_v),
            # re-ranked by L2. Match per-query extras count to ARWGI's
            # observed extras (we recompute ARWGI in this same loop).
            # Match exactly: use the same extras_target as ARWGI per-query.
            t_b0 = time.time()
            ids_c = np.full((nq, K_local), -1, dtype=np.int64)
            extras_b_list = []
            # (c) ARWGI in same loop for matched extras_target
            t_a0 = time.time()
            ids_a = np.full((nq, K_local), -1, dtype=np.int64)
            extras_a_list = []
            for i in range(nq):
                mask = phi_masks[i]
                # ARWGI seed
                seed_a = set()
                for x in I[i]:
                    if x >= 0 and mask[x]:
                        seed_a.add(int(x))
                hnsw_hits = len(seed_a)
                for x in I[i]:
                    if x < 0 or int(x) not in node_pos:
                        continue
                    idx_v = node_pos[int(x)]
                    rev = int(r_eff[idx_v])
                    for w in Wv[idx_v, :rev]:
                        if mask[int(w)]:
                            seed_a.add(int(w))
                extras_a = len(seed_a) - hnsw_hits
                extras_a_list.append(extras_a)
                if seed_a:
                    ids_arr = np.array(sorted(seed_a), dtype=np.int64)
                    diffs = base_f32[ids_arr] - queries_f32[i]
                    d2 = np.einsum("ij,ij->i", diffs, diffs)
                    order = np.argsort(d2)[:K_local]
                    top = ids_arr[order]
                    ids_a[i, :len(top)] = top

                # control seed: same HNSW hits + matched random extras from U_v
                seed_c = set()
                for x in I[i]:
                    if x >= 0 and mask[x]:
                        seed_c.add(int(x))
                # match extras_a in count using random U_v draws
                if extras_a > 0:
                    extras_set = random_extras_from_uv(i, I[i], extras_a, rng)
                    seed_c.update(extras_set)
                extras_b_list.append(len(seed_c) - hnsw_hits)
                if seed_c:
                    ids_arr = np.array(sorted(seed_c), dtype=np.int64)
                    diffs = base_f32[ids_arr] - queries_f32[i]
                    d2 = np.einsum("ij,ij->i", diffs, diffs)
                    order = np.argsort(d2)[:K_local]
                    top = ids_arr[order]
                    ids_c[i, :len(top)] = top
            t_c = time.time() - t_b0
            t_a = time.time() - t_a0
            hits_c = sum(len(set(ids_c[i].tolist()) & set(gt_ids[i, :K_local].tolist())) for i in range(nq))
            rec_c = hits_c / (nq * K_local)
            hits_a = sum(len(set(ids_a[i].tolist()) & set(gt_ids[i, :K_local].tolist())) for i in range(nq))
            rec_a = hits_a / (nq * K_local)
            mean_extras_a = float(np.mean(extras_a_list))
            mean_extras_b = float(np.mean(extras_b_list))

            row = {
                "ef": ef, "over_k": over, "k": K_local,
                "recall@10_unmodified": rec_u,
                "recall@10_control_random_2hop": rec_c,
                "recall@10_arwgi": rec_a,
                "delta_arwgi_vs_unmodified": rec_a - rec_u,
                "delta_arwgi_vs_control": rec_a - rec_c,
                "delta_control_vs_unmodified": rec_c - rec_u,
                "mean_extras_arwgi": mean_extras_a,
                "mean_extras_control": mean_extras_b,
                "hnsw_search_time_s": t_search,
            }
            print(f"  ef={ef:>3} over={over}  unmod={rec_u:.4f}  ctrl-rand-2hop={rec_c:.4f}  "
                  f"arwgi={rec_a:.4f}  d(arwgi-ctrl)={rec_a-rec_c:+.4f}  "
                  f"extras a={mean_extras_a:.1f} b={mean_extras_b:.1f}", flush=True)
            bench_rows.append(row)

    out = OUT_DIR / f"arwgi_control_bench_yfcc10m_r{r_req}.json"
    out.write_text(json.dumps({
        "rows": bench_rows,
        "qid_start": QID_START, "qid_end": QID_END,
        "r_requested": r_req,
        "control_definition": "HNSW + N random extras drawn uniformly without replacement from "
                              "each HNSW candidate's 2-hop layer-0 pool U_v (NOT W_v witnesses), "
                              "phi-filtered, re-ranked by L2. N matched per-query to ARWGI extras.",
    }, indent=2))
    print(f"  wrote {out}", flush=True)
    return out


def stage_uncond_audit(args, wv_path: Path):
    """GAP B: unconditioned concentration audit. Sample 200 uniformly-random
    nodes per query from Wv_node_ids (no faiss top-(ef) filter). Report
    Pearson r and Bernstein envelope coverage at p_v >= 0.10 only.
    """
    print(f"=== STAGE: uncond_audit on YFCC10M [{QID_START}, {QID_END}) ===", flush=True)
    print(f"  load W_v from {wv_path}", flush=True)
    d = np.load(wv_path)
    Wv_node_ids = d["node_ids"]
    Uv_indptr = d["Uv_indptr"]; Uv_indices = d["Uv_indices"]; Uv_sizes = d["Uv_sizes"]
    Wv = d["Wv"]; r_eff = d["r_eff"]; r_req = int(d["r_requested"])

    print("  load YFCC labels (no base needed for audit)", flush=True)
    t0 = time.time()
    nrows, _, _, b_indptr, b_indices = read_spmat(BASE_SPMAT)
    qn, _, _, q_indptr, q_indices = read_spmat(QUERY_SPMAT)
    row_id_per_nnz = np.repeat(np.arange(nrows, dtype=np.int32), np.diff(b_indptr).astype(np.int64))
    sort_idx = np.argsort(b_indices, kind="stable")
    sorted_labels = b_indices[sort_idx]; sorted_row_ids = row_id_per_nnz[sort_idx]
    boundaries = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
    label_to_rs = {}
    for i in range(len(boundaries) - 1):
        l = int(sorted_labels[boundaries[i]])
        g = sorted_row_ids[boundaries[i]: boundaries[i + 1]].astype(np.int64); g.sort()
        label_to_rs[l] = g
    phi_masks = []
    for qi_local in range(QID_END - QID_START):
        qi = QID_START + qi_local
        q_labels = list(q_indices[q_indptr[qi]: q_indptr[qi + 1]])
        m = compute_xphi_bitmap(b_indptr, b_indices, q_labels, nrows, label_to_rs)
        phi_masks.append(m)
    print(f"  setup in {time.time()-t0:.0f}s", flush=True)

    nq = len(phi_masks)
    n_pool = len(Wv_node_ids)
    rng = np.random.default_rng(args.seed ^ 0xBEEF)
    samples_per_q = args.max_visited_per_q  # 200 by default

    rows = []
    for qi in range(nq):
        mask = phi_masks[qi]
        # uniform random sample of node positions
        if n_pool <= samples_per_q:
            picks = np.arange(n_pool)
        else:
            picks = rng.choice(n_pool, size=samples_per_q, replace=False)
        for idx_v in picks:
            idx_v = int(idx_v)
            rev = int(r_eff[idx_v])
            if rev == 0:
                continue
            start = int(Uv_indptr[idx_v]); end = int(Uv_indptr[idx_v + 1])
            uv_ids = Uv_indices[start:end].astype(np.int64)
            if len(uv_ids) == 0:
                continue
            pv = float(mask[uv_ids].sum()) / float(len(uv_ids))
            wrow = Wv[idx_v, :rev].astype(np.int64)
            wv_hits = int(mask[wrow].sum())
            wv_rate = wv_hits / rev
            rows.append({
                "qid": int(QID_START + qi),
                "node": int(Wv_node_ids[idx_v]),
                "r_eff": rev,
                "Uv_size": int(Uv_sizes[idx_v]),
                "p_v_phi": pv,
                "wv_hits": wv_hits,
                "wv_rate": wv_rate,
                "abs_gap": float(abs(wv_rate - pv)),
                "rel_gap": float(abs(wv_rate - pv) / pv) if pv > 0 else 0.0,
            })

    pv_arr = np.array([r["p_v_phi"] for r in rows])
    rate_arr = np.array([r["wv_rate"] for r in rows])
    print(f"  total obs (unconditioned): {len(rows):,}  "
          f"frac with pv>=0.10: {float((pv_arr >= 0.10).mean()):.4f}", flush=True)

    # Bernstein envelope at p_v >= 0.10:
    # For Binomial(r, p), additive deviation bound at confidence 1-delta:
    #   |X/r - p| <= sqrt(2 p (1-p) ln(2/delta) / r) + (2 ln(2/delta)) / (3 r)
    # We report the empirical fraction within this envelope (per row).
    delta = 0.05
    def bernstein_eps(p, r):
        if r <= 0:
            return float("inf")
        ln_term = np.log(2.0 / delta)
        return np.sqrt(2.0 * p * (1.0 - p) * ln_term / r) + (2.0 * ln_term) / (3.0 * r)

    sel = pv_arr >= 0.10
    cov_p10_within = 0.0
    pearson_p10 = float("nan")
    if sel.sum() > 0:
        ps = pv_arr[sel]
        rs = rate_arr[sel]
        # bernstein env per-row using r_eff per row
        rs_eff = np.array([rows[i]["r_eff"] for i in range(len(rows))])[sel]
        eps_per = np.array([bernstein_eps(ps[i], int(rs_eff[i])) for i in range(len(ps))])
        within = np.abs(rs - ps) <= eps_per
        cov_p10_within = float(within.mean())
        # Pearson r
        if len(ps) > 1 and ps.std() > 0 and rs.std() > 0:
            pearson_p10 = float(np.corrcoef(ps, rs)[0, 1])

    # full distribution
    pearson_all = float("nan")
    if len(pv_arr) > 1 and pv_arr.std() > 0 and rate_arr.std() > 0:
        pearson_all = float(np.corrcoef(pv_arr, rate_arr)[0, 1])

    summary = {
        "sampling": "uniform_random_from_Wv_node_ids",
        "samples_per_query": samples_per_q,
        "n_queries": nq,
        "n_obs_total": len(rows),
        "n_obs_pv_geq_0p10": int(sel.sum()),
        "pearson_r_all": pearson_all,
        "pearson_r_pv_geq_0p10": pearson_p10,
        "bernstein_delta": delta,
        "bernstein_coverage_pv_geq_0p10": cov_p10_within,
        "comment": ("Round-2 audit used faiss top-(ef=64) visited nodes, a "
                    "query-conditioned high-similarity subset. This run samples "
                    "uniformly from Wv_node_ids and reports stats only on "
                    "p_v >= 0.10 (per Round-3 request: low p_v trivially "
                    "concentrates near 0)."),
    }

    full_path = OUT_DIR / f"arwgi_concentration_yfcc10m_r{r_req}_unconditioned.json"
    full_path.write_text(json.dumps({
        "summary": summary,
        "rows": rows[:20_000] + ([{"truncated": True, "total": len(rows)}] if len(rows) > 20_000 else []),
    }, indent=2))
    print(f"  wrote {full_path}", flush=True)
    print(f"  Pearson r (pv>=0.10) = {pearson_p10:.4f}   Bernstein coverage = {cov_p10_within:.4f}", flush=True)
    return full_path


# --------------------------- SIFT10M support ---------------------------

SIFT10M_BASE = ROOT / "data/raw/sift10m/sift10m_base_fvecs_real.fvecs"
SIFT_QUERY = ROOT / "data/raw/sift/sift_query.fvecs"
SIFT10M_HNSW = SANITY / "sift10m_hnsw_M32_efc200.faiss"


def read_fvecs(path: Path, n: int | None = None) -> np.ndarray:
    """Read .fvecs: each row = int32 dim + dim float32. View-cast safe.
    Per project memory: NEVER use .astype to convert int32 ids to float32 here.
    """
    a = np.fromfile(str(path), dtype=np.int32)
    d = int(a[0])
    rec = a.reshape(-1, d + 1)
    if n is not None and n < rec.shape[0]:
        rec = rec[:n]
    return rec[:, 1:].view(np.float32).copy()


def stage_sift_audit(args):
    """GAP C: build a small W_v on SIFT10M and run concentration audit with
    synthetic Bernoulli labels at s in {0.05, 0.10, 0.20}. 100 queries,
    100 sampled nodes each (uniform random over built nodes) -> 10K obs.
    """
    print(f"=== STAGE: sift_audit on SIFT10M (synthetic Bernoulli labels) ===", flush=True)
    print(f"  load HNSW {SIFT10M_HNSW}", flush=True)
    idx = faiss.read_index(str(SIFT10M_HNSW))
    view = HNSWView(idx)
    print(f"  N={view.N} layer0_width={view.layer0_width} ep={view.entry_point}", flush=True)

    print(f"  load queries from {SIFT_QUERY}", flush=True)
    queries = read_fvecs(SIFT_QUERY, n=args.sift_nq)
    nq = queries.shape[0]
    print(f"  queries: {queries.shape}", flush=True)

    # Discover nodes to build W_v on (use ef_visit + top-(visit_topk))
    print(f"  discover visited nodes (ef_visit={args.ef_visit}, top-{args.visit_topk})", flush=True)
    idx.hnsw.efSearch = max(args.ef_visit, 128)
    _, I = idx.search(queries, args.visit_topk)
    visited_set = set(int(x) for x in I.flatten() if x >= 0)
    visited_set.add(view.entry_point)
    visited_arr = np.array(sorted(visited_set), dtype=np.int64)
    print(f"  built-node set: {len(visited_arr):,}", flush=True)

    print(f"  build U_v (2-hop) + W_v (r={args.r}) for {len(visited_arr):,} nodes", flush=True)
    t0 = time.time()
    Uv_indptr, Uv_indices, Uv_sizes, Wv, r_eff = build_uv_wv(
        view, visited_arr, r=args.r, seed=args.seed, log_every=2000)
    print(f"  built in {time.time()-t0:.0f}s; mean |U_v|={Uv_sizes.mean():.1f} mean r_eff={r_eff.mean():.1f}", flush=True)

    # Synthetic Bernoulli labels at s in {0.05, 0.10, 0.20}.
    # For each s, create one global mask (Bernoulli(s) over [0, N)).
    # This is the "phi" universally for all queries (same predicate, since
    # synthetic). Compute p_v(phi) on U_v and Wv_rate.
    rng = np.random.default_rng(args.seed ^ 0xC0DE)
    all_summary = []
    full_obs_by_s = {}
    samples_per_q = args.sift_samples_per_q
    for s in args.sift_s:
        print(f"  --- s={s} ---", flush=True)
        mask = (rng.random(view.N) < s)
        # for each "query", uniformly sample samples_per_q built nodes
        rows = []
        for qi in range(nq):
            picks = rng.choice(len(visited_arr), size=min(samples_per_q, len(visited_arr)), replace=False)
            for idx_v in picks:
                idx_v = int(idx_v)
                rev = int(r_eff[idx_v])
                if rev == 0:
                    continue
                start = int(Uv_indptr[idx_v]); end = int(Uv_indptr[idx_v + 1])
                uv_ids = Uv_indices[start:end].astype(np.int64)
                if len(uv_ids) == 0:
                    continue
                pv = float(mask[uv_ids].sum()) / float(len(uv_ids))
                wrow = Wv[idx_v, :rev].astype(np.int64)
                wv_hits = int(mask[wrow].sum())
                wv_rate = wv_hits / rev
                rows.append({
                    "qid": qi, "node": int(visited_arr[idx_v]),
                    "r_eff": rev, "Uv_size": int(Uv_sizes[idx_v]),
                    "p_v_phi": pv, "wv_hits": wv_hits, "wv_rate": wv_rate,
                    "abs_gap": float(abs(wv_rate - pv)),
                })
        pv_arr = np.array([r["p_v_phi"] for r in rows])
        rate_arr = np.array([r["wv_rate"] for r in rows])
        rs_eff = np.array([r["r_eff"] for r in rows])

        # Bernstein envelope at p_v >= 0.10
        delta = 0.05
        sel = pv_arr >= 0.10
        ln_term = np.log(2.0 / delta)
        eps = np.sqrt(2.0 * pv_arr * (1.0 - pv_arr) * ln_term / np.maximum(rs_eff, 1)) + \
              (2.0 * ln_term) / (3.0 * np.maximum(rs_eff, 1))
        within = np.abs(rate_arr - pv_arr) <= eps
        cov_p10 = float(within[sel].mean()) if sel.sum() > 0 else 0.0

        pearson = float("nan")
        if len(pv_arr) > 1 and pv_arr.std() > 0 and rate_arr.std() > 0:
            pearson = float(np.corrcoef(pv_arr, rate_arr)[0, 1])
        pearson_p10 = float("nan")
        if sel.sum() > 1 and pv_arr[sel].std() > 0 and rate_arr[sel].std() > 0:
            pearson_p10 = float(np.corrcoef(pv_arr[sel], rate_arr[sel])[0, 1])

        summary = {
            "s_target": s,
            "s_empirical_global": float(mask.mean()),
            "n_obs": len(rows),
            "n_obs_pv_geq_0p10": int(sel.sum()),
            "pearson_r_all": pearson,
            "pearson_r_pv_geq_0p10": pearson_p10,
            "bernstein_delta": delta,
            "bernstein_coverage_pv_geq_0p10": cov_p10,
            "mean_pv": float(pv_arr.mean()),
            "mean_wv_rate": float(rate_arr.mean()),
        }
        all_summary.append(summary)
        full_obs_by_s[str(s)] = rows[:5000]
        print(f"    n_obs={len(rows):,}  Pearson r (all)={pearson:.4f}  "
              f"r(pv>=0.10)={pearson_p10:.4f}  Bernstein cov(pv>=0.10)={cov_p10:.4f}", flush=True)

    out = OUT_DIR / f"arwgi_concentration_sift10m_r{args.r}.json"
    out.write_text(json.dumps({
        "dataset": "sift10m",
        "hnsw": str(SIFT10M_HNSW),
        "n_built_nodes": len(visited_arr),
        "r_requested": args.r,
        "n_queries_used_for_node_discovery_and_audit": nq,
        "samples_per_query": samples_per_q,
        "label_type": "synthetic_bernoulli_global",
        "summaries": all_summary,
        "obs_sample_by_s": full_obs_by_s,
    }, indent=2))
    print(f"  wrote {out}", flush=True)
    return out


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "audit_bench", "all",
                                          "control_bench", "uncond_audit", "sift_audit"],
                    default="all")
    ap.add_argument("--r", type=int, default=24, help="witnesses per node (paper uses r=M*gamma); 24 fits 2-hop closure")
    ap.add_argument("--seed", type=int, default=0xA17)
    ap.add_argument("--visit-topk", dest="visit_topk", type=int, default=512,
                    help="per-query top-k candidates used to discover nodes needing W_v")
    ap.add_argument("--ef-visit", dest="ef_visit", type=int, default=128,
                    help="efSearch used during build-time visited-node discovery")
    ap.add_argument("--ef-audit", dest="ef_audit", type=int, default=64,
                    help="efSearch used during concentration audit")
    ap.add_argument("--p-min", dest="p_min", type=float, default=0.001,
                    help="ignore audit points with p_v(phi) < p_min")
    ap.add_argument("--max-visited-per-q", dest="max_visited_per_q", type=int, default=200)
    ap.add_argument("--ef-bench", dest="ef_bench", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--over-bench", dest="over_bench", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--wv-path", dest="wv_path", default=None)
    # SIFT-specific
    ap.add_argument("--sift-nq", dest="sift_nq", type=int, default=100)
    ap.add_argument("--sift-samples-per-q", dest="sift_samples_per_q", type=int, default=100)
    ap.add_argument("--sift-s", dest="sift_s", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    args = ap.parse_args()

    wv_path = Path(args.wv_path) if args.wv_path else OUT_DIR / f"arwgi_yfcc10m_r{args.r}_wv.npz"

    if args.stage == "build":
        stage_build(args)
    elif args.stage == "audit_bench":
        stage_audit_and_bench(args, wv_path)
    elif args.stage == "all":
        wv_path = stage_build(args)
        stage_audit_and_bench(args, wv_path)
    elif args.stage == "control_bench":
        stage_control_bench(args, wv_path)
    elif args.stage == "uncond_audit":
        stage_uncond_audit(args, wv_path)
    elif args.stage == "sift_audit":
        stage_sift_audit(args)


if __name__ == "__main__":
    main()
