"""
PAC-certified T-1A theorem audit (Phase 18, T-1A^PAC).

Implements the T-1A theorem:
  With prob >= 1 - delta1 - delta2 over iid audit sample S (split S_A + S_V):
    E_{q~D} |S_H(q,phi) - cert_e(q,phi)|
       <= rho_T1A(N_cell) + delta_A_UB + sqrt(eta_H_UB / N_cell)
  where:
    delta_A_UB = Clopper-Pearson (1-delta1) upper bound on Bernoulli anchor-miss rate (from S_A)
    eta_H_UB = Maurer-Pontil empirical-Bernstein (1-delta2) upper bound on Var_w[p_w] (from S_V)
    V_i in [0, 1/4], so worst-case s_V^2 <= 1/64

Default: delta1 = delta2 = 0.025 (joint 95%).
Sample sizes: n_A = 2500, n_V = 2500 (total M_audit = 5000).
"""
from __future__ import annotations
import argparse, time, json, math
import os
from pathlib import Path
import numpy as np
import faiss
from scipy import stats as sstats

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
def read_u8bin(p):
    with open(p, "rb") as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        n, d = int(hdr[0]), int(hdr[1])
        data = np.frombuffer(f.read(n * d), dtype=np.uint8).reshape(n, d).copy()
    return data


def extract_layer0_neighbors(idx, n_nodes, M):
    """Per-node layer-0 neighbor list."""
    nbrs = faiss.vector_to_array(idx.hnsw.neighbors).astype(np.int64)
    offsets = faiss.vector_to_array(idx.hnsw.offsets).astype(np.int64)
    L0 = 2 * M
    out = np.full((n_nodes, L0), -1, dtype=np.int64)
    for v in range(n_nodes):
        s = offsets[v]
        e = offsets[v + 1] if v + 1 < len(offsets) else len(nbrs)
        region = nbrs[s:e]
        if len(region) >= L0:
            out[v] = region[-L0:]
        else:
            out[v, : len(region)] = region
    return out


def clopper_pearson_upper(k, n, alpha):
    """Clopper-Pearson upper bound on Bernoulli p at (1-alpha) confidence.
    Returns UB such that Pr(p <= UB) >= 1 - alpha."""
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    if k == n:
        return 1.0
    return sstats.beta.ppf(1 - alpha, k + 1, n - k)


def maurer_pontil_eb_upper(v_samples, alpha, v_min=0.0, v_max=0.25):
    """Maurer-Pontil empirical Bernstein 1-alpha upper bound on E[V].
    v_samples bounded in [v_min, v_max]. Returns UB on E[V]."""
    n = len(v_samples)
    if n < 2:
        return v_max
    mean_v = float(np.mean(v_samples))
    var_v = float(np.var(v_samples, ddof=1))  # sample variance
    R = v_max - v_min
    ub = (
        mean_v
        + math.sqrt(2.0 * var_v * math.log(2.0 / alpha) / n)
        + 7.0 * R * math.log(2.0 / alpha) / (3.0 * (n - 1))
    )
    return min(ub, v_max)


def audit_predicate(q_idx, q_vecs, base_vecs, idx, M_layer0, label_arr, predicate_mask,
                    ef_search=200, anchor_top=1):
    """Run T-1A audit for a single query against a single predicate.

    Returns:
      A_i: 1 if anchor missed by HNSW visit, 0 otherwise.
      V_i: variance of p_w over visited HNSW nodes (cell-restricted), in [0, 1/4].
    """
    idx.hnsw.efSearch = ef_search
    q = q_vecs[q_idx:q_idx + 1].astype(np.float32)
    # find true anchor (top-K filtered NN) by brute on predicate-positive subset
    pos_ids = np.where(predicate_mask)[0]
    if len(pos_ids) < 1:
        return None, None
    # Sample-anchor: top-1 of true NN within predicate-positive subset
    sub_d2 = np.sum((base_vecs[pos_ids].astype(np.float32) - q) ** 2, axis=1)
    anchor_id = int(pos_ids[np.argmin(sub_d2)])
    # Did HNSW (unfiltered) visit the anchor? Approximate: anchor in top-ef neighbors
    _, I_visit = idx.search(q, ef_search)
    visited = set(int(x) for x in I_visit[0] if x >= 0)
    anchor_missed = 1 if anchor_id not in visited else 0
    # Variance V_i: over HNSW-layer-0 neighbors of visited nodes, p_w = pos_frac
    visited_arr = np.fromiter(visited, dtype=np.int64)
    if len(visited_arr) < 2:
        return anchor_missed, 0.0
    # for each visited node v, p_v = fraction of layer-0 neighbors satisfying predicate
    p_vs = []
    for v in visited_arr:
        nbrs = M_layer0[v]
        nbrs = nbrs[nbrs >= 0]
        if len(nbrs) == 0:
            continue
        p_v = float(predicate_mask[nbrs].mean())
        p_vs.append(p_v)
    if len(p_vs) < 2:
        return anchor_missed, 0.0
    V_i = float(np.var(p_vs, ddof=1))
    V_i = min(V_i, 0.25)  # clip to [0, 1/4]
    return anchor_missed, V_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yfcc10m", choices=["yfcc10m", "sift10m", "gist1m", "deep10m"])
    ap.add_argument("--n-audit", type=int, default=5000)
    ap.add_argument("--delta1", type=float, default=0.025)
    ap.add_argument("--delta2", type=float, default=0.025)
    ap.add_argument("--ef-search", type=int, default=200)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--rho-t1a", type=float, default=0.0007,
                    help="Pre-measured T-1A ARWGI deviation (default 0.07% from SIFT100M)")
    ap.add_argument("--n-cell", type=int, default=1000,
                    help="Effective cell size for ratio bound")
    args = ap.parse_args()

    # Locate index + data
    paths = {
        "yfcc10m": dict(
            faiss="03_experiment_bridge/results/raw/sanity_t1a/yfcc10m_hnsw_M32_efc200.faiss",
            base="data/raw/yfcc100m/base.10M.u8bin",
            queries="data/raw/yfcc100m/query.public.100K.u8bin",
            labels="data/raw/yfcc100m/base.metadata.10M.spmat",
            q_labels="data/raw/yfcc100m/query.metadata.public.100K.spmat",
        ),
    }
    if args.dataset not in paths:
        print(f"[ERR] dataset {args.dataset} not configured")
        return 1
    pcfg = paths[args.dataset]

    print(f"[load] HNSW: {pcfg['faiss']}", flush=True)
    t0 = time.time()
    idx = faiss.read_index(str(ROOT / pcfg["faiss"]))
    print(f"  loaded in {time.time() - t0:.0f}s, ntotal={idx.ntotal}", flush=True)

    print(f"[load] base.u8bin -> float32", flush=True)
    t0 = time.time()
    base = read_u8bin(ROOT / pcfg["base"]).astype(np.float32)
    print(f"  base shape {base.shape}, {time.time() - t0:.0f}s", flush=True)

    print(f"[load] queries", flush=True)
    queries_all = read_u8bin(ROOT / pcfg["queries"]).astype(np.float32)
    print(f"  queries shape {queries_all.shape}", flush=True)

    # Load YFCC labels (sparse matrix)
    print(f"[load] base labels (spmat)", flush=True)
    def load_spmat(p):
        with open(p, "rb") as f:
            n, m, nnz = np.frombuffer(f.read(24), dtype=np.int64)
            indptr = np.frombuffer(f.read((n + 1) * 8), dtype=np.int64).copy()
            indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32).copy()
        return indptr, indices

    b_indptr, b_indices = load_spmat(ROOT / pcfg["labels"])
    q_indptr, q_indices = load_spmat(ROOT / pcfg["q_labels"])
    print(f"  base labels loaded: nnz={len(b_indices)}", flush=True)

    # Build per-label base membership for fast mask construction
    print(f"[build] label -> base IDs", flush=True)
    t0 = time.time()
    n_labels = int(max(b_indices.max(), q_indices.max())) + 1
    label_to_base = [[] for _ in range(n_labels)]
    for vec_id in range(len(b_indptr) - 1):
        for lbl in b_indices[b_indptr[vec_id]:b_indptr[vec_id + 1]]:
            label_to_base[int(lbl)].append(vec_id)
    for lbl in range(n_labels):
        label_to_base[lbl] = np.array(label_to_base[lbl], dtype=np.int64)
    print(f"  {time.time() - t0:.0f}s", flush=True)

    # Pre-extract layer-0 neighbors
    print(f"[extract] layer-0 neighbors (M={args.M})", flush=True)
    t0 = time.time()
    M_layer0 = extract_layer0_neighbors(idx, idx.ntotal, args.M)
    print(f"  {time.time() - t0:.0f}s, shape {M_layer0.shape}", flush=True)

    # Audit sample: use queries [HOLDOUT_START, +n_audit), held out from any prior measurement
    # YFCC10M has 100K queries; we'll use [85K, 90K) as held-out audit set
    HOLDOUT_START = 85_000
    if HOLDOUT_START + args.n_audit > len(queries_all):
        print(f"[ERR] holdout exceeds query count")
        return 1
    audit_q_ids = list(range(HOLDOUT_START, HOLDOUT_START + args.n_audit))
    # split into S_A and S_V
    rng = np.random.default_rng(42)
    perm = rng.permutation(audit_q_ids)
    n_A = args.n_audit // 2
    S_A = perm[:n_A].tolist()
    S_V = perm[n_A:].tolist()
    print(f"[audit] |S_A|={len(S_A)}, |S_V|={len(S_V)}", flush=True)

    print(f"[run] S_A anchor-miss audit ...", flush=True)
    t0 = time.time()
    A_samples = []
    V_samples_A = []  # also collect V on S_A for cross-validation
    skip = 0
    for i, qi in enumerate(S_A):
        q_labels_i = q_indices[q_indptr[qi]:q_indptr[qi + 1]]
        if len(q_labels_i) == 0:
            skip += 1; continue
        # use first label as predicate
        lbl = int(q_labels_i[0])
        if len(label_to_base[lbl]) < 10:
            skip += 1; continue
        # build mask
        mask = np.zeros(idx.ntotal, dtype=bool)
        mask[label_to_base[lbl]] = True
        A_i, V_i = audit_predicate(qi, queries_all, base, idx, M_layer0, None, mask,
                                    ef_search=args.ef_search)
        if A_i is None:
            skip += 1; continue
        A_samples.append(A_i)
        V_samples_A.append(V_i)
        if (i + 1) % 100 == 0:
            print(f"  S_A {i+1}/{len(S_A)}, elapsed {time.time()-t0:.0f}s, skipped {skip}", flush=True)
    print(f"  S_A done in {time.time() - t0:.0f}s, n_used={len(A_samples)}, skipped={skip}", flush=True)

    print(f"[run] S_V variance audit ...", flush=True)
    t0 = time.time()
    V_samples = []
    skip_V = 0
    for i, qi in enumerate(S_V):
        q_labels_i = q_indices[q_indptr[qi]:q_indptr[qi + 1]]
        if len(q_labels_i) == 0:
            skip_V += 1; continue
        lbl = int(q_labels_i[0])
        if len(label_to_base[lbl]) < 10:
            skip_V += 1; continue
        mask = np.zeros(idx.ntotal, dtype=bool)
        mask[label_to_base[lbl]] = True
        _, V_i = audit_predicate(qi, queries_all, base, idx, M_layer0, None, mask,
                                  ef_search=args.ef_search)
        if V_i is None:
            skip_V += 1; continue
        V_samples.append(V_i)
        if (i + 1) % 100 == 0:
            print(f"  S_V {i+1}/{len(S_V)}, elapsed {time.time()-t0:.0f}s, skipped {skip_V}", flush=True)
    print(f"  S_V done in {time.time() - t0:.0f}s, n_used={len(V_samples)}, skipped={skip_V}", flush=True)

    n_A_used = len(A_samples)
    n_V_used = len(V_samples)
    k_miss = sum(A_samples)
    delta_A_point = k_miss / n_A_used if n_A_used > 0 else 1.0
    delta_A_UB = clopper_pearson_upper(k_miss, n_A_used, args.delta1)
    eta_H_point = float(np.mean(V_samples)) if n_V_used > 0 else 0.25
    eta_H_UB = maurer_pontil_eb_upper(np.array(V_samples), args.delta2)

    # PAC theorem bound
    bound = args.rho_t1a + delta_A_UB + math.sqrt(eta_H_UB / max(args.n_cell, 1))

    print(f"\n=== T-1A^PAC certificate ===")
    print(f"  Dataset: {args.dataset}, HNSW M={args.M}, efSearch={args.ef_search}")
    print(f"  Audit sample size: n_A={n_A_used}, n_V={n_V_used}")
    print(f"  Confidence: delta1={args.delta1}, delta2={args.delta2}, joint = {1-args.delta1-args.delta2:.3f}")
    print(f"  delta_A: point={delta_A_point:.4f}, CP-UB={delta_A_UB:.4f}")
    print(f"  eta_H: point={eta_H_point:.5f}, EB-UB={eta_H_UB:.5f}")
    print(f"  rho_T1A (input): {args.rho_t1a:.4f}")
    print(f"  N_cell (input): {args.n_cell}")
    print(f"  PAC bound: {bound:.4f}")

    out = {
        'dataset': args.dataset, 'M': args.M, 'ef_search': args.ef_search,
        'n_A': n_A_used, 'n_V': n_V_used,
        'delta1': args.delta1, 'delta2': args.delta2,
        'delta_A_point': delta_A_point, 'delta_A_UB': delta_A_UB,
        'eta_H_point': eta_H_point, 'eta_H_UB': eta_H_UB,
        'rho_T1A': args.rho_t1a, 'n_cell': args.n_cell, 'pac_bound': bound,
    }
    outp = ROOT / f"03_experiment_bridge/results/raw/sanity_t1a/pac_t1a_audit_{args.dataset}.json"
    json.dump(out, open(outp, 'w'), indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
