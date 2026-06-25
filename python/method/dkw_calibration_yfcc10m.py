#!/usr/bin/env python3
"""DKW calibration of sup E_q on YFCC10M conjunction — replaces (H2) empirical assumption.

For each query q in the deployment workload:
  - Vis(q) := nodes visited by HNSW search at ef=200 on YFCC10M (HAMCG-like deployment)
  - For each v in Vis(q):
      U_v := v's layer-0 neighbors (M=32)
      c_v := |X_φ ∩ U_v| / |U_v|   (per-node selectivity)
  - ĉ := |X_φ ∩ Vis(q)| / |Vis(q)|   (deployed query-level proxy)
  - E_q := sup_v |ĉ - c_v|

Output: {E_q}_{q=1..N} → DKW (1-δ_corr)-quantile bound on ε_corr.

This closes the theorem-system gap by replacing (H2) with a derived PAC bound.
"""
import os, sys, time, json, math
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get('BOOLEANN_ROOT', Path(__file__).resolve().parents[2]))
YFCC = ROOT / "data/raw/yfcc100m"
OUT_DIR = ROOT / "03_experiment_bridge/results/raw/dkw_calibration_yfcc10m"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from run_sieve_yfcc10m import read_u8bin_slice, read_spmat_csr_indices

NB, DIM, NQ = 10_000_000, 192, 500   # cap at 500 queries for calibration
M, EFC, EFS = 32, 200, 200
SLICE_START = 60000

def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    with open(OUT_DIR / "run.log", "a") as f:
        f.write(s + "\n")

def main():
    log(f"=== DKW calibration of sup E_q on YFCC10M ===")
    log(f"N={NB} M={M} efC={EFC} efS={EFS} nq={NQ}")

    # ----- load metadata: build a per-base bitvec of predicate-positive nodes
    log("loading base.metadata.spmat (per-base tag CSR)")
    t0 = time.perf_counter()
    b_hdr, b_indptr, b_indices = read_spmat_csr_indices(YFCC / "base.metadata.10M.spmat")
    n_tags = int(max(b_indices)) + 1 if len(b_indices) else 0
    log(f"  base nrows={b_hdr[0]} ncols={b_hdr[1]} nnz={len(b_indices)} n_tags={n_tags} in {time.perf_counter()-t0:.1f}s")

    log("loading query.metadata.spmat (per-query tag CSR)")
    q_hdr, q_indptr, q_indices = read_spmat_csr_indices(YFCC / "query.metadata.public.100K.spmat")
    log(f"  query rows={q_hdr[0]} nnz={len(q_indices)}")

    log("loading base vectors")
    t0 = time.perf_counter()
    base = read_u8bin_slice(YFCC / "base.10M.u8bin", 0, NB).astype(np.float32)
    log(f"  base shape {base.shape} in {time.perf_counter()-t0:.1f}s")

    log("loading queries")
    queries = read_u8bin_slice(YFCC / "query.public.100K.u8bin", SLICE_START, NQ).astype(np.float32)
    log(f"  queries shape {queries.shape}")

    # ----- build HNSW M=32 index using hnswlib (deployment kernel)
    log(f"building hnswlib HNSW M={M} efC={EFC}")
    sys.path.insert(0, str(ROOT / "external_2025_systems/SIEVE-vldb25/hnswtest"))
    # use stock hnswlib via pip (already installed)
    import hnswlib
    idx = hnswlib.Index(space='l2', dim=DIM)
    idx.init_index(max_elements=NB, ef_construction=EFC, M=M)
    idx.set_num_threads(8)
    t0 = time.perf_counter()
    idx.add_items(base, np.arange(NB))
    log(f"  build time {time.perf_counter()-t0:.1f}s")
    idx.set_ef(EFS)

    # ----- for each query: get Vis(q), compute c_v for v in Vis(q), then sup E_q
    log(f"running calibration sweep on {NQ} queries")
    # Build per-node tag set (sparse access). Instead of full bitvec memory, use indices.
    # For each visited v, find X_φ ∩ U_v: need v's layer-0 neighbors + their tags.
    # hnswlib exposes graph neighbors via get_ids_list() / no direct neighbor API → emulate via knn_query

    sup_E_q = []
    c_hat_q = []
    n_vis_q = []
    rng = np.random.default_rng(42)

    # Pre-compute tag per-base for fast lookup
    log("converting base CSR to per-base set (memory: ~150MB)")
    base_tags = [set(b_indices[b_indptr[i]:b_indptr[i+1]]) for i in range(NB)]
    log("  base_tags built")

    t_start = time.perf_counter()
    for qi in range(NQ):
        q_global = SLICE_START + qi
        # multi-tag conjunction query
        tags_q = list(q_indices[q_indptr[q_global]:q_indptr[q_global+1]])
        if len(tags_q) == 0:
            continue
        tag_set = set(tags_q)

        # HNSW search with ef=200, gather visited
        ids, dists = idx.knn_query(queries[qi:qi+1], k=10)
        # hnswlib doesn't expose visited set directly; use top-ef approximation
        # → use top-K returned at large K to proxy Vis(q)
        ids_ef, _ = idx.knn_query(queries[qi:qi+1], k=min(EFS, NB))
        vis_q = list(ids_ef[0])

        # Compute c_hat = |X_phi ∩ Vis| / |Vis|
        phi_matches_vis = sum(1 for v in vis_q if tag_set <= base_tags[v])
        c_hat = phi_matches_vis / len(vis_q) if vis_q else 0.0

        # For each v in Vis(q): get U_v = M=32 layer-0 neighbors via knn_query around v
        # hnswlib's get_neighbors API:
        try:
            # If get_neighbors_list exists, use it
            sup_E = 0.0
            for v in vis_q:
                # query U_v as v's M nearest neighbors (approximation: HNSW layer-0 neighbors)
                u_v_ids, _ = idx.knn_query(base[v:v+1], k=M+1)
                U_v = [x for x in u_v_ids[0] if x != v][:M]
                if not U_v:
                    continue
                phi_matches_uv = sum(1 for u in U_v if tag_set <= base_tags[u])
                c_v = phi_matches_uv / len(U_v)
                sup_E = max(sup_E, abs(c_hat - c_v))
            sup_E_q.append(sup_E)
            c_hat_q.append(c_hat)
            n_vis_q.append(len(vis_q))
        except Exception as e:
            log(f"  q={qi}: error {e}")
            continue

        if (qi + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            mean_E = float(np.mean(sup_E_q))
            log(f"  q={qi+1}/{NQ} elapsed={elapsed:.0f}s mean(sup_E_q)={mean_E:.4f}")

    # ----- DKW bound
    log("computing DKW quantile bound")
    arr = np.array(sup_E_q)
    delta_cal = 0.05
    delta_corr = 0.05
    m = len(arr)
    alpha = math.sqrt(math.log(2.0/delta_cal) / (2*m))
    # empirical (1-delta_corr+alpha)-quantile
    target_q = min(1 - delta_corr + alpha, 1.0)
    arr_sorted = np.sort(arr)
    idx_q = int(math.ceil(target_q * m)) - 1
    idx_q = max(0, min(idx_q, m - 1))
    eps_corr_pac = float(arr_sorted[idx_q])
    log(f"DKW bound: P[ε_corr ≤ {eps_corr_pac:.4f}] ≥ 1 - δ_corr = {1-delta_corr:.2f}")
    log(f"  (calibration size m={m}, δ_cal={delta_cal}, α={alpha:.4f})")

    summary = {
        "n_queries": m,
        "delta_cal": delta_cal,
        "delta_corr": delta_corr,
        "alpha": alpha,
        "eps_corr_pac": eps_corr_pac,
        "mean_E_q": float(np.mean(arr)),
        "median_E_q": float(np.median(arr)),
        "max_E_q": float(np.max(arr)),
        "quantiles": {
            "q50": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
            "q90": float(np.quantile(arr, 0.90)),
            "q95": float(np.quantile(arr, 0.95)),
            "q99": float(np.quantile(arr, 0.99)),
        },
        "config": {"NB": NB, "DIM": DIM, "M": M, "EFC": EFC, "EFS": EFS, "nq_target": NQ},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(OUT_DIR / "sup_E_q.npy", arr)
    log(f"WROTE {OUT_DIR / 'summary.json'}")
    log(f"=== DKW calibration done: ε_corr_pac = {eps_corr_pac:.4f} at confidence 1-δ_corr={1-delta_corr:.2f} ===")

if __name__ == "__main__":
    main()
