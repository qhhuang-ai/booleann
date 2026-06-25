"""
cert_e bridge: compute expected-cell certificate cert_e = E_v[s_v] alongside the
existing min(...) certificate, for the same (query, predicate) pairs as
full_cert_calibration. Output table for §3 Path B reformulation.

For each (query, predicate):
  - proxy:  in-cell selectivity (existing)
  - p_e:    E_v[p_v]  = unweighted mean over sampled cells (Ass 10)
  - phi_e:  E_v[Phi_v]
  - C/D_max: query-level (existing)
  - s_v_min: min(p_v/p_min, C/D_max, Phi_v) per cell (existing aggregated value)
  - cert_e:  E_v[min(p_v/p_min, C/D_max, Phi_v)]
  - cert_min: min over cells (existing)

Re-uses 8 sampled cells × 4 nodes per cell layout from full_cert_calibration.
"""
from __future__ import annotations
import argparse, time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data/raw/sift"
HNSW_CACHE = ROOT / "03_experiment_bridge/results/raw/sanity_t1a"
KMEANS_CACHE = ROOT / "03_experiment_bridge/results/raw/sanity_td"
RESULTS_DIR = ROOT / "03_experiment_bridge/results/raw/real_recall"

from full_cert_calibration import (
    read_fvecs, extract_layer0_neighbors,
    predicate_independent_equality, predicate_clustered_negative,
    cert_query_correlation,
)


def per_cell_pv_phi(M_layer0, V_sample, mask):
    """Returns per-node p_v(phi) and Phi_v(phi). V_sample is one cell's nodes."""
    if len(V_sample) == 0:
        return np.array([]), np.array([])
    nbrs = M_layer0[V_sample]
    valid = (nbrs >= 0)
    nbrs_safe = np.where(valid, nbrs, 0)
    in_phi = mask[nbrs_safe] & valid
    cnt_phi = in_phi.sum(axis=1)
    cnt_valid = valid.sum(axis=1)
    p_v = np.where(cnt_valid > 0, cnt_phi / cnt_valid, 0.0)
    # Filtered conductance: cross-edges / total edges
    v_in_phi = mask[V_sample]
    cross = np.zeros(len(V_sample))
    for i, v in enumerate(V_sample):
        if not v_in_phi[i]: continue
        cross[i] = (~in_phi[i]).sum() / max(1, cnt_valid[i])
    return p_v, cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--n-predicates-per-sel", type=int, default=4)
    ap.add_argument("--C", type=int, default=16384)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--n-cells-sample", type=int, default=8)
    ap.add_argument("--n-nodes-per-cell", type=int, default=4)
    ap.add_argument("--n-subsamples-C", type=int, default=20)
    ap.add_argument("--p-min-norm", type=float, default=0.05)
    ap.add_argument("--tau-A", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--indep-selectivities", type=float, nargs="+",
                    default=[0.005, 0.016, 0.063, 0.109, 0.262, 1.0])
    ap.add_argument("--neg-selectivities", type=float, nargs="+",
                    default=[0.053, 0.104, 0.203])
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--out", type=str,
                    default="03_experiment_bridge/results/raw/real_recall/cert_e_bridge_sift1m_M32_r024.parquet")
    args = ap.parse_args()

    print("Loading SIFT1M base + queries...")
    base = read_fvecs(DATA_DIR / "sift_base.fvecs").astype(np.float32)
    queries = read_fvecs(DATA_DIR / "sift_query.fvecs").astype(np.float32)
    n = len(base)

    print(f"Loading HNSW M={args.M}...")
    hnsw = faiss.read_index(str(HNSW_CACHE / f"sift1m_hnsw_M{args.M}_efc200.faiss"))
    M_layer0 = extract_layer0_neighbors(hnsw.hnsw_index if hasattr(hnsw, 'hnsw_index') else hnsw, n, args.M)

    print(f"Loading k-means C={args.C}...")
    kmeans = np.load(KMEANS_CACHE / f"sift1m_kmeans_C{args.C}.npz")
    cluster_id = kmeans["cluster_id"]
    centroids = kmeans["centroids"]
    cell_points = [np.where(cluster_id == c)[0] for c in range(args.C)]

    # k-means C=256 for negative-correlation predicates
    kmeans_256 = np.load(KMEANS_CACHE / f"sift1m_kmeans_C256.npz")
    cluster_id_256 = kmeans_256["cluster_id"]

    # Per-query R-nearest cells via Faiss
    index_c = faiss.IndexFlatL2(centroids.shape[1])
    index_c.add(centroids.astype(np.float32))
    _, nearest_R = index_c.search(queries[:args.n_queries].astype(np.float32), args.R)

    # Query cluster sets for neg-correlation predicate generator
    _, query_c256 = faiss.IndexFlatL2(kmeans_256["centroids"].shape[1]).search(
        queries[:args.n_queries].astype(np.float32), 1) if False else (
        None, None)
    # cheap version
    qkm = faiss.IndexFlatL2(kmeans_256["centroids"].shape[1])
    qkm.add(kmeans_256["centroids"].astype(np.float32))
    _, query_cluster_id_256 = qkm.search(queries[:args.n_queries].astype(np.float32), 1)
    query_cluster_set_256 = set(int(c) for c in query_cluster_id_256.ravel())

    label_arr = np.random.default_rng(args.seed).integers(0, 64, size=n).astype(np.int32)
    D_max = float(np.linalg.norm(base.max(0) - base.min(0)))

    predicates = []
    for s in args.indep_selectivities:
        for j in range(args.n_predicates_per_sel):
            seed = args.seed + 100 * j + int(s * 1e6)
            mask = predicate_independent_equality(label_arr, s, seed)
            predicates.append({"family": "indep", "target_s": s, "actual_s": float(mask.mean()),
                               "mask": mask, "id": f"indep_{s}_{j}"})
    for s in args.neg_selectivities:
        for j in range(args.n_predicates_per_sel):
            seed = args.seed + 200 * j + int(s * 1e6)
            mask = predicate_clustered_negative(cluster_id_256, s, query_cluster_set_256, seed)
            predicates.append({"family": "neg", "target_s": s, "actual_s": float(mask.mean()),
                               "mask": mask, "id": f"neg_{s}_{j}"})

    rng_sample = np.random.default_rng(args.seed + 1)

    print(f"Computing cert_e + cert_min for {len(predicates)} predicates × {args.n_queries} queries...")
    rows = []
    t0_total = time.time()
    for pred_i, pred in enumerate(predicates):
        mask = pred["mask"]
        t0 = time.time()
        for qi in range(args.n_queries):
            Rq = nearest_R[qi]
            # Proxy
            union = np.concatenate([cell_points[c] for c in Rq])
            proxy = float(mask[union].mean()) if len(union) > 0 else 0.0
            # Sample cells
            sample_cells = Rq[rng_sample.choice(args.R, size=min(args.n_cells_sample, args.R), replace=False)]
            pv_list, phi_list, sv_list = [], [], []
            C_norm = cert_query_correlation(queries[qi], base, mask, D_max,
                                             n_subsamples=args.n_subsamples_C,
                                             seed=args.seed + qi)
            for c in sample_cells:
                pts = cell_points[c]
                if len(pts) == 0: continue
                take = min(args.n_nodes_per_cell, len(pts))
                V_sample = rng_sample.choice(pts, size=take, replace=False)
                pv, phi = per_cell_pv_phi(M_layer0, V_sample, mask)
                if len(pv) == 0: continue
                pv_cell = float(pv.mean())
                phi_cell = float(phi.mean())
                sv_cell = min(pv_cell / args.p_min_norm, C_norm, phi_cell)
                pv_list.append(pv_cell); phi_list.append(phi_cell); sv_list.append(sv_cell)
            if not sv_list:
                p_e, phi_e, cert_e, cert_min = 0.0, 0.0, 0.0, 0.0
                p_min, phi_min = 0.0, 0.0
            else:
                p_e = float(np.mean(pv_list)); phi_e = float(np.mean(phi_list))
                cert_e = float(np.mean(sv_list))
                cert_min = float(np.min(sv_list))
                p_min = float(np.min(pv_list)); phi_min = float(np.min(phi_list))
            full_cert_min = min(p_min / args.p_min_norm, C_norm, phi_min)
            rows.append({
                "family": pred["family"], "target_s": pred["target_s"],
                "actual_s": pred["actual_s"], "pred_id": pred["id"],
                "query_i": qi,
                "proxy_cert": proxy,
                "p_e": p_e, "p_min": p_min,
                "phi_e": phi_e, "phi_min": phi_min,
                "C_norm": C_norm,
                "cert_e": cert_e, "cert_min": cert_min,
                "full_cert_min": full_cert_min,
            })
        if pred_i % 4 == 0:
            print(f"  pred {pred_i+1}/{len(predicates)} done in {time.time()-t0:.1f}s "
                  f"(total {time.time()-t0_total:.0f}s)")

    df = pd.DataFrame(rows)
    out = ROOT / args.out
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")

    # Bridge statistics: proxy vs cert_e
    print("\n=== Proxy ↔ cert_e bridge ===")
    print(f"Pearson corr proxy ↔ p_e: {df.proxy_cert.corr(df.p_e):.3f}")
    print(f"Pearson corr proxy ↔ cert_e: {df.proxy_cert.corr(df.cert_e):.3f}")
    print(f"Pearson corr proxy ↔ cert_min: {df.proxy_cert.corr(df.cert_min):.3f}")
    print(f"Pearson corr proxy ↔ full_cert_min: {df.proxy_cert.corr(df.full_cert_min):.3f}")
    print(f"Spearman corr proxy ↔ cert_e: {df.proxy_cert.corr(df.cert_e, method='spearman'):.3f}")

    print("\n=== Per family/sel: mean(proxy), mean(cert_e), mean(cert_min) ===")
    g = df.groupby(["family", "target_s"]).agg(
        proxy_mean=("proxy_cert", "mean"),
        cert_e_mean=("cert_e", "mean"),
        cert_min_mean=("cert_min", "mean"),
        full_cert_min_mean=("full_cert_min", "mean"),
    )
    print(g.to_string())


if __name__ == "__main__":
    main()
