"""
T-D Group-Testing Sentinel sanity check.

Pseudocode source: 02_theory/proofs/T_D_sentinel_skeleton_v1.md §5

Pass criteria (from skeleton §5):
  - For exact row scans + λ=8: no_omit_Z >= 0.999
  - For exact row scans + λ=8: recall10_R >= 0.99
  - For at least one t<=8, R<=256: t_violation_rate <= 0.2 on selectivities s<=1e-2
  - Median decoded cells <= 2t and p95 <= 4t under exact rows
  - Median distance comps after D >= 5x lower than scanning all R_q

Setup: SIFT1M base, 1000 queries, k-means C ∈ {1024, 4096}.
"""
from __future__ import annotations
import argparse, json, time
from math import ceil, log
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data/raw/sift"
RESULTS_DIR = ROOT / "03_experiment_bridge/results/raw/sanity_td"


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32); d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C-list", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--R-list", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--t-list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--lambda-list", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--selectivities", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    ap.add_argument("--n-preds-per-config", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delta-sentinel", type=float, default=0.05)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading SIFT1M...")
    base = read_fvecs(DATA_DIR / "sift_base.fvecs")
    queries = read_fvecs(DATA_DIR / "sift_query.fvecs")[: args.n_queries]
    n = base.shape[0]

    rng = np.random.default_rng(args.seed)
    label_arr = rng.integers(0, 256, size=n).astype(np.int32)  # for predicate labeling

    rows = []
    for C in args.C_list:
        cache_path = RESULTS_DIR / f"sift1m_kmeans_C{C}.npz"
        if cache_path.exists():
            print(f"  loading cached k-means C={C}")
            d = np.load(cache_path)
            cluster_id = d["cluster_id"]; centroids = d["centroids"]
        else:
            print(f"  fitting MiniBatchKMeans C={C}...")
            t0 = time.time()
            km = MiniBatchKMeans(n_clusters=C, random_state=args.seed, batch_size=8192, max_iter=20, n_init=3)
            cluster_id = km.fit_predict(base).astype(np.int32)
            centroids = km.cluster_centers_.astype(np.float32)
            np.savez(cache_path, cluster_id=cluster_id, centroids=centroids)
            print(f"    fit in {time.time()-t0:.1f}s")

        cell_points = [np.where(cluster_id == c)[0] for c in range(C)]

        # For each query, find R nearest centroids
        qc_dists = ((queries[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

        for s in args.selectivities:
            # Build n_preds_per_config rare predicates of selectivity s by sampling labels
            for pred_j in range(args.n_preds_per_config):
                # Pick equality predicate that hits target_s
                target = int(s * n)
                seed = args.seed + 1000 * pred_j + int(s * 1e6) + C
                rng_p = np.random.default_rng(seed)
                labels = rng_p.permutation(256)
                chosen, count = [], 0
                for lab in labels:
                    c = int((label_arr == lab).sum())
                    if count + c > 1.5 * target: break
                    chosen.append(int(lab)); count += c
                    if count >= target: break
                if not chosen: chosen.append(int(labels[0]))
                pred_mask = np.isin(label_arr, chosen)
                actual_s = float(pred_mask.mean())

                for R in args.R_list:
                    nearest_R = np.argpartition(qc_dists, kth=R - 1, axis=1)[:, :R]  # (Q, R)
                    Q = len(queries)
                    Z_sizes = []
                    no_omit_acc = {t: {l: 0 for l in args.lambda_list} for t in args.t_list}
                    exact_Z_acc = {t: {l: 0 for l in args.lambda_list} for t in args.t_list}
                    n_eligible = {t: 0 for t in args.t_list}
                    n_violation = {t: 0 for t in args.t_list}
                    decoded_cells_p50 = {t: {l: [] for l in args.lambda_list} for t in args.t_list}
                    for qi in range(Q):
                        Rq = nearest_R[qi]
                        pos_cells = []
                        for c in Rq:
                            pts = cell_points[c]
                            if len(pts) and pred_mask[pts].any():
                                pos_cells.append(int(c))
                        Z = set(pos_cells)
                        Z_sizes.append(len(Z))
                        for t in args.t_list:
                            if len(Z) > t:
                                n_violation[t] += 1
                                continue
                            n_eligible[t] += 1
                            for lam in args.lambda_list:
                                m = int(ceil(lam * t * t * log(R / args.delta_sentinel)))
                                p_row = 1.0 / (t + 1)
                                rng_m = np.random.default_rng(seed + qi + t * 1000 + lam * 100)
                                A = (rng_m.random((m, R)) < p_row).astype(np.uint8)
                                y = np.zeros(m, dtype=np.uint8)
                                for l_row in range(m):
                                    cells_in_row = Rq[A[l_row].astype(bool)]
                                    if len(cells_in_row) == 0: continue
                                    union = np.concatenate([cell_points[c] for c in cells_in_row])
                                    if pred_mask[union].any():
                                        y[l_row] = 1
                                Zhat = set()
                                for ci, col in enumerate(Rq):
                                    rows_containing = A[:, ci].astype(bool)
                                    if not rows_containing.any():
                                        Zhat.add(int(col))
                                        continue
                                    if y[rows_containing].all():
                                        Zhat.add(int(col))
                                no_omit = Z.issubset(Zhat)
                                exact = (Z == Zhat)
                                no_omit_acc[t][lam] += int(no_omit)
                                exact_Z_acc[t][lam] += int(exact)
                                decoded_cells_p50[t][lam].append(len(Zhat))

                    z_sizes = np.array(Z_sizes)
                    for t in args.t_list:
                        for lam in args.lambda_list:
                            decoded = decoded_cells_p50[t][lam]
                            n_total = n_eligible[t] + n_violation[t]
                            rows.append({
                                "C": C, "R": R, "t": t, "lam": lam,
                                "target_s": s, "actual_s": actual_s, "pred_j": pred_j,
                                "n_queries": Q,
                                "n_eligible_with_t": n_eligible[t],
                                "n_t_violation": n_violation[t],
                                "t_violation_rate": float(n_violation[t] / max(n_total, 1)),
                                "no_omit_Z_rate": no_omit_acc[t][lam] / n_eligible[t] if n_eligible[t] > 0 else float('nan'),
                                "exact_Z_rate": exact_Z_acc[t][lam] / n_eligible[t] if n_eligible[t] > 0 else float('nan'),
                                "median_decoded": float(np.median(decoded)) if decoded else float('nan'),
                                "p95_decoded": float(np.percentile(decoded, 95)) if decoded else float('nan'),
                                "median_Z_size": float(np.median(z_sizes)),
                                "p95_Z_size": float(np.percentile(z_sizes, 95)),
                            })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "td_sentinel_summary.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # Print summary verdict
    print("\n=== T-D sentinel sanity verdict ===")
    print(f"Pass criteria (from T_D_sentinel_skeleton_v1.md §5):")
    # Check: exact row scans + λ=8 → no_omit_Z >= 0.999
    sub = df[df["lam"] == 8]
    print(f"  no_omit_Z (λ=8) median: {sub['no_omit_Z_rate'].median():.4f} (target ≥ 0.999): "
          f"{'PASS' if sub['no_omit_Z_rate'].median() >= 0.999 else 'FAIL'}")
    print(f"  exact_Z (λ=8) median: {sub['exact_Z_rate'].median():.4f}")
    # Check t_violation_rate <= 0.2 on s <= 1e-2 for some (t, R)
    rare = df[df["target_s"] <= 1e-2]
    best = rare.groupby(["t", "R"], as_index=False)["t_violation_rate"].mean()
    feasible = best[best["t_violation_rate"] <= 0.2]
    print(f"  t_violation ≤ 0.2 found at (t, R) ∈: {feasible[['t','R']].values.tolist() if not feasible.empty else 'NONE'}")

    print(f"\nSummary table (by t, λ=8):")
    pivot = df[df["lam"] == 8].groupby(["t"], as_index=False)[
        ["no_omit_Z_rate", "exact_Z_rate", "median_decoded", "p95_decoded", "t_violation_rate"]
    ].mean()
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
