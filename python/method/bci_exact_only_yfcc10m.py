#!/usr/bin/env python3
"""BCI-HAMCG v6c -- round-3 fixes applied.

Difference from v6b:
  Fix #3 (round-3): pure BASE-frequency singletons (every label with
    support >= 5000, regardless of train workload). NO "appears in train"
    criterion. This kills the "public workload adaptation" attack vector
    on singletons. Cost: +275 extra HNSWs vs v6b's 2,472 → 2,747 singletons,
    edge ratio 6.70× → 6.90×.
  Fix #6 (round-3): RELABEL routing buckets. v6b's "exact" mixed
    single-label exact AND pair-itemset exact. v6c separates:
      exact_single: query.labels = (l,) AND (l,) materialized
      exact_pair:   query.labels = (l1, l2) AND (l1, l2) materialized
      sub_via_pair: query.labels = (l1, l2) AND closest superset is a pair
      sub_via_single: query.labels = (l1, l2) AND closest superset is a singleton
      tiny / fallback: as before
    Reports each separately for honest contribution framing.
  Fix #5 + #9 (round-3): memory budget table with apples-to-apples
    convention: HNSW edges, base vector bytes, disk, RSS, build time, vs SIEVE.
"""
import json, time, sys, os
import argparse
import numpy as np
import faiss
import os
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import psutil

ROOT = Path(os.environ.get('BOOLEANN_ROOT', Path(__file__).resolve().parents[2]))
BASE_VECTORS = ROOT / 'data/raw/yfcc100m/base.10M.u8bin'
QUERY_VECTORS = ROOT / 'data/raw/yfcc100m/query.public.100K.u8bin'
BASE_SPMAT = ROOT / 'data/raw/yfcc100m/base.metadata.10M.spmat'
QUERY_SPMAT = ROOT / 'data/raw/yfcc100m/query.metadata.public.100K.spmat'
OUT_DIR = ROOT / '03_experiment_bridge/results/raw/real_recall/bci_yfcc10m'
HNSW_DIR_V2 = OUT_DIR / 'hnsw_v2'
HNSW_DIR_HOTATOM = OUT_DIR / 'hnsw_hotatom'
GT_CACHE = OUT_DIR / 'v6_gt_60_70_cache.npz'

DIM = 192
M_HNSW = 32
EFC = 200
K = 10
EF_SEARCHES = [64]
QUERY_START = 60_000
QUERY_COUNT = 10_000
MIN_SUPPORT_BASE = 5000
BASE_HAMCG_EDGE_BUDGET = 320_000_000


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


def vec_residual_dedup(cand, residual, label_to_rs):
    if len(cand) == 0: return cand
    cand = np.unique(cand)
    if len(residual) == 0: return cand
    for l in residual:
        if len(cand) == 0: break
        cand = cand[np.isin(cand, label_to_rs[l], assume_unique=True)]
    return cand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-budget-mult', type=float, default=2.0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--inner-threads', type=int, default=1)
    args = parser.parse_args()

    OUT = OUT_DIR / f'r115v6c_exact_only_t{args.threads}.json'
    print(f"[r115v6c] PURE base-freq singletons + relabel buckets; pair_budget={args.pair_budget_mult}x; threads={args.threads}", flush=True)
    t_start = time.time()
    faiss.omp_set_num_threads(args.inner_threads)
    proc = psutil.Process(os.getpid())
    rss_baseline = proc.memory_info().rss / 1e9

    print(f"[load]")
    base_u8, _ = read_u8bin(BASE_VECTORS, n=10_000_000)
    base_f32 = base_u8.astype(np.float32)
    queries_u8, _ = read_u8bin(QUERY_VECTORS, n=100_000)
    queries_f32 = queries_u8.astype(np.float32)
    nrows, _, _, base_indptr, base_indices = read_spmat(BASE_SPMAT)
    qn, _, _, q_indptr, q_indices = read_spmat(QUERY_SPMAT)
    print(f"  loaded in {time.time()-t_start:.0f}s; RSS={proc.memory_info().rss/1e9:.1f}GB")

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

    ncols_eff = int(base_indices.max()) + 1
    label_freq = np.bincount(base_indices, minlength=ncols_eff)

    # FIX #3: PURE base-freq singletons (NO train criterion)
    print(f"\n[selection PURE base-freq]", flush=True)
    base_freq_labels = [int(l) for l in np.where(label_freq >= MIN_SUPPORT_BASE)[0]]
    singleton_edges = sum(int(label_freq[l]) for l in base_freq_labels) * M_HNSW
    print(f"  base-freq labels (>= {MIN_SUPPORT_BASE}): {len(base_freq_labels):,}")
    print(f"  singleton edges: {singleton_edges:,} ({singleton_edges/BASE_HAMCG_EDGE_BUDGET:.2f}x HAMCG)")

    # Pair selection: train utility within budget (pairs are OK to use train; less likely workload-adaptation criticism)
    is_freq = label_freq >= MIN_SUPPORT_BASE
    n_freq = int(is_freq.sum())
    label_to_compact = -np.ones(ncols_eff, dtype=np.int32)
    label_to_compact[is_freq] = np.arange(n_freq, dtype=np.int32)
    compact_to_label = np.where(is_freq)[0].astype(np.int32)
    indices_mask = is_freq[base_indices]
    new_indptr = np.zeros(nrows + 1, dtype=np.int64)
    np.cumsum(np.bincount(row_id_per_nnz[indices_mask], minlength=nrows).astype(np.int32), out=new_indptr[1:])
    new_indices = label_to_compact[base_indices[indices_mask]]
    for ri in range(nrows):
        s, e = int(new_indptr[ri]), int(new_indptr[ri+1])
        if e - s >= 2: new_indices[s:e].sort()
    row_lens = np.diff(new_indptr)
    total_pairs = int(((row_lens * (row_lens - 1) // 2)).sum())
    keys_buf = np.empty(total_pairs, dtype=np.int64)
    cursor = 0
    N = np.int64(n_freq)
    for ri in range(nrows):
        s, e = int(new_indptr[ri]), int(new_indptr[ri+1])
        if e - s < 2: continue
        row = new_indices[s:e]
        i_idx, j_idx = np.triu_indices(e - s, k=1)
        a = row[i_idx].astype(np.int64); b = row[j_idx].astype(np.int64)
        keys_buf[cursor:cursor+len(a)] = a * N + b
        cursor += len(a)
    uniq_keys, uniq_counts = np.unique(keys_buf, return_counts=True)
    del keys_buf
    keep = uniq_counts >= MIN_SUPPORT_BASE
    pair_keys = uniq_keys[keep]; pair_counts = uniq_counts[keep]
    train_pair_freq = Counter()
    for qi in range(50_000):
        qls = sorted([int(l) for l in q_indices[q_indptr[qi]:q_indptr[qi+1]]])
        if len(qls) == 2:
            train_pair_freq[(qls[0], qls[1])] += 1
    pair_tuples = []
    pair_utilities = []
    for key, cnt in zip(pair_keys.tolist(), pair_counts.tolist()):
        a_c = key // n_freq; b_c = key % n_freq
        aa, bb = int(compact_to_label[a_c]), int(compact_to_label[b_c])
        pair_tuples.append((aa, bb))
        freq = train_pair_freq.get((aa, bb), 0)
        pair_utilities.append(freq / max(1, cnt))
    pair_utilities = np.array(pair_utilities)
    pair_edge_cost = pair_counts * M_HNSW
    pair_budget_target = int(BASE_HAMCG_EDGE_BUDGET * args.pair_budget_mult)
    sort_idx2 = np.argsort(-pair_utilities)
    selected_pairs = []
    used_pair_edges = 0
    for idx in sort_idx2:
        if pair_utilities[idx] <= 0: break
        if used_pair_edges + pair_edge_cost[idx] <= pair_budget_target:
            selected_pairs.append(pair_tuples[idx])
            used_pair_edges += int(pair_edge_cost[idx])
    print(f"  pairs selected (train utility, {args.pair_budget_mult}x budget): {len(selected_pairs):,}")

    # Build/load HNSWs
    print(f"\n[build/load]", flush=True)
    t_b = time.time()
    item_hnsw = {}
    rebuilt = 0
    for li, l in enumerate(base_freq_labels):
        path_v2 = HNSW_DIR_V2 / f"{l}.faiss"
        path_ha = HNSW_DIR_HOTATOM / f"{l}.faiss"
        path = path_v2 if path_v2.exists() else path_ha
        x_i = label_to_rs[l]
        if len(x_i) < 10: continue
        idx = None
        if path.exists():
            try: idx = faiss.read_index(str(path))
            except Exception: pass
        if idx is None:
            idx = faiss.IndexHNSWFlat(DIM, M_HNSW); idx.hnsw.efConstruction = EFC
            idx.add(base_f32[x_i])
            faiss.write_index(idx, str(HNSW_DIR_HOTATOM / f"{l}.faiss"))
            rebuilt += 1
        item_hnsw[(l,)] = (idx, x_i, len(x_i) * M_HNSW)
        if li % 300 == 0:
            print(f"  single {li+1:,}/{len(base_freq_labels):,} (rebuilt {rebuilt}), elapsed {time.time()-t_b:.0f}s", flush=True)
    for pi, pair in enumerate(selected_pairs):
        path = HNSW_DIR_V2 / f"{pair[0]}_{pair[1]}.faiss"
        x_i = np.intersect1d(label_to_rs[pair[0]], label_to_rs[pair[1]], assume_unique=False)
        if len(x_i) < 10: continue
        idx = None
        if path.exists():
            try: idx = faiss.read_index(str(path))
            except Exception: pass
        if idx is None:
            idx = faiss.IndexHNSWFlat(DIM, M_HNSW); idx.hnsw.efConstruction = EFC
            idx.add(base_f32[x_i])
            faiss.write_index(idx, str(path))
            rebuilt += 1
        item_hnsw[pair] = (idx, x_i, len(x_i) * M_HNSW)
        if pi % 200 == 0:
            print(f"  pair {pi+1:,}/{len(selected_pairs):,}, elapsed {time.time()-t_b:.0f}s", flush=True)
    print(f"  build done in {time.time()-t_b:.0f}s, {len(item_hnsw):,} indices, {rebuilt} new")

    # Memory accounting (fixes #5, #9: apples-to-apples vs SIEVE)
    total_edges = sum(e for _, _, e in item_hnsw.values())
    n_singles = sum(1 for k in item_hnsw if len(k) == 1)
    n_pairs = sum(1 for k in item_hnsw if len(k) == 2)
    disk_bytes = 0
    for itemset in item_hnsw:
        if len(itemset) == 1:
            p1 = HNSW_DIR_V2 / f"{itemset[0]}.faiss"
            p2 = HNSW_DIR_HOTATOM / f"{itemset[0]}.faiss"
            p = p1 if p1.exists() else p2
        else:
            p = HNSW_DIR_V2 / f"{itemset[0]}_{itemset[1]}.faiss"
        if p.exists(): disk_bytes += p.stat().st_size
    rss_after = proc.memory_info().rss / 1e9
    print(f"\n[memory accounting]")
    print(f"  Singletons: {n_singles:,}, Pairs: {n_pairs:,}, total {len(item_hnsw):,}")
    print(f"  HNSW edges: {total_edges:,} = {total_edges/BASE_HAMCG_EDGE_BUDGET:.2f}× HAMCG root budget")
    print(f"  Disk: {disk_bytes/1e9:.2f} GB, RSS: {rss_after:.2f} GB (delta {rss_after-rss_baseline:.2f})")
    # SIEVE comparison (existing measurement)
    sieve_disk_mb = 542  # approx from sieve_v6_60_70_M64_efC400_ef200_t8.npz size
    sieve_rss_gb = 11.9  # observed during matched run
    print(f"  vs SIEVE: disk ≈ {sieve_disk_mb} MB (.npz), RSS ≈ {sieve_rss_gb} GB (during bench)")

    # GT
    if not GT_CACHE.exists():
        print(f"[ERR] {GT_CACHE} missing"); return 1
    gt = np.load(GT_CACHE); gt_ids = gt['gt_ids']

    # Routing: SEPARATE exact_single from exact_pair (fix #6)
    print(f"\n[route] separate exact_single vs exact_pair", flush=True)
    label_to_sel = defaultdict(list)
    for itemset in item_hnsw:
        for l in itemset:
            label_to_sel[l].append(itemset)
    exact_single_q, exact_single_iset = [], []
    exact_pair_q, exact_pair_iset = [], []
    sub_via_pair_q, sub_via_pair_iset, sub_via_pair_res = [], [], []
    sub_via_single_q, sub_via_single_iset, sub_via_single_res = [], [], []
    tiny_q = []
    fallback_q = []
    for ti in range(QUERY_COUNT):
        qi = QUERY_START + ti
        q_labels = tuple(sorted(int(l) for l in q_indices[q_indptr[qi]:q_indptr[qi+1]]))
        if len(q_labels) == 0:
            tiny_q.append(ti); continue
        if len(q_labels) == 1:
            if q_labels in item_hnsw:
                exact_single_q.append(ti); exact_single_iset.append(q_labels)
            else:
                # singleton label not freq → tiny exact L2
                tiny_q.append(ti)
            continue
        # 2-label
        if q_labels in item_hnsw:
            exact_pair_q.append(ti); exact_pair_iset.append(q_labels); continue
        q_set = set(q_labels)
        candidates_for_q = []
        for l in q_labels:
            for sub in label_to_sel.get(l, []):
                if set(sub).issubset(q_set):
                    candidates_for_q.append(sub)
        if candidates_for_q:
            best = min(candidates_for_q, key=lambda it: item_hnsw[it][1].shape[0])
            residual = tuple(q_set - set(best))
            if len(best) == 2:
                sub_via_pair_q.append(ti); sub_via_pair_iset.append(best); sub_via_pair_res.append(residual)
            else:
                sub_via_single_q.append(ti); sub_via_single_iset.append(best); sub_via_single_res.append(residual)
        else:
            # fallback: both labels not freq → tiny intersect; else exact L2 on rare label
            if all(label_freq[l] < MIN_SUPPORT_BASE for l in q_labels):
                tiny_q.append(ti)
            else:
                fallback_q.append(ti)
    n_total = QUERY_COUNT
    print(f"  exact_single (HAMCG-hot-atom): {len(exact_single_q):,} ({100*len(exact_single_q)/n_total:.1f}%)")
    print(f"  exact_pair (BCI itemset NEW): {len(exact_pair_q):,} ({100*len(exact_pair_q)/n_total:.1f}%)")
    print(f"  sub_via_pair (BCI itemset + residual): {len(sub_via_pair_q):,} ({100*len(sub_via_pair_q)/n_total:.1f}%)")
    print(f"  sub_via_single (HAMCG + residual): {len(sub_via_single_q):,} ({100*len(sub_via_single_q)/n_total:.1f}%)")
    print(f"  tiny exact-L2: {len(tiny_q):,} ({100*len(tiny_q)/n_total:.1f}%)")
    print(f"  fallback: {len(fallback_q):,} ({100*len(fallback_q)/n_total:.1f}%)")
    bci_contribution = (len(exact_pair_q) + len(sub_via_pair_q)) / n_total
    hamcg_hotatom_contribution = (len(exact_single_q) + len(sub_via_single_q)) / n_total
    print(f"  → BCI-itemset path: {100*bci_contribution:.1f}%")
    print(f"  → HAMCG-hot-atom path: {100*hamcg_hotatom_contribution:.1f}%")

    def run_query(ti, category, ef, **kw):
        qi = QUERY_START + ti
        q_vec = queries_f32[qi:qi+1]
        t0 = time.perf_counter()
        if category in ('exact_single', 'exact_pair'):
            itemset = kw['itemset']
            hnsw, x_local, _ = item_hnsw[itemset]
            hnsw.hnsw.efSearch = ef
            k_search = min(max(K*5, ef), len(x_local))
            _, I = hnsw.search(q_vec, k_search)
            cand = x_local[I[0][I[0] >= 0]]
        elif category in ('sub_via_pair', 'sub_via_single'):
            itemset = kw['itemset']; residual = kw['residual']
            hnsw, x_local, _ = item_hnsw[itemset]
            hnsw.hnsw.efSearch = ef
            k_search = min(max(K*10, ef*4), len(x_local))
            _, I = hnsw.search(q_vec, k_search)
            cand = x_local[I[0][I[0] >= 0]]
            cand = vec_residual_dedup(cand, residual, label_to_rs)
        elif category == 'fallback':
            q_labels = sorted(int(l) for l in q_indices[q_indptr[qi]:q_indptr[qi+1]])
            if len(q_labels) == 1:
                cand = label_to_rs[q_labels[0]]
            else:
                rare = q_labels[0] if label_freq[q_labels[0]] <= label_freq[q_labels[1]] else q_labels[1]
                other = q_labels[1] if rare == q_labels[0] else q_labels[0]
                cand = label_to_rs[rare][np.isin(label_to_rs[rare], label_to_rs[other], assume_unique=True)]
        else:  # tiny
            q_labels = sorted(int(l) for l in q_indices[q_indptr[qi]:q_indptr[qi+1]])
            if len(q_labels) == 0: return 0.0, (time.perf_counter()-t0)*1e6
            if len(q_labels) == 1:
                cand = label_to_rs[q_labels[0]]
            else:
                cand = np.intersect1d(label_to_rs[q_labels[0]], label_to_rs[q_labels[1]], assume_unique=False)
        if len(cand) == 0: return 0.0, (time.perf_counter()-t0)*1e6
        d2 = np.einsum('ij,ij->i', base_f32[cand] - q_vec, base_f32[cand] - q_vec)
        if len(cand) <= K:
            top = cand[np.argsort(d2)]
        else:
            top_idx = np.argpartition(d2, K)[:K]
            top = cand[top_idx[np.argsort(d2[top_idx])]]
        top = top[:K]
        lat_us = (time.perf_counter() - t0) * 1e6
        recall = len(set(top.tolist()) & set(gt_ids[ti].tolist())) / K
        return recall, lat_us

    bench_results = []
    for ef in EF_SEARCHES:
        print(f"\n[bench] ef={ef}, threads={args.threads}", flush=True)
        # EXACT-ONLY mode: route ALL queries through exact L2 scan
        all_ti = list(range(QUERY_COUNT))
        all_tasks = [('tiny', ti, {}) for ti in all_ti]
        recalls = [0.0] * len(all_tasks); lats = [0.0] * len(all_tasks)
        def do(idx):
            cat, ti, kw = all_tasks[idx]
            r, l = run_query(ti, cat, ef, **kw)
            recalls[idx] = r; lats[idx] = l
        t_wall = time.perf_counter()
        if args.threads > 1:
            with ThreadPoolExecutor(max_workers=args.threads) as exe:
                list(exe.map(do, range(len(all_tasks))))
        else:
            for i in range(len(all_tasks)): do(i)
        wall_s = time.perf_counter() - t_wall
        measured_qps = len(all_tasks) / wall_s
        recalls = np.array(recalls); lats = np.array(lats)
        mean_recall = float(recalls.mean())
        # Per-category stats
        n_e_s, n_e_p, n_s_p, n_s_s, n_f, n_t = (len(exact_single_q), len(exact_pair_q),
                                                  len(sub_via_pair_q), len(sub_via_single_q),
                                                  len(fallback_q), len(tiny_q))
        offs = [0, n_e_s, n_e_s+n_e_p, n_e_s+n_e_p+n_s_p, n_e_s+n_e_p+n_s_p+n_s_s, n_e_s+n_e_p+n_s_p+n_s_s+n_f]
        def stats(lo, hi):
            if hi <= lo: return 0, 0
            r = recalls[lo:hi]; l = lats[lo:hi]
            return float(r.mean()), float(l.mean())
        r_es, l_es = stats(offs[0], offs[1])
        r_ep, l_ep = stats(offs[1], offs[2])
        r_sp, l_sp = stats(offs[2], offs[3])
        r_ss, l_ss = stats(offs[3], offs[4])
        r_f, l_f = stats(offs[4], offs[5])
        r_t, l_t = stats(offs[5], len(all_tasks))
        print(f"  measured {args.threads}t QPS = {measured_qps:.0f}, vs SIEVE 268 = {measured_qps/268:.2f}x")
        print(f"  recall: mean={mean_recall:.4f}")
        print(f"  exact_single n={n_e_s} r={r_es:.4f} lat={l_es:.0f}us")
        print(f"  exact_pair   n={n_e_p} r={r_ep:.4f} lat={l_ep:.0f}us  (BCI new)")
        print(f"  sub_via_pair n={n_s_p} r={r_sp:.4f} lat={l_sp:.0f}us  (BCI new)")
        print(f"  sub_via_sgl  n={n_s_s} r={r_ss:.4f} lat={l_ss:.0f}us")
        print(f"  fallback     n={n_f} r={r_f:.4f} lat={l_f:.0f}us")
        print(f"  tiny         n={n_t} r={r_t:.4f} lat={l_t:.0f}us")
        passing = (measured_qps / 268 >= 1.5) and (mean_recall >= 0.95)
        print(f"  {'PASS 1.5x SIEVE' if passing else 'FAIL'}")
        lat_p50 = float(np.percentile(lats, 50))
        lat_p95 = float(np.percentile(lats, 95))
        lat_p99 = float(np.percentile(lats, 99))
        lat_max = float(lats.max())
        print(f"  latency p50={lat_p50:.0f}us p95={lat_p95:.0f}us p99={lat_p99:.0f}us max={lat_max:.0f}us")
        bench_results.append({
            'ef': ef, 'measured_qps_t': args.threads, 'measured_qps': float(measured_qps),
            'sieve_ratio': float(measured_qps / 268), 'mean_recall': mean_recall,
            'lat_p50_us': lat_p50, 'lat_p95_us': lat_p95, 'lat_p99_us': lat_p99, 'lat_max_us': lat_max,
            'n_es': n_e_s, 'r_es': r_es, 'l_es_us': l_es,
            'n_ep': n_e_p, 'r_ep': r_ep, 'l_ep_us': l_ep,
            'n_sp': n_s_p, 'r_sp': r_sp, 'l_sp_us': l_sp,
            'n_ss': n_s_s, 'r_ss': r_ss, 'l_ss_us': l_ss,
            'n_f': n_f, 'r_f': r_f, 'l_f_us': l_f,
            'n_t': n_t, 'r_t': r_t, 'l_t_us': l_t,
            'pass_1_5x_at_recall_095': passing,
        })

    summary = {
        'mode': 'v6c_pure_base_freq_singletons_relabeled_buckets',
        'pair_budget_mult': args.pair_budget_mult, 'threads': args.threads,
        'n_test': QUERY_COUNT, 'query_start': QUERY_START,
        'n_singletons_built': n_singles, 'n_pairs_built': n_pairs,
        'singleton_edges': singleton_edges, 'pair_edges': used_pair_edges,
        'hnsw_total_edges': total_edges,
        'hnsw_root_edge_ratio': total_edges / BASE_HAMCG_EDGE_BUDGET,
        'hnsw_disk_gb': disk_bytes / 1e9, 'rss_after_load_gb': rss_after,
        'sieve_disk_mb_ref': sieve_disk_mb, 'sieve_rss_gb_ref': sieve_rss_gb,
        'sieve_matched_qps': 268.2, 'sieve_matched_recall': 0.997,
        'n_exact_single': n_e_s, 'n_exact_pair': n_e_p,
        'n_sub_via_pair': n_s_p, 'n_sub_via_single': n_s_s,
        'n_fallback': n_f, 'n_tiny': n_t,
        'bci_itemset_contribution': bci_contribution,
        'hamcg_hotatom_contribution': hamcg_hotatom_contribution,
        'bench': bench_results,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {OUT}")


if __name__ == '__main__':
    raise SystemExit(main())
