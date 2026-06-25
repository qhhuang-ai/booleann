#!/usr/bin/env python3
"""BCI-HAMCG on LAION-1M with RICHER caption-derived filters.

Variant of bci_hamcg_laion1m_real.py pointing at *.richer.spmat / gt.10K.richer.bin.

Changes vs the top-200 unigram version:
  * MIN_SUPPORT_BASE lowered to 200 (richer vocab supports more labels).
  * Vocabulary is ~2200 labels (uni+bi); singleton/pair budget adjusted.
  * Baseline HNSW + post-filter is benchmarked in the same run for direct
    apples-to-apples comparison at matched recall.

Outputs:
  03_experiment_bridge/results/raw/real_recall/bci_laion1m/
    r115v6c_laion1m_RICHER_summary_pb{X}x_t{T}.json
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

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
BASE_VECTORS = ROOT / 'data/raw/laion1m/base.1M.f32bin'
QUERY_VECTORS = ROOT / 'data/raw/laion1m/query.10K.f32bin'
BASE_SPMAT = ROOT / 'data/raw/laion1m/base.metadata.laion1m.richer.spmat'
QUERY_SPMAT = ROOT / 'data/raw/laion1m/query.metadata.laion1m.richer.spmat'
GT_PATH = ROOT / 'data/raw/laion1m/gt.10K.richer.bin'
OUT_DIR = ROOT / '03_experiment_bridge/results/raw/real_recall/bci_laion1m'
HNSW_DIR = OUT_DIR / 'hnsw_richer'
BASELINE_DIR = OUT_DIR / 'baseline'  # reuse synthetic-filter baseline HNSW (filter-agnostic)
HNSW_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

DIM = 512
M_HNSW = 32
EFC = 200
K = 10
EF_SEARCHES = [16, 32, 64, 128, 256, 512, 1024]
QUERY_START = 0
QUERY_COUNT = 10_000
MIN_SUPPORT_BASE = 200
BASE_HAMCG_EDGE_BUDGET = 32_000_000


def read_f32bin(path, n=None):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        total_n, d = int(hdr[0]), int(hdr[1])
        if n is None or n > total_n: n = total_n
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy(), d


def read_spmat(path):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(24), dtype=np.int64)
        nrows = int(hdr[0]); ncols = int(hdr[1]); nnz = int(hdr[2])
        indptr = np.frombuffer(f.read((nrows+1)*8), dtype=np.int64).copy()
        indices = np.frombuffer(f.read(nnz*4), dtype=np.int32).copy()
    return nrows, ncols, nnz, indptr, indices


def read_gt(path):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        n, k = int(hdr[0]), int(hdr[1])
        ids = np.frombuffer(f.read(n*k*4), dtype=np.int32).reshape(n, k).copy()
    return ids


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
    parser.add_argument('--build-threads', type=int, default=16)
    parser.add_argument('--mode', choices=['bci', 'baseline', 'both'], default='both')
    args = parser.parse_args()

    OUT = OUT_DIR / f'r115v6c_laion1m_RICHER_summary_pb{args.pair_budget_mult:g}x_t{args.threads}.json'
    print(f"[BCI-LAION1M-RICHER] pair_budget={args.pair_budget_mult}x; threads={args.threads}; mode={args.mode}", flush=True)
    t_start = time.time()
    faiss.omp_set_num_threads(args.build_threads)
    proc = psutil.Process(os.getpid())

    print(f"[load]", flush=True)
    base_f32, _ = read_f32bin(BASE_VECTORS, n=1_000_000)
    queries_f32, _ = read_f32bin(QUERY_VECTORS, n=10_000)
    nrows, ncols, _, base_indptr, base_indices = read_spmat(BASE_SPMAT)
    qn, _, _, q_indptr, q_indices = read_spmat(QUERY_SPMAT)
    gt_ids = read_gt(GT_PATH)
    print(f"  base={base_f32.shape}, queries={queries_f32.shape}, "
          f"base_spmat: nrows={nrows} ncols={ncols} nnz={len(base_indices)}, gt={gt_ids.shape}", flush=True)

    # Inverted index
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
    print(f"  label_to_rs in {time.time()-t_inv:.0f}s, {len(label_to_rs)} non-empty labels", flush=True)

    ncols_eff = int(base_indices.max()) + 1 if len(base_indices) > 0 else 1
    label_freq = np.bincount(base_indices, minlength=ncols_eff)

    summary = {
        'dataset': 'laion1m_RICHER_filters',
        'mode': 'v6c_pure_base_freq_singletons',
        'dim': DIM, 'n_base': int(nrows), 'n_query': QUERY_COUNT,
        'min_support_base': MIN_SUPPORT_BASE,
        'pair_budget_mult': args.pair_budget_mult, 'threads': args.threads,
        'filter_source': 'real_caption_uni1800_bi400_DF>=200_OOV',
        'n_labels_total': int(ncols),
        'n_labels_nonempty': int(len(label_to_rs)),
    }

    has_gt_mask = (gt_ids >= 0).any(axis=1)
    n_eval = int(has_gt_mask.sum())
    print(f"  queries with non-empty GT: {n_eval}/{QUERY_COUNT}", flush=True)
    summary['n_eval_queries_with_gt'] = n_eval

    # Query-label distribution
    q_lens = np.diff(q_indptr).astype(np.int64)
    n_q_zero = int((q_lens == 0).sum()); n_q_single = int((q_lens == 1).sum()); n_q_multi = int((q_lens >= 2).sum())
    print(f"  query labels: 0={n_q_zero}, 1={n_q_single}, >=2={n_q_multi}, avg={q_lens.mean():.2f}", flush=True)
    summary['query_label_dist'] = {'zero': n_q_zero, 'one': n_q_single, 'two_plus': n_q_multi,
                                    'avg_labels_per_query': float(q_lens.mean())}

    if args.mode in ('bci', 'both'):
        print(f"\n[BCI selection PURE base-freq]", flush=True)
        base_freq_labels = [int(l) for l in np.where(label_freq >= MIN_SUPPORT_BASE)[0]]
        singleton_edges = sum(int(label_freq[l]) for l in base_freq_labels) * M_HNSW
        print(f"  base-freq labels (>= {MIN_SUPPORT_BASE}): {len(base_freq_labels):,}", flush=True)
        print(f"  singleton edges: {singleton_edges:,}", flush=True)

        # Pair selection (vectorized row-wise pair extraction)
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
        print(f"  total candidate pair-instances: {total_pairs:,}", flush=True)
        if total_pairs > 0:
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
        else:
            pair_keys = np.array([], dtype=np.int64); pair_counts = np.array([], dtype=np.int64)
        print(f"  candidate pairs >= {MIN_SUPPORT_BASE}: {len(pair_keys):,}", flush=True)

        # Train utility from first 5K queries
        train_pair_freq = Counter()
        for qi in range(min(5000, qn)):
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
        print(f"  pairs selected: {len(selected_pairs):,}, edges={used_pair_edges:,}", flush=True)

        # Build HNSWs
        print(f"\n[BCI build]", flush=True)
        t_b = time.time()
        item_hnsw = {}
        rebuilt = 0
        for li, l in enumerate(base_freq_labels):
            path = HNSW_DIR / f"single_{l}.faiss"
            x_i = label_to_rs[l]
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
            item_hnsw[(l,)] = (idx, x_i, len(x_i) * M_HNSW)
            if (li+1) % 200 == 0:
                print(f"  single {li+1:,}/{len(base_freq_labels):,} (rebuilt {rebuilt}), elapsed {time.time()-t_b:.0f}s", flush=True)
        for pi, pair in enumerate(selected_pairs):
            path = HNSW_DIR / f"pair_{pair[0]}_{pair[1]}.faiss"
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
            if (pi+1) % 500 == 0:
                print(f"  pair {pi+1:,}/{len(selected_pairs):,}, elapsed {time.time()-t_b:.0f}s", flush=True)
        print(f"  build done in {time.time()-t_b:.0f}s, {len(item_hnsw):,} indices, {rebuilt} new", flush=True)

        faiss.omp_set_num_threads(args.inner_threads)
        print(f"  [post-build] reset faiss OMP threads to {args.inner_threads}", flush=True)

        total_edges = sum(e for _, _, e in item_hnsw.values())
        n_singles = sum(1 for k in item_hnsw if len(k) == 1)
        n_pairs = sum(1 for k in item_hnsw if len(k) == 2)
        disk_bytes = sum(p.stat().st_size for p in HNSW_DIR.glob('*.faiss'))
        rss_after = proc.memory_info().rss / 1e9
        print(f"\n[memory] Singletons: {n_singles:,}, Pairs: {n_pairs:,}", flush=True)
        print(f"  HNSW edges: {total_edges:,}, Disk: {disk_bytes/1e9:.2f} GB, RSS: {rss_after:.2f} GB", flush=True)

        # Routing
        print(f"\n[BCI route]", flush=True)
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
                    tiny_q.append(ti)
                continue
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
                if all(label_freq[l] < MIN_SUPPORT_BASE for l in q_labels):
                    tiny_q.append(ti)
                else:
                    fallback_q.append(ti)
        n_total = QUERY_COUNT
        n_e_s, n_e_p = len(exact_single_q), len(exact_pair_q)
        n_s_p, n_s_s = len(sub_via_pair_q), len(sub_via_single_q)
        n_f, n_t = len(fallback_q), len(tiny_q)
        print(f"  exact_single: {n_e_s} ({100*n_e_s/n_total:.1f}%)", flush=True)
        print(f"  exact_pair: {n_e_p} ({100*n_e_p/n_total:.1f}%)", flush=True)
        print(f"  sub_via_pair: {n_s_p} ({100*n_s_p/n_total:.1f}%)", flush=True)
        print(f"  sub_via_single: {n_s_s} ({100*n_s_s/n_total:.1f}%)", flush=True)
        print(f"  tiny: {n_t} ({100*n_t/n_total:.1f}%)", flush=True)
        print(f"  fallback: {n_f} ({100*n_f/n_total:.1f}%)", flush=True)

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
            else:  # tiny / 0-label
                q_labels = sorted(int(l) for l in q_indices[q_indptr[qi]:q_indptr[qi+1]])
                if len(q_labels) == 0: return None, (time.perf_counter()-t0)*1e6
                if len(q_labels) == 1:
                    cand = label_to_rs[q_labels[0]] if q_labels[0] in label_to_rs else np.empty(0, dtype=np.int64)
                else:
                    if q_labels[0] in label_to_rs and q_labels[1] in label_to_rs:
                        cand = np.intersect1d(label_to_rs[q_labels[0]], label_to_rs[q_labels[1]], assume_unique=False)
                    else:
                        cand = np.empty(0, dtype=np.int64)
            if len(cand) == 0: return None, (time.perf_counter()-t0)*1e6
            d2 = np.einsum('ij,ij->i', base_f32[cand] - q_vec, base_f32[cand] - q_vec)
            if len(cand) <= K:
                top = cand[np.argsort(d2)]
            else:
                top_idx = np.argpartition(d2, K)[:K]
                top = cand[top_idx[np.argsort(d2[top_idx])]]
            top = top[:K]
            lat_us = (time.perf_counter() - t0) * 1e6
            valid_gt = gt_ids[ti][gt_ids[ti] >= 0]
            if len(valid_gt) == 0:
                return None, lat_us
            recall = len(set(top.tolist()) & set(valid_gt.tolist())) / len(valid_gt)
            return recall, lat_us

        bench_results_bci = []
        for ef in EF_SEARCHES:
            print(f"\n[BCI bench] ef={ef}", flush=True)
            all_tasks = (
                [('exact_single', ti, {'itemset': iset}) for ti, iset in zip(exact_single_q, exact_single_iset)] +
                [('exact_pair', ti, {'itemset': iset}) for ti, iset in zip(exact_pair_q, exact_pair_iset)] +
                [('sub_via_pair', ti, {'itemset': iset, 'residual': res}) for ti, iset, res in zip(sub_via_pair_q, sub_via_pair_iset, sub_via_pair_res)] +
                [('sub_via_single', ti, {'itemset': iset, 'residual': res}) for ti, iset, res in zip(sub_via_single_q, sub_via_single_iset, sub_via_single_res)] +
                [('fallback', ti, {}) for ti in fallback_q] +
                [('tiny', ti, {}) for ti in tiny_q]
            )
            recalls = [None] * len(all_tasks); lats = [0.0] * len(all_tasks)
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
            valid_recalls = [r for r in recalls if r is not None]
            mean_recall = float(np.mean(valid_recalls)) if valid_recalls else 0.0
            lats_arr = np.array(lats)
            print(f"  measured {args.threads}t QPS={measured_qps:.0f}, "
                  f"recall={mean_recall:.4f} (n_eval={len(valid_recalls)}/{QUERY_COUNT})", flush=True)
            print(f"  lat p50={np.percentile(lats_arr,50):.0f}us p95={np.percentile(lats_arr,95):.0f}us", flush=True)
            bench_results_bci.append({
                'ef': ef, 'measured_qps': float(measured_qps),
                'mean_recall': mean_recall, 'n_eval': len(valid_recalls),
                'lat_p50_us': float(np.percentile(lats_arr,50)),
                'lat_p95_us': float(np.percentile(lats_arr,95)),
                'lat_p99_us': float(np.percentile(lats_arr,99)),
            })

        summary.update({
            'n_freq_labels': len(base_freq_labels),
            'n_selected_pairs': len(selected_pairs),
            'n_indices_built': len(item_hnsw),
            'singleton_edges': singleton_edges,
            'pair_edges': used_pair_edges,
            'hnsw_total_edges': total_edges,
            'hnsw_disk_gb': disk_bytes / 1e9,
            'rss_gb': rss_after,
            'route_breakdown': {
                'exact_single': n_e_s, 'exact_pair': n_e_p,
                'sub_via_pair': n_s_p, 'sub_via_single': n_s_s,
                'fallback': n_f, 'tiny': n_t,
            },
            'bci_bench': bench_results_bci,
        })
        # Save after BCI phase in case baseline is slow.
        OUT.write_text(json.dumps(summary, indent=2))
        print(f"  [intermediate save] wrote {OUT}", flush=True)

    # ----- Baseline: unfiltered HNSW + post-filter -----
    if args.mode in ('baseline', 'both'):
        print(f"\n[baseline] HNSW over full 1M + post-filter", flush=True)
        faiss.omp_set_num_threads(args.build_threads)
        baseline_path = BASELINE_DIR / 'full_hnsw.faiss'
        base_hnsw = None
        if baseline_path.exists():
            print(f"  loading existing {baseline_path}", flush=True)
            try:
                base_hnsw = faiss.read_index(str(baseline_path))
            except Exception as ex:
                print(f"  load failed: {ex}", flush=True)
                base_hnsw = None
        if base_hnsw is None:
            t_b = time.time()
            base_hnsw = faiss.IndexHNSWFlat(DIM, M_HNSW)
            base_hnsw.hnsw.efConstruction = EFC
            base_hnsw.add(base_f32)
            print(f"  built in {time.time()-t_b:.0f}s", flush=True)
            faiss.write_index(base_hnsw, str(baseline_path))
        faiss.omp_set_num_threads(args.inner_threads)
        print(f"  [post-build] reset faiss OMP threads to {args.inner_threads}", flush=True)

        def run_baseline_query(ti, ef, k_search):
            qi = QUERY_START + ti
            q_vec = queries_f32[qi:qi+1]
            t0 = time.perf_counter()
            base_hnsw.hnsw.efSearch = ef
            _, I = base_hnsw.search(q_vec, k_search)
            cand_all = I[0][I[0] >= 0]
            q_labels = sorted(int(l) for l in q_indices[q_indptr[qi]:q_indptr[qi+1]])
            cand = cand_all
            for l in q_labels:
                if len(cand) == 0: break
                if l in label_to_rs:
                    cand = cand[np.isin(cand, label_to_rs[l], assume_unique=True)]
                else:
                    cand = np.empty(0, dtype=np.int64); break
            if len(cand) == 0:
                lat_us = (time.perf_counter()-t0)*1e6
                valid_gt = gt_ids[ti][gt_ids[ti] >= 0]
                if len(valid_gt) == 0: return None, lat_us
                return 0.0, lat_us
            d2 = np.einsum('ij,ij->i', base_f32[cand] - q_vec, base_f32[cand] - q_vec)
            if len(cand) <= K:
                top = cand[np.argsort(d2)]
            else:
                top_idx = np.argpartition(d2, K)[:K]
                top = cand[top_idx[np.argsort(d2[top_idx])]]
            top = top[:K]
            lat_us = (time.perf_counter() - t0) * 1e6
            valid_gt = gt_ids[ti][gt_ids[ti] >= 0]
            if len(valid_gt) == 0:
                return None, lat_us
            recall = len(set(top.tolist()) & set(valid_gt.tolist())) / len(valid_gt)
            return recall, lat_us

        bench_results_base = []
        for ef, ks_mult in [(64, 10), (128, 10), (256, 10), (256, 30), (512, 30), (1024, 30), (2048, 30)]:
            k_search = min(K * ks_mult, ef * 4, 100_000)
            k_search = max(k_search, ef)
            print(f"  [baseline bench] ef={ef}, k_search={k_search}", flush=True)
            recalls_b = [None] * QUERY_COUNT; lats_b = [0.0] * QUERY_COUNT
            def do_b(ti):
                r, l = run_baseline_query(ti, ef, k_search)
                recalls_b[ti] = r; lats_b[ti] = l
            t_wall = time.perf_counter()
            if args.threads > 1:
                with ThreadPoolExecutor(max_workers=args.threads) as exe:
                    list(exe.map(do_b, range(QUERY_COUNT)))
            else:
                for ti in range(QUERY_COUNT): do_b(ti)
            wall_s = time.perf_counter() - t_wall
            qps = QUERY_COUNT / wall_s
            valid = [r for r in recalls_b if r is not None]
            recall_mean = float(np.mean(valid)) if valid else 0.0
            lats_arr = np.array(lats_b)
            print(f"    QPS={qps:.0f}, recall={recall_mean:.4f} (n_eval={len(valid)}/{QUERY_COUNT}), "
                  f"p50={np.percentile(lats_arr,50):.0f}us, p95={np.percentile(lats_arr,95):.0f}us", flush=True)
            bench_results_base.append({
                'ef': ef, 'k_search': k_search,
                'measured_qps': float(qps), 'mean_recall': recall_mean,
                'n_eval': len(valid),
                'lat_p50_us': float(np.percentile(lats_arr,50)),
                'lat_p95_us': float(np.percentile(lats_arr,95)),
                'lat_p99_us': float(np.percentile(lats_arr,99)),
            })
        summary['baseline_bench'] = bench_results_base

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {OUT}, total time {time.time()-t_start:.0f}s", flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
