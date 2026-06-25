#!/usr/bin/env python3
"""laion_richer_filters.py — richer vocabulary version of laion_real_filters.py.

Goal: drive empty-query rate from 36.1% (top-200 unigrams) down to <5%, and
push avg labels/query from 0.97 to ~1.5-2.0 so BCI's pair structure has
something to bite into.

Strategy:
  * Vocabulary = top-1800 unigrams + top-400 bigrams (total ~2200 labels),
    MIN_DF=200 (was 500).
  * Bigrams add multi-word concepts (e.g. "ice cream", "new york") that a
    pure unigram vocab misses for visual scenes.
  * Per-query: keep up to MAX_Q_LABELS_PER_ROW=2 rarest labels (same as
    real_filters.py); but with richer vocab the chance of zero hits drops.
  * If a query row STILL has zero in-vocab tokens after this expanded vocab,
    fall back to a sentinel "OOV" label (id=ncols-1) so the query is not
    empty — that label has support across rows with no vocab match, which
    is honest: it means "any item with non-vocab caption".

Outputs (under data/raw/laion1m/):
  base.metadata.laion1m.richer.spmat
  query.metadata.laion1m.richer.spmat
  gt.10K.richer.bin
  gt.10K.richer.dist.bin
  richer_filters_summary.json
  richer_vocab.json

Run:
  python laion_richer_filters.py
"""
import json, time, os, re, sys
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / 'data/raw/laion1m/raw'
OUT = ROOT / 'data/raw/laion1m'
META = RAW / 'metadata_0.parquet'

N_TARGET = 1_000_000
N_QUERY = 10_000
N_UNIGRAMS = 1800
N_BIGRAMS = 400
MIN_SUPPORT_BASE = 200
MAX_Q_LABELS_PER_ROW = 2
USE_OOV_SENTINEL = True  # If a query row has 0 vocab tokens, assign OOV.

STOPWORDS = set("""
a about above after again against all also am an and any are aren as at
be because been before being below between both but by can cant could couldnt
did didnt do does doesnt doing dont down during each few for from further
had hadnt has hasnt have havent having he her here hers herself him himself his
how i if im in into is isnt it its itself just lets me more most must my myself
no nor not now of off on once only or other our ours ourselves out over own
same shall she should shouldnt so some such than that thats the their theirs
them themselves then there these they this those through to too under until up
very was wasnt we were werent what when where which while who whom why with
would wouldnt you your yours yourself yourselves
new old big small high low best great good top get one two three made make
made via inc com http https www jpg jpeg png gif image photo picture pic
stock vector free shutterstock istock alamy 123rf gettyimages dreamstime
""".split())

# Token: alphabetic, length 3..20.
TOKEN_RE = re.compile(r"[a-z]{3,20}")


def spmat_write(path, indptr, indices, ncols):
    nrows = len(indptr) - 1
    nnz = len(indices)
    assert indptr.dtype == np.int64
    assert indices.dtype == np.int32
    with open(path, 'wb') as f:
        f.write(np.array([nrows, ncols, nnz], dtype=np.int64).tobytes())
        indptr.tofile(f)
        indices.tofile(f)


def tokenize(cap_lower: str):
    """Return list of (non-stopword) tokens preserving order for bigram extraction."""
    return [m.group(0) for m in TOKEN_RE.finditer(cap_lower) if m.group(0) not in STOPWORDS]


def main():
    t0 = time.time()
    assert META.exists(), f"missing {META}"
    print(f"[load] {META} ({META.stat().st_size/1e6:.1f} MB)", flush=True)
    df = pd.read_parquet(META, columns=['caption'])
    print(f"  rows={len(df):,}", flush=True)
    captions = df['caption'].fillna('').astype(str).values[:N_TARGET]
    if len(captions) < N_TARGET:
        print(f"  WARN: only {len(captions)} captions, < {N_TARGET}", file=sys.stderr)

    # Pass 1: tokenize + count DF for unigrams and bigrams.
    print(f"[vocab] tokenizing {len(captions):,} captions (uni+bi)...", flush=True)
    t_tok = time.time()
    row_unigram_sets = [None] * len(captions)
    row_bigram_sets = [None] * len(captions)
    uni_df = Counter()
    bi_df = Counter()
    for i, cap in enumerate(captions):
        toks = tokenize(cap.lower())
        uni_set = set(toks)
        # Bigrams: consecutive non-stopword pairs.
        bi_set = set()
        for a, b in zip(toks, toks[1:]):
            bi_set.add(a + '_' + b)
        row_unigram_sets[i] = uni_set
        row_bigram_sets[i] = bi_set
        for tk in uni_set:
            uni_df[tk] += 1
        for bg in bi_set:
            bi_df[bg] += 1
        if (i + 1) % 200_000 == 0:
            print(f"  tok {i+1:,}/{len(captions):,} elapsed={time.time()-t_tok:.0f}s "
                  f"vocab uni={len(uni_df):,} bi={len(bi_df):,}", flush=True)
    print(f"  tokenization done {time.time()-t_tok:.0f}s; "
          f"unique uni={len(uni_df):,}, bi={len(bi_df):,}", flush=True)

    # Pick top-N unigrams with df>=MIN_SUPPORT_BASE.
    uni_candidates = [(tk, c) for tk, c in uni_df.items() if c >= MIN_SUPPORT_BASE]
    uni_candidates.sort(key=lambda x: -x[1])
    uni_chosen = uni_candidates[:N_UNIGRAMS]
    bi_candidates = [(tk, c) for tk, c in bi_df.items() if c >= MIN_SUPPORT_BASE]
    bi_candidates.sort(key=lambda x: -x[1])
    bi_chosen = bi_candidates[:N_BIGRAMS]
    print(f"  selected {len(uni_chosen)} unigrams (min_df={uni_chosen[-1][1] if uni_chosen else 0}); "
          f"{len(bi_chosen)} bigrams (min_df={bi_chosen[-1][1] if bi_chosen else 0})", flush=True)
    for tk, c in uni_chosen[:5]:
        print(f"    UNI {tk:20s} df={c:,}", flush=True)
    for tk, c in bi_chosen[:5]:
        print(f"    BI  {tk:25s} df={c:,}", flush=True)

    # Combined vocabulary with one shared label space.
    # IDs: 0..len(uni_chosen)-1 = unigrams, then bigrams, then optional OOV at end.
    combined = [('uni', tk, c) for tk, c in uni_chosen] + [('bi', tk, c) for tk, c in bi_chosen]
    if USE_OOV_SENTINEL:
        n_labels_real = len(combined)
        OOV_ID = n_labels_real
        n_labels = n_labels_real + 1
    else:
        n_labels_real = len(combined)
        OOV_ID = None
        n_labels = n_labels_real
    label_to_id = {}
    for i, (kind, tk, _) in enumerate(combined):
        key = ('u', tk) if kind == 'uni' else ('b', tk)
        label_to_id[key] = i
    print(f"  total label vocab: {n_labels} (real={n_labels_real}, OOV={OOV_ID})", flush=True)

    # Pass 2: build base.spmat from intersection of row's tokens with vocab.
    print(f"[build base spmat]", flush=True)
    base_indptr = np.zeros(len(captions) + 1, dtype=np.int64)
    base_indices_list = []
    rows_with_zero_real_labels = 0
    for i, (uni_set, bi_set) in enumerate(zip(row_unigram_sets, row_bigram_sets)):
        row_ids = set()
        for t in uni_set:
            li = label_to_id.get(('u', t))
            if li is not None: row_ids.add(li)
        for t in bi_set:
            li = label_to_id.get(('b', t))
            if li is not None: row_ids.add(li)
        if len(row_ids) == 0:
            rows_with_zero_real_labels += 1
            if USE_OOV_SENTINEL:
                row_ids.add(OOV_ID)
        ids_sorted = sorted(row_ids)
        base_indptr[i + 1] = base_indptr[i] + len(ids_sorted)
        base_indices_list.append(np.asarray(ids_sorted, dtype=np.int32))
    base_indices = np.concatenate(base_indices_list) if base_indices_list else np.zeros(0, dtype=np.int32)
    print(f"  base nnz={len(base_indices):,}, avg/row={len(base_indices)/len(captions):.2f} "
          f"(rows w/ 0 real-label = {rows_with_zero_real_labels}, OOV applied)", flush=True)
    if len(captions) < N_TARGET:
        pad = N_TARGET - len(captions)
        print(f"  padding {pad} unlabeled rows", flush=True)
        new_indptr = np.zeros(N_TARGET + 1, dtype=np.int64)
        new_indptr[: len(captions) + 1] = base_indptr
        new_indptr[len(captions) + 1:] = base_indptr[-1]
        base_indptr = new_indptr

    label_freq = np.bincount(base_indices, minlength=n_labels)
    print(f"  per-label support: min={label_freq.min()}, max={label_freq.max()}, "
          f"mean={label_freq.mean():.0f}, >={MIN_SUPPORT_BASE}: {int((label_freq >= MIN_SUPPORT_BASE).sum())}", flush=True)
    base_path = OUT / 'base.metadata.laion1m.richer.spmat'
    spmat_write(base_path, base_indptr, base_indices, n_labels)
    print(f"[write] {base_path} ({base_path.stat().st_size/1e6:.2f} MB)", flush=True)

    # Build query.spmat — last N_QUERY base rows.
    # Strategy: pick the MAX_Q_LABELS_PER_ROW rarest labels from row's labels. If the
    # row had OOV-only (sentinel), the query gets OOV; that's a deliberate
    # honest design choice — those rows have non-vocab captions.
    print(f"[build query spmat]", flush=True)
    q_indptr = np.zeros(N_QUERY + 1, dtype=np.int64)
    q_indices_list = []
    q_label_counts = []
    for qi in range(N_QUERY):
        bi = N_TARGET - N_QUERY + qi
        s, e = int(base_indptr[bi]), int(base_indptr[bi + 1])
        row_labels = base_indices[s:e]
        if len(row_labels) == 0:
            chosen_q = np.empty(0, dtype=np.int32)
        else:
            order = np.argsort(label_freq[row_labels])
            keep = row_labels[order][:MAX_Q_LABELS_PER_ROW]
            chosen_q = np.sort(keep).astype(np.int32)
        q_indices_list.append(chosen_q)
        q_label_counts.append(len(chosen_q))
        q_indptr[qi + 1] = q_indptr[qi] + len(chosen_q)
    q_indices = np.concatenate(q_indices_list) if q_indices_list else np.zeros(0, dtype=np.int32)
    q_label_counts_np = np.array(q_label_counts)
    n_q_zero = int((q_label_counts_np == 0).sum())
    n_q_single = int((q_label_counts_np == 1).sum())
    n_q_multi = int((q_label_counts_np >= 2).sum())
    print(f"  query nnz={len(q_indices):,}, avg/query={q_label_counts_np.mean():.2f}", flush=True)
    print(f"  empty queries: {n_q_zero} ({100*n_q_zero/N_QUERY:.2f}%), "
          f"1-label: {n_q_single} ({100*n_q_single/N_QUERY:.2f}%), "
          f">=2 labels: {n_q_multi} ({100*n_q_multi/N_QUERY:.2f}%)", flush=True)
    q_path = OUT / 'query.metadata.laion1m.richer.spmat'
    spmat_write(q_path, q_indptr, q_indices, n_labels)
    print(f"[write] {q_path} ({q_path.stat().st_size/1e3:.0f} KB)", flush=True)

    # ---------- GT computation ----------
    print(f"\n[GT] computing filtered top-10 brute-force GT over {N_QUERY} queries...", flush=True)
    K = 10
    BASE_F32 = OUT / 'base.1M.f32bin'
    QUERY_F32 = OUT / 'query.10K.f32bin'

    def _read_f32(p, n=None):
        with open(p, 'rb') as f:
            hdr = np.frombuffer(f.read(8), dtype=np.int32)
            tn, dd = int(hdr[0]), int(hdr[1])
            if n is None or n > tn: n = tn
            return np.frombuffer(f.read(n * dd * 4), dtype=np.float32).reshape(n, dd).copy(), dd
    base, _ = _read_f32(BASE_F32, n=N_TARGET)
    queries, _ = _read_f32(QUERY_F32, n=N_QUERY)
    print(f"  loaded base={base.shape}, queries={queries.shape}", flush=True)

    row_id_per_nnz = np.repeat(np.arange(N_TARGET, dtype=np.int32), np.diff(base_indptr).astype(np.int64))
    sort_idx = np.argsort(base_indices, kind='stable')
    sorted_labels = base_indices[sort_idx]
    sorted_rows = row_id_per_nnz[sort_idx]
    boundaries = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
    label_to_rs = {}
    for i in range(len(boundaries) - 1):
        ll = int(sorted_labels[boundaries[i]])
        g = sorted_rows[boundaries[i]:boundaries[i+1]].astype(np.int64); g.sort()
        label_to_rs[ll] = g

    gt_ids = np.zeros((N_QUERY, K), dtype=np.int32)
    gt_dists = np.zeros((N_QUERY, K), dtype=np.float32)
    t_gt = time.time()
    n_empty_gt = 0
    for qi in range(N_QUERY):
        qls = q_indices[q_indptr[qi]:q_indptr[qi+1]]
        if len(qls) == 0:
            gt_ids[qi] = -1; gt_dists[qi] = np.inf; n_empty_gt += 1; continue
        if len(qls) == 1:
            cand = label_to_rs[int(qls[0])]
        else:
            cand = label_to_rs[int(qls[0])]
            for ll in qls[1:]:
                cand = cand[np.isin(cand, label_to_rs[int(ll)], assume_unique=True)]
        if len(cand) == 0:
            gt_ids[qi] = -1; gt_dists[qi] = np.inf; n_empty_gt += 1; continue
        self_id = (N_TARGET - N_QUERY) + qi
        cand = cand[cand != self_id]
        if len(cand) == 0:
            gt_ids[qi] = -1; gt_dists[qi] = np.inf; n_empty_gt += 1; continue
        q_vec = queries[qi:qi+1]
        diffs = base[cand] - q_vec
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        if len(cand) <= K:
            ti = np.argsort(d2)
            gt_ids[qi, :len(cand)] = cand[ti]
            gt_ids[qi, len(cand):] = -1
            gt_dists[qi, :len(cand)] = d2[ti]
            gt_dists[qi, len(cand):] = np.inf
        else:
            ti = np.argpartition(d2, K)[:K]
            order = np.argsort(d2[ti])
            ti = ti[order]
            gt_ids[qi] = cand[ti]
            gt_dists[qi] = d2[ti]
        if (qi + 1) % 2000 == 0:
            print(f"    GT {qi+1:,}/{N_QUERY:,} elapsed={time.time()-t_gt:.0f}s", flush=True)
    print(f"  GT done in {time.time()-t_gt:.0f}s; {n_empty_gt} queries with no matches", flush=True)

    gt_path = OUT / 'gt.10K.richer.bin'
    gt_dist_path = OUT / 'gt.10K.richer.dist.bin'
    with open(gt_path, 'wb') as f:
        f.write(np.array([N_QUERY, K], dtype=np.int32).tobytes())
        gt_ids.tofile(f)
    with open(gt_dist_path, 'wb') as f:
        f.write(np.array([N_QUERY, K], dtype=np.int32).tobytes())
        gt_dists.tofile(f)
    print(f"[write] {gt_path}, {gt_dist_path}", flush=True)

    summary = {
        'metadata_source': str(META),
        'n_base': N_TARGET,
        'n_query': N_QUERY,
        'n_unigrams': len(uni_chosen),
        'n_bigrams': len(bi_chosen),
        'n_labels_total': n_labels,
        'oov_id': OOV_ID,
        'use_oov_sentinel': USE_OOV_SENTINEL,
        'min_support_base': MIN_SUPPORT_BASE,
        'max_q_labels_per_row': MAX_Q_LABELS_PER_ROW,
        'avg_labels_per_base_row': float(len(base_indices) / N_TARGET),
        'avg_labels_per_query': float(q_label_counts_np.mean()),
        'min_label_support': int(label_freq.min()),
        'max_label_support': int(label_freq.max()),
        'mean_label_support': float(label_freq.mean()),
        'labels_above_500': int((label_freq >= 500).sum()),
        'labels_above_min': int((label_freq >= MIN_SUPPORT_BASE).sum()),
        'mean_sparsity': float(label_freq.mean() / N_TARGET),
        'rows_with_zero_real_labels': rows_with_zero_real_labels,
        'zero_label_queries': n_q_zero,
        'zero_label_query_rate': float(n_q_zero / N_QUERY),
        'one_label_queries': n_q_single,
        'multi_label_queries': n_q_multi,
        'empty_gt_queries': int(n_empty_gt),
        'tokenization_n_unique_uni': len(uni_df),
        'tokenization_n_unique_bi': len(bi_df),
        'top_10_unigrams': [{'token': tk, 'df': int(c)} for tk, c in uni_chosen[:10]],
        'top_10_bigrams': [{'token': tk, 'df': int(c)} for tk, c in bi_chosen[:10]],
        'total_time_s': time.time() - t0,
    }
    (OUT / 'richer_filters_summary.json').write_text(json.dumps(summary, indent=2))
    vocab_dump = (
        [{'id': i, 'kind': 'uni', 'token': tk, 'df': int(c)} for i, (tk, c) in enumerate(uni_chosen)] +
        [{'id': len(uni_chosen)+j, 'kind': 'bi', 'token': tk, 'df': int(c)} for j, (tk, c) in enumerate(bi_chosen)]
    )
    if USE_OOV_SENTINEL:
        vocab_dump.append({'id': OOV_ID, 'kind': 'oov', 'token': '<OOV>', 'df': int(label_freq[OOV_ID])})
    (OUT / 'richer_vocab.json').write_text(json.dumps(vocab_dump, indent=2))
    print(f"\n[done] total time {time.time()-t0:.0f}s", flush=True)
    print(f"  empty-query rate dropped from 36.1%% (top-200 uni) to {100*n_q_zero/N_QUERY:.2f}%", flush=True)
    print(f"  avg labels/query: {q_label_counts_np.mean():.2f}", flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
