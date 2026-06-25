#!/usr/bin/env python3
"""Build SIFT100M 2-tag conjunction BCI dataset.

Outputs (under <BOOLEANN_ROOT>/data/raw/sift100m/bci/):
  - query.10K.f32bin              (10000 queries × 128 float32, header N + dim)
  - base.metadata.spmat           (100M points × 12 filters; each row = 1 tag)
  - query.metadata.spmat          (200 queries × 12 filters; each row = 2 tags)
  - conjunction_pairs.txt          (chosen pairs + joint selectivity manifest)

Predicate semantics:
  - 12 base equality predicates already exist (labels_s005_p[0-3], labels_s050_p[0-3], labels_s100_p[0-3]).
  - We reuse them as 12 filter columns (filter ID = predicate index 0..11).
  - Per labels_s*_p*.txt we set bit[col] for rows where label == target_class+1 (1-indexed).
  - 8 conjunction pairs picked: prefer cross-sel-band pairs (s005 × s050, s050 × s100, etc.)
    to span joint selectivities ~0.025% (s005 × s005), ~0.25% (s005 × s050), ~1% (s050 × s100).

Notes:
  - filter id == predicate index in predicates_manifest_gt.txt order.
  - target_label is in [target_class+1]; we DO NOT store the raw label values; only one
    bit (=1) per matching row per filter. This keeps the spmat compact.
"""
import numpy as np
import os
import struct
import time
import sys
import os
from pathlib import Path

ROOT = Path(os.environ.get('BOOLEANN_ROOT', Path(__file__).resolve().parents[2]))
SIFT_DIR = ROOT / 'data/raw/sift100m/bci'
DA_DIR = ROOT / 'baselines/MS_DiskANN/datasets/sift100m'
SIFT_QUERY = ROOT / 'data/raw/sift/sift_query.fvecs'

SIFT_DIR.mkdir(parents=True, exist_ok=True)
N = 100_000_000
DIM = 128
NQ_TOTAL = 10000   # query.10K
NQ_ACTIVE = 200    # first 200 queries get conjunction predicates

# Load manifest of 12 predicates
MAN = DA_DIR / 'predicates_manifest_gt.txt'
preds = []
with open(MAN) as f:
    header = f.readline().rstrip()
    for line in f:
        s_target, pred_i, target_label, actual_s, label_file, gt_file = line.strip().split(',')
        preds.append(dict(
            s_target=float(s_target),
            pred_i=int(pred_i),
            target_label=int(target_label),
            actual_s=float(actual_s),
            label_file=label_file,
            gt_file=gt_file,
            pred_idx=len(preds),  # filter ID in spmat (0..11)
        ))

print(f"[manifest] {len(preds)} equality predicates loaded", flush=True)
for p in preds:
    print(f"  pred_idx={p['pred_idx']}: s={p['s_target']:.3f} target_label={p['target_label']} actual_s={p['actual_s']:.6f}")

# -- Step 1: read queries (sift_query.fvecs has 10K queries × 128 dim) ----
def read_fvecs(p, max_n=None):
    with open(p, 'rb') as f:
        data = f.read()
    d = struct.unpack('<i', data[:4])[0]
    n = len(data) // (4 + d*4)
    if max_n: n = min(n, max_n)
    return np.frombuffer(data, dtype=np.float32).reshape(-1, d+1)[:n, 1:].copy()

print(f"\n[query] reading {SIFT_QUERY}", flush=True)
queries = read_fvecs(SIFT_QUERY)
print(f"  shape={queries.shape}, dtype={queries.dtype}")
assert queries.shape[1] == DIM
assert queries.shape[0] >= NQ_TOTAL, f"want {NQ_TOTAL} queries, file has {queries.shape[0]}"

q_out = SIFT_DIR / 'query.10K.f32bin'
print(f"\n[query] writing {q_out}", flush=True)
with open(q_out, 'wb') as f:
    f.write(struct.pack('<II', NQ_TOTAL, DIM))
    queries[:NQ_TOTAL].astype(np.float32).tofile(f)
print(f"  wrote {q_out.stat().st_size} bytes")

# -- Step 2: load each predicate's matching-row indices (the valid_idx set) ----
# Quickest way: for each labels_*.txt, find lines where int(line) == target_label.
# 100M lines × 4-byte int read each → can use np.loadtxt or stream parse.
# We just need a boolean mask per predicate.
print("\n[labels] loading all 12 predicate masks", flush=True)
pred_masks = []
for p in preds:
    t0 = time.time()
    # use int8 mask: 1 if label == target_label else 0
    target = p['target_label']
    # quick load: ASCII labels are small integers (max ~200 for s005 case), one per line
    arr = np.loadtxt(p['label_file'], dtype=np.int32)
    assert arr.shape[0] == N, f"{p['label_file']} has {arr.shape[0]} rows, expected {N}"
    mask = (arr == target)
    n_match = int(mask.sum())
    actual = n_match / N
    print(f"  pred_idx={p['pred_idx']}: |match|={n_match} (s={actual:.6f}) in {time.time()-t0:.1f}s", flush=True)
    pred_masks.append(mask)
    del arr

# -- Step 3: build conjunction pairs ----
# 8 pairs spanning joint selectivities.
# Strategy: pair sizes (small_sel, big_sel), repeat across 4 reps for diversity.
# Joint sel ~= sel_a × sel_b assuming independence (which IS true since masks are i.i.d.)
# Targeted joint sel bands: 0.025% (s005×s005), 0.25% (s005×s050), 1% (s050×s050), 5% (s050×s100)
# 8 pairs:
#   (s005_p0, s005_p1) joint ~0.000025 = 0.0025%
#   (s005_p2, s050_p0) joint ~0.00025  = 0.025%
#   (s005_p3, s050_p1) joint ~0.00025
#   (s050_p2, s050_p3) joint ~0.0025   = 0.25%
#   (s050_p0, s100_p0) joint ~0.005    = 0.5%
#   (s050_p1, s100_p1) joint ~0.005
#   (s050_p2, s100_p2) joint ~0.005
#   (s100_p0, s100_p3) joint ~0.01     = 1%

def find_pred(s_target, pred_i):
    for idx, p in enumerate(preds):
        if abs(p['s_target'] - s_target) < 1e-9 and p['pred_i'] == pred_i:
            return idx
    raise KeyError(f"no pred sel={s_target} pred_i={pred_i}")

PAIRS_SPEC = [
    (0.005, 0, 0.005, 1),
    (0.005, 2, 0.05, 0),
    (0.005, 3, 0.05, 1),
    (0.05, 2, 0.05, 3),
    (0.05, 0, 0.1, 0),
    (0.05, 1, 0.1, 1),
    (0.05, 2, 0.1, 2),
    (0.1, 0, 0.1, 3),
]
pairs = []
for sa, pa, sb, pb in PAIRS_SPEC:
    pairs.append((find_pred(sa, pa), find_pred(sb, pb)))

print("\n[pairs] computing joint selectivities", flush=True)
pair_meta = []
for i, (a, b) in enumerate(pairs):
    joint = (pred_masks[a] & pred_masks[b]).sum()
    js = joint / N
    print(f"  pair {i}: ({preds[a]['s_target']:.3f}_p{preds[a]['pred_i']}) ∧ ({preds[b]['s_target']:.3f}_p{preds[b]['pred_i']})"
          f"  |A∩B|={joint} joint_s={js:.6f}", flush=True)
    pair_meta.append(dict(pair=i, a=a, b=b, joint=int(joint), joint_s=js))

# Save pair manifest
with open(SIFT_DIR / 'conjunction_pairs.txt', 'w') as f:
    f.write("pair,filter_a,filter_b,sel_a,pred_a_i,sel_b,pred_b_i,joint_count,joint_s\n")
    for m in pair_meta:
        pa, pb = preds[m['a']], preds[m['b']]
        f.write(f"{m['pair']},{m['a']},{m['b']},{pa['s_target']},{pa['pred_i']},{pb['s_target']},{pb['pred_i']},{m['joint']},{m['joint_s']:.6f}\n")
print(f"\n[pair-manifest] wrote {SIFT_DIR / 'conjunction_pairs.txt'}")

# -- Step 4: build BASE metadata spmat (100M rows × 12 filters; CSR per-point) ----
# Per filter, we have a mask. CSR format expects: row[p] = list of filter ids where mask=1.
# Equivalent to: for each point, scan its 12 bits and collect the on bits.
print("\n[base.spmat] building per-point CSR", flush=True)
t0 = time.time()
# Stack masks: (12, N) bool → transpose to (N, 12)
stacked = np.stack(pred_masks, axis=0)  # (12, N) bool
# count per-row tag count
row_counts = stacked.sum(axis=0)  # (N,) int8/int32
n_nonzero = int(row_counts.sum())
# row_offsets[N+1]
row_offsets = np.zeros(N+1, dtype=np.int64)
np.cumsum(row_counts, dtype=np.int64, out=row_offsets[1:])
# row_indices: for each row, list of filter idx where mask=1
# Vectorize via finding nonzero positions of stacked (transposed iteration).
# stacked[:, i] tells us which filters i belongs to.
# Use np.argwhere on stacked.T = (N, 12)
print(f"  total nonzeros = {n_nonzero}, building row_indices...", flush=True)
row_indices = np.zeros(n_nonzero, dtype=np.int32)
# Stream: for each filter idx in order, append filter idx at row positions in mask
# But that gives wrong order (need per-row sorted). Easier: iterate by row in chunks.
CHUNK = 10_000_000
pos = 0
for r0 in range(0, N, CHUNK):
    r1 = min(r0 + CHUNK, N)
    sub = stacked[:, r0:r1]   # (12, chunk)
    sub_T = sub.T.copy()      # (chunk, 12)
    # for each row, collect filter ids where True
    # use np.where on sub_T (returns row-major)
    row_ids, fil_ids = np.where(sub_T)
    # row_ids is in [0, chunk-1]; filter ids 0..11
    # row_ids is sorted since np.where iterates row-major → good, sorted per row
    # we just need to copy fil_ids to row_indices[pos:pos+len(fil_ids)]
    row_indices[pos:pos+len(fil_ids)] = fil_ids.astype(np.int32)
    pos += len(fil_ids)
    if (r0 // CHUNK) % 5 == 0:
        print(f"    {r1/1e6:.0f}M rows done ({time.time()-t0:.1f}s, pos={pos})", flush=True)
assert pos == n_nonzero, f"pos {pos} != nnz {n_nonzero}"
print(f"  built CSR in {time.time()-t0:.1f}s. n_nonzero={n_nonzero}", flush=True)

# Write spmat
n_filters = 12
out_path = SIFT_DIR / 'base.metadata.spmat'
with open(out_path, 'wb') as f:
    f.write(struct.pack('<qqq', N, n_filters, n_nonzero))
    row_offsets.tofile(f)
    row_indices.tofile(f)
print(f"  wrote {out_path} ({out_path.stat().st_size} bytes)")

# Sanity: verify offsets[0]=0, offsets[N]=nnz, last few rows look right.
print(f"  offsets[0]={row_offsets[0]}, offsets[N]={row_offsets[N]} (should be {n_nonzero})")

del stacked

# -- Step 5: build QUERY metadata spmat (10K queries × 12 filters) ----
# First 200 queries get the conjunction pair assignment (25 queries per pair × 8 pairs = 200).
# Remaining queries get empty (n_tags=0) → skipped by bench.
print(f"\n[query.spmat] building query metadata for {NQ_TOTAL} queries; first {NQ_ACTIVE} = 8 pairs × 25 each", flush=True)
n_per_pair = NQ_ACTIVE // len(pairs)  # 25
assert n_per_pair * len(pairs) == NQ_ACTIVE
q_row_counts = np.zeros(NQ_TOTAL, dtype=np.int32)
q_row_indices_buf = []
for q in range(NQ_TOTAL):
    if q < NQ_ACTIVE:
        pair_id = q // n_per_pair
        a, b = pairs[pair_id]
        # csr_filters sorts on load, but we keep small-first for safety
        if a < b:
            q_row_indices_buf.extend([a, b])
        else:
            q_row_indices_buf.extend([b, a])
        q_row_counts[q] = 2
    else:
        q_row_counts[q] = 0
q_row_offsets = np.zeros(NQ_TOTAL+1, dtype=np.int64)
np.cumsum(q_row_counts, dtype=np.int64, out=q_row_offsets[1:])
q_row_indices = np.array(q_row_indices_buf, dtype=np.int32)
q_nnz = q_row_indices.size

q_spmat_path = SIFT_DIR / 'query.metadata.spmat'
with open(q_spmat_path, 'wb') as f:
    f.write(struct.pack('<qqq', NQ_TOTAL, n_filters, q_nnz))
    q_row_offsets.tofile(f)
    q_row_indices.tofile(f)
print(f"  wrote {q_spmat_path} ({q_spmat_path.stat().st_size} bytes); active queries={NQ_ACTIVE}")

# Manifest summary
print("\n[done] all metadata artifacts written:")
for f in ['query.10K.f32bin', 'base.metadata.spmat', 'query.metadata.spmat', 'conjunction_pairs.txt']:
    p = SIFT_DIR / f
    print(f"  {p}: {p.stat().st_size if p.exists() else 'MISSING'} bytes")
