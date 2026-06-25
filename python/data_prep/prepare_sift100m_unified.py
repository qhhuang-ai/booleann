#!/usr/bin/env python3
"""Unified data prep for SIFT100M baselines (UNG, DiskANN, iRangeGraph, WoW, HAMCG, ACORN).

Step 1: Stream-convert SIFT100M uint8 raw → float32 (with per-vec fvecs dim prefix)
Step 2: Per-baseline binary formats (DiskANN .bin, WoW fvecs symlink, iRangeGraph sorted, etc.)
Step 3: Generate shared 12-predicate manifest (same seeds as SIFT10M, 4 per sel × 3 sels)
Step 4: Per-baseline label files + GT (top-10 filtered nearest neighbor, GPU brute-force)

All baselines share:
- Same float32 base data
- Same 12 predicates (same seeds)
- Same 200 queries (sift_query.fvecs first 200)
- Same GT
"""
import numpy as np
import struct
import time
import os
from pathlib import Path

SIFT100M_U8 = Path(os.environ.get('BOOLEANN_ROOT', str(Path(__file__).resolve().parents[2]))) / 'data/raw/bigann/sift100m_base.u8raw'
SIFT_QUERY = Path(os.environ.get('BOOLEANN_ROOT', str(Path(__file__).resolve().parents[2]))) / 'data/raw/sift/sift_query.fvecs'
F32_DIR = Path(os.environ.get('BOOLEANN_ROOT', str(Path(__file__).resolve().parents[2]))) / 'data/raw/sift100m'
F32_DIR.mkdir(parents=True, exist_ok=True)
F32_FVECS = F32_DIR / 'sift100m_base_f32.fvecs'   # with dim prefix
F32_RAW = F32_DIR / 'sift100m_base_f32.raw'        # no header (for HAMCG faiss)

N = 100_000_000
DIM = 128
NQ = 200
K = 10
SEED = 42
SELS = [0.005, 0.05, 0.10]
N_PRED_PER_SEL = 4

# --- 1. Convert uint8 → float32 (chunked streaming) ---
CHUNK = 200_000  # 200K × 128 = 25.6 MB / chunk float32

if not F32_RAW.exists():
    print(f"[convert] uint8 → float32 raw, N={N} dim={DIM}")
    t0 = time.time()
    with open(SIFT100M_U8, 'rb') as fin, open(F32_RAW, 'wb') as fout:
        for chunk_start in range(0, N, CHUNK):
            chunk_n = min(CHUNK, N - chunk_start)
            buf = fin.read(chunk_n * DIM * 1)  # uint8 = 1 byte
            arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(chunk_n, DIM)
            arr.tofile(fout)
            if chunk_start % 10_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {chunk_start//1_000_000}M / {N//1_000_000}M ({elapsed:.0f}s elapsed)")
    print(f"  done: {F32_RAW}, {F32_RAW.stat().st_size/1e9:.2f}GB in {time.time()-t0:.0f}s")
else:
    print(f"[skip] {F32_RAW} exists ({F32_RAW.stat().st_size/1e9:.2f}GB)")

if not F32_FVECS.exists():
    print(f"[convert] raw → fvecs (with dim prefix)")
    t0 = time.time()
    with open(F32_RAW, 'rb') as fin, open(F32_FVECS, 'wb') as fout:
        for chunk_start in range(0, N, CHUNK):
            chunk_n = min(CHUNK, N - chunk_start)
            buf = fin.read(chunk_n * DIM * 4)  # float32 = 4 bytes
            arr = np.frombuffer(buf, dtype=np.float32).reshape(chunk_n, DIM)
            dim_arr = np.full(chunk_n, DIM, dtype=np.int32)
            out_arr = np.concatenate(
                [dim_arr.reshape(-1, 1).view(np.float32), arr], axis=1
            )
            out_arr.astype(np.float32).tofile(fout)
            if chunk_start % 10_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {chunk_start//1_000_000}M / {N//1_000_000}M ({elapsed:.0f}s)")
    print(f"  done: {F32_FVECS}, {F32_FVECS.stat().st_size/1e9:.2f}GB in {time.time()-t0:.0f}s")
else:
    print(f"[skip] {F32_FVECS} exists ({F32_FVECS.stat().st_size/1e9:.2f}GB)")

# --- 2. UNG / DiskANN .bin format (no dim prefix per vec; just n×d header + data) ---
def write_diskann_bin_from_raw(raw_path, out_path, n, d):
    """Convert raw float32 (no header) to DiskANN .bin (n int32 + d int32 + n*d float32)."""
    with open(raw_path, 'rb') as fin, open(out_path, 'wb') as fout:
        fout.write(struct.pack('<ii', n, d))
        # Stream the rest
        CHUNK_BYTES = CHUNK * d * 4
        while True:
            buf = fin.read(CHUNK_BYTES)
            if not buf:
                break
            fout.write(buf)
    print(f"  wrote {out_path}: {n}x{d}, {out_path.stat().st_size/1e9:.2f}GB")

UNG_DIR = Path(os.environ.get('BOOLEANN_ROOT', str(Path(__file__).resolve().parents[2]))) / 'baselines/UNG/data/sift100m'
DA_DIR = Path(os.environ.get('BOOLEANN_ROOT', str(Path(__file__).resolve().parents[2]))) / 'baselines/MS_DiskANN/datasets/sift100m'
UNG_DIR.mkdir(parents=True, exist_ok=True)
DA_DIR.mkdir(parents=True, exist_ok=True)

ung_base_bin = UNG_DIR / 'sift100m_base.bin'
da_base_bin = DA_DIR / 'sift100m_base.bin'
if not ung_base_bin.exists():
    print(f"[bin] UNG base from raw → {ung_base_bin}")
    write_diskann_bin_from_raw(F32_RAW, ung_base_bin, N, DIM)
if not da_base_bin.exists():
    print(f"[bin] DiskANN base from raw → {da_base_bin}")
    write_diskann_bin_from_raw(F32_RAW, da_base_bin, N, DIM)

# Queries
def read_fvecs(p, max_n=None):
    with open(p, 'rb') as f: data = f.read()
    d = struct.unpack('<i', data[:4])[0]
    n = len(data) // (4 + d*4)
    if max_n: n = min(n, max_n)
    return np.frombuffer(data, dtype=np.float32).reshape(-1, d+1)[:n, 1:].copy()

print(f"[load] {NQ} queries...")
queries = read_fvecs(SIFT_QUERY)[:NQ]
print(f"  shape: {queries.shape}")

def write_diskann_bin(arr, path):
    n, d = arr.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<ii', n, d))
        arr.astype(np.float32).tofile(f)

ung_query_bin = UNG_DIR / 'sift100m_query.bin'
da_query_bin = DA_DIR / 'sift100m_query.bin'
if not ung_query_bin.exists(): write_diskann_bin(queries, ung_query_bin)
if not da_query_bin.exists(): write_diskann_bin(queries, da_query_bin)

# --- 3. WoW symlinks ---
WOW_VECS = Path(os.environ.get('BOOLEANN_ROOT', str(Path(__file__).resolve().parents[2]))) / 'baselines/WoW/exp/data/vecs/sift100m'
WOW_VECS.mkdir(parents=True, exist_ok=True)
WOW_BASE = WOW_VECS / 'sift_base.fvecs'
WOW_QUERY = WOW_VECS / 'sift_query.fvecs'
if not WOW_BASE.exists():
    WOW_BASE.symlink_to(F32_FVECS)
    print(f"  symlinked {WOW_BASE} → {F32_FVECS}")
if not WOW_QUERY.exists():
    WOW_QUERY.symlink_to(SIFT_QUERY)
    print(f"  symlinked {WOW_QUERY} → {SIFT_QUERY}")

# --- 4. Generate shared predicate manifest + per-baseline labels ---
print(f"\n[predicates] generating shared manifest for {len(SELS) * N_PRED_PER_SEL} = {len(SELS)*N_PRED_PER_SEL} predicates")

ung_manifest = []
da_manifest = []

for s_target in SELS:
    n_classes = max(2, int(round(1.0 / s_target)))
    for pred_i in range(N_PRED_PER_SEL):
        rng = np.random.RandomState(SEED + int(s_target * 1000) + pred_i)
        labels = rng.randint(0, n_classes, size=N).astype(np.int32)
        target_class = rng.randint(0, n_classes)
        actual_s = float((labels == target_class).mean())

        pred_id = f"s{int(s_target*1000):03d}_p{pred_i}"
        # UNG: 1 label per line, 1-indexed (target + 1)
        ung_bl = UNG_DIR / f'base_label_{pred_id}.txt'
        ung_ql = UNG_DIR / f'query_label_{pred_id}.txt'
        if not ung_bl.exists():
            with open(ung_bl, 'w') as f:
                f.write('\n'.join(str(int(l)+1) for l in labels) + '\n')
        if not ung_ql.exists():
            with open(ung_ql, 'w') as f:
                f.write('\n'.join([str(int(target_class)+1)] * NQ) + '\n')

        # DiskANN: 1 label per line, 1-indexed
        da_bl = DA_DIR / f'labels_{pred_id}.txt'
        if not da_bl.exists():
            with open(da_bl, 'w') as f:
                f.write('\n'.join(str(int(l)+1) for l in labels) + '\n')

        ung_manifest.append((s_target, pred_i, target_class+1, actual_s, str(ung_bl), str(ung_ql)))
        da_manifest.append((s_target, pred_i, target_class+1, actual_s, str(da_bl)))
        print(f"  {pred_id}: target={target_class}, actual_s={actual_s:.6f}")

# Write manifests
with open(UNG_DIR / 'predicates_manifest.txt', 'w') as f:
    f.write("s_target,pred_i,target_label,actual_s,base_label_file,query_label_file\n")
    for r in ung_manifest:
        f.write(','.join(map(str, r)) + '\n')

with open(DA_DIR / 'predicates_manifest.txt', 'w') as f:
    f.write("s_target,pred_i,target_label,actual_s,label_file\n")
    for r in da_manifest:
        f.write(','.join(map(str, r)) + '\n')

print(f"\n[done] SIFT100M unified prep")
print(f"  Base float32 raw: {F32_RAW}")
print(f"  Base float32 fvecs: {F32_FVECS}")
print(f"  UNG: {ung_base_bin}, {len(ung_manifest)} predicates")
print(f"  DiskANN: {da_base_bin}, {len(da_manifest)} predicates")
print(f"  WoW: {WOW_BASE}")
print(f"\nNext: generate GT (GPU brute-force, separate script) before any baseline run.")
