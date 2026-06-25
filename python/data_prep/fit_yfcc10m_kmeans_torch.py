"""Fit k-means C∈{256, 16384} on YFCC-10M using PyTorch GPU (A6000)."""
from __future__ import annotations
import time
import os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data/raw/yfcc100m/base.10M.u8bin"
OUT = ROOT / "03_experiment_bridge/results/raw/sanity_td"


def read_u8bin(p):
    with open(p, "rb") as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        n, d = int(hdr[0]), int(hdr[1])
        data = np.frombuffer(f.read(n * d), dtype=np.uint8).reshape(n, d).copy()
    return data, n, d


@torch.no_grad()
def gpu_kmeans(X, k, niter=15, seed=42, device="cuda:0"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, d = X.shape
    batch_size = min(200_000, max(8_000, int(6e9 / (k * 4))))
    print(f"  batch_size for k={k}: {batch_size}", flush=True)
    torch.cuda.empty_cache()
    idx = torch.randperm(n, generator=g)[:k]
    centroids = X[idx].to(device).float()
    centroids_norm = (centroids * centroids).sum(1)
    counts = torch.zeros(k, device=device)
    for it in range(niter):
        t_it = time.time()
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch = X[start:end].to(device, non_blocking=True).float()
            batch_norm = (batch * batch).sum(1, keepdim=True)
            d2 = batch_norm + centroids_norm[None, :] - 2 * batch @ centroids.T
            assign = d2.argmin(1)
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, assign, batch)
            cnt = torch.zeros(k, device=device).index_add_(
                0, assign, torch.ones_like(assign, dtype=torch.float32))
            new_total = counts + cnt
            w = (cnt / new_total.clamp(min=1)).unsqueeze(1)
            centroids = centroids * (1 - w) + (sums / cnt.clamp(min=1).unsqueeze(1)) * w
            counts = new_total
            centroids_norm = (centroids * centroids).sum(1)
        print(f"  iter {it+1}/{niter} in {time.time()-t_it:.1f}s", flush=True)
    cluster_id = torch.empty(n, dtype=torch.int32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = X[start:end].to(device, non_blocking=True).float()
        batch_norm = (batch * batch).sum(1, keepdim=True)
        d2 = batch_norm + centroids_norm[None, :] - 2 * batch @ centroids.T
        cluster_id[start:end] = d2.argmin(1).cpu().int()
    return centroids.cpu().numpy().astype(np.float32), cluster_id.numpy()


def main():
    print("Loading YFCC-10M u8bin...", flush=True)
    t0 = time.time()
    base, n, d = read_u8bin(DATA)
    print(f"  base: {base.shape}, load {time.time()-t0:.0f}s", flush=True)
    X = torch.from_numpy(base.astype(np.float32)).pin_memory()

    OUT.mkdir(parents=True, exist_ok=True)
    for C in [256, 16384]:
        cache = OUT / f"yfcc10m_kmeans_C{C}.npz"
        if cache.exists():
            print(f"  [SKIP] cached: {cache.name}", flush=True)
            continue
        print(f"\nFitting torch GPU k-means C={C} on YFCC-10M...", flush=True)
        t0 = time.time()
        torch.cuda.empty_cache()
        centroids, cluster_id = gpu_kmeans(X, C, niter=15 if C > 1024 else 25)
        np.savez(cache, cluster_id=cluster_id.astype(np.int32), centroids=centroids)
        print(f"  ✓ done in {time.time()-t0:.0f}s, saved {cache.name}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
