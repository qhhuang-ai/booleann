#!/usr/bin/env python3
"""Plot ARWGI concentration: empirical |W_v cap X_phi|/r_v vs p_v(phi),
with theorem envelope (1+/-eps)*p_v where eps = c/sqrt(r * p_v) (Bernstein-style).

Saves PNG + PDF + supporting JSON. Reads from arwgi_proper/.
"""
from __future__ import annotations
import json
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("BOOLEANN_ROOT", Path(__file__).resolve().parents[2]))
OUT_DIR = ROOT / "03_experiment_bridge/results/raw/arwgi_proper"
R = int(sys.argv[1]) if len(sys.argv) > 1 else 24

audit_path = OUT_DIR / f"arwgi_concentration_yfcc10m_r{R}.json"
summary_path = OUT_DIR / f"arwgi_concentration_summary_yfcc10m_r{R}.json"

data = json.loads(audit_path.read_text())
rows = data["rows"]
print(f"Loaded {len(rows)} (node, query) observations")

pv = np.array([r["p_v_phi"] for r in rows])
rate = np.array([r["wv_rate"] for r in rows])
r_eff = np.array([r["r_eff"] for r in rows])

# Theorem envelope: by Hoeffding/Bernstein on hypergeometric mean,
# |rate - pv| <= eps * pv with eps = sqrt(2 log(1/delta) / (r * pv)).
# For visualization use delta = 0.05: log(1/delta) ~ 3 -> eps = sqrt(6/(r*pv)).
delta = 0.05
eps_envelope = np.sqrt(2 * np.log(1.0/delta) / np.maximum(R * pv, 1e-6))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: scatter + identity line
ax = axes[0]
ax.scatter(pv, rate, s=4, alpha=0.20, color="C0", label="(p_v, |W_v cap X_phi|/r)")
pv_grid = np.linspace(1e-3, 1, 200)
ax.plot(pv_grid, pv_grid, color="black", lw=1.2, label="identity (theorem mean)")
eps_grid = np.sqrt(2 * np.log(1.0/delta) / np.maximum(R * pv_grid, 1e-6))
ax.plot(pv_grid, pv_grid * (1 + eps_grid), color="red", lw=1.0, ls="--",
        label=f"(1+eps) bound, delta={delta}")
ax.plot(pv_grid, np.maximum(0, pv_grid * (1 - eps_grid)), color="red", lw=1.0, ls="--")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("predicate selectivity p_v(phi) = |U_v cap X_phi| / |U_v|")
ax.set_ylabel("empirical |W_v cap X_phi| / r_v")
ax.set_title(f"ARWGI concentration on YFCC10M (r={R}, 200 held-out queries)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

# Panel B: bucketed mean rate vs mean p_v, with theorem envelope
bins = np.linspace(0, 1, 21)
digit = np.digitize(pv, bins) - 1
bucket_pv = []
bucket_rate = []
bucket_rate_std = []
bucket_n = []
for b in range(20):
    mask = digit == b
    if mask.sum() < 5:
        continue
    bucket_pv.append(pv[mask].mean())
    bucket_rate.append(rate[mask].mean())
    bucket_rate_std.append(rate[mask].std())
    bucket_n.append(mask.sum())
bucket_pv = np.array(bucket_pv); bucket_rate = np.array(bucket_rate)
bucket_rate_std = np.array(bucket_rate_std); bucket_n = np.array(bucket_n)

ax = axes[1]
ax.errorbar(bucket_pv, bucket_rate, yerr=bucket_rate_std, fmt="o", color="C0", capsize=3,
            label="bucketed mean +/- 1 std")
ax.plot(pv_grid, pv_grid, color="black", lw=1.2, label="identity")
ax.plot(pv_grid, pv_grid * (1 + eps_grid), color="red", lw=1.0, ls="--",
        label=f"(1+/-eps) bound, delta={delta}")
ax.plot(pv_grid, np.maximum(0, pv_grid * (1 - eps_grid)), color="red", lw=1.0, ls="--")
for i, n in enumerate(bucket_n):
    ax.annotate(f"n={n}", (bucket_pv[i], bucket_rate[i]), fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("predicate selectivity p_v(phi)")
ax.set_ylabel("empirical mean |W_v cap X_phi|/r")
ax.set_title(f"Bucketed concentration vs theorem envelope")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

png_path = OUT_DIR / f"arwgi_concentration_yfcc10m_r{R}.png"
pdf_path = OUT_DIR / f"arwgi_concentration_yfcc10m_r{R}.pdf"
plt.tight_layout()
plt.savefig(png_path, dpi=140)
plt.savefig(pdf_path)
print(f"wrote {png_path}")
print(f"wrote {pdf_path}")

# How many points lie within the (1+/-eps) envelope?
hi = pv * (1 + eps_envelope)
lo = np.maximum(0, pv * (1 - eps_envelope))
within = ((rate <= hi) & (rate >= lo)).mean()
# only count observations with pv >= 0.01 (low-pv buckets dominated by zero-count noise)
mask01 = pv >= 0.01
within01 = ((rate[mask01] <= hi[mask01]) & (rate[mask01] >= lo[mask01])).mean()
mask05 = pv >= 0.05
within05 = ((rate[mask05] <= hi[mask05]) & (rate[mask05] >= lo[mask05])).mean()
mask10 = pv >= 0.10
within10 = ((rate[mask10] <= hi[mask10]) & (rate[mask10] >= lo[mask10])).mean()

env_stats = {
    "delta": delta,
    "r": R,
    "n_observations": len(rows),
    "fraction_within_envelope_all": float(within),
    "fraction_within_envelope_pv_geq_0p01": float(within01),
    "fraction_within_envelope_pv_geq_0p05": float(within05),
    "fraction_within_envelope_pv_geq_0p10": float(within10),
    "abs_pearson_pv_vs_rate": float(np.corrcoef(pv, rate)[0, 1]),
}
env_path = OUT_DIR / f"arwgi_concentration_envelope_yfcc10m_r{R}.json"
env_path.write_text(json.dumps(env_stats, indent=2))
print(json.dumps(env_stats, indent=2))
