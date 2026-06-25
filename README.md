# Boole-ANN: The Correlation Frontier of Filtered ANN

Reference implementation and reproducibility scripts for the **Boole-ANN** filtered
approximate nearest-neighbour framework. This bundle contains the code
required to reproduce our reported results: the ARWGI theorem deployment, the
HAMCG hot-atom subgraph, the BCI deployed system (selectivity-tiered dispatcher + PACH cluster cache + per-tag bitvecs + pair-posting cache), the DKW
calibration of $\varepsilon_{\mathrm{corr}}$ for hypothesis (H2), and the
$t$-disjunct group-testing sentinel validation for Theorem T-D.

Baseline systems (SIEVE, Parlay-IVF, UNG, iRangeGraph, Filtered-DiskANN, JAG,
ACORN, etc.) are **not included**; please obtain them from their original
upstream repositories if you wish to reproduce the baseline comparisons.

```
boole-ann-code/
├── README.md                          # this file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT
├── python/
│   ├── method/                        # core method (11 files)
│   │   ├── arwgi_proper_deploy.py     # T-1A faithful deployment + concentration audit
│   │   ├── arwgi_canonical_60_70.py   # canonical YFCC10M slice [60K,70K)
│   │   ├── arwgi_sift100m_deploy.py   # SIFT100M scale-out (HAMCG inner HNSW)
│   │   ├── arwgi_proper_plot.py       # diagnostic plots
│   │   ├── bci_exact_only_yfcc10m.py  # BCI deployed system on YFCC10M
│   │   ├── hamcg_sift1b_clean.py      # SIFT1B scale-out
│   │   ├── bci_hamcg_laion1m_richer.py# LAION1M conjunctions
│   │   ├── dkw_calibration_yfcc10m.py # DKW calibration of epsilon_corr (Lemma 3)
│   │   ├── sanity_check_td_sentinel.py# T-D group-testing sentinel validation
│   │   ├── pac_t1a_audit.py           # T-1A PAC audit (per-query envelope)
│   │   └── cert_e_bridge.py           # cert_e -> deployed proxy bridge
│   └── data_prep/                     # dataset preparation (7 files)
│       ├── gen_yfcc10m_gt_matched_v6.py  # YFCC10M filtered top-k ground truth
│       ├── prepare_sift100m_unified.py   # SIFT100M shard preparation
│       ├── sift100m_build_conjunction_data.py
│       ├── sift100m_gen_conjunction_gt.py
│       ├── laion_prepare.py
│       ├── laion_richer_filters.py
│       └── fit_yfcc10m_kmeans_torch.py   # k-means clustering (T-D row index)
├── cpp/                                # C++17 deeply-optimised port
│   ├── CMakeLists.txt
│   ├── src/                            # 12 C++ sources
│   │   ├── bci_main.cpp
│   │   ├── bci_bench.cpp               # YFCC10M / SIFT10M benchmark driver
│   │   ├── sift100m_bci_bench.cpp      # SIFT100M variant
│   │   ├── laion_bci_bench.cpp         # LAION1M variant
│   │   ├── build_hamcg_shards.cpp      # HAMCG per-atom subgraph build
│   │   ├── build_hamcg_shards_laion.cpp
│   │   ├── build_shard_clusters.cpp    # k-means clustering of each shard
│   │   ├── build_shard_clusters_laion.cpp
│   │   ├── extract_sweet_spot_tags.cpp # vocabulary selection
│   │   ├── extract_sweet_spot_tags_laion.cpp
│   │   ├── yfcc_probe.cpp              # YFCC label probe utility
│   │   └── query_shard_test.cpp        # shard query smoke test
│   └── include/                        # local headers (if any)
├── scripts/                            # one-shot runners
│   ├── 01_install.sh                   # pip install + build C++ port
│   ├── 02_download_datasets.sh         # YFCC10M / SIFT100M / SIFT1B / LAION1M URLs
│   ├── 10_reproduce_dkw_calibration.sh # epsilon_corr calibration (T-1A H2)
│   ├── 11_reproduce_yfcc10m.sh         # YFCC10M conjunction Pareto (Fig. 3)
│   ├── 12_reproduce_sift100m.sh        # SIFT100M scale-out (Table 2)
│   ├── 13_reproduce_sift1b.sh          # SIFT1B scale-out (Table 2)
│   ├── 14_reproduce_laion1m.sh         # LAION1M conjunctions
│   └── 15_reproduce_td_sentinel.sh     # T-D sentinel validation (Sec. 6.3)
└── data/
    └── README.md                       # dataset download instructions
```

---

## 1. System requirements

* Linux x86-64 (tested on AMD EPYC 7642, 96 cores, 251 GB RAM).
* GCC 9+ or clang 12+, CMake 3.16+, OpenMP.
* Python 3.9+ with the packages in `requirements.txt`.
* ~50 GB free disk for YFCC10M + SIFT100M, ~1.5 TB for SIFT1B.

The C++ port uses **parlaylib** (a parallel runtime library, Apache 2.0) for
work-stealing primitives. Install it via:

```bash
git clone https://github.com/cmuparlay/parlaylib.git /tmp/parlaylib
export PARLAYANN_ROOT=/tmp/parlaylib   # CMake reads this
```

(The `PARLAYANN_ROOT` CMake variable is historical; we only use `parlaylib`
headers from that tree.)

---

## 2. Quick start

```bash
# All scripts honour BOOLEANN_ROOT; default to the repo root.
export BOOLEANN_ROOT=$PWD

bash scripts/01_install.sh                 # Python deps + build cpp/
bash scripts/02_download_datasets.sh       # download YFCC10M (others gated)
bash scripts/10_reproduce_dkw_calibration.sh
bash scripts/11_reproduce_yfcc10m.sh       # ~3 hours on EPYC 7642 96-core
```

Per-run artefacts are written under
`$BOOLEANN_ROOT/03_experiment_bridge/results/raw/`. Each Python entrypoint
picks up data from `$BOOLEANN_ROOT/data/raw/` (overridable via the same
env var); module-level constants in the scripts (`NB`, `DIM`, `M`,
`EFC`, `EFS`, etc.) control non-CLI parameters --- see the docstring at
the top of each file.

---

## 3. Mapping from results to scripts

| Paper result                                              | Driver                                      | Output |
|-----------------------------------------------------------|---------------------------------------------|--------|
| Fig. 3 (BCI vs Parlay-IVF, 15/15 cross-slice cells)        | `scripts/11_reproduce_yfcc10m.sh`           | `results/yfcc10m/cross_slice.json` |
| Sec. 6.3, T-D sentinel (t in {8,16,24})                    | `scripts/15_reproduce_td_sentinel.sh`       | `results/sift1m/td_sentinel.json` |
| Sec. 6.4, $\varepsilon_{\mathrm{corr}}=0.075$ at $m=594$    | `scripts/10_reproduce_dkw_calibration.sh`   | `results/dkw_calibration_yfcc10m/summary.json` |
| Table 2, SIFT100M 0.999@3.6 ms                             | `scripts/12_reproduce_sift100m.sh`          | `results/sift100m/bci.json` |
| Table 2, SIFT1B 0.995@5.7 ms                               | `scripts/13_reproduce_sift1b.sh`            | `results/sift1b/bci.json` |
| LAION1M 5.30x (text only, supplemental measurement)         | `scripts/14_reproduce_laion1m.sh`           | `results/laion1m/bci.json` |
| T-1A above-frontier deployment (Sec. 6.2)                  | `python python/method/arwgi_canonical_60_70.py` | `results/arwgi/canonical_60_70.json` |

---

## 4. Hyperparameter defaults (reproduce paper numbers)

All defaults match the paper's canonical configuration; no per-slice retuning.

* **HNSW** (graph index for ARWGI and HAMCG inner search): `M = 32`,
  `ef_construction = 200`, `ef_search = 200` (calibration) / `400` (HAMCG inner).
* **HAMCG vocabulary** (per-atom subgraph): edge budget
  `B = 3.2e8` for YFCC10M (~2747 atoms selected),
  `B = 1.0e8` for LAION1M (top-~400 tags).
* **T-D sentinel** (Sec. 6.3): `C = 16384` k-means cells, `R = 32` candidate
  cells, sparsity bound `t in {8, 16, 24}` (so `R >= t + 1` holds for Thm. 3).
* **BCI dispatcher** (CAPD, Sec. 3.4 / 4.3): canonical slice config
  `beam = 4096`, `cut = 1.35`, `tau = 5e5`, plus per-tag bitvecs + pair-posting
  two-tag cache.
* **DKW calibration** (Lemma 3): `m = 594` calibration queries,
  `delta_cal = delta_corr = 0.05`, yields
  `epsilon_corr <= 0.075` with probability `>= 0.95`.

---

## 5. License

This bundle is released under the MIT License (see `LICENSE`). External
dependencies retain their own licenses.

---

## 6. Authors

* Qionghao Huang &lt;qhhuang@zjnu.edu.cn&gt;, Zhejiang Normal University, Jinhua, China
* Feiyang Shu &lt;feiyangshu@zjnu.edu.cn&gt;, Zhejiang Normal University, Jinhua, China
* Changqin Huang &lt;cqhuang@zju.edu.cn&gt;, Zhejiang University, Hangzhou, China

## 7. Citation

```bibtex
@article{huang2026booleann,
  title  = {Boole-ANN: The Correlation Frontier of Filtered ANN},
  author = {Huang, Qionghao and Shu, Feiyang and Huang, Changqin},
  year   = {2026}
}
```
