# Reproducing the submitted experiments

## 1. Dependencies

- CMake 3.16 or newer and a C++17 compiler with AVX2/FMA support.
- Python 3.9 or newer with NumPy and SciPy.
- OpenSSL development headers for the hierarchical range executable.
- An unmodified ParlayANN checkout at revision
  `870dc7101ed796673c63ffd4c199a58936c8124c`.
- For the SIEVE comparison only, an unmodified SIEVE checkout at revision
  `9f128f4916ca2fc5cf839a292359050be85eab0f`.

Additional baseline revisions are listed in `EXTERNAL_DEPENDENCIES.md`.
Baseline repositories and their licenses remain separate from this repository.

## 2. Paper-to-code map

| Paper experiment | Boole-ANN implementation | External comparison |
|---|---|---|
| Natural tags and exact endpoints | `src/categorical/categorical_bench.cpp` | Parlay-IVF, ACORN |
| Predicate families against SIEVE | `src/predicate_cube/layout_selector_x3.cpp` | SIEVE |
| Range and two-clause DNF | `src/ordered/range_bench.cpp` | WoW, SeRF |
| Budgeted fragment admission | `src/advisor/fragment_advisor.py` | Oracle and stagewise controls |
| LAION mixed serving | `src/laion/` | Forced leaf/direct/fragment routes |
| Pair materialization | `src/categorical/sift100m_bench.cpp` | Pair-disabled control, Parlay-IVF |
| Support/layout/plan controls | `src/predicate_cube/mechanism_ablation.cpp` | Forced component controls |
| Hierarchical range layout | `src/ordered/hierarchical_range_bench.cpp` | beta-WST |

## 3. Input conventions

Full-scale commands accept paths to local dataset and index directories. The
default relative layout is:

```text
data/
  yfcc10m/
  sift10m/
  sift100m/
  laion1m/
indexes/
  yfcc10m/
  sift10m/
  sift100m/
outputs/
```

YFCC vectors use the BigANN `u8bin` container. SIFT vectors use the public
BIGANN containers named by each executable. Structured predicates are stored
as sorted postings or fixed-width integer ranges. Ground truth contains exact
top-10 global IDs and is evaluated as an unordered set for strict recall.

## 4. Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPARLAYANN_ROOT=/path/to/ParlayANN \
  -DSIEVE_ROOT=/path/to/SIEVE
cmake --build build -j
ctest --test-dir build --output-on-failure
```

No baseline source is copied or patched by this build. The two root variables
only add include paths to independent upstream checkouts.

Before publishing a release archive, verify that no external checkout or
symlink entered the tree:

```bash
scripts/check_release_boundaries.sh
```

## 5. Result checks

The compact record in `results/submission_results.json` contains the values
reported in the submitted tables. Check ratios, intervals, and recall bounds
with:

```bash
python scripts/verify_submission_results.py results/submission_results.json
```

The compact record is not a substitute for raw result logs. Large raw outputs
and derived indexes are kept outside Git because of their size; the public
release can attach them as a separate archival asset.
