# Boole-ANN

This repository contains the implementation and compact result record for
**Boole-ANN: Joint Physical Design for Structured Filtered Vector Search**.
Boole-ANN selects reusable predicate-support fragments, their physical
layouts, and recall-eligible execution plans under an explicit byte budget.

The repository snapshot corresponds to the submitted paper in
[`paper/boole-ann.pdf`](paper/boole-ann.pdf).

## Repository layout

| Path | Contents |
|---|---|
| `src/advisor` | Resource-bounded fragment and plan advisor |
| `src/categorical` | Equality and conjunction engine used for YFCC10M and SIFT100M |
| `src/ordered` | Fixed-block range, DNF, and hierarchical range executors |
| `src/predicate_cube` | Contiguous Predicate-Cube executor and component controls |
| `src/laion` | U24/Delta16 support layouts, fragment composition, and exact leaf merge |
| `scripts` | Submission-result consistency checks |
| `results` | Compact values reported in the submitted paper |
| `EXTERNAL_DEPENDENCIES.md` | Upstream locations and revisions; no baseline source is vendored |

## Quick check

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/advisor/fragment_advisor.py --self-test
python src/advisor/fragment_advisor.py \
  examples/advisor_toy.json \
  --output outputs/advisor_toy_solution.json \
  --mip-gap 0
python scripts/verify_submission_results.py results/submission_results.json
scripts/check_release_boundaries.sh
```

The native LAION kernels have no external baseline dependency:

```bash
cmake -S . -B build
cmake --build build -j --target \
  fixed_block_compositor packed_fixed_block_compositor leaf_posting_merge
```

To build the categorical and ordered engines, provide an unmodified ParlayANN
checkout. To build Predicate-Cube, also provide an unmodified SIEVE checkout:

```bash
cmake -S . -B build \
  -DPARLAYANN_ROOT=/path/to/ParlayANN \
  -DSIEVE_ROOT=/path/to/SIEVE
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The repository does not redistribute datasets, indexes, or baseline source.
See [`REPRODUCING.md`](REPRODUCING.md) for input formats and experiment mapping.

## License

Boole-ANN is released under the [MIT License](LICENSE). External systems and
datasets remain subject to their respective upstream licenses.

## Baseline policy

Baseline implementations are obtained from their authors' repositories and
executed at the revisions listed in
[`EXTERNAL_DEPENDENCIES.md`](EXTERNAL_DEPENDENCIES.md). They are not copied,
submoduled, patched, or redistributed by this repository. Boole-ANN adapters
translate common vectors, predicates, thread counts, and returned IDs while
calling unmodified upstream checkouts through their public interfaces.

## Artifact availability

The paper and compact result record are included. Large public datasets and
derived indexes are intentionally excluded from Git. Full-scale runs write to
user-supplied `data/`, `indexes/`, and `outputs/` directories, all ignored by
Git.
