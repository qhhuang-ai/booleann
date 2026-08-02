# Source provenance

All files under `src/` are Boole-ANN implementation, adapters, or experiment
drivers maintained for this project. No baseline checkout, source file, patch,
generated amalgamation, or submodule is present in this release directory.

External APIs referenced by include name are:

- ParlayANN headers under `parlay/`, `utils/`, and `vamana/`;
- SIEVE's `partitioned_hnsw.h`.

Those headers are resolved only from user-supplied upstream checkouts through
the `PARLAYANN_ROOT` and `SIEVE_ROOT` CMake variables. Their source and licenses
remain in the respective upstream repositories.

The release-boundary check also rejects nested repositories, symlinks, common
baseline directory names, patch files, and upstream license files.
