# External dependencies

This repository does not vendor baseline source. Obtain each system from its
authors and retain its original license.

| System | Repository | Revision used |
|---|---|---|
| ParlayANN / Parlay-IVF | <https://github.com/cmuparlay/ParlayANN> | `870dc7101ed796673c63ffd4c199a58936c8124c` |
| SIEVE | <https://github.com/BillyZhaohengLi/SIEVE-vldb25> | `9f128f4916ca2fc5cf839a292359050be85eab0f` |
| ACORN | <https://github.com/TAG-Research/ACORN> | `c259f11cdbec880671658eaa0d8747e2cc9de79d` |
| SeRF | <https://github.com/rutgers-db/SeRF> | `5666c2a461ed75b10b8b5aafb43b6eb884538ddf` |
| WoW | <https://github.com/nju-websoft/WoW> | `31de2f44ad344410ff6eddb535f508f25ae712a6` |

Boole-ANN consumes ParlayANN and SIEVE through explicit include roots. ACORN,
SeRF, and WoW are built independently with their released build systems.
Experiment adapters exchange only dataset paths, predicates, thread counts,
and returned global IDs.
