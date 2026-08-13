#!/usr/bin/env python3
"""Check arithmetic and invariants in the compact submission result record."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def close(actual: float, expected: float, tolerance: float = 0.002) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual} differs from {expected}")


def validate(document: dict[str, Any]) -> None:
    if document.get("schema") != "booleann-submission-results-v1":
        raise ValueError("unexpected result schema")
    experiments = document["experiments"]

    for name in ("yfcc10m_high_recall", "yfcc10m_exact", "laion1m_mixed"):
        row = experiments[name]
        close(float(row["boole_qps"]) / float(row.get("baseline_qps", row.get("leaf_qps"))),
              float(row["ratio"]))

    hierarchy = experiments["sift100k_hierarchy"]
    close(float(hierarchy["hierarchy_qps"]) / float(hierarchy["baseline_qps"]),
          float(hierarchy["ratio"]))

    complete = experiments["laion1m_complete_support_owner"]
    if int(complete["distinct_query_roles"]) != (
        int(complete["predicates"]) * int(complete["blocks"])
    ):
        raise AssertionError("LAION complete-support main-role multiplicity drift")
    if int(complete["timed_records"]) != 4 * int(complete["distinct_query_roles"]):
        raise AssertionError("LAION complete-support period multiplicity drift")
    if int(complete["unexposed_fallback_roles"]) != (
        int(complete["unexposed_fallback_formulas"]) * int(complete["blocks"])
    ):
        raise AssertionError("LAION fallback-role multiplicity drift")
    if max(int(complete["owner_optional_bytes"]),
           int(complete["control_optional_bytes"])) > int(complete["optional_byte_cap"]):
        raise AssertionError("LAION complete-support optional-state cap exceeded")
    lower, upper = map(float, complete["block_ratio_range"])
    effect = float(complete["control_over_owner_qps_ratio"])
    if not lower <= effect <= upper:
        raise AssertionError("LAION complete-support effect is outside block range")
    if float(complete["one_sided_exact_p"]) != 1.0 / 64.0:
        raise AssertionError("LAION complete-support exact p-value drift")
    for key in ("owner_tie_aware_recall", "control_tie_aware_recall",
                "fallback_tie_aware_recall"):
        if float(complete[key]) != 1.0:
            raise AssertionError(f"{key} must remain one")

    sieve = experiments["laion1m_vs_sieve_mixed"]
    sieve_lower, sieve_upper = map(float, sieve["block_ratio_range"])
    if not sieve_lower <= float(sieve["ratio"]) <= sieve_upper:
        raise AssertionError("LAION/SIEVE ratio is outside its block range")
    if int(sieve["fresh_process_pairs"]) != 8:
        raise AssertionError("LAION/SIEVE process-pair count drift")
    if float(sieve["two_sided_exact_p"]) != 1.0 / 128.0:
        raise AssertionError("LAION/SIEVE exact p-value drift")
    if int(sieve["system_executions"]) != 12800:
        raise AssertionError("LAION/SIEVE execution census drift")
    if not sieve["id_and_distance_bits_exact"] or not sieve["resource_gate"]:
        raise AssertionError("LAION/SIEVE correctness or resource gate failed")

    drift = experiments["laion1m_vocabulary_drift_control"]
    close(float(drift["boole_qps"]) / float(drift["leaf_qps"]),
          float(drift["ratio"]))
    if (
        int(drift["fresh_process_pairs"]) != 8
        or int(drift["system_executions"]) != 51200
        or int(drift["predecessor_disjoint_non_equality_requests_per_arm_block"])
        != 2400
        or int(drift["selected_equality_requests_per_treatment_block"]) != 582
        or not drift["id_and_distance_bits_exact"]
    ):
        raise AssertionError("LAION vocabulary-drift contract differs")

    frontier = experiments["sift100m_pivf_frontier"]
    close(float(frontier["boole_low"]["qps"]) /
          float(frontier["pivf_low"]["qps"]),
          float(frontier["low_over_low_ratio"]))
    if float(frontier["boole_low"]["recall"]) <= float(frontier["pivf_high"]["recall"]):
        raise AssertionError("SIFT100M frontier recall dominance drift")
    if (
        int(frontier["queries"]) != 3248
        or int(frontier["predicate_classes"]) != 58
        or int(frontier["fresh_processes_per_cell"]) != 5
        or float(frontier["minimum_cross_tier_lower95_family9"]) <= 1.2
        or not frontier["all_validity_gates"]
    ):
        raise AssertionError("SIFT100M frontier contract differs")

    mechanism = experiments["sift100m_exact_mechanism"]
    close(float(mechanism["pair_over_atom_ratio"]) *
          float(mechanism["packed_over_pair_ratio"]),
          float(mechanism["combined_ratio"]))
    if (
        int(mechanism["fresh_processes"]) != 15
        or int(mechanism["passes"]) != 90
        or int(mechanism["exact_rank_rows"]) != 2923200
        or float(mechanism["combined_lower95"]) <= 2.0
        or not mechanism["all_validity_gates"]
    ):
        raise AssertionError("SIFT100M exact-mechanism contract differs")

    for name, row in experiments.items():
        for key in ("recall", "boole_recall", "baseline_recall"):
            if key in row and not 0.0 <= float(row[key]) <= 1.0:
                raise AssertionError(f"{name}.{key} is outside [0,1]")
        if "ci95" in row:
            lower, upper = map(float, row["ci95"])
            center = float(row.get("ratio", row.get("throughput_ratio")))
            if not lower <= center <= upper:
                raise AssertionError(f"{name} ratio is outside its interval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=Path)
    args = parser.parse_args()
    document = json.loads(args.result_file.read_text())
    validate(document)
    print(f"submission result check: PASS ({len(document['experiments'])} experiments)")


if __name__ == "__main__":
    main()
