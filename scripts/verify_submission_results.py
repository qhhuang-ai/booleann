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
