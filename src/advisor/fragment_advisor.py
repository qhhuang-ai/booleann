#!/usr/bin/env python3
"""Resource-bounded fragment/layout advisor with a certified MILP gap.

Input JSON contains candidate physical units and calibration-only per-query
latency savings at a fixed recall constraint.  A selected unit can serve any
number of queries, while each query is assigned to at most one selected unit.
The objective is the weighted maximum realised saving under one byte budget.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(instance: dict[str, Any]) -> None:
    if int(instance["budget_bytes"]) < 0:
        raise ValueError("budget_bytes must be nonnegative")
    units = instance["units"]
    queries = instance["queries"]
    unit_ids = [str(unit["id"]) for unit in units]
    query_ids = [str(query["id"]) for query in queries]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("duplicate unit id")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query id")
    known = set(unit_ids)
    for unit in units:
        if int(unit["cost_bytes"]) <= 0:
            raise ValueError(f"unit {unit['id']} must have positive byte cost")
    for query in queries:
        if float(query.get("weight", 1.0)) < 0:
            raise ValueError(f"query {query['id']} has negative weight")
        for unit_id, saving in query.get("saving_us", {}).items():
            if unit_id not in known:
                raise ValueError(f"query {query['id']} references unknown unit {unit_id}")
            if float(saving) < 0:
                raise ValueError(f"query {query['id']} has negative saving for {unit_id}")


def solve(instance: dict[str, Any], time_limit: float, mip_gap: float) -> dict[str, Any]:
    validate(instance)
    units = instance["units"]
    queries = instance["queries"]
    n_units = len(units)
    unit_pos = {str(unit["id"]): index for index, unit in enumerate(units)}

    # Only positive-benefit assignment edges are variables.  x_u occupies
    # [0,n_units); z_(q,u) follows.  All variables are binary.
    edges: list[tuple[int, int, float]] = []
    for q_index, query in enumerate(queries):
        weight = float(query.get("weight", 1.0))
        for unit_id, raw_saving in query.get("saving_us", {}).items():
            benefit = weight * float(raw_saving)
            if benefit > 0:
                edges.append((q_index, unit_pos[str(unit_id)], benefit))
    n_vars = n_units + len(edges)
    objective = np.zeros(n_vars, dtype=np.float64)
    for edge_index, (_q, _u, benefit) in enumerate(edges):
        objective[n_units + edge_index] = -benefit

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    # One global resource constraint.
    for unit_index, unit in enumerate(units):
        row_indices.append(0)
        col_indices.append(unit_index)
        data.append(float(unit["cost_bytes"]))
    lower.append(-np.inf)
    upper.append(float(instance["budget_bytes"]))
    next_row = 1

    # Each query chooses at most one physical alternative.
    edges_by_query: list[list[int]] = [[] for _ in queries]
    edges_by_unit: list[list[int]] = [[] for _ in units]
    for edge_index, (q_index, unit_index, _benefit) in enumerate(edges):
        edges_by_query[q_index].append(edge_index)
        edges_by_unit[unit_index].append(edge_index)
    for query_edges in edges_by_query:
        if not query_edges:
            continue
        for edge_index in query_edges:
            row_indices.append(next_row)
            col_indices.append(n_units + edge_index)
            data.append(1.0)
        lower.append(-np.inf)
        upper.append(1.0)
        next_row += 1

    # Assignment implies physical admission: z_(q,u) <= x_u.
    for edge_index, (_q_index, unit_index, _benefit) in enumerate(edges):
        row_indices.extend((next_row, next_row))
        col_indices.extend((n_units + edge_index, unit_index))
        data.extend((1.0, -1.0))
        lower.append(-np.inf)
        upper.append(0.0)
        next_row += 1

    # Do not admit a unit unless at least one query uses it: x_u <= sum_q z_qu.
    for unit_index, unit_edges in enumerate(edges_by_unit):
        if not unit_edges:
            # x_u <= 0 for a candidate with no positive benefit.
            row_indices.append(next_row)
            col_indices.append(unit_index)
            data.append(1.0)
        else:
            row_indices.append(next_row)
            col_indices.append(unit_index)
            data.append(1.0)
            for edge_index in unit_edges:
                row_indices.append(next_row)
                col_indices.append(n_units + edge_index)
                data.append(-1.0)
        lower.append(-np.inf)
        upper.append(0.0)
        next_row += 1

    matrix = coo_matrix(
        (np.asarray(data), (np.asarray(row_indices), np.asarray(col_indices))),
        shape=(next_row, n_vars),
    ).tocsr()
    result = milp(
        c=objective,
        integrality=np.ones(n_vars, dtype=np.uint8),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"MILP produced no incumbent: status={result.status} {result.message}")

    selected_positions = [index for index in range(n_units) if result.x[index] > 0.5]
    assignments = []
    for edge_index, (q_index, unit_index, benefit) in enumerate(edges):
        if result.x[n_units + edge_index] > 0.5:
            assignments.append({
                "query_id": str(queries[q_index]["id"]),
                "unit_id": str(units[unit_index]["id"]),
                "weighted_saving_us": benefit,
            })
    selected = [units[index] for index in selected_positions]
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective_weighted_saving_us": -float(result.fun),
        "dual_bound_weighted_saving_us": -float(result.mip_dual_bound),
        "certified_relative_gap": float(result.mip_gap),
        "mip_node_count": int(result.mip_node_count),
        "budget_bytes": int(instance["budget_bytes"]),
        "selected_bytes": sum(int(unit["cost_bytes"]) for unit in selected),
        "selected_unit_ids": [str(unit["id"]) for unit in selected],
        "assignments": assignments,
        "n_units": n_units,
        "n_queries": len(queries),
        "n_assignment_edges": len(edges),
    }


def brute_force(instance: dict[str, Any]) -> tuple[float, set[str]]:
    """Tiny-instance oracle used by --self-test."""
    units = instance["units"]
    best_value = -1.0
    best_ids: set[str] = set()
    for mask in range(1 << len(units)):
        selected = {str(units[index]["id"]) for index in range(len(units)) if mask & (1 << index)}
        cost = sum(int(unit["cost_bytes"]) for unit in units if str(unit["id"]) in selected)
        if cost > int(instance["budget_bytes"]):
            continue
        value = 0.0
        for query in instance["queries"]:
            value += float(query.get("weight", 1.0)) * max(
                [float(saving) for unit_id, saving in query.get("saving_us", {}).items()
                 if unit_id in selected] or [0.0]
            )
        if value > best_value:
            best_value, best_ids = value, selected
    return best_value, best_ids


def self_test() -> None:
    instance = {
        "budget_bytes": 7,
        "units": [
            {"id": "a", "cost_bytes": 3},
            {"id": "b", "cost_bytes": 4},
            {"id": "c", "cost_bytes": 5},
        ],
        "queries": [
            {"id": "q0", "weight": 2, "saving_us": {"a": 5, "b": 7}},
            {"id": "q1", "weight": 1, "saving_us": {"a": 3, "c": 20}},
            {"id": "q2", "weight": 1, "saving_us": {"b": 8, "c": 8}},
        ],
    }
    expected_value, _expected_ids = brute_force(instance)
    actual = solve(instance, time_limit=30, mip_gap=0.0)
    if not np.isclose(actual["objective_weighted_saving_us"], expected_value):
        raise AssertionError((actual, expected_value))
    # Exhaustively test deterministic small variations against brute force.
    for budget, scale in itertools.product(range(0, 10), (0.5, 1.0, 2.0)):
        variant = json.loads(json.dumps(instance))
        variant["budget_bytes"] = budget
        variant["queries"][1]["saving_us"]["c"] *= scale
        expected, _ = brute_force(variant)
        got = solve(variant, time_limit=30, mip_gap=0.0)
        if not np.isclose(got["objective_weighted_saving_us"], expected):
            raise AssertionError((budget, scale, got, expected))
    print("fragment_advisor self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.instance is None or args.output is None:
        parser.error("instance and --output are required unless --self-test is used")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite append-only output: {args.output}")
    instance = json.loads(args.instance.read_text())
    result = solve(instance, args.time_limit, args.mip_gap)
    result["instance_path"] = str(args.instance)
    result["instance_sha256"] = sha256(args.instance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), **result}, indent=2))


if __name__ == "__main__":
    main()
