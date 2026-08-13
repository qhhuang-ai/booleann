#!/usr/bin/env python3
"""Analyze only the frozen mixed-q800 LAION1M v19 formal endpoint."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import run_laion1m_v19_same_resident_paired_formal as controller
from laion1m_paired_formal_block_adapter import (
    ACK_SCHEMA,
    RELEASE_SCHEMA,
    canonical_json_bytes,
    snapshot_nonempty_regular_file,
)


SCHEMA = "laion1m-v19-same-resident-paired-formal-analysis/v1"
SCHEDULE = "SHHSSHHSSHHSSHHS"
PAIRS = 8
BLOCKS = 16
PRIMARY_RATIO_MINIMUM = 1.10
P_MAXIMUM = 0.05


def fail(message: str) -> None:
    raise RuntimeError(message)


def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_exact_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=exact_object)
    if not isinstance(value, dict):
        fail(f"{path}: expected object")
    return value


def read_one_jsonl(path: Path) -> dict[str, Any]:
    before = snapshot_nonempty_regular_file(path, "formal block summary")
    payload = path.read_bytes()
    after = snapshot_nonempty_regular_file(path, "formal block summary")
    if before != after or len(payload) != before.bytes or not payload.endswith(b"\n"):
        fail("formal block summary is unstable or not newline terminated")
    lines = payload.splitlines()
    if len(lines) != 1:
        fail("formal block summary must contain one row")
    value = json.loads(lines[0].decode("ascii"), object_pairs_hook=exact_object)
    if not isinstance(value, dict):
        fail("formal block summary row is not an object")
    return value


def require_exact_keys(value: dict[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    if set(value) != expected:
        fail(f"{label} keys differ")


def validate_release(
    ack_directory: Path,
    state_id: str,
    block: int,
    summary_path: Path,
    results_path: Path,
) -> None:
    ack_path = ack_directory / f"block_{block:02d}.json"
    release_path = ack_directory / f"block_{block:02d}.release.json"
    summary = snapshot_nonempty_regular_file(summary_path, "formal block summary")
    results = snapshot_nonempty_regular_file(results_path, "formal block results")
    ack = read_exact_json(ack_path)
    require_exact_keys(
        ack,
        {"schema", "state_id", "pass", "passes_jsonl_bytes", "results_jsonl_bytes"},
        "block ACK",
    )
    if ack != {
        "schema": ACK_SCHEMA,
        "state_id": state_id,
        "pass": block,
        "passes_jsonl_bytes": summary.bytes,
        "results_jsonl_bytes": results.bytes,
    }:
        fail("block ACK does not bind exact durable file sizes")
    ack_payload = canonical_json_bytes(ack)
    if ack_path.read_bytes() != ack_payload:
        fail("block ACK is not canonical")
    release = read_exact_json(release_path)
    require_exact_keys(release, {"schema", "state_id", "pass", "ack_sha256"},
                       "block release")
    if release != {
        "schema": RELEASE_SCHEMA,
        "state_id": state_id,
        "pass": block,
        "ack_sha256": hashlib.sha256(ack_payload).hexdigest(),
    }:
        fail("block release does not bind the durable ACK")
    if release_path.read_bytes() != canonical_json_bytes(release):
        fail("block release is not canonical")


def validate_summary(row: dict[str, Any], state_id: str, block: int) -> float:
    require_exact_keys(row, controller.SUMMARY_FIELDS, "formal block summary")
    arm = SCHEDULE[block]
    expected_pair_order = "SH" if (block // 2) % 2 == 0 else "HS"
    if (
        row.get("schema") != controller.SUMMARY_SCHEMA
        or row.get("state_id") != state_id
        or row.get("block") != block
        or row.get("pair") != block // 2
        or row.get("pair_order") != expected_pair_order
        or row.get("arm") != arm
        or row.get("audit_after_wall") is not True
        or row.get("queries") != 800
        or row.get("frozen_ef") != 800
        or row.get("timed_worker_explicit_io_calls") != 0
        or row.get("validity_gate") is not True
        or row.get("recall_gate") is not True
        or row.get("block_gate") is not True
    ):
        fail(f"block {block}: identity, endpoint, or gate differs")
    for name in (
        "invalid", "predicate_fail", "duplicate", "forbidden_self",
        "underfull", "nondeterministic",
    ):
        if row.get(name) != 0:
            fail(f"block {block}: {name} is nonzero")
    if row.get("removed_self") != 800:
        fail(f"block {block}: leave-one-out census differs")
    wall_ns = row.get("wall_ns")
    qps = row.get("complete_batch_qps")
    if (
        not isinstance(wall_ns, int)
        or isinstance(wall_ns, bool)
        or wall_ns <= 0
        or not isinstance(qps, (int, float))
        or isinstance(qps, bool)
        or not math.isfinite(float(qps))
        or not math.isclose(float(qps), 800e9 / wall_ns, rel_tol=1e-12)
    ):
        fail(f"block {block}: complete-batch QPS arithmetic differs")
    return float(qps)


def pair_ratios(block_qps: list[float]) -> list[float]:
    if len(block_qps) != BLOCKS:
        fail("formal block QPS census differs")
    ratios: list[float] = []
    for pair in range(PAIRS):
        values = {
            SCHEDULE[2 * pair]: block_qps[2 * pair],
            SCHEDULE[2 * pair + 1]: block_qps[2 * pair + 1],
        }
        if set(values) != {"S", "H"}:
            fail("adjacent pair does not contain one H and one S")
        ratios.append(values["H"] / values["S"])
    return ratios


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        fail("paired ratio is not finite and positive")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def exact_sign_flip_p_values(ratios: list[float]) -> tuple[float, float]:
    logs = [math.log(value) for value in ratios]
    observed = math.fsum(logs)
    one_sided_extreme = 0
    two_sided_extreme = 0
    tolerance = 1e-15 * max(1.0, abs(observed))
    for signs in itertools.product((-1.0, 1.0), repeat=len(logs)):
        statistic = math.fsum(sign * value for sign, value in zip(signs, logs))
        one_sided_extreme += statistic >= observed - tolerance
        two_sided_extreme += abs(statistic) >= abs(observed) - tolerance
    denominator = 1 << len(logs)
    return one_sided_extreme / denominator, two_sided_extreme / denominator


def analyze(manifest_path: Path) -> dict[str, Any]:
    manifest = controller.parse_manifest(manifest_path.resolve(strict=True), activation=True)
    qps: list[float] = []
    for block in range(BLOCKS):
        summary_path = manifest.raw_directory / f"block_{block:02d}_summary.jsonl"
        results_path = manifest.raw_directory / f"block_{block:02d}_results.jsonl"
        validate_release(
            manifest.ack_directory, manifest.state_id, block,
            summary_path, results_path,
        )
        qps.append(validate_summary(read_one_jsonl(summary_path), manifest.state_id, block))
    ratios = pair_ratios(qps)
    gm = geometric_mean(ratios)
    one_sided_p, two_sided_p = exact_sign_flip_p_values(ratios)
    gate = gm >= PRIMARY_RATIO_MINIMUM and two_sided_p <= P_MAXIMUM
    return {
        "schema": SCHEMA,
        "status": "PASS" if gate else "STOP",
        "state_id": manifest.state_id,
        "performance_endpoint": "mixed_q800_complete_batch_wall_only",
        "family_qps_from_mixed_concurrency_computed": False,
        "paired_qps_ratios_h_over_s": ratios,
        "geometric_mean_qps_ratio_h_over_s": gm,
        "exact_one_sided_sign_flip_p": one_sided_p,
        "exact_two_sided_sign_flip_p": two_sided_p,
        "primary_report_p_value": "exact_two_sided_sign_flip_p",
        "sign_flip_assignments": 1 << PAIRS,
        "effect_threshold": PRIMARY_RATIO_MINIMUM,
        "p_threshold": P_MAXIMUM,
        "all_16_block_gates_pass": True,
        "all_16_ack_release_bindings_pass": True,
        "formal_endpoint_gate": gate,
    }


def write_create_only(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or not path.is_absolute() or not path.parent.is_dir():
        fail("analysis output must be a new absolute file in an existing directory")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def self_test() -> None:
    positive = [1.20] * PAIRS
    one_sided, two_sided = exact_sign_flip_p_values(positive)
    if (
        not math.isclose(geometric_mean(positive), 1.20, rel_tol=1e-15)
        or one_sided != 1 / 256
        or two_sided != 2 / 256
    ):
        fail("positive exact-sign-flip fixture differs")
    alternating_qps: list[float] = []
    for pair in range(PAIRS):
        if pair % 2 == 0:
            alternating_qps.extend((100.0, 120.0))  # S,H
        else:
            alternating_qps.extend((120.0, 100.0))  # H,S
    if any(not math.isclose(value, 1.20) for value in pair_ratios(alternating_qps)):
        fail("SH/HS adjacent-pair fixture differs")
    print(
        "SELF_TEST schema=laion1m-v19-formal-analysis/v1 status=PASS "
        "endpoint=mixed_q800_only pair_first=true assignments=256 "
        "all_positive_one_sided_p=0.00390625 "
        "all_positive_two_sided_p=0.0078125 family_qps_computed=false"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    if arguments.self_test:
        if arguments.manifest is not None or arguments.out is not None:
            fail("self-test accepts no manifest/output")
        self_test()
        return 0
    if arguments.manifest is None or arguments.out is None:
        fail("--manifest and --out are required")
    result = analyze(arguments.manifest)
    write_create_only(arguments.out.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["formal_endpoint_gate"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"STOP {error}", file=os.sys.stderr)
        raise SystemExit(2)
