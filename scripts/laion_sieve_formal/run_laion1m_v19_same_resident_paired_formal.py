#!/usr/bin/env python3
"""Prospective controller for the LAION1M v19 paired formal runner.

The controller consumes, but never creates, a later frozen manifest.  During
activation it validates every post-wall block payload, durably publishes the
existing LAION ACK and then its bound release, and never authorizes block N+1
before block N's exact raw/summary bytes have crossed that boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

from laion1m_paired_formal_block_adapter import (
    FileSnapshot,
    publish_block_ack,
    publish_block_release,
    snapshot_nonempty_regular_file,
    validate_state_id,
)


MANIFEST_SCHEMA = "laion1m-v19-same-resident-paired-formal-manifest/v1"
SYSTEM_FREEZE_SCHEMA = "laion1m-equality-local-graph-v19-system-freeze/v1"
POOL_SCHEMA = "laion1m-equality-local-graph-v19-pool/v1"
SCIENCE_SCHEMA = "laion1m-equality-local-graph-v19-science/v1"
QUERY_SCHEMA = "laion1m-v19-formal-query-audit/v1"
SUMMARY_SCHEMA = "laion1m-v19-formal-block-summary/v1"
CONTROLLER_SCHEMA = "laion1m_v19_formal_controller_v1"
PAIRS = 8
BLOCKS = 16
LANES = 8
K = 10
N = 1_000_000
SCHEDULE = "SHHSSHHSSHHSSHHS"
FAMILIES = ("equality", "conjunction", "range", "dnf2")
H_EXACT_FAMILIES = ("range", "dnf2")
REQUIRE_EXACT_FAMILY_BINDING = False
BACKENDS = {
    ("S", "equality"): "official_sieve_saturated_1m",
    ("S", "conjunction"): "official_sieve_saturated_1m",
    ("S", "range"): "official_sieve_saturated_1m",
    ("S", "dnf2"): "official_sieve_saturated_1m",
    ("H", "equality"): "clean_local_graph_r16",
    ("H", "conjunction"): "official_sieve_reference_100k",
    ("H", "range"): "complete_support_exact_topk",
    ("H", "dnf2"): "complete_support_exact_topk",
}
FROZEN_RUNNER_VALUES = {
    "frozen-ef": 800,
    "timed-per-family": 200,
    "minimum-recall-numerator": 9995,
    "minimum-recall-denominator": 10000,
    "expected-reference-partitions": 18,
    "expected-reference-memberships": 109757,
    "expected-saturated-partitions": 42,
    "expected-saturated-memberships": 167220,
}
COMMIT_SIGNALS = frozenset((signal.SIGINT, signal.SIGTERM))
PAIR1_MIN_H_OVER_S = 0.90
FIRST4_PAIR_GM_MIN_H_OVER_S = 1.00
FINAL8_PAIR_GM_MIN_H_OVER_S = 1.10
FINAL8_ONE_SIDED_P_MAX = 0.05
FINAL8_TWO_SIDED_P_MAX = 0.05
SUMMARY_FIELDS = {
    "arm",
    "audit_after_wall",
    "block",
    "block_gate",
    "complete_batch_qps",
    "duplicate",
    "forbidden_self",
    "frozen_ef",
    "graph_beam",
    "graph_cut_milli",
    "graph_pool",
    "h_sieve_routes",
    "invalid",
    "max_child_peak_rss_kib",
    "max_child_rss_kib",
    "nondeterministic",
    "pair",
    "pair_order",
    "predicate_fail",
    "queries",
    "recall_gate",
    "removed_self",
    "schema",
    "state_id",
    "strict_denominator",
    "strict_hits",
    "tie_denominator",
    "tie_hits",
    "timed_worker_explicit_io_calls",
    "underfull",
    "validity_gate",
    "wall_ns",
}

RUNNER_ARGUMENTS = (
    "base",
    "spmat",
    "endpoints",
    "history",
    "queries",
    "graph-root",
    "numeric-similarity",
    "numeric-original-width",
    "numeric-original-height",
    "raw-directory",
    "ack-directory",
    "cpu-list",
    "frozen-ef",
    "timed-per-family",
    "query-permutation-seed",
    "query-permutation-fingerprint",
    "minimum-recall-numerator",
    "minimum-recall-denominator",
    "expected-reference-partitions",
    "expected-reference-memberships",
    "expected-saturated-partitions",
    "expected-saturated-memberships",
    "expected-graph-file-bytes",
    "expected-graph-subset-bytes",
    "expected-graph-dense-bytes",
    "expected-graph-memberships",
    "graph-authority-persistent-bytes",
    "graph-authority-runtime-bytes",
    "ack-timeout-seconds",
    "ack-poll-milliseconds",
)
PATH_ARGUMENTS = {
    "base",
    "spmat",
    "endpoints",
    "history",
    "queries",
    "graph-root",
    "numeric-similarity",
    "numeric-original-width",
    "numeric-original-height",
    "raw-directory",
    "ack-directory",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=exact_object)
    if not isinstance(value, dict):
        fail("formal manifest must be a JSON object")
    return value


def require_exact_keys(value: dict[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    observed = set(value)
    if observed != expected:
        fail(
            f"{label} keys differ; missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        fail(f"{label} must be a positive integer")
    return value


def nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        fail(f"{label} must be a nonnegative integer")
    return value


def validate_frozen_runner_arguments(arguments: dict[str, Any]) -> None:
    """Reject drift from the single formal q800/census/recall contract."""

    for name, expected in FROZEN_RUNNER_VALUES.items():
        if arguments.get(name) != expected:
            fail(f"runner argument {name} must equal frozen value {expected}")


def parse_worker_cpus(value: Any) -> tuple[int, ...]:
    if not isinstance(value, str):
        fail("cpu-list must be a comma-separated string")
    raw_values = value.split(",")
    if len(raw_values) != LANES or len(set(raw_values)) != LANES:
        fail("cpu-list must contain eight unique CPUs")
    if any(not item.isascii() or not item.isdecimal() for item in raw_values):
        fail("cpu-list contains a non-decimal CPU")
    return tuple(int(item) for item in raw_values)


def validate_cpu_partition(
    controller_cpu: int,
    worker_cpus: tuple[int, ...],
    *,
    allowed_cpus: set[int] | None = None,
) -> None:
    nonnegative_int(controller_cpu, "controller_cpu")
    if controller_cpu in worker_cpus:
        fail("controller_cpu must be outside the eight-worker CPU set")
    if allowed_cpus is not None:
        required = set(worker_cpus) | {controller_cpu}
        if not required.issubset(allowed_cpus):
            fail("controller/worker CPU partition is outside process affinity")


class SignalMaskedCommitLock:
    """A non-reentrant commit lock whose holder cannot run stop handlers.

    Python dispatches SIGINT/SIGTERM handlers on the main thread.  Blocking
    those signals before acquiring the ordinary lock and restoring the mask
    only after releasing it prevents a handler from re-entering the same lock.
    The under-lock stop check and the publication link are therefore ordered.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()

    def __enter__(self) -> "SignalMaskedCommitLock":
        if getattr(self._local, "active", False):
            raise RuntimeError("commit lock is not reentrant")
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, COMMIT_SIGNALS)
        self._local.active = True
        self._local.previous_mask = previous
        try:
            self._lock.acquire()
        except BaseException:
            self._local.active = False
            del self._local.previous_mask
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            raise
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> bool:
        previous = self._local.previous_mask
        self._lock.release()
        self._local.active = False
        del self._local.previous_mask
        # Any pending stop handler runs only here, after the ordinary lock has
        # been released, so request_stop cannot self-deadlock.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        return False


@dataclass(frozen=True)
class Manifest:
    path: Path
    state_id: str
    runner_binary: Path
    runner_bytes: int
    runner_mtime_ns: int
    status_log: Path
    controller_cpu: int
    system_freeze_path: Path
    system_freeze_id: str
    frozen_pool_path: Path
    arguments: dict[str, str | int]

    @property
    def raw_directory(self) -> Path:
        return Path(str(self.arguments["raw-directory"]))

    @property
    def ack_directory(self) -> Path:
        return Path(str(self.arguments["ack-directory"]))

    @property
    def timed_per_family(self) -> int:
        return int(self.arguments["timed-per-family"])

    @property
    def frozen_ef(self) -> int:
        return int(self.arguments["frozen-ef"])

    @property
    def minimum_recall(self) -> tuple[int, int]:
        return (
            int(self.arguments["minimum-recall-numerator"]),
            int(self.arguments["minimum-recall-denominator"]),
        )


def read_stat_pinned_json(
    path: Path,
    *,
    expected_bytes: int,
    expected_mtime_ns: int,
    label: str,
) -> dict[str, Any]:
    """Read a small control document under an ordinary immutable stat pin."""

    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    before = snapshot_nonempty_regular_file(path, label)
    if before.bytes != expected_bytes or before.modified_ns != expected_mtime_ns:
        fail(f"{label} ordinary stat pin differs")
    value = read_json_file(path)
    after = snapshot_nonempty_regular_file(path, label)
    if after != before:
        fail(f"{label} changed while being read")
    return value


def validate_activation_documents(
    *,
    system: dict[str, Any],
    pool: dict[str, Any],
    science: dict[str, Any],
    system_freeze_path: Path,
    system_freeze_id: str,
    runner_arguments: dict[str, Any],
    controller_cpu: int,
    runner_binary: Path,
    runner_bytes: int,
    runner_mtime_ns: int,
) -> None:
    """Cross-bind the query-free system, pool, science, and runner contract."""

    if (
        system.get("schema") != SYSTEM_FREEZE_SCHEMA
        or system.get("freeze_id") != system_freeze_id
        or system.get("formal_execution_authorized") is not False
    ):
        fail("system-freeze schema or freeze_id differs from manifest")
    execution = system.get("execution")
    statistics = system.get("statistical_contract")
    census = system.get("sieve_census")
    inputs = system.get("inputs")
    graph = system.get("graph")
    complete_pins = system.get("complete_file_stat_pins")
    expected_runner_pin = {
        "path": str(runner_binary.resolve()),
        "bytes": runner_bytes,
        "mtime_ns": runner_mtime_ns,
    }
    if (
        not isinstance(complete_pins, list)
        or sum(pin == expected_runner_pin for pin in complete_pins) != 1
    ):
        fail("system-freeze runner ordinary stat pin differs from manifest")
    expected_workers = list(parse_worker_cpus(runner_arguments["cpu-list"]))
    if (
        not isinstance(execution, dict)
        or execution.get("controller_cpu") != controller_cpu
        or execution.get("cpus") != expected_workers
        or execution.get("lanes") != LANES
        or execution.get("schedule") != SCHEDULE
        or execution.get("controller_cpu_disjoint_from_worker_cpus") is not True
    ):
        fail("system-freeze execution/CPU contract differs from manifest")
    if (
        not isinstance(statistics, dict)
        or statistics.get("timed_per_family") != 200
        or statistics.get("strict_and_tie_recall_floor")
        != {"numerator": 9995, "denominator": 10000}
        or statistics.get("primary_geomean_qps_ratio_minimum")
        != FINAL8_PAIR_GM_MIN_H_OVER_S
        or statistics.get("only_performance_endpoint")
        != "mixed_q800_complete_batch_wall"
        or statistics.get("family_gates")
        != "strict_tie_recall_and_validity_only"
        or statistics.get("family_qps_from_mixed_concurrency_prohibited") is not True
        or statistics.get("one_sided_p_maximum") != FINAL8_ONE_SIDED_P_MAX
        or statistics.get("two_sided_p_maximum") != FINAL8_TWO_SIDED_P_MAX
        or statistics.get("primary_report_uses_two_sided_p") is not True
        or (
            REQUIRE_EXACT_FAMILY_BINDING
            and statistics.get("h_exact_families") != list(H_EXACT_FAMILIES)
        )
    ):
        fail("system-freeze mixed-q800 statistical contract differs")
    if (
        not isinstance(census, dict)
        or census.get("reference")
        != {
            "partitions": FROZEN_RUNNER_VALUES["expected-reference-partitions"],
            "memberships": FROZEN_RUNNER_VALUES["expected-reference-memberships"],
        }
        or census.get("saturated")
        != {
            "partitions": FROZEN_RUNNER_VALUES["expected-saturated-partitions"],
            "memberships": FROZEN_RUNNER_VALUES["expected-saturated-memberships"],
        }
    ):
        fail("system-freeze SIEVE census differs from runner contract")
    expected_input_arguments = {
        "base": "base",
        "spmat": "spmat",
        "endpoints": "endpoints",
        "history": "history",
        "numeric-similarity": "numeric_similarity",
        "numeric-original-width": "numeric_original_width",
        "numeric-original-height": "numeric_original_height",
    }
    if not isinstance(inputs, dict) or any(
        Path(str(runner_arguments[argument])).resolve()
        != Path(str(inputs.get(system_name))).resolve()
        for argument, system_name in expected_input_arguments.items()
    ):
        fail("system-freeze input paths differ from runner contract")
    if (
        not isinstance(graph, dict)
        or Path(str(runner_arguments["graph-root"])).resolve()
        != Path(str(graph.get("root"))).resolve()
        or runner_arguments["expected-graph-file-bytes"]
        != graph.get("graph_file_bytes")
        or runner_arguments["expected-graph-subset-bytes"]
        != graph.get("subset_file_bytes")
        or runner_arguments["expected-graph-dense-bytes"]
        != graph.get("runtime_dense_graph_bytes")
        or runner_arguments["expected-graph-memberships"] != 1_254_304
        or runner_arguments["graph-authority-persistent-bytes"]
        != graph.get("persistent_design_bytes")
        or runner_arguments["graph-authority-runtime-bytes"]
        != graph.get("runtime_design_payload_bytes")
    ):
        fail("system-freeze graph path/census/resource contract differs")

    if (
        pool.get("schema") != POOL_SCHEMA
        or pool.get("formal_execution_authorized") is not False
        or pool.get("system_freeze_id") != system_freeze_id
        or Path(str(pool.get("system_freeze_path"))).resolve()
        != system_freeze_path.resolve()
        or pool.get("timed_requests") != 800
        or pool.get("timed_per_family") != 200
        or pool.get("query_permutation_seed")
        != runner_arguments["query-permutation-seed"]
        or pool.get("query_permutation_fingerprint")
        != runner_arguments["query-permutation-fingerprint"]
        or pool.get("pair_schedule") != SCHEDULE
        or Path(str(pool.get("timed_requests_path"))).resolve()
        != Path(str(runner_arguments["queries"])).resolve()
    ):
        fail("frozen pool identity/census/permutation binding differs")

    endpoints = science.get("endpoints")
    diagnostics = science.get("family_diagnostics")
    performance_gate = science.get("performance_gate")
    science_system = science.get("system_freeze")
    system_snapshot = snapshot_nonempty_regular_file(
        system_freeze_path, "science-bound system freeze"
    )
    if (
        science.get("schema") != SCIENCE_SCHEMA
        or science.get("formal_execution_authorized") is not False
        or science.get("system_freeze_id") != system_freeze_id
        or not isinstance(science_system, dict)
        or science_system.get("path") != str(system_freeze_path.resolve())
        or science_system.get("bytes") != system_snapshot.bytes
        or science_system.get("mtime_ns") != system_snapshot.modified_ns
        or not isinstance(endpoints, dict)
        or set(endpoints) != {"primary_mixed_q800"}
        or endpoints["primary_mixed_q800"]
        != {"requests": 800, "families": list(FAMILIES)}
        or not isinstance(diagnostics, dict)
        or diagnostics.get("strict_tie_and_validity_gates") != list(FAMILIES)
        or diagnostics.get("family_qps_from_mixed_concurrency_prohibited") is not True
        or (
            REQUIRE_EXACT_FAMILY_BINDING
            and diagnostics.get("h_exact_families") != list(H_EXACT_FAMILIES)
        )
        or not isinstance(performance_gate, dict)
        or set(performance_gate)
        != {
            "primary_geometric_mean_qps_ratio_minimum",
            "one_sided_p_maximum",
            "two_sided_p_maximum",
            "primary_report_uses_two_sided_p",
            "rep1_futility_ratio",
            "rep4_futility_ratio",
        }
        or performance_gate.get("primary_geometric_mean_qps_ratio_minimum")
        != FINAL8_PAIR_GM_MIN_H_OVER_S
        or performance_gate.get("one_sided_p_maximum") != FINAL8_ONE_SIDED_P_MAX
        or performance_gate.get("two_sided_p_maximum")
        != FINAL8_TWO_SIDED_P_MAX
        or performance_gate.get("primary_report_uses_two_sided_p") is not True
        or performance_gate.get("rep1_futility_ratio") != PAIR1_MIN_H_OVER_S
        or performance_gate.get("rep4_futility_ratio")
        != FIRST4_PAIR_GM_MIN_H_OVER_S
    ):
        fail("science gate is not the unique frozen mixed-q800 endpoint")


def parse_manifest(path: Path, *, activation: bool) -> Manifest:
    if not path.is_absolute():
        fail("manifest path must be absolute")
    value = read_json_file(path)
    require_exact_keys(
        value,
        {
            "schema",
            "activation",
            "state_id",
            "runner_binary",
            "runner_bytes",
            "runner_mtime_ns",
            "status_log",
            "controller_cpu",
            "system_freeze_path",
            "system_freeze_id",
            "system_freeze_bytes",
            "system_freeze_mtime_ns",
            "frozen_pool_path",
            "frozen_pool_bytes",
            "frozen_pool_mtime_ns",
            "runner_arguments",
        },
        "formal manifest",
    )
    if value["schema"] != MANIFEST_SCHEMA:
        fail("formal manifest schema differs")
    if activation and value["activation"] != "FROZEN_FORMAL":
        fail("execution requires activation=FROZEN_FORMAL")
    if not activation and value["activation"] not in {
        "PROSPECTIVE",
        "FROZEN_FORMAL",
    }:
        fail("manifest activation field differs")
    state_id = value["state_id"]
    validate_state_id(state_id)
    arguments = value["runner_arguments"]
    if not isinstance(arguments, dict):
        fail("runner_arguments must be an object")
    require_exact_keys(arguments, RUNNER_ARGUMENTS, "runner_arguments")
    for name in PATH_ARGUMENTS:
        raw = arguments[name]
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            fail(f"runner argument {name} must be an absolute path string")
    for name in set(RUNNER_ARGUMENTS) - PATH_ARGUMENTS - {"cpu-list"}:
        positive_int(arguments[name], f"runner argument {name}")
    cpu_values = parse_worker_cpus(arguments["cpu-list"])
    controller_cpu = nonnegative_int(value["controller_cpu"], "controller_cpu")
    validate_cpu_partition(controller_cpu, cpu_values)
    validate_frozen_runner_arguments(arguments)
    runner_raw = value["runner_binary"]
    status_log_raw = value["status_log"]
    if (
        not isinstance(runner_raw, str)
        or not Path(runner_raw).is_absolute()
        or not isinstance(status_log_raw, str)
        or not Path(status_log_raw).is_absolute()
    ):
        fail("runner_binary and status_log must be absolute path strings")
    runner = Path(runner_raw)
    status_log = Path(status_log_raw)
    runner_bytes = positive_int(value["runner_bytes"], "runner_bytes")
    runner_mtime_ns = positive_int(value["runner_mtime_ns"], "runner_mtime_ns")
    runner_snapshot = snapshot_nonempty_regular_file(runner, "formal runner binary")
    if (
        runner_snapshot.bytes != runner_bytes
        or runner_snapshot.modified_ns != runner_mtime_ns
    ):
        fail("formal runner ordinary stat pin differs")
    system_freeze_id = value["system_freeze_id"]
    validate_state_id(system_freeze_id)
    system_freeze_path_raw = value["system_freeze_path"]
    frozen_pool_path_raw = value["frozen_pool_path"]
    if (
        not isinstance(system_freeze_path_raw, str)
        or not Path(system_freeze_path_raw).is_absolute()
        or not isinstance(frozen_pool_path_raw, str)
        or not Path(frozen_pool_path_raw).is_absolute()
    ):
        fail("system-freeze and frozen-pool paths must be absolute strings")
    system_freeze_path = Path(system_freeze_path_raw)
    frozen_pool_path = Path(frozen_pool_path_raw)
    system_freeze_bytes = positive_int(
        value["system_freeze_bytes"], "system_freeze_bytes"
    )
    system_freeze_mtime_ns = positive_int(
        value["system_freeze_mtime_ns"], "system_freeze_mtime_ns"
    )
    frozen_pool_bytes = positive_int(
        value["frozen_pool_bytes"], "frozen_pool_bytes"
    )
    frozen_pool_mtime_ns = positive_int(
        value["frozen_pool_mtime_ns"], "frozen_pool_mtime_ns"
    )
    system = read_stat_pinned_json(
        system_freeze_path,
        expected_bytes=system_freeze_bytes,
        expected_mtime_ns=system_freeze_mtime_ns,
        label="system freeze",
    )
    pool = read_stat_pinned_json(
        frozen_pool_path,
        expected_bytes=frozen_pool_bytes,
        expected_mtime_ns=frozen_pool_mtime_ns,
        label="frozen pool",
    )
    science_path_raw = pool.get("science_gate_path")
    if not isinstance(science_path_raw, str) or not Path(science_path_raw).is_absolute():
        fail("frozen pool science_gate_path must be absolute")
    science_path = Path(science_path_raw)
    science_before = snapshot_nonempty_regular_file(science_path, "science gate")
    science = read_json_file(science_path)
    if snapshot_nonempty_regular_file(science_path, "science gate") != science_before:
        fail("science gate changed while being read")
    validate_activation_documents(
        system=system,
        pool=pool,
        science=science,
        system_freeze_path=system_freeze_path,
        system_freeze_id=system_freeze_id,
        runner_arguments=arguments,
        controller_cpu=controller_cpu,
        runner_binary=runner,
        runner_bytes=runner_bytes,
        runner_mtime_ns=runner_mtime_ns,
    )
    if snapshot_nonempty_regular_file(runner, "formal runner binary") != runner_snapshot:
        fail("formal runner binary changed during activation validation")
    raw_directory = Path(str(arguments["raw-directory"]))
    ack_directory = Path(str(arguments["ack-directory"]))
    if status_log.parent in {raw_directory, ack_directory}:
        fail("status_log must be outside the raw and ACK directories")
    return Manifest(
        path=path,
        state_id=state_id,
        runner_binary=runner,
        runner_bytes=runner_bytes,
        runner_mtime_ns=runner_mtime_ns,
        status_log=status_log,
        controller_cpu=controller_cpu,
        system_freeze_path=system_freeze_path,
        system_freeze_id=system_freeze_id,
        frozen_pool_path=frozen_pool_path,
        arguments=arguments,
    )


def runner_command(manifest: Manifest) -> list[str]:
    # Recheck the mutable mapping at the final command boundary.
    validate_frozen_runner_arguments(manifest.arguments)
    command = [
        str(manifest.runner_binary),
        "--manifest",
        str(manifest.path),
        "--state-id",
        manifest.state_id,
    ]
    for name in RUNNER_ARGUMENTS:
        command.extend((f"--{name}", str(manifest.arguments[name])))
    return command


def pin_controller_to_reserved_cpu(manifest: Manifest) -> None:
    worker_cpus = parse_worker_cpus(manifest.arguments["cpu-list"])
    allowed_cpus = set(os.sched_getaffinity(0))
    validate_cpu_partition(
        manifest.controller_cpu,
        worker_cpus,
        allowed_cpus=allowed_cpus,
    )
    os.sched_setaffinity(0, {manifest.controller_cpu})
    if set(os.sched_getaffinity(0)) != {manifest.controller_cpu}:
        fail("controller CPU affinity did not become the reserved singleton")


def validate_empty_real_directory(path: Path, label: str) -> None:
    if not path.is_absolute():
        fail(f"{label} must be absolute")
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode):
        fail(f"{label} must be an existing real directory")
    if any(path.iterdir()):
        fail(f"{label} must be empty at activation")


def read_stable_jsonl(
    path: Path, label: str, maximum_bytes: int = 64 * 1024 * 1024
) -> tuple[list[dict[str, Any]], FileSnapshot]:
    before = snapshot_nonempty_regular_file(path, label)
    if before.bytes > maximum_bytes:
        fail(f"{label} exceeds bounded payload size")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        remaining = before.bytes
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                fail(f"short read from {label}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            fail(f"{label} grew while being read")
    finally:
        os.close(descriptor)
    after = snapshot_nonempty_regular_file(path, label)
    if after != before:
        fail(f"{label} changed while being read")
    payload = b"".join(chunks)
    if not payload.endswith(b"\n"):
        fail(f"{label} is not newline terminated")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(payload.splitlines(), start=1):
        if not raw:
            fail(f"{label} contains a blank row")
        try:
            value = json.loads(raw.decode("ascii"), object_pairs_hook=exact_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid {label} row {number}") from error
        if not isinstance(value, dict):
            fail(f"{label} row {number} is not an object")
        rows.append(value)
    return rows, before


def f32_from_bits(value: Any, label: str) -> float:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
        fail(f"{label} is not uint32 bits")
    result = struct.unpack("<f", struct.pack("<I", value))[0]
    if not math.isfinite(result) or result < 0.0:
        fail(f"{label} does not encode a finite nonnegative float")
    return result


@dataclass
class Aggregate:
    queries: int = 0
    strict_hits: int = 0
    strict_denominator: int = 0
    tie_hits: int = 0
    tie_denominator: int = 0
    invalid: int = 0
    predicate_fail: int = 0
    duplicate: int = 0
    forbidden_self: int = 0
    removed_self: int = 0
    underfull: int = 0
    nondeterministic: int = 0
    families: dict[str, "Aggregate"] = field(default_factory=dict)

    def add(self, row: dict[str, Any]) -> None:
        self.queries += 1
        for name in (
            "strict_hits",
            "strict_denominator",
            "tie_hits",
            "tie_denominator",
            "invalid",
            "predicate_fail",
            "duplicate",
            "forbidden_self",
            "removed_self",
            "underfull",
            "nondeterministic",
        ):
            value = row[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                fail(f"query audit field {name} is not a nonnegative integer")
            setattr(self, name, getattr(self, name) + value)

    def validity(self) -> bool:
        return (
            self.invalid == 0
            and self.predicate_fail == 0
            and self.duplicate == 0
            and self.forbidden_self == 0
            and self.underfull == 0
            and self.nondeterministic == 0
        )


def fraction_passes(hits: int, denominator: int, threshold: tuple[int, int]) -> bool:
    numerator, threshold_denominator = threshold
    return denominator > 0 and hits * threshold_denominator >= denominator * numerator


def family_recall_passes(
    aggregate: Aggregate,
    threshold: tuple[int, int],
    *,
    require_exact: bool,
) -> bool:
    """Evaluate one family's strict/tie contract with optional exactness."""

    threshold_pass = fraction_passes(
        aggregate.strict_hits, aggregate.strict_denominator, threshold
    ) and fraction_passes(
        aggregate.tie_hits, aggregate.tie_denominator, threshold
    )
    if not threshold_pass:
        return False
    return not require_exact or (
        aggregate.strict_hits == aggregate.strict_denominator
        and aggregate.tie_hits == aggregate.tie_denominator
    )


def validate_query_rows(
    rows: list[dict[str, Any]], manifest: Manifest, block: int
) -> Aggregate:
    expected_queries = 4 * manifest.timed_per_family
    if len(rows) != expected_queries:
        fail("formal query audit row census differs")
    arm = SCHEDULE[block]
    aggregate = Aggregate(families={family: Aggregate() for family in FAMILIES})
    permuted_rows: set[int] = set()
    original_rows: set[int] = set()
    for row in rows:
        if row.get("schema") != QUERY_SCHEMA:
            fail("formal query audit schema differs")
        if row.get("state_id") != manifest.state_id or row.get("block") != block:
            fail("formal query audit state/block binding differs")
        if row.get("pair") != block // 2 or row.get("pair_order") != (
            "SH" if (block // 2) % 2 == 0 else "HS"
        ):
            fail("formal query audit pair binding differs")
        if row.get("arm") != arm or row.get("audit_after_wall") is not True:
            fail("formal query audit arm/timing binding differs")
        family = row.get("family")
        if family not in FAMILIES:
            fail("formal query audit family differs")
        if row.get("backend") != BACKENDS[(arm, family)]:
            fail("formal query route/backend binding differs")
        expected_ef: int | None = (
            manifest.frozen_ef if arm == "S" or family == "conjunction" else None
        )
        if row.get("frozen_ef") != expected_ef:
            fail("formal query route did not propagate the one frozen ef")
        permuted = row.get("permuted_row")
        original = row.get("original_workload_row")
        if (
            not isinstance(permuted, int)
            or isinstance(permuted, bool)
            or not 0 <= permuted < expected_queries
            or not isinstance(original, int)
            or isinstance(original, bool)
            or not 0 <= original < expected_queries
        ):
            fail("formal query row index is outside the query pool")
        permuted_rows.add(permuted)
        original_rows.add(original)
        worker_lane = row.get("worker_lane")
        if (
            not isinstance(worker_lane, int)
            or isinstance(worker_lane, bool)
            or worker_lane not in range(LANES)
        ):
            fail("formal query worker lane differs")
        if row.get("strict_denominator") != K or row.get("tie_denominator") != K:
            fail("formal query recall denominator differs")
        truth = row.get("strict_truth_ids")
        observed = row.get("observed_ids")
        distances = row.get("observed_distance_f32_bits")
        result_size = row.get("result_size")
        if not isinstance(truth, list) or len(truth) != K or any(
            not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < N
            for item in truth
        ):
            fail("strict truth must be a sorted unique ten-ID set")
        if truth != sorted(set(truth)):
            fail("strict truth must be a sorted unique ten-ID set")
        if (
            not isinstance(result_size, int)
            or isinstance(result_size, bool)
            or not 0 <= result_size <= K
            or not isinstance(observed, list)
            or not isinstance(distances, list)
            or len(observed) != result_size
            or len(distances) != result_size
        ):
            fail("observed result/distance extent differs")
        query_base_id = row.get("query_base_id")
        service_ns = row.get("service_ns")
        if (
            not isinstance(query_base_id, int)
            or isinstance(query_base_id, bool)
            or not 0 <= query_base_id < N
            or not isinstance(service_ns, int)
            or isinstance(service_ns, bool)
            or service_ns <= 0
        ):
            fail("query identity or service time differs")
        if any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or not 0 <= item < 2**32
            for item in observed
        ):
            fail("observed ID is outside uint32")
        strict_hits = sum(item in set(truth) for item in observed)
        cutoff = f32_from_bits(row.get("kth_distance_f32_bits"), "kth cutoff")
        tie_hits = 0
        for item, bits in zip(observed, distances):
            if item >= N:
                if bits is not None:
                    fail("invalid observed ID must have null distance bits")
                continue
            distance = f32_from_bits(bits, "observed distance")
            tie_hits += distance <= cutoff
        if row.get("strict_hits") != strict_hits or row.get("tie_hits") != tie_hits:
            fail("strict-set or tie-aware query arithmetic differs")
        if (
            row.get("invalid") != sum(item >= N for item in observed)
            or row.get("duplicate") != len(observed) - len(set(observed))
            or row.get("forbidden_self")
            != sum(item == query_base_id for item in observed)
            or row.get("underfull") != int(result_size < K)
        ):
            fail("query structural validity arithmetic differs")
        support = row.get("support_after_leave_one_out")
        if (
            not isinstance(support, int)
            or isinstance(support, bool)
            or support < K
        ):
            fail("post-LOO support is smaller than k")
        # exact_support has already verified that the base ID belonged to the
        # predicate support and removed it before constructing ground truth.
        # An ANN route need not retrieve self among its candidates; correctness
        # requires only that self is absent from the returned top-k.
        if row.get("removed_self") not in {0, 1}:
            fail("self-candidate observation must be zero or one")
        leave_one_out = row.get("forbidden_self") == 0
        if row.get("leave_one_out_gate") is not leave_one_out:
            fail("leave-one-out query gate arithmetic differs")
        aggregate.add(row)
        aggregate.families[family].add(row)
    if permuted_rows != set(range(expected_queries)) or original_rows != set(
        range(expected_queries)
    ):
        fail("formal query permutation is not a full bijection")
    for family in FAMILIES:
        if aggregate.families[family].queries != manifest.timed_per_family:
            fail("formal query family census differs")
    return aggregate


def validate_summary(
    rows: list[dict[str, Any]], manifest: Manifest, block: int, aggregate: Aggregate
) -> bool:
    if len(rows) != 1:
        fail("formal block summary must contain exactly one row")
    row = rows[0]
    # The one performance endpoint is complete mixed-q800 throughput.  Exact
    # schema closure prevents family QPS, nested-equality, or post-hoc testing
    # fields from silently becoming controller gates; families contribute only
    # the recall and validity audits recomputed below.
    require_exact_keys(row, SUMMARY_FIELDS, "formal block summary")
    arm = SCHEDULE[block]
    if (
        row.get("schema") != SUMMARY_SCHEMA
        or row.get("state_id") != manifest.state_id
        or row.get("block") != block
        or row.get("pair") != block // 2
        or row.get("pair_order") != ("SH" if (block // 2) % 2 == 0 else "HS")
        or row.get("arm") != arm
        or row.get("audit_after_wall") is not True
        or row.get("frozen_ef") != manifest.frozen_ef
        or row.get("graph_beam") != 256
        or row.get("graph_pool") != 32
        or row.get("graph_cut_milli") != 1350
        or row.get("h_sieve_routes") != "conjunction_only"
        or row.get("timed_worker_explicit_io_calls") != 0
    ):
        fail("formal block summary identity/route/timing binding differs")
    fields = {
        "queries": aggregate.queries,
        "strict_hits": aggregate.strict_hits,
        "strict_denominator": aggregate.strict_denominator,
        "tie_hits": aggregate.tie_hits,
        "tie_denominator": aggregate.tie_denominator,
        "invalid": aggregate.invalid,
        "predicate_fail": aggregate.predicate_fail,
        "duplicate": aggregate.duplicate,
        "forbidden_self": aggregate.forbidden_self,
        "removed_self": aggregate.removed_self,
        "underfull": aggregate.underfull,
        "nondeterministic": aggregate.nondeterministic,
    }
    for name, expected in fields.items():
        if row.get(name) != expected:
            fail(f"formal block summary {name} differs from query rows")
    child_rss = nonnegative_int(
        row.get("max_child_rss_kib"), "formal block max_child_rss_kib"
    )
    child_peak_rss = nonnegative_int(
        row.get("max_child_peak_rss_kib"),
        "formal block max_child_peak_rss_kib",
    )
    if child_peak_rss < child_rss:
        fail("formal block child peak RSS is below current RSS")
    wall_ns = positive_int(row.get("wall_ns"), "formal block wall_ns")
    qps = row.get("complete_batch_qps")
    expected_qps = aggregate.queries * 1e9 / wall_ns
    if (
        not isinstance(qps, (int, float))
        or isinstance(qps, bool)
        or not math.isfinite(qps)
        or not math.isclose(float(qps), expected_qps, rel_tol=1e-12)
    ):
        fail("formal block complete-batch QPS arithmetic differs")
    validity = aggregate.validity()
    recall = fraction_passes(
        aggregate.strict_hits, aggregate.strict_denominator, manifest.minimum_recall
    ) and fraction_passes(
        aggregate.tie_hits, aggregate.tie_denominator, manifest.minimum_recall
    )
    for family, family_value in aggregate.families.items():
        recall = recall and family_recall_passes(
            family_value,
            manifest.minimum_recall,
            require_exact=arm == "H" and family in H_EXACT_FAMILIES,
        )
    if (
        row.get("validity_gate") is not validity
        or row.get("recall_gate") is not recall
        or row.get("block_gate") is not (validity and recall)
    ):
        fail("formal block gate arithmetic differs")
    return validity and recall


def streaming_pair_gate(
    h_over_s_ratios: list[float],
) -> tuple[bool, str, float | None]:
    """Apply only the two prospectively frozen mixed-q800 early-stop rules."""

    if not h_over_s_ratios or any(
        not math.isfinite(value) or value <= 0.0 for value in h_over_s_ratios
    ):
        fail("streaming H/S pair ratios must be finite and positive")
    first_four_gm: float | None = None
    if len(h_over_s_ratios) >= 4:
        first_four_gm = math.exp(
            sum(math.log(value) for value in h_over_s_ratios[:4]) / 4.0
        )
    if len(h_over_s_ratios) == 1 and h_over_s_ratios[0] < PAIR1_MIN_H_OVER_S:
        return (
            False,
            f"pair1_h_over_s_below_{PAIR1_MIN_H_OVER_S:.2f}",
            first_four_gm,
        )
    if (
        len(h_over_s_ratios) == 4
        and first_four_gm is not None
        and first_four_gm < FIRST4_PAIR_GM_MIN_H_OVER_S
    ):
        return (
            False,
            f"first4_pair_gm_h_over_s_below_{FIRST4_PAIR_GM_MIN_H_OVER_S:.2f}",
            first_four_gm,
        )
    return True, "continue", first_four_gm


def final_mixed_q800_gate(
    h_over_s_ratios: list[float],
) -> tuple[bool, float, float, float]:
    """Evaluate the frozen eight-pair GM and exact one-sided sign-flip gate."""

    if len(h_over_s_ratios) != PAIRS or any(
        not math.isfinite(value) or value <= 0.0 for value in h_over_s_ratios
    ):
        fail("final mixed-q800 gate requires eight finite positive pair ratios")
    logs = [math.log(value) for value in h_over_s_ratios]
    observed = sum(logs)
    geometric_mean = math.exp(observed / PAIRS)
    one_sided_extreme = 0
    two_sided_extreme = 0
    for assignment in range(1 << PAIRS):
        permuted = sum(
            value if assignment & (1 << index) else -value
            for index, value in enumerate(logs)
        )
        if permuted >= observed - 1e-15:
            one_sided_extreme += 1
        if abs(permuted) >= abs(observed) - 1e-15:
            two_sided_extreme += 1
    one_sided_p = one_sided_extreme / float(1 << PAIRS)
    two_sided_p = two_sided_extreme / float(1 << PAIRS)
    gate = (
        geometric_mean >= FINAL8_PAIR_GM_MIN_H_OVER_S
        and two_sided_p <= FINAL8_TWO_SIDED_P_MAX
    )
    return gate, geometric_mean, one_sided_p, two_sided_p


@dataclass
class ControllerState:
    commit_lock: SignalMaskedCommitLock = field(
        default_factory=SignalMaskedCommitLock
    )
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_signal: int | None = None
    phase: str = "startup"
    completed_blocks: int = 0
    latest_block: int | None = None
    latest_gate: str = "unknown"
    child_pid: int | None = None

    def request_stop(self, signal_number: int) -> None:
        with self.commit_lock:
            if self.stop_signal is None:
                self.stop_signal = signal_number

    def stopped(self) -> int | None:
        return self.stop_signal


def append_status(descriptor: int, state: ControllerState, line: str) -> None:
    payload = (line.rstrip("\n") + "\n").encode("ascii")
    with state.log_lock:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                fail("zero-byte controller status write")
            offset += written
        os.fdatasync(descriptor)


def heartbeat_loop(
    descriptor: int, state: ControllerState, stop: threading.Event
) -> None:
    while not stop.wait(25.0):
        rss = "unknown"
        if state.child_pid is not None:
            try:
                for line in Path(f"/proc/{state.child_pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss = line.split()[1]
                        break
            except (FileNotFoundError, PermissionError, OSError):
                pass
        append_status(
            descriptor,
            state,
            "CONTROLLER_HEARTBEAT "
            f"utc_epoch_s={int(time.time())} pid={os.getpid()} "
            f"child_pid={state.child_pid or 0} phase={state.phase} "
            f"completed_blocks={state.completed_blocks} total_blocks={BLOCKS} "
            f"latest_block={state.latest_block if state.latest_block is not None else 'unknown'} "
            f"latest_gate={state.latest_gate} child_rss_kib={rss} eta=unknown",
        )


def wait_for_block_files(
    process: subprocess.Popen[bytes], results: Path, summary: Path,
    state: ControllerState,
) -> None:
    while not (results.is_file() and summary.is_file()):
        if state.stop_signal is not None:
            fail(f"controller stop requested: {state.stop_signal}")
        return_code = process.poll()
        if return_code is not None:
            fail(f"runner exited {return_code} before publishing the next block")
        time.sleep(0.02)


def open_status_log(path: Path) -> int:
    if not path.parent.is_dir():
        fail("status-log parent directory does not exist")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC,
        0o644,
    )
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return descriptor


def execute(manifest: Manifest) -> int:
    validate_frozen_runner_arguments(manifest.arguments)
    runner_snapshot = snapshot_nonempty_regular_file(
        manifest.runner_binary, "formal runner binary"
    )
    if (
        runner_snapshot.bytes != manifest.runner_bytes
        or runner_snapshot.modified_ns != manifest.runner_mtime_ns
    ):
        fail("formal runner ordinary stat pin differs at execution")
    validate_empty_real_directory(manifest.raw_directory, "raw-directory")
    validate_empty_real_directory(manifest.ack_directory, "ack-directory")
    if manifest.raw_directory == manifest.ack_directory:
        fail("raw and ACK directories must be distinct")
    # The polling/controller thread and its heartbeat inherit this reserved
    # singleton CPU, outside all eight timed workers.
    pin_controller_to_reserved_cpu(manifest)
    descriptor = open_status_log(manifest.status_log)
    state = ControllerState()

    def handle_signal(signal_number: int, _frame: Any) -> None:
        state.request_stop(signal_number)

    old_int = signal.signal(signal.SIGINT, handle_signal)
    old_term = signal.signal(signal.SIGTERM, handle_signal)
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=heartbeat_loop,
        args=(descriptor, state, stop_heartbeat),
        daemon=True,
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        append_status(
            descriptor,
            state,
            f"CONTROLLER_START schema={CONTROLLER_SCHEMA} "
            f"state_id={manifest.state_id} blocks={BLOCKS} schedule={SCHEDULE} "
            f"controller_cpu={manifest.controller_cpu}",
        )
        process = subprocess.Popen(
            runner_command(manifest),
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=descriptor,
            close_fds=True,
            start_new_session=True,
        )
        if (
            snapshot_nonempty_regular_file(
                manifest.runner_binary, "formal runner binary"
            )
            != runner_snapshot
        ):
            fail("formal runner binary changed across process activation")
        state.child_pid = process.pid
        heartbeat.start()
        all_pass = True
        pair_arm_qps: dict[int, dict[str, float]] = {}
        h_over_s_ratios: list[float] = []
        for block in range(BLOCKS):
            state.phase = "await_raw_summary"
            state.latest_block = block
            results_path = manifest.raw_directory / f"block_{block:02d}_results.jsonl"
            summary_path = manifest.raw_directory / f"block_{block:02d}_summary.jsonl"
            wait_for_block_files(process, results_path, summary_path, state)
            state.phase = "audit_block"
            result_rows, results_snapshot = read_stable_jsonl(
                results_path, "formal block results"
            )
            summary_rows, summary_snapshot = read_stable_jsonl(
                summary_path, "formal block summary"
            )
            aggregate = validate_query_rows(result_rows, manifest, block)
            gate = validate_summary(summary_rows, manifest, block, aggregate)
            pair = block // 2
            arm = SCHEDULE[block]
            pair_arm_qps.setdefault(pair, {})[arm] = float(
                summary_rows[0]["complete_batch_qps"]
            )
            pair_gate = True
            pair_gate_reason = "incomplete_pair"
            first_four_gm: float | None = None
            if set(pair_arm_qps[pair]) == {"H", "S"}:
                ratio = pair_arm_qps[pair]["H"] / pair_arm_qps[pair]["S"]
                h_over_s_ratios.append(ratio)
                pair_gate, pair_gate_reason, first_four_gm = streaming_pair_gate(
                    h_over_s_ratios
                )
                append_status(
                    descriptor,
                    state,
                    "CONTROLLER_PAIR_GATE "
                    f"pair={pair + 1} endpoint=mixed_q800 "
                    f"h_over_s={ratio:.9f} "
                    f"first4_gm={first_four_gm if first_four_gm is not None else 'pending'} "
                    f"gate={'PASS' if pair_gate else 'STOP'} "
                    f"reason={pair_gate_reason}",
                )
                if len(h_over_s_ratios) == PAIRS:
                    (
                        final_gate,
                        final_gm,
                        final_one_sided_p,
                        final_two_sided_p,
                    ) = final_mixed_q800_gate(h_over_s_ratios)
                    append_status(
                        descriptor,
                        state,
                        "CONTROLLER_FINAL_ENDPOINT_GATE endpoint=mixed_q800 "
                        f"pairs={PAIRS} h_over_s_gm={final_gm:.9f} "
                        f"one_sided_sign_flip_p={final_one_sided_p:.9f} "
                        f"two_sided_sign_flip_p={final_two_sided_p:.9f} "
                        "primary_p=two_sided "
                        f"gate={'PASS' if final_gate else 'STOP'}",
                    )
                    if not final_gate:
                        pair_gate = False
                        pair_gate_reason = "final_mixed_q800_gm_or_p"
            gate = gate and pair_gate
            if (
                snapshot_nonempty_regular_file(results_path, "formal block results")
                != results_snapshot
                or snapshot_nonempty_regular_file(summary_path, "formal block summary")
                != summary_snapshot
            ):
                fail("formal block files changed after controller audit")
            state.phase = "publish_ack"
            published = publish_block_ack(
                ack_directory=manifest.ack_directory,
                state_id=manifest.state_id,
                block=block,
                summary_path=summary_path,
                results_path=results_path,
                stop_requested=state.stopped,
                commit_lock=state.commit_lock,
            )
            if not gate:
                # ACK preserves the audited partial block, but withholding the
                # canonical release guarantees that block N+1 is never declared
                # or started after a scientific kill gate.
                all_pass = False
                state.latest_gate = "STOP"
                state.phase = "release_withheld"
                append_status(
                    descriptor,
                    state,
                    f"CONTROLLER_BLOCK_WITHHELD block={block} arm={arm} "
                    f"summary_bytes={summary_snapshot.bytes} "
                    f"results_bytes={results_snapshot.bytes} gate=STOP "
                    f"reason={pair_gate_reason if not pair_gate else 'block_audit'}",
                )
                break
            state.phase = "publish_release"
            publish_block_release(published, stop_requested=state.stopped)
            state.completed_blocks = block + 1
            state.latest_gate = "PASS" if gate else "STOP"
            append_status(
                descriptor,
                state,
                f"CONTROLLER_BLOCK_RELEASED block={block} arm={SCHEDULE[block]} "
                f"summary_bytes={summary_snapshot.bytes} "
                f"results_bytes={results_snapshot.bytes} "
                f"gate={state.latest_gate}",
            )
        if not all_pass:
            state.phase = "terminate_after_stop"
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            state.phase = "complete"
            append_status(
                descriptor,
                state,
                f"CONTROLLER_FINAL completed_blocks={state.completed_blocks} "
                "experiment_gate=STOP release_withheld=true",
            )
            return 2
        state.phase = "wait_runner_exit"
        return_code = process.wait(timeout=60)
        expected = 0 if state.completed_blocks == BLOCKS else 2
        if return_code != expected:
            fail(f"runner exit {return_code} differs from expected {expected}")
        state.phase = "complete"
        append_status(
            descriptor,
            state,
            f"CONTROLLER_FINAL completed_blocks={state.completed_blocks} "
            f"experiment_gate={'PASS' if expected == 0 else 'STOP'}",
        )
        return expected
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stop_heartbeat.set()
        if heartbeat.is_alive():
            heartbeat.join(timeout=2)
        os.close(descriptor)
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def query_free_activation_binding_self_test() -> None:
    """Exercise activation-document joins without opening workload rows."""

    freeze_id = "synthetic-freeze-v19"
    runner_path = Path("/synthetic/runner")
    runner_bytes = 1234
    runner_mtime_ns = 5678
    arguments: dict[str, Any] = {
        **FROZEN_RUNNER_VALUES,
        "cpu-list": "0,1,2,3,4,5,6,7",
        "query-permutation-seed": 123,
        "query-permutation-fingerprint": 456,
        "base": "/synthetic/base",
        "spmat": "/synthetic/spmat",
        "endpoints": "/synthetic/endpoints",
        "history": "/synthetic/history",
        "queries": "/synthetic/timed_requests.jsonl",
        "graph-root": "/synthetic/graph",
        "numeric-similarity": "/synthetic/similarity",
        "numeric-original-width": "/synthetic/width",
        "numeric-original-height": "/synthetic/height",
        "expected-graph-file-bytes": 11,
        "expected-graph-subset-bytes": 12,
        "expected-graph-dense-bytes": 13,
        "expected-graph-memberships": 1_254_304,
        "graph-authority-persistent-bytes": 15,
        "graph-authority-runtime-bytes": 16,
    }
    with tempfile.TemporaryDirectory(prefix="laion_v19_activation_selftest_") as root:
        directory = Path(root)
        system_path = directory / "system_freeze.json"
        science_path = directory / "science_gate.json"
        system = {
            "schema": SYSTEM_FREEZE_SCHEMA,
            "freeze_id": freeze_id,
            "formal_execution_authorized": False,
            "complete_file_stat_pins": [
                {
                    "path": str(runner_path.resolve()),
                    "bytes": runner_bytes,
                    "mtime_ns": runner_mtime_ns,
                }
            ],
            "execution": {
                "controller_cpu": 8,
                "cpus": list(range(8)),
                "lanes": LANES,
                "schedule": SCHEDULE,
                "controller_cpu_disjoint_from_worker_cpus": True,
            },
            "statistical_contract": {
                "timed_per_family": 200,
                "strict_and_tie_recall_floor": {
                    "numerator": 9995,
                    "denominator": 10000,
                },
                "primary_geomean_qps_ratio_minimum":
                    FINAL8_PAIR_GM_MIN_H_OVER_S,
                "only_performance_endpoint": "mixed_q800_complete_batch_wall",
                "family_gates": "strict_tie_recall_and_validity_only",
                "family_qps_from_mixed_concurrency_prohibited": True,
                "one_sided_p_maximum": 0.05,
                "two_sided_p_maximum": 0.05,
                "primary_report_uses_two_sided_p": True,
            },
            "sieve_census": {
                "reference": {"partitions": 18, "memberships": 109757},
                "saturated": {"partitions": 42, "memberships": 167220},
            },
            "inputs": {
                "base": "/synthetic/base",
                "spmat": "/synthetic/spmat",
                "endpoints": "/synthetic/endpoints",
                "history": "/synthetic/history",
                "numeric_similarity": "/synthetic/similarity",
                "numeric_original_width": "/synthetic/width",
                "numeric_original_height": "/synthetic/height",
            },
            "graph": {
                "root": "/synthetic/graph",
                "graph_file_bytes": 11,
                "subset_file_bytes": 12,
                "runtime_dense_graph_bytes": 13,
                "indexed_memberships": 14,
                "persistent_design_bytes": 15,
                "runtime_design_payload_bytes": 16,
            },
        }
        if REQUIRE_EXACT_FAMILY_BINDING:
            system["statistical_contract"]["h_exact_families"] = list(
                H_EXACT_FAMILIES
            )
        system_path.write_text(
            json.dumps(system, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        system_snapshot = snapshot_nonempty_regular_file(
            system_path, "synthetic system freeze"
        )
        pool = {
            "schema": POOL_SCHEMA,
            "formal_execution_authorized": False,
            "system_freeze_id": freeze_id,
            "system_freeze_path": str(system_path.resolve()),
            "timed_requests": 800,
            "timed_per_family": 200,
            "query_permutation_seed": 123,
            "query_permutation_fingerprint": 456,
            "pair_schedule": SCHEDULE,
            "timed_requests_path": "/synthetic/timed_requests.jsonl",
            "science_gate_path": str(science_path.resolve()),
        }
        science = {
            "schema": SCIENCE_SCHEMA,
            "formal_execution_authorized": False,
            "system_freeze_id": freeze_id,
            "system_freeze": {
                "path": str(system_path.resolve()),
                "bytes": system_snapshot.bytes,
                "mtime_ns": system_snapshot.modified_ns,
            },
            "endpoints": {
                "primary_mixed_q800": {
                    "requests": 800,
                    "families": list(FAMILIES),
                }
            },
            "family_diagnostics": {
                "strict_tie_and_validity_gates": list(FAMILIES),
                "family_qps_from_mixed_concurrency_prohibited": True,
            },
            "performance_gate": {
                "primary_geometric_mean_qps_ratio_minimum":
                    FINAL8_PAIR_GM_MIN_H_OVER_S,
                "one_sided_p_maximum": 0.05,
                "two_sided_p_maximum": 0.05,
                "primary_report_uses_two_sided_p": True,
                "rep1_futility_ratio": PAIR1_MIN_H_OVER_S,
                "rep4_futility_ratio": FIRST4_PAIR_GM_MIN_H_OVER_S,
            },
        }
        if REQUIRE_EXACT_FAMILY_BINDING:
            science["family_diagnostics"]["h_exact_families"] = list(
                H_EXACT_FAMILIES
            )
        science_path.write_text(
            json.dumps(science, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        validate_activation_documents(
            system=system,
            pool=pool,
            science=science,
            system_freeze_path=system_path,
            system_freeze_id=freeze_id,
            runner_arguments=arguments,
            controller_cpu=8,
            runner_binary=runner_path,
            runner_bytes=runner_bytes,
            runner_mtime_ns=runner_mtime_ns,
        )
        if REQUIRE_EXACT_FAMILY_BINDING:
            bad_system = {
                **system,
                "statistical_contract": dict(system["statistical_contract"]),
            }
            bad_system["statistical_contract"].pop("h_exact_families")
            try:
                validate_activation_documents(
                    system=bad_system,
                    pool=pool,
                    science=science,
                    system_freeze_path=system_path,
                    system_freeze_id=freeze_id,
                    runner_arguments=arguments,
                    controller_cpu=8,
                    runner_binary=runner_path,
                    runner_bytes=runner_bytes,
                    runner_mtime_ns=runner_mtime_ns,
                )
            except RuntimeError as error:
                if "statistical contract differs" not in str(error):
                    raise
            else:
                fail("missing H exact-family system binding was accepted")
            bad_science = {
                **science,
                "family_diagnostics": dict(science["family_diagnostics"]),
            }
            bad_science["family_diagnostics"].pop("h_exact_families")
            try:
                validate_activation_documents(
                    system=system,
                    pool=pool,
                    science=bad_science,
                    system_freeze_path=system_path,
                    system_freeze_id=freeze_id,
                    runner_arguments=arguments,
                    controller_cpu=8,
                    runner_binary=runner_path,
                    runner_bytes=runner_bytes,
                    runner_mtime_ns=runner_mtime_ns,
                )
            except RuntimeError as error:
                if "science gate" not in str(error):
                    raise
            else:
                fail("missing H exact-family science binding was accepted")
        read_stat_pinned_json(
            system_path,
            expected_bytes=system_snapshot.bytes,
            expected_mtime_ns=system_snapshot.modified_ns,
            label="synthetic system freeze",
        )
        try:
            read_stat_pinned_json(
                system_path,
                expected_bytes=system_snapshot.bytes + 1,
                expected_mtime_ns=system_snapshot.modified_ns,
                label="synthetic system freeze",
            )
        except RuntimeError as error:
            if "stat pin differs" not in str(error):
                raise
        else:
            fail("synthetic system-freeze stat-pin mismatch was accepted")
        bad_pool = dict(pool)
        bad_pool["timed_requests"] = 799
        try:
            validate_activation_documents(
                system=system,
                pool=bad_pool,
                science=science,
                system_freeze_path=system_path,
                system_freeze_id=freeze_id,
                runner_arguments=arguments,
                controller_cpu=8,
                runner_binary=runner_path,
                runner_bytes=runner_bytes,
                runner_mtime_ns=runner_mtime_ns,
            )
        except RuntimeError as error:
            if "pool identity/census/permutation" not in str(error):
                raise
        else:
            fail("synthetic q799 pool entered the q800 activation contract")


def query_free_self_test() -> int:
    if SCHEDULE != "SHHSSHHSSHHSSHHS" or len(SCHEDULE) != BLOCKS:
        fail("frozen 8-pair schedule differs")
    query_free_activation_binding_self_test()
    for pair in range(PAIRS):
        expected = "SH" if pair % 2 == 0 else "HS"
        if SCHEDULE[2 * pair : 2 * pair + 2] != expected:
            fail("pair order does not alternate SH/HS")
    synthetic_bits = struct.unpack("<I", struct.pack("<f", 0.3))[0]
    roundtrip_bits = struct.unpack(
        "<I", struct.pack("<f", f32_from_bits(synthetic_bits, "synthetic"))
    )[0]
    if roundtrip_bits != synthetic_bits:
        fail("f32 distance-bit round trip differs")
    synthetic_manifest = Manifest(
        path=Path("/synthetic/manifest.json"),
        state_id="synthetic-v19",
        runner_binary=Path("/synthetic/runner"),
        runner_bytes=1,
        runner_mtime_ns=1,
        status_log=Path("/synthetic/status.log"),
        controller_cpu=8,
        system_freeze_path=Path("/synthetic/system_freeze.json"),
        system_freeze_id="synthetic-freeze-v19",
        frozen_pool_path=Path("/synthetic/frozen_pool.json"),
        arguments={
            "timed-per-family": 1,
            "frozen-ef": 800,
            "minimum-recall-numerator": 9995,
            "minimum-recall-denominator": 10000,
        },
    )
    validate_frozen_runner_arguments(dict(FROZEN_RUNNER_VALUES))
    near_exact = Aggregate(
        strict_hits=1999,
        strict_denominator=2000,
        tie_hits=2000,
        tie_denominator=2000,
    )
    if not family_recall_passes(
        near_exact, (9995, 10000), require_exact=False
    ):
        fail("synthetic 0.9995 family recall did not pass the threshold gate")
    if family_recall_passes(
        near_exact, (9995, 10000), require_exact=True
    ):
        fail("synthetic 0.9995 family recall entered an exact-family gate")
    for name, frozen in FROZEN_RUNNER_VALUES.items():
        counterexample = dict(FROZEN_RUNNER_VALUES)
        counterexample[name] = frozen + 1
        try:
            validate_frozen_runner_arguments(counterexample)
        except RuntimeError as error:
            if name not in str(error):
                raise
        else:
            fail(f"synthetic frozen-gate counterexample accepted: {name}")
    validate_cpu_partition(8, tuple(range(8)), allowed_cpus=set(range(9)))
    try:
        validate_cpu_partition(7, tuple(range(8)), allowed_cpus=set(range(9)))
    except RuntimeError as error:
        if "outside" not in str(error):
            raise
    else:
        fail("synthetic controller/worker CPU overlap was accepted")
    distance_bits = struct.unpack("<I", struct.pack("<f", 0.25))[0]
    truth = list(range(K))
    for block in (0, 1):
        arm = SCHEDULE[block]
        query_rows: list[dict[str, Any]] = []
        for index, family in enumerate(FAMILIES):
            query_rows.append(
                {
                    "schema": QUERY_SCHEMA,
                    "state_id": synthetic_manifest.state_id,
                    "block": block,
                    "pair": block // 2,
                    "pair_order": "SH",
                    "arm": arm,
                    "audit_after_wall": True,
                    "family": family,
                    "backend": BACKENDS[(arm, family)],
                    "frozen_ef": (
                        synthetic_manifest.frozen_ef
                        if arm == "S" or family == "conjunction"
                        else None
                    ),
                    "permuted_row": index,
                    "original_workload_row": index,
                    "worker_lane": index,
                    "strict_denominator": K,
                    "tie_denominator": K,
                    "strict_truth_ids": truth,
                    "observed_ids": truth,
                    "observed_distance_f32_bits": [distance_bits] * K,
                    "result_size": K,
                    "query_base_id": 100,
                    "service_ns": 1000 + index,
                    "strict_hits": K,
                    "tie_hits": K,
                    "kth_distance_f32_bits": distance_bits,
                    "support_after_leave_one_out": K,
                    "removed_self": 1,
                    "forbidden_self": 0,
                    "leave_one_out_gate": True,
                    "invalid": 0,
                    "duplicate": 0,
                    "underfull": 0,
                    "predicate_fail": 0,
                    "nondeterministic": 0,
                }
            )
        aggregate = validate_query_rows(query_rows, synthetic_manifest, block)
        wall_ns = 1_000_000
        summary = {
            "schema": SUMMARY_SCHEMA,
            "state_id": synthetic_manifest.state_id,
            "block": block,
            "pair": block // 2,
            "pair_order": "SH",
            "arm": arm,
            "audit_after_wall": True,
            "frozen_ef": synthetic_manifest.frozen_ef,
            "graph_beam": 256,
            "graph_pool": 32,
            "graph_cut_milli": 1350,
            "h_sieve_routes": "conjunction_only",
            "timed_worker_explicit_io_calls": 0,
            "max_child_peak_rss_kib": 1,
            "max_child_rss_kib": 1,
            "queries": 4,
            "strict_hits": 4 * K,
            "strict_denominator": 4 * K,
            "tie_hits": 4 * K,
            "tie_denominator": 4 * K,
            "invalid": 0,
            "predicate_fail": 0,
            "duplicate": 0,
            "forbidden_self": 0,
            "removed_self": 4,
            "underfull": 0,
            "nondeterministic": 0,
            "wall_ns": wall_ns,
            "complete_batch_qps": 4 * 1e9 / wall_ns,
            "validity_gate": True,
            "recall_gate": True,
            "block_gate": True,
        }
        if not validate_summary([summary], synthetic_manifest, block, aggregate):
            fail("synthetic positive block did not pass")
        if block == 0:
            no_self_candidate_rows = [dict(row) for row in query_rows]
            no_self_candidate_rows[0]["removed_self"] = 0
            no_self_candidate_rows[0]["leave_one_out_gate"] = True
            no_self_candidate_aggregate = validate_query_rows(
                no_self_candidate_rows, synthetic_manifest, block
            )
            no_self_candidate_summary = dict(summary)
            no_self_candidate_summary["removed_self"] = 3
            if not validate_summary(
                [no_self_candidate_summary], synthetic_manifest, block,
                no_self_candidate_aggregate,
            ):
                fail("absent self candidate incorrectly failed validity")
            forbidden_family_performance = dict(summary)
            forbidden_family_performance["family_qps"] = {"equality": 1.0}
            try:
                validate_summary(
                    [forbidden_family_performance], synthetic_manifest, block, aggregate
                )
            except RuntimeError as error:
                if "keys differ" not in str(error):
                    raise
            else:
                fail("family performance field entered the mixed-q800 summary gate")
        if block == 1:
            conjunction = query_rows[1]
            conjunction["frozen_ef"] = None
            try:
                validate_query_rows(query_rows, synthetic_manifest, block)
            except RuntimeError as error:
                if "one frozen ef" not in str(error):
                    raise
            else:
                fail("H conjunction accepted a missing frozen ef")
    if streaming_pair_gate([PAIR1_MIN_H_OVER_S - 0.001])[0]:
        fail("pair-1 H/S counterexample did not stop")
    if streaming_pair_gate([FIRST4_PAIR_GM_MIN_H_OVER_S - 0.01] * 4)[0]:
        fail("first-four-pair GM counterexample did not stop")
    if not streaming_pair_gate([FIRST4_PAIR_GM_MIN_H_OVER_S + 0.01] * 4)[0]:
        fail("positive first-four-pair GM fixture did not continue")
    if final_mixed_q800_gate([FINAL8_PAIR_GM_MIN_H_OVER_S - 0.01] * PAIRS)[0]:
        fail("sub-threshold final mixed-q800 GM counterexample passed")
    final_gate, _gm, one_sided_p, two_sided_p = final_mixed_q800_gate(
        [FINAL8_PAIR_GM_MIN_H_OVER_S + 0.20] * PAIRS
    )
    if (
        not final_gate
        or one_sided_p != 1 / 256
        or two_sided_p != 2 / 256
    ):
        fail("positive final mixed-q800 GM/sign-flip fixture stopped")
    print(
        f"SELF_TEST schema={CONTROLLER_SCHEMA} status=PASS "
        "queries_opened=0 indexes_built=0 performance_processes_started=0 "
        f"schedule={SCHEDULE} pairs={PAIRS} blocks={BLOCKS} "
        "ack_then_release_adapter_imported=true block_audit_fixture=true "
        "wrong_h_conjunction_ef_counterexample=STOP "
        "eight_frozen_argument_counterexamples=STOP "
        "controller_worker_cpu_overlap_counterexample=STOP "
        "absent_self_candidate_validity_fixture=PASS "
        "family_qps_field_counterexample=STOP "
        "exact_family_0p9995_counterexample=STOP "
        f"missing_exact_family_activation_bindings="
        f"{'STOP' if REQUIRE_EXACT_FAMILY_BINDING else 'NOT_APPLICABLE'} "
        "pair1_and_first4_streaming_counterexamples=STOP "
        "final8_mixed_q800_gm_two_sided_signflip_counterexample=STOP"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query-free-self-test", action="store_true")
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args()
    if arguments.query_free_self_test:
        if arguments.manifest is not None:
            fail("query-free self-test does not accept a manifest")
        return query_free_self_test()
    if arguments.manifest is None:
        fail("--manifest is required")
    manifest = parse_manifest(arguments.manifest, activation=arguments.execute)
    if arguments.validate_only:
        print(
            "VALID manifest_schema=PASS activation_structure=PASS "
            "data_or_query_files_opened=0 performance_processes_started=0"
        )
        return 0
    return execute(manifest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"BLOCKED {error}", file=sys.stderr)
        raise SystemExit(2)
