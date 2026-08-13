#!/usr/bin/env python3
"""Shared, fail-closed constants and byte-level helpers for range/DNF v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(os.environ.get("BOOLEANN_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2]))
CODE = ROOT / "03_experiment_bridge/code"
CONFIG = ROOT / "03_experiment_bridge/configs"
PLANS = ROOT / "03_experiment_bridge/plans"
RAW_RESULTS = ROOT / "03_experiment_bridge/results/raw"

PROTOCOL = PLANS / "protocol_addendum_sift10m_range_postbuild_v2_before_workload_20260722.md"
GATE_A_INPUTS = CONFIG / "sift10m_range_v2_gate_a_inputs_20260722.json"
GATE_A = PLANS / "system_freeze_sift10m_range_v2_before_workload_20260722.json"
SCHEDULE = CONFIG / "sift10m_range_v2_point_schedule_20260722.json"
ACTIVATION = PLANS / "activation_sift10m_range_v2_after_gt_20260722.json"
AUTHORIZATION = PLANS / "authorization_sift10m_range_v2_formal_execution_20260722.json"

DATA_ROOT = ROOT / "data/raw/sift10m/range_postbuild_v2_20260722"
RANGE_PREDICATES = DATA_ROOT / "range_postbuild_v2.bin"
RANGE_GT = DATA_ROOT / "range_postbuild_v2_gt.bin"
DNF_PREDICATES = DATA_ROOT / "dnf2_postbuild_v2.bin"
DNF_GT = DATA_ROOT / "dnf2_postbuild_v2_gt.bin"
WORKLOAD_MANIFEST = DATA_ROOT / "manifest.json"
GT_VERIFICATION = DATA_ROOT / "independent_verification.json"

GT_PROGRESS_ROOT = RAW_RESULTS / "sift10m_range_v2_gt_generation_20260722"
CALIBRATION_ROOT = RAW_RESULTS / "sift10m_range_v2_original_calibration_20260722"
CALIBRATION_COMPLETION = CALIBRATION_ROOT / "campaign_completion.json"
FORMAL_ROOT = RAW_RESULTS / "sift10m_range_v2_formal_20260722"
SCALING_ROOT = RAW_RESULTS / "sift10m_range_v2_scaling_20260722"

BASE_RAW = ROOT / "data/raw/sift10m/sift10m_base.fvecs"
QUERY_FVECS = ROOT / "data/raw/sift/sift_query.fvecs"
ATTRIBUTES = ROOT / "baselines/WoW/exp/data/meta/meta_c5000_n10000000.bin"
ORIGINAL_PREDICATES = ROOT / "baselines/WoW/exp/data/ranges/common_ranges_c5000_nq1000.bin"
ORIGINAL_GT = ROOT / "baselines/WoW/test_results/common_sift10m_gt_c5000_nq1000.bin"

WORKLOADS = ("original", "postbuild_range", "postbuild_dnf2")
BEAMS = (64, 80, 96, 112, 128, 160, 192, 256)
WOW_EFS = (900, 1100, 1400, 1700)
SERF_EFS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
PARAMETERS = {"bci": BEAMS, "wow": WOW_EFS, "serf": SERF_EFS}
N = 10_000_000
NQ = 1_000
DIM = 128
K = 10
THREADS = 8
PASSES = 6
REPS = 5
FAMILY_COMPARISONS = 336
HEARTBEAT_SECONDS = 30


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"dependency is not a regular file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(resolved),
    }


def verify_pin(record: dict[str, Any], *, rehash: bool = True) -> None:
    expected = {"path", "bytes", "mtime_ns", "sha256"}
    if set(record) != expected:
        raise RuntimeError(f"malformed file pin keys: {set(record)}")
    path = Path(record["path"]).resolve(strict=True)
    stat = path.stat()
    if (str(path) != record["path"] or stat.st_size != record["bytes"] or
            stat.st_mtime_ns != record["mtime_ns"]):
        raise RuntimeError(f"frozen file changed: {record['path']}")
    if rehash and sha256(path) != record["sha256"]:
        raise RuntimeError(f"frozen file content changed: {record['path']}")


def deterministic_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def load_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def write_json_exclusive(path: Path, value: Any, *, deterministic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = deterministic_json_bytes(value) if deterministic else (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = deterministic_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(descriptor, "ab", buffering=0) as handle:
        handle.write(payload)
        os.fsync(handle.fileno())


def derive_seed(freeze_digest: str, label: bytes) -> int:
    if len(freeze_digest) != 64:
        raise RuntimeError("Gate A SHA-256 must contain 64 hexadecimal characters")
    try:
        frozen_bytes = bytes.fromhex(freeze_digest)
    except ValueError as exc:
        raise RuntimeError("invalid Gate A SHA-256") from exc
    return int.from_bytes(hashlib.sha256(label + b"\0" + frozen_bytes).digest()[:8], "big")


def workload_paths(workload: str) -> tuple[Path, Path, str]:
    if workload == "original":
        return ORIGINAL_PREDICATES, ORIGINAL_GT, "range"
    if workload == "postbuild_range":
        return RANGE_PREDICATES, RANGE_GT, "range"
    if workload == "postbuild_dnf2":
        return DNF_PREDICATES, DNF_GT, "dnf2"
    raise RuntimeError(f"not a frozen workload: {workload}")


def assert_absent(paths: Iterable[Path], label: str) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise RuntimeError(f"{label} must not exist yet: {present}")


def verify_gate_a(*, rehash: bool = True) -> tuple[dict[str, Any], str]:
    raw = GATE_A.read_bytes()
    gate = json.loads(raw)
    digest = sha256_bytes(raw)
    if gate.get("schema") != "sift10m-range-v2-system-freeze/v1":
        raise RuntimeError("unsupported Gate A schema")
    if gate.get("self_payload_sha256") is not None:
        raise RuntimeError("Gate A must not contain a recursive self hash")
    for key in ("protocol", "gate_a_input", "schedule"):
        verify_pin(gate[key], rehash=rehash)
    if gate["grid"] != {
        "bci_beams": list(BEAMS),
        "wow_efs": list(WOW_EFS),
        "serf_efs": list(SERF_EFS),
        "k": K,
        "threads": THREADS,
        "passes": PASSES,
        "process_reps": REPS,
        "workloads": list(WORKLOADS),
        "points_per_workload_rep": 22,
        "states_per_rep": 66,
        "formal_states": 330,
        "comparison_family": FAMILY_COMPARISONS,
        "comparison_family_formula": "8*(4+10)*3",
    }:
        raise RuntimeError("Gate A grid is not the hard-coded v2 confirmatory grid")
    for group in gate["pinned_files"].values():
        for record in group:
            verify_pin(record, rehash=rehash)
    for system, index_record in gate["index_records"].items():
        if system not in ("bci", "wow", "serf"):
            raise RuntimeError(f"unexpected frozen index system: {system}")
        verify_pin(index_record["manifest"], rehash=rehash)
        for record in index_record["contents"]:
            verify_pin(record, rehash=rehash)
    return gate, digest


def verify_schedule(schedule: dict[str, Any]) -> None:
    if schedule.get("schema") != "sift10m-range-v2-schedule/v1":
        raise RuntimeError("unsupported schedule schema")
    formal = schedule.get("formal_states", [])
    calibration = schedule.get("calibration_states", [])
    if len(formal) != 330 or len({row["state_id"] for row in formal}) != 330:
        raise RuntimeError("formal schedule is not exactly 330 unique states")
    if len(calibration) != 22 or len({row["state_id"] for row in calibration}) != 22:
        raise RuntimeError("calibration schedule is not exactly 22 unique states")
    expected = {
        (rep, workload, system, parameter)
        for rep in range(1, REPS + 1)
        for workload in WORKLOADS
        for system, values in PARAMETERS.items()
        for parameter in values
    }
    observed = {
        (int(row["rep"]), row["workload"], row["system"], int(row["parameter"]))
        for row in formal
    }
    if observed != expected:
        raise RuntimeError("formal schedule coverage differs from the frozen grid")
    expected_calibration = {
        ("original", system, parameter)
        for system, values in PARAMETERS.items()
        for parameter in values
    }
    observed_calibration = {
        (row["workload"], row["system"], int(row["parameter"])) for row in calibration
    }
    if observed_calibration != expected_calibration:
        raise RuntimeError("calibration schedule coverage differs from the 22-point grid")


def artifact_groups_from_gate(gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for records in gate["pinned_files"].values() for record in records]
