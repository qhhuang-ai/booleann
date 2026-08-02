#!/usr/bin/env python3
"""Checked binding for the project-owned four-family native exact leaf.

Request descriptors and posting validation are built outside timed execution.
The timed call performs no Python per-ID loop and writes only into caller-owned
int32 output and uint64 audit arrays.  ``ctypes.CDLL`` releases the GIL while
the C++ merge/filter kernel runs.
"""

from __future__ import annotations

import ctypes
import math
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native/leaf_posting_merge.cpp"
DEFAULT_LIBRARY = HERE / "native/libleaf_posting_merge.so"

ABI_VERSION = 1
AUDIT_SLOTS = 8
UNIVERSE = 1_000_000
LANES = 8
OUTPUT_CAPACITY_IDS = 33_646
OUTPUT_BYTES_EIGHT_LANES = LANES * OUTPUT_CAPACITY_IDS * 4
AUDIT_BYTES_EIGHT_LANES = LANES * AUDIT_SLOTS * 8
SCORER_BYTES_EIGHT_LANES = LANES * 168
TOTAL_LEAF_BYTES = (
    OUTPUT_BYTES_EIGHT_LANES
    + AUDIT_BYTES_EIGHT_LANES
    + SCORER_BYTES_EIGHT_LANES
)

MODES = {"equality": 0, "conjunction": 1, "range": 2, "dnf2": 3}
VALUE_NONE = 0
VALUE_FLOAT64 = 1
VALUE_INT64 = 2
STATUS = {
    0: "OK",
    -1: "BAD_ABI",
    -2: "BAD_ARGUMENT",
    -3: "BAD_POSTING",
    -4: "BAD_INTERVAL",
    -5: "MISSING_OR_MULTIPLE_SELF",
    -6: "OUTPUT_OVERFLOW",
    -7: "NONCANONICAL_OUTPUT",
}
AUDIT_NAMES = (
    "abi_version",
    "output_rows",
    "removed_self",
    "left_rows_visited",
    "right_rows_visited",
    "unique_candidates",
    "predicate_checks",
    "overflow_required_minimum",
)

I32_PTR = ctypes.POINTER(ctypes.c_int32)
U64_PTR = ctypes.POINTER(ctypes.c_uint64)
NULL_I32 = I32_PTR()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _i32(array: np.ndarray, label: str) -> np.ndarray:
    require(
        type(array) is np.ndarray
        and array.dtype == np.int32
        and array.ndim == 1
        and array.flags.c_contiguous,
        f"{label} must be contiguous one-dimensional int32",
    )
    return array


def _u64(array: np.ndarray, label: str) -> np.ndarray:
    require(
        type(array) is np.ndarray
        and array.dtype == np.uint64
        and array.ndim == 1
        and array.flags.c_contiguous,
        f"{label} must be contiguous one-dimensional uint64",
    )
    return array


def _bound(value: Any) -> float:
    if value == "+inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    result = float(value)
    require(not math.isnan(result), "interval endpoint is NaN")
    return result


def _numeric_descriptor(
    interval: Mapping[str, Any] | None,
    numeric: Mapping[str, np.ndarray],
    universe: int,
) -> tuple[np.ndarray | None, int, float, float, int]:
    if interval is None:
        return None, VALUE_NONE, 0.0, 0.0, 0
    attribute = str(interval["attribute"])
    require(attribute in numeric, f"numeric attribute absent: {attribute}")
    values = numeric[attribute]
    require(
        type(values) is np.ndarray
        and values.ndim == 1
        and values.flags.c_contiguous
        and len(values) == universe,
        f"{attribute}: numeric column contract drift",
    )
    if values.dtype == np.float64:
        kind = VALUE_FLOAT64
    elif values.dtype == np.int64:
        kind = VALUE_INT64
    else:
        raise RuntimeError(f"{attribute}: unsupported dtype {values.dtype}")
    lo, hi = _bound(interval["lo"]), _bound(interval["hi"])
    require(lo < hi, f"{attribute}: empty or reversed interval")
    return values, kind, lo, hi, int(attribute == "similarity")


class CompiledLeafRequest:
    """Immutable pointer descriptor compiled before a measured period."""

    __slots__ = (
        "family", "mode", "left", "right", "left_pointer", "right_pointer",
        "left_rows", "right_rows", "left_values", "right_values",
        "left_value_pointer", "right_value_pointer", "left_kind",
        "right_kind", "left_lo", "left_hi", "right_lo", "right_hi",
        "left_similarity", "right_similarity", "self_id", "universe",
    )

    def __init__(
        self,
        *,
        family: str,
        left: np.ndarray,
        right: np.ndarray | None,
        left_numeric: tuple[np.ndarray | None, int, float, float, int],
        right_numeric: tuple[np.ndarray | None, int, float, float, int],
        self_id: int,
        universe: int,
    ) -> None:
        require(family in MODES, f"unknown family {family}")
        self.family = family
        self.mode = MODES[family]
        self.left = _i32(left, "left posting")
        self.right = None if right is None else _i32(right, "right posting")
        self.left_pointer = self.left.ctypes.data_as(I32_PTR)
        self.right_pointer = (
            NULL_I32 if self.right is None else self.right.ctypes.data_as(I32_PTR)
        )
        self.left_rows = len(self.left)
        self.right_rows = 0 if self.right is None else len(self.right)
        (
            self.left_values,
            self.left_kind,
            self.left_lo,
            self.left_hi,
            self.left_similarity,
        ) = left_numeric
        (
            self.right_values,
            self.right_kind,
            self.right_lo,
            self.right_hi,
            self.right_similarity,
        ) = right_numeric
        self.left_value_pointer = ctypes.c_void_p(
            0 if self.left_values is None else int(self.left_values.ctypes.data)
        )
        self.right_value_pointer = ctypes.c_void_p(
            0 if self.right_values is None else int(self.right_values.ctypes.data)
        )
        require(0 <= int(self_id) < universe, "self ID outside universe")
        self.self_id = int(self_id)
        self.universe = int(universe)


class NativeLeafMerge:
    """Reentrant native ABI; all mutable state belongs to the caller."""

    def __init__(self, library: Path = DEFAULT_LIBRARY) -> None:
        self.path = library.resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        self.library.laion1m_leaf_merge_abi_version_v1.restype = ctypes.c_uint32
        self.library.laion1m_leaf_merge_audit_slots_v1.restype = ctypes.c_uint64
        require(
            int(self.library.laion1m_leaf_merge_abi_version_v1()) == ABI_VERSION
            and int(self.library.laion1m_leaf_merge_audit_slots_v1())
            == AUDIT_SLOTS,
            "native leaf ABI drift",
        )
        self.validate_fn = self.library.laion1m_leaf_validate_posting_v1
        self.validate_fn.argtypes = [I32_PTR, ctypes.c_uint64, ctypes.c_int32]
        self.validate_fn.restype = ctypes.c_int32
        self.execute_fn = self.library.laion1m_leaf_sorted_posting_merge_v1
        self.execute_fn.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            I32_PTR,
            ctypes.c_uint64,
            I32_PTR,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            I32_PTR,
            ctypes.c_uint64,
            U64_PTR,
            ctypes.c_uint64,
        ]
        self.execute_fn.restype = ctypes.c_int32

    def validate_posting(self, posting: np.ndarray, universe: int = UNIVERSE) -> None:
        values = _i32(posting, "posting")
        status = int(
            self.validate_fn(
                values.ctypes.data_as(I32_PTR), len(values), int(universe)
            )
        )
        require(status == 0, f"posting preflight failed: {STATUS.get(status, status)}")

    def compile_request(
        self,
        row: Mapping[str, Any],
        postings: list[np.ndarray],
        numeric: Mapping[str, np.ndarray],
        *,
        universe: int = UNIVERSE,
        validate_postings: bool = True,
    ) -> CompiledLeafRequest:
        family = str(row["family"])
        require(family in MODES, f"unknown family {family}")
        primary = int(row["primary_token_id"])
        require(0 <= primary < len(postings), "primary token outside postings")
        left = postings[primary]
        secondary = row.get("secondary_token_id")
        right = None
        if family in {"conjunction", "dnf2"}:
            require(type(secondary) is int, f"{family}: secondary token absent")
            require(0 <= int(secondary) < len(postings), "secondary token outside postings")
            right = postings[int(secondary)]
        if validate_postings:
            self.validate_posting(left, universe)
            if right is not None:
                self.validate_posting(right, universe)
        left_interval = row.get("interval_1") if family in {"range", "dnf2"} else None
        right_interval = row.get("interval_2") if family == "dnf2" else None
        return CompiledLeafRequest(
            family=family,
            left=left,
            right=right,
            left_numeric=_numeric_descriptor(left_interval, numeric, universe),
            right_numeric=_numeric_descriptor(right_interval, numeric, universe),
            self_id=int(row["query_base_id"]),
            universe=universe,
        )

    def execute_compiled_pointers(
        self,
        route: CompiledLeafRequest,
        output_pointer: I32_PTR,
        output_capacity: int,
        audit_pointer: U64_PTR,
        audit_slots: int = AUDIT_SLOTS,
    ) -> int:
        """Timed primitive: no Python per-ID work and no native heap allocation."""
        status = int(
            self.execute_fn(
                ABI_VERSION,
                route.mode,
                route.left_pointer,
                route.left_rows,
                route.right_pointer,
                route.right_rows,
                route.left_value_pointer,
                route.left_kind,
                route.left_lo,
                route.left_hi,
                route.left_similarity,
                route.right_value_pointer,
                route.right_kind,
                route.right_lo,
                route.right_hi,
                route.right_similarity,
                route.self_id,
                route.universe,
                output_pointer,
                int(output_capacity),
                audit_pointer,
                int(audit_slots),
            )
        )
        if status != 0:
            raise RuntimeError(f"native leaf failed: {STATUS.get(status, status)}")
        return status

    def execute_compiled(
        self,
        route: CompiledLeafRequest,
        output: np.ndarray,
        audit: np.ndarray,
    ) -> np.ndarray:
        output = _i32(output, "output")
        audit = _u64(audit, "audit")
        require(len(audit) >= AUDIT_SLOTS, "audit descriptor too short")
        self.execute_compiled_pointers(
            route,
            output.ctypes.data_as(I32_PTR),
            len(output),
            audit.ctypes.data_as(U64_PTR),
            len(audit),
        )
        rows = int(audit[1])
        require(
            int(audit[0]) == ABI_VERSION
            and int(audit[2]) == 1
            and 0 <= rows <= len(output)
            and (rows < 2 or bool(np.all(output[1:rows] > output[: rows - 1]))),
            "native leaf audit/output invariant failed",
        )
        return output[:rows]


class LeafLanePool:
    """Eight preallocated arenas with explicit batch/lane generation binding."""

    def __init__(
        self,
        native: NativeLeafMerge,
        *,
        lanes: int = LANES,
        capacity_ids: int = OUTPUT_CAPACITY_IDS,
    ) -> None:
        require(lanes == LANES, "native leaf requires exactly eight lanes")
        require(
            capacity_ids == OUTPUT_CAPACITY_IDS,
            "native leaf requires the frozen 33,646-ID capacity",
        )
        self.native = native
        self.lanes = lanes
        self.capacity_ids = capacity_ids
        self._outputs = [
            np.empty(capacity_ids, dtype=np.int32) for _ in range(lanes)
        ]
        self._audits = [
            np.empty(AUDIT_SLOTS, dtype=np.uint64) for _ in range(lanes)
        ]
        self._output_pointers = [
            value.ctypes.data_as(I32_PTR) for value in self._outputs
        ]
        self._audit_pointers = [
            value.ctypes.data_as(U64_PTR) for value in self._audits
        ]
        self._local = threading.local()
        self._lock = threading.Lock()
        self._generation = 0
        self._prepared = False
        self._bindings: dict[int, int] = {}

    @property
    def allocated_output_bytes(self) -> int:
        return sum(int(value.nbytes) for value in self._outputs)

    @property
    def allocated_audit_bytes(self) -> int:
        return sum(int(value.nbytes) for value in self._audits)

    def addresses(self) -> dict[str, list[int]]:
        return {
            "output": [int(value.ctypes.data) for value in self._outputs],
            "audit": [int(value.ctypes.data) for value in self._audits],
        }

    def prepare_batch(self) -> None:
        with self._lock:
            require(not self._prepared, "native leaf batch already prepared")
            self._bindings.clear()
            self._generation += 1
            self._prepared = True

    def lane_ready(self, lane_id: int) -> None:
        require(0 <= lane_id < self.lanes, "native leaf lane outside range")
        thread_id = threading.get_ident()
        with self._lock:
            require(
                self._prepared
                and thread_id not in self._bindings
                and lane_id not in self._bindings.values(),
                "native leaf duplicate thread/lane binding",
            )
            self._bindings[thread_id] = lane_id
        self._local.generation = self._generation
        self._local.lane_id = lane_id
        self._local.active = False
        self._local.cursor = 0

    def validate_batch_bindings(self) -> None:
        with self._lock:
            require(
                self._prepared
                and set(self._bindings.values()) == set(range(self.lanes)),
                "native leaf batch lacks eight unique lane bindings",
            )

    def begin_request(self) -> None:
        require(
            self._prepared
            and getattr(self._local, "generation", None) == self._generation
            and not getattr(self._local, "active", False),
            "native leaf request outside current lane generation",
        )
        self._local.cursor = 0
        self._local.active = True

    def execute(self, route: CompiledLeafRequest) -> np.ndarray:
        require(getattr(self._local, "active", False), "native leaf execute without begin")
        lane_id = int(self._local.lane_id)
        self.native.execute_compiled_pointers(
            route,
            self._output_pointers[lane_id],
            self.capacity_ids,
            self._audit_pointers[lane_id],
            AUDIT_SLOTS,
        )
        rows = int(self._audits[lane_id][1])
        require(
            int(self._audits[lane_id][0]) == ABI_VERSION
            and int(self._audits[lane_id][2]) == 1
            and 0 < rows <= self.capacity_ids,
            "native leaf lane audit drift",
        )
        self._local.cursor = rows
        return self._outputs[lane_id][:rows]

    def current_lane_arena(self) -> tuple[np.ndarray, int]:
        require(
            self._prepared
            and getattr(self._local, "generation", None) == self._generation
            and getattr(self._local, "active", False),
            "native leaf arena outside active request",
        )
        lane_id = int(self._local.lane_id)
        return self._outputs[lane_id], int(self._local.cursor)

    def current_audit(self) -> np.ndarray:
        require(getattr(self._local, "active", False), "native leaf audit outside request")
        return self._audits[int(self._local.lane_id)]

    def end_request(self) -> int:
        require(getattr(self._local, "active", False), "native leaf end without begin")
        cursor = int(self._local.cursor)
        require(cursor > 0, "native leaf ended without exact support")
        self._local.active = False
        return cursor

    def abort_request(self) -> None:
        self._local.active = False
        self._local.cursor = 0

    def finish_batch(self) -> None:
        self.validate_batch_bindings()
        with self._lock:
            self._prepared = False

    def release(self) -> dict[str, int]:
        require(not self._prepared, "cannot release native leaf during batch")
        output_bytes = self.allocated_output_bytes
        audit_bytes = self.allocated_audit_bytes
        self._output_pointers.clear()
        self._audit_pointers.clear()
        self._outputs.clear()
        self._audits.clear()
        return {
            "released_output_bytes": output_bytes,
            "released_audit_bytes": audit_bytes,
            "released_total_bytes": output_bytes + audit_bytes,
        }


def audit_dict(audit: np.ndarray) -> dict[str, int]:
    values = _u64(audit, "audit")
    require(len(values) >= AUDIT_SLOTS, "audit descriptor too short")
    return {name: int(values[index]) for index, name in enumerate(AUDIT_NAMES)}


def allocated_capacity_ledger() -> dict[str, int]:
    return {
        "lanes": LANES,
        "output_capacity_ids_per_lane": OUTPUT_CAPACITY_IDS,
        "output_scratch_bytes_eight_lanes": OUTPUT_BYTES_EIGHT_LANES,
        "audit_slots_per_lane": AUDIT_SLOTS,
        "audit_bytes_eight_lanes": AUDIT_BYTES_EIGHT_LANES,
        "scorer_workspace_bytes_eight_lanes": SCORER_BYTES_EIGHT_LANES,
        "native_leaf_total_bytes": TOTAL_LEAF_BYTES,
    }
