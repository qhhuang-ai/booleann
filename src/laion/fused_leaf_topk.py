#!/usr/bin/env python3
"""Checked binding for the one-pass project-owned exact range/DNF leaf."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native/fused_leaf_topk.cpp"
DEFAULT_LIBRARY = HERE / "native/libfused_leaf_topk.so"

ABI_VERSION = 1
DIMENSION = 512
K = 10
AUDIT_SLOTS = 12
MODE_RANGE = 2
MODE_DNF2 = 3
MODES = {"range": MODE_RANGE, "dnf2": MODE_DNF2}
VALUE_FLOAT64 = 1
VALUE_INT64 = 2

STATUS = {
    0: "OK",
    -1: "BAD_ABI",
    -2: "BAD_ARGUMENT",
    -3: "BAD_POSTING",
    -4: "BAD_INTERVAL",
    -5: "MISSING_OR_MULTIPLE_SELF",
    -6: "SUPPORT_BELOW_K",
    -7: "NONFINITE_DISTANCE",
}
AUDIT_NAMES = (
    "abi_version",
    "mode",
    "left_rows_visited",
    "right_rows_visited",
    "predicate_checks",
    "clause_admissions",
    "unique_before_self",
    "removed_self",
    "distance_rows",
    "both_clauses_admitted",
    "cutoff_tie",
    "interleaved_batches",
)

F32_PTR = ctypes.POINTER(ctypes.c_float)
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


def _bound(value: Any) -> float:
    if value == "+inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    result = float(value)
    require(not math.isnan(result), "interval endpoint is NaN")
    return result


def _numeric_descriptor(
    interval: Mapping[str, Any],
    numeric: Mapping[str, np.ndarray],
    base_rows: int,
) -> tuple[np.ndarray, int, float, float, int]:
    attribute = str(interval["attribute"])
    require(attribute in numeric, f"numeric attribute absent: {attribute}")
    values = numeric[attribute]
    require(
        type(values) is np.ndarray
        and values.ndim == 1
        and values.flags.c_contiguous
        and len(values) == base_rows,
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


class CompiledFusedLeafRequest:
    """Immutable pointer descriptor built outside the measured call."""

    __slots__ = (
        "family",
        "mode",
        "base_rows",
        "left",
        "right",
        "left_pointer",
        "right_pointer",
        "left_rows",
        "right_rows",
        "left_values",
        "right_values",
        "left_value_pointer",
        "right_value_pointer",
        "left_kind",
        "right_kind",
        "left_lo",
        "left_hi",
        "right_lo",
        "right_hi",
        "left_similarity",
        "right_similarity",
        "self_id",
    )

    def __init__(
        self,
        *,
        family: str,
        base_rows: int,
        left: np.ndarray,
        right: np.ndarray | None,
        left_numeric: tuple[np.ndarray, int, float, float, int],
        right_numeric: tuple[np.ndarray, int, float, float, int] | None,
        self_id: int,
    ) -> None:
        require(family in MODES, f"unsupported fused family {family}")
        require(0 < int(base_rows) <= np.iinfo(np.int32).max, "invalid base rows")
        self.family = family
        self.mode = MODES[family]
        self.base_rows = int(base_rows)
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
        if right_numeric is None:
            self.right_values = None
            self.right_kind = 0
            self.right_lo = 0.0
            self.right_hi = 0.0
            self.right_similarity = 0
        else:
            (
                self.right_values,
                self.right_kind,
                self.right_lo,
                self.right_hi,
                self.right_similarity,
            ) = right_numeric
        self.left_value_pointer = ctypes.c_void_p(int(self.left_values.ctypes.data))
        self.right_value_pointer = ctypes.c_void_p(
            0 if self.right_values is None else int(self.right_values.ctypes.data)
        )
        require(0 <= int(self_id) < self.base_rows, "self ID outside base")
        self.self_id = int(self_id)


class NativeFusedLeafTopK:
    """Reentrant CDLL binding; the production call allocates no scratch."""

    def __init__(self, library: Path = DEFAULT_LIBRARY) -> None:
        self.path = library.resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        for name, restype in (
            ("abi_version", ctypes.c_uint32),
            ("dimension", ctypes.c_uint32),
            ("k", ctypes.c_uint32),
            ("audit_slots", ctypes.c_uint64),
        ):
            function = getattr(
                self.library, f"laion1m_fused_leaf_topk_{name}_v1"
            )
            function.argtypes = ()
            function.restype = restype
        observed = (
            int(self.library.laion1m_fused_leaf_topk_abi_version_v1()),
            int(self.library.laion1m_fused_leaf_topk_dimension_v1()),
            int(self.library.laion1m_fused_leaf_topk_k_v1()),
            int(self.library.laion1m_fused_leaf_topk_audit_slots_v1()),
        )
        require(
            observed == (ABI_VERSION, DIMENSION, K, AUDIT_SLOTS),
            f"native fused leaf ABI drift: {observed}",
        )
        cpu = self.library.laion1m_exact_topk_avx2_v1_cpu_supported
        cpu.argtypes = ()
        cpu.restype = ctypes.c_int
        require(bool(cpu()), "native fused leaf requires AVX2")

        self.validate_fn = (
            self.library.laion1m_fused_leaf_topk_validate_posting_v1
        )
        self.validate_fn.argtypes = [I32_PTR, ctypes.c_uint64, ctypes.c_uint64]
        self.validate_fn.restype = ctypes.c_int32
        self.execute_fn = (
            self.library.laion1m_fused_leaf_top10_i32_f32_retained_v1
        )
        self.execute_fn.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            F32_PTR,
            ctypes.c_uint64,
            F32_PTR,
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
            I32_PTR,
            F32_PTR,
            U64_PTR,
            ctypes.c_uint64,
        ]
        self.execute_fn.restype = ctypes.c_int32

    def validate_posting(self, posting: np.ndarray, base_rows: int) -> None:
        values = _i32(posting, "posting")
        status = int(
            self.validate_fn(
                values.ctypes.data_as(I32_PTR), len(values), int(base_rows)
            )
        )
        require(status == 0, f"posting preflight failed: {STATUS.get(status, status)}")

    def compile_request(
        self,
        row: Mapping[str, Any],
        postings: list[np.ndarray],
        numeric: Mapping[str, np.ndarray],
        *,
        base_rows: int,
        validate_postings: bool = True,
    ) -> CompiledFusedLeafRequest:
        family = str(row["family"])
        require(family in MODES, f"unsupported fused family {family}")
        primary = int(row["primary_token_id"])
        require(0 <= primary < len(postings), "primary token outside postings")
        left = postings[primary]
        right: np.ndarray | None = None
        if family == "dnf2":
            secondary = row.get("secondary_token_id")
            require(type(secondary) is int, "dnf2 secondary token absent")
            require(
                0 <= secondary < len(postings),
                "secondary token outside postings",
            )
            right = postings[secondary]
        if validate_postings:
            self.validate_posting(left, base_rows)
            if right is not None:
                self.validate_posting(right, base_rows)
        first = row.get("interval_1")
        require(isinstance(first, Mapping), f"{family}: interval_1 absent")
        second = row.get("interval_2")
        if family == "dnf2":
            require(isinstance(second, Mapping), "dnf2 interval_2 absent")
        return CompiledFusedLeafRequest(
            family=family,
            base_rows=base_rows,
            left=left,
            right=right,
            left_numeric=_numeric_descriptor(first, numeric, base_rows),
            right_numeric=(
                None
                if family == "range"
                else _numeric_descriptor(second, numeric, base_rows)
            ),
            self_id=int(row["query_base_id"]),
        )

    @staticmethod
    def _base_query(base: np.ndarray, query: np.ndarray, base_rows: int) -> None:
        require(
            type(base) is np.ndarray
            and base.dtype == np.float32
            and base.shape == (base_rows, DIMENSION)
            and base.flags.c_contiguous,
            "base must be contiguous float32[N,512] matching the route",
        )
        require(
            type(query) is np.ndarray
            and query.dtype == np.float32
            and query.shape == (DIMENSION,)
            and query.flags.c_contiguous,
            "query must be contiguous float32[512]",
        )

    def execute_compiled_into(
        self,
        route: CompiledFusedLeafRequest,
        base: np.ndarray,
        query: np.ndarray,
        output_ids: np.ndarray,
        output_squared_l2: np.ndarray,
        audit: np.ndarray,
    ) -> None:
        require(
            isinstance(route, CompiledFusedLeafRequest),
            "route must be a CompiledFusedLeafRequest",
        )
        self._base_query(base, query, route.base_rows)
        require(
            type(output_ids) is np.ndarray
            and output_ids.dtype == np.int32
            and output_ids.shape == (K,)
            and output_ids.flags.c_contiguous
            and output_ids.flags.writeable,
            "output IDs must be writable contiguous int32[10]",
        )
        require(
            type(output_squared_l2) is np.ndarray
            and output_squared_l2.dtype == np.float32
            and output_squared_l2.shape == (K,)
            and output_squared_l2.flags.c_contiguous
            and output_squared_l2.flags.writeable,
            "output distances must be writable contiguous float32[10]",
        )
        require(
            type(audit) is np.ndarray
            and audit.dtype == np.uint64
            and audit.ndim == 1
            and len(audit) >= AUDIT_SLOTS
            and audit.flags.c_contiguous
            and audit.flags.writeable,
            "audit must be writable contiguous uint64[>=12]",
        )
        status = int(
            self.execute_fn(
                ABI_VERSION,
                route.mode,
                base.ctypes.data_as(F32_PTR),
                route.base_rows,
                query.ctypes.data_as(F32_PTR),
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
                output_ids.ctypes.data_as(I32_PTR),
                output_squared_l2.ctypes.data_as(F32_PTR),
                audit.ctypes.data_as(U64_PTR),
                len(audit),
            )
        )
        require(status == 0, f"native fused leaf failed: {STATUS.get(status, status)}")
        values = audit[:AUDIT_SLOTS]
        require(
            int(values[0]) == ABI_VERSION
            and int(values[1]) == route.mode
            and int(values[7]) == 1
            and int(values[8]) + 1 == int(values[6])
            and int(values[8]) >= K,
            "native fused leaf audit invariant failed",
        )
        order = np.lexsort((output_ids, output_squared_l2))
        require(
            len(np.unique(output_ids)) == K
            and np.array_equal(order, np.arange(K)),
            "native fused leaf top-k order/uniqueness failed",
        )

    def workspace(self) -> "FusedLeafTopKWorkspace":
        return FusedLeafTopKWorkspace(self)


class FusedLeafTopKWorkspace:
    """Fixed caller-owned 128-byte result/audit workspace."""

    __slots__ = ("native", "top_ids", "top_squared_l2", "audit")

    def __init__(self, native: NativeFusedLeafTopK) -> None:
        self.native = native
        self.top_ids = np.empty(K, dtype=np.int32)
        self.top_squared_l2 = np.empty(K, dtype=np.float32)
        self.audit = np.empty(AUDIT_SLOTS, dtype=np.uint64)

    @property
    def allocated_bytes(self) -> int:
        return int(self.top_ids.nbytes + self.top_squared_l2.nbytes + self.audit.nbytes)

    def run_into(
        self,
        route: CompiledFusedLeafRequest,
        base: np.ndarray,
        query: np.ndarray,
    ) -> dict[str, Any]:
        self.native.execute_compiled_into(
            route,
            base,
            query,
            self.top_ids,
            self.top_squared_l2,
            self.audit,
        )
        result = {
            name: int(self.audit[index])
            for index, name in enumerate(AUDIT_NAMES)
        }
        result.update(
            {
                "top_ids": self.top_ids,
                "top_squared_l2": self.top_squared_l2,
                "stable_order": "retained_squared_l2_then_id",
                "materialized_union_rows": 0,
                "workspace_bytes": self.allocated_bytes,
            }
        )
        return result
