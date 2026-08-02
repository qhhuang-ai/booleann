#!/usr/bin/env python3
"""Query-free binding for the packed low-memory fixed-block candidate v2.

The descriptor is one owner with the frozen capacity equation
``align64(128 + 4R + 2S + S + 20C)``.  ``R`` is the selected route count,
``S`` its segment count, and ``C`` its clause count.  Route words pack a
stable 16-bit plan ordinal with a 16-bit segment begin, so arbitrary selected
subsets do not require a dense route table.  No performance entry point is
provided here.
"""

from __future__ import annotations

import ctypes
import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import compact_support_arena as logical
import fixed_block_compositor as checked_v1


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE / "native/"
    "packed_fixed_block_compositor.cpp"
)
DEFAULT_LIBRARY = (
    HERE / "native/"
    "libpacked_fixed_block_compositor.so"
)
ABI_VERSION = 2
DIMENSION = 512
TOP_K = 10
MAX_OBJECTS = 64
MAX_SEGMENTS = 64
MAX_DECODED_ROWS = 14_044
LOCAL_SCRATCH_BYTES = 1_904
DECODE_SCRATCH_BYTES = MAX_DECODED_ROWS * 4
AUDIT_BYTES = 48
OUTPUT_BYTES = TOP_K * 8 + AUDIT_BYTES
LANE_WORKSPACE_BYTES = DECODE_SCRATCH_BYTES + LOCAL_SCRATCH_BYTES + OUTPUT_BYTES
EIGHT_LANE_WORKSPACE_BYTES = 8 * LANE_WORKSPACE_BYTES

U8P = ctypes.POINTER(ctypes.c_uint8)
I32P = ctypes.POINTER(ctypes.c_int32)
F32P = ctypes.POINTER(ctypes.c_float)
UPTR = ctypes.c_size_t


class FixedBlockAuditV2(ctypes.Structure):
    _fields_ = [
        ("decoded_input_rows", ctypes.c_uint64),
        ("segment_input_rows", ctypes.c_uint64),
        ("boundary_predicate_checks", ctypes.c_uint64),
        ("support_before_leave_one_out", ctypes.c_uint64),
        ("support_after_leave_one_out", ctypes.c_uint64),
        ("self_fragment_occurrences", ctypes.c_uint64),
    ]


class PackedRouteV2:
    """Ephemeral compile product; it is not retained by the packed owner."""

    __slots__ = (
        "family", "plan_ordinal", "request_id", "predicate_qid",
        "query_base_id", "predicate", "segment_positions",
        "segment_clause_flags", "clauses", "expected_decoded_input_rows",
        "expected_segment_input_rows", "expected_boundary_checks",
        "expected_support_before", "expected_support_after",
    )

    def __init__(
        self, *, family: str, plan_ordinal: int, request_id: int,
        predicate_qid: int, query_base_id: int, predicate: str,
        segment_positions: np.ndarray, segment_clause_flags: np.ndarray,
        clauses: tuple[tuple[int, float, float], ...],
        expected_decoded_input_rows: int,
        expected_segment_input_rows: int,
        expected_boundary_checks: int,
        expected_support_before: int | None,
        expected_support_after: int | None,
    ) -> None:
        self.family = family
        self.plan_ordinal = int(plan_ordinal)
        self.request_id = int(request_id)
        self.predicate_qid = int(predicate_qid)
        self.query_base_id = int(query_base_id)
        self.predicate = predicate
        self.segment_positions = segment_positions
        self.segment_clause_flags = segment_clause_flags
        self.clauses = clauses
        self.expected_decoded_input_rows = int(expected_decoded_input_rows)
        self.expected_segment_input_rows = int(expected_segment_input_rows)
        self.expected_boundary_checks = int(expected_boundary_checks)
        self.expected_support_before = expected_support_before
        self.expected_support_after = expected_support_after
        self.segment_positions.setflags(write=False)
        self.segment_clause_flags.setflags(write=False)

    @property
    def segment_count(self) -> int:
        return len(self.segment_positions)


def compile_packed_route_v2(
    row: Mapping[str, Any], fragment_cardinalities: Mapping[str, int],
    numeric: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray],
    *, plan_ordinal: int, universe: int = 1_000_000,
) -> PackedRouteV2:
    """Compile semantics only; no query outcome or performance field is read."""
    checked = checked_v1.compile_formula_route(
        row, fragment_cardinalities, numeric, valid, universe=universe
    )
    if not 0 <= int(plan_ordinal) <= 0xFFFF:
        raise ValueError("stable plan ordinal outside uint16")
    clause_inputs: list[tuple[int, Mapping[str, Any]]] = [
        (int(row["primary_token_id"]), row["interval_1"])
    ]
    if checked.family == "dnf2":
        clause_inputs.append(
            (int(row["secondary_token_id"]), row["interval_2"])
        )
    positions: list[int] = []
    flags: list[int] = []
    clauses: list[tuple[int, float, float]] = []
    unique_cardinalities: dict[str, int] = {}
    segment_rows = 0
    boundary_checks = 0
    for clause_index, (token, interval) in enumerate(clause_inputs):
        attribute = str(interval["attribute"])
        if attribute not in logical.ATTRIBUTE_INDEX:
            raise KeyError(f"noncanonical numeric attribute: {attribute}")
        lower = checked_v1._decode_endpoint(interval["lo"])
        upper = checked_v1._decode_endpoint(interval["hi"])
        clauses.append((logical.ATTRIBUTE_INDEX[attribute], lower, upper))
        full = tuple(int(value) for value in interval["full_coarse_blocks"])
        boundary = tuple(
            int(value) for value in interval["boundary_coarse_blocks"]
        )
        for is_boundary, blocks in ((False, full), (True, boundary)):
            for block in blocks:
                object_id = f"X|{token}|{attribute}|{block}"
                cardinality = int(fragment_cardinalities[object_id])
                position = logical.key_to_position(object_id)
                if not 0 <= position <= 0xFFFF:
                    raise OverflowError("logical object position exceeds uint16")
                positions.append(position)
                flags.append(clause_index + 1 if is_boundary else 0)
                segment_rows += cardinality
                if is_boundary:
                    boundary_checks += cardinality
                previous = unique_cardinalities.setdefault(
                    object_id, cardinality
                )
                if previous != cardinality:
                    raise ValueError("fragment cardinality drift")
    decoded = sum(unique_cardinalities.values())
    if not (
        len(positions) == checked.segment_count
        and decoded == checked.expected_decoded_input_rows
        and segment_rows == checked.expected_segment_input_rows
        and boundary_checks == checked.expected_boundary_checks
    ):
        raise AssertionError("packed route differs from checked v1 semantics")
    return PackedRouteV2(
        family=checked.family,
        plan_ordinal=plan_ordinal,
        request_id=checked.request_id,
        predicate_qid=checked.predicate_qid,
        query_base_id=checked.query_base_id,
        predicate=checked.predicate,
        segment_positions=np.asarray(positions, dtype=np.uint16),
        segment_clause_flags=np.asarray(flags, dtype=np.uint8),
        clauses=tuple(clauses),
        expected_decoded_input_rows=decoded,
        expected_segment_input_rows=segment_rows,
        expected_boundary_checks=boundary_checks,
        expected_support_before=checked.expected_support_before,
        expected_support_after=checked.expected_support_after,
    )


def _align64(value: int) -> int:
    return (int(value) + 63) // 64 * 64


class PackedDescriptorTableV2:
    """One packed owner for an arbitrary same-family selected route subset."""

    HEADER = struct.Struct("<8s9I84s")
    CLAUSE = struct.Struct("<Idd")
    MAGIC = b"L1MFBP2\0"
    VERSION = 2

    __slots__ = (
        "owner", "routes", "segments", "clauses", "family",
        "plan_ordinals", "layout", "u24_owner_view", "_u24_pointer",
        "_attribute_values", "_attribute_valid", "_attribute_kinds",
        "_source_owners",
    )

    def __init__(
        self, routes: Sequence[PackedRouteV2],
        u24_owner_view: checked_v1.CompactU24OwnerView,
        numeric: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray],
        *, universe: int = 1_000_000,
    ) -> None:
        ordered = tuple(sorted(routes, key=lambda route: route.plan_ordinal))
        if not ordered or len({route.plan_ordinal for route in ordered}) != len(ordered):
            raise ValueError("packed descriptor needs unique selected routes")
        family = ordered[0].family
        clauses_per_route = 1 if family == "range" else 2
        if any(
            route.family != family or len(route.clauses) != clauses_per_route
            for route in ordered
        ):
            raise ValueError("packed descriptor must contain one formula family")
        route_count = len(ordered)
        segment_count = sum(route.segment_count for route in ordered)
        clause_count = sum(len(route.clauses) for route in ordered)
        if not 0 < segment_count <= 0xFFFF:
            raise OverflowError("packed segment population exceeds uint16 begin")

        route_offset = self.HEADER.size
        position_offset = route_offset + 4 * route_count
        flag_offset = position_offset + 2 * segment_count
        clause_offset = flag_offset + segment_count
        logical_bytes = clause_offset + self.CLAUSE.size * clause_count
        capacity = _align64(logical_bytes)
        owner = bytearray(capacity)
        self.HEADER.pack_into(
            owner, 0, self.MAGIC, self.VERSION, capacity, route_count,
            segment_count, clause_count, route_offset, position_offset,
            flag_offset, clause_offset, b"\0" * 84,
        )
        segment_cursor = 0
        clause_cursor = 0
        for slot, route in enumerate(ordered):
            struct.pack_into(
                "<I", owner, route_offset + 4 * slot,
                (route.plan_ordinal << 16) | segment_cursor,
            )
            count = route.segment_count
            owner[
                position_offset + 2 * segment_cursor:
                position_offset + 2 * (segment_cursor + count)
            ] = route.segment_positions.astype("<u2", copy=False).tobytes()
            owner[
                flag_offset + segment_cursor:
                flag_offset + segment_cursor + count
            ] = route.segment_clause_flags.tobytes()
            for clause in route.clauses:
                self.CLAUSE.pack_into(
                    owner, clause_offset + self.CLAUSE.size * clause_cursor,
                    *clause,
                )
                clause_cursor += 1
            segment_cursor += count
        if segment_cursor != segment_count or clause_cursor != clause_count:
            raise AssertionError("packed descriptor terminal drift")

        # All required logical objects, including required zero-cardinality
        # objects, must be selected in the U24 owner.
        for route in ordered:
            for position in route.segment_positions:
                object_id = logical.position_to_key(int(position))
                if object_id not in u24_owner_view:
                    raise KeyError(f"required U24 object absent: {object_id}")

        attribute_values: list[int] = []
        attribute_valid: list[int] = []
        attribute_kinds: list[int] = []
        source_owners: list[np.ndarray] = []
        for attribute in logical.ATTRIBUTES:
            value_ptr, valid_ptr, kind = checked_v1._checked_column(
                numeric[attribute], valid[attribute], universe, attribute
            )
            attribute_values.append(value_ptr)
            attribute_valid.append(valid_ptr)
            attribute_kinds.append(kind)
            source_owners.extend((numeric[attribute], valid[attribute]))

        self.owner = owner
        self.routes = route_count
        self.segments = segment_count
        self.clauses = clause_count
        self.family = family
        self.plan_ordinals = tuple(route.plan_ordinal for route in ordered)
        self.layout = {
            "schema": "laion1m-lowmem-fixedblock-packed-descriptor/v2",
            "selected_subset": True,
            "stable_plan_ordinal_in_route_word": True,
            "capacity_equation": "align64(128 + 4R + 2S + S + 20C)",
            "route_count": route_count,
            "segment_count": segment_count,
            "clause_count": clause_count,
            "logical_bytes": logical_bytes,
            "allocated_capacity_bytes": capacity,
            "route_words_offset": route_offset,
            "segment_logical_positions_offset": position_offset,
            "segment_clause_flags_offset": flag_offset,
            "clause_records_offset": clause_offset,
            "per_route_numpy_owners": 0,
            "global_payload_pointer_table": False,
            "global_payload_row_table": False,
        }
        self.u24_owner_view = u24_owner_view
        self._u24_pointer = int(u24_owner_view._base_address)
        self._attribute_values = tuple(attribute_values)
        self._attribute_valid = tuple(attribute_valid)
        self._attribute_kinds = tuple(attribute_kinds)
        self._source_owners = tuple(source_owners)

    @property
    def allocated_capacity_bytes(self) -> int:
        return len(self.owner)


class LowMemoryFixedBlockPackedNativeV2:
    def __init__(self, library: Path = DEFAULT_LIBRARY) -> None:
        self.path = library.resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        probes = (
            ("abi_version", ABI_VERSION),
            ("max_objects", MAX_OBJECTS),
            ("max_segments", MAX_SEGMENTS),
            ("max_decoded_rows", MAX_DECODED_ROWS),
            ("lane_decode_scratch_bytes", DECODE_SCRATCH_BYTES),
            ("lane_local_scratch_bytes", LOCAL_SCRATCH_BYTES),
            ("lane_output_workspace_bytes", OUTPUT_BYTES),
            ("audit_struct_bytes", AUDIT_BYTES),
        )
        for suffix, expected in probes:
            function = getattr(
                self.library,
                f"laion1m_lowmem_fixedblock_packed_v2_{suffix}",
            )
            function.argtypes = ()
            function.restype = ctypes.c_uint32
            observed = int(function())
            if observed != expected:
                raise RuntimeError(
                    f"packed fixed-block ABI drift: {suffix}={observed}"
                )
        retained = (
            self.library.laion1m_exact_topk_avx2_v1_retained_prefetch_policy
        )
        retained.argtypes = ()
        retained.restype = ctypes.c_uint32
        if int(retained()) != 2:
            raise RuntimeError("packed compositor is not retained NTA ABI-v7")
        self.function = (
            self.library.laion1m_lowmem_fixedblock_exact_top10_packed_v2
        )
        self.function.argtypes = (
            F32P, ctypes.c_uint64, F32P,
            U8P, ctypes.c_uint64, U8P, ctypes.c_uint64,
            ctypes.c_uint64,
            UPTR, UPTR, ctypes.c_uint8,
            UPTR, UPTR, ctypes.c_uint8,
            UPTR, UPTR, ctypes.c_uint8,
            ctypes.c_int32,
            I32P, ctypes.c_uint64, U8P, ctypes.c_uint64,
            I32P, F32P, ctypes.POINTER(FixedBlockAuditV2),
            I32P, ctypes.c_uint64,
        )
        self.function.restype = ctypes.c_int

    def workspace(
        self, *, diagnostic_support: bool = False
    ) -> "PackedWorkspaceV2":
        return PackedWorkspaceV2(self, diagnostic_support=diagnostic_support)


class PackedWorkspaceV2:
    __slots__ = (
        "native", "decode_scratch", "local_scratch", "top_ids",
        "top_squared_l2", "audit", "diagnostic_support",
    )

    def __init__(
        self, native: LowMemoryFixedBlockPackedNativeV2,
        *, diagnostic_support: bool,
    ) -> None:
        self.native = native
        self.decode_scratch = np.empty(MAX_DECODED_ROWS, dtype=np.int32)
        # uint64 ownership guarantees the alignment required by the native
        # scratch struct without adding hidden padding or a second allocation.
        self.local_scratch = np.empty(
            LOCAL_SCRATCH_BYTES // 8, dtype=np.uint64
        )
        if (
            self.local_scratch.nbytes != LOCAL_SCRATCH_BYTES
            or self.local_scratch.ctypes.data % 8
        ):
            raise AssertionError("caller-owned local scratch alignment drift")
        self.top_ids = np.empty(TOP_K, dtype=np.int32)
        self.top_squared_l2 = np.empty(TOP_K, dtype=np.float32)
        self.audit = FixedBlockAuditV2()
        self.diagnostic_support = (
            np.empty(MAX_DECODED_ROWS, dtype=np.int32)
            if diagnostic_support else None
        )

    @property
    def production_allocated_bytes(self) -> int:
        return (
            self.decode_scratch.nbytes + self.local_scratch.nbytes
            + self.top_ids.nbytes + self.top_squared_l2.nbytes
            + ctypes.sizeof(self.audit)
        )

    def run_into(
        self, base: np.ndarray, query: np.ndarray,
        table: PackedDescriptorTableV2, *, plan_ordinal: int,
        self_id: int, expected: PackedRouteV2 | None = None,
    ) -> FixedBlockAuditV2:
        if not (
            isinstance(base, np.ndarray) and base.dtype == np.float32
            and base.ndim == 2 and base.shape[1] == DIMENSION
            and base.flags.c_contiguous
            and isinstance(query, np.ndarray) and query.dtype == np.float32
            and query.shape == (DIMENSION,) and query.flags.c_contiguous
            and isinstance(table, PackedDescriptorTableV2)
        ):
            raise TypeError("packed fixed-block request contract failed")
        owner_pointer = ctypes.addressof(ctypes.c_uint8.from_buffer(table.owner))
        diagnostic = self.diagnostic_support
        status = int(self.native.function(
            base.ctypes.data_as(F32P), len(base), query.ctypes.data_as(F32P),
            ctypes.cast(table._u24_pointer, U8P),
            len(table.u24_owner_view.owner),
            ctypes.cast(owner_pointer, U8P), len(table.owner), plan_ordinal,
            table._attribute_values[0], table._attribute_valid[0],
            table._attribute_kinds[0],
            table._attribute_values[1], table._attribute_valid[1],
            table._attribute_kinds[1],
            table._attribute_values[2], table._attribute_valid[2],
            table._attribute_kinds[2], self_id,
            self.decode_scratch.ctypes.data_as(I32P), len(self.decode_scratch),
            self.local_scratch.ctypes.data_as(U8P), self.local_scratch.nbytes,
            self.top_ids.ctypes.data_as(I32P),
            self.top_squared_l2.ctypes.data_as(F32P), ctypes.byref(self.audit),
            (ctypes.cast(0, I32P) if diagnostic is None
             else diagnostic.ctypes.data_as(I32P)),
            (0 if diagnostic is None else len(diagnostic)),
        ))
        if status:
            meanings = {
                1: "null pointer/output contract",
                2: "base or decoded-input capacity",
                3: "fewer than ten rows after leave-one-out",
                4: "decoded ID outside base domain",
                5: "fragment not strictly increasing",
                6: "non-finite retained distance",
                7: "invalid object/segment count",
                8: "required U24 object/payload missing",
                9: "malformed clause descriptor",
                10: "query base absent before leave-one-out",
                11: "diagnostic support buffer contract",
                12: "retained U24 decoder failure",
                13: "packed descriptor/U24 owner drift",
            }
            raise ValueError(
                f"packed fixed-block failed closed ({status}: "
                f"{meanings.get(status, 'unknown')})"
            )
        if expected is not None:
            observed = (
                int(self.audit.decoded_input_rows),
                int(self.audit.segment_input_rows),
                int(self.audit.boundary_predicate_checks),
                int(self.audit.support_before_leave_one_out),
                int(self.audit.support_after_leave_one_out),
            )
            declared = (
                expected.expected_decoded_input_rows,
                expected.expected_segment_input_rows,
                expected.expected_boundary_checks,
                expected.expected_support_before,
                expected.expected_support_after,
            )
            if observed != declared:
                raise AssertionError(
                    f"packed compositor audit mismatch: {observed} != {declared}"
                )
        return self.audit

    def diagnostic_support_view(self) -> np.ndarray:
        if self.diagnostic_support is None:
            raise RuntimeError("diagnostic support was not provisioned")
        return self.diagnostic_support[
            : int(self.audit.support_after_leave_one_out)
        ]


def descriptor_capacity(route_count: int, segments: int, clauses: int) -> int:
    return _align64(128 + 4 * route_count + 3 * segments + 20 * clauses)


def integration_interface() -> dict[str, Any]:
    return {
        "schema": "laion1m-lowmem-fixedblock-packed-integration/v2",
        "status": "QUERY_FREE_CANDIDATE_ONLY",
        "selected_subset_supported": True,
        "stable_plan_ordinal_lookup": "u16 high half of packed route word",
        "descriptor_capacity": "align64(128 + 4R + 3S + 20C)",
        "payload_lookup_inside_native_request": "U24 selection/rank/offset",
        "caller_owned_explicit_local_scratch_bytes": LOCAL_SCRATCH_BYTES,
        "lane_workspace_bytes": LANE_WORKSPACE_BYTES,
        "eight_lane_workspace_bytes": EIGHT_LANE_WORKSPACE_BYTES,
        "performance_started": False,
        "performance_authorized": False,
    }
