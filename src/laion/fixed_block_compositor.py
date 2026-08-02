#!/usr/bin/env python3
"""Checked binding for the project-owned low-memory fixed-block compositor.

The production call is one native invocation: it decodes referenced objects
from a single-owner U24 arena through the retained U24 decoder, applies each
segment's boundary predicate, performs global union/dedup and leave-one-out,
and computes exact retained-arithmetic top-10.  Route/payload compilation is
outside the future timer.  This module exposes no performance entry point.
"""

from __future__ import annotations

import ctypes
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import compact_support_arena as logical


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "native/"
    "fixed_block_compositor.cpp"
)
DEFAULT_LIBRARY = (
    HERE
    / "native/"
    "libfixed_block_compositor.so"
)
RETAINED_U24_LIBRARY = HERE / "native/libu24_decode.so"
IMPLEMENTATION_NOTE = HERE / "README.md"

ABI_VERSION = 1
DIMENSION = 512
TOP_K = 10
MAX_OBJECTS = 64
MAX_SEGMENTS = 64
MAX_GLOBAL_OBJECTS = 65_535
MAX_DECODED_ROWS = 14_044
BATCH_IDS = 64
LANES = 8
LANE_DECODE_SCRATCH_BYTES = MAX_DECODED_ROWS * 4
AUDIT_STRUCT_BYTES = 48
LANE_OUTPUT_WORKSPACE_BYTES = TOP_K * 8 + AUDIT_STRUCT_BYTES
LANE_WORKSPACE_BYTES = (
    LANE_DECODE_SCRATCH_BYTES + LANE_OUTPUT_WORKSPACE_BYTES
)
EIGHT_LANE_WORKSPACE_BYTES = LANES * LANE_WORKSPACE_BYTES
RETAINED_PREFETCH_POLICY_NTA = 2

U8_POINTER = ctypes.POINTER(ctypes.c_uint8)
U8_POINTER_POINTER = ctypes.POINTER(U8_POINTER)
U8_ARRAY_POINTER = ctypes.POINTER(ctypes.c_uint8)
U16_POINTER = ctypes.POINTER(ctypes.c_uint16)
U32_POINTER = ctypes.POINTER(ctypes.c_uint32)
U64_POINTER = ctypes.POINTER(ctypes.c_uint64)
UPTR_POINTER = ctypes.POINTER(ctypes.c_size_t)
F64_POINTER = ctypes.POINTER(ctypes.c_double)
F32_POINTER = ctypes.POINTER(ctypes.c_float)
I32_POINTER = ctypes.POINTER(ctypes.c_int32)


class FixedBlockAuditV1(ctypes.Structure):
    _fields_ = [
        ("decoded_input_rows", ctypes.c_uint64),
        ("segment_input_rows", ctypes.c_uint64),
        ("boundary_predicate_checks", ctypes.c_uint64),
        ("support_before_leave_one_out", ctypes.c_uint64),
        ("support_after_leave_one_out", ctypes.c_uint64),
        ("self_fragment_occurrences", ctypes.c_uint64),
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _decode_endpoint(value: Any) -> float:
    if value == "-inf":
        return -math.inf
    if value == "+inf":
        return math.inf
    if type(value) not in {int, float}:
        raise TypeError("interval endpoint is malformed")
    result = float(value)
    if math.isnan(result):
        raise ValueError("interval endpoint is NaN")
    return result


def _checked_column(
    values: np.ndarray, valid: np.ndarray, universe: int, attribute: str
) -> tuple[int, int, int]:
    if not (
        type(values) is np.ndarray
        and values.dtype in (np.dtype(np.float64), np.dtype(np.int64))
        and values.shape == (universe,)
        and values.flags.c_contiguous
        and type(valid) is np.ndarray
        and valid.dtype in (np.dtype(np.bool_), np.dtype(np.uint8))
        and valid.shape == (universe,)
        and valid.flags.c_contiguous
        and valid.itemsize == 1
    ):
        raise TypeError(f"{attribute}: numeric/valid column contract failed")
    return (
        int(values.ctypes.data),
        int(valid.ctypes.data),
        1 if values.dtype == np.float64 else 2,
    )


class CompiledFixedBlockRoute:
    """Ephemeral query-free route used to build one production SoA owner."""

    __slots__ = (
        "family", "request_id", "predicate_qid", "query_base_id",
        "predicate", "decode_object_ids", "decode_cardinalities",
        "segment_object_ordinals", "boundary_flags",
        "segment_attribute_names", "lower_bounds", "upper_bounds",
        "expected_decoded_input_rows", "expected_segment_input_rows",
        "expected_boundary_checks", "expected_support_before",
        "expected_support_after", "zero_cardinality_object_ids",
        "cross_clause_shared_object_count",
    )

    def __init__(
        self,
        *,
        family: str,
        request_id: int,
        predicate_qid: int,
        query_base_id: int,
        predicate: str,
        decode_object_ids: Sequence[str],
        decode_cardinalities: np.ndarray,
        segment_object_ordinals: np.ndarray,
        boundary_flags: np.ndarray,
        segment_attribute_names: Sequence[str],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        expected_support_before: int | None,
        expected_support_after: int | None,
    ) -> None:
        self.family = family
        self.request_id = int(request_id)
        self.predicate_qid = int(predicate_qid)
        self.query_base_id = int(query_base_id)
        self.predicate = predicate
        self.decode_object_ids = tuple(decode_object_ids)
        self.decode_cardinalities = decode_cardinalities
        self.segment_object_ordinals = segment_object_ordinals
        self.boundary_flags = boundary_flags
        self.segment_attribute_names = tuple(segment_attribute_names)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.expected_decoded_input_rows = int(
            np.sum(decode_cardinalities, dtype=np.uint64)
        )
        self.expected_segment_input_rows = int(np.sum(
            decode_cardinalities[segment_object_ordinals], dtype=np.uint64
        ))
        self.expected_boundary_checks = int(np.sum(
            decode_cardinalities[segment_object_ordinals][
                boundary_flags.astype(bool)
            ],
            dtype=np.uint64,
        ))
        self.expected_support_before = expected_support_before
        self.expected_support_after = expected_support_after
        self.zero_cardinality_object_ids = tuple(
            object_id
            for object_id, cardinality in zip(
                self.decode_object_ids, self.decode_cardinalities
            )
            if int(cardinality) == 0
        )
        self.cross_clause_shared_object_count = (
            len(self.segment_object_ordinals) - len(self.decode_object_ids)
        )
        for array in self.descriptor_arrays:
            array.setflags(write=False)

    @property
    def descriptor_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.decode_cardinalities,
            self.segment_object_ordinals,
            self.boundary_flags,
            self.lower_bounds,
            self.upper_bounds,
        )

    @property
    def descriptor_bytes(self) -> int:
        return int(sum(array.nbytes for array in self.descriptor_arrays))

    @property
    def object_count(self) -> int:
        return len(self.decode_object_ids)

    @property
    def segment_count(self) -> int:
        return len(self.segment_object_ordinals)


def compile_formula_route(
    row: Mapping[str, Any],
    fragment_cardinalities: Mapping[str, int],
    numeric: Mapping[str, np.ndarray],
    valid: Mapping[str, np.ndarray],
    *,
    universe: int = 1_000_000,
) -> CompiledFixedBlockRoute:
    """Compile range/DNF formula semantics without reading an outcome."""
    family = str(row.get("family"))
    if family not in {"range", "dnf2"}:
        raise ValueError("low-memory fixed-block route requires range/dnf2")
    if not 0 <= int(row["query_base_id"]) < universe:
        raise ValueError("query base outside fixed universe")

    clauses: list[tuple[int, Mapping[str, Any]]] = [
        (int(row["primary_token_id"]), row["interval_1"])
    ]
    if family == "dnf2":
        clauses.append((int(row["secondary_token_id"]), row["interval_2"]))

    object_ids: list[str] = []
    object_to_ordinal: dict[str, int] = {}
    cardinalities: list[int] = []
    segment_ordinals: list[int] = []
    boundaries: list[int] = []
    segment_attributes: list[str] = []
    lower: list[float] = []
    upper: list[float] = []

    for token, interval in clauses:
        if not isinstance(interval, Mapping):
            raise TypeError("clause interval is absent")
        attribute = str(interval["attribute"])
        full = tuple(int(value) for value in interval["full_coarse_blocks"])
        boundary = tuple(
            int(value) for value in interval["boundary_coarse_blocks"]
        )
        overlap = tuple(
            int(value) for value in interval["overlap_coarse_blocks"]
        )
        if (
            not overlap
            or len(overlap) != len(set(overlap))
            or len(full) != len(set(full))
            or len(boundary) != len(set(boundary))
            or set(full) & set(boundary)
            or set(full) | set(boundary) != set(overlap)
        ):
            raise ValueError("full/boundary/overlap partition is malformed")
        lo = _decode_endpoint(interval["lo"])
        hi = _decode_endpoint(interval["hi"])
        if not lo < hi:
            raise ValueError("interval is empty")
        if attribute not in numeric or attribute not in valid:
            raise KeyError(f"numeric descriptor missing: {attribute}")
        _checked_column(
            numeric[attribute], valid[attribute], universe, attribute
        )
        clause_object_ids = [
            f"X|{token}|{attribute}|{block}"
            for block in full + boundary
        ]
        if len(clause_object_ids) != len(set(clause_object_ids)):
            raise ValueError("same clause references one X object twice")

        for is_boundary, blocks in ((False, full), (True, boundary)):
            for block in blocks:
                object_id = f"X|{token}|{attribute}|{block}"
                if object_id not in fragment_cardinalities:
                    raise KeyError(f"required X object missing: {object_id}")
                cardinality = int(fragment_cardinalities[object_id])
                if cardinality < 0:
                    raise ValueError(f"negative fragment cardinality: {object_id}")
                ordinal = object_to_ordinal.get(object_id)
                if ordinal is None:
                    ordinal = len(object_ids)
                    object_to_ordinal[object_id] = ordinal
                    object_ids.append(object_id)
                    cardinalities.append(cardinality)
                elif cardinalities[ordinal] != cardinality:
                    raise ValueError(f"fragment cardinality drift: {object_id}")
                segment_ordinals.append(ordinal)
                boundaries.append(int(is_boundary))
                segment_attributes.append(attribute if is_boundary else "")
                lower.append(lo if is_boundary else 0.0)
                upper.append(hi if is_boundary else 0.0)

    if not (
        0 < len(object_ids) <= MAX_OBJECTS
        and 0 < len(segment_ordinals) <= MAX_SEGMENTS
    ):
        raise ValueError("object/segment capacity exceeded")
    decoded = sum(cardinalities)
    if decoded <= 0 or decoded > MAX_DECODED_ROWS:
        raise OverflowError(
            f"decoded route rows {decoded} exceed frozen cap {MAX_DECODED_ROWS}"
        )

    expected_before = row.get("support_before_any_query_removal")
    expected_after = row.get("support_after_new_leave_one_out")
    if expected_before is not None or expected_after is not None:
        if not (
            type(expected_before) is int
            and type(expected_after) is int
            and expected_before == expected_after + 1
            and expected_after >= TOP_K
        ):
            raise ValueError("declared before/after support counts are malformed")

    return CompiledFixedBlockRoute(
        family=family,
        request_id=int(row["request_id"]),
        predicate_qid=int(row["predicate_qid"]),
        query_base_id=int(row["query_base_id"]),
        predicate=str(row["predicate"]),
        decode_object_ids=object_ids,
        decode_cardinalities=np.asarray(cardinalities, dtype=np.uint32),
        segment_object_ordinals=np.asarray(
            segment_ordinals, dtype=np.uint8
        ),
        boundary_flags=np.asarray(boundaries, dtype=np.uint8),
        segment_attribute_names=segment_attributes,
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
        expected_support_before=(
            None if expected_before is None else int(expected_before)
        ),
        expected_support_after=(
            None if expected_after is None else int(expected_after)
        ),
    )


class CompactU24OwnerView:
    """Read-only checked view of a dynamic compact-U24 single owner."""

    HEADER = struct.Struct("<8s11I12s")
    MAGIC = b"BAJ7U24\0"
    VERSION = 1

    __slots__ = (
        "owner", "words_offset", "words", "prefix_offset", "prefixes",
        "offsets_offset", "offset_entries", "arena_offset",
        "payload_bytes", "selected_objects", "allocated_capacity_bytes",
        "_pin", "_base_address",
    )

    def __init__(self, owner: bytearray) -> None:
        if type(owner) is not bytearray or len(owner) < self.HEADER.size:
            raise TypeError("compact U24 owner must be a mutable bytearray")
        values = self.HEADER.unpack_from(owner, 0)
        (
            magic, version, allocated, words_offset, words,
            prefix_offset, prefixes, offsets_offset, offset_entries,
            arena_offset, payload_bytes, selected_objects, reserved,
        ) = values
        if not (
            magic == self.MAGIC
            and version == self.VERSION
            and allocated == len(owner)
            and words_offset == self.HEADER.size
            and words == logical.WORDS
            and prefix_offset == words_offset + 8 * words
            and prefixes == logical.PREFIXES
            and offsets_offset == prefix_offset + 2 * prefixes
            and offset_entries == selected_objects + 1
            and offsets_offset + 4 * offset_entries <= arena_offset
            and arena_offset % 64 == 0
            and arena_offset + payload_bytes <= len(owner)
            and reserved == b"\0" * 12
        ):
            raise ValueError("compact U24 owner header/section drift")
        offsets = np.frombuffer(
            owner, dtype="<u4", count=offset_entries, offset=offsets_offset
        )
        if not (
            int(offsets[0]) == 0
            and int(offsets[-1]) == payload_bytes
            and bool(np.all(offsets[1:] >= offsets[:-1]))
            and bool(np.all((offsets[1:] - offsets[:-1]) % 3 == 0))
        ):
            raise ValueError("compact U24 offset directory drift")
        running = 0
        for index in range(words):
            word = struct.unpack_from("<Q", owner, words_offset + 8 * index)[0]
            prefix = struct.unpack_from(
                "<H", owner, prefix_offset + 2 * index
            )[0]
            if prefix != running:
                raise ValueError("compact U24 prefix recurrence drift")
            running += bin(word).count("1")
        terminal_prefix = struct.unpack_from(
            "<H", owner, prefix_offset + 2 * words
        )[0]
        if running != selected_objects or terminal_prefix != running:
            raise ValueError("compact U24 selected-object popcount drift")
        self.owner = owner
        self.words_offset = words_offset
        self.words = words
        self.prefix_offset = prefix_offset
        self.prefixes = prefixes
        self.offsets_offset = offsets_offset
        self.offset_entries = offset_entries
        self.arena_offset = arena_offset
        self.payload_bytes = payload_bytes
        self.selected_objects = selected_objects
        self.allocated_capacity_bytes = len(owner)
        self._pin = memoryview(owner)
        self._base_address = ctypes.addressof(
            ctypes.c_uint8.from_buffer(owner)
        )

    def _ordinal(self, object_id: str) -> int:
        try:
            position = logical.key_to_position(object_id)
        except KeyError as error:
            raise KeyError(object_id) from error
        word_index, bit = divmod(position, 64)
        word = struct.unpack_from(
            "<Q", self.owner, self.words_offset + 8 * word_index
        )[0]
        if not word & (1 << bit):
            raise KeyError(object_id)
        prefix = struct.unpack_from(
            "<H", self.owner, self.prefix_offset + 2 * word_index
        )[0]
        return prefix + bin(word & ((1 << bit) - 1)).count("1")

    def payload_bounds(self, object_id: str) -> tuple[int, int]:
        ordinal = self._ordinal(object_id)
        return self.ordinal_payload_bounds(ordinal)

    def ordinal_payload_bounds(self, ordinal: int) -> tuple[int, int]:
        if not 0 <= int(ordinal) < self.selected_objects:
            raise IndexError("compact U24 ordinal outside selected inventory")
        start = struct.unpack_from(
            "<I", self.owner, self.offsets_offset + 4 * ordinal
        )[0]
        end = struct.unpack_from(
            "<I", self.owner, self.offsets_offset + 4 * (ordinal + 1)
        )[0]
        if not 0 <= start <= end <= self.payload_bytes or (end - start) % 3:
            raise ValueError(f"ordinal {ordinal}: payload bounds drift")
        return start, end

    def ordinal_payload_pointer_and_rows(self, ordinal: int) -> tuple[int, int]:
        start, end = self.ordinal_payload_bounds(ordinal)
        rows = (end - start) // 3
        pointer = (
            0 if rows == 0 else self._base_address + self.arena_offset + start
        )
        return pointer, rows

    def __contains__(self, object_id: object) -> bool:
        if not isinstance(object_id, str):
            return False
        try:
            self._ordinal(object_id)
        except (KeyError, ValueError):
            return False
        return True


def _align64(value: int) -> int:
    return (int(value) + 63) // 64 * 64


class FixedBlockDescriptorTable:
    """One aligned SoA owner for every route and global object descriptor."""

    MAGIC = b"L1MFBDS1"
    VERSION = 1
    HEADER = struct.Struct("<8s15I60s")  # exactly 128 bytes
    ROUTE_META_FIELDS = (
        "request_id", "predicate_qid", "query_base_id",
        "decoded_input_rows", "segment_input_rows", "boundary_checks",
        "support_before", "support_after",
    )

    __slots__ = (
        "owner", "routes", "segments", "global_objects", "attributes",
        "attribute_names", "route_offsets", "route_meta",
        "segment_global_object_ordinals", "segment_filter_codes",
        "lower_bounds", "upper_bounds", "global_object_payload_ptrs",
        "global_object_rows", "attribute_value_ptrs",
        "attribute_valid_ptrs", "attribute_value_kinds", "layout",
        "u24_owner_view", "_source_owners", "_native_pointers",
    )

    def __init__(
        self,
        routes: Sequence[CompiledFixedBlockRoute],
        u24_owner_view: CompactU24OwnerView,
        numeric: Mapping[str, np.ndarray],
        valid: Mapping[str, np.ndarray],
        *,
        universe: int = 1_000_000,
    ) -> None:
        if not routes:
            raise ValueError("descriptor table needs at least one route")
        route_count = len(routes)
        segment_count = sum(route.segment_count for route in routes)
        global_objects = int(u24_owner_view.selected_objects)
        attribute_names = tuple(sorted({
            attribute
            for route in routes
            for attribute in route.segment_attribute_names
            if attribute
        }))
        if not (
            route_count > 0
            and segment_count > 0
            and 0 < global_objects <= MAX_GLOBAL_OBJECTS
            and 0 < len(attribute_names) <= 255
        ):
            raise ValueError("descriptor table population exceeds ABI")

        cursor = self.HEADER.size
        sections: dict[str, tuple[int, int, int]] = {}

        def section(name: str, logical_bytes: int) -> int:
            nonlocal cursor
            offset = cursor
            capacity = _align64(logical_bytes)
            sections[name] = (offset, logical_bytes, capacity)
            cursor += capacity
            return offset

        route_offsets_offset = section(
            "route_offsets", 4 * (route_count + 1)
        )
        route_meta_offset = section(
            "route_meta", 4 * len(self.ROUTE_META_FIELDS) * route_count
        )
        segment_objects_offset = section(
            "segment_global_object_ordinals", 2 * segment_count
        )
        segment_filters_offset = section(
            "segment_filter_codes", segment_count
        )
        lower_offset = section("lower_bounds", 8 * segment_count)
        upper_offset = section("upper_bounds", 8 * segment_count)
        object_ptrs_offset = section(
            "global_object_payload_ptrs", 8 * global_objects
        )
        object_rows_offset = section(
            "global_object_rows", 4 * global_objects
        )
        attribute_desc_offset = section(
            "attribute_descriptors", 17 * len(attribute_names)
        )
        total_capacity = cursor
        owner = bytearray(total_capacity)
        self.HEADER.pack_into(
            owner, 0, self.MAGIC, self.VERSION, total_capacity,
            route_count, segment_count, global_objects,
            len(attribute_names), route_offsets_offset, route_meta_offset,
            segment_objects_offset, segment_filters_offset, lower_offset,
            upper_offset, object_ptrs_offset, object_rows_offset,
            attribute_desc_offset, b"\0" * 60,
        )

        self.owner = owner
        self.routes = route_count
        self.segments = segment_count
        self.global_objects = global_objects
        self.attributes = len(attribute_names)
        self.attribute_names = attribute_names
        self.route_offsets = np.frombuffer(
            owner, dtype=np.uint32, count=route_count + 1,
            offset=route_offsets_offset,
        )
        self.route_meta = np.frombuffer(
            owner, dtype=np.uint32,
            count=len(self.ROUTE_META_FIELDS) * route_count,
            offset=route_meta_offset,
        ).reshape(len(self.ROUTE_META_FIELDS), route_count)
        self.segment_global_object_ordinals = np.frombuffer(
            owner, dtype=np.uint16, count=segment_count,
            offset=segment_objects_offset,
        )
        self.segment_filter_codes = np.frombuffer(
            owner, dtype=np.uint8, count=segment_count,
            offset=segment_filters_offset,
        )
        self.lower_bounds = np.frombuffer(
            owner, dtype=np.float64, count=segment_count,
            offset=lower_offset,
        )
        self.upper_bounds = np.frombuffer(
            owner, dtype=np.float64, count=segment_count,
            offset=upper_offset,
        )
        self.global_object_payload_ptrs = np.frombuffer(
            owner, dtype=np.uintp, count=global_objects,
            offset=object_ptrs_offset,
        )
        self.global_object_rows = np.frombuffer(
            owner, dtype=np.uint32, count=global_objects,
            offset=object_rows_offset,
        )
        self.attribute_value_ptrs = np.frombuffer(
            owner, dtype=np.uintp, count=len(attribute_names),
            offset=attribute_desc_offset,
        )
        valid_offset = attribute_desc_offset + 8 * len(attribute_names)
        self.attribute_valid_ptrs = np.frombuffer(
            owner, dtype=np.uintp, count=len(attribute_names),
            offset=valid_offset,
        )
        kind_offset = valid_offset + 8 * len(attribute_names)
        self.attribute_value_kinds = np.frombuffer(
            owner, dtype=np.uint8, count=len(attribute_names),
            offset=kind_offset,
        )

        for ordinal in range(global_objects):
            pointer, rows = (
                u24_owner_view.ordinal_payload_pointer_and_rows(ordinal)
            )
            self.global_object_payload_ptrs[ordinal] = pointer
            self.global_object_rows[ordinal] = rows

        source_owners: list[np.ndarray] = []
        attribute_codes = {
            attribute: index + 1
            for index, attribute in enumerate(attribute_names)
        }
        for index, attribute in enumerate(attribute_names):
            if attribute not in numeric or attribute not in valid:
                raise KeyError(f"numeric descriptor missing: {attribute}")
            value_ptr, valid_ptr, value_kind = _checked_column(
                numeric[attribute], valid[attribute], universe, attribute
            )
            self.attribute_value_ptrs[index] = value_ptr
            self.attribute_valid_ptrs[index] = valid_ptr
            self.attribute_value_kinds[index] = value_kind
            source_owners.extend((numeric[attribute], valid[attribute]))

        segment_cursor = 0
        self.route_offsets[0] = 0
        for route_index, route in enumerate(routes):
            begin = segment_cursor
            for local_segment in range(route.segment_count):
                local_object = int(route.segment_object_ordinals[local_segment])
                object_id = route.decode_object_ids[local_object]
                if object_id not in u24_owner_view:
                    raise KeyError(
                        f"required compact U24 object missing: {object_id}"
                    )
                global_ordinal = u24_owner_view._ordinal(object_id)
                _start, _end = u24_owner_view.ordinal_payload_bounds(
                    global_ordinal
                )
                if (_end - _start) // 3 != int(
                    route.decode_cardinalities[local_object]
                ):
                    raise ValueError(
                        f"{object_id}: payload/cardinality mismatch"
                    )
                attribute = route.segment_attribute_names[local_segment]
                self.segment_global_object_ordinals[segment_cursor] = (
                    global_ordinal
                )
                self.segment_filter_codes[segment_cursor] = (
                    0 if not attribute else attribute_codes[attribute]
                )
                self.lower_bounds[segment_cursor] = route.lower_bounds[
                    local_segment
                ]
                self.upper_bounds[segment_cursor] = route.upper_bounds[
                    local_segment
                ]
                segment_cursor += 1
            self.route_offsets[route_index + 1] = segment_cursor
            meta_values = (
                route.request_id,
                route.predicate_qid,
                route.query_base_id,
                route.expected_decoded_input_rows,
                route.expected_segment_input_rows,
                route.expected_boundary_checks,
                route.expected_support_before,
                route.expected_support_after,
            )
            if any(value is None or not 0 <= int(value) <= 0xFFFFFFFF
                   for value in meta_values):
                raise ValueError("route metadata absent/outside uint32")
            self.route_meta[:, route_index] = np.asarray(
                meta_values, dtype=np.uint32
            )
            if segment_cursor - begin != route.segment_count:
                raise AssertionError("route descriptor terminal drift")
        if segment_cursor != segment_count:
            raise AssertionError("descriptor segment count drift")

        self.layout = {
            "schema": "laion1m-lowmem-fixedblock-descriptor-layout/v1",
            "alignment_bytes": 64,
            "header_bytes": self.HEADER.size,
            "route_count": route_count,
            "segment_count": segment_count,
            "global_object_count": global_objects,
            "attribute_count": len(attribute_names),
            "attribute_names": list(attribute_names),
            "sections": {
                name: {
                    "offset": values[0],
                    "logical_bytes": values[1],
                    "allocated_capacity_bytes": values[2],
                }
                for name, values in sections.items()
            },
            "allocated_capacity_bytes": total_capacity,
            "single_preallocated_owner": True,
            "per_route_numpy_owners": 0,
            "global_payload_table_stored_once": True,
        }
        if (
            total_capacity != self.HEADER.size
            + sum(value[2] for value in sections.values())
            or any(value[0] % 64 for value in sections.values())
        ):
            raise AssertionError("descriptor capacity/alignment drift")
        self.u24_owner_view = u24_owner_view
        self._source_owners = tuple(source_owners)
        self._native_pointers = (
            self.global_object_payload_ptrs.ctypes.data_as(
                U8_POINTER_POINTER
            ),
            self.global_object_rows.ctypes.data_as(U32_POINTER),
            self.segment_global_object_ordinals.ctypes.data_as(U16_POINTER),
            self.segment_filter_codes.ctypes.data_as(U8_ARRAY_POINTER),
            self.lower_bounds.ctypes.data_as(F64_POINTER),
            self.upper_bounds.ctypes.data_as(F64_POINTER),
            self.attribute_value_ptrs.ctypes.data_as(UPTR_POINTER),
            self.attribute_valid_ptrs.ctypes.data_as(UPTR_POINTER),
            self.attribute_value_kinds.ctypes.data_as(U8_ARRAY_POINTER),
        )

    @property
    def allocated_capacity_bytes(self) -> int:
        return len(self.owner)

    def route_bounds(self, route_index: int) -> tuple[int, int]:
        if not 0 <= int(route_index) < self.routes:
            raise IndexError("descriptor route index outside table")
        return (
            int(self.route_offsets[route_index]),
            int(self.route_offsets[route_index + 1]),
        )

    def route_metadata(self, route_index: int) -> dict[str, int]:
        self.route_bounds(route_index)
        return {
            name: int(self.route_meta[index, route_index])
            for index, name in enumerate(self.ROUTE_META_FIELDS)
        }


class LowMemoryFixedBlockNative:
    def __init__(self, library: Path = DEFAULT_LIBRARY) -> None:
        self.path = library.resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        for suffix in (
            "abi_version", "max_objects", "max_segments",
            "max_global_objects",
            "max_decoded_rows", "batch_ids", "lane_decode_scratch_bytes",
            "lane_output_workspace_bytes", "explicit_local_array_bytes",
            "audit_struct_bytes",
        ):
            function = getattr(
                self.library, f"laion1m_lowmem_fixedblock_v1_{suffix}"
            )
            function.argtypes = ()
            function.restype = ctypes.c_uint32
        observed = (
            int(self.library.laion1m_lowmem_fixedblock_v1_abi_version()),
            int(self.library.laion1m_lowmem_fixedblock_v1_max_objects()),
            int(self.library.laion1m_lowmem_fixedblock_v1_max_segments()),
            int(
                self.library
                .laion1m_lowmem_fixedblock_v1_max_global_objects()
            ),
            int(self.library.laion1m_lowmem_fixedblock_v1_max_decoded_rows()),
            int(self.library.laion1m_lowmem_fixedblock_v1_batch_ids()),
            int(
                self.library
                .laion1m_lowmem_fixedblock_v1_lane_decode_scratch_bytes()
            ),
            int(
                self.library
                .laion1m_lowmem_fixedblock_v1_lane_output_workspace_bytes()
            ),
            int(self.library.laion1m_lowmem_fixedblock_v1_audit_struct_bytes()),
        )
        expected = (
            ABI_VERSION, MAX_OBJECTS, MAX_SEGMENTS, MAX_GLOBAL_OBJECTS,
            MAX_DECODED_ROWS,
            BATCH_IDS, LANE_DECODE_SCRATCH_BYTES,
            LANE_OUTPUT_WORKSPACE_BYTES, AUDIT_STRUCT_BYTES,
        )
        if observed != expected:
            raise RuntimeError(f"low-memory fixed-block ABI drift: {observed}")
        retained_policy = self.library.laion1m_exact_topk_avx2_v1_retained_prefetch_policy
        retained_policy.argtypes = ()
        retained_policy.restype = ctypes.c_uint32
        if int(retained_policy()) != RETAINED_PREFETCH_POLICY_NTA:
            raise RuntimeError("fixed-block compositor is not retained NTA ABI-v7")

        self.function = self.library.laion1m_lowmem_fixedblock_exact_top10_v1
        self.function.argtypes = (
            F32_POINTER, ctypes.c_uint64, F32_POINTER,
            U8_POINTER_POINTER, U32_POINTER, ctypes.c_uint64,
            U16_POINTER, U8_ARRAY_POINTER, F64_POINTER, F64_POINTER,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            UPTR_POINTER, UPTR_POINTER, U8_ARRAY_POINTER, ctypes.c_uint64,
            ctypes.c_int32,
            I32_POINTER, ctypes.c_uint64, I32_POINTER, F32_POINTER,
            ctypes.POINTER(FixedBlockAuditV1), I32_POINTER,
            ctypes.c_uint64,
        )
        self.function.restype = ctypes.c_int

    @property
    def explicit_local_array_bytes(self) -> int:
        return int(
            self.library
            .laion1m_lowmem_fixedblock_v1_explicit_local_array_bytes()
        )

    def workspace(self, *, diagnostic_support: bool = False) -> "FixedBlockWorkspace":
        return FixedBlockWorkspace(self, diagnostic_support=diagnostic_support)

    def implementation_audit(self) -> dict[str, Any]:
        return {
            "schema": "laion1m-lowmem-fixedblock-native-audit/v1",
            "status": "CANDIDATE_QUERY_FREE_ONLY",
            "library": str(self.path),
            "retained_u24_decoder": str(RETAINED_U24_LIBRARY),
            "retained_scorer_abi": 7,
            "retained_prefetch_policy": "NTA_BLOCKED_SAME_OFFSET",
            "maximum_objects": MAX_OBJECTS,
            "maximum_segments": MAX_SEGMENTS,
            "maximum_global_objects": MAX_GLOBAL_OBJECTS,
            "maximum_decoded_rows": MAX_DECODED_ROWS,
            "lane_decode_scratch_bytes": LANE_DECODE_SCRATCH_BYTES,
            "lane_output_workspace_bytes": LANE_OUTPUT_WORKSPACE_BYTES,
            "lane_workspace_bytes": LANE_WORKSPACE_BYTES,
            "eight_lane_workspace_bytes": EIGHT_LANE_WORKSPACE_BYTES,
            "source_level_explicit_local_array_bytes": (
                self.explicit_local_array_bytes
            ),
            "hot_native_heap_allocations": 0,
            "hot_native_global_mutations": 0,
            "decode_inside_native_call": True,
            "boundary_filter_inside_native_call": True,
            "global_union_dedup_inside_native_call": True,
            "leave_one_out_inside_native_call": True,
            "exact_top10_inside_native_call": True,
            "production_materializes_complete_support": False,
            "query_free_diagnostic_support_optional": True,
            "descriptor_layout": (
                "single aligned SoA owner; one global payload table; compact "
                "u16 object ordinal + u8 filter code + f64 lo/hi per segment"
            ),
            "performance_claim_authorized": False,
            "implementation_note": str(IMPLEMENTATION_NOTE),
        }


class FixedBlockWorkspace:
    """One lane's fully preallocated request scratch and outputs."""

    __slots__ = (
        "native", "decode_scratch", "top_ids", "top_squared_l2", "audit",
        "diagnostic_support",
    )

    def __init__(
        self, native: LowMemoryFixedBlockNative, *, diagnostic_support: bool
    ) -> None:
        self.native = native
        self.decode_scratch = np.empty(MAX_DECODED_ROWS, dtype=np.int32)
        self.top_ids = np.empty(TOP_K, dtype=np.int32)
        self.top_squared_l2 = np.empty(TOP_K, dtype=np.float32)
        self.audit = FixedBlockAuditV1()
        self.diagnostic_support = (
            np.empty(MAX_DECODED_ROWS, dtype=np.int32)
            if diagnostic_support else None
        )

    @property
    def allocated_bytes(self) -> int:
        return int(
            self.decode_scratch.nbytes
            + self.top_ids.nbytes
            + self.top_squared_l2.nbytes
            + ctypes.sizeof(self.audit)
            + (
                0
                if self.diagnostic_support is None
                else self.diagnostic_support.nbytes
            )
        )

    @property
    def production_allocated_bytes(self) -> int:
        return int(
            self.decode_scratch.nbytes
            + self.top_ids.nbytes
            + self.top_squared_l2.nbytes
            + ctypes.sizeof(self.audit)
        )

    def run_table_into(
        self,
        base: np.ndarray,
        query: np.ndarray,
        table: FixedBlockDescriptorTable,
        route_index: int,
    ) -> FixedBlockAuditV1:
        if not (
            isinstance(base, np.ndarray)
            and base.dtype == np.float32
            and base.ndim == 2
            and base.shape[1] == DIMENSION
            and base.flags.c_contiguous
            and isinstance(query, np.ndarray)
            and query.dtype == np.float32
            and query.shape == (DIMENSION,)
            and query.flags.c_contiguous
            and isinstance(table, FixedBlockDescriptorTable)
        ):
            raise TypeError("fixed-block base/query/descriptor contract failed")
        begin, end = table.route_bounds(route_index)
        metadata = table.route_metadata(route_index)
        diagnostic = self.diagnostic_support
        diagnostic_pointer = (
            ctypes.cast(0, I32_POINTER)
            if diagnostic is None
            else diagnostic.ctypes.data_as(I32_POINTER)
        )
        diagnostic_capacity = 0 if diagnostic is None else len(diagnostic)
        status = int(self.native.function(
            base.ctypes.data_as(F32_POINTER),
            len(base),
            query.ctypes.data_as(F32_POINTER),
            table._native_pointers[0],
            table._native_pointers[1],
            table.global_objects,
            table._native_pointers[2],
            table._native_pointers[3],
            table._native_pointers[4],
            table._native_pointers[5],
            table.segments,
            begin,
            end - begin,
            table._native_pointers[6],
            table._native_pointers[7],
            table._native_pointers[8],
            table.attributes,
            metadata["query_base_id"],
            self.decode_scratch.ctypes.data_as(I32_POINTER),
            len(self.decode_scratch),
            self.top_ids.ctypes.data_as(I32_POINTER),
            self.top_squared_l2.ctypes.data_as(F32_POINTER),
            ctypes.byref(self.audit),
            diagnostic_pointer,
            diagnostic_capacity,
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
                8: "malformed U24 payload/object ordinal",
                9: "malformed boundary descriptor",
                10: "query base absent before leave-one-out",
                11: "diagnostic support buffer contract",
                12: "retained U24 decoder failure",
            }
            raise ValueError(
                "low-memory fixed-block compositor failed closed "
                f"({status}: {meanings.get(status, 'unknown')})"
            )
        if not (
            int(self.audit.decoded_input_rows)
            == metadata["decoded_input_rows"]
            and int(self.audit.segment_input_rows)
            == metadata["segment_input_rows"]
            and int(self.audit.boundary_predicate_checks)
            == metadata["boundary_checks"]
            and int(self.audit.support_before_leave_one_out)
            == int(self.audit.support_after_leave_one_out) + 1
            and int(self.audit.support_after_leave_one_out) >= TOP_K
        ):
            raise AssertionError("fixed-block native audit/count drift")
        if (
            int(self.audit.support_before_leave_one_out)
            != metadata["support_before"]
            or int(self.audit.support_after_leave_one_out)
            != metadata["support_after"]
        ):
            raise AssertionError("fixed-block declared support count mismatch")
        return self.audit

    def diagnostic_support_view(self) -> np.ndarray:
        if self.diagnostic_support is None:
            raise RuntimeError("workspace was not provisioned for diagnostics")
        return self.diagnostic_support[
            : int(self.audit.support_after_leave_one_out)
        ]


def integration_interface() -> dict[str, Any]:
    """Machine-readable handoff; no build or execution is performed."""
    return {
        "schema": "laion1m-lowmem-fixedblock-integration-interface/v1",
        "status": "CREATE_ONLY_NOT_BUILT",
        "compile_formula_route": (
            "compile_formula_route(row, fragment_cardinalities, numeric, valid)"
        ),
        "owner_view": "CompactU24OwnerView(owner_bytearray)",
        "descriptor_table": (
            "FixedBlockDescriptorTable(compiled_routes, owner_view, "
            "numeric, valid)"
        ),
        "lane_workspace": (
            "LowMemoryFixedBlockNative(library).workspace()"
        ),
        "timed_call": (
            "workspace.run_table_into(base, query, descriptor_table, "
            "route_index)"
        ),
        "top_ids": "workspace.top_ids",
        "top_squared_l2": "workspace.top_squared_l2",
        "audit": "workspace.audit",
        "lane_workspace_bytes": LANE_WORKSPACE_BYTES,
        "eight_lane_workspace_bytes": EIGHT_LANE_WORKSPACE_BYTES,
        "production_materializes_complete_support": False,
        "diagnostic_support_is_query_free_only": True,
        "performance_started": False,
        "performance_authorized": False,
    }
