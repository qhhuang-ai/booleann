#!/usr/bin/env python3
"""Single-owner compact raw-int32 support arena for LAION1M J4/J5.

The resident representation is one charged bytearray.  Directory values are
read with fixed-offset ``struct`` operations, and support lookups create only
ephemeral NumPy views into the same owner.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import struct
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


TOKENS = 200
ATTRIBUTES = ("similarity", "original_width", "original_height")
ATTRIBUTE_INDEX = {name: index for index, name in enumerate(ATTRIBUTES)}
BLOCKS = 32
PAIR_POSITIONS = TOKENS * (TOKENS - 1) // 2
LOGICAL_POSITIONS = PAIR_POSITIONS + TOKENS * len(ATTRIBUTES) * BLOCKS
WORDS = (LOGICAL_POSITIONS + 63) // 64
PREFIXES = WORDS + 1
SELECTED_OBJECTS = 8_136
ARENA_IDS = 1_600_810

HEADER_OFFSET = 0
HEADER_BYTES = 64
WORDS_OFFSET = 64
PREFIX_OFFSET = WORDS_OFFSET + WORDS * 8
OFFSETS_OFFSET = PREFIX_OFFSET + PREFIXES * 2
ARENA_OFFSET = 38_784
ARENA_BYTES = ARENA_IDS * 4
TOTAL_BYTES = ARENA_OFFSET + ARENA_BYTES
ALIGNMENT_PADDING_BYTES = ARENA_OFFSET - (
    OFFSETS_OFFSET + (SELECTED_OBJECTS + 1) * 4
)

MAGIC = b"BAJ4I32\0"
VERSION = 1
HEADER = struct.Struct("<8s11I12s")
PAIR_STARTS = tuple(
    a * (2 * TOKENS - a - 1) // 2 for a in range(TOKENS)
)


def key_to_position(object_id: str) -> int:
    fields = object_id.split("|")
    try:
        if len(fields) == 3 and fields[0] == "C":
            a, b = int(fields[1]), int(fields[2])
            if not (0 <= a < b < TOKENS):
                raise ValueError
            return a * (2 * TOKENS - a - 1) // 2 + b - a - 1
        if len(fields) == 4 and fields[0] == "X":
            token, attribute, block = (
                int(fields[1]), fields[2], int(fields[3])
            )
            if not (
                0 <= token < TOKENS
                and attribute in ATTRIBUTE_INDEX
                and 0 <= block < BLOCKS
            ):
                raise ValueError
            return (
                PAIR_POSITIONS
                + token * len(ATTRIBUTES) * BLOCKS
                + ATTRIBUTE_INDEX[attribute] * BLOCKS
                + block
            )
    except ValueError as error:
        raise KeyError(f"noncanonical compact key: {object_id}") from error
    raise KeyError(f"noncanonical compact key: {object_id}")


def position_to_key(position: int) -> str:
    if not 0 <= position < LOGICAL_POSITIONS:
        raise KeyError(f"logical position out of range: {position}")
    if position < PAIR_POSITIONS:
        a = bisect.bisect_right(PAIR_STARTS, position) - 1
        if not 0 <= a < TOKENS - 1:
            raise AssertionError("pair inverse overflow")
        b = a + 1 + position - PAIR_STARTS[a]
        return f"C|{a}|{b}"
    value = position - PAIR_POSITIONS
    token, remainder = divmod(value, len(ATTRIBUTES) * BLOCKS)
    attribute_index, block = divmod(remainder, BLOCKS)
    return f"X|{token}|{ATTRIBUTES[attribute_index]}|{block}"


def _active(selectable: bool, authorized: bool, complete: bool) -> bool:
    return bool(selectable and authorized and complete)


class CompactI32Arena(Mapping[str, np.ndarray]):
    """Read-only mapping backed by exactly one persistent allocation."""

    __slots__ = ("_owner", "_released")

    def __init__(self, owner: bytearray):
        if type(owner) is not bytearray or len(owner) != TOTAL_BYTES:
            raise AssertionError("compact arena owner size/type drift")
        self._owner: bytearray | None = owner
        self._released = False
        self._validate_header()

    @classmethod
    def build(
        cls,
        manifest: Mapping[str, Any],
        complete_selected: dict[str, np.ndarray],
    ) -> tuple["CompactI32Arena", dict[str, Any]]:
        schema = str(manifest.get("schema"))
        arm = str(manifest.get("arm"))
        if not (
            schema
            in {
                "laion1m-tight-budget-selector-manifest-j4-compact-i32/v1",
                "laion1m-tight-budget-selector-manifest-j5-family-gated-compact-i32/v1",
            }
            and arm in {"J4", "J5"}
            and manifest.get("layout")
            == "compact_raw_little_endian_int32_arena"
            and int(manifest["selected_objects"]) == SELECTED_OBJECTS
            and 0 < int(manifest["selected_cardinalities"]) <= ARENA_IDS
        ):
            raise AssertionError("compact manifest identity drift")
        rows = sorted(
            manifest["objects"], key=lambda row: int(row["logical_position"])
        )
        if len(rows) != SELECTED_OBJECTS:
            raise AssertionError("compact selected object count drift")
        if sum(int(row["cardinality"]) for row in rows) != int(
            manifest["selected_cardinalities"]
        ):
            raise AssertionError("compact manifest cardinality sum drift")
        owner = cls._build_owner(rows, complete_selected, production=True)
        source_objects = len(complete_selected)
        complete_selected.clear()
        del complete_selected
        gc.collect()
        arena = cls(owner)
        audit = arena.audit(full_payload=True)
        if audit["selected_objects"] != SELECTED_OBJECTS:
            raise AssertionError("J4 compact setup audit drift")
        return arena, {
            "arm": arm,
            "layout": "single_owner_raw_little_endian_int32_arena",
            "selected_objects": SELECTED_OBJECTS,
            "selected_cardinalities": int(manifest["selected_cardinalities"]),
            "selected_cx_accounted_bytes": TOTAL_BYTES,
            "arena_capacity_cardinalities": ARENA_IDS,
            "arena_unused_cardinalities": (
                ARENA_IDS - int(manifest["selected_cardinalities"])
            ),
            "persistent_backing_allocations": 1,
            "persistent_numpy_views": 0,
            "complete_source_objects_released": source_objects,
            "complete_source_int32_released_before_warmup": True,
            "setup_full_payload_audit": audit,
        }

    @staticmethod
    def _build_owner(
        rows: list[Mapping[str, Any]],
        selected: Mapping[str, np.ndarray],
        *,
        production: bool,
    ) -> bytearray:
        selected_count = len(rows)
        arena_ids = sum(int(row["cardinality"]) for row in rows)
        if production and (
            selected_count != SELECTED_OBJECTS or arena_ids > ARENA_IDS
        ):
            raise AssertionError("production compact inventory drift")
        if selected_count > SELECTED_OBJECTS or arena_ids > ARENA_IDS:
            raise AssertionError("compact synthetic inventory overflow")
        positions = [int(row["logical_position"]) for row in rows]
        if positions != sorted(set(positions)):
            raise AssertionError("arena rows must be unique logical-position order")
        owner = bytearray(TOTAL_BYTES)
        HEADER.pack_into(
            owner,
            HEADER_OFFSET,
            MAGIC,
            VERSION,
            TOTAL_BYTES,
            WORDS_OFFSET,
            WORDS,
            PREFIX_OFFSET,
            PREFIXES,
            OFFSETS_OFFSET,
            selected_count + 1,
            ARENA_OFFSET,
            arena_ids,
            selected_count,
            b"\0" * 12,
        )
        words = [0] * WORDS
        for position in positions:
            if not 0 <= position < LOGICAL_POSITIONS:
                raise AssertionError("selected logical position out of range")
            words[position // 64] |= 1 << (position % 64)
        running = 0
        for index, word in enumerate(words):
            struct.pack_into("<Q", owner, WORDS_OFFSET + 8 * index, word)
            struct.pack_into("<H", owner, PREFIX_OFFSET + 2 * index, running)
            running += bin(word).count("1")
        struct.pack_into("<H", owner, PREFIX_OFFSET + 2 * WORDS, running)
        if running != selected_count:
            raise AssertionError("selection rank count drift")

        cursor = 0
        struct.pack_into("<I", owner, OFFSETS_OFFSET, 0)
        for ordinal, row in enumerate(rows):
            object_id = str(row["object_id"])
            position = key_to_position(object_id)
            cardinality = int(row["cardinality"])
            if position != int(row["logical_position"]):
                raise AssertionError("manifest key/logical position drift")
            source = selected.get(object_id)
            if source is None:
                raise AssertionError(f"complete source missing {object_id}")
            array = np.asarray(source)
            if not (
                array.dtype == np.int32
                and array.ndim == 1
                and array.flags.c_contiguous
                and len(array) == cardinality
                and (
                    len(array) < 2
                    or bool(np.all(array[1:] > array[:-1]))
                )
                and (not len(array) or int(array[0]) >= 0)
                and (not len(array) or int(array[-1]) < 1_000_000)
            ):
                raise AssertionError(f"source support contract drift: {object_id}")
            destination = np.frombuffer(
                owner,
                dtype="<i4",
                count=cardinality,
                offset=ARENA_OFFSET + 4 * cursor,
            )
            np.copyto(destination, array, casting="no")
            del destination, array, source
            cursor += cardinality
            struct.pack_into(
                "<I", owner, OFFSETS_OFFSET + 4 * (ordinal + 1), cursor
            )
        if cursor != arena_ids:
            raise AssertionError("arena terminal offset drift")
        return owner

    def _require_owner(self) -> bytearray:
        if self._released or self._owner is None:
            raise AssertionError("compact arena used after release")
        return self._owner

    def _header(self) -> tuple[Any, ...]:
        return HEADER.unpack_from(self._require_owner(), HEADER_OFFSET)

    def _validate_header(self) -> None:
        values = self._header()
        if not (
            values[0] == MAGIC
            and values[1] == VERSION
            and values[2] == TOTAL_BYTES
            and values[3] == WORDS_OFFSET
            and values[4] == WORDS
            and values[5] == PREFIX_OFFSET
            and values[6] == PREFIXES
            and values[7] == OFFSETS_OFFSET
            and 1 <= values[8] <= SELECTED_OBJECTS + 1
            and values[9] == ARENA_OFFSET
            and 0 <= values[10] <= ARENA_IDS
            and values[11] + 1 == values[8]
            and values[12] == b"\0" * 12
            and WORDS_OFFSET + WORDS * 8 == PREFIX_OFFSET
            and PREFIX_OFFSET + PREFIXES * 2 == OFFSETS_OFFSET
            and OFFSETS_OFFSET + (SELECTED_OBJECTS + 1) * 4
            + ALIGNMENT_PADDING_BYTES == ARENA_OFFSET
            and ARENA_OFFSET % 64 == 0
            and ARENA_OFFSET + ARENA_BYTES == TOTAL_BYTES
        ):
            raise AssertionError("compact header/section contract drift")

    def _word(self, word_index: int) -> int:
        return struct.unpack_from(
            "<Q", self._require_owner(), WORDS_OFFSET + 8 * word_index
        )[0]

    def _ordinal(self, position: int) -> int:
        word_index, bit = divmod(position, 64)
        word = self._word(word_index)
        if not word & (1 << bit):
            raise KeyError(position_to_key(position))
        rank_before_word = struct.unpack_from(
            "<H", self._require_owner(), PREFIX_OFFSET + 2 * word_index
        )[0]
        return rank_before_word + bin(word & ((1 << bit) - 1)).count("1")

    def resolve(self, object_id: str) -> np.ndarray:
        position = key_to_position(object_id)
        ordinal = self._ordinal(position)
        owner = self._require_owner()
        start = struct.unpack_from(
            "<I", owner, OFFSETS_OFFSET + 4 * ordinal
        )[0]
        end = struct.unpack_from(
            "<I", owner, OFFSETS_OFFSET + 4 * (ordinal + 1)
        )[0]
        arena_ids = self._header()[10]
        if not 0 <= start < end <= arena_ids:
            raise AssertionError("compact offset monotonicity drift")
        view = np.frombuffer(
            owner, dtype="<i4", count=end - start,
            offset=ARENA_OFFSET + 4 * start,
        )
        view.flags.writeable = False
        return view

    def __getitem__(self, object_id: str) -> np.ndarray:
        return self.resolve(object_id)

    def __contains__(self, object_id: object) -> bool:
        if not isinstance(object_id, str):
            return False
        try:
            position = key_to_position(object_id)
        except KeyError:
            return False
        word_index, bit = divmod(position, 64)
        return bool(self._word(word_index) & (1 << bit))

    def __len__(self) -> int:
        return int(self._header()[11])

    def __iter__(self) -> Iterator[str]:
        for position in range(LOGICAL_POSITIONS):
            if self._word(position // 64) & (1 << (position % 64)):
                yield position_to_key(position)

    def audit(self, *, full_payload: bool) -> dict[str, Any]:
        self._validate_header()
        owner = self._require_owner()
        selected_count = len(self)
        arena_ids = int(self._header()[10])
        terminal = struct.unpack_from(
            "<I", owner, OFFSETS_OFFSET + 4 * selected_count
        )[0]
        if terminal != arena_ids:
            raise AssertionError("compact terminal offset/header drift")
        tail_bits = self._word(WORDS - 1) >> (LOGICAL_POSITIONS % 64)
        if tail_bits:
            raise AssertionError("compact final-word high-tail bits set")
        running_popcount = 0
        for word_index in range(WORDS):
            prefix = struct.unpack_from(
                "<H", owner, PREFIX_OFFSET + 2 * word_index
            )[0]
            if prefix != running_popcount:
                raise AssertionError("compact rank-prefix recurrence drift")
            running_popcount += bin(self._word(word_index)).count("1")
        terminal_prefix = struct.unpack_from(
            "<H", owner, PREFIX_OFFSET + 2 * WORDS
        )[0]
        if (
            terminal_prefix != running_popcount
            or running_popcount != selected_count
        ):
            raise AssertionError("compact selection popcount drift")
        previous = 0
        for ordinal in range(selected_count + 1):
            current = struct.unpack_from(
                "<I", owner, OFFSETS_OFFSET + 4 * ordinal
            )[0]
            if current < previous or current > arena_ids:
                raise AssertionError("compact offsets not monotone")
            previous = current
        payload_valid = None
        if full_payload:
            payload_valid = True
            for object_id in self:
                view = self.resolve(object_id)
                if not (
                    view.dtype == np.dtype("<i4")
                    and not view.flags.writeable
                    and (
                        len(view) < 2
                        or bool(np.all(view[1:] > view[:-1]))
                    )
                ):
                    raise AssertionError(
                        f"compact payload invalid: {object_id}"
                    )
                del view
        return {
            "status": "PASS",
            "selected_objects": selected_count,
            "selected_cardinalities": arena_ids,
            "cx_accounted_bytes": TOTAL_BYTES,
            "persistent_backing_allocations": 1,
            "persistent_numpy_views": 0,
            "logical_position_order": True,
            "terminal_offset_valid": True,
            "tail_bits_zero": True,
            "full_payload_valid": payload_valid,
        }

    def inventory(self) -> dict[str, Any]:
        return self.audit(full_payload=False)

    def release(self) -> dict[str, Any]:
        if self._released:
            raise AssertionError("compact arena released twice")
        count = len(self)
        cardinalities = int(self._header()[10])
        self._owner = None
        self._released = True
        gc.collect()
        return {
            "arm": "J4",
            "released_objects": count,
            "released_cardinalities": cardinalities,
            "released_backing_bytes": TOTAL_BYTES,
        }


def semantic_self_test() -> dict[str, Any]:
    keys = [
        *(f"C|{a}|{b}" for a in range(TOKENS) for b in range(a + 1, TOKENS)),
        *(
            f"X|{token}|{attribute}|{block}"
            for token in range(TOKENS)
            for attribute in ATTRIBUTES
            for block in range(BLOCKS)
        ),
    ]
    positions = [key_to_position(key) for key in keys]
    if (
        len(keys) != LOGICAL_POSITIONS
        or set(positions) != set(range(LOGICAL_POSITIONS))
        or any(position_to_key(position) != key for key, position in zip(keys, positions))
    ):
        raise AssertionError("full key/position bijection failed")
    if not (
        "C|0|116" < "C|0|12"
        and key_to_position("C|0|116") > key_to_position("C|0|12")
    ):
        raise AssertionError("lexicographic-order counterexample failed")
    rows = [
        {"object_id": position_to_key(position), "logical_position": position,
         "cardinality": 1}
        for position in (63, 64, LOGICAL_POSITIONS - 1)
    ]
    selected = {
        rows[0]["object_id"]: np.array([3], dtype=np.int32),
        rows[1]["object_id"]: np.array([5], dtype=np.int32),
        rows[2]["object_id"]: np.array([7], dtype=np.int32),
    }
    owner = CompactI32Arena._build_owner(rows, selected, production=False)
    arena = CompactI32Arena(owner)
    if not (
        int(arena.resolve(rows[0]["object_id"])[0]) == 3
        and int(arena.resolve(rows[1]["object_id"])[0]) == 5
        and int(arena.resolve(rows[2]["object_id"])[0]) == 7
        and position_to_key(62) not in arena
    ):
        raise AssertionError("rank-before-bit boundary/absence failed")
    try:
        arena.resolve(position_to_key(62))
    except KeyError:
        pass
    else:
        raise AssertionError("absent bit resolved")
    arena.audit(full_payload=True)
    word_one = arena._word(1)
    struct.pack_into("<Q", owner, WORDS_OFFSET + 8, word_one & ~1)
    try:
        arena.audit(full_payload=False)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing selected bit accepted")
    struct.pack_into("<Q", owner, WORDS_OFFSET + 8, word_one)
    prefix_one = struct.unpack_from("<H", owner, PREFIX_OFFSET + 2)[0]
    struct.pack_into("<H", owner, PREFIX_OFFSET + 2, 0)
    try:
        arena.audit(full_payload=False)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt rank prefix accepted")
    struct.pack_into("<H", owner, PREFIX_OFFSET + 2, prefix_one)
    struct.pack_into(
        "<Q", owner, WORDS_OFFSET + 8 * (WORDS - 1),
        arena._word(WORDS - 1) | (1 << 63),
    )
    try:
        arena.audit(full_payload=False)
    except AssertionError:
        pass
    else:
        raise AssertionError("high-tail bit corruption accepted")
    struct.pack_into(
        "<Q", owner, WORDS_OFFSET + 8 * (WORDS - 1),
        arena._word(WORDS - 1) & ~(1 << 63),
    )
    if _active(True, False, True):
        raise AssertionError("P=false physical-complete route activated")
    if not (
        CompactI32Arena.__slots__ == ("_owner", "_released")
        and type(arena._owner) is bytearray
        and len(arena._owner) == TOTAL_BYTES
        and not any(
            isinstance(getattr(arena, slot), np.ndarray)
            for slot in CompactI32Arena.__slots__
        )
    ):
        raise AssertionError("single-owner liveness contract failed")
    released = arena.release()
    return {
        "schema": "laion1m-compact-i32-arena-selftest/v1",
        "status": "PASS",
        "logical_positions": LOGICAL_POSITIONS,
        "full_key_position_bijection": True,
        "lexicographic_order_counterexample": True,
        "rank_before_bit_63_64": True,
        "absent_bit_rejected": True,
        "missing_selected_bit_rejected": True,
        "corrupt_rank_prefix_rejected": True,
        "last_word_high_tail_rejected": True,
        "p_false_complete_route_inactive": True,
        "single_persistent_owner": True,
        "persistent_numpy_views": 0,
        "cx_accounted_bytes": TOTAL_BYTES,
        "release": released,
        "real_data_loaded": False,
        "performance_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        raise SystemExit("choose --self-test")
    print(json.dumps(semantic_self_test(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
