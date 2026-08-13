#!/usr/bin/env python3
"""Dataset-free ABI, corruption, and exactness tests for Elias--Fano."""

from __future__ import annotations

import ctypes
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAION = ROOT / "src/laion"
sys.path.insert(0, str(LAION))
import elias_fano_topk as ef


class EliasFanoTopKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.native = ef.NativeEliasFanoTopK(Path(os.environ[
            "LAION_ELIAS_FANO_TOPK_LIBRARY"
        ]))
        cls.rng = np.random.default_rng(20260802)

    def encode(self, ids: np.ndarray) -> tuple[np.ndarray, ef.Directory]:
        account = self.native.account(len(ids))
        owner = np.zeros(64 + account.payload_bytes, dtype=np.uint8)
        directory, _ = self.native.encode_sorted(ids, owner, 64)
        return owner, directory

    def assert_roundtrip(self, ids: np.ndarray) -> None:
        owner, directory = self.encode(ids)
        decoded, audit = self.native.decode(directory, owner)
        self.assertTrue(np.array_equal(ids, decoded))
        self.assertEqual(int(audit[5]), len(ids))
        self.assertEqual(int(audit[12]), len(ids))
        checksum, scan_audit = self.native.scan_checksum(directory, owner)
        expected = 1_469_598_103_934_665_603
        for value in ids:
            expected ^= int(value)
            expected = (expected * 1_099_511_628_211) & ((1 << 64) - 1)
        self.assertEqual(checksum, expected)
        self.assertEqual(int(scan_audit[5]), len(ids))

    def test_roundtrip_boundaries_and_every_real_lower_width(self) -> None:
        for ids in (
            np.asarray([0], np.int32),
            np.asarray([999_999], np.int32),
            np.asarray([0, 999_999], np.int32),
            np.asarray([1, 2, 3, 900_000, 999_999], np.int32),
        ):
            self.assert_roundtrip(ids)
        self.assertEqual(self.native.account(15_625).lower_bits, 6)
        self.assertEqual(self.native.account(15_626).lower_bits, 5)
        for lower_bits in range(5, 13):
            rows = next(
                n for n in range(1, 50_000)
                if math.floor(math.log2(ef.UNIVERSE / n)) == lower_bits
            )
            ids = np.sort(self.rng.choice(
                ef.UNIVERSE, rows, replace=False
            ).astype(np.int32))
            self.assert_roundtrip(ids)

    def test_union_coalesces_overlap_once(self) -> None:
        left = np.asarray([0, 2, 4, 9, 999_999], np.int32)
        right = np.asarray([1, 2, 7, 9], np.int32)
        expected = np.union1d(left, right).astype(np.int32)
        account = self.native.account(len(expected))
        owner = np.zeros(64 + account.payload_bytes, np.uint8)
        directory = ef.Directory()
        audit = np.zeros(ef.AUDIT_SLOTS, np.uint64)
        status = int(self.native.encode_union_fn(
            ef.ABI_VERSION, left.ctypes.data_as(ef.I32_PTR), len(left),
            right.ctypes.data_as(ef.I32_PTR), len(right),
            owner.ctypes.data_as(ef.U8_PTR), len(owner), 64,
            ctypes.byref(directory), audit.ctypes.data_as(ef.U64_PTR),
            len(audit),
        ))
        self.assertEqual(status, 0)
        decoded, _ = self.native.decode(directory, owner)
        self.assertTrue(np.array_equal(decoded, expected))

    def test_input_order_wrong_abi_and_transactional_corruption(self) -> None:
        for ids in (
            np.asarray([1, 1], np.int32),
            np.asarray([2, 1], np.int32),
        ):
            account = self.native.account(len(ids))
            owner = np.zeros(64 + account.payload_bytes, np.uint8)
            directory = ef.Directory()
            audit = np.zeros(ef.AUDIT_SLOTS, np.uint64)
            status = int(self.native.encode_sorted_fn(
                ef.ABI_VERSION, ids.ctypes.data_as(ef.I32_PTR), len(ids),
                owner.ctypes.data_as(ef.U8_PTR), len(owner), 64,
                ctypes.byref(directory), audit.ctypes.data_as(ef.U64_PTR),
                len(audit),
            ))
            self.assertEqual(status, -3)

        ids = np.sort(self.rng.choice(
            ef.UNIVERSE, 5_000, replace=False
        ).astype(np.int32))
        owner, directory = self.encode(ids)
        output = np.full(len(ids), 424_242, np.int32)
        audit = np.zeros(ef.AUDIT_SLOTS, np.uint64)
        status = int(self.native.decode_fn(
            ef.ABI_VERSION + 1, ctypes.byref(directory),
            owner.ctypes.data_as(ef.U8_PTR), len(owner),
            output.ctypes.data_as(ef.I32_PTR), len(output),
            audit.ctypes.data_as(ef.U64_PTR), len(audit),
        ))
        self.assertEqual(status, -1)
        self.assertTrue(np.all(output == 424_242))

        corrupt = owner.copy()
        corrupt[-1] |= np.uint8(0x80)
        status = int(self.native.decode_fn(
            ef.ABI_VERSION, ctypes.byref(directory),
            corrupt.ctypes.data_as(ef.U8_PTR), len(corrupt),
            output.ctypes.data_as(ef.I32_PTR), len(output),
            audit.ctypes.data_as(ef.U64_PTR), len(audit),
        ))
        self.assertEqual(status, -6)
        self.assertTrue(np.all(output == 424_242))

        bad = ef.Directory(
            2**64 - 8, 0, directory.rows, directory.lower_bits, 1, 0
        )
        status = int(self.native.decode_fn(
            ef.ABI_VERSION, ctypes.byref(bad),
            owner.ctypes.data_as(ef.U8_PTR), len(owner),
            output.ctypes.data_as(ef.I32_PTR), len(output),
            audit.ctypes.data_as(ef.U64_PTR), len(audit),
        ))
        self.assertEqual(status, -5)

    def test_swapped_low_high_directory_offsets_fail_closed(self) -> None:
        ids = np.asarray(
            [0, 1, 7, 31, 255, 4096, 65_535, 400_000, 800_000, 999_999],
            np.int32,
        )
        owner, directory = self.encode(ids)
        account = self.native.account(len(ids))
        self.assertEqual((account.low_bytes, account.high_bytes), (24, 8))
        swapped = ef.Directory(
            directory.high_offset, directory.low_offset, directory.rows,
            directory.lower_bits, directory.version, directory.reserved,
        )
        output = np.full(len(ids), 424_242, np.int32)
        audit = np.zeros(ef.AUDIT_SLOTS, np.uint64)
        status = int(self.native.decode_fn(
            ef.ABI_VERSION, ctypes.byref(swapped),
            owner.ctypes.data_as(ef.U8_PTR), len(owner),
            output.ctypes.data_as(ef.I32_PTR), len(output),
            audit.ctypes.data_as(ef.U64_PTR), len(audit),
        ))
        self.assertEqual(status, -5)
        self.assertTrue(np.all(output == 424_242))

    def test_streaming_topk_matches_retained_bits_and_removes_self(self) -> None:
        base = self.rng.normal(size=(160, ef.DIMENSION)).astype(np.float32)
        self_id = 73
        query = base[self_id].copy()
        support = np.arange(150, dtype=np.int32)
        owner, directory = self.encode(support)
        ids, distances, audit = self.native.exact_top10(
            directory, owner, base, query, self_id
        )
        filtered = support[support != self_id]
        differences = np.subtract(base[filtered], query, dtype=np.float32)
        reference = np.einsum(
            "ij,ij->i", differences, differences,
            dtype=np.float32, optimize=False,
        )
        order = np.lexsort((filtered, reference))[: ef.K]
        self.assertTrue(np.array_equal(ids, filtered[order]))
        self.assertTrue(np.array_equal(
            distances.view(np.uint32), reference[order].view(np.uint32)
        ))
        self.assertEqual(int(audit[7]), 1)
        self.assertEqual(int(audit[6]), len(filtered))

    def test_cutoff_tie_is_stable(self) -> None:
        base = np.zeros((64, ef.DIMENSION), np.float32)
        query = np.zeros(ef.DIMENSION, np.float32)
        support = np.arange(32, dtype=np.int32)
        owner, directory = self.encode(support)
        ids, distances, audit = self.native.exact_top10(
            directory, owner, base, query, 31
        )
        self.assertEqual(ids.tolist(), list(range(10)))
        self.assertTrue(np.all(distances.view(np.uint32) == 0))
        self.assertEqual(int(audit[10]), 1)


if __name__ == "__main__":
    unittest.main()
