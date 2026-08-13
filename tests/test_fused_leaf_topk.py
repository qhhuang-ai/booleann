#!/usr/bin/env python3
"""Dataset-free exactness and ABI tests for the fused LAION range/DNF leaf."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAION = ROOT / "src/laion"
if str(LAION) not in sys.path:
    sys.path.insert(0, str(LAION))

import fused_leaf_topk as fused


def interval_keep(
    values: np.ndarray, row_id: int, interval: Mapping[str, Any]
) -> bool:
    value = float(values[row_id])
    if not np.isfinite(value):
        return False
    attribute = str(interval["attribute"])
    domain = 0.0 <= value <= 1.0 if attribute == "similarity" else value > 0.0
    return bool(domain and value >= float(interval["lo"]) and value < float(interval["hi"]))


def reference_support(
    row: Mapping[str, Any],
    postings: list[np.ndarray],
    numeric: Mapping[str, np.ndarray],
) -> np.ndarray:
    first = row["interval_1"]
    left = {
        int(row_id)
        for row_id in postings[int(row["primary_token_id"])]
        if interval_keep(numeric[str(first["attribute"])], int(row_id), first)
    }
    support = left
    if row["family"] == "dnf2":
        second = row["interval_2"]
        support |= {
            int(row_id)
            for row_id in postings[int(row["secondary_token_id"])]
            if interval_keep(
                numeric[str(second["attribute"])], int(row_id), second
            )
        }
    support.discard(int(row["query_base_id"]))
    return np.asarray(sorted(support), dtype=np.int32)


def reference_top10(
    base: np.ndarray, query: np.ndarray, support: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    differences = np.subtract(base[support], query, dtype=np.float32)
    distances = np.einsum(
        "ij,ij->i",
        differences,
        differences,
        dtype=np.float32,
        optimize=False,
    )
    order = np.lexsort((support, distances))[: fused.K]
    return support[order], distances[order]


def row(
    family: str,
    self_id: int,
    first: Mapping[str, Any],
    *,
    secondary: int | None = None,
    second: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "family": family,
        "primary_token_id": 0,
        "secondary_token_id": secondary,
        "query_base_id": self_id,
        "interval_1": dict(first),
        "interval_2": None if second is None else dict(second),
    }


class FusedLeafTopKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        library = Path(
            os.environ.get("LAION_FUSED_LEAF_TOPK_LIBRARY", fused.DEFAULT_LIBRARY)
        )
        cls.native = fused.NativeFusedLeafTopK(library)
        cls.rng = np.random.default_rng(20260802)
        cls.base = cls.rng.standard_normal(
            (160, fused.DIMENSION), dtype=np.float32
        )
        cls.query = cls.rng.standard_normal(fused.DIMENSION, dtype=np.float32)
        cls.width = np.arange(1, 161, dtype=np.int64)
        cls.similarity = np.linspace(0.0, 1.0, 160, dtype=np.float64)
        cls.similarity[7] = np.nan
        cls.numeric = {"width": cls.width, "similarity": cls.similarity}

    def assert_exact(
        self,
        request: Mapping[str, Any],
        postings: list[np.ndarray],
        numeric: Mapping[str, np.ndarray] | None = None,
        base: np.ndarray | None = None,
        query: np.ndarray | None = None,
    ) -> dict[str, Any]:
        numeric = self.numeric if numeric is None else numeric
        base = self.base if base is None else base
        query = self.query if query is None else query
        route = self.native.compile_request(
            request, postings, numeric, base_rows=len(base)
        )
        workspace = self.native.workspace()
        result = workspace.run_into(route, base, query)
        support = reference_support(request, postings, numeric)
        expected_ids, expected_distances = reference_top10(base, query, support)
        self.assertTrue(np.array_equal(workspace.top_ids, expected_ids))
        self.assertTrue(
            np.array_equal(
                workspace.top_squared_l2.view(np.uint32),
                expected_distances.view(np.uint32),
            )
        )
        self.assertEqual(result["unique_before_self"], len(support) + 1)
        self.assertEqual(result["distance_rows"], len(support))
        self.assertEqual(result["removed_self"], 1)
        self.assertEqual(result["materialized_union_rows"], 0)
        self.assertEqual(result["workspace_bytes"], 176)
        return result

    def test_range_exact_ids_and_float32_bits_across_ring_boundaries(self) -> None:
        interval = {"attribute": "width", "lo": 0.0, "hi": 1000.0}
        for count in (11, 12, 31, 32, 33, 39, 40, 41, 63, 64, 65, 95):
            with self.subTest(count=count):
                posting = np.arange(count, dtype=np.int32)
                request = row("range", count // 2, interval)
                result = self.assert_exact(request, [posting])
                self.assertEqual(result["left_rows_visited"], count)
                self.assertEqual(result["predicate_checks"], count)
                self.assertEqual(
                    result["interleaved_batches"], (count - 1) // 8
                )

    def test_dnf_overlap_is_scored_once_and_self_removed_once(self) -> None:
        left = np.arange(0, 150, 2, dtype=np.int32)
        right = np.arange(0, 150, 3, dtype=np.int32)
        first = {"attribute": "width", "lo": 10.0, "hi": 130.0}
        second = {"attribute": "similarity", "lo": 0.10, "hi": 0.90}
        request = row(
            "dnf2", 60, first, secondary=1, second=second
        )
        result = self.assert_exact(request, [left, right])
        left_pass = {
            int(value)
            for value in left
            if interval_keep(self.width, int(value), first)
        }
        right_pass = {
            int(value)
            for value in right
            if interval_keep(self.similarity, int(value), second)
        }
        self.assertEqual(result["clause_admissions"], len(left_pass) + len(right_pass))
        self.assertEqual(result["both_clauses_admitted"], len(left_pass & right_pass))
        self.assertEqual(result["unique_before_self"], len(left_pass | right_pass))
        self.assertEqual(result["distance_rows"], len(left_pass | right_pass) - 1)
        self.assertEqual(result["left_rows_visited"], len(left))
        self.assertEqual(result["right_rows_visited"], len(right))

    def test_identical_clause_postings_do_not_duplicate_scores(self) -> None:
        posting = np.arange(128, dtype=np.int32)
        first = {"attribute": "width", "lo": 0.0, "hi": 1000.0}
        request = row("dnf2", 64, first, secondary=1, second=first)
        result = self.assert_exact(request, [posting, posting])
        self.assertEqual(result["clause_admissions"], 256)
        self.assertEqual(result["both_clauses_admitted"], 128)
        self.assertEqual(result["unique_before_self"], 128)
        self.assertEqual(result["distance_rows"], 127)

    def test_cutoff_tie_has_stable_distance_then_id_order(self) -> None:
        base = np.zeros((160, fused.DIMENSION), dtype=np.float32)
        query = np.zeros(fused.DIMENSION, dtype=np.float32)
        posting = np.arange(128, dtype=np.int32)
        first = {"attribute": "width", "lo": 0.0, "hi": 1000.0}
        request = row("range", 64, first)
        result = self.assert_exact(
            request, [posting], base=base, query=query
        )
        self.assertEqual(result["cutoff_tie"], 1)
        self.assertEqual(result["top_ids"].tolist(), list(range(10)))
        self.assertEqual(
            result["top_squared_l2"].view(np.uint32).tolist(), [0] * 10
        )

    def test_nan_and_numeric_domain_are_filtered(self) -> None:
        posting = np.arange(5, 80, dtype=np.int32)
        first = {"attribute": "similarity", "lo": 0.0, "hi": 0.5}
        request = row("range", 20, first)
        result = self.assert_exact(request, [posting])
        self.assertNotIn(7, result["top_ids"].tolist())

    def test_half_open_hi_is_excluded_and_lo_is_included(self) -> None:
        posting = np.arange(32, dtype=np.int32)
        first = {"attribute": "width", "lo": 1.0, "hi": 20.0}
        request = row("range", 5, first)
        base = self.base.copy()
        base[19] = self.query
        result = self.assert_exact(request, [posting], base=base)
        # width[id] == id + 1: ID 0 equals lo and is admitted; ID 19 equals hi
        # and must be absent even though it would have exact distance zero.
        self.assertEqual(result["unique_before_self"], 19)
        self.assertEqual(result["distance_rows"], 18)
        self.assertIn(0, reference_support(request, [posting], self.numeric))
        self.assertNotIn(19, reference_support(request, [posting], self.numeric))
        self.assertNotIn(19, result["top_ids"].tolist())

    def test_wrong_native_abi_is_called_and_rejected_without_output_write(self) -> None:
        posting = np.arange(32, dtype=np.int32)
        first = {"attribute": "width", "lo": 0.0, "hi": 1000.0}
        request = row("range", 5, first)
        route = self.native.compile_request(
            request, [posting], self.numeric, base_rows=len(self.base)
        )
        output_ids = np.full(fused.K, -4242, dtype=np.int32)
        output_distances = np.full(fused.K, np.float32(-1.0), dtype=np.float32)
        audit = np.full(fused.AUDIT_SLOTS, np.uint64(0xDEADBEEF), dtype=np.uint64)
        status = int(
            self.native.execute_fn(
                fused.ABI_VERSION + 1,
                route.mode,
                self.base.ctypes.data_as(fused.F32_PTR),
                route.base_rows,
                self.query.ctypes.data_as(fused.F32_PTR),
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
                output_ids.ctypes.data_as(fused.I32_PTR),
                output_distances.ctypes.data_as(fused.F32_PTR),
                audit.ctypes.data_as(fused.U64_PTR),
                len(audit),
            )
        )
        self.assertEqual(status, -1)
        self.assertEqual(int(audit[0]), fused.ABI_VERSION)
        self.assertEqual(output_ids.tolist(), [-4242] * fused.K)
        self.assertEqual(output_distances.tolist(), [-1.0] * fused.K)

    def test_bad_posting_missing_self_and_small_support_fail_closed(self) -> None:
        for posting in (
            np.asarray([0, 2, 1, 3], dtype=np.int32),
            np.asarray([0, 1, 1, 3], dtype=np.int32),
            np.asarray([0, 1, 2, 160], dtype=np.int32),
        ):
            with self.assertRaises(RuntimeError):
                self.native.validate_posting(posting, len(self.base))

        first = {"attribute": "width", "lo": 0.0, "hi": 1000.0}
        missing = row("range", 100, first)
        missing_route = self.native.compile_request(
            missing,
            [np.arange(32, dtype=np.int32)],
            self.numeric,
            base_rows=len(self.base),
        )
        workspace = self.native.workspace()
        workspace.top_ids.fill(-777)
        workspace.top_squared_l2.view(np.uint32).fill(0xDEADBEEF)
        with self.assertRaisesRegex(RuntimeError, "MISSING_OR_MULTIPLE_SELF"):
            workspace.run_into(missing_route, self.base, self.query)
        self.assertEqual(workspace.top_ids.tolist(), [-777] * fused.K)
        self.assertEqual(
            workspace.top_squared_l2.view(np.uint32).tolist(),
            [0xDEADBEEF] * fused.K,
        )

        small = row("range", 5, first)
        small_route = self.native.compile_request(
            small,
            [np.arange(10, dtype=np.int32)],
            self.numeric,
            base_rows=len(self.base),
        )
        with self.assertRaisesRegex(RuntimeError, "SUPPORT_BELOW_K"):
            workspace.run_into(small_route, self.base, self.query)

    def test_python_input_abi_rejects_dtype_and_shape_drift(self) -> None:
        first = {"attribute": "width", "lo": 0.0, "hi": 1000.0}
        request = row("range", 5, first)
        with self.assertRaises(RuntimeError):
            self.native.compile_request(
                request,
                [np.arange(20, dtype=np.int64)],
                self.numeric,
                base_rows=len(self.base),
            )
        bad_numeric = {**self.numeric, "width": self.width.astype(np.float32)}
        with self.assertRaises(RuntimeError):
            self.native.compile_request(
                request,
                [np.arange(20, dtype=np.int32)],
                bad_numeric,
                base_rows=len(self.base),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
