#!/usr/bin/env python3
"""Checked binding for exact-support Elias--Fano encoding and top-k scan."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_LIBRARY = HERE / "native/libelias_fano_topk.so"
ABI_VERSION = 1
UNIVERSE = 1_000_000
DIMENSION = 512
K = 10
AUDIT_SLOTS = 16

STATUS = {
    0: "OK",
    -1: "BAD_ABI",
    -2: "BAD_ARGUMENT",
    -3: "BAD_INPUT_ORDER",
    -4: "CAPACITY",
    -5: "CORRUPT_DIRECTORY",
    -6: "CORRUPT_PAYLOAD",
    -7: "SUPPORT_BELOW_K",
    -8: "NONFINITE_DISTANCE",
    -9: "FP_ENVIRONMENT",
    -10: "UNSUPPORTED_CPU",
}

U8_PTR = ctypes.POINTER(ctypes.c_uint8)
I32_PTR = ctypes.POINTER(ctypes.c_int32)
U64_PTR = ctypes.POINTER(ctypes.c_uint64)
F32_PTR = ctypes.POINTER(ctypes.c_float)


class Directory(ctypes.Structure):
    _fields_ = [
        ("low_offset", ctypes.c_uint64),
        ("high_offset", ctypes.c_uint64),
        ("rows", ctypes.c_uint32),
        ("lower_bits", ctypes.c_uint8),
        ("version", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
    ]


@dataclass(frozen=True)
class Account:
    lower_bits: int
    low_bytes: int
    high_bytes: int
    payload_bytes: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_i32(values: np.ndarray, label: str) -> np.ndarray:
    require(
        type(values) is np.ndarray
        and values.dtype == np.int32
        and values.ndim == 1
        and values.flags.c_contiguous,
        f"{label} must be contiguous one-dimensional int32",
    )
    return values


class NativeEliasFanoTopK:
    def __init__(self, library: Path = DEFAULT_LIBRARY) -> None:
        self.path = library.resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        for name, restype in (
            ("abi_version", ctypes.c_uint32),
            ("universe", ctypes.c_uint64),
            ("directory_bytes", ctypes.c_uint32),
            ("dimension", ctypes.c_uint32),
            ("k", ctypes.c_uint32),
            ("audit_slots", ctypes.c_uint64),
        ):
            fn = getattr(self.library, f"laion1m_elias_fano_v1_{name}")
            fn.argtypes = ()
            fn.restype = restype
        observed = (
            int(self.library.laion1m_elias_fano_v1_abi_version()),
            int(self.library.laion1m_elias_fano_v1_universe()),
            int(self.library.laion1m_elias_fano_v1_directory_bytes()),
            int(self.library.laion1m_elias_fano_v1_dimension()),
            int(self.library.laion1m_elias_fano_v1_k()),
            int(self.library.laion1m_elias_fano_v1_audit_slots()),
        )
        require(
            observed == (
                ABI_VERSION, UNIVERSE, ctypes.sizeof(Directory),
                DIMENSION, K, AUDIT_SLOTS,
            ),
            f"native Elias--Fano ABI drift: {observed}",
        )

        self.account_fn = self.library.laion1m_elias_fano_v1_account
        self.account_fn.argtypes = [
            ctypes.c_uint32, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8), U64_PTR, U64_PTR, U64_PTR,
        ]
        self.account_fn.restype = ctypes.c_int32
        self.encode_sorted_fn = self.library.laion1m_elias_fano_v1_encode_sorted
        self.encode_sorted_fn.argtypes = [
            ctypes.c_uint32, I32_PTR, ctypes.c_uint64, U8_PTR,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(Directory),
            U64_PTR, ctypes.c_uint64,
        ]
        self.encode_sorted_fn.restype = ctypes.c_int32
        self.encode_union_fn = self.library.laion1m_elias_fano_v1_encode_union
        self.encode_union_fn.argtypes = [
            ctypes.c_uint32, I32_PTR, ctypes.c_uint64, I32_PTR,
            ctypes.c_uint64, U8_PTR, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(Directory), U64_PTR, ctypes.c_uint64,
        ]
        self.encode_union_fn.restype = ctypes.c_int32
        self.decode_fn = self.library.laion1m_elias_fano_v1_decode
        self.decode_fn.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(Directory), U8_PTR,
            ctypes.c_uint64, I32_PTR, ctypes.c_uint64, U64_PTR,
            ctypes.c_uint64,
        ]
        self.decode_fn.restype = ctypes.c_int32
        self.scan_checksum_fn = (
            self.library.laion1m_elias_fano_v1_scan_checksum
        )
        self.scan_checksum_fn.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(Directory), U8_PTR,
            ctypes.c_uint64, U64_PTR, U64_PTR, ctypes.c_uint64,
        ]
        self.scan_checksum_fn.restype = ctypes.c_int32
        self.exact_fn = self.library.laion1m_elias_fano_v1_exact_top10
        self.exact_fn.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(Directory), U8_PTR,
            ctypes.c_uint64, F32_PTR, ctypes.c_uint64, F32_PTR,
            ctypes.c_int32, I32_PTR, F32_PTR, U64_PTR, ctypes.c_uint64,
        ]
        self.exact_fn.restype = ctypes.c_int32

    def account(self, rows: int) -> Account:
        lower = ctypes.c_uint8()
        low = ctypes.c_uint64()
        high = ctypes.c_uint64()
        payload = ctypes.c_uint64()
        status = int(self.account_fn(
            ABI_VERSION, int(rows), ctypes.byref(lower), ctypes.byref(low),
            ctypes.byref(high), ctypes.byref(payload),
        ))
        require(status == 0, f"Elias--Fano account failed: {STATUS.get(status)}")
        return Account(lower.value, low.value, high.value, payload.value)

    def encode_sorted(
        self, ids: np.ndarray, owner: np.ndarray, offset: int = 0,
    ) -> tuple[Directory, np.ndarray]:
        ids = require_i32(ids, "ids")
        require(
            type(owner) is np.ndarray and owner.dtype == np.uint8
            and owner.ndim == 1 and owner.flags.c_contiguous,
            "owner must be contiguous one-dimensional uint8",
        )
        directory = Directory()
        audit = np.zeros(AUDIT_SLOTS, dtype=np.uint64)
        status = int(self.encode_sorted_fn(
            ABI_VERSION, ids.ctypes.data_as(I32_PTR), len(ids),
            owner.ctypes.data_as(U8_PTR), len(owner), int(offset),
            ctypes.byref(directory), audit.ctypes.data_as(U64_PTR),
            len(audit),
        ))
        require(status == 0, f"Elias--Fano encode failed: {STATUS.get(status)}")
        return directory, audit

    def decode(
        self, directory: Directory, owner: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        output = np.empty(int(directory.rows), dtype=np.int32)
        audit = np.zeros(AUDIT_SLOTS, dtype=np.uint64)
        status = int(self.decode_fn(
            ABI_VERSION, ctypes.byref(directory), owner.ctypes.data_as(U8_PTR),
            len(owner), output.ctypes.data_as(I32_PTR), len(output),
            audit.ctypes.data_as(U64_PTR), len(audit),
        ))
        require(status == 0, f"Elias--Fano decode failed: {STATUS.get(status)}")
        return output, audit

    def scan_checksum(
        self, directory: Directory, owner: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        checksum = ctypes.c_uint64()
        audit = np.zeros(AUDIT_SLOTS, dtype=np.uint64)
        status = int(self.scan_checksum_fn(
            ABI_VERSION, ctypes.byref(directory), owner.ctypes.data_as(U8_PTR),
            len(owner), ctypes.byref(checksum), audit.ctypes.data_as(U64_PTR),
            len(audit),
        ))
        require(status == 0, f"Elias--Fano scan failed: {STATUS.get(status)}")
        return checksum.value, audit

    def exact_top10(
        self, directory: Directory, owner: np.ndarray, base: np.ndarray,
        query: np.ndarray, self_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        require(
            type(base) is np.ndarray and base.dtype == np.float32
            and base.ndim == 2 and base.shape[1] == DIMENSION
            and base.flags.c_contiguous,
            "base must be contiguous float32[N,512]",
        )
        require(
            type(query) is np.ndarray and query.dtype == np.float32
            and query.shape == (DIMENSION,) and query.flags.c_contiguous,
            "query must be contiguous float32[512]",
        )
        ids = np.full(K, -1, dtype=np.int32)
        distances = np.full(K, np.nan, dtype=np.float32)
        audit = np.zeros(AUDIT_SLOTS, dtype=np.uint64)
        status = int(self.exact_fn(
            ABI_VERSION, ctypes.byref(directory), owner.ctypes.data_as(U8_PTR),
            len(owner), base.ctypes.data_as(F32_PTR), len(base),
            query.ctypes.data_as(F32_PTR), int(self_id),
            ids.ctypes.data_as(I32_PTR), distances.ctypes.data_as(F32_PTR),
            audit.ctypes.data_as(U64_PTR), len(audit),
        ))
        require(status == 0, f"Elias--Fano top-k failed: {STATUS.get(status)}")
        return ids, distances, audit
