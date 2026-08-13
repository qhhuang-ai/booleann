#!/usr/bin/env python3
"""Reusable ACK-then-release publisher for LAION paired formal blocks.

This module performs no query, ground-truth, recall, or outcome work.  It
reuses the established SIFT v2 adapter's atomic no-replace preparation/link/
directory-fsync primitives and supplies only LAION schema and block binding.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Callable, Optional

from sift10m_range_v2_cpp_adapter import (
    finish_prepared_json_durability,
    fsync_directory,
    link_prepared_json_atomic_exclusive,
    prepare_json_atomic_exclusive,
)


ACK_SCHEMA = "laion1m-paired-formal-block-adapter-ack/v1"
RELEASE_SCHEMA = "laion1m-paired-formal-block-adapter-release/v1"
MAX_CONTRACT_BYTES = 64 * 1024

StopRequested = Callable[[], Optional[int]]
DurabilityCommit = Callable[[Path, Path], None]


class BlockPublicationStopped(RuntimeError):
    """The stop decision won before an ACK or release publication edge."""


class BlockReleaseNotAuthorized(RuntimeError):
    """Release preparation failed before the canonical permission edge."""


@dataclass(frozen=True)
class FileSnapshot:
    device: int
    inode: int
    bytes: int
    modified_ns: int


@dataclass(frozen=True)
class PublishedBlockAck:
    block: int
    state_id: str
    summary_bytes: int
    results_bytes: int
    path: Path
    payload: bytes
    commit_lock: Any


def validate_state_id(state_id: str) -> None:
    if not isinstance(state_id, str) or not 1 <= len(state_id) <= 128:
        raise ValueError("state_id must contain 1..128 safe characters")
    if any(not (value.isascii() and (value.isalnum() or value in "._-"))
           for value in state_id):
        raise ValueError(
            "state_id may contain only ASCII letters, digits, dot, underscore, or dash"
        )


def block_stem(block: int) -> str:
    if (not isinstance(block, int) or isinstance(block, bool) or
            not 0 <= block <= 9999):
        raise ValueError("block must be an integer in 0..9999")
    return f"block_{block:02d}"


def canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def snapshot_nonempty_regular_file(path: Path, label: str) -> FileSnapshot:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise RuntimeError(f"{label} must be a nonempty regular file")
    return FileSnapshot(
        device=value.st_dev,
        inode=value.st_ino,
        bytes=value.st_size,
        modified_ns=value.st_mtime_ns,
    )


def _read_small_regular_file(path: Path, label: str) -> bytes:
    before = snapshot_nonempty_regular_file(path, label)
    if before.bytes > MAX_CONTRACT_BYTES:
        raise RuntimeError(f"{label} exceeds the bounded contract size")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        chunks: list[bytes] = []
        remaining = before.bytes
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise RuntimeError(f"short read from {label}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while being read")
        after_stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = FileSnapshot(
        device=after_stat.st_dev,
        inode=after_stat.st_ino,
        bytes=after_stat.st_size,
        modified_ns=after_stat.st_mtime_ns,
    )
    if before != after:
        raise RuntimeError(f"{label} changed while being read")
    return b"".join(chunks)


def _raise_if_stopped(stop_requested: StopRequested, stage: str) -> None:
    signal_number = stop_requested()
    if signal_number is not None:
        raise BlockPublicationStopped(
            f"stop decision won before {stage}: {signal_number}"
        )


def _cleanup_private(temporary: Path) -> None:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass


def _ack_value(
    *, state_id: str, block: int, summary_bytes: int, results_bytes: int,
) -> dict[str, object]:
    return {
        "schema": ACK_SCHEMA,
        "state_id": state_id,
        "pass": block,
        # These shared field names intentionally remain compatible with
        # pass_release_contract.h: the first binds this block's summary.
        "passes_jsonl_bytes": summary_bytes,
        "results_jsonl_bytes": results_bytes,
    }


def publish_block_ack(
    *,
    ack_directory: Path,
    state_id: str,
    block: int,
    summary_path: Path,
    results_path: Path,
    stop_requested: StopRequested = lambda: None,
    commit_lock: Any | None = None,
    durability_commit: DurabilityCommit = finish_prepared_json_durability,
) -> PublishedBlockAck:
    """Publish and directory-fsync one no-replace block ACK.

    The caller's ``commit_lock`` must also serialize its timeout/signal latch.
    Raw summary/result files must already be independently durable.
    """

    validate_state_id(state_id)
    stem = block_stem(block)
    try:
        directory_status = ack_directory.lstat()
    except FileNotFoundError as error:
        raise ValueError(
            "ack_directory must be an existing absolute real directory"
        ) from error
    if (not ack_directory.is_absolute() or
            not stat.S_ISDIR(directory_status.st_mode)):
        raise ValueError(
            "ack_directory must be an existing absolute real directory"
        )
    lock = commit_lock if commit_lock is not None else threading.Lock()
    summary = snapshot_nonempty_regular_file(summary_path, "formal block summary")
    results = snapshot_nonempty_regular_file(results_path, "formal block results")
    if (summary.device, summary.inode) == (results.device, results.inode):
        raise RuntimeError("formal block summary and results must be distinct files")
    value = _ack_value(
        state_id=state_id,
        block=block,
        summary_bytes=summary.bytes,
        results_bytes=results.bytes,
    )
    payload = canonical_json_bytes(value)
    acknowledgement = ack_directory / f"{stem}.json"
    temporary = prepare_json_atomic_exclusive(acknowledgement, value)
    try:
        if (snapshot_nonempty_regular_file(summary_path, "formal block summary") != summary or
                snapshot_nonempty_regular_file(results_path, "formal block results") != results):
            raise RuntimeError(
                "formal block summary/results changed before ACK publication"
            )
        with lock:
            _raise_if_stopped(stop_requested, "ACK publication")
            link_prepared_json_atomic_exclusive(acknowledgement, temporary)
        # Slow directory fsync deliberately stays outside the linearization lock.
        durability_commit(acknowledgement, temporary)
    except BaseException:
        _cleanup_private(temporary)
        raise
    observed = _read_small_regular_file(acknowledgement, "block acknowledgement")
    if observed != payload:
        raise RuntimeError("published block acknowledgement is not exact canonical JSON")
    return PublishedBlockAck(
        block=block,
        state_id=state_id,
        summary_bytes=summary.bytes,
        results_bytes=results.bytes,
        path=acknowledgement,
        payload=payload,
        commit_lock=lock,
    )


def publish_block_release(
    published_ack: PublishedBlockAck,
    *,
    stop_requested: StopRequested = lambda: None,
    durability_commit: DurabilityCommit = finish_prepared_json_durability,
) -> Path:
    """Publish release only after revalidating the exact durable ACK bytes.

    ``*.release.ready.json`` is a durable, non-authorizing recovery name.  The
    runner recognizes only ``*.release.json``.  Consequently every fallible
    durability operation precedes the canonical hard-link linearization edge:
    a preparation/fsync failure leaves the next block unauthorized, while a
    canonical link that becomes visible has already won and is recoverable from
    the durable ready inode.
    """

    observed_ack = _read_small_regular_file(
        published_ack.path, "block acknowledgement"
    )
    if observed_ack != published_ack.payload:
        raise RuntimeError("block acknowledgement changed before release")
    release_value = {
        "schema": RELEASE_SCHEMA,
        "state_id": published_ack.state_id,
        "pass": published_ack.block,
        "ack_sha256": hashlib.sha256(observed_ack).hexdigest(),
    }
    release = published_ack.path.with_name(
        f"{block_stem(published_ack.block)}.release.json"
    )
    ready = published_ack.path.with_name(
        f"{block_stem(published_ack.block)}.release.ready.json"
    )
    release_payload = canonical_json_bytes(release_value)

    # A prior interrupted attempt may have linked the non-authorizing ready
    # name before its directory fsync failed.  Exact bytes plus a successful
    # retry fsync turn that inode into the durable recovery anchor.
    if ready.exists():
        if _read_small_regular_file(ready, "block release ready record") != release_payload:
            raise RuntimeError("existing block release ready record is not exact")
        try:
            fsync_directory(ready.parent)
        except BaseException as error:
            raise BlockReleaseNotAuthorized(
                "release permission remains absent; ready-record directory "
                "durability retry failed"
            ) from error
    else:
        temporary = prepare_json_atomic_exclusive(ready, release_value)
        try:
            link_prepared_json_atomic_exclusive(ready, temporary)
            durability_commit(ready, temporary)
        except BaseException as error:
            _cleanup_private(temporary)
            raise BlockReleaseNotAuthorized(
                "release permission remains absent; a visible ready record is "
                "non-authorizing and may be retried"
            ) from error
        if _read_small_regular_file(ready, "block release ready record") != release_payload:
            raise RuntimeError("durable block release ready record is not exact")

    # The canonical hard link is the sole permission edge.  Do not place a
    # directory fsync, reread, cleanup, or other fallible operation after it.
    # If a crash loses this alias, recovery recreates it from the durable ready
    # inode; if it remains visible, the runner's next block was authorized.
    try:
        with published_ack.commit_lock:
            _raise_if_stopped(stop_requested, "release publication")
            link_prepared_json_atomic_exclusive(release, ready)
    except BaseException:
        # An exception after link visibility (for example, signal delivery as
        # the masked commit exits) cannot revoke permission.  Same-inode
        # identity makes that outcome explicit and idempotent.
        try:
            if release.samefile(ready):
                return release
        except (FileNotFoundError, OSError):
            pass
        raise
    return release


def publish_block_ack_then_release(
    *,
    ack_directory: Path,
    state_id: str,
    block: int,
    summary_path: Path,
    results_path: Path,
    stop_requested: StopRequested = lambda: None,
    commit_lock: Any | None = None,
) -> PublishedBlockAck:
    """Convenience adapter boundary: durable ACK first, then bound release."""

    published = publish_block_ack(
        ack_directory=ack_directory,
        state_id=state_id,
        block=block,
        summary_path=summary_path,
        results_path=results_path,
        stop_requested=stop_requested,
        commit_lock=commit_lock,
    )
    publish_block_release(published, stop_requested=stop_requested)
    return published
