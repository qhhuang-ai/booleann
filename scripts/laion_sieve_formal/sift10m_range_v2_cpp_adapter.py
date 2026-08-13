#!/usr/bin/env python3
"""Pass-boundary adapter for v2 C++ binaries with zero timed-region I/O."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Optional

from sift10m_range_v2_common import append_jsonl


PassConverter = Callable[
    [int, Path, Path], tuple[dict[str, Any], list[dict[str, Any]]]
]
BeforeAck = Callable[
    [int, dict[str, Any], dict[str, Optional[int]]], Optional[dict[str, Any]]
]
AckPrecommitHook = Callable[[], None]


# ``stop_requested`` is sampled on the ACK publication lock so its production
# implementation must be a non-blocking in-memory read.  Enforcing a generous
# upper bound detects accidental filesystem/network/blocking implementations
# without imposing telemetry I/O on the timed child.
STOP_POLL_MAX_SECONDS = 0.100


class AdapterStop(RuntimeError):
    pass


def terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 15.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def process_memory_kb(pid: int) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "rss_kb": None, "rss_high_water_kb": None, "vm_swap_kb": None,
    }
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return result
    for line in lines:
        if line.startswith("VmRSS:"):
            result["rss_kb"] = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            result["rss_high_water_kb"] = int(line.split()[1])
        elif line.startswith("VmSwap:"):
            result["vm_swap_kb"] = int(line.split()[1])
    return result


def append_many_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(descriptor, "ab", buffering=0) as handle:
        view = memoryview(payload)
        while view:
            written = handle.write(view)
            if written is None or written <= 0:
                raise RuntimeError(f"short append while writing {path}")
            view = view[written:]
        os.fsync(handle.fileno())


def prepare_json_atomic_exclusive(path: Path, value: dict[str, Any]) -> Path:
    """Write and fsync a private JSON inode without publishing ``path``."""

    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()
    temporary = path.parent / (
        f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            view = memoryview(payload)
            while view:
                written = handle.write(view)
                if written is None or written <= 0:
                    raise RuntimeError(f"short atomic JSON write while writing {path}")
                view = view[written:]
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def commit_prepared_json_atomic_exclusive(path: Path, temporary: Path) -> None:
    """Publish a prepared inode at ``path`` atomically and without replacement."""

    try:
        # A hard link is an atomic no-replace publication: link() fails if the
        # final acknowledgement already exists, unlike os.replace().
        os.link(temporary, path)
        os.unlink(temporary)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def publish_json_atomic_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Durably publish JSON without exposing a partial final acknowledgement."""

    temporary = prepare_json_atomic_exclusive(path, value)
    commit_prepared_json_atomic_exclusive(path, temporary)


def fsync_directory(path: Path) -> None:
    """Make prior directory-entry operations durable."""

    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def link_prepared_json_atomic_exclusive(path: Path, temporary: Path) -> None:
    """Atomically reserve ``path``; durability is completed separately."""

    os.link(temporary, path)


def finish_prepared_json_durability(path: Path, temporary: Path) -> None:
    """Remove the private name and durably commit the published name."""

    temporary.unlink()
    fsync_directory(path.parent)


def read_started_passes(path: Path) -> list[int]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        return []
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames != ["event", "pass"]:
        raise RuntimeError("C++ pass-event CSV has an unexpected schema")
    rows = list(reader)
    passes = []
    for row in rows:
        if row["event"] != "pass_start":
            raise RuntimeError("C++ pass-event CSV contains an unknown event")
        passes.append(int(row["pass"]))
    if passes != list(range(1, len(passes) + 1)) or len(set(passes)) != len(passes):
        raise RuntimeError("C++ pass-start events are not the exact ordered prefix 1..p")
    return passes


def raw_pass_paths(raw_root: Path, pass_number: int) -> tuple[Path, Path]:
    return (
        raw_root / f"pass_{pass_number:02d}_summary.csv",
        raw_root / f"pass_{pass_number:02d}_results.csv",
    )


def emit_progress(progress: Path, event: dict[str, Any]) -> None:
    record = dict(event)
    record.setdefault("pid", os.getpid())
    record.setdefault("utc", datetime.now(timezone.utc).isoformat())
    append_jsonl(progress, record)


def run_cpp_adapter(
    *,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    run_dir: Path,
    state_id: str,
    expected_passes: int,
    timeout_seconds: float,
    convert_pass: PassConverter,
    stop_requested: Callable[[], int | None],
    before_ack: BeforeAck | None = None,
    heartbeat_seconds: float = 30.0,
    ack_precommit_hook: AckPrecommitHook | None = None,
) -> int:
    """Run one C++ point and publish only independently audited pass outputs.

    The C++ process writes no telemetry inside its worker-timed interval. After
    each pass it atomically publishes a one-row summary CSV and a 1,000-row raw
    result CSV, then waits for an adapter release marker bound to the durable
    acknowledgement before starting the next pass. This prevents adapter
    parsing/auditing from overlapping another timed pass.

    ``stop_requested`` is a production synchronization primitive, not a
    general callback: it must be a non-blocking, side-effect-free in-memory
    read returning ``None`` or an integer signal number.  It is invoked while
    serializing the final timeout/stop decision against ACK publication.
    """

    if (not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0 or
            heartbeat_seconds > 30):
        raise ValueError("heartbeat_seconds must be finite and in (0, 30]")
    if not isinstance(expected_passes, int) or isinstance(expected_passes, bool) or expected_passes <= 0:
        raise ValueError("expected_passes must be a positive integer")
    if (not isinstance(timeout_seconds, (int, float)) or
            isinstance(timeout_seconds, bool) or
            not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    if not callable(stop_requested) or inspect.iscoroutinefunction(stop_requested):
        raise TypeError("stop_requested must be a synchronous non-blocking callable")

    raw_root = run_dir / "cpp_raw"
    ack_root = run_dir / "adapter_ack"
    events_path = raw_root / "pass_events.csv"
    passes_jsonl = run_dir / "passes.jsonl"
    results_jsonl = run_dir / "results.jsonl"
    progress = run_dir / "child_progress.jsonl"
    log_path = run_dir / "run.log"
    stale_streams = [
        path for path in (passes_jsonl, results_jsonl, progress)
        if os.path.lexists(path)
    ]
    if stale_streams:
        raise FileExistsError(
            "adapter refuses stale append-only stream(s): "
            + ", ".join(str(path) for path in stale_streams)
        )
    raw_root.mkdir(exist_ok=False)
    ack_root.mkdir(exist_ok=False)
    started = time.monotonic()
    timeout_deadline = started + float(timeout_seconds)
    emitted_starts: set[int] = set()
    completed = 0
    latest_metrics: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    monitor_stop = threading.Event()
    stop_latched = threading.Event()
    state_lock = threading.Lock()
    stop_poll_lock = threading.Lock()
    progress_lock = threading.Lock()
    # This lock defines publication linearization.  Every stop/deadline
    # observation and each final no-replace link share it.  The ACK link only
    # reserves evidence; the release link is the next-pass linearization point.
    # A stop observed before either corresponding critical section wins that
    # publication decision.
    commit_lock = threading.RLock()
    monitor_thread: threading.Thread | None = None
    monitor_error: list[BaseException] = []
    stop_reason: list[dict[str, Any]] = []
    stop_trigger_emitted = False
    monitor_state: dict[str, Any] = {
        "phase": "launching_cpp",
        "completed_passes": 0,
        "total_passes": expected_passes,
        "current_pass": None,
        "latest_valid_metrics": None,
    }

    def update_monitor_state(**changes: Any) -> None:
        with state_lock:
            monitor_state.update(changes)

    def write_progress(event: dict[str, Any]) -> None:
        # Both the adapter and its monitor write this parent-owned stream.  A
        # single lock preserves whole-record and monotonic-elapsed ordering.
        with progress_lock:
            record = dict(event)
            record["elapsed_seconds"] = time.monotonic() - started
            emit_progress(progress, record)

    def latch_reason_locked(reason: dict[str, Any]) -> None:
        """Retain exactly one reason while the caller owns ``commit_lock``."""

        with state_lock:
            if not stop_reason:
                stop_reason.append(dict(reason))
                stop_latched.set()

    def latch_reason(reason: dict[str, Any]) -> None:
        with commit_lock:
            latch_reason_locked(reason)

    def refresh_stop_latch() -> None:
        """Poll the caller's signal state and retain the first observation."""

        with commit_lock:
            if stop_latched.is_set():
                return
            # Deadline evaluation is part of the same critical section as the
            # first-reason latch and final publication link.  A timeout already
            # due can therefore never lose merely because the monitor has not
            # yet scheduled another iteration.
            if time.monotonic() >= timeout_deadline:
                latch_reason_locked({
                    "kind": "point_timeout",
                    "reason": "frozen per-point timeout exceeded",
                })
                return
            poll_started = time.monotonic()
            try:
                # stop_requested may read caller-owned state; serialize calls
                # from the adapter and monitor so it never needs to be
                # thread-safe.  Polling while holding commit_lock makes the
                # observation itself linearizable against ACK publication.
                with stop_poll_lock:
                    signal_number = stop_requested()
            except BaseException as exc:
                poll_finished = time.monotonic()
                with state_lock:
                    if not monitor_error:
                        monitor_error.append(exc)
                if poll_finished >= timeout_deadline:
                    latch_reason_locked({
                        "kind": "point_timeout",
                        "reason": "frozen per-point timeout exceeded",
                    })
                else:
                    latch_reason_locked({
                        "kind": "stop_poll_failure", "reason": repr(exc),
                    })
                return
            poll_finished = time.monotonic()
            # If the callback crossed the frozen deadline, timeout was due no
            # later than its return and wins this sampling transaction.
            if poll_finished >= timeout_deadline:
                latch_reason_locked({
                    "kind": "point_timeout",
                    "reason": "frozen per-point timeout exceeded",
                })
                return
            if poll_finished - poll_started > STOP_POLL_MAX_SECONDS:
                error = RuntimeError(
                    "stop_requested violated the non-blocking production contract: "
                    f"{poll_finished - poll_started:.6f}s > {STOP_POLL_MAX_SECONDS:.3f}s"
                )
                with state_lock:
                    if not monitor_error:
                        monitor_error.append(error)
                latch_reason_locked({
                    "kind": "stop_poll_contract_violation",
                    "reason": str(error),
                })
                return
            if signal_number is not None:
                if not isinstance(signal_number, int) or isinstance(signal_number, bool):
                    error = TypeError(
                        "stop_requested must return None or an integer signal number"
                    )
                    with state_lock:
                        if not monitor_error:
                            monitor_error.append(error)
                    latch_reason_locked({
                        "kind": "stop_poll_failure", "reason": str(error),
                    })
                else:
                    latch_reason_locked({
                        "kind": "monitored_signal", "signal": signal_number,
                    })

    def monitor_loop(child_pid: int) -> None:
        """Observe callbacks without touching C++ input, output, or workers."""

        next_heartbeat = time.monotonic()
        poll_seconds = min(0.2, heartbeat_seconds)
        peak_rss_kb: int | None = None
        peak_hwm_kb: int | None = None
        try:
            while not monitor_stop.is_set():
                refresh_stop_latch()
                now = time.monotonic()
                if now >= next_heartbeat:
                    with state_lock:
                        snapshot = dict(monitor_state)
                        reason = dict(stop_reason[0]) if stop_reason else None
                    memory = process_memory_kb(child_pid)
                    if memory["rss_kb"] is not None:
                        peak_rss_kb = max(peak_rss_kb or 0, int(memory["rss_kb"]))
                    if memory["rss_high_water_kb"] is not None:
                        peak_hwm_kb = max(
                            peak_hwm_kb or 0, int(memory["rss_high_water_kb"])
                        )
                    completed_passes = int(snapshot["completed_passes"])
                    eta_seconds: float | str = (
                        (now - started) / completed_passes
                        * (expected_passes - completed_passes)
                        if completed_passes else "unknown"
                    )
                    write_progress({
                        "event": "adapter_heartbeat",
                        "state_id": state_id,
                        "child_pid": child_pid,
                        **snapshot,
                        "total_passes": expected_passes,
                        "completed_work": completed_passes,
                        "total_work": expected_passes,
                        "work_unit": "passes",
                        "latest_metrics": (
                            snapshot["latest_valid_metrics"]
                            if snapshot["latest_valid_metrics"] is not None
                            else "unknown"
                        ),
                        "current_rss_kb": memory["rss_kb"],
                        "peak_rss_kb": peak_rss_kb,
                        "rss_kb": memory["rss_kb"],
                        "rss_high_water_kb": memory["rss_high_water_kb"],
                        "vm_swap_kb": memory["vm_swap_kb"],
                        "peak_hwm_kb": peak_hwm_kb,
                        "eta_seconds": eta_seconds,
                        "stop_latched": reason is not None,
                        "stop_reason": reason,
                    })
                    next_heartbeat = now + heartbeat_seconds
                monitor_stop.wait(poll_seconds)
        except BaseException as exc:
            with state_lock:
                if not monitor_error:
                    monitor_error.append(exc)
            latch_reason({"kind": "monitor_failure", "reason": repr(exc)})

    def raise_if_stopped(stage: str) -> None:
        """Refresh and fail before ACK, retaining one durable stop record."""

        nonlocal stop_trigger_emitted
        refresh_stop_latch()
        with state_lock:
            error = monitor_error[0] if monitor_error else None
            reason = dict(stop_reason[0]) if stop_reason else None
            snapshot = dict(monitor_state)
        if reason is None:
            if error is not None:
                raise RuntimeError("adapter monitor failed") from error
            return
        if not stop_trigger_emitted:
            memory = process_memory_kb(process.pid) if process is not None else {
                "rss_kb": None, "rss_high_water_kb": None, "vm_swap_kb": None,
            }
            write_progress({
                "event": "early_stop_trigger",
                "state_id": state_id,
                "child_pid": process.pid if process is not None else None,
                "pass": snapshot["completed_passes"] or snapshot["current_pass"],
                "completed_passes": snapshot["completed_passes"],
                "latest_valid_metrics": snapshot["latest_valid_metrics"],
                "rss_kb": memory["rss_kb"],
                "rss_high_water_kb": memory["rss_high_water_kb"],
                "vm_swap_kb": memory["vm_swap_kb"],
                "stage": stage,
                "trigger": reason,
            })
            stop_trigger_emitted = True
        if error is not None:
            raise RuntimeError(
                "adapter monitor failed after durable early-stop trigger"
            ) from error
        raise AdapterStop(
            f"adapter stopped at {stage}: " + json.dumps(reason, sort_keys=True)
        )

    try:
        with log_path.open("xb") as log_handle:
            raise_if_stopped("before_cpp_spawn")
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            update_monitor_state(phase="waiting_for_cpp")
            monitor_thread = threading.Thread(
                target=monitor_loop,
                args=(process.pid,),
                name="sift10m-range-adapter-monitor",
                daemon=True,
            )
            monitor_thread.start()
            try:
                while True:
                    raise_if_stopped("adapter_loop")

                    started_passes = read_started_passes(events_path)
                    for pass_number in started_passes:
                        if pass_number not in emitted_starts:
                            if pass_number != completed + 1:
                                raise RuntimeError(
                                    "C++ started a new pass before the prior adapter acknowledgement"
                                )
                            update_monitor_state(
                                phase="waiting_for_cpp", current_pass=pass_number,
                            )
                            write_progress({
                                "event": "pass_start",
                                "state_id": state_id,
                                "child_pid": process.pid,
                                "pass": pass_number,
                                "completed_passes": completed,
                                "total_passes": expected_passes,
                                "completed_work": completed,
                                "total_work": expected_passes,
                                "work_unit": "passes",
                            })
                            emitted_starts.add(pass_number)

                    next_pass = completed + 1
                    if next_pass <= expected_passes and next_pass in emitted_starts:
                        summary_path, results_path = raw_pass_paths(raw_root, next_pass)
                        if summary_path.exists() and results_path.exists():
                            update_monitor_state(
                                phase="convert_pass", current_pass=next_pass,
                            )
                            pass_record, result_records = convert_pass(
                                next_pass, summary_path, results_path
                            )
                            # Observe a stop immediately after the synchronous
                            # converter.  Delay raising only long enough to make
                            # its already-produced pass evidence durable.
                            refresh_stop_latch()
                            append_many_jsonl(results_jsonl, result_records)
                            append_many_jsonl(passes_jsonl, [pass_record])
                            completed = next_pass
                            latest_metrics = {
                                "qps": pass_record["qps"],
                                "strict_recall": (
                                    pass_record["strict_correct"] / pass_record["strict_total"]
                                ),
                                "tie_aware_recall": (
                                    pass_record["tie_correct"] / pass_record["tie_total"]
                                ),
                            }
                            update_monitor_state(
                                phase="pass_converted",
                                completed_passes=completed,
                                current_pass=completed,
                                latest_valid_metrics=latest_metrics,
                            )
                            elapsed = time.monotonic() - started
                            memory = process_memory_kb(process.pid)
                            write_progress({
                                "event": "pass_complete",
                                "state_id": state_id,
                                "child_pid": process.pid,
                                "pass": completed,
                                "completed_passes": completed,
                                "total_passes": expected_passes,
                                "completed_work": completed,
                                "total_work": expected_passes,
                                "work_unit": "passes",
                                "latest_valid_metrics": latest_metrics,
                                "latest_metrics": latest_metrics,
                                "current_rss_kb": memory["rss_kb"],
                                "rss_kb": memory["rss_kb"],
                                "rss_high_water_kb": memory["rss_high_water_kb"],
                                "vm_swap_kb": memory["vm_swap_kb"],
                                "peak_rss_kb": memory["rss_high_water_kb"],
                                "peak_hwm_kb": memory["rss_high_water_kb"],
                                "eta_seconds": (
                                    elapsed / completed * (expected_passes - completed)
                                ),
                            })
                            raise_if_stopped("after_convert_pass")
                            # This is the sole safe online-kill boundary: the C++
                            # timed workers have ended, raw output and independently
                            # converted JSONL are durable, and the child is blocked
                            # waiting for an acknowledgement.  A frozen controller
                            # may inspect only the completed pass and direct-child
                            # memory here.  If it stops, no acknowledgement is ever
                            # published and therefore no later timed pass can start.
                            if before_ack is not None:
                                update_monitor_state(phase="before_ack")
                                try:
                                    trigger = before_ack(
                                        completed, dict(pass_record), dict(memory)
                                    )
                                except AdapterStop as exc:
                                    trigger = {
                                        "kind": "before_ack_exception",
                                        "reason": str(exc),
                                    }
                                # This is intentionally distinct from the
                                # monitor's asynchronous polling: it closes the
                                # synchronous callback boundary itself.
                                refresh_stop_latch()
                                raise_if_stopped("after_before_ack")
                                if trigger is not None:
                                    if not isinstance(trigger, dict) or not trigger:
                                        raise RuntimeError(
                                            "before_ack must return None or a nonempty trigger object"
                                        )
                                    write_progress({
                                        "event": "early_stop_trigger",
                                        "state_id": state_id,
                                        "child_pid": process.pid,
                                        "pass": completed,
                                        "completed_passes": completed,
                                        "total_passes": expected_passes,
                                        "completed_work": completed,
                                        "total_work": expected_passes,
                                        "work_unit": "passes",
                                        "latest_valid_metrics": latest_metrics,
                                        "latest_metrics": latest_metrics,
                                        "current_rss_kb": memory["rss_kb"],
                                        "rss_kb": memory["rss_kb"],
                                        "rss_high_water_kb": memory["rss_high_water_kb"],
                                        "vm_swap_kb": memory["vm_swap_kb"],
                                        "peak_rss_kb": memory["rss_high_water_kb"],
                                        "peak_hwm_kb": memory["rss_high_water_kb"],
                                        "stage": "before_ack_gate",
                                        "trigger": trigger,
                                    })
                                    stop_trigger_emitted = True
                                    raise AdapterStop(
                                        "frozen before-ack gate stopped the point: "
                                        + json.dumps(trigger, sort_keys=True)
                                    )
                            update_monitor_state(phase="pre_ack")
                            # This check catches stops before potentially slow
                            # ACK serialization/fsync.  It is not the commit
                            # check: a monitor may still latch while the private
                            # inode is prepared below.
                            raise_if_stopped("immediately_before_ack")
                            acknowledgement = ack_root / f"pass_{completed:02d}.json"
                            release = ack_root / f"pass_{completed:02d}.release.json"
                            prepared_ack = prepare_json_atomic_exclusive(
                                acknowledgement, {
                                    "schema": "sift10m-range-v2-adapter-ack/v1",
                                    "state_id": state_id,
                                    "pass": completed,
                                    "passes_jsonl_bytes": passes_jsonl.stat().st_size,
                                    "results_jsonl_bytes": results_jsonl.stat().st_size,
                                },
                            )
                            try:
                                # Test-only fault injection runs after the
                                # private ACK inode is durable but before commit.
                                # The default production path has no callback.
                                if ack_precommit_hook is not None:
                                    update_monitor_state(phase="ack_precommit")
                                    ack_precommit_hook()
                                with commit_lock:
                                    # refresh_stop_latch re-enters this same
                                    # lock.  No monitor can observe/latch a stop
                                    # or newly-due timeout between this check and
                                    # the ACK reservation hard-link.
                                    raise_if_stopped("ack_precommit")
                                    link_prepared_json_atomic_exclusive(
                                        acknowledgement, prepared_ack,
                                    )
                                # The child does not consume ACK reservation.
                                # Directory fsync is deliberately outside the
                                # commit lock, so the parent heartbeat and stop
                                # sampler remain live even on a slow filesystem.
                                update_monitor_state(phase="ack_directory_fsync")
                                finish_prepared_json_durability(
                                    acknowledgement, prepared_ack,
                                )
                            except BaseException:
                                try:
                                    prepared_ack.unlink()
                                except FileNotFoundError:
                                    pass
                                raise

                            # A release marker, not ACK existence, permits the
                            # C++ child to start another timed pass.  Therefore
                            # ACK may be visible but cannot release the child
                            # until its directory entry is durably fsynced.
                            raise_if_stopped("after_ack_durability")
                            ack_raw = acknowledgement.read_bytes()
                            prepared_release = prepare_json_atomic_exclusive(
                                release, {
                                    "schema": "sift10m-range-v2-adapter-release/v1",
                                    "state_id": state_id,
                                    "pass": completed,
                                    "ack_sha256": hashlib.sha256(ack_raw).hexdigest(),
                                },
                            )
                            try:
                                update_monitor_state(phase="release_precommit")
                                with commit_lock:
                                    # Stops and timeouts arising during the ACK
                                    # directory fsync win before child release.
                                    raise_if_stopped("release_precommit")
                                    link_prepared_json_atomic_exclusive(
                                        release, prepared_release,
                                    )
                                update_monitor_state(phase="release_directory_fsync")
                                finish_prepared_json_durability(
                                    release, prepared_release,
                                )
                            except BaseException:
                                try:
                                    prepared_release.unlink()
                                except FileNotFoundError:
                                    pass
                                raise
                            update_monitor_state(
                                phase="waiting_for_cpp", current_pass=None,
                            )

                    return_code = process.poll()
                    if return_code is not None:
                        if return_code != 0:
                            raise RuntimeError(f"C++ point process exited with code {return_code}")
                        if completed != expected_passes:
                            raise RuntimeError(
                                f"C++ exited after {completed}/{expected_passes} audited passes"
                            )
                        return return_code
                    time.sleep(0.2)
            finally:
                monitor_stop.set()
                if monitor_thread is not None:
                    monitor_thread.join()
    except BaseException:
        if process is not None:
            terminate_process_group(process)
        raise
