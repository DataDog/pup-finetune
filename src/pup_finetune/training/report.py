# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""How a remote job reports a run's state back: snapshots of its files, as log lines.

The job's worker (``_worker --report-state``) runs inside :class:`StateReporter`, which
writes a :class:`Snapshot` of the run's ``state.json``, ``epochs.json`` and ``result.json``
to stdout when it starts, whenever those files change (a heartbeat alone does not count) or
:data:`KEEPALIVE_S` passed, and once more, ``final``, when the worker exits, with the error
it recorded in ``worker.json``. A snapshot is canonical JSON, compressed and base64-encoded,
cut into lines ``PUPFT1 <run id> <seq> <i>/<n> <digest> <chunk>`` of at most
:data:`MAX_CHUNK` characters. The machine that started the run reads the job's logs back:
:func:`newest_snapshot` reassembles the run's highest complete snapshot, and
:func:`apply_snapshot` writes its files into the local run directory.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import re
import sys
import threading
import time
import zlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Self

from pup_finetune.errors import StateError
from pup_finetune.providers.base import LogLine
from pup_finetune.runs import Clock, RunDir, RunState, utcnow
from pup_finetune.storage import (
    atomic_write_json,
    canonical_json,
    finite_json,
    loads_finite,
    read_json,
)
from pup_finetune.training.records import EpochMetrics, TrainResult

__all__ = [
    "CHECK_S",
    "KEEPALIVE_S",
    "MARKER",
    "MAX_CHUNK",
    "Snapshot",
    "StateReporter",
    "apply_snapshot",
    "encode",
    "newest_snapshot",
    "read_snapshot",
]

MARKER: Final = "PUPFT1"
MAX_CHUNK: Final = 2000
"""The longest line a snapshot is written as."""
CHECK_S: Final = 15.0
KEEPALIVE_S: Final = 60.0
FORMAT: Final = 1
DIGEST_CHARS: Final = 12
MAX_SNAPSHOT_BYTES: Final = 64 * 1024 * 1024

_IGNORED: Final = frozenset({"heartbeat_at", "pid", "elapsed_seconds"})
"""What every heartbeat changes in ``state.json``, so a change there alone is not reported."""
_LINE: Final = re.compile(
    rf"{MARKER} (\S+) (\d+) (\d+)/(\d+) ([0-9a-f]{{{DIGEST_CHARS}}}) ([A-Za-z0-9+/=]+)"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Snapshot:
    """A run's files as a job's worker saw them: ``seq`` counts its reports from 0, ``at`` is
    when (the job's clock), ``final`` marks the one written as it exited, with the error it
    ended with (``exit_error``). ``log_ns`` is when the newest of its lines was logged, once
    read back."""

    run_id: str
    seq: int
    at: str
    final: bool
    files: dict[str, Any]
    exit_error: dict[str, Any] | None
    log_ns: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "run_id": self.run_id,
            "seq": self.seq,
            "at": self.at,
            "final": self.final,
            "files": self.files,
            "exit_error": self.exit_error,
        }


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:DIGEST_CHARS]


def encode(snapshot: Snapshot) -> list[str]:
    """``snapshot`` as the log lines that carry it, each at most :data:`MAX_CHUNK` long."""
    raw = canonical_json(snapshot.to_json(), ensure_ascii=True).encode("ascii")
    payload = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    digest = _digest(payload)
    head = f"{MARKER} {snapshot.run_id} {snapshot.seq} "
    room = MAX_CHUNK - len(head) - 2 * len(str(len(payload))) - 1 - len(digest) - 2
    chunks = [payload[start : start + room] for start in range(0, len(payload), room)]
    count = len(chunks)
    return [f"{head}{i}/{count} {digest} {chunk}" for i, chunk in enumerate(chunks, start=1)]


def newest_snapshot(lines: Iterable[LogLine], run_id: str) -> Snapshot | None:
    """The highest complete snapshot of run ``run_id`` in ``lines``, in any order; lines of
    other runs, and anything else, are ignored. A snapshot is complete when it has every part
    and they decode to what its digest names."""
    parts: dict[tuple[int, str], dict[int, str]] = {}
    counts: dict[tuple[int, str], int] = {}
    newest: dict[tuple[int, str], int] = {}
    for line in lines:
        match = _LINE.search(line.message)
        if match is None or match[1] != run_id:
            continue
        index, count = int(match[3]), int(match[4])
        key = (int(match[2]), match[5])
        if not 1 <= index <= count or counts.setdefault(key, count) != count:
            continue
        parts.setdefault(key, {})[index] = match[6]
        newest[key] = max(newest.get(key, line.timestamp_ns), line.timestamp_ns)
    for key in sorted(parts, reverse=True):
        data = _decoded(parts[key], counts[key], key[1])
        snapshot = None if data is None else _snapshot(data, run_id, key[0], newest[key])
        if snapshot is not None:
            return snapshot
    return None


def _decoded(chunks: Mapping[int, str], count: int, digest: str) -> dict[str, Any] | None:
    if len(chunks) != count:
        return None
    payload = "".join(chunks[index] for index in range(1, count + 1))
    if _digest(payload) != digest:
        return None
    try:
        inflate = zlib.decompressobj()
        raw = inflate.decompress(base64.b64decode(payload, validate=True), MAX_SNAPSHOT_BYTES)
        if inflate.unconsumed_tail:
            return None
        data = loads_finite(raw.decode("utf-8"))
    except (ValueError, zlib.error, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _snapshot(data: Mapping[str, Any], run_id: str, seq: int, log_ns: int) -> Snapshot | None:
    files, at, final, error = (data.get(key) for key in ("files", "at", "final", "exit_error"))
    if not (
        data.get("format") == FORMAT
        and data.get("run_id") == run_id
        and data.get("seq") == seq
        and isinstance(files, dict)
        and isinstance(at, str)
        and isinstance(final, bool)
        and (error is None or isinstance(error, dict))
    ):
        return None
    return Snapshot(
        run_id=run_id, seq=seq, at=at, final=final, files=files, exit_error=error, log_ns=log_ns
    )


def read_snapshot(run: RunDir, *, seq: int, final: bool, now: Clock = utcnow) -> Snapshot:
    """``run``'s files now; ``final`` adds the error ``worker.json`` records.

    They are read while training may write them: ``epochs.json`` first, so the
    ``state.json`` read next holds a training checkpoint at least as new as the last finished
    epoch, and ``result.json`` last, since training writes it before its final phase.
    """
    exit_error = None
    if final:
        with contextlib.suppress(FileNotFoundError, StateError):
            worker = read_json(run.worker_path)
            found = worker.get("exit_error") if isinstance(worker, dict) else None
            exit_error = found if isinstance(found, dict) else None
    epochs = run.read_epochs()
    state = read_json(run.state_path)
    result = run.read_result()
    return Snapshot(
        run_id=run.run_id,
        seq=seq,
        at=now().isoformat(timespec="seconds"),
        final=final,
        files={"state.json": state, "epochs.json": epochs, "result.json": result},
        exit_error=exit_error,
    )


def _compared(files: Mapping[str, Any]) -> dict[str, Any]:
    state = files.get("state.json")
    if isinstance(state, Mapping):
        state = {key: value for key, value in state.items() if key not in _IGNORED}
    return {**files, "state.json": state}


def _print(line: str) -> None:
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


class StateReporter:
    """Reports ``run``'s state while a remote job's worker runs; see the module docstring.

    ``write`` takes each line (by default, stdout, flushed) and ``clock`` is monotonic
    seconds. With ``thread``, a daemon thread calls :meth:`tick` every ``check_s`` seconds;
    without, the caller does. It never raises: a report it cannot read or write is skipped.
    """

    def __init__(
        self,
        run: RunDir,
        *,
        write: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        check_s: float = CHECK_S,
        keepalive_s: float = KEEPALIVE_S,
        thread: bool = True,
    ) -> None:
        self._run = run
        self._write = write or _print
        self._clock = clock
        self._check_s = check_s
        self._keepalive_s = keepalive_s
        self._threaded = thread
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._sent: dict[str, Any] | None = None
        self._sent_at = 0.0

    def __enter__(self) -> Self:
        self._report(final=False, force=True)
        if self._threaded:
            self._thread = threading.Thread(target=self._loop, name="state-reporter", daemon=True)
            self._thread.start()
        return self

    def tick(self) -> bool:
        """Report the state when it changed or ``keepalive_s`` passed since the last report;
        whether it did."""
        return self._report(final=False, force=False)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._report(final=True, force=True)

    def _loop(self) -> None:
        while not self._stop.wait(self._check_s):
            self.tick()

    def _report(self, *, final: bool, force: bool) -> bool:
        with self._lock:
            try:
                snapshot = read_snapshot(self._run, seq=self._seq, final=final)
                compared = _compared(snapshot.files)
                recent = self._clock() - self._sent_at < self._keepalive_s
                if not force and compared == self._sent and recent:
                    return False
                for line in encode(snapshot):
                    self._write(line)
            except Exception:
                return False
            self._seq += 1
            self._sent, self._sent_at = compared, self._clock()
            return True


class StaleReport(StateError):
    """A state report whose last training checkpoint does not follow its finished epochs:
    read while training wrote them, so out of date rather than damaged."""


def apply_snapshot(run: RunDir, snapshot: Snapshot) -> None:
    """Write ``snapshot``'s files into ``run``: ``epochs.json``, ``result.json`` when it has
    one, then ``state.json`` without the job's process id (its heartbeat kept). Raises
    :class:`StateError`, writing nothing, when they are not a run's files, and
    :class:`StaleReport` when a resume could not continue from them."""
    state, epochs, result = (
        snapshot.files.get(name) for name in ("state.json", "epochs.json", "result.json")
    )
    if not isinstance(state, dict):
        raise StateError("the state report holds no state.json")
    point = RunState.from_json(run, state).last_resume_point()
    if result is not None:
        if not isinstance(result, dict):
            raise StateError("the state report's result.json is not an object")
        if result.get("stopped_reason") is not None:  # an interrupted run's result has none
            TrainResult.from_json(result)
    if not isinstance(epochs, list) or not all(isinstance(epoch, dict) for epoch in epochs):
        raise StateError("the state report's epochs.json is not a list of objects")
    if point is not None and not point.follows([EpochMetrics.from_json(e) for e in epochs]):
        raise StaleReport(
            f"its last training checkpoint, at epoch {point.epoch} step {point.step}, does not "
            f"follow its {len(epochs)} finished epoch(s)"
        )
    run.write_epochs(epochs)
    if result is not None:
        run.write_result(result)
    atomic_write_json(run.state_path, finite_json({**state, "pid": None}))
