# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training a run in the foreground or in a detached worker, and watching or stopping it.

:func:`run_in_dir` trains, resumes or only finalizes a run in this process, the way both the
foreground commands and the worker do: SIGTERM and Ctrl-C cancel gracefully (the loop stops
at a step boundary, saves a checkpoint and promotes the best epoch) and the state's heartbeat
is refreshed every :data:`HEARTBEAT_INTERVAL_S` seconds.

:func:`spawn_worker` starts ``python -P -m pup_finetune _worker RUN_DIR --mode MODE`` in the
run directory, in a new session, with its output in ``worker.log`` (``0600``), and records
it in ``worker.json``. ``-P`` keeps the caller's working directory off ``sys.path``. The
worker inherits this process's environment and nothing else: no secret is passed through
its arguments, extra environment variables or files; it resolves the provider's credentials
again itself. When it ends, it notes that in ``worker.json`` (:func:`record_worker_exit`),
with the error it ended with, so ``status`` can tell a worker that failed before training.
Both updates of ``worker.json`` happen under a lock on ``.worker.lock`` and merge into the
record of the same pid, so a worker that ends before its launcher has recorded it keeps its
exit record.

:func:`worker_status` tells whether a run's training process is alive (its pid, and how old
the heartbeat is); :func:`cancel_training` sends SIGTERM to it after checking that the
process in ``state.json`` is alive, ours, and holds the run's lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import stat
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, get_args

from pup_finetune.errors import StateError, UsageError
from pup_finetune.providers.base import Provider
from pup_finetune.runs import Clock, RunDir, error_json, utcnow
from pup_finetune.storage import atomic_write_json, read_json
from pup_finetune.training.inputs import RunInputs
from pup_finetune.training.loop import EventSink
from pup_finetune.training.runner import resume_training, start_training

__all__ = [
    "HEARTBEAT_INTERVAL_S",
    "STALE_AFTER_S",
    "WORKER_MODES",
    "ProcessProbe",
    "WorkerMode",
    "WorkerStatus",
    "cancel_training",
    "cancellable_pid",
    "lock_held",
    "pid_alive",
    "probe_pid",
    "record_worker_exit",
    "require_detach_supported",
    "run_in_dir",
    "spawn_worker",
    "worker_status",
]

WorkerMode = Literal["train", "resume", "finalize"]
"""What a worker does: train a new run, resume one, or only retry its promotion."""
WORKER_MODES: tuple[WorkerMode, ...] = get_args(WorkerMode)

HEARTBEAT_INTERVAL_S = 15.0
"""How often a training process refreshes the run's heartbeat, at the least."""

STALE_AFTER_S = 60.0
"""A training run whose heartbeat is older than this is reported ``stale``."""

DETACH_UNSUPPORTED = "--detach is not supported on Windows yet"


def require_detach_supported() -> None:
    """Raise :class:`UsageError` where a detached worker cannot run (Windows)."""
    if sys.platform == "win32":
        raise UsageError(DETACH_UNSUPPORTED, hint="run without --detach to train in the foreground")


# --- Running -----------------------------------------------------------------------------


def run_in_dir(
    provider: Provider,
    run_dir: RunDir,
    mode: WorkerMode,
    inputs: RunInputs,
    *,
    on_event: EventSink | None = None,
    heartbeat_s: float = HEARTBEAT_INTERVAL_S,
) -> dict[str, Any]:
    """Train (``train``), resume (``resume``) or only finalize (``finalize``) ``run_dir``.

    SIGTERM and Ctrl-C cancel gracefully (see :func:`~pup_finetune.runs.execute_run`).
    ``finalize`` needs a run whose promotion is what is left to retry. Returns the run's
    ``result.json``.
    """
    if mode == "finalize" and run_dir.load_state().resume_mode != "finalize_only":
        raise UsageError(
            f"the run in {run_dir.path} has training left to do, not only its promotion",
            hint="resume it instead",
        )
    run = start_training if mode == "train" else resume_training
    return run(
        provider,
        run_dir,
        train_rows=inputs.train_rows,
        validation_rows=inputs.validation_rows,
        on_event=on_event,
        signals="cancel",
        heartbeat_s=heartbeat_s,
    )


def spawn_worker(run_dir: RunDir, mode: WorkerMode, *, now: Clock = utcnow) -> int:
    """Start a detached worker for ``run_dir`` in ``mode``; returns its pid.

    The worker runs in a new session (so closing the terminal does not stop it), in the run
    directory, reads nothing from stdin, and writes its output to ``worker.log``. Its pid,
    mode and start time go to ``worker.json``. Raises :class:`StateError` when ``worker.log``
    is not a regular file of this user's (e.g. a symlink).
    """
    require_detach_supported()
    if mode not in WORKER_MODES:
        raise ValueError(f"unknown worker mode {mode!r}")
    path = run_dir.path.resolve()
    log_fd = _open_log(run_dir)
    try:
        process = subprocess.Popen(  # noqa: S603 - our own interpreter and module, no shell
            [sys.executable, "-P", "-m", "pup_finetune", "_worker", str(path), "--mode", mode],
            cwd=path,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=log_fd,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_fd)
    started_at = now().isoformat(timespec="seconds")
    launch = {"pid": process.pid, "mode": mode, "started_at": started_at}
    with _worker_record_lock(run_dir):
        recorded = _launched(run_dir) or {}
        if recorded.get("pid") == process.pid:  # the worker already ended: keep its exit
            launch = {**recorded, **launch}
        atomic_write_json(run_dir.worker_path, launch)
    return process.pid


WORKER_LOCK = ".worker.lock"
"""The lock file ``worker.json`` updates are made under, in the run directory."""


@contextlib.contextmanager
def _worker_record_lock(run_dir: RunDir) -> Iterator[None]:
    """Hold the (blocking) lock that serializes updates of ``worker.json``.

    Not the run's lock, which the worker holds while it trains. The lock file is never
    opened through a symlink, and must be a regular file of this user's.
    """
    path = run_dir.path / WORKER_LOCK
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StateError(f"cannot lock {path}: {exc.strerror or exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise StateError(f"{path} is not a regular file of yours; not locking it")
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _open_log(run_dir: RunDir) -> int:
    """``worker.log`` opened for appending, ``0600``; never through a symlink."""
    log = run_dir.worker_log_path
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(log, flags, 0o600)
    except OSError as exc:
        raise StateError(f"cannot open {log}: {exc.strerror or exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise StateError(f"{log} is not a regular file of yours; not writing to it")
        os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        raise
    return fd


def record_worker_exit(
    run_dir: RunDir, mode: WorkerMode, error: BaseException | None, *, now: Clock = utcnow
) -> None:
    """Note in ``worker.json`` that this worker ended, and the error it ended with (redacted).

    Writes nothing outside a run directory. Never raises: it runs as the worker exits.
    """
    with contextlib.suppress(Exception):
        if not run_dir.run_path.is_file():
            return
        with _worker_record_lock(run_dir):
            launched = _launched(run_dir) or {}
            if launched.get("pid") != os.getpid():  # run by hand, or not yet recorded
                launched = {"pid": os.getpid(), "mode": mode, "started_at": None}
            atomic_write_json(
                run_dir.worker_path,
                {
                    **launched,
                    "ended_at": now().isoformat(),
                    "exit_error": None if error is None else error_json(error),
                },
            )


# --- Watching ----------------------------------------------------------------------------


ProcessProbe = Literal["ours", "foreign", "gone"]
"""A process that exists and that we may signal, one of another user's, or none."""


def probe_pid(pid: int | None) -> ProcessProbe:
    """Whether process ``pid`` exists and is ours to signal. An exited child is reaped."""
    if pid is None or pid <= 0:
        return "gone"
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # not our child: ask the kernel below
    else:
        if reaped == pid:
            return "gone"  # our child had exited (a zombie until now)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "foreign"  # it exists, but belongs to another user
    return "ours"


def pid_alive(pid: int | None) -> bool:
    """Whether a process ``pid`` exists (ours or not). An exited child of ours is reaped."""
    return probe_pid(pid) != "gone"


def lock_held(run_dir: RunDir) -> bool:
    """Whether a process holds ``run_dir``'s lock, i.e. is training in it right now.

    Probes by taking the lock for an instant, so call it only when no process is about to
    take it (e.g. while the state says the run is training).
    """
    return run_dir.lock().is_held()


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """The process training a run, as far as the run directory and the OS tell.

    ``pid`` is the training process from ``state.json`` while the run is training, else the
    last worker started for it. ``stale`` means the run is training but its heartbeat is
    older than :data:`STALE_AFTER_S`, or a worker has been starting for longer than that.
    ``starting`` means a worker was started after the state last changed, less than
    :data:`STALE_AFTER_S` ago, and is still alive, so it has not begun (or resumed) training
    yet. ``running`` is whether training is under way or about to be: watch it until it is
    not. ``log`` is the last worker's ``worker.log`` (``None`` when none was started).
    ``problem`` says what went wrong with that worker since the state last changed, if
    anything (it failed, or exited or hangs before training), and ``error`` is the error it
    recorded when it ended.
    """

    pid: int | None
    alive: bool
    heartbeat_age_s: float | None
    stale: bool
    starting: bool
    running: bool
    log: str | None = None
    error: dict[str, Any] | None = None
    problem: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "alive": self.alive,
            "running": self.running,
            "starting": self.starting,
            "heartbeat_age_s": self.heartbeat_age_s,
            "stale": self.stale,
            "log": self.log,
            "error": self.error,
        }


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _launched(run_dir: RunDir) -> dict[str, Any] | None:
    """``worker.json``, or ``None`` when there is none or it cannot be read."""
    try:
        data = read_json(run_dir.worker_path)
    except (FileNotFoundError, StateError):
        return None
    return data if isinstance(data, dict) else None


@dataclass(frozen=True, slots=True)
class _Launch:
    """The last worker started for a run (``worker.json``), against the run's heartbeat."""

    pid: int | None
    alive: bool
    started: datetime | None
    ended: datetime | None
    error: dict[str, Any] | None
    log: str

    @classmethod
    def read(cls, run_dir: RunDir) -> _Launch | None:
        data = _launched(run_dir)
        if data is None:
            return None
        pid = data.get("pid")
        pid = pid if isinstance(pid, int) else None
        error = data.get("exit_error")
        return cls(
            pid=pid,
            alive=pid_alive(pid),
            started=_parse_time(data.get("started_at")),
            ended=_parse_time(data.get("ended_at")),
            error=error if isinstance(error, dict) else None,
            log=str(run_dir.worker_log_path),
        )

    def started_after(self, heartbeat: datetime | None) -> bool:
        """Whether it started after the state last changed (so it has not changed it)."""
        return self.started is not None and (heartbeat is None or self.started >= heartbeat)

    def waiting_s(self, heartbeat: datetime | None, now: datetime) -> float | None:
        """How long it has been alive without starting to train, if it has not yet."""
        if not self.alive or self.ended is not None or self.started is None:
            return None
        if heartbeat is not None and self.started < heartbeat:
            return None
        return (now - self.started).total_seconds()

    def problem(self, heartbeat: datetime | None, now: datetime) -> str | None:
        """What went wrong with it since the state last changed, if anything."""
        who = f"the worker (process {self.pid})"
        where = f"its output is in {self.log}"
        if self.alive:
            waiting = self.waiting_s(heartbeat, now)
            if waiting is not None and waiting > STALE_AFTER_S:
                return f"{who} started {round(waiting)}s ago and has not begun training; {where}"
            return None
        if self.ended is None:
            if self.started_after(heartbeat):
                return f"{who} exited before training, without recording why; {where}"
            return None
        if self.error is not None and (heartbeat is None or self.ended >= heartbeat):
            why = self.error.get("message") or self.error.get("type")
            return f"{who} failed: {why}; {where}"
        return None


def worker_status(
    run_dir: RunDir,
    *,
    phase: str,
    pid: int | None,
    heartbeat_at: str | None,
    now: Clock = utcnow,
) -> WorkerStatus:
    """The :class:`WorkerStatus` of the run in ``run_dir``, from its state's ``phase``,
    training ``pid`` and ``heartbeat_at``."""
    current = now()
    heartbeat = _parse_time(heartbeat_at)
    age = None if heartbeat is None else max(0.0, round((current - heartbeat).total_seconds(), 1))
    launch = _Launch.read(run_dir)
    problem = None if launch is None else launch.problem(heartbeat, current)
    about_launch: dict[str, Any] = {
        "log": None if launch is None else launch.log,
        "error": None if launch is None or problem is None else launch.error,
        "problem": problem,
    }
    if phase == "training":
        alive = pid_alive(pid)
        stale = age is None or age > STALE_AFTER_S
        running = alive and not stale
        return WorkerStatus(pid, alive, age, stale, False, running, **about_launch)
    if launch is None or launch.pid is None:
        return WorkerStatus(pid, pid_alive(pid), age, False, False, False, **about_launch)
    waiting = launch.waiting_s(heartbeat, current)
    starting = waiting is not None and waiting <= STALE_AFTER_S
    hanging = waiting is not None and not starting
    return WorkerStatus(launch.pid, launch.alive, age, hanging, starting, starting, **about_launch)


# --- Stopping ----------------------------------------------------------------------------


def cancellable_pid(run_dir: RunDir) -> int:
    """The pid of the live process of ours training ``run_dir``, checked; else raise.

    Raises :class:`UsageError` when the run is not training, and :class:`StateError` when
    the process recorded in ``state.json`` is gone, belongs to another user, or is not the
    one holding the run's lock (so it is not the one training it).
    """
    state = run_dir.load_state()
    resume = f"pup finetune resume {run_dir.path}"
    if state.phase != "training":
        why = (
            "it has not started training yet; retry once `status` shows it training"
            if state.phase == "created"
            else "there is nothing to cancel"
        )
        raise UsageError(f"the run in {run_dir.path} is {state.phase}, not training", hint=why)
    pid = state.pid
    probe = probe_pid(pid)
    if pid is None or probe == "gone":
        raise StateError(
            f"the process training the run in {run_dir.path} ({pid}) is no longer running",
            hint=f"it stopped without recording why; resume the run with `{resume}`",
        )
    if probe == "foreign":
        raise StateError(f"process {pid} belongs to another user; not signalling it")
    lock = run_dir.lock()
    if not lock.is_held():
        raise StateError(
            f"process {pid} is not training the run in {run_dir.path} (the run is not locked)",
            hint=f"the training process is gone; resume the run with `{resume}`",
        )
    owner = lock.owner_pid()
    if owner != pid:
        raise StateError(
            f"process {pid} is not training the run in {run_dir.path} (process {owner} holds "
            "its lock); not signalling it",
            hint=f"check the run with `pup finetune status {run_dir.path}`",
        )
    return pid


def cancel_training(run_dir: RunDir) -> int:
    """Ask the process training ``run_dir`` to cancel gracefully (SIGTERM); returns its pid.

    It finishes the current step, saves a training checkpoint, promotes the best epoch and
    ends the run ``cancelled``. Checks the process first (see :func:`cancellable_pid`).
    """
    pid = cancellable_pid(run_dir)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        raise StateError(f"process {pid} exited before it could be cancelled") from None
    return pid
