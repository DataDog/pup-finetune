# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Process liveness, the run lock probe, worker status and the cancel checks.

Spawning and cancelling real workers is covered end to end in
``tests/cli/test_train_detach.py``.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest

from pup_finetune.errors import StateError, UsageError
from pup_finetune.providers.base import TrainSpec
from pup_finetune.providers.fake import FAKE_PROFILE, FakeProvider, FakeRenderer
from pup_finetune.runs import RunDir, RunState
from pup_finetune.runtime import Runtime
from pup_finetune.storage import atomic_write_json, read_json
from pup_finetune.training.datums import build_examples
from pup_finetune.training.inputs import RunInputs
from pup_finetune.training.worker import (
    STALE_AFTER_S,
    cancellable_pid,
    lock_held,
    pid_alive,
    record_worker_exit,
    run_in_dir,
    spawn_worker,
    worker_status,
)
from tests.fakes.training import chat_rows

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def at(seconds: float) -> str:
    return (NOW + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def new_run(path: Path, **spec: Any) -> RunDir:
    return RunDir.create(
        path,
        provider="fake",
        profile=FAKE_PROFILE,
        spec=TrainSpec(**spec),
        dataset_dir=path.parent / "dataset",
        dataset_sha256="d" * 64,
        rendering=FakeRenderer().identity(),
        run_id="ft-worker-test",
    )


def exited_child(*, reap: bool) -> int:
    """The pid of a child of ours that has exited: reaped, or left a zombie."""
    process = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE)
    if reap:
        process.communicate()
        return process.pid
    assert process.stdout is not None
    process.stdout.read()  # EOF once it has exited; nothing reaps it
    process.stdout.close()
    deadline = time.monotonic() + 10
    while _state_of(process.pid) not in ("Z", ""):
        assert time.monotonic() < deadline, "the child did not exit"
        time.sleep(0.01)
    return process.pid


def _state_of(pid: int) -> str:
    ps = subprocess.run(  # noqa: S603 - a fixed command, list args
        ["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S607 - ps from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return ps.stdout.strip()[:1]


def test_pid_alive() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(None)
    assert not pid_alive(0)
    assert not pid_alive(-1)
    assert not pid_alive(exited_child(reap=True))


def test_an_exited_child_of_ours_is_not_alive() -> None:
    assert not pid_alive(exited_child(reap=False))


def test_lock_held(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    assert not lock_held(run_dir)
    with run_dir.lock():
        assert lock_held(run_dir)
    assert not lock_held(run_dir)


def test_lock_held_reports_other_lock_failures(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="cannot lock"):
        lock_held(RunDir(tmp_path / "missing"))


def test_a_training_run_is_running_while_its_heartbeat_is_fresh(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fresh = worker_status(
        run_dir, phase="training", pid=os.getpid(), heartbeat_at=at(0), now=lambda: NOW
    )
    assert (fresh.alive, fresh.stale, fresh.running, fresh.heartbeat_age_s) == (
        True,
        False,
        True,
        0.0,
    )
    later = NOW + timedelta(seconds=STALE_AFTER_S + 1)
    stale = worker_status(
        run_dir, phase="training", pid=os.getpid(), heartbeat_at=at(0), now=lambda: later
    )
    assert (stale.stale, stale.running) == (True, False)
    gone = worker_status(
        run_dir,
        phase="training",
        pid=exited_child(reap=True),
        heartbeat_at=at(0),
        now=lambda: NOW,
    )
    assert (gone.alive, gone.running) == (False, False)


def test_a_worker_started_after_the_last_state_change_is_starting(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    atomic_write_json(
        run_dir.worker_path, {"pid": os.getpid(), "mode": "train", "started_at": at(1)}
    )
    starting = worker_status(
        run_dir, phase="created", pid=None, heartbeat_at=at(0), now=lambda: NOW
    )
    assert (starting.pid, starting.starting, starting.running) == (os.getpid(), True, True)
    # Once the state changed after the worker started, it is no longer starting.
    done = worker_status(
        run_dir, phase="completed", pid=os.getpid(), heartbeat_at=at(5), now=lambda: NOW
    )
    assert (done.starting, done.running, done.stale) == (False, False, False)


def test_a_dead_worker_is_not_starting(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    pid = exited_child(reap=True)
    atomic_write_json(run_dir.worker_path, {"pid": pid, "mode": "train", "started_at": at(1)})
    status = worker_status(run_dir, phase="created", pid=None, heartbeat_at=at(0), now=lambda: NOW)
    assert (status.pid, status.alive, status.running) == (pid, False, False)


def test_an_unreadable_worker_file_is_ignored(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    run_dir.worker_path.write_text("{not json")
    status = worker_status(run_dir, phase="created", pid=None, heartbeat_at=at(0))
    assert (status.pid, status.running) == (None, False)


def test_only_a_training_run_can_be_cancelled(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    with pytest.raises(UsageError, match="is created, not training"):
        cancellable_pid(run_dir)


def test_cancel_needs_the_training_process_to_hold_the_lock(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    state = run_dir.load_state()
    state.start()  # this process, which is alive, but does not hold the lock
    with pytest.raises(StateError, match="not locked"):
        cancellable_pid(run_dir)
    with run_dir.lock():
        # Held (here by this process, as the training process would): the pid is accepted.
        assert cancellable_pid(run_dir) == os.getpid()


def test_cancel_refuses_a_pid_that_does_not_own_the_lock(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    state = run_dir.load_state()
    state.start()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        state.pid = sleeper.pid  # alive and ours, but this process holds the lock
        state.save()
        with run_dir.lock(), pytest.raises(StateError, match=f"process {os.getpid()} holds"):
            cancellable_pid(run_dir)
    finally:
        sleeper.kill()
        sleeper.wait()


# --- spawning ----------------------------------------------------------------------------


class FakePopen:
    """Records how a worker would be started, and starts nothing."""

    calls: ClassVar[list[tuple[list[str], dict[str, Any]]]] = []

    def __init__(self, args: list[str], **kwargs: Any) -> None:
        FakePopen.calls.append((args, kwargs))
        self.pid = 4242


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[FakePopen]]:
    FakePopen.calls = []
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    yield FakePopen


def test_spawn_worker_runs_our_module_safely(tmp_path: Path, fake_popen: type[FakePopen]) -> None:
    run_dir = new_run(tmp_path / "run")
    assert spawn_worker(run_dir, "resume", now=lambda: NOW) == 4242
    [(args, kwargs)] = fake_popen.calls
    path = run_dir.path.resolve()
    # -P: the caller's working directory is not on sys.path, so nothing there is imported.
    assert args == [
        sys.executable,
        "-P",
        "-m",
        "pup_finetune",
        "_worker",
        str(path),
        "--mode",
        "resume",
    ]
    assert kwargs["cwd"] == path
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert "env" not in kwargs  # this process's environment, nothing added
    assert stat.S_IMODE(run_dir.worker_log_path.stat().st_mode) == 0o600
    launched = run_dir.worker_path.read_text()
    assert '"pid": 4242' in launched
    assert '"mode": "resume"' in launched


def test_spawn_worker_refuses_a_symlinked_log(tmp_path: Path, fake_popen: type[FakePopen]) -> None:
    run_dir = new_run(tmp_path / "run")
    target = tmp_path / "elsewhere"
    target.write_text("keep me")
    target.chmod(0o644)
    run_dir.worker_log_path.symlink_to(target)
    with pytest.raises(StateError, match=r"worker\.log"):
        spawn_worker(run_dir, "train")
    assert fake_popen.calls == []
    assert target.read_text() == "keep me"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_record_worker_exit_notes_the_error(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    atomic_write_json(
        run_dir.worker_path, {"pid": os.getpid(), "mode": "train", "started_at": at(0)}
    )
    record_worker_exit(run_dir, "train", StateError("no credentials"), now=lambda: NOW)
    launched = read_json(run_dir.worker_path)
    assert launched["started_at"] == at(0)
    assert launched["ended_at"] == NOW.isoformat()
    assert launched["exit_error"]["message"] == "no credentials"


def test_record_worker_exit_writes_nothing_outside_a_run(tmp_path: Path) -> None:
    record_worker_exit(RunDir(tmp_path), "train", None)
    assert list(tmp_path.iterdir()) == []


def test_a_worker_that_failed_after_the_state_changed_is_reported(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    dead = exited_child(reap=True)
    error = {"type": "StateError", "message": "no credentials", "notes": []}
    launched = {"pid": dead, "mode": "train", "started_at": at(1), "ended_at": at(2)}
    atomic_write_json(run_dir.worker_path, {**launched, "exit_error": error})
    failed = worker_status(run_dir, phase="created", pid=None, heartbeat_at=at(0), now=lambda: NOW)
    assert failed.error == error
    assert failed.problem is not None and "failed: no credentials" in failed.problem
    # A later run (here: the heartbeat moved on) makes that failure old news.
    later = worker_status(
        run_dir, phase="completed", pid=None, heartbeat_at=at(10), now=lambda: NOW
    )
    assert (later.error, later.problem) == (None, None)


# --- running -----------------------------------------------------------------------------


def _inputs() -> RunInputs:
    def build(count: int, prefix: str) -> Any:
        return build_examples(chat_rows(count, prefix=prefix), FakeRenderer(), max_seq_len=4096)

    return RunInputs(
        rows={"train": 2, "validation": 1, "test": 0},
        train=build(2, "row"),
        validation=build(1, "val"),
    )


def test_run_in_dir_keeps_the_heartbeat_fresh_through_a_slow_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = new_run(
        tmp_path / "run", max_epochs=1, batch_size=2, provider_options={"step_delay_s": "0.4"}
    )
    beats: list[str] = []
    save = RunState.save

    def counting_save(self: RunState) -> None:
        if threading.current_thread().name == "run-heartbeat":
            beats.append(self.phase)
        save(self)

    monkeypatch.setattr(RunState, "save", counting_save)
    provider = FakeProvider(Runtime(dev_mode=True))
    result = run_in_dir(provider, run_dir, "train", _inputs(), heartbeat_s=0.05)
    assert result["phase"] == "completed"
    assert len(beats) >= 2  # saved from the heartbeat thread during the one slow step
    assert set(beats) == {"training"}


def test_run_in_dir_refuses_finalize_for_a_run_with_training_left(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    state = run_dir.load_state()
    state.start()
    state.transition("interrupted")
    provider = FakeProvider(Runtime(dev_mode=True))
    with pytest.raises(UsageError, match="has training left to do"):
        run_in_dir(provider, run_dir, "finalize", _inputs())
    assert run_dir.load_state().phase == "interrupted"
