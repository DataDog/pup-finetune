# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The detached worker and graceful cancellation, with real processes and the fake provider.

`train --detach` runs as a real ``python -m pup_finetune`` process, so the worker it starts
outlives it, as it does for a user. Every wait has a timeout.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.runs import RunDir

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
TIMEOUT_S = 60.0
SECRET = "zq9Xsynthetic0000workerKEYw7Pm"  # noqa: S105 - synthetic test value

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="--detach needs POSIX")


def _env() -> dict[str, str]:
    return {**os.environ, "PUP_FINETUNE_DEV": "1", "FIREWORKS_API_KEY": SECRET}


def cli(*args: str, timeout: float = TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - our own interpreter and module, list args
        [sys.executable, "-m", "pup_finetune", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(),
        check=False,
    )


def one_json(stdout: str) -> dict[str, Any]:
    """The only JSON document on stdout; ``json.loads`` fails on anything more."""
    data: dict[str, Any] = json.loads(stdout)
    return data


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    imported = cli("dataset", "import", str(path), "--file", str(ROWS))
    assert imported.returncode == 0, imported.stderr
    return path


@pytest.fixture
def hold(tmp_path: Path) -> Iterator[Path]:
    """The fake provider's ``hold_file``, there before the test starts its run and removed
    however the test ends, so no held worker outlives it."""
    path = tmp_path / "hold"
    path.touch()
    yield path
    path.unlink(missing_ok=True)


def train_args(dataset: Path, run_dir: Path, *extra: str) -> list[str]:
    return [
        "train",
        "--provider",
        "fake",
        "--model",
        "fake-model",
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run_dir),
        *extra,
    ]


def wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def state(run_dir: Path) -> dict[str, Any]:
    try:
        return RunDir(run_dir).load_state().to_json()
    except Exception:
        return {}


def watch(run_dir: Path) -> dict[str, Any]:
    """`status --watch` until the run stops: progress lines on stderr, one JSON on stdout."""
    watched = cli("status", str(run_dir), "--watch", "--interval", "0.2")
    assert watched.returncode == 0, watched.stderr
    final = one_json(watched.stdout)
    *progress, summary = watched.stderr.splitlines()
    # Off a terminal each progress line starts with the time of day; the closing summary,
    # the run's status line and the next command, comes last and is the only line that
    # says how the run ended. Until a resumed run's worker begins, its state is still how
    # the last attempt ended, and the line says the worker is starting.
    for line in progress:
        training = re.fullmatch(r"\d\d:\d\d:\d\d (training|created): epoch .*", line)
        starting = re.fullmatch(
            r"\d\d:\d\d:\d\d (created|interrupted|failed|cancelled)( \([a-z_]+\))?: "
            r"epoch .* \(worker pid \d+ starting\)",
            line,
        )
        assert training or starting, line
    assert summary.startswith(f"{final['phase']}: epoch "), summary
    assert summary.endswith(f". Next: {final['next_command']}"), summary
    return final


def command_line(pid: int) -> str:
    ps = subprocess.run(  # noqa: S603 - a fixed command, list args
        ["ps", "-ww", "-o", "command=", "-p", str(pid)],  # noqa: S607 - ps from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return ps.stdout.strip()


def test_a_detached_run_completes_and_status_watch_follows_it(
    dataset: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    started = cli(*train_args(dataset, run_dir, "--max-epochs", "2", "--detach", "--confirm"))
    assert started.returncode == 0, started.stderr
    data = one_json(started.stdout)
    assert data["status"] == "started"
    assert data["run_dir"] == str(run_dir)
    assert data["next_command"] == f"pup finetune status {run_dir} --watch"
    assert isinstance(data["pid"], int)

    final = watch(run_dir)
    assert final["phase"] == "completed"
    assert final["promoted_model"]["model_id"] == data["run_id"]
    assert final["worker"]["state"] == "exited"
    assert final["worker"]["log"] == str(run_dir / "worker.log")
    assert final["next_command"] == f"pup finetune evaluate {run_dir}"
    assert json.loads((run_dir / "worker.json").read_text())["pid"] == data["pid"]

    log = run_dir / "worker.log"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    text = log.read_text()
    # The log is not a terminal: every line is stamped, and each epoch line names its
    # checkpoint instead of a line of its own.
    assert all(re.match(r"\d\d:\d\d:\d\d ", line) for line in text.splitlines()), text
    assert re.search(r"epoch 2/2 done: train [\d.]+, val [\d.]+ \(new best\), checkpoint t2s", text)
    assert "saved training checkpoint" not in text
    assert f"promoted {data['run_id']}" in text
    assert "finished: the run is completed" in text
    for path in run_dir.iterdir():
        assert SECRET not in path.read_text(errors="replace"), path.name


def test_cancel_stops_a_detached_worker_gracefully_and_promotes(
    dataset: Path, tmp_path: Path, hold: Path
) -> None:
    run_dir = tmp_path / "run"
    # One step per row. The worker holds itself at its second epoch's first step while the
    # hold file exists, so it is still training when the cancel arrives, however slowly the
    # commands below start, and its first epoch's best checkpoint is there to promote.
    planned = cli(*train_args(dataset, run_dir, "--batch-size", "1"))
    assert planned.returncode == 2, planned.stderr
    second_epoch = one_json(planned.stdout)["preview"]["rows"]["train"] + 1
    held = (
        "--batch-size",
        "1",
        "--provider-option",
        f"hold_file={json.dumps(str(hold))}",
        "--provider-option",
        f"hold_from_step={second_epoch}",
    )
    # The --detach preview's next_command runs as written: it starts the worker.
    previewed = cli(*train_args(dataset, run_dir, *held, "--detach"))
    assert previewed.returncode == 2, previewed.stderr
    detach = shlex.split(one_json(previewed.stdout)["next_command"])
    assert detach[:2] == ["pup", "finetune"] and detach[-1] == "--confirm"
    assert "--detach" in detach
    started = cli(*detach[2:])
    assert started.returncode == 0, started.stderr
    pid = one_json(started.stdout)["pid"]
    wait_until(lambda: state(run_dir).get("epochs", 0) >= 1, "the first epoch")
    # Its command line carries no secret: exactly our module, the run and the mode.
    command = command_line(pid)
    assert SECRET not in command
    path = str(run_dir.resolve())
    assert command.split()[1:] == ["-P", "-m", "pup_finetune", "_worker", path, "--mode", "train"]

    preview = cli("cancel", str(run_dir))
    assert preview.returncode == 2
    cancel = one_json(preview.stdout)
    assert cancel["preview"]["pid"] == pid
    # Its next_command runs as written too.
    words = shlex.split(cancel["next_command"])
    assert words == ["pup", "finetune", "cancel", str(run_dir), "--confirm"]

    cancelled = cli(*words[2:])
    assert cancelled.returncode == 0, cancelled.stderr
    assert one_json(cancelled.stdout)["status"] == "cancelling"

    hold.unlink()
    final = watch(run_dir)
    assert final["phase"] == "cancelled"
    assert final["stopped_reason"] == "cancelled"
    assert final["promoted_model"] is not None
    # The best epoch was promoted: evaluate it; training the remaining epochs is a step.
    assert final["next_command"] == f"pup finetune evaluate {run_dir}"
    assert final["next_steps"][0]["command"] == f"pup finetune resume {run_dir}"
    assert "finished: the run is cancelled" in (run_dir / "worker.log").read_text()


def _first_line(stream: Any, pattern: str) -> str:
    """The first line ``stream`` (a process's stderr) prints that ``pattern`` matches at its
    start, skipping anything else, within :data:`TIMEOUT_S`.

    A thread reads up to that line. ``communicate`` then reads the pipe itself, not the
    stream's buffer, so lines ``readline`` read ahead past the match are lost to it.
    """
    lines: queue.Queue[str] = queue.Queue()

    def read() -> None:
        while line := stream.readline():
            lines.put(line)
            if re.match(pattern, line):
                return
        lines.put("")

    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        try:
            line = lines.get(timeout=max(deadline - time.monotonic(), 0))
        except queue.Empty:
            raise AssertionError(f"no line matching {pattern!r} in {TIMEOUT_S:g}s") from None
        if not line:
            raise AssertionError(f"the process ended before printing a line matching {pattern!r}")
        if re.match(pattern, line):
            return line


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@pytest.mark.parametrize("sigint_ignored", [False, True], ids=["default", "inherited-ignored"])
def test_ctrl_c_stops_watching_a_detached_run_but_never_its_training(
    dataset: Path, tmp_path: Path, hold: Path, sigint_ignored: bool
) -> None:
    run_dir = tmp_path / "run"
    # Every step's result is held while the hold file exists, so the run is still training
    # whenever the watch is interrupted, however slowly the processes start.
    slow = ("--batch-size", "1", "--provider-option", f"hold_file={json.dumps(str(hold))}")
    # The --detach preview's next_command runs as written: it starts the worker.
    previewed = cli(*train_args(dataset, run_dir, *slow, "--detach"))
    assert previewed.returncode == 2, previewed.stderr
    detach = shlex.split(one_json(previewed.stdout)["next_command"])
    assert detach[:2] == ["pup", "finetune"] and detach[-1] == "--confirm"
    assert "--detach" in detach
    started = cli(*detach[2:])
    assert started.returncode == 0, started.stderr
    pid = one_json(started.stdout)["pid"]
    wait_until(lambda: state(run_dir).get("phase") == "training", "training to start")
    watcher = subprocess.Popen(  # noqa: S603 - our own interpreter and module, list args
        [
            sys.executable,
            "-m",
            "pup_finetune",
            "status",
            str(run_dir),
            "--watch",
            "--interval",
            "0.2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(),
        # Started with SIGINT ignored (a background job, some agents' shells), the watch
        # still stops on Ctrl-C: it installs the default handler while it watches.
        preexec_fn=_ignore_sigint if sigint_ignored else None,
    )
    try:
        assert watcher.stderr is not None
        first = _first_line(watcher.stderr, r"\d\d:\d\d:\d\d training: epoch ")  # under way
        watcher.send_signal(signal.SIGINT)
        stdout, rest = watcher.communicate(timeout=TIMEOUT_S)
        stderr = first + rest
    finally:
        if watcher.poll() is None:
            watcher.kill()
            watcher.wait()
    assert watcher.returncode == 0, stderr
    data = one_json(stdout)
    assert (data["phase"], data["watch_timed_out"]) == ("training", False)
    assert f"stopped watching; the run keeps training (pid {pid})" in stderr
    # The worker trains on and completes once its steps are let go.
    hold.unlink()
    final = watch(run_dir)
    assert final["phase"] == "completed"
    assert "cancel" not in (run_dir / "worker.log").read_text()


def test_ctrl_c_cancels_a_foreground_run_gracefully(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    args = train_args(
        dataset, run_dir, "--batch-size", "1", "--provider-option", "step_delay_s=0.2"
    )
    process = subprocess.Popen(  # noqa: S603 - our own interpreter and module, list args
        [sys.executable, "-m", "pup_finetune", *args, "--confirm"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(),
    )
    try:
        wait_until(lambda: state(run_dir).get("epochs", 0) >= 1, "the first epoch")
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=TIMEOUT_S)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 0, stderr
    result = one_json(stdout)
    assert result["status"] == "cancelled"
    assert result["promoted_model"] is not None
    assert "cancel requested" in stderr


def test_a_second_ctrl_c_stops_at_once_with_a_resume_hint(
    dataset: Path, tmp_path: Path, hold: Path
) -> None:
    run_dir = tmp_path / "run"
    # Every step from the second on is held while the hold file exists, and a cancel waits
    # for the steps in flight: once the first step is recorded, with the second already
    # sent, the run cannot end before the hold is let go, however slowly this test runs.
    args = train_args(
        dataset,
        run_dir,
        "--batch-size",
        "1",
        "--provider-option",
        f"hold_file={json.dumps(str(hold))}",
        "--provider-option",
        "hold_from_step=2",
    )
    process = subprocess.Popen(  # noqa: S603 - our own interpreter and module, list args
        [sys.executable, "-m", "pup_finetune", *args, "--confirm"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(),
    )
    try:
        wait_until(lambda: state(run_dir).get("step", 0) >= 1, "the first step")
        # The second step is sent before the first one's result is read only with more than
        # one step in flight.
        assert RunDir(run_dir).spec().pipeline_depth > 1
        process.send_signal(signal.SIGINT)
        # Two Ctrl-Cs sent before the process next runs arrive as one: send the second only
        # once the first is handled.
        assert process.stderr is not None
        _first_line(process.stderr, "cancel requested")
        process.send_signal(signal.SIGINT)
        hold.unlink()
        stdout, stderr = process.communicate(timeout=TIMEOUT_S)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 1, stderr
    assert stdout == ""
    assert "the run in" in stderr and "is interrupted" in stderr
    assert f"pup finetune resume {run_dir}" in stderr
    assert state(run_dir)["phase"] == "interrupted"


def test_a_detached_resume_completes_a_crashed_run(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    crash = ("--batch-size", "1", "--max-epochs", "2", "--provider-option", "fail_at_step=9")
    crashed = cli(*train_args(dataset, run_dir, *crash, "--confirm"))
    assert crashed.returncode == 1

    started = cli("resume", str(run_dir), "--detach", "--confirm")
    assert started.returncode == 0, started.stderr
    assert one_json(started.stdout)["status"] == "started"
    final = watch(run_dir)
    assert final["phase"] == "completed"
    assert final["promoted_model"] is not None


def test_a_crashed_worker_s_hints_name_the_run_by_its_absolute_path(
    dataset: Path, tmp_path: Path
) -> None:
    """The worker runs in the run directory, so a relative path would read ``resume .``."""
    run_dir = tmp_path / "run"
    crash = ("--batch-size", "1", "--provider-option", "fail_at_step=2")
    started = cli(*train_args(dataset, run_dir, *crash, "--detach", "--confirm"))
    assert started.returncode == 0, started.stderr
    wait_until(lambda: state(run_dir).get("phase") == "interrupted", "the crash")
    log = run_dir / "worker.log"
    wait_until(lambda: "Next: " in log.read_text(), "the worker's hint")
    text = log.read_text()
    assert f"Next: pup finetune resume {run_dir.resolve()}" in text, text
    assert "resume ." not in text
