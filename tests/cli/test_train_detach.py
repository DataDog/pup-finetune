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
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable
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
    lines = watched.stderr.splitlines()
    assert lines, "no progress lines on stderr"
    assert lines[-1].startswith(f"{final['phase']}: epoch "), lines[-1]
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
    assert final["worker"]["running"] is False
    assert final["next_command"] == f"pup finetune evaluate {run_dir}"
    assert json.loads((run_dir / "worker.json").read_text())["pid"] == data["pid"]

    log = run_dir / "worker.log"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    text = log.read_text()
    assert "epoch 2 done" in text
    assert "finished: the run is completed" in text
    for path in run_dir.iterdir():
        assert SECRET not in path.read_text(errors="replace"), path.name


def test_cancel_stops_a_slow_worker_gracefully_and_promotes(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    slow = ("--batch-size", "1", "--provider-option", "step_delay_s=0.2")
    started = cli(*train_args(dataset, run_dir, *slow, "--detach", "--confirm"))
    assert started.returncode == 0, started.stderr
    pid = one_json(started.stdout)["pid"]
    # Wait until the first epoch's best checkpoint exists, so there is something to promote.
    wait_until(lambda: state(run_dir).get("epochs", 0) >= 1, "the first epoch")
    # Its command line carries no secret: exactly our module, the run and the mode.
    command = command_line(pid)
    assert SECRET not in command
    path = str(run_dir.resolve())
    assert command.split()[1:] == ["-P", "-m", "pup_finetune", "_worker", path, "--mode", "train"]

    preview = cli("cancel", str(run_dir))
    assert preview.returncode == 2
    assert one_json(preview.stdout)["preview"]["pid"] == pid

    cancelled = cli("cancel", str(run_dir), "--confirm")
    assert cancelled.returncode == 0, cancelled.stderr
    assert one_json(cancelled.stdout)["status"] == "cancelling"

    final = watch(run_dir)
    assert final["phase"] == "cancelled"
    assert final["stopped_reason"] == "cancelled"
    assert final["promoted_model"] is not None
    assert final["next_command"] == f"pup finetune resume {run_dir}"
    assert "finished: the run is cancelled" in (run_dir / "worker.log").read_text()


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
    assert result["phase"] == "cancelled"
    assert result["promoted_model"] is not None
    assert "cancel requested" in stderr


def test_a_second_ctrl_c_stops_at_once_with_a_resume_hint(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    args = train_args(dataset, run_dir, "--batch-size", "1", "--provider-option", "step_delay_s=2")
    process = subprocess.Popen(  # noqa: S603 - our own interpreter and module, list args
        [sys.executable, "-m", "pup_finetune", *args, "--confirm"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(),
    )
    try:
        wait_until(lambda: state(run_dir).get("phase") == "training", "training to start")
        process.send_signal(signal.SIGINT)
        time.sleep(0.3)  # well within the 2s step it waits for
        process.send_signal(signal.SIGINT)
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
