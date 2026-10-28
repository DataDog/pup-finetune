# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`train`, `resume`, `status` and `cancel` in the foreground, through the Typer app.

Every test runs in development mode, so the offline fake provider is available. The
detached worker is covered by ``test_train_detach.py``.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import train as train_module
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.runs import RunDir
from pup_finetune.storage import atomic_write_json
from pup_finetune.training import worker as worker_module

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    """The one JSON document on stdout (``json.loads`` fails on anything more)."""
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def import_dataset(path: Path, rows: Path = ROWS) -> Path:
    result = invoke("dataset", "import", str(path), "--file", str(rows))
    assert result.exit_code == 0, result.stderr
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return import_dataset(tmp_path / "ds")


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


def rerun(next_command: str) -> Result:
    """Run a ``pup finetune ...`` next_command through the app."""
    words = shlex.split(next_command)
    assert words[:2] == ["pup", "finetune"]
    return invoke(*words[2:])


def kept_rows_per_split(dataset: Path) -> dict[str, int]:
    return {split: len(rows) for split, rows in DatasetDir(dataset).rows_by_split().items()}


# --- train: the preview ------------------------------------------------------------------


def test_the_preview_exits_2_and_pins_the_rerun(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    extra = ("--max-epochs", "3", "--learning-rate", "0.0001")
    options = ("--provider-option", "step_delay_s=0", "--provider-option", "val_curve=[2.0,1.5]")
    result = invoke(*train_args(dataset, run_dir, *extra, *options, "--detach"))
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["status"] == "confirmation_required"
    preview = data["preview"]
    run_id = preview["run_id"]
    assert preview["run_dir"] == str(run_dir)
    assert not run_dir.exists()  # a preview writes nothing
    words = shlex.split(data["next_command"])
    assert words[-1] == "--confirm"
    assert words[words.index("--run-id") + 1] == run_id
    assert words[words.index("--run-dir") + 1] == str(run_dir)
    assert words[words.index("--max-epochs") + 1] == "3"
    assert words[words.index("--learning-rate") + 1] == "0.0001"
    given = [words[i + 1] for i, word in enumerate(words) if word == "--provider-option"]
    assert given == ["step_delay_s=0", "val_curve=[2.0,1.5]"]
    assert "--detach" in words

    # Replayed in the foreground, it trains exactly what was previewed.
    words.remove("--detach")
    confirmed = rerun(shlex.join(words))
    assert confirmed.exit_code == 0, confirmed.stderr
    assert output(confirmed)["run_id"] == run_id
    assert RunDir(run_dir).run_id == run_id
    recorded = RunDir(run_dir).manifest()["settings"]
    # run.json records the settings the preview showed, and what no flag changes.
    assert {k: v for k, v in recorded.items() if k not in ("pipeline_depth", "optimizer")} == (
        preview["settings"]
    )
    assert recorded["optimizer"]["beta2"] == 0.95


def test_the_preview_counts_rows_examples_and_tokens(dataset: Path, tmp_path: Path) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--max-epochs", "4"))
    preview = output(result)["preview"]
    counts = kept_rows_per_split(dataset)
    assert preview["provider"] == "fake"
    assert preview["model"] == "fake-model"
    assert preview["rows"] == {**counts, "rejected": 0}
    assert preview["examples"] == {"train": counts["train"], "validation": counts["validation"]}
    assert preview["rejected_by_code"] == {}
    assert preview["tokens_per_epoch"] > 0
    assert preview["max_tokens_total"] == preview["tokens_per_epoch"] * 4
    assert preview["settings"]["max_epochs"] == 4
    assert preview["data_destination"] == "training data is sent to fake"
    assert preview["data_recipients"] == ["training rows -> fake"]


def test_the_spend_flags_reach_the_spec_and_the_rerun(dataset: Path, tmp_path: Path) -> None:
    spend = ("--max-spend-usd", "5", "--hourly-rate-usd", "2.5")
    result = invoke(
        *train_args(dataset, tmp_path / "run", *spend, "--spend-reserve-fraction", "0.25")
    )
    data = output(result)
    spec = data["preview"]["settings"]
    assert (spec["max_spend_usd"], spec["hourly_rate_usd"]) == (5.0, 2.5)
    assert spec["spend_reserve_fraction"] == 0.25
    words = shlex.split(data["next_command"])
    assert words[words.index("--spend-reserve-fraction") + 1] == "0.25"
    default = output(invoke(*train_args(dataset, tmp_path / "other", *spend)))["preview"]
    assert default["settings"]["spend_reserve_fraction"] == 0.1


def test_the_preview_reports_rows_over_the_context(tmp_path: Path) -> None:
    lines = ROWS.read_text().splitlines()
    long_rows = []
    for index, line in enumerate(lines):
        row = json.loads(line)
        row["id"] = f"long-{index}"
        if "source" in row:
            row["source"] = {**row["source"], "trace_id": f"long-trace-{index}"}
        row["messages"][-1]["content"] = " ".join(["word"] * 5000)
        long_rows.append(json.dumps(row))
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join([*lines, *long_rows]) + "\n")
    dataset = import_dataset(tmp_path / "ds", rows)
    by_split = DatasetDir(dataset).rows_by_split()
    expected = sum(
        row.id.startswith("long-") for split in ("train", "validation") for row in by_split[split]
    )
    assert expected > 0

    preview = output(invoke(*train_args(dataset, tmp_path / "run")))["preview"]
    assert preview["rows"]["rejected"] == expected
    assert preview["rejected_by_code"] == {"exceeds_context": expected}
    assert {row["code"] for row in preview["rejected_rows"]} == {"exceeds_context"}


def test_the_preview_warns_without_validation_rows(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    imported = invoke(
        "dataset", "import", str(ds), "--file", str(ROWS), "--validation-fraction", "0"
    )
    assert imported.exit_code == 0, imported.stderr
    preview = output(invoke(*train_args(ds, tmp_path / "run")))["preview"]
    assert preview["rows"]["validation"] == 0
    assert any("no rows to validate on" in warning for warning in preview["warnings"])


def test_a_dataset_whose_train_rows_all_exceed_the_context_is_refused(tmp_path: Path) -> None:
    long_rows = []
    for line in ROWS.read_text().splitlines():
        row = json.loads(line)
        row["messages"][-1]["content"] = " ".join(["word"] * 5000)
        long_rows.append(json.dumps(row))
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(long_rows) + "\n")
    dataset = import_dataset(tmp_path / "ds", rows)
    result = invoke(*train_args(dataset, tmp_path / "run"))
    assert result.exit_code == 2
    assert "no rows to train on" in result.stderr
    assert "rejected as exceeds_context" in result.stderr
    assert result.stdout == ""


def test_read_only_allows_training_on_the_offline_fake_provider(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing leaves the machine, so PUP_READ_ONLY has nothing to refuse (a real provider's
    # --confirm is refused: tests/cli/test_train_fireworks.py).
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    run_dir = tmp_path / "run"
    preview = invoke(*train_args(dataset, run_dir))
    assert preview.exit_code == 2
    data = output(preview)
    assert data["status"] == "confirmation_required"
    assert "read_only" not in data["preview"]
    confirmed = rerun(data["next_command"])
    assert confirmed.exit_code == 0, confirmed.stderr


def test_the_fake_provider_needs_dev_mode(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PUP_FINETUNE_DEV")
    result = invoke(*train_args(dataset, tmp_path / "run"))
    assert result.exit_code == 2
    assert "PUP_FINETUNE_DEV=1" in result.stderr


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--max-epochs", "0"), "--max-epochs"),
        (("--max-spend-usd", "5"), "--hourly-rate-usd"),
        (("--spend-reserve-fraction", "1"), "--spend-reserve-fraction"),
        (
            ("--lr-schedule", "step"),
            "Error [usage]: invalid training settings: --lr-schedule: Input should be "
            "'constant', 'linear' or 'cosine'",
        ),
        (("--warmup-fraction", "1"), "--warmup-fraction: Input should be less than 1"),
        (("--provider-option", "no-equals-sign"), "key=value"),
        (("--provider-option", "nope=1"), "unknown fake provider option"),
        (("--run-id", "Not Valid"), "invalid run id"),
    ],
)
def test_invalid_settings_are_usage_errors(
    dataset: Path, tmp_path: Path, extra: tuple[str, ...], message: str
) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", *extra))
    assert result.exit_code == 2
    assert message in result.stderr


def test_the_run_dir_must_be_new_or_empty(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "something").write_text("x")
    result = invoke(*train_args(dataset, run_dir))
    assert result.exit_code == 2
    assert f"Error [run_dir_not_empty]: {run_dir} is not empty" in result.stderr


def test_detach_is_refused_on_windows(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_sys = SimpleNamespace(platform="win32", executable=sys.executable)
    monkeypatch.setattr(worker_module, "sys", fake_sys)
    result = invoke(*train_args(dataset, tmp_path / "run", "--detach"))
    assert result.exit_code == 2
    assert "not supported on Windows yet" in result.stderr


# --- train: in the foreground -------------------------------------------------------------


def test_confirm_trains_in_the_foreground(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = invoke(*train_args(dataset, run_dir, "--max-epochs", "2", "--confirm"))
    assert result.exit_code == 0, result.stderr
    data = output(result)  # exactly one JSON document
    assert data["status"] == "completed"
    assert data["run_dir"] == str(run_dir)
    assert (data["model"], data["provider"]) == ("fake-model", "fake")
    assert data["promoted_model"]["model_id"] == data["run_id"]
    assert data["next_command"] == f"pup finetune evaluate {run_dir}"
    assert data["files"] == {
        "result": str(run_dir / "result.json"),
        "epochs": str(run_dir / "epochs.json"),
    }
    # The checkpoints stay in result.json; stdout names the best epoch's and the losses.
    assert "checkpoints" not in data
    assert data["epochs"] == [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 2.1},
        {"epoch": 2, "train_loss": 1.4, "val_loss": 1.6, "best": True},
    ]
    assert isinstance(data["elapsed_seconds"], int)
    on_disk = json.loads((run_dir / "result.json").read_text())
    assert set(on_disk["checkpoints"]) == {"best_sampler", "best_training", "last_training"}
    assert data["deployable_checkpoint"] == on_disk["deployable_checkpoint"]["name"]
    assert on_disk["promoted_model"]["model_id"] == data["promoted_model"]["model_id"]
    assert on_disk["promoted_model"]["uri"] == data["promoted_model"]["uri"]
    # Off a terminal (as here), the steps of these one-step epochs get no lines of their own,
    # each epoch line names its checkpoint, and the outcome closes stderr in one line.
    lines = result.stderr.splitlines()
    assert "18:03:16 epoch 1/2 done: train 2.000, val 2.100 (new best), checkpoint t1s1-" in (
        result.stderr
    )
    assert "18:03:16 promoting epoch 2 (val 1.600) ..." in lines
    assert f"18:03:16 promoted {data['run_id']}" in lines
    assert not [line for line in lines if " step " in line or "saved " in line]
    assert lines[-1] == (
        f"run {data['run_id']} completed in {train_module.format_duration(data['elapsed_seconds'])}"
        ": 2 epochs, best epoch 2 (validation loss 2.100 -> 1.600); promoted model: "
        f"{data['run_id']} (fake). Next: pup finetune evaluate {run_dir}"
    )


# --- status ------------------------------------------------------------------------------


STATUS_FIELDS = {
    "run_id",
    "phase",
    "epoch",
    "max_epochs",
    "step",
    "max_steps",
    "elapsed_seconds",
    "approx_eta_seconds",
    "last_train_loss",
    "val_losses",
    "best_epoch",
    "best_val_loss",
    "promoted_model",
    "deployable_checkpoint",
    "error",
    "finalize_error",
    "warnings",
    "next_command",
    "next_steps",
    "files",
}


@pytest.fixture
def trained(dataset: Path, tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    result = invoke(*train_args(dataset, run_dir, "--max-epochs", "2", "--confirm"))
    assert result.exit_code == 0, result.stderr
    return run_dir


def test_status_reports_a_completed_run(trained: Path) -> None:
    result = invoke("status", str(trained))
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert set(data) >= STATUS_FIELDS
    assert data["phase"] == "completed"
    assert (data["epoch"], data["step"]) == (2, 2)
    assert data["last_train_loss"] == pytest.approx(1.4)
    assert data["val_losses"] == pytest.approx([2.1, 1.6])
    assert data["best_epoch"] == 2
    assert data["promoted_model"]["model_id"] == data["run_id"]
    on_disk = json.loads((trained / "result.json").read_text())
    assert on_disk["deployable_checkpoint"]["kind"] == "sampler"
    assert data["deployable_checkpoint"] == on_disk["deployable_checkpoint"]["name"]
    assert data["error"] is None
    assert data["finalize_error"] is None
    # A completed run is not resumable, and it trained in this process, which has no worker.
    assert "resume_mode" not in data
    assert "worker" not in data
    assert data["files"]["result"] == str(trained / "result.json")
    assert data["next_command"] == f"pup finetune evaluate {trained}"


def test_status_watch_on_a_finished_run_returns_at_once(trained: Path) -> None:
    result = invoke("status", str(trained), "--watch", "--interval", "0.1")
    assert result.exit_code == 0, result.stderr
    assert output(result)["phase"] == "completed"
    assert "completed: epoch 2/2" in result.stderr


def _set_state(run_dir: Path, **fields: Any) -> None:
    path = run_dir / "state.json"
    state = json.loads(path.read_text())
    atomic_write_json(path, {**state, **fields})


def _ago(seconds: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_a_training_run_with_an_old_heartbeat_is_stale(trained: Path) -> None:
    _set_state(trained, phase="training", pid=os.getpid(), heartbeat_at=_ago(300))
    data = output(invoke("status", str(trained)))
    worker = data["worker"]
    assert worker["pid"] == os.getpid()
    assert worker["state"] == "stale"
    assert worker["heartbeat_age_s"] >= 300
    assert worker["log"] is None  # it trains in the foreground
    assert data["next_command"] == f"pup finetune cancel {trained}"


def test_a_training_run_with_a_fresh_heartbeat_is_running(trained: Path) -> None:
    _set_state(trained, phase="training", pid=os.getpid(), heartbeat_at=_ago(5))
    data = output(invoke("status", str(trained)))
    assert data["worker"]["state"] == "running"
    assert data["next_command"] == f"pup finetune status {trained} --watch"


def test_a_training_run_whose_process_died_is_resumable(trained: Path) -> None:
    _set_state(trained, phase="training", pid=_dead_pid(), heartbeat_at=_ago(5))
    data = output(invoke("status", str(trained)))
    assert data["worker"]["state"] == "exited"
    assert data["next_command"] == f"pup finetune resume {trained}"


def test_a_cancelled_run_is_evaluated_once_its_best_epoch_was_promoted(trained: Path) -> None:
    _set_state(trained, phase="cancelled")
    data = output(invoke("status", str(trained)))
    assert data["next_command"] == f"pup finetune evaluate {trained}"
    assert data["next_steps"] == [
        {
            "command": f"pup finetune resume {trained}",
            "why": "to train the remaining epochs (paid; it previews first)",
        }
    ]
    # Cancelled before any epoch: nothing was promoted, so resuming is all there is.
    result = json.loads((trained / "result.json").read_text())
    atomic_write_json(
        trained / "result.json", {**result, "promoted_model": None, "deployable_checkpoint": None}
    )
    data = output(invoke("status", str(trained)))
    assert data["next_command"] == f"pup finetune resume {trained}"
    assert data["next_steps"] == []


def test_non_finite_numbers_are_emitted_as_null(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = RunDir.status

    def with_nan(self: RunDir) -> dict[str, Any]:
        summary = original(self)
        nan_best = {**summary["best"], "val_loss": math.nan}
        return {**summary, "last_train_loss": math.nan, "val_losses": [math.inf], "best": nan_best}

    monkeypatch.setattr(RunDir, "status", with_nan)
    result = invoke("status", str(trained))
    assert result.exit_code == 0, result.stderr
    assert "NaN" not in result.stdout
    assert "Infinity" not in result.stdout
    data = output(result)
    assert data["last_train_loss"] is None
    assert data["val_losses"] == [None]
    assert data["best_val_loss"] is None


def test_status_watch_returns_for_a_training_run_whose_process_died(trained: Path) -> None:
    _set_state(trained, phase="training", pid=_dead_pid(), heartbeat_at=_ago(300))
    result = invoke("status", str(trained), "--watch", "--interval", "0.1")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["worker"]["state"] == "exited"
    assert data["watch_timed_out"] is False
    assert data["next_command"] == f"pup finetune resume {trained}"
    assert "training: epoch 2/2" in result.stderr


def test_status_reports_a_run_in_training_without_its_earlier_result(trained: Path) -> None:
    # A resumed run is training again: the old result.json describes the earlier attempt.
    _set_state(trained, phase="training", pid=os.getpid(), heartbeat_at=_ago(1))
    data = output(invoke("status", str(trained)))
    assert data["promoted_model"] is None
    assert data["deployable_checkpoint"] is None
    assert data["stopped_reason"] is None
    assert (trained / "result.json").exists()


def test_status_points_at_a_worker_that_failed_before_training(trained: Path) -> None:
    _set_state(trained, phase="interrupted", heartbeat_at=_ago(30))
    error = {"type": "StateError", "message": "the dataset changed", "notes": []}
    atomic_write_json(
        trained / "worker.json",
        {
            "pid": _dead_pid(),
            "mode": "resume",
            "started_at": _ago(10),
            "ended_at": _ago(9),
            "exit_error": error,
        },
    )
    data = output(invoke("status", str(trained)))
    assert data["worker"]["log"] == str(trained / "worker.log")
    assert data["worker"]["state"] == "exited"
    assert data["error"] == error
    assert data["resume_mode"] == "train"  # an interrupted run is resumable
    assert any("failed: the dataset changed" in w and "worker.log" in w for w in data["warnings"])
    assert data["next_command"] == f"pup finetune resume {trained}"


def test_status_points_at_a_worker_that_exited_without_a_word(trained: Path) -> None:
    _set_state(trained, phase="created", heartbeat_at=_ago(30), pid=None)
    atomic_write_json(
        trained / "worker.json", {"pid": _dead_pid(), "mode": "train", "started_at": _ago(10)}
    )
    data = output(invoke("status", str(trained)))
    assert data["error"] is None
    assert any("exited before training" in w for w in data["warnings"])


def test_a_worker_starting_for_too_long_is_stale(trained: Path) -> None:
    _set_state(trained, phase="created", heartbeat_at=_ago(300), pid=None)
    atomic_write_json(
        trained / "worker.json", {"pid": os.getpid(), "mode": "train", "started_at": _ago(200)}
    )
    data = output(invoke("status", str(trained), "--watch", "--interval", "0.1"))
    assert data["worker"]["state"] == "stale"
    assert any("has not begun training" in w for w in data["warnings"])


def test_status_says_a_worker_is_starting_to_resume_a_stopped_run(trained: Path) -> None:
    # Right after `resume --detach`, until the worker begins: the state is still the crash's.
    crash = {"type": "FakeCrash", "code": "external", "message": "simulated crash", "notes": []}
    _set_state(trained, phase="interrupted", heartbeat_at=_ago(30), error=crash)
    atomic_write_json(
        trained / "worker.json", {"pid": os.getpid(), "mode": "resume", "started_at": _ago(1)}
    )
    result = invoke("status", str(trained))
    assert output(result)["worker"]["state"] == "starting"
    headline = result.stderr.splitlines()[-1].split("; ")[0]
    assert headline.startswith("interrupted (external): epoch 2/2 after step 2/2 "), headline
    assert headline.endswith(f" (worker pid {os.getpid()} starting)"), headline


def test_status_shows_how_a_run_ended_when_its_worker_ends_while_status_reads(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = json.loads((trained / "state.json").read_text())
    _set_state(trained, phase="interrupted", heartbeat_at=_ago(30))
    atomic_write_json(
        trained / "worker.json", {"pid": os.getpid(), "mode": "resume", "started_at": _ago(1)}
    )
    real = train_module.worker_status
    ends: list[int] = []

    def worker_ends_first(run: RunDir, **state: Any) -> worker_module.WorkerStatus:
        # The resumed worker finishes after status read the state and before it looks at the
        # worker: it saves its last state, then records that it ended.
        if not ends:
            ends.append(_dead_pid())
            atomic_write_json(trained / "state.json", {**completed, "heartbeat_at": _ago(0)})
            worker = {"pid": ends[0], "mode": "resume", "started_at": _ago(1)}
            atomic_write_json(
                trained / "worker.json", {**worker, "ended_at": _ago(0), "exit_error": None}
            )
        return real(run, **state)

    monkeypatch.setattr(train_module, "worker_status", worker_ends_first)
    data = output(invoke("status", str(trained)))
    assert data["phase"] == "completed"
    assert data["next_command"] == f"pup finetune evaluate {trained}"


# --- status: progress numbers and a bounded watch -------------------------------------


def _training(run_dir: Path, **fields: Any) -> None:
    """Make ``run_dir`` look mid-training in this (live) process, with a fresh heartbeat."""
    _set_state(run_dir, phase="training", pid=os.getpid(), heartbeat_at=_ago(0), **fields)


def test_status_reports_steps_elapsed_time_and_no_eta_once_completed(trained: Path) -> None:
    data = output(invoke("status", str(trained)))
    assert (data["step"], data["max_steps"]) == (2, 2)
    assert isinstance(data["elapsed_seconds"], int)
    assert data["approx_eta_seconds"] is None
    assert "estimated_spend_usd" not in data  # the run records no hourly rate


def test_status_estimates_the_time_left_from_the_measured_step_time(trained: Path) -> None:
    _training(trained, epoch=1, step=10, epochs=0, steps_per_epoch=10, max_steps=40)
    _set_state(trained, elapsed_seconds=100.0)
    data = output(invoke("status", str(trained)))
    # 10 s a step, 30 steps left to max_epochs (a rough estimate: it may stop early).
    assert data["approx_eta_seconds"] == 300
    assert 100 <= data["elapsed_seconds"] <= 102  # plus the time since the last save


def test_status_prices_the_elapsed_time_when_the_run_has_an_hourly_rate(
    dataset: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    spend = ("--max-spend-usd", "5", "--hourly-rate-usd", "3.6")
    assert invoke(*train_args(dataset, run_dir, *spend, "--confirm")).exit_code == 0
    _set_state(run_dir, elapsed_seconds=1000.0)
    data = output(invoke("status", str(run_dir)))
    assert data["estimated_spend_usd"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90s", 90.0), ("10m", 600.0), ("2h", 7200.0), ("45", 45.0), ("0.5", 0.5), ("1.5m", 90.0)],
)
def test_watch_timeouts_take_seconds_minutes_hours_or_a_bare_number(
    text: str, seconds: float
) -> None:
    assert train_module.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["0", "0s", "-5", "5x", "m", "1d", "ten", "", "1e3"])
def test_a_watch_timeout_that_is_not_a_positive_duration_is_a_usage_error(
    trained: Path, text: str
) -> None:
    result = invoke("status", str(trained), "--watch", "--timeout", text)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--timeout" in result.stderr


@pytest.mark.parametrize("flags", [("--timeout", "90s"), ("--interval", "5")])
def test_timeout_and_interval_need_watch(trained: Path, flags: tuple[str, str]) -> None:
    result = invoke("status", str(trained), *flags)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert f"{flags[0]} needs --watch" in result.stderr


def test_a_watch_that_times_out_returns_the_status_with_the_same_watch_next(
    trained: Path,
) -> None:
    _training(trained, epoch=1, step=1, epochs=0)
    result = invoke("status", str(trained), "--watch", "--interval", "0.1", "--timeout", "0.3")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["phase"] == "training"
    assert data["watch_timed_out"] is True
    again = f"pup finetune status {trained} --watch --interval 0.1 --timeout 0.3"
    assert data["next_command"] == again
    assert f"stopped watching after 0.3s; the run keeps training (pid {os.getpid()})" in (
        result.stderr
    )
    assert f"Watch again: {again}" in result.stderr


def test_a_watch_that_ends_with_training_has_not_timed_out(trained: Path) -> None:
    data = output(invoke("status", str(trained), "--watch", "--timeout", "1h"))
    assert data["watch_timed_out"] is False
    assert data["next_command"] == f"pup finetune evaluate {trained}"


def test_status_without_watch_has_no_watch_field(trained: Path) -> None:
    assert "watch_timed_out" not in output(invoke("status", str(trained)))


@pytest.mark.parametrize(
    "env", [{"PUP_AGENT_MODE": "true"}, {"AI_AGENT": "an-agent"}, {"CLAUDECODE": "1"}]
)
def test_a_watch_under_a_coding_agent_times_out_by_default(
    trained: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(train_module, "AGENT_WATCH_TIMEOUT_S", 0.2)
    _training(trained, epoch=1, step=1, epochs=0)
    result = invoke("status", str(trained), "--watch", "--interval", "0.1")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert (data["phase"], data["watch_timed_out"]) == ("training", True)
    # The default is not pinned: the same watch under an agent times out again.
    assert data["next_command"] == f"pup finetune status {trained} --watch --interval 0.1"


def test_the_agent_default_is_90_seconds() -> None:
    assert train_module.AGENT_WATCH_TIMEOUT_S == 90.0


def test_a_watch_off_an_agent_has_no_default_timeout(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without an agent and --timeout the watch only ends with training: here, when a
    # (simulated) step finishes the run while it sleeps.
    monkeypatch.setattr(train_module, "AGENT_WATCH_TIMEOUT_S", 0.0)
    _training(trained, epoch=1, step=1, epochs=0)
    naps: list[float] = []

    def nap(seconds: float) -> None:
        naps.append(seconds)
        if len(naps) == 3:
            _set_state(trained, phase="completed")

    monkeypatch.setattr(train_module.time, "sleep", nap)
    data = output(invoke("status", str(trained), "--watch", "--interval", "0.1"))
    assert (data["phase"], data["watch_timed_out"]) == ("completed", False)
    assert naps == [0.1, 0.1, 0.1]


def test_a_watch_prints_a_line_when_the_run_moves_not_when_only_time_passes(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _training(trained, epoch=1, step=1, epochs=0, elapsed_seconds=10.0)
    naps: list[float] = []

    def nap(seconds: float) -> None:
        naps.append(seconds)
        if len(naps) < 3:  # time passes since the last save, nothing else
            _set_state(trained, heartbeat_at=_ago(5 * len(naps)))
        elif len(naps) == 3:
            _set_state(trained, step=2, heartbeat_at=_ago(0))
        else:
            _set_state(trained, phase="completed")

    monkeypatch.setattr(train_module.time, "sleep", nap)
    result = invoke("status", str(trained), "--watch", "--interval", "0.1")
    *lines, summary = result.stderr.splitlines()
    # The run's end is said once, by the closing summary, not by a last progress line too.
    assert [line.split(" loss")[0] for line in lines] == [
        "18:03:16 training: epoch 1/2 step 1/2",
        "18:03:16 training: epoch 1/2 step 2/2",
    ]
    assert summary.startswith("completed: epoch 1/2 step 2/2")


def test_a_quiet_watch_still_prints_the_last_line(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _training(trained, epoch=1, step=2, epochs=0, elapsed_seconds=10.0)
    monkeypatch.setenv("PUP_FINETUNE_QUIET", "1")
    monkeypatch.setattr(
        train_module.time, "sleep", lambda seconds: _set_state(trained, phase="completed")
    )
    result = invoke("status", str(trained), "--watch", "--interval", "0.1")
    assert [line.split(" loss")[0] for line in result.stderr.splitlines()] == [
        "18:03:16 training: epoch 1/2 step 2/2",
        "18:03:16 completed: epoch 1/2 step 2/2",
    ]


def test_ctrl_c_stops_only_the_watch_and_prints_the_status(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _training(trained, epoch=1, step=1, epochs=0)

    def interrupted(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(train_module.time, "sleep", interrupted)
    result = invoke("status", str(trained), "--watch", "--timeout", "10m")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert (data["phase"], data["watch_timed_out"]) == ("training", False)
    again = f"pup finetune status {trained} --watch --timeout 10m"
    assert data["next_command"] == again
    assert (
        f"stopped watching; the run keeps training (pid {os.getpid()}). Watch again: {again}"
        in result.stderr
    )


def test_ctrl_c_after_training_stopped_says_so_and_points_at_the_next_step(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _training(trained, epoch=1, step=1, epochs=0)

    def interrupted(seconds: float) -> None:
        _set_state(trained, pid=_dead_pid())  # the training process died meanwhile
        raise KeyboardInterrupt

    monkeypatch.setattr(train_module.time, "sleep", interrupted)
    result = invoke("status", str(trained), "--watch")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["next_command"] == f"pup finetune resume {trained}"
    assert "the run is training and no process is training it" in result.stderr
    assert f"Next: pup finetune resume {trained}" in result.stderr


def _line(
    steps_per_epoch: int | None = 7, epochs_completed: int | None = 2, **overrides: Any
) -> str:
    data: dict[str, Any] = {
        "phase": "training",
        "epoch": 3,
        "max_epochs": 5,
        "step": 21,
        "max_steps": 35,
        "last_train_loss": 1.1,
        "val_losses": [1.5, 1.35],
        "best_epoch": 2,
        "elapsed_seconds": 252,
        "approx_eta_seconds": 361,
        "worker": {"pid": 42, "state": "running", "heartbeat_age_s": 1.0, "log": None},
    }
    return train_module.status_line(
        {**data, **overrides}, steps_per_epoch=steps_per_epoch, epochs_completed=epochs_completed
    )


def test_the_watch_line_shows_steps_losses_best_epoch_elapsed_and_eta() -> None:
    assert _line(step=17) == (
        "training: epoch 3/5 step 17/35 loss 1.100 val 1.350 (best epoch 2) 4m12s elapsed, eta ~6m"
    )


def test_the_watch_line_says_when_the_epoch_is_validating() -> None:
    assert _line().startswith("training: epoch 3/5 step 21/35 loss 1.100 val 1.350 validating ")
    assert "validating" not in _line(epochs_completed=3)
    # What status prints alone (its closing summary) cannot tell, so it does not say.
    assert "validating" not in _line(steps_per_epoch=None, epochs_completed=None)


def test_the_watch_line_leaves_out_what_is_not_known_yet() -> None:
    line = _line(
        epoch=0,
        step=0,
        max_steps=None,
        steps_per_epoch=None,
        epochs_completed=0,
        last_train_loss=None,
        val_losses=[],
        best_epoch=None,
        elapsed_seconds=0,
        approx_eta_seconds=None,
    )
    assert line == "training: epoch 0/5 step 0"


def test_the_watch_line_says_when_the_training_process_is_gone() -> None:
    gone = {"pid": 42, "state": "exited", "heartbeat_age_s": 300.0, "log": None}
    assert _line(worker=gone).endswith("(pid 42 not running)")
    assert _line(worker={**gone, "state": "stale"}).endswith("(pid 42 not running)")
    assert _line(phase="completed", worker=gone).endswith("eta ~6m")
    # A completed run without a worker prints none.
    assert _line(phase="completed", worker=None).endswith("eta ~6m")


@pytest.mark.parametrize(
    ("seconds", "exact", "rough"),
    [(0, "0s", "0s"), (45.4, "45s", "45s"), (252, "4m12s", "4m"), (7500, "2h05m", "2h05m")],
)
def test_durations_read_as_seconds_minutes_and_hours(
    seconds: float, exact: str, rough: str
) -> None:
    assert train_module.format_duration(seconds) == exact
    assert train_module.format_duration(seconds, rough=True) == rough


def test_status_of_a_missing_run_is_an_error(tmp_path: Path) -> None:
    result = invoke("status", str(tmp_path / "nothing"))
    assert result.exit_code == 2
    assert "Error [not_a_run_dir]: " in result.stderr
    assert "no such run directory or run id" in result.stderr


# --- resume ------------------------------------------------------------------------------


def test_resume_after_a_crash_completes(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    batch = ("--batch-size", "1", "--max-epochs", "2")
    crashed = invoke(
        *train_args(dataset, run_dir, *batch, "--provider-option", "fail_at_step=9", "--confirm")
    )
    assert crashed.exit_code == 1
    assert "crashed (fail_at_step)" in crashed.stderr
    # The error itself says how to go on, as status does.
    assert f"Next: pup finetune resume {run_dir}\n" in crashed.stderr
    status = output(invoke("status", str(run_dir)))
    assert status["phase"] == "interrupted"
    assert status["error"]["type"] == "FakeCrash"
    assert status["next_command"] == f"pup finetune resume {run_dir}"
    assert status["next_steps"] == []

    preview = invoke("resume", str(run_dir))
    assert preview.exit_code == 2
    data = output(preview)
    assert data["preview"]["mode"] == "resume"
    assert data["preview"]["resume_from"]["epoch"] == 1
    assert data["preview"]["remaining_epochs"] == 1
    assert data["preview"]["action"].startswith("continue training with epoch 2")
    assert data["preview"]["data_recipients"] == ["training rows -> fake"]
    assert data["next_command"] == f"pup finetune resume {run_dir} --confirm"

    resumed = rerun(data["next_command"])
    assert resumed.exit_code == 0, resumed.stderr
    result = output(resumed)
    assert result["status"] == "completed"
    assert result["promoted_model"]["model_id"] == result["run_id"]
    assert [epoch["epoch"] for epoch in result["epochs"]] == [1, 2]
    assert result["next_command"] == f"pup finetune evaluate {run_dir}"


def test_a_finalize_only_resume_only_promotes(trained: Path) -> None:
    epochs_before = (trained / "epochs.json").read_text()
    _set_state(trained, phase="failed", resume_mode="finalize_only")
    status = output(invoke("status", str(trained)))
    assert status["next_command"] == f"pup finetune resume {trained}"

    preview = invoke("resume", str(trained))
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)["preview"]
    assert data["mode"] == "finalize"
    assert data["remaining_epochs"] == 0
    assert data["resume_from"] is None
    assert data["action"].startswith("finalize only")

    resumed = invoke("resume", str(trained), "--confirm")
    assert resumed.exit_code == 0, resumed.stderr
    result = output(resumed)
    assert result["status"] == "completed"
    assert result["promoted_model"]["model_id"] == result["run_id"]
    assert "epoch 1 step" not in resumed.stderr  # nothing was trained again
    assert (trained / "epochs.json").read_text() == epochs_before
    assert result["next_command"] == f"pup finetune evaluate {trained}"


def test_the_worker_retries_only_the_promotion(trained: Path) -> None:
    _set_state(trained, phase="failed", resume_mode="finalize_only")
    result = invoke("_worker", str(trained), "--mode", "finalize")
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""
    assert "finished: the run is completed" in result.stderr
    assert "epoch 1 step" not in result.stderr
    assert RunDir(trained).read_result()["promoted_model"] is not None  # type: ignore[index]
    launched = json.loads((trained / "worker.json").read_text())
    assert launched["pid"] == os.getpid()
    assert launched["exit_error"] is None
    assert launched["ended_at"]


def test_finalize_is_refused_for_a_run_with_training_left(trained: Path) -> None:
    _set_state(trained, phase="interrupted", resume_mode="train")
    state_before = (trained / "state.json").read_bytes()
    result = invoke("_worker", str(trained), "--mode", "finalize")
    assert result.exit_code == 2
    assert "has training left to do" in result.stderr
    assert (trained / "state.json").read_bytes() == state_before
    launched = json.loads((trained / "worker.json").read_text())
    assert launched["exit_error"]["type"] == "WrongPhase"


def test_the_worker_is_blocked_by_read_only(trained: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_state(trained, phase="interrupted")
    # A run on a real provider: the fake one never leaves the machine and is allowed.
    manifest = json.loads((trained / "run.json").read_text())
    (trained / "run.json").write_text(json.dumps({**manifest, "provider": "fireworks"}))
    state_before = (trained / "state.json").read_bytes()
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    result = invoke("_worker", str(trained), "--mode", "resume")
    assert result.exit_code == 2
    assert "PUP_READ_ONLY" in result.stderr
    assert (trained / "state.json").read_bytes() == state_before
    launched = json.loads((trained / "worker.json").read_text())
    assert launched["exit_error"]["type"] == "ReadOnlyError"


def test_resume_starts_a_run_that_never_started(trained: Path) -> None:
    _set_state(trained, phase="created")
    preview = output(invoke("resume", str(trained)))["preview"]
    assert preview["mode"] == "train"
    assert preview["action"].startswith("train from the start")


def test_resume_refuses_a_run_that_is_still_training(trained: Path) -> None:
    _set_state(trained, phase="training", pid=os.getpid(), heartbeat_at=_ago(1))
    with RunDir(trained).lock():
        result = invoke("resume", str(trained), "--confirm")
    assert result.exit_code == 2
    assert "Error [wrong_phase]: " in result.stderr
    assert "is still training" in result.stderr


def test_resume_refuses_while_a_worker_is_starting(trained: Path) -> None:
    _set_state(trained, phase="interrupted", heartbeat_at=_ago(30))
    atomic_write_json(
        trained / "worker.json", {"pid": os.getpid(), "mode": "resume", "started_at": _ago(1)}
    )
    result = invoke("resume", str(trained), "--confirm")
    assert result.exit_code == 2
    assert "Error [wrong_phase]: " in result.stderr
    assert "is starting" in result.stderr


def test_resume_refuses_a_changed_renderer(trained: Path) -> None:
    _set_state(trained, phase="interrupted")
    manifest = json.loads((trained / "run.json").read_text())
    atomic_write_json(trained / "run.json", {**manifest, "rendering": {"renderer": "other"}})
    result = invoke("resume", str(trained))
    assert result.exit_code == 2
    assert "Error [settings_conflict]: " in result.stderr
    assert "renderer changed since the run" in result.stderr


def test_resume_refuses_a_completed_run(trained: Path) -> None:
    result = invoke("resume", str(trained))
    assert result.exit_code == 2
    assert "it already completed" in result.stderr


def test_resume_refuses_a_changed_dataset(trained: Path, dataset: Path) -> None:
    _set_state(trained, phase="interrupted")
    extra = ROWS.parent / "tool_use.jsonl"
    lines = extra.read_text().splitlines()
    row = json.loads(lines[0])
    row["id"] = "added-later"
    row.pop("source", None)
    row["messages"][0]["content"] = "a question that was not there before"
    added = trained.parent / "added.jsonl"
    added.write_text(json.dumps(row) + "\n")
    import_dataset(dataset, added)
    result = invoke("resume", str(trained))
    assert result.exit_code == 2
    assert "Error [settings_conflict]: " in result.stderr
    assert "changed since the run" in result.stderr


def test_resume_refuses_a_relabeled_dataset(trained: Path, dataset: Path, tmp_path: Path) -> None:
    # Labels change a run's kept rows without touching rows.jsonl.
    _set_state(trained, phase="interrupted")
    row_id = DatasetDir(dataset).rows_for_split("train")[0].id
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": row_id, "keep": False, "reason": "off topic"}) + "\n")
    assert invoke("dataset", "label", str(dataset), "--file", str(labels)).exit_code == 0
    result = invoke("resume", str(trained))
    assert result.exit_code == 2
    assert "Error [settings_conflict]: " in result.stderr
    assert "changed since the run" in result.stderr


def test_read_only_allows_resuming_on_the_offline_fake_provider(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_state(trained, phase="interrupted")
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    preview = invoke("resume", str(trained))
    assert preview.exit_code == 2
    assert output(preview)["next_command"] is not None


# --- cancel ------------------------------------------------------------------------------


def test_cancel_refuses_a_run_that_is_not_training(trained: Path) -> None:
    result = invoke("cancel", str(trained), "--confirm")
    assert result.exit_code == 2
    assert "is completed, not training" in result.stderr


def test_cancel_refuses_a_dead_process(trained: Path) -> None:
    _set_state(trained, phase="training", pid=_dead_pid())
    result = invoke("cancel", str(trained), "--confirm")
    assert result.exit_code == 2
    assert "Error [wrong_phase]: " in result.stderr
    assert "no longer running" in result.stderr


def test_cancel_refuses_a_live_process_that_does_not_hold_the_lock(trained: Path) -> None:
    # This test process is alive and ours, but is not training the run: never signal it.
    _set_state(trained, phase="training", pid=os.getpid())
    result = invoke("cancel", str(trained), "--confirm")
    assert result.exit_code == 2
    assert "Error [wrong_phase]: " in result.stderr
    assert "not locked" in result.stderr
