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
    assert RunDir(run_dir).spec().model_dump(mode="json") == preview["spec"]


def test_the_preview_counts_rows_examples_and_tokens(dataset: Path, tmp_path: Path) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--max-epochs", "4"))
    preview = output(result)["preview"]
    counts = kept_rows_per_split(dataset)
    assert preview["provider"] == "fake"
    assert preview["model"] == "fake-model"
    assert preview["rows"] == counts
    assert preview["examples"] == {"train": counts["train"], "validation": counts["validation"]}
    assert preview["rejected"] == 0
    assert preview["rejected_by_code"] == {}
    assert preview["tokens_per_epoch"] > 0
    assert preview["max_tokens_total"] == preview["tokens_per_epoch"] * 4
    assert preview["spec"]["max_epochs"] == 4
    assert preview["data_destination"] == "training data is sent to fake"


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
    assert preview["rejected"] == expected
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


def test_read_only_allows_the_preview_but_not_training(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    run_dir = tmp_path / "run"
    preview = invoke(*train_args(dataset, run_dir))
    assert preview.exit_code == 2
    assert output(preview)["status"] == "confirmation_required"

    confirmed = invoke(*train_args(dataset, run_dir, "--confirm"))
    assert confirmed.exit_code == 2
    assert "PUP_READ_ONLY" in confirmed.stderr
    assert confirmed.stdout == ""
    assert not run_dir.exists()


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
    assert result.exit_code == 1
    assert "is not empty" in result.stderr


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
    assert data["phase"] == "completed"
    assert data["run_dir"] == str(run_dir)
    assert data["promoted_model"]["model_id"] == data["run_id"]
    assert data["next_command"] == f"pup finetune evaluate {run_dir}"
    on_disk = json.loads((run_dir / "result.json").read_text())
    assert on_disk["promoted_model"] == data["promoted_model"]
    assert "epoch 1 step 1: loss" in result.stderr
    assert "epoch 2 done" in result.stderr


# --- status ------------------------------------------------------------------------------


STATUS_FIELDS = {
    "run_id",
    "phase",
    "epoch",
    "step",
    "last_train_loss",
    "val_losses",
    "best_epoch",
    "promoted_model",
    "deployable_checkpoint",
    "resume_mode",
    "worker",
    "error",
    "finalize_error",
    "next_command",
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
    assert data["deployable_checkpoint"]["kind"] == "sampler"
    assert data["resume_mode"] == "train"
    assert data["error"] is None
    assert data["finalize_error"] is None
    assert set(data["worker"]) >= {"pid", "alive", "heartbeat_age_s", "stale"}
    assert data["worker"]["pid"] == os.getpid()  # it trained in this process
    assert data["worker"]["running"] is False
    assert data["worker"]["stale"] is False
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
    assert worker["alive"] is True
    assert worker["stale"] is True
    assert worker["heartbeat_age_s"] >= 300
    assert worker["running"] is False
    assert data["next_command"] == f"pup finetune cancel {trained}"


def test_a_training_run_with_a_fresh_heartbeat_is_running(trained: Path) -> None:
    _set_state(trained, phase="training", pid=os.getpid(), heartbeat_at=_ago(5))
    data = output(invoke("status", str(trained)))
    assert data["worker"]["stale"] is False
    assert data["worker"]["running"] is True
    assert data["next_command"] == f"pup finetune status {trained} --watch"


def test_a_training_run_whose_process_died_is_resumable(trained: Path) -> None:
    _set_state(trained, phase="training", pid=_dead_pid(), heartbeat_at=_ago(5))
    data = output(invoke("status", str(trained)))
    assert data["worker"]["alive"] is False
    assert data["worker"]["running"] is False
    assert data["next_command"] == f"pup finetune resume {trained}"


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
    assert (data["worker"]["alive"], data["worker"]["stale"]) == (False, True)
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
    assert data["worker_log"] == str(trained / "worker.log")
    assert data["worker"]["error"] == error
    assert data["worker"]["running"] is False
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
    worker = data["worker"]
    assert (worker["alive"], worker["starting"], worker["running"], worker["stale"]) == (
        True,
        False,
        False,
        True,
    )
    assert any("has not begun training" in w for w in data["warnings"])


def test_status_of_a_missing_run_is_an_error(tmp_path: Path) -> None:
    result = invoke("status", str(tmp_path / "nothing"))
    assert result.exit_code == 1
    assert "does not exist" in result.stderr


# --- resume ------------------------------------------------------------------------------


def test_resume_after_a_crash_completes(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    batch = ("--batch-size", "1", "--max-epochs", "2")
    crashed = invoke(
        *train_args(dataset, run_dir, *batch, "--provider-option", "fail_at_step=9", "--confirm")
    )
    assert crashed.exit_code == 1
    assert "crashed (fail_at_step)" in crashed.stderr
    status = output(invoke("status", str(run_dir)))
    assert status["phase"] == "interrupted"
    assert status["error"]["type"] == "FakeCrash"
    assert status["next_command"] == f"pup finetune resume {run_dir}"

    preview = invoke("resume", str(run_dir))
    assert preview.exit_code == 2
    data = output(preview)
    assert data["preview"]["mode"] == "resume"
    assert data["preview"]["resume_from"]["epoch"] == 1
    assert data["preview"]["remaining_epochs"] == 1
    assert data["preview"]["action"].startswith("continue training with epoch 2")
    assert data["next_command"] == f"pup finetune resume {run_dir} --confirm"

    resumed = rerun(data["next_command"])
    assert resumed.exit_code == 0, resumed.stderr
    result = output(resumed)
    assert result["phase"] == "completed"
    assert result["promoted_model"]["model_id"] == result["run_id"]
    assert [epoch["epoch"] for epoch in result["epochs"]] == [1, 2]


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
    assert result["phase"] == "completed"
    assert result["promoted_model"]["model_id"] == result["run_id"]
    assert "epoch 1 step" not in resumed.stderr  # nothing was trained again
    assert (trained / "epochs.json").read_text() == epochs_before


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
    assert launched["exit_error"]["type"] == "UsageError"


def test_the_worker_is_blocked_by_read_only(trained: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_state(trained, phase="interrupted")
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
    assert result.exit_code == 1
    assert "is still training" in result.stderr


def test_resume_refuses_while_a_worker_is_starting(trained: Path) -> None:
    _set_state(trained, phase="interrupted", heartbeat_at=_ago(30))
    atomic_write_json(
        trained / "worker.json", {"pid": os.getpid(), "mode": "resume", "started_at": _ago(1)}
    )
    result = invoke("resume", str(trained), "--confirm")
    assert result.exit_code == 1
    assert "is starting" in result.stderr


def test_resume_refuses_a_changed_renderer(trained: Path) -> None:
    _set_state(trained, phase="interrupted")
    manifest = json.loads((trained / "run.json").read_text())
    atomic_write_json(trained / "run.json", {**manifest, "rendering": {"renderer": "other"}})
    result = invoke("resume", str(trained))
    assert result.exit_code == 1
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
    assert result.exit_code == 1
    assert "changed since the run" in result.stderr


def test_resume_is_blocked_by_read_only(trained: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_state(trained, phase="interrupted")
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    assert invoke("resume", str(trained)).exit_code == 2  # the preview
    result = invoke("resume", str(trained), "--confirm")
    assert result.exit_code == 2
    assert "PUP_READ_ONLY" in result.stderr


# --- cancel ------------------------------------------------------------------------------


def test_cancel_refuses_a_run_that_is_not_training(trained: Path) -> None:
    result = invoke("cancel", str(trained), "--confirm")
    assert result.exit_code == 2
    assert "is completed, not training" in result.stderr


def test_cancel_refuses_a_dead_process(trained: Path) -> None:
    _set_state(trained, phase="training", pid=_dead_pid())
    result = invoke("cancel", str(trained), "--confirm")
    assert result.exit_code == 1
    assert "no longer running" in result.stderr


def test_cancel_refuses_a_live_process_that_does_not_hold_the_lock(trained: Path) -> None:
    # This test process is alive and ours, but is not training the run: never signal it.
    _set_state(trained, phase="training", pid=os.getpid())
    result = invoke("cancel", str(trained), "--confirm")
    assert result.exit_code == 1
    assert "not locked" in result.stderr
