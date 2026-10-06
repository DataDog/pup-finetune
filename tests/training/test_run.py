# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from pup_finetune.errors import StateError
from pup_finetune.storage import DirectoryLock
from pup_finetune.training import EPOCHS_FILE, PLAN_FILE, RESULT_FILE, STATE_FILE, RunDir


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_file_names() -> None:
    assert (PLAN_FILE, EPOCHS_FILE, STATE_FILE, RESULT_FILE) == (
        "plan.json",
        "epochs.json",
        "run-state.json",
        "result.json",
    )


def test_create_makes_a_private_directory_with_parents(tmp_path: Path) -> None:
    run = RunDir(tmp_path / "runs" / "sft-1")
    run.create()
    assert run.path.is_dir()
    assert mode(run.path) == 0o700


def test_create_accepts_an_empty_directory(tmp_path: Path) -> None:
    RunDir(tmp_path).create()
    assert list(tmp_path.iterdir()) == []


def test_create_accepts_a_directory_holding_only_its_lock(tmp_path: Path) -> None:
    with DirectoryLock(tmp_path):
        RunDir(tmp_path).create()


def test_create_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / PLAN_FILE).write_text("{}")
    with pytest.raises(StateError, match="is not empty") as info:
        RunDir(tmp_path).create()
    assert info.value.hint == "use a new --run-dir"


def test_create_refuses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "run"
    path.write_text("")
    with pytest.raises(StateError, match="not a directory"):
        RunDir(path).create()


def test_writes_are_private_json_and_read_back(tmp_path: Path) -> None:
    run = RunDir(tmp_path)
    run.write_plan({"run_id": "sft-1", "provider": "baseten"})
    run.write_result({"status": "completed"})
    run.write_epochs([{"epoch": 1, "train_loss": 0.5}])
    assert run.read_plan() == {"run_id": "sft-1", "provider": "baseten"}
    assert run.read_result() == {"status": "completed"}
    assert json.loads((tmp_path / EPOCHS_FILE).read_text()) == [{"epoch": 1, "train_loss": 0.5}]
    for name in (PLAN_FILE, RESULT_FILE, EPOCHS_FILE):
        assert mode(tmp_path / name) == 0o600


def test_update_state_merges_fields(tmp_path: Path) -> None:
    run = RunDir(tmp_path)
    assert run.update_state(status="planned", run_id="sft-1") == {
        "status": "planned",
        "run_id": "sft-1",
    }
    assert run.update_state(status="training", completed_epoch=1) == {
        "status": "training",
        "run_id": "sft-1",
        "completed_epoch": 1,
    }
    assert run.read_state() == {"status": "training", "run_id": "sft-1", "completed_epoch": 1}
    assert mode(tmp_path / STATE_FILE) == 0o600


def test_writes_refuse_non_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        RunDir(tmp_path).write_result({"loss": float("nan")})
    assert not (tmp_path / RESULT_FILE).exists()


@pytest.mark.parametrize("method", ["read_plan", "read_result", "read_state"])
def test_reading_a_missing_file_is_a_state_error(tmp_path: Path, method: str) -> None:
    with pytest.raises(StateError, match="does not exist") as info:
        getattr(RunDir(tmp_path), method)()
    assert info.value.hint == "pass the --run-dir of a training run"


def test_reading_a_non_object_is_a_state_error(tmp_path: Path) -> None:
    (tmp_path / RESULT_FILE).write_text("[]")
    with pytest.raises(StateError, match="expected a JSON object"):
        RunDir(tmp_path).read_result()


def test_reading_invalid_json_is_a_state_error(tmp_path: Path) -> None:
    (tmp_path / STATE_FILE).write_text("{")
    with pytest.raises(StateError, match="invalid JSON"):
        RunDir(tmp_path).update_state(status="failed")
