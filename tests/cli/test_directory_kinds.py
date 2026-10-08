# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A dataset directory given to a run command, or a run directory given to a dataset one,
gets an error that names both kinds and the command that takes it."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import dataset as dataset_cli
from tests.fakes.pup import FakePup

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    assert invoke("dataset", "import", str(path), "--file", str(ROWS)).exit_code == 0
    return path


@pytest.fixture
def run(dataset: Path, tmp_path: Path) -> Path:
    path = tmp_path / "run"
    result = invoke(
        "train",
        "--provider",
        "fake",
        "--model",
        "fake-model",
        "--dataset",
        str(dataset),
        "--run-dir",
        str(path),
        "--max-epochs",
        "1",
        "--confirm",
    )
    assert result.exit_code == 0, result.stderr
    return path


@pytest.mark.parametrize(
    "command", ["status", "resume", "cancel", "deploy", "undeploy", "evaluate"]
)
def test_a_dataset_given_to_a_run_command_is_named_as_one(dataset: Path, command: str) -> None:
    result = invoke(command, str(dataset))
    assert result.exit_code == 2
    assert result.stdout == ""
    assert f"Error [not_a_run_dir]: {dataset} is a dataset directory, not a training run" in (
        result.stderr
    )
    assert f"`pup finetune train --dataset {dataset} --model <MODEL>`" in result.stderr
    assert "run.json does not exist" not in result.stderr


def test_evaluate_says_it_takes_the_run_directory(dataset: Path) -> None:
    result = invoke("evaluate", str(dataset))
    assert "evaluate takes the run directory or run id as its argument" in result.stderr
    assert "`pup finetune evaluate <RUN_DIR>`" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("dataset", "show"),
        ("dataset", "sample"),
        ("dataset", "push", "--name", "x"),
        ("dataset", "label", "--file", str(ROWS)),
        ("dataset", "import", "--file", str(ROWS)),
        ("dataset", "triage", "--rule", "stay on topic"),
        ("dataset", "pull", "--ml-app", "a"),
    ],
    ids=lambda args: args[1],
)
def test_a_run_given_to_a_dataset_command_is_named_as_one(
    run: Path, dataset: Path, args: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePup()  # pull and push would reach Datadog: nothing may be called
    monkeypatch.setattr(dataset_cli, "Pup", fake.client)
    before = sorted(entry.name for entry in run.iterdir())
    result = invoke(*args[:2], str(run), *args[2:])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert f"Error [not_a_dataset_dir]: {run} is a training run directory, not a dataset" in (
        result.stderr
    )
    assert f"the run trained on the dataset in {dataset.resolve()}" in result.stderr
    assert f"`pup finetune status {run}`" in result.stderr
    # Nothing was written into the run, e.g. by an import or a pull, and nothing was called.
    assert sorted(entry.name for entry in run.iterdir()) == before
    assert fake.calls == []


def test_train_names_both_kinds_too(run: Path, dataset: Path, tmp_path: Path) -> None:
    base = ["train", "--provider", "fake", "--model", "fake-model"]
    on_run = invoke(*base, "--dataset", str(run), "--run-dir", str(tmp_path / "new"))
    assert on_run.exit_code == 2
    assert f"{run} is a training run directory, not a dataset" in on_run.stderr
    into_dataset = invoke(*base, "--dataset", str(dataset), "--run-dir", str(dataset))
    assert into_dataset.exit_code == 2
    assert f"{dataset} is a dataset directory, not a new run directory" in into_dataset.stderr
    assert f"--dataset {dataset}" in into_dataset.stderr
