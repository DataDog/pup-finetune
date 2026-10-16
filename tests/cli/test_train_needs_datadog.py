# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`train` and `benchmark` on a real provider read only a dataset that is in Datadog: every
kept row pushed, no push in flight. The offline fake provider trains on any directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.data.push import PUSH_FILE
from tests.fakes.pushed import mark_pushed

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    result = invoke("dataset", "import", "ds", "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return Path("ds")


def train(dataset: Path) -> Result:
    return invoke("train", "--dataset", str(dataset), "--model", "qwen3p8-27b")


def test_train_refuses_a_dataset_never_pushed_and_names_the_push(dataset: Path) -> None:
    result = train(dataset)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        "Error [usage]: train reads a dataset that is in Datadog, and ds was never pushed\n"
    )
    assert "`pup finetune dataset push ds --name <NAME>`" in result.stderr
    assert "`pup finetune dataset fetch <DIR> --project <PROJECT> --name <NAME>`" in result.stderr


def test_train_refuses_kept_rows_not_pushed_yet(dataset: Path) -> None:
    mark_pushed(dataset)
    receipt = json.loads((dataset / PUSH_FILE).read_text())
    receipt["record_ids"].popitem()
    (dataset / PUSH_FILE).write_text(json.dumps(receipt))
    result = train(dataset)
    assert result.exit_code == 2
    assert "1 of its kept rows are not in dataset 'synthetic-sft' yet" in result.stderr
    assert "`pup finetune dataset push ds` sends them" in result.stderr


def test_train_refuses_a_push_still_in_flight(dataset: Path) -> None:
    mark_pushed(dataset)
    receipt = json.loads((dataset / PUSH_FILE).read_text())
    receipt["in_flight"] = [next(iter(receipt["record_ids"]))]
    (dataset / PUSH_FILE).write_text(json.dumps(receipt))
    result = train(dataset)
    assert result.exit_code == 2
    assert "its last push did not finish" in result.stderr


def test_train_refuses_an_unreadable_push_receipt(dataset: Path) -> None:
    (dataset / PUSH_FILE).write_text("{not json")
    result = train(dataset)
    assert result.exit_code == 2
    assert f"{PUSH_FILE} is unreadable" in result.stderr
    assert "`pup finetune dataset show ds`" in result.stderr


def test_dropped_rows_need_no_push(dataset: Path, tmp_path: Path) -> None:
    mark_pushed(dataset)
    row_id = json.loads((dataset / "rows.jsonl").read_text().splitlines()[0])["id"]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": row_id, "keep": False, "reason": "synthetic"}) + "\n")
    assert invoke("dataset", "label", str(dataset), "--file", str(labels)).exit_code == 0
    result = train(dataset)
    # Past the push check: the preview goes on to the provider, which has no key here.
    assert "reads a dataset that is in Datadog" not in result.stderr


def test_benchmark_on_a_real_model_refuses_a_dataset_never_pushed(dataset: Path) -> None:
    result = invoke("benchmark", str(dataset), "--models", "qwen3p8-27b")
    assert result.exit_code == 2
    assert result.stderr.startswith(
        "Error [usage]: benchmark reads a dataset that is in Datadog, and ds was never pushed\n"
    )


def test_the_offline_fake_provider_trains_on_a_dataset_never_pushed(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    result = invoke("train", "--dataset", str(dataset), "--model", "fake-model")
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "confirmation_required"
    result = invoke("benchmark", str(dataset), "--models", "fake-model")
    assert json.loads(result.stdout)["status"] == "confirmation_required"
