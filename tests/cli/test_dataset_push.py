# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset push`, end to end through the Typer app over a fake pup."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import dataset as dataset_cli
from pup_finetune.data.push import PUSH_FILE, SOURCE_TAG
from pup_finetune.data.split import SPLITS
from tests.fakes.datasets import FakeDataset, FakeDatasetsApi
from tests.fakes.pup import FakePup

ROWS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rows"
TOOL_USE = ROWS_DIR / "tool_use.jsonl"
ROW_IDS = [json.loads(line)["id"] for line in TOOL_USE.read_text().splitlines()]
PROJECT = "pup-finetune-dev"
NAME = "synthetic-push"

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@pytest.fixture
def fake() -> FakePup:
    return FakePup()


@pytest.fixture
def api(fake: FakePup, monkeypatch: pytest.MonkeyPatch) -> FakeDatasetsApi:
    """Fake datasets the dataset commands reach through a fake pup."""
    remote = FakeDatasetsApi()
    remote.install(fake)
    monkeypatch.setattr(dataset_cli, "Pup", fake.client)
    return remote


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    target = tmp_path / "ds"
    result = invoke("import", str(target), "--file", str(TOOL_USE))
    assert result.exit_code == 0, result.stderr
    return target


def invoke(*args: str, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(app, ["dataset", *args], env=env)


def push(directory: Path, *extra: str, env: dict[str, str] | None = None) -> Result:
    return invoke("push", str(directory), "--name", NAME, "--project", PROJECT, *extra, env=env)


def record_id_of(remote: FakeDataset, row_id: str) -> str:
    (record_id,) = [
        record_id
        for record_id, record in zip(remote.record_ids, remote.records, strict=True)
        if record["metadata"]["row_id"] == row_id
    ]
    return record_id


def output_json(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def test_push_without_confirm_previews_and_sends_nothing(
    dataset: Path, api: FakeDatasetsApi, fake: FakePup
) -> None:
    result = push(dataset)
    assert result.exit_code == 2, result.stderr
    assert "Confirmation required" in result.stderr
    data = output_json(result)
    assert data["status"] == "confirmation_required"
    preview = data["preview"]
    assert preview["project"] == PROJECT
    assert preview["dataset_name"] == NAME
    assert preview["dataset_id"] is None
    assert preview["records_to_push"] == len(ROW_IDS)
    assert sum(preview["records_to_push_by_split"].values()) == len(ROW_IDS)
    assert set(preview["records_to_push_by_split"]) == set(SPLITS)
    assert preview["already_pushed"] == 0
    assert preview["stale_records"] == {}
    sample = preview["sample_record"]
    assert set(sample) == {"input", "expected_output", "metadata", "tags"}
    assert sample["metadata"]["row_id"] in ROW_IDS
    assert SOURCE_TAG in sample["tags"]
    assert "create dataset" in preview["action"]
    assert shlex.split(data["next_command"]) == [
        "pup",
        "finetune",
        "dataset",
        "push",
        str(dataset),
        "--name",
        NAME,
        "--project",
        PROJECT,
        "--confirm",
    ]
    assert fake.calls == []
    assert not (dataset / PUSH_FILE).exists()


def test_push_with_confirm_pushes_and_emits_the_receipt(
    dataset: Path, api: FakeDatasetsApi
) -> None:
    result = push(dataset, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    remote = api.dataset_named(NAME)
    assert data["dataset_dir"] == str(dataset.resolve())
    assert (data["project"], data["project_id"]) == (PROJECT, remote.project_id)
    assert (data["dataset_name"], data["dataset_id"]) == (NAME, remote.id)
    assert data["dataset_version"] == remote.version
    assert (data["pushed"], data["already_pushed"], data["records"]) == (len(ROW_IDS), 0, 14)
    assert sum(data["records_by_split"].values()) == len(ROW_IDS)
    assert data["stale_records"] == {}
    assert data["receipt_file"] == str((dataset / PUSH_FILE).resolve())
    assert data["next_command"] == f"pup finetune train --dataset {dataset} ..."
    assert "dataset_url" not in data
    assert len(remote.records) == len(ROW_IDS)


def test_push_is_refused_read_only_even_as_a_preview(
    dataset: Path, api: FakeDatasetsApi, fake: FakePup
) -> None:
    for extra in ((), ("--confirm",)):
        result = push(dataset, *extra, env={"PUP_READ_ONLY": "true"})
        assert result.exit_code == 2
        assert "PUP_READ_ONLY" in result.stderr
    assert fake.calls == []
    assert not (dataset / PUSH_FILE).exists()


def test_confirm_is_not_implied_by_auto_approve(
    dataset: Path, api: FakeDatasetsApi, fake: FakePup
) -> None:
    result = push(dataset, env={"PUP_AUTO_APPROVE": "true", "PUP_AGENT_MODE": "true"})
    assert result.exit_code == 2
    assert output_json(result)["status"] == "confirmation_required"
    assert fake.calls == []


def label(dataset: Path, tmp_path: Path, *row_ids: str) -> None:
    labels = tmp_path / "labels.jsonl"
    lines = [json.dumps({"id": row_id, "keep": False, "reason": "wrong"}) for row_id in row_ids]
    labels.write_text("\n".join(lines) + "\n")
    assert invoke("label", str(dataset), "--file", str(labels)).exit_code == 0


def test_nothing_left_to_push_reports_the_receipt_without_confirm(
    dataset: Path, api: FakeDatasetsApi, fake: FakePup, tmp_path: Path
) -> None:
    assert push(dataset, "--confirm").exit_code == 0
    label(dataset, tmp_path, ROW_IDS[0])
    calls = len(fake.calls)
    result = push(dataset)
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    remote = api.dataset_named(NAME)
    assert "status" not in data
    assert data["dataset_id"] == remote.id
    assert (data["pushed"], data["already_pushed"]) == (0, len(ROW_IDS) - 1)
    assert data["stale_records"] == {ROW_IDS[0]: record_id_of(remote, ROW_IDS[0])}
    assert "not deleted" in data["warnings"][0]
    assert data["next_command"].startswith("pup finetune train")
    assert len(fake.calls) == calls  # nothing reached Datadog
    assert output_json(push(dataset, "--confirm")) == data


def test_rows_changed_after_a_push_are_reported_with_a_warning(
    dataset: Path, api: FakeDatasetsApi, fake: FakePup, tmp_path: Path
) -> None:
    assert push(dataset, "--confirm").exit_code == 0
    remote = api.dataset_named(NAME)
    # Reasoning filled in for a stored row under the same id, as a resumed pull can.
    rows = dataset / "rows.jsonl"
    lines = rows.read_text().splitlines()
    row = json.loads(lines[0])
    row["messages"][-1]["reasoning_content"] = "Check the units first."
    rows.write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    calls = len(fake.calls)
    result = push(dataset)
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    assert data["changed_records"] == {row["id"]: record_id_of(remote, row["id"])}
    assert (data["pushed"], data["already_pushed"]) == (0, len(ROW_IDS))
    (warning,) = data["warnings"]
    assert "not updated" in warning and PUSH_FILE in warning and "--name" in warning
    assert len(fake.calls) == calls
    # With rows to push, the preview reports them too.
    more = tmp_path / "more.jsonl"
    more.write_text(
        json.dumps(
            {
                "id": "row-synthetic-extra",
                "messages": [
                    {"role": "user", "content": "Synthetic question?"},
                    {"role": "assistant", "content": "Synthetic answer."},
                ],
                "source": {"trace_id": "synthetic-trace-extra"},
            }
        )
        + "\n"
    )
    assert invoke("import", str(dataset), "--file", str(more)).exit_code == 0
    preview = output_json(push(dataset))["preview"]
    assert preview["changed_records"] == data["changed_records"]
    assert any("changed after they were pushed" in w for w in preview["warnings"])


def test_the_preview_after_more_rows_were_added_shows_only_those(
    tmp_path: Path, api: FakeDatasetsApi
) -> None:
    rows = TOOL_USE.read_text().splitlines()
    first, rest = tmp_path / "first.jsonl", tmp_path / "rest.jsonl"
    first.write_text("\n".join(rows[:6]) + "\n")
    rest.write_text("\n".join(rows[6:]) + "\n")
    directory = tmp_path / "ds"
    assert invoke("import", str(directory), "--file", str(first)).exit_code == 0
    assert push(directory, "--confirm").exit_code == 0
    assert invoke("import", str(directory), "--file", str(rest)).exit_code == 0
    result = push(directory)
    assert result.exit_code == 2
    preview = output_json(result)["preview"]
    remote = api.dataset_named(NAME)
    new_ids = set(ROW_IDS[6:])
    assert preview["action"].startswith(f"add {len(new_ids)} records")
    assert preview["dataset_id"] == remote.id
    assert preview["records_to_push"] == len(new_ids)
    assert preview["already_pushed"] == 6
    assert preview["sample_record"]["metadata"]["row_id"] in new_ids


def test_the_preview_after_a_lost_batch_says_it_checks_first(
    dataset: Path, api: FakeDatasetsApi
) -> None:
    api.records_add_lost_responses = {1}
    assert push(dataset, "--confirm").exit_code == 1
    preview = output_json(push(dataset))["preview"]
    assert preview["action"].startswith(f"check dataset {api.dataset_named(NAME).id}")
    assert "interrupted push" in preview["warnings"][0]
    data = output_json(push(dataset, "--confirm"))
    assert (data["pushed"], data["records"]) == (0, len(ROW_IDS))
    assert len(api.dataset_named(NAME).records) == len(ROW_IDS)


def test_a_dataset_without_kept_rows_fails_before_the_preview(
    dataset: Path, api: FakeDatasetsApi, fake: FakePup, tmp_path: Path
) -> None:
    label(dataset, tmp_path, *ROW_IDS)
    result = push(dataset)
    assert result.exit_code == 2
    assert "no kept rows" in result.stderr
    assert "confirmation_required" not in result.stdout
    assert fake.calls == []


def test_another_name_is_a_usage_error(dataset: Path, api: FakeDatasetsApi) -> None:
    assert push(dataset, "--confirm").exit_code == 0
    result = invoke("push", str(dataset), "--name", "other", "--project", PROJECT)
    assert result.exit_code == 2
    assert "was pushed to dataset" in result.stderr
    assert PUSH_FILE in result.stderr


def test_the_default_project(dataset: Path, api: FakeDatasetsApi) -> None:
    result = invoke("push", str(dataset), "--name", NAME)
    assert output_json(result)["preview"]["project"] == "pup-finetune"


def test_name_is_required(dataset: Path) -> None:
    result = invoke("push", str(dataset))
    assert result.exit_code == 2
    assert "--name" in _ANSI.sub("", result.stderr)


def test_show_reports_the_push(dataset: Path, api: FakeDatasetsApi) -> None:
    assert output_json(invoke("show", str(dataset)))["push"] is None
    assert push(dataset, "--confirm").exit_code == 0
    shown = output_json(invoke("show", str(dataset)))["push"]
    remote = api.dataset_named(NAME)
    assert shown["dataset_id"] == remote.id
    assert shown["records"] == len(ROW_IDS)
    assert (shown["project"], shown["dataset_name"]) == (PROJECT, NAME)
    assert shown["dataset_version"] == remote.version
