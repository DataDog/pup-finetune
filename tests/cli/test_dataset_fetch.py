# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset fetch`, end to end through the Typer app over a fake pup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import dataset as dataset_cli
from pup_finetune.data.push import PUSH_FILE
from tests.fakes.datasets import FakeDataset, FakeDatasetsApi
from tests.fakes.pup import FakePup

TOOL_USE = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
PROJECT = "pup-finetune-dev"
NAME = "synthetic-fetch"

runner = CliRunner()


@pytest.fixture
def fake() -> FakePup:
    return FakePup()


@pytest.fixture
def api(fake: FakePup, monkeypatch: pytest.MonkeyPatch) -> FakeDatasetsApi:
    remote = FakeDatasetsApi()
    remote.install(fake)
    monkeypatch.setattr(dataset_cli, "Pup", fake.client)
    return remote


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["dataset", *args])


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.fixture
def pushed(api: FakeDatasetsApi, tmp_path: Path) -> Path:
    """A dataset imported from the fixture rows and pushed to the fake Datadog."""
    source = tmp_path / "source"
    assert invoke("import", str(source), "--file", str(TOOL_USE)).exit_code == 0
    result = invoke("push", str(source), "--name", NAME, "--project", PROJECT, "--confirm")
    assert result.exit_code == 0, result.stderr
    return source


def fetch(directory: Path) -> Result:
    return invoke("fetch", str(directory), "--project", PROJECT, "--name", NAME)


def rows(directory: Path) -> dict[str, Any]:
    lines = (directory / "rows.jsonl").read_text().splitlines()
    return {json.loads(line)["id"]: json.loads(line) for line in lines}


def splits(directory: Path) -> dict[str, str]:
    assignments: dict[str, str] = json.loads((directory / "splits.json").read_text())["assignments"]
    return assignments


def test_a_pushed_dataset_fetches_back_to_the_same_rows_splits_and_records(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    target = tmp_path / "fetched"
    result = fetch(target)
    assert result.exit_code == 0, result.stderr
    data = output(result)
    remote = api.dataset_named(NAME)
    assert data["fetch"]["dataset_id"] == remote.id
    assert data["fetch"]["dataset_version"] == remote.version
    assert data["excluded"] == 0
    assert {row_id: row["messages"] for row_id, row in rows(target).items()} == {
        row_id: row["messages"] for row_id, row in rows(pushed).items()
    }
    assert splits(target) == {key: splits(pushed)[key] for key in splits(target)}
    receipt = json.loads((target / PUSH_FILE).read_text())
    original = json.loads((pushed / PUSH_FILE).read_text())
    assert receipt["dataset_id"] == remote.id
    assert receipt["record_ids"] == original["record_ids"]
    assert data["pipeline"]["push"]["status"] == "pushed"
    # Curated in Datadog: the rows go straight to training.
    assert any("train --dataset" in step["command"] for step in data["next_steps"])


def test_records_that_are_not_conversations_or_name_conflicting_splits_are_excluded(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    remote: FakeDataset = api.dataset_named(NAME)
    first = remote.records[0]
    twin = json.loads(json.dumps(first))
    twin["metadata"]["row_id"] = "twin-of-first"
    twin["input"]["messages"][-1]["content"] = "a different question, same trace"
    twin["metadata"]["split"] = next(
        s for s in ("train", "test") if s != first["metadata"]["split"]
    )
    twin["tags"] = []
    remote.records += [{"input": "free text", "expected_output": "an answer", "metadata": {}}, twin]
    remote.record_ids += ["record-text", "record-twin"]
    result = fetch(tmp_path / "fetched")
    assert result.exit_code == 0, result.stderr
    assert output(result)["excluded_by_code"] == {"split_conflict": 1, "unreadable_record": 1}


def test_a_record_without_a_split_is_assigned_one(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    remote = api.dataset_named(NAME)
    record = remote.records[0]
    record["metadata"].pop("split")
    record["tags"] = [tag for tag in record["tags"] if not tag.startswith("split:")]
    result = fetch(tmp_path / "fetched")
    assert result.exit_code == 0, result.stderr
    assert sum(output(result)["splits"].values()) == len(remote.records)


def test_a_newer_version_is_fetched_into_a_new_directory(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    target = tmp_path / "fetched"
    assert fetch(target).exit_code == 0
    api.dataset_named(NAME).version += 1
    again = fetch(target)
    assert again.exit_code == 2
    assert "settings" in again.stderr.lower()


def test_an_unknown_dataset_or_project_is_a_usage_error(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    missing = invoke("fetch", str(tmp_path / "a"), "--project", PROJECT, "--name", "nope")
    assert missing.exit_code == 2
    assert "no dataset named 'nope'" in missing.stderr
    nowhere = invoke("fetch", str(tmp_path / "b"), "--project", "nope", "--name", NAME)
    assert nowhere.exit_code == 2
    assert "no Agent Observability project named 'nope'" in nowhere.stderr


def test_an_imported_directory_is_not_fetched_into(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    result = fetch(pushed)
    assert result.exit_code == 2
    assert "dataset import" in result.stderr


def test_a_listing_with_unreadable_items_is_refused(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    api.records_all_extra = ["junk"]
    result = fetch(tmp_path / "fetched")
    assert result.exit_code == 1
    assert "cannot be read as records" in result.stderr
    assert not (tmp_path / "fetched" / "rows.jsonl").exists()


def test_a_copied_record_keeps_its_own_row(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    remote = api.dataset_named(NAME)
    copy = json.loads(json.dumps(remote.records[0]))
    copy["expected_output"]["content"] = "An edited answer to the same question."
    remote.records.append(copy)
    remote.record_ids.append("record-copy")
    result = fetch(tmp_path / "fetched")
    assert result.exit_code == 0, result.stderr
    assert "record-copy" in rows(tmp_path / "fetched")


def test_a_split_named_only_by_a_tag_is_kept(
    api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    remote = api.dataset_named(NAME)
    record = next(r for r in remote.records if r["metadata"]["split"] == "test")
    record["metadata"].pop("split")
    assert "split:test" in record["tags"]
    target = tmp_path / "fetched"
    assert fetch(target).exit_code == 0
    trace = record["metadata"]["trace_id"]
    assert splits(target)[trace] == "test"


def test_a_dataset_that_changes_while_it_is_read_is_refused(
    fake: FakePup, api: FakeDatasetsApi, pushed: Path, tmp_path: Path
) -> None:
    listed = fake.handler("llm-obs", "datasets", "records-all")
    assert listed is not None

    def edited_meanwhile(call: Any) -> Any:
        answer = listed(call)
        api.dataset_named(NAME).version += 1  # someone edits the dataset during the read
        return answer

    fake.on("llm-obs", "datasets", "records-all", handler=edited_meanwhile)
    result = fetch(tmp_path / "fetched")
    assert result.exit_code == 1
    assert "changed while its records were read" in result.stderr
    assert not (tmp_path / "fetched" / "rows.jsonl").exists()
