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
    assert result.stderr.startswith("Preview only, nothing was run: push")
    data = output_json(result)
    assert data["status"] == "confirmation_required"
    preview = data["preview"]
    assert preview["project"] == PROJECT
    assert preview["dataset_name"] == NAME
    assert preview["dataset_id"] is None
    assert preview["records_to_push"] == len(ROW_IDS)
    assert sum(preview["rows"].values()) == len(ROW_IDS)
    assert set(preview["rows"]) == set(SPLITS)
    # Each count once: the records to push by split are the preview's rows.
    assert not {"records_to_push_by_split", "kept_by_split"} & set(preview)
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


def test_a_warnings_key_inside_trace_data_is_never_echoed_as_a_warning(
    tmp_path: Path, api: FakeDatasetsApi, fake: FakePup
) -> None:
    # Trace content is untrusted: a `warnings` list in a row's tool schema reaches the
    # preview's sample_record, but stderr must not print it as one of our warnings.
    injected = "INJECTED TEXT from a trace"
    rows = []
    for line in TOOL_USE.read_text().splitlines():
        row = json.loads(line)
        if not row.get("tools"):
            continue
        for tool in row["tools"]:
            tool["function"]["parameters"]["warnings"] = [injected]
        rows.append(json.dumps(row))
    file = tmp_path / "rows.jsonl"
    file.write_text("\n".join(rows) + "\n")
    target = tmp_path / "ds"
    assert invoke("import", str(target), "--file", str(file)).exit_code == 0

    result = push(target)
    assert result.exit_code == 2, result.stderr
    assert injected in result.stdout  # the data is still in the JSON, as data
    assert injected not in result.stderr
    assert "warning:" not in result.stderr


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
    assert data["next_command"] == f"pup finetune dataset sample {dataset} --count 20"
    assert "dataset_url" not in data
    assert len(remote.records) == len(ROW_IDS)


def test_push_previews_read_only_but_refuses_confirm(
    dataset: Path, api: FakeDatasetsApi, fake: FakePup
) -> None:
    refused = push(dataset, "--confirm", env={"PUP_READ_ONLY": "true"})
    assert refused.exit_code == 2
    assert refused.stdout == ""
    assert "PUP_READ_ONLY" in refused.stderr
    assert "pup --read-only" in refused.stderr
    # The preview reads local files only, so it runs, but offers no --confirm.
    preview = push(dataset, env={"PUP_READ_ONLY": "yes"})
    assert preview.exit_code == 2, preview.stderr
    data = output_json(preview)
    assert data["status"] == "confirmation_required"
    assert data["next_command"] is None
    assert data["preview"]["read_only"] == "--confirm is refused while PUP_READ_ONLY is set"
    assert data["preview"]["records_to_push"] > 0
    assert data["preview"]["data_recipients"] == [
        f"kept rows -> Datadog (dataset {NAME!r} in project {PROJECT!r})"
    ]
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
    # Labeled and pushed: training is the step left (no push step any more).
    assert data["next_command"] is None
    assert [step["command"].split()[2] for step in data["next_steps"]] == [
        "models",
        "benchmark",
        "train",
    ]
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


def test_name_is_required_until_the_first_push(dataset: Path) -> None:
    result = invoke("push", str(dataset))
    assert result.exit_code == 2
    assert "Error [usage]: dataset push needs --name (" in _ANSI.sub("", result.stderr)
    assert "was never pushed)" in result.stderr


def test_project_without_name_is_refused(
    dataset: Path, api: FakeDatasetsApi, fake: FakePup
) -> None:
    """Before a first push, --project alone names no dataset; after one, another project
    than the receipt's is refused too."""
    never = invoke("push", str(dataset), "--project", "other")
    assert never.exit_code == 2
    assert "Error [usage]: dataset push needs --name (" in _ANSI.sub("", never.stderr)
    assert fake.calls == []
    assert push(dataset, "--confirm").exit_code == 0
    calls = len(fake.calls)
    other = invoke("push", str(dataset), "--project", "other")
    assert other.exit_code == 2
    assert f"was pushed to dataset {NAME!r} in project {PROJECT!r}" in other.stderr
    assert len(fake.calls) == calls


def test_a_rerun_takes_the_dataset_from_the_receipt(tmp_path: Path, api: FakeDatasetsApi) -> None:
    """Once pushed, `dataset push DIR` needs no flag; another --name names the receipt's."""
    rows = TOOL_USE.read_text().splitlines()
    first, rest = tmp_path / "first.jsonl", tmp_path / "rest.jsonl"
    first.write_text("\n".join(rows[:6]) + "\n")
    rest.write_text("\n".join(rows[6:]) + "\n")
    directory = tmp_path / "ds"
    assert invoke("import", str(directory), "--file", str(first)).exit_code == 0
    pushed = push(directory, "--confirm")
    assert pushed.exit_code == 0, pushed.stderr
    assert output_json(pushed)["pipeline"]["push"] == {
        "status": "pushed",
        "pending": 0,
        "changed": 0,
    }

    imported = output_json(invoke("import", str(directory), "--file", str(rest)))
    new_rows = len(rows) - 6
    assert imported["pipeline"]["push"] == {
        "status": "pushed_with_new_rows",
        "pending": new_rows,
        "changed": 0,
    }
    flagless = f"pup finetune dataset push {directory}"
    assert flagless in [step["command"] for step in imported["next_steps"]]

    result = invoke("push", str(directory))
    assert result.exit_code == 2, result.stderr
    data = output_json(result)
    # The receipt's project, not the default one.
    assert (data["preview"]["project"], data["preview"]["dataset_name"]) == (PROJECT, NAME)
    assert data["preview"]["defaults_used"] == []
    assert data["preview"]["records_to_push"] == new_rows
    words = shlex.split(data["next_command"])
    assert words[:4] == ["pup", "finetune", "dataset", "push"]
    assert "--name" not in words and "--project" not in words
    confirmed = invoke(*words[3:])
    assert confirmed.exit_code == 0, confirmed.stderr
    receipt = output_json(confirmed)
    assert (receipt["pushed"], receipt["records"]) == (new_rows, len(rows))
    assert receipt["pipeline"]["push"] == {"status": "pushed", "pending": 0, "changed": 0}
    assert len(api.dataset_named(NAME).records) == len(rows)

    other = invoke("push", str(directory), "--name", "other")
    assert other.exit_code == 2
    assert f"was pushed to dataset {NAME!r} in project {PROJECT!r}, not 'other'" in other.stderr
    assert "leave --name and --project out" in other.stderr


def test_show_reports_the_push(dataset: Path, api: FakeDatasetsApi) -> None:
    assert output_json(invoke("show", str(dataset)))["push"] is None
    assert push(dataset, "--confirm").exit_code == 0
    shown = output_json(invoke("show", str(dataset)))["push"]
    remote = api.dataset_named(NAME)
    assert shown["dataset_id"] == remote.id
    assert shown["records"] == len(ROW_IDS)
    assert (shown["project"], shown["dataset_name"]) == (PROJECT, NAME)
    assert shown["dataset_version"] == remote.version
