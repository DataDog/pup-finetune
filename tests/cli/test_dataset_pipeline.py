# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``pipeline`` block of the dataset commands, the next steps it leads to, and the
skewed-split warning, over the offline fixture rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.curation.pipeline import pipeline_state
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.push import PUSH_FILE
from tests.fakes.pushed import mark_pushed

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
ROW_IDS = [json.loads(line)["id"] for line in ROWS.read_text().splitlines()]
SKEW = (
    "validation has 5 of 14 rows (36%, 4 of 13 traces) though --validation-fraction is 0.1: "
    "splits are assigned per trace and request, so small datasets vary; import more rows"
)

runner = CliRunner()


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["dataset", *args])


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def imported() -> Result:
    result = invoke("import", "ds", "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return result


def test_import_reports_the_pipeline_and_warns_about_the_skewed_split(imported: Result) -> None:
    data = output(imported)
    assert data["pipeline"] == {
        "rows": {"accepted": 14, "kept": 14, "dropped": 0, "unlabeled": 14},
        "triage": "none",
        "push": {"status": "not_pushed", "pending": 14, "changed": 0},
    }
    assert data["splits"] == {"train": 7, "validation": 5, "test": 2}
    assert SKEW in data["warnings"]
    assert f"warning: {SKEW}" in imported.stderr.splitlines()
    assert data["next_command"] == "pup finetune dataset sample ds --count 20"


@pytest.mark.usefixtures("imported")
def test_show_reports_the_same_pipeline_and_the_warning() -> None:
    result = invoke("show", "ds")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["pipeline"]["rows"]["unlabeled"] == 14
    assert SKEW in data["summary"]["warnings"]
    assert f"warning: {SKEW}" in result.stderr.splitlines()


@pytest.mark.usefixtures("imported")
def test_labels_move_the_pipeline_on_to_a_first_push_then_training(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"id": ROW_IDS[0], "keep": False, "reason": "synthetic drop"})
        + "\n"
        + json.dumps({"id": ROW_IDS[1], "keep": True})
        + "\n"
    )
    result = invoke("label", "ds", "--file", str(labels))
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["pipeline"] == {
        "rows": {"accepted": 14, "kept": 13, "dropped": 1, "unlabeled": 12},
        "triage": "none",
        "push": {"status": "not_pushed", "pending": 13, "changed": 0},
    }
    assert data["next_command"] is None
    assert [step["command"] for step in data["next_steps"]] == [
        "pup finetune models list",
        "pup finetune dataset push ds --name <NAME>",
    ]
    mark_pushed(Path("ds"))
    shown = output(invoke("show", "ds"))
    assert [step["command"] for step in shown["next_steps"]] == [
        "pup finetune models list",
        "pup finetune benchmark ds --models <MODELS>",
        "pup finetune train --provider <PROVIDER> --dataset ds --model <MODEL>",
    ]


@pytest.mark.usefixtures("imported")
def test_with_every_row_dropped_there_is_nothing_to_push_or_train_on(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"id": row_id, "keep": False, "reason": "synthetic drop"}) + "\n"
            for row_id in ROW_IDS
        )
    )
    data = output(invoke("label", "ds", "--file", str(labels)))
    assert data["pipeline"]["rows"]["kept"] == 0
    commands = [step["command"] for step in data["next_steps"]]
    assert not any(" push " in c or " train " in c or " benchmark " in c for c in commands)


@pytest.mark.usefixtures("imported")
def test_an_unreadable_receipt_is_reported_not_guessed() -> None:
    (Path("ds") / PUSH_FILE).write_text("{not json")
    state = pipeline_state(DatasetDir(Path("ds")))
    assert (state.push, state.pending) == ("unreadable", None)
    result = invoke("sample", "ds", "--count", "1")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["pipeline"]["push"] == {"status": "unreadable", "pending": None, "changed": None}
    assert "pup finetune dataset show ds" in [step["command"] for step in data["next_steps"]]
    # The warning names the file and the problem.
    [warning] = [w for w in data["warnings"] if PUSH_FILE in w]
    assert "push receipt cannot be read (ds/push.json:1: invalid JSON" in warning


@pytest.mark.usefixtures("imported")
@pytest.mark.parametrize(
    ("name", "content", "problem"),
    [
        ("judgments.jsonl", "{not json\n", "ds/judgments.jsonl:1: "),
        ("triage.json", "{not json", "ds/triage.json:1: invalid JSON"),
        ("triage.json", '{"judges": []}', "ds/triage.json: not the settings of a triage"),
    ],
)
def test_unreadable_triage_files_are_reported_not_taken_as_incomplete(
    name: str, content: str, problem: str
) -> None:
    (Path("ds") / name).write_text(content)
    state = pipeline_state(DatasetDir(Path("ds")))
    assert state.triage == "unreadable"
    result = invoke("show", "ds")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["pipeline"]["triage"] == "unreadable"
    [warning] = [w for w in data["warnings"] if "triage files cannot be read" in w]
    assert problem in warning and "--reset-triage" in warning
    assert f"warning: {warning}" in result.stderr.splitlines()
    assert "pup finetune dataset triage ds --rubric <RUBRIC_FILE> --reset-triage" in [
        step["command"] for step in data["next_steps"]
    ]


def test_an_empty_dataset_has_nothing_next(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    result = invoke("import", "ds", "--file", str(empty))
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["pipeline"]["rows"] == {"accepted": 0, "kept": 0, "dropped": 0, "unlabeled": 0}
    assert (data["next_command"], data["next_steps"]) == (None, [])
