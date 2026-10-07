# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset label` and `dataset sample`, end to end through the Typer app."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.data.dataset_dir import DatasetDir

ROWS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rows"
TOOL_USE = ROWS_DIR / "tool_use.jsonl"
FIXTURE_IDS = sorted(json.loads(line)["id"] for line in TOOL_USE.read_text().splitlines())

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["dataset", *args])


def output_json(result: Result) -> dict[str, Any]:
    assert result.exit_code == 0, result.stderr
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    target = tmp_path / "ds"
    result = invoke("import", str(target), "--file", str(TOOL_USE))
    assert result.exit_code == 0, result.stderr
    return target


def label_file(tmp_path: Path, *records: dict[str, Any], name: str = "labels.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def label(directory: Path, file: Path, *extra: str) -> Result:
    return invoke("label", str(directory), "--file", str(file), *extra)


# --- label ------------------------------------------------------------------------------


def test_label_applies_labels_and_emits_the_report(dataset: Path, tmp_path: Path) -> None:
    file = label_file(
        tmp_path,
        {"id": FIXTURE_IDS[0], "keep": False, "reason": "wrong tool"},
        {"id": FIXTURE_IDS[1], "keep": True, "reason": ""},
    )
    data = output_json(label(dataset, file))
    assert data["dataset_dir"] == str(dataset.resolve())
    assert data["label"] == {
        "file": str(file),
        "labels": 2,
        "applied": 2,
        "unchanged": 0,
        "kept": 1,
        "dropped": 1,
    }
    assert (data["accepted"], data["kept"], data["dropped"]) == (len(FIXTURE_IDS), 13, 1)
    assert data["drop_reasons"] == [{"reason": "wrong tool", "count": 1}]
    assert sum(data["splits"].values()) == 13
    # Training needs a provider, a model and a run directory nobody chose yet: steps only.
    quoted = shlex.quote(str(dataset))
    assert data["next_command"] is None
    assert [step["command"] for step in data["next_steps"]] == [
        "pup finetune models list --provider <PROVIDER>",
        f"pup finetune train --provider <PROVIDER> --model <MODEL> --dataset {quoted} "
        "--run-dir <RUN_DIR>",
        f"pup finetune dataset push {quoted} --name <NAME>",
    ]
    assert FIXTURE_IDS[0] not in {row.id for row in DatasetDir(dataset).kept_rows()}
    show = output_json(invoke("show", str(dataset)))
    assert show["summary"]["dropped"] == 1


def test_labels_are_manual_and_at_is_filled_in(dataset: Path, tmp_path: Path) -> None:
    file = label_file(
        tmp_path,
        {"id": FIXTURE_IDS[0], "keep": True},
        {"id": FIXTURE_IDS[1], "keep": True, "at": "2026-01-02T03:04:05+01:00"},
    )
    output_json(label(dataset, file))
    records = [json.loads(line) for line in (dataset / "labels.jsonl").read_text().splitlines()]
    assert [record["by"] for record in records] == ["manual", "manual"]
    assert records[0]["at"].endswith("Z")
    assert records[1]["at"] == "2026-01-02T02:04:05Z"


@pytest.mark.parametrize("old", ["agent", "human"])
def test_a_label_file_from_before_says_agent_or_human_and_reads_as_manual(
    dataset: Path, tmp_path: Path, old: str
) -> None:
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True, "by": old})
    output_json(label(dataset, file))
    [record] = [json.loads(line) for line in (dataset / "labels.jsonl").read_text().splitlines()]
    assert record["by"] == "manual"


def test_a_label_file_cannot_claim_the_judge_council(dataset: Path, tmp_path: Path) -> None:
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True, "by": "judge"})
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "judge" in result.stderr
    assert not (dataset / "labels.jsonl").exists()


def test_label_has_no_by_option(dataset: Path, tmp_path: Path) -> None:
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True})
    result = label(dataset, file, "--by", "human")
    assert result.exit_code == 2


def test_label_last_label_wins_across_runs(dataset: Path, tmp_path: Path) -> None:
    row_id = FIXTURE_IDS[0]
    drop = label_file(tmp_path, {"id": row_id, "keep": False, "reason": "bad"}, name="a.jsonl")
    keep = label_file(tmp_path, {"id": row_id, "keep": True}, name="b.jsonl")
    assert output_json(label(dataset, drop))["dropped"] == 1
    assert output_json(label(dataset, keep))["dropped"] == 0
    again = output_json(label(dataset, keep))
    assert (again["label"]["applied"], again["label"]["unchanged"]) == (0, 1)


def test_label_rejects_unknown_ids(dataset: Path, tmp_path: Path) -> None:
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True}, {"id": "nope", "keep": True})
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "1 labeled ids are not accepted rows of the dataset: 'nope'" in result.stderr
    assert result.stdout == ""
    assert not (dataset / "labels.jsonl").exists()


def test_label_requires_a_reason_to_drop(dataset: Path, tmp_path: Path) -> None:
    file = label_file(
        tmp_path, {"id": FIXTURE_IDS[0], "keep": True}, {"id": FIXTURE_IDS[1], "keep": False}
    )
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "labels.jsonl:2: invalid label: a reason is required when keep is false" in (
        result.stderr
    )
    assert not (dataset / "labels.jsonl").exists()


def test_label_rejects_a_lone_surrogate_in_a_reason(dataset: Path, tmp_path: Path) -> None:
    keep = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True}, name="keep.jsonl")
    output_json(label(dataset, keep))
    before = (dataset / "labels.jsonl").read_bytes()
    file = label_file(
        tmp_path,
        {"id": FIXTURE_IDS[1], "keep": True},
        {"id": FIXTURE_IDS[2], "keep": False, "reason": "bad \ud800"},
    )
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "labels.jsonl:2: invalid label: reason: has a lone surrogate (U+D800)" in result.stderr
    assert (dataset / "labels.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ("not json", "invalid JSON"),
        ('["a"]', "expected a JSON object"),
        ('{"id": "x", "keep": true, "extra": 1}', "extra"),
        ('{"keep": true}', "id"),
    ],
)
def test_label_reports_bad_lines(dataset: Path, tmp_path: Path, line: str, problem: str) -> None:
    file = tmp_path / "labels.jsonl"
    file.write_text("\n" + line + "\n")
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "labels.jsonl:2: " in result.stderr
    assert problem in result.stderr


def test_label_rejects_an_empty_file(dataset: Path, tmp_path: Path) -> None:
    file = tmp_path / "labels.jsonl"
    file.write_text("\n")
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "has no labels" in result.stderr


def test_label_needs_a_dataset_directory(tmp_path: Path) -> None:
    file = label_file(tmp_path, {"id": "a", "keep": True})
    result = label(tmp_path / "missing", file)
    assert result.exit_code == 2
    assert "is not a dataset directory" in result.stderr


# --- sample -----------------------------------------------------------------------------


def sample(directory: Path, *extra: str) -> dict[str, Any]:
    return output_json(invoke("sample", str(directory), *extra))


def test_sample_shows_rows_in_brief(dataset: Path) -> None:
    data = sample(dataset, "--n", "3")
    assert data["dataset_dir"] == str(dataset.resolve())
    assert (data["split"], data["seed"], data["eligible"]) == ("all", 0, len(FIXTURE_IDS))
    assert len(data["rows"]) == 3
    for row in data["rows"]:
        assert set(row) == {
            "id",
            "split",
            "label",
            "turns",
            "tools_used",
            "first_user_message",
            "final_assistant_message",
        }
        assert row["label"] == "none"
    assert data["rows_file"] == str((dataset / "rows.jsonl").resolve())
    # Labels come from a file the reader writes, so labeling is a step, not next_command.
    assert data["next_command"] is None
    assert data["next_steps"][0]["command"] == (
        f"pup finetune dataset label {shlex.quote(str(dataset))} --file <LABELS_FILE>"
    )


def test_sample_is_deterministic_for_a_seed(dataset: Path) -> None:
    first = sample(dataset, "--n", "4", "--seed", "11")
    assert sample(dataset, "--n", "4", "--seed", "11") == first
    assert (
        sample(dataset, "--n", "100")["rows"]
        == sample(dataset, "--n", "100", "--seed", "5")["rows"]
    )


def test_sample_of_one_split(dataset: Path) -> None:
    train = {row.id for row in DatasetDir(dataset).rows_for_split("train")}
    data = sample(dataset, "--split", "train", "--n", "100")
    assert {row["id"] for row in data["rows"]} == train
    assert {row["split"] for row in data["rows"]} == {"train"}


def test_sample_hides_dropped_rows_unless_asked(dataset: Path, tmp_path: Path) -> None:
    dropped = FIXTURE_IDS[0]
    output_json(label(dataset, label_file(tmp_path, {"id": dropped, "keep": False, "reason": "x"})))
    shown = {row["id"] for row in sample(dataset, "--n", "100")["rows"]}
    assert dropped not in shown
    everything = sample(dataset, "--n", "100", "--include-dropped")["rows"]
    assert {row["id"]: row["label"] for row in everything}[dropped] == "drop"


def test_sample_renders_tool_calls_of_a_final_message(tmp_path: Path) -> None:
    row = {
        "id": "tool-final",
        "messages": [
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "lookup", "arguments": '{"key": "k1"}'}}
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row) + "\n")
    target = tmp_path / "ds"
    assert invoke("import", str(target), "--file", str(rows)).exit_code == 0
    [described] = sample(target)["rows"]
    assert described["final_assistant_message"] == 'lookup({"key": "k1"})'
    assert described["tools_used"] == ["lookup"]


def test_sample_rejects_an_unknown_split(dataset: Path) -> None:
    result = invoke("sample", str(dataset), "--split", "holdout")
    assert result.exit_code == 2
    assert "--split" in _ANSI.sub("", result.stderr)


def test_sample_needs_a_dataset_directory(tmp_path: Path) -> None:
    result = invoke("sample", str(tmp_path / "missing"))
    assert result.exit_code == 2
    assert "is not a dataset directory" in result.stderr


# --- read-only mode ---------------------------------------------------------------------


def test_label_and_sample_are_allowed_read_only(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": False, "reason": "noise"})
    assert output_json(label(dataset, file))["dropped"] == 1
    assert sample(dataset, "--include-dropped")["eligible"] == len(FIXTURE_IDS)
