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
from pup_finetune.curation.pipeline import require_in_datadog
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.push import PUSH_FILE
from pup_finetune.errors import DirectoryUsageError
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
    return invoke(
        "train", "--provider", "fireworks", "--dataset", str(dataset), "--model", "qwen3p8-27b"
    )


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
    assert "1 of its kept rows is not in dataset 'synthetic-sft' yet" in result.stderr
    assert (
        "push the 1 kept row not pushed yet to dataset 'synthetic-sft', which push.json "
        "names (a remote write; it previews first): `pup finetune dataset push ds`"
    ) in result.stderr


def edit_rows(dataset: Path, count: int) -> list[str]:
    """Edit the answer of the first ``count`` rows after the push; their ids, sorted."""
    rows = [json.loads(line) for line in (dataset / "rows.jsonl").read_text().splitlines()]
    for row in rows[:count]:
        row["messages"][-1]["content"] = "An answer edited after the push."
    (dataset / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return sorted(row["id"] for row in rows[:count])


def test_train_refuses_kept_rows_changed_after_the_push_and_names_them(dataset: Path) -> None:
    mark_pushed(dataset)
    (edited,) = edit_rows(dataset, 1)
    result = train(dataset)
    assert result.exit_code == 2
    assert (
        f"1 of its kept rows changed after the push to dataset 'synthetic-sft' (row {edited})"
        in result.stderr
    )
    fetch = "`pup finetune dataset fetch <NEW_DIR> --project pup-finetune-dev --name synthetic-sft`"
    assert fetch in result.stderr
    assert "to train on what Datadog holds" in result.stderr
    assert "--name <NEW_NAME>" not in result.stderr
    assert "--project <PROJECT>" not in result.stderr
    assert "brings back" not in result.stderr
    show = json.loads(invoke("dataset", "show", str(dataset)).stdout)
    assert show["pipeline"]["push"] == {
        "status": "pushed_with_changed_rows",
        "pending": 0,
        "changed": 1,
    }
    assert fetch.strip("`") in [step["command"] for step in show["next_steps"]]
    assert not any(step["command"].startswith("pup finetune train") for step in show["next_steps"])


def test_the_refusal_names_a_few_of_many_changed_rows(dataset: Path) -> None:
    mark_pushed(dataset)
    edited = edit_rows(dataset, 7)
    result = invoke("benchmark", str(dataset), "--models", "fireworks/qwen3p8-27b")
    assert result.exit_code == 2
    shown = ", ".join(edited[:5])
    assert (
        "7 of its kept rows changed after the push to dataset 'synthetic-sft' "
        f"(rows {shown} and 2 more)"
    ) in result.stderr
    assert "fetch it into a new directory to benchmark on what Datadog holds" in result.stderr
    assert "train on" not in result.stderr


def all_row_ids(dataset: Path) -> list[str]:
    return [json.loads(line)["id"] for line in (dataset / "rows.jsonl").read_text().splitlines()]


def drop(dataset: Path, tmp_path: Path, row_ids: list[str]) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps({"id": i, "keep": False, "reason": "synthetic"}) + "\n" for i in row_ids)
    )
    result = invoke("dataset", "label", str(dataset), "--file", str(labels))
    assert result.exit_code == 0, result.stderr


def test_the_fetch_for_changed_rows_says_it_brings_back_rows_dropped_since_the_push(
    dataset: Path, tmp_path: Path
) -> None:
    mark_pushed(dataset)
    edited = edit_rows(dataset, 1)
    dropped = sorted(row for row in all_row_ids(dataset) if row not in edited)[:2]
    drop(dataset, tmp_path, dropped)
    result = train(dataset)
    assert result.exit_code == 2
    assert (
        "fetch it into a new directory to train on what Datadog holds; the fetch also brings "
        f"back the 2 rows dropped or retired since the push (rows {dropped[0]}, {dropped[1]}), "
        "which you drop again with `dataset label` after the fetch"
    ) in " ".join(result.stderr.split())


def test_the_fetch_for_changed_rows_names_a_few_of_many_rows_dropped_since_the_push(
    dataset: Path, tmp_path: Path
) -> None:
    mark_pushed(dataset)
    edited = edit_rows(dataset, 1)
    dropped = sorted(row for row in all_row_ids(dataset) if row not in edited)[:7]
    drop(dataset, tmp_path, dropped)
    result = train(dataset)
    assert result.exit_code == 2
    shown = ", ".join(dropped[:5])
    assert (
        f"brings back the 7 rows dropped or retired since the push (rows {shown} and 2 more)"
    ) in " ".join(result.stderr.split())


def test_the_check_counts_the_rows_the_command_reads_not_a_later_state(dataset: Path) -> None:
    store = DatasetDir(dataset)
    by_split = store.rows_by_split()
    rows = [row.id for split in by_split.values() for row in split]
    drop(dataset, dataset.parent, rows)
    assert store.summary()["kept"] == 0
    with pytest.raises(DirectoryUsageError, match="ds was never pushed"):
        require_in_datadog(store, by_split, "train", directory=dataset, offline=False)


def test_train_refuses_a_push_still_in_flight(dataset: Path) -> None:
    mark_pushed(dataset)
    receipt = json.loads((dataset / PUSH_FILE).read_text())
    receipt["in_flight"] = [next(iter(receipt["record_ids"]))]
    (dataset / PUSH_FILE).write_text(json.dumps(receipt))
    result = train(dataset)
    assert result.exit_code == 2
    assert "its last push to dataset 'synthetic-sft' did not finish" in result.stderr
    assert "finish the interrupted push to dataset 'synthetic-sft'" in result.stderr


def test_train_refuses_an_unreadable_push_receipt(dataset: Path) -> None:
    (dataset / PUSH_FILE).write_text("{not json")
    result = train(dataset)
    assert result.exit_code == 2
    assert f"whether ds was pushed is unknown: {PUSH_FILE} cannot be read" in result.stderr
    assert "`pup finetune dataset show ds`" in result.stderr


def test_dropped_rows_need_no_push(dataset: Path, tmp_path: Path) -> None:
    mark_pushed(dataset)
    row_id = json.loads((dataset / "rows.jsonl").read_text().splitlines()[0])["id"]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": row_id, "keep": False, "reason": "synthetic"}) + "\n")
    assert invoke("dataset", "label", str(dataset), "--file", str(labels)).exit_code == 0
    result = train(dataset)
    assert "reads a dataset that is in Datadog" not in result.stderr


def test_benchmark_on_a_real_model_refuses_a_dataset_never_pushed(dataset: Path) -> None:
    result = invoke("benchmark", str(dataset), "--models", "fireworks/qwen3p8-27b")
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
