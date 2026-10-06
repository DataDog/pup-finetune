# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Rendering a dataset for a run, and rendering it again for a resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings
from pup_finetune.data.schema import TrainingRow, validate_row
from pup_finetune.data.split import SplitName, SplitParams
from pup_finetune.errors import StateError
from pup_finetune.providers.base import TrainSpec
from pup_finetune.providers.fake import FAKE_PROFILE, FakeRenderer
from pup_finetune.runs import RunDir, dataset_content_hash, rows_content_hash
from pup_finetune.runtime import Runtime
from pup_finetune.training import inputs as inputs_module
from pup_finetune.training.inputs import build_inputs, load_run_inputs

SETTINGS = DatasetSettings("import", SplitParams(42, 0.0, 0.0))  # every row trains


def _row(row_id: str) -> TrainingRow:
    return validate_row(
        {
            "id": row_id,
            "messages": [
                {"role": "user", "content": f"question {row_id}"},
                {"role": "assistant", "content": f"answer {row_id}"},
            ],
        }
    )


def _dataset(path: Path, ids: list[str]) -> DatasetDir:
    dataset = DatasetDir(path)
    dataset.add([_row(row_id) for row_id in ids], settings=SETTINGS)
    return dataset


def _train_ids(dataset: DatasetDir) -> list[str]:
    inputs = build_inputs(dataset, FakeRenderer(), FAKE_PROFILE)
    return [row.row_id for row in inputs.train_rows]


def test_rows_render_in_id_order_whatever_their_order_on_disk(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "ds", ["c", "a", "b"])
    before = dataset_content_hash(dataset)
    assert _train_ids(dataset) == ["a", "b", "c"]
    lines = dataset.rows_path.read_text().splitlines(keepends=True)
    dataset.rows_path.write_text("".join(reversed(lines)))
    # The same rows in another order: the same identity, and the same training order.
    assert dataset_content_hash(dataset) == before
    assert _train_ids(dataset) == ["a", "b", "c"]
    assert build_inputs(dataset, FakeRenderer(), FAKE_PROFILE).dataset_sha256 == before


def _two_turn_row(row_id: str, weights: tuple[int | None, int | None]) -> TrainingRow:
    first, second = (
        {"role": "assistant", "content": f"{part} {row_id}"}
        | ({} if weight is None else {"weight": weight})
        for part, weight in zip(("answer", "more"), weights, strict=True)
    )
    return validate_row(
        {
            "id": row_id,
            "messages": [
                {"role": "user", "content": f"question {row_id}"},
                first,
                {"role": "user", "content": f"and then {row_id}?"},
                second,
            ],
        }
    )


def test_the_run_hash_covers_loss_weights_as_trained() -> None:
    def run_hash(weights: tuple[int | None, int | None]) -> str:
        rows = [_two_turn_row("a", weights)]
        return rows_content_hash({"train": rows, "validation": [], "test": []})

    # An unset weight trains like 1, so it is the same run; a 0 trains on less.
    assert run_hash((None, None)) == run_hash((1, 1)) == run_hash((1, None))
    assert run_hash((0, None)) != run_hash((None, None))
    assert run_hash((0, None)) != run_hash((None, 0))


def _tool_row(call_id: str) -> TrainingRow:
    call = {"id": call_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}
    return validate_row(
        {
            "id": "a",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "content": "result", "tool_call_id": call_id},
                {"role": "assistant", "content": "answer"},
            ],
        }
    )


def test_the_run_hash_covers_the_stored_tool_call_ids() -> None:
    # The content hash names calls by position, but a chat template may render the ids.
    def run_hash(call_id: str) -> str:
        return rows_content_hash({"train": [_tool_row(call_id)], "validation": [], "test": []})

    assert run_hash("call_1") == run_hash("call_1")
    assert run_hash("call_1") != run_hash("call_2")


def test_a_resume_refuses_a_dataset_whose_loss_weights_changed(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([_two_turn_row("a", (None, None)), _row("b")], settings=SETTINGS)
    run_dir = RunDir.create(
        tmp_path / "run",
        provider="fake",
        profile=FAKE_PROFILE,
        spec=TrainSpec(),
        dataset_dir=dataset.path,
        dataset_sha256=dataset_content_hash(dataset),
        rendering=FakeRenderer().identity(),
    )
    lines = dataset.rows_path.read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    [edited] = [row for row in rows if row["id"] == "a"]
    edited["messages"][1]["weight"] = 0  # the conversation, and so the dedupe hash, is the same
    dataset.rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(StateError, match="changed since the run"):
        load_run_inputs(run_dir, Runtime(dev_mode=True))


class GrowingDataset(DatasetDir):
    """A dataset another process imports into after every read of its rows."""

    def rows_by_split(self) -> dict[SplitName, list[TrainingRow]]:
        rows = super().rows_by_split()
        self.add([_row(f"late-{len(self.rows())}")], settings=SETTINGS)
        return rows


def test_the_new_run_hash_describes_the_rows_rendered(tmp_path: Path) -> None:
    _dataset(tmp_path / "ds", ["a", "b"])
    dataset = GrowingDataset(tmp_path / "ds")
    inputs = build_inputs(dataset, FakeRenderer(), FAKE_PROFILE)
    rendered: dict[SplitName, list[TrainingRow]] = {
        "train": [_row("a"), _row("b")],
        "validation": [],
        "test": [],
    }
    assert [row.row_id for row in inputs.train_rows] == ["a", "b"]
    assert inputs.dataset_sha256 == rows_content_hash(rendered)


def test_a_resume_renders_the_rows_it_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path / "ds", ["a", "b"])
    run_dir = RunDir.create(
        tmp_path / "run",
        provider="fake",
        profile=FAKE_PROFILE,
        spec=TrainSpec(),
        dataset_dir=dataset.path,
        dataset_sha256=dataset_content_hash(dataset),
        rendering=FakeRenderer().identity(),
    )
    # Rows imported right after the check must not reach the rendered inputs.
    monkeypatch.setattr(inputs_module, "DatasetDir", GrowingDataset)
    _, inputs = load_run_inputs(run_dir, Runtime(dev_mode=True))
    assert [row.row_id for row in inputs.train_rows] == ["a", "b"]
    assert inputs.dataset_sha256 == run_dir.manifest()["dataset_sha256"]
