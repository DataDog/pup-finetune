# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Pushing a dataset directory's kept rows to a (fake) Agent Observability dataset."""

from __future__ import annotations

import json
import os
import signal
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.curation.labels import apply_labels
from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings
from pup_finetune.data.jsonl import read_rows
from pup_finetune.data.labels import Label
from pup_finetune.data.push import (
    IN_FLIGHT_SETTLE,
    PUSH_FILE,
    SOURCE_TAG,
    PushTerminated,
    dataset_description,
    dataset_record,
    plan_push,
    push_dataset,
    read_push_state,
)
from pup_finetune.data.schema import TrainingRow, validate_row
from pup_finetune.data.split import SPLITS
from pup_finetune.datadog.datasets import RECORDS_BATCH_SIZE
from pup_finetune.errors import ExternalError, LockBusy, StateError, UsageError
from pup_finetune.storage import DirectoryLock
from tests.fakes.datasets import FakeDatasetsApi
from tests.fakes.pup import FakePup, PupCall

ROWS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rows"
TOOL_USE = ROWS_DIR / "tool_use.jsonl"
ALL_ROWS = list(read_rows(TOOL_USE))
FIRST = 10
"""Rows in the dataset at first; the rest are added later."""
PROJECT = "pup-finetune-dev"
NAME = "synthetic-weather-and-more"
IMPORT = DatasetSettings("import")
SENT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
"""When a :class:`FakeClock` starts."""


class FakeClock:
    """A clock that stands still until moved on."""

    def __init__(self) -> None:
        self.now = SENT

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@pytest.fixture
def api() -> FakeDatasetsApi:
    return FakeDatasetsApi()


@pytest.fixture
def fake(api: FakeDatasetsApi) -> FakePup:
    return api.install(FakePup())


@pytest.fixture
def dataset(tmp_path: Path) -> DatasetDir:
    target = DatasetDir(tmp_path / "ds")
    target.add(ALL_ROWS[:FIRST], settings=IMPORT)
    return target


def push(
    dataset: DatasetDir, fake: FakePup, *, name: str = NAME, project: str = PROJECT, **extra: Any
) -> Any:
    extra.setdefault("batch_size", 4)
    return push_dataset(dataset, project=project, name=name, pup=fake.client(), **extra)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def drop(dataset: DatasetDir, *row_ids: str) -> None:
    apply_labels(dataset, [Label(id=row_id, keep=False, reason="off-topic") for row_id in row_ids])


def records_add_calls(fake: FakePup) -> int:
    return sum(1 for call in fake.calls if call.args[:3] == ("llm-obs", "datasets", "records-add"))


def push_order(dataset: DatasetDir) -> list[str]:
    """The kept row ids in the order a first push sends them: by split, then as added."""
    by_split = dataset.rows_by_split()
    return [row.id for split in SPLITS for row in by_split[split]]


def remote_row_ids(api: FakeDatasetsApi) -> list[str]:
    return [record["metadata"]["row_id"] for record in api.dataset_named(NAME).records]


def synthetic_rows(count: int) -> list[TrainingRow]:
    return [
        validate_row(
            {
                "id": f"row-synthetic-{index:04d}",
                "messages": [
                    {"role": "user", "content": f"Synthetic question {index}?"},
                    {"role": "assistant", "content": f"Synthetic answer {index}."},
                ],
                "source": {"trace_id": f"synthetic-trace-{index:04d}"},
            }
        )
        for index in range(count)
    ]


# --- records --------------------------------------------------------------------------------


def test_a_record_answers_the_final_assistant_turn() -> None:
    row = validate_row(
        {
            "id": "row-1",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather in Northvale?"},
                {
                    "role": "assistant",
                    "content": "",
                    "weight": 0,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "N"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "rain"},
                {"role": "assistant", "content": "Rain.", "reasoning_content": "It said rain."},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "source": {
                "trace_id": "synthetic-trace-1",
                "span_id": "synthetic-span-1",
                "ml_app": "synthetic-app",
                "model_name": "example-model",
            },
        }
    )
    record = dataset_record(row, "validation")
    assert record == {
        "input": {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather in Northvale?"},
                {
                    "role": "assistant",
                    "content": "",
                    "weight": 0,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "N"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "rain"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "description": "", "parameters": {}},
                }
            ],
        },
        "expected_output": {
            "role": "assistant",
            "content": "Rain.",
            "reasoning_content": "It said rain.",
        },
        "metadata": {
            "row_id": "row-1",
            "split": "validation",
            "trace_id": "synthetic-trace-1",
            "span_id": "synthetic-span-1",
            "ml_app": "synthetic-app",
            "source_model": "example-model",
            "schema_version": 1,
        },
        "tags": ["split:validation", SOURCE_TAG],
    }


def test_a_record_of_a_row_without_tools_or_source() -> None:
    row = TrainingRow.model_validate(
        {
            "id": "row-2",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
    )
    record = dataset_record(row, "train")
    assert record["input"] == {"messages": [{"role": "user", "content": "hi"}], "tools": []}
    assert record["metadata"]["trace_id"] is None
    assert record["tags"] == ["split:train", SOURCE_TAG]


# --- pushing --------------------------------------------------------------------------------


def test_push_creates_the_project_and_dataset_and_adds_records_in_batches(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    receipt = push(dataset, fake)
    remote = api.dataset_named(NAME)
    assert api.projects == {remote.project_id: PROJECT}
    assert records_add_calls(fake) == 3  # 10 rows in batches of 4
    assert remote_row_ids(api) == push_order(dataset)
    for record in remote.records:
        assert record["tags"] == [f"split:{record['metadata']['split']}", SOURCE_TAG]

    state = read_push_state(dataset)
    assert state is not None
    assert state == receipt.state
    assert (state.project, state.project_id) == (PROJECT, remote.project_id)
    assert (state.dataset_id, state.dataset_name) == (remote.id, NAME)
    assert state.dataset_version == remote.version == 3
    assert state.record_ids == dict(zip(push_order(dataset), remote.record_ids, strict=True))
    assert state.in_flight == ()
    assert dataset_description(state.push_id) == remote.description
    assert state.pushed_at is not None
    assert (receipt.pushed, receipt.already_pushed, receipt.stale_records) == (FIRST, 0, {})
    by_split = dataset.rows_by_split()
    assert receipt.records_by_split == {split: len(by_split[split]) for split in SPLITS}
    assert sum(receipt.records_by_split.values()) == FIRST


def test_push_json_is_private_and_holds_the_receipt(dataset: DatasetDir, fake: FakePup) -> None:
    push(dataset, fake)
    path = dataset.path / PUSH_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = json.loads(path.read_text())
    assert set(data) == {
        "project",
        "project_id",
        "dataset_id",
        "dataset_name",
        "dataset_version",
        "push_id",
        "record_ids",
        "in_flight",
        "in_flight_at",
        "pushed_at",
        "settings_hash",
    }
    assert data["in_flight"] == [] and data["in_flight_at"] is None


def test_records_split_tags_match_the_dataset_splits(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    push(dataset, fake)
    by_split = dataset.rows_by_split()
    expected = {row.id: split for split in SPLITS for row in by_split[split]}
    remote = api.dataset_named(NAME)
    assert {r["metadata"]["row_id"]: r["metadata"]["split"] for r in remote.records} == expected


def test_push_reuses_an_existing_project(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    project_id = api.add_project(PROJECT)
    push(dataset, fake)
    assert api.projects == {project_id: PROJECT}
    assert not any(call.args[:3] == ("llm-obs", "projects", "create") for call in fake.calls)


def test_pushing_an_unchanged_dataset_again_is_a_no_op(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    first = push(dataset, fake)
    calls = len(fake.calls)
    again = push(dataset, fake)
    assert len(fake.calls) == calls  # not even a lookup
    assert (again.pushed, again.already_pushed) == (0, FIRST)
    assert again.state == first.state
    assert len(api.dataset_named(NAME).records) == FIRST


def test_rows_added_after_a_push_are_appended(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    push(dataset, fake)
    dataset.add(ALL_ROWS, settings=IMPORT)
    receipt = push(dataset, fake)
    remote = api.dataset_named(NAME)
    assert (receipt.pushed, receipt.already_pushed) == (len(ALL_ROWS) - FIRST, FIRST)
    assert sorted(remote_row_ids(api)) == sorted(row.id for row in ALL_ROWS)
    assert len(api.datasets) == 1
    state = read_push_state(dataset)
    assert state is not None
    assert set(state.record_ids) == {row.id for row in ALL_ROWS}
    assert state.dataset_version == remote.version


def test_an_interrupted_push_resumes_without_repeating_records(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi, clock: FakeClock
) -> None:
    api.records_add_failures = {2}
    order = push_order(dataset)
    with pytest.raises(ExternalError, match="502") as caught:
        push(dataset, fake, clock=clock)
    remote = api.dataset_named(NAME)
    assert f"a batch of 4 records to dataset {remote.id}" in caught.value.message
    assert "in_flight" in (caught.value.hint or "")
    state = read_push_state(dataset)
    assert state is not None
    assert set(state.record_ids) == set(order[:4])  # the first batch
    assert state.in_flight == tuple(order[4:8])
    assert state.in_flight_at == SENT
    clock.advance(IN_FLIGHT_SETTLE)
    receipt = push(dataset, fake, clock=clock)
    assert (receipt.pushed, receipt.already_pushed) == (FIRST - 4, 4)
    assert remote_row_ids(api) == order
    assert len(set(remote.record_ids)) == FIRST
    assert receipt.state.record_ids == dict(zip(order, remote.record_ids, strict=True))
    assert receipt.state.in_flight == ()
    assert receipt.state.in_flight_at is None


def test_missing_rows_of_a_batch_are_not_sent_again_until_it_has_settled(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi, clock: FakeClock
) -> None:
    # The call failed, but the server may still be adding the batch: wait before resending.
    api.records_add_failures = {2}
    order = push_order(dataset)
    with pytest.raises(ExternalError):
        push(dataset, fake, clock=clock)
    clock.advance(IN_FLIGHT_SETTLE - timedelta(seconds=1))
    with pytest.raises(StateError, match="may still be being added") as caught:
        push(dataset, fake, clock=clock)
    assert "4 of the 4 records" in caught.value.message
    assert "at 03:04:05 UTC" in caught.value.message
    assert all(row_id in caught.value.message for row_id in order[4:8])
    assert caught.value.hint is not None
    assert caught.value.hint.startswith("retry after 03:15 UTC")  # 03:14:05, rounded up
    assert records_add_calls(fake) == 2  # nothing sent again
    state = read_push_state(dataset)
    assert state is not None
    assert (state.in_flight, state.in_flight_at) == (tuple(order[4:8]), SENT)
    clock.advance(timedelta(seconds=1))
    receipt = push(dataset, fake, clock=clock)
    assert remote_row_ids(api) == order
    assert receipt.state.in_flight == ()


def test_rows_of_an_unsettled_batch_found_are_adopted_and_the_rest_wait(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi, clock: FakeClock
) -> None:
    api.records_add_lost_responses = {2}
    order = push_order(dataset)
    with pytest.raises(ExternalError, match="timed out"):
        push(dataset, fake, clock=clock)
    # Only half the lost batch is listed yet: the rest is still being written.
    remote = api.dataset_named(NAME)
    pending = (remote.records[6:], remote.record_ids[6:])
    del remote.records[6:], remote.record_ids[6:]
    clock.advance(timedelta(minutes=1))
    with pytest.raises(StateError, match="2 of the 4 records") as caught:
        push(dataset, fake, clock=clock)
    assert "2 found were recorded" in caught.value.message
    state = read_push_state(dataset)
    assert state is not None
    assert set(state.record_ids) == set(order[:6])
    assert (state.in_flight, state.in_flight_at) == (tuple(order[6:8]), SENT)
    assert state.dataset_version is None
    assert records_add_calls(fake) == 2
    # The write lands; within the window the push adopts it and sends the remaining batch.
    remote.records.extend(pending[0])
    remote.record_ids.extend(pending[1])
    receipt = push(dataset, fake, clock=clock)
    assert sorted(remote_row_ids(api)) == sorted(order)
    assert receipt.state.record_ids == dict(zip(order, remote.record_ids, strict=True))
    assert records_add_calls(fake) == 3


def test_a_receipt_without_in_flight_at_is_taken_as_settled(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi, clock: FakeClock
) -> None:
    api.records_add_failures = {2}
    with pytest.raises(ExternalError):
        push(dataset, fake, clock=clock)
    path = dataset.path / PUSH_FILE
    data = json.loads(path.read_text())
    del data["in_flight_at"]  # written before it was recorded
    path.write_text(json.dumps(data))
    receipt = push(dataset, fake, clock=clock)
    assert remote_row_ids(api) == push_order(dataset)
    assert receipt.state.in_flight == ()


def test_a_batch_whose_response_was_lost_is_not_added_twice(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.records_add_lost_responses = {2}
    order = push_order(dataset)
    with pytest.raises(ExternalError, match="timed out"):
        push(dataset, fake)
    remote = api.dataset_named(NAME)
    assert len(remote.records) == 8  # the second batch went through all the same
    state = read_push_state(dataset)
    assert state is not None
    assert set(state.record_ids) == set(order[:4])
    assert state.in_flight == tuple(order[4:8])
    messages: list[str] = []
    receipt = push(dataset, fake, progress=messages.append)
    assert sorted(remote_row_ids(api)) == sorted(order)  # exactly one record per row
    assert len(remote.records) == FIRST
    assert receipt.state.record_ids == dict(zip(order, remote.record_ids, strict=True))
    assert receipt.state.in_flight == ()
    assert receipt.state.dataset_version == remote.version
    assert records_add_calls(fake) == 3  # two, then only the third batch: none sent twice
    assert any("4 of the 4 records were already in the dataset" in m for m in messages)


def test_a_lost_last_batch_is_adopted_without_sending_anything(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.records_add_lost_responses = {3}
    with pytest.raises(ExternalError):
        push(dataset, fake)
    receipt = push(dataset, fake)
    assert records_add_calls(fake) == 3
    assert receipt.pushed == 0
    assert len(receipt.state.record_ids) == FIRST
    assert receipt.state.dataset_version is None  # unknown: the lost answer carried it
    assert len(api.dataset_named(NAME).records) == FIRST


def test_an_unlisted_batch_is_not_sent_again(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.records_add_lost_responses = {1}
    with pytest.raises(ExternalError):
        push(dataset, fake)
    api.records_all_fails = True
    with pytest.raises(StateError, match="may already be there") as caught:
        push(dataset, fake)
    remote = api.dataset_named(NAME)
    assert remote.id in caught.value.message
    assert all(row_id in caught.value.message for row_id in push_order(dataset)[:4])
    assert records_add_calls(fake) == 1
    state = read_push_state(dataset)
    assert state is not None and len(state.in_flight) == 4


def test_a_truncated_listing_keeps_the_batch_in_flight(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.records_add_lost_responses = {2}
    order = push_order(dataset)
    with pytest.raises(ExternalError):
        push(dataset, fake)
    # The listing ends inside the lost batch: its last two records look absent.
    api.records_all_limit = 6
    with pytest.raises(StateError, match="may already be there") as caught:
        push(dataset, fake)
    assert "only part of the records" in caught.value.message
    assert caught.value.hint is not None and "in_flight" in caught.value.hint
    assert all(row_id in caught.value.message for row_id in order[4:8])
    assert records_add_calls(fake) == 2  # nothing sent again
    assert len(api.dataset_named(NAME).records) == 8
    state = read_push_state(dataset)
    assert state is not None
    assert state.in_flight == tuple(order[4:8])
    assert set(state.record_ids) == set(order[:4])


def test_an_unreadable_listing_keeps_the_batch_in_flight(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.records_add_lost_responses = {2}
    order = push_order(dataset)
    with pytest.raises(ExternalError):
        push(dataset, fake)
    # An item without an id may be one of the lost batch's records.
    api.records_all_extra = [{"attributes": {"metadata": {"row_id": order[4]}}}]
    with pytest.raises(StateError, match="may already be there") as caught:
        push(dataset, fake)
    assert "1 items that are not readable records" in caught.value.message
    assert caught.value.hint is not None and "in_flight" in caught.value.hint
    assert records_add_calls(fake) == 2  # nothing sent again
    state = read_push_state(dataset)
    assert state is not None
    assert state.in_flight == tuple(order[4:8])
    assert set(state.record_ids) == set(order[:4])
    api.records_all_extra = [{"id": "rec-other", "attributes": {"metadata": None}}]
    receipt = push(dataset, fake)  # a record without metadata is readable, and not ours
    assert sorted(remote_row_ids(api)) == sorted(order)
    assert receipt.state.in_flight == ()


def test_a_lost_create_response_is_recovered_in_the_same_push(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.dataset_create_answer = "fail_after_create"
    api.records_add_failures = {1}
    with pytest.raises(ExternalError):
        push(dataset, fake)
    created = api.dataset_named(NAME)
    # The create's id was found by name at once, before the records failed.
    state = read_push_state(dataset)
    assert state is not None and state.record_ids == {}
    assert state.dataset_id == created.id
    receipt = push(dataset, fake, clock=lambda: datetime.now(UTC) + IN_FLIGHT_SETTLE)
    assert receipt.state.dataset_id == created.id
    assert len(api.datasets) == 1
    assert len(created.records) == FIRST


def test_a_push_that_stopped_before_the_dataset_id_was_saved_finds_it(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.dataset_create_answer = "fail"
    with pytest.raises(ExternalError, match="500"):
        push(dataset, fake)
    state = read_push_state(dataset)
    assert state is not None and state.dataset_id is None
    assert api.datasets == {}
    # Meanwhile the dataset was created (e.g. the create went through after all).
    created = api.add_dataset(state.project_id, NAME, dataset_description(state.push_id))
    api.dataset_create_answer = "ok"
    receipt = push(dataset, fake)
    assert receipt.state.dataset_id == created.id
    assert len(api.datasets) == 1


def test_a_same_name_dataset_created_out_of_band_is_not_adopted(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.dataset_create_answer = "fail"
    with pytest.raises(ExternalError, match="500"):
        push(dataset, fake)
    state = read_push_state(dataset)
    assert state is not None
    api.add_dataset(state.project_id, NAME, "someone else's dataset")
    api.dataset_create_answer = "ok"
    with pytest.raises(UsageError, match="did not create") as caught:
        push(dataset, fake)
    assert "--name" in (caught.value.hint or "") and PUSH_FILE in (caught.value.hint or "")
    assert records_add_calls(fake) == 0


def test_a_dataset_the_directory_did_not_create_is_refused(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    api.add_dataset(api.add_project(PROJECT), NAME)
    with pytest.raises(UsageError, match="already has a dataset named") as caught:
        push(dataset, fake)
    assert caught.value.hint is not None and "--name" in caught.value.hint
    assert records_add_calls(fake) == 0
    assert read_push_state(dataset) is None


def test_dropped_rows_are_never_pushed(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    dropped = ALL_ROWS[1].id
    drop(dataset, dropped)
    receipt = push(dataset, fake)
    pushed = [r["metadata"]["row_id"] for r in api.dataset_named(NAME).records]
    assert dropped not in pushed
    assert len(pushed) == FIRST - 1
    assert sum(receipt.records_by_split.values()) == FIRST - 1


def test_rows_dropped_after_a_push_are_stale_and_kept_remotely(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    first = push(dataset, fake)
    dropped = ALL_ROWS[2].id
    drop(dataset, dropped)
    plan = plan_push(dataset, project=PROJECT, name=NAME)
    stale = {dropped: first.state.record_ids[dropped]}
    assert plan.stale_records == stale
    receipt = push(dataset, fake)
    assert receipt.stale_records == stale
    assert receipt.pushed == 0
    assert len(api.dataset_named(NAME).records) == FIRST  # nothing was deleted
    state = read_push_state(dataset)
    assert state is not None and dropped in state.record_ids
    # Kept again, the row is no longer stale and is not sent a second time.
    apply_labels(dataset, [Label(id=dropped, keep=True)])
    again = push(dataset, fake)
    assert (again.stale_records, again.pushed) == ({}, 0)
    assert len(api.dataset_named(NAME).records) == FIRST


@pytest.mark.parametrize(
    ("name", "project"), [("another-name", PROJECT), (NAME, "another-project")]
)
def test_another_name_or_project_than_the_receipt_is_a_usage_error(
    dataset: DatasetDir, fake: FakePup, name: str, project: str
) -> None:
    push(dataset, fake)
    calls = len(fake.calls)
    with pytest.raises(UsageError, match="was pushed to dataset") as caught:
        push(dataset, fake, name=name, project=project)
    hint = caught.value.hint or ""
    assert f"--name {NAME}" in hint and f"--project {PROJECT}" in hint
    assert PUSH_FILE in hint
    assert len(fake.calls) == calls


def test_a_receipt_of_other_settings_is_a_state_error(dataset: DatasetDir, fake: FakePup) -> None:
    push(dataset, fake)
    path = dataset.path / PUSH_FILE
    data = json.loads(path.read_text())
    data["settings_hash"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(StateError, match="other settings"):
        push(dataset, fake)


@pytest.mark.parametrize(
    "change",
    [
        {"record_ids": ["not", "a", "mapping"]},
        {"project_id": ""},
        {"dataset_version": "3"},
        {"dataset_id": 7},
        {"push_id": None},
        {"in_flight": "row-1"},
        {"in_flight_at": "yesterday"},
        {"in_flight_at": "2026-01-02T03:04:05"},
    ],
)
def test_a_corrupt_receipt_is_a_state_error(
    dataset: DatasetDir, fake: FakePup, change: dict[str, Any]
) -> None:
    push(dataset, fake)
    path = dataset.path / PUSH_FILE
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(StateError, match=PUSH_FILE):
        push(dataset, fake)


def test_a_dataset_without_kept_rows_has_nothing_to_push(
    dataset: DatasetDir, fake: FakePup
) -> None:
    drop(dataset, *(row.id for row in ALL_ROWS[:FIRST]))
    with pytest.raises(UsageError, match="no kept rows"):
        push(dataset, fake)
    assert fake.calls == []


def test_push_sends_batches_of_100_by_default(tmp_path: Path) -> None:
    assert RECORDS_BATCH_SIZE == 100
    dataset = DatasetDir(tmp_path / "big")
    dataset.add(synthetic_rows(130), settings=IMPORT)
    sizes: list[int] = []

    class CountingApi(FakeDatasetsApi):
        def _records_add(self, call: PupCall) -> Any:
            sizes.append(
                len(json.loads(Path(call.args[call.args.index("--file") + 1]).read_text()))
            )
            return super()._records_add(call)

    fake = CountingApi().install(FakePup())
    receipt = push_dataset(dataset, project=PROJECT, name=NAME, pup=fake.client())
    assert sizes == [100, 30]
    assert receipt.pushed == 130


def test_a_missing_version_is_recorded_as_unknown(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    push(dataset, fake)
    dataset.add(ALL_ROWS, settings=IMPORT)
    fake.on("llm-obs", "datasets", "records-add", json={"created_record_ids": ["r-1"] * 4})
    receipt = push(dataset, fake)
    assert receipt.state.dataset_version is None
    state = read_push_state(dataset)
    assert state is not None and state.dataset_version is None


def test_sigterm_stops_the_push_and_removes_the_batch_file(
    dataset: DatasetDir, fake: FakePup, api: FakeDatasetsApi
) -> None:
    def terminate(_call: PupCall) -> Any:
        os.kill(os.getpid(), signal.SIGTERM)
        return {}

    fake.on("llm-obs", "datasets", "records-add", handler=terminate)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(PushTerminated):
        push(dataset, fake)
    assert signal.getsignal(signal.SIGTERM) == before
    batch_file = Path(fake.last.args[fake.last.args.index("--file") + 1])
    assert not batch_file.parent.exists()
    state = read_push_state(dataset)
    assert state is not None and len(state.in_flight) == 4


def test_push_needs_a_dataset_directory(tmp_path: Path, fake: FakePup) -> None:
    with pytest.raises(UsageError, match="not a dataset directory"):
        push(DatasetDir(tmp_path / "nothing"), fake)


def test_push_rejects_empty_names(dataset: DatasetDir, fake: FakePup) -> None:
    with pytest.raises(UsageError, match="--name"):
        push(dataset, fake, name=" ")
    with pytest.raises(UsageError, match="--project"):
        push(dataset, fake, project="")


def test_push_holds_the_directory_lock(dataset: DatasetDir, fake: FakePup) -> None:
    with DirectoryLock(dataset.path), pytest.raises(LockBusy):
        push(dataset, fake)


def test_push_reports_progress(dataset: DatasetDir, fake: FakePup) -> None:
    messages: list[str] = []
    push(dataset, fake, progress=messages.append)
    assert messages[-1] == f"pushed {FIRST}/{FIRST} records"
    assert any("creating dataset" in message for message in messages)


def test_plan_push_counts_by_split_and_writes_nothing(dataset: DatasetDir) -> None:
    before = sorted(path.name for path in dataset.path.iterdir())
    plan = plan_push(dataset, project=PROJECT, name=NAME)
    assert sorted(path.name for path in dataset.path.iterdir()) == before
    assert plan.state is None
    assert len(plan.pending) == FIRST
    assert plan.pending_by_split() == plan.kept_by_split
    assert sum(plan.kept_by_split.values()) == FIRST
