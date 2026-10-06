# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Agent Observability projects, datasets and records through a fake pup."""

from __future__ import annotations

import json

import pytest

from pup_finetune.datadog.datasets import (
    JSON_API,
    RECORDS_ALL_MAX_PAGES,
    DatasetInfo,
    DatasetRecord,
    LLMObsDatasets,
    RecordsListing,
    RecordsListTruncated,
)
from pup_finetune.errors import ExternalError
from tests.fakes.datasets import FakeDatasetsApi
from tests.fakes.pup import FakePup, PupCall, PupExit


@pytest.fixture
def api() -> FakeDatasetsApi:
    return FakeDatasetsApi()


@pytest.fixture
def fake(api: FakeDatasetsApi) -> FakePup:
    return api.install(FakePup())


@pytest.fixture
def client(fake: FakePup) -> LLMObsDatasets:
    return LLMObsDatasets(fake.client())


def commands(fake: FakePup) -> list[tuple[str, ...]]:
    return [call.args[:3] for call in fake.calls]


# --- projects -------------------------------------------------------------------------------


def test_find_project_picks_the_exact_name(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    api.add_project("pup-finetune-dev")
    wanted = api.add_project("pup-finetune")
    assert client.find_project("pup-finetune") == wanted
    assert fake.last.args == ("llm-obs", "projects", "list", "--filter-name", "pup-finetune")
    assert client.find_project("pup") is None


def test_ensure_project_returns_an_existing_project_without_creating(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    project_id = api.add_project("pup-finetune")
    assert client.ensure_project("pup-finetune") == project_id
    assert commands(fake) == [("llm-obs", "projects", "list")]


def test_ensure_project_creates_a_missing_project(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    project_id = client.ensure_project("pup-finetune", description="synthetic")
    assert api.projects == {project_id: "pup-finetune"}
    assert commands(fake) == [("llm-obs", "projects", "list"), ("llm-obs", "projects", "create")]
    create = fake.last.args
    assert create[3] == "--file"
    # The body file was private while pup read it, and is gone afterwards.
    assert api.file_modes == [(0o600, 0o700)]
    assert not api.file_paths[0].exists()
    assert not api.file_paths[0].parent.exists()


@pytest.mark.parametrize("answer", ["unreadable", "fail_after_create", "no_id"])
def test_ensure_project_finds_a_project_whose_create_response_was_lost(
    api: FakeDatasetsApi, client: LLMObsDatasets, answer: str
) -> None:
    api.project_create_answer = answer  # type: ignore[assignment]
    project_id = client.ensure_project("pup-finetune")
    assert api.projects == {project_id: "pup-finetune"}


def test_ensure_project_reports_a_failed_create(
    api: FakeDatasetsApi, client: LLMObsDatasets
) -> None:
    api.project_create_answer = "fail"
    with pytest.raises(ExternalError, match="projects create"):
        client.ensure_project("pup-finetune")


# --- datasets -------------------------------------------------------------------------------


def test_create_dataset_posts_a_json_api_body(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    project_id = api.add_project("pup-finetune")
    dataset_id = client.create_dataset(project_id, "synthetic-ds", description="about it")
    dataset = api.datasets[dataset_id]
    assert (dataset.name, dataset.description, dataset.project_id) == (
        "synthetic-ds",
        "about it",
        project_id,
    )
    call = fake.last
    assert call.args[:4] == ("api", f"v2/llm-obs/v1/{project_id}/datasets", "-X", "POST")
    assert call.stdin_json() == {
        "data": {
            "type": "datasets",
            "attributes": {"name": "synthetic-ds", "description": "about it"},
        }
    }
    assert api.content_types == [JSON_API]
    # Never through `pup llm-obs datasets create`, whose response pup cannot read.
    assert not any(c.args[:3] == ("llm-obs", "datasets", "create") for c in fake.calls)


@pytest.mark.parametrize("answer", ["unreadable", "fail_after_create", "no_id"])
def test_create_dataset_finds_the_dataset_when_the_response_is_unreadable(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup, answer: str
) -> None:
    project_id = api.add_project("pup-finetune")
    api.dataset_create_answer = answer  # type: ignore[assignment]
    dataset_id = client.create_dataset(project_id, "synthetic-ds")
    assert api.datasets[dataset_id].name == "synthetic-ds"
    assert len(api.datasets) == 1
    assert fake.last.args[-2:] == ("-X", "GET")


def test_create_dataset_reports_a_failed_create(
    api: FakeDatasetsApi, client: LLMObsDatasets
) -> None:
    project_id = api.add_project("pup-finetune")
    api.dataset_create_answer = "fail"
    with pytest.raises(ExternalError, match="500"):
        client.create_dataset(project_id, "synthetic-ds")


def test_create_without_an_id_and_nothing_found_is_an_error(
    fake: FakePup, client: LLMObsDatasets
) -> None:
    fake.on("api", "v2/llm-obs/v1/p1/datasets", json={"data": {"type": "datasets"}})
    fake.on("api", "v2/llm-obs/v1/p1/datasets?filter[name]=ds", json={"data": []})
    with pytest.raises(ExternalError, match="returned no id"):
        client.create_dataset("p1", "ds")


def test_find_dataset_filters_by_name_and_picks_the_exact_one(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    project_id = api.add_project("pup-finetune")
    api.add_dataset(project_id, "trip planner v2")
    wanted = api.add_dataset(project_id, "trip planner", "about trips")
    assert client.find_dataset(project_id, "trip planner") == DatasetInfo(wanted.id, "about trips")
    assert fake.last.args[1] == f"v2/llm-obs/v1/{project_id}/datasets?filter[name]=trip%20planner"
    assert client.find_dataset(project_id, "trip") is None


def test_create_dataset_adopts_only_a_dataset_carrying_the_marker(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    project_id = api.add_project("pup-finetune")
    api.add_dataset(project_id, "synthetic-ds", "someone else's")
    api.dataset_create_answer = "fail"
    with pytest.raises(ExternalError, match="500"):
        client.create_dataset(project_id, "synthetic-ds", description="x (m-1)", marker="m-1")
    mine = api.add_dataset(project_id, "synthetic-ds v2", "x (m-2)")
    found = client.create_dataset(project_id, "synthetic-ds v2", description="x", marker="m-2")
    assert found == mine.id


def test_a_failed_lookup_after_a_failed_create_reports_the_create(
    fake: FakePup, client: LLMObsDatasets
) -> None:
    fake.on("api", "v2/llm-obs/v1/p1/datasets", handler=lambda _call: PupExit(stderr="Error: 403"))
    fake.on("api", "v2/llm-obs/v1/p1/datasets?filter[name]=ds", json={"errors": ["nope"]})
    with pytest.raises(ExternalError, match="403") as caught:
        client.create_dataset("p1", "ds")
    assert "no data list" not in str(caught.value)


def test_lists_are_read_page_by_page_until_the_name_is_found(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    api.page_size = 2
    for suffix in ("-a", "-b", "-c"):
        api.add_project(f"pup-finetune{suffix}")
    wanted = api.add_project("pup-finetune")
    assert client.find_project("pup-finetune") == wanted
    assert [call.args[5:] for call in fake.calls] == [(), ("--cursor", "2")]
    for suffix in ("-a", "-b", "-c"):
        api.add_dataset(wanted, f"ds{suffix}")
    dataset = api.add_dataset(wanted, "ds")
    found = client.find_dataset(wanted, "ds")
    assert found is not None and found.id == dataset.id
    assert fake.last.args[1].endswith("&page[cursor]=2")
    assert client.find_dataset(wanted, "missing") is None


def test_a_list_whose_cursor_repeats_stops(fake: FakePup, client: LLMObsDatasets) -> None:
    fake.on("llm-obs", "projects", "list", json={"data": [], "meta": {"after": "same"}})
    assert client.find_project("pup-finetune") is None
    assert len(fake.calls) == 2


def test_a_list_without_data_is_an_error(fake: FakePup, client: LLMObsDatasets) -> None:
    fake.on("llm-obs", "projects", "list", json={"errors": ["nope"]})
    with pytest.raises(ExternalError, match="no data list"):
        client.find_project("pup-finetune")


# --- records --------------------------------------------------------------------------------


def records(count: int) -> list[dict[str, object]]:
    return [{"input": {"q": i}, "expected_output": {"a": i}} for i in range(count)]


def test_add_records_confirms_and_returns_ids_in_order(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    project_id = api.add_project("pup-finetune")
    dataset = api.add_dataset(project_id, "synthetic-ds")
    added = client.add_records(project_id, dataset.id, records(3))
    assert added.record_ids == dataset.record_ids
    assert added.dataset_version == 1
    assert dataset.records == records(3)
    args = fake.last.args
    assert args[:3] == ("llm-obs", "datasets", "records-add")
    assert args[args.index("--project-id") + 1] == project_id
    assert args[args.index("--dataset-id") + 1] == dataset.id
    assert args[-1] == "--confirm"
    assert api.file_modes == [(0o600, 0o700)]
    assert not api.file_paths[0].parent.exists()


def test_add_records_removes_the_file_when_pup_fails(
    api: FakeDatasetsApi, client: LLMObsDatasets
) -> None:
    project_id = api.add_project("pup-finetune")
    dataset = api.add_dataset(project_id, "synthetic-ds")
    api.records_add_failures = {1}
    with pytest.raises(ExternalError, match="502"):
        client.add_records(project_id, dataset.id, records(2))
    assert not api.file_paths[0].parent.exists()


def test_add_records_rejects_a_response_that_does_not_match(
    fake: FakePup, client: LLMObsDatasets
) -> None:
    fake.on("llm-obs", "datasets", "records-add", json={"created_record_ids": ["r1"]})
    with pytest.raises(ExternalError, match="created 1 records in dataset d1 for the 2 sent"):
        client.add_records("p1", "d1", records(2))
    fake.on("llm-obs", "datasets", "records-add", json={"created_count": 2})
    with pytest.raises(ExternalError, match="no created_record_ids"):
        client.add_records("p1", "d1", records(2))
    fake.on("llm-obs", "datasets", "records-add", json=["r1"])
    with pytest.raises(ExternalError, match="not a JSON object"):
        client.add_records("p1", "d1", records(1))


def test_add_records_tolerates_a_missing_version(fake: FakePup, client: LLMObsDatasets) -> None:
    fake.on("llm-obs", "datasets", "records-add", json={"created_record_ids": ["r1"]})
    added = client.add_records("p1", "d1", records(1))
    assert (added.record_ids, added.dataset_version) == (["r1"], None)


def test_list_records_reads_ids_and_metadata(
    api: FakeDatasetsApi, client: LLMObsDatasets, fake: FakePup
) -> None:
    project_id = api.add_project("pup-finetune")
    dataset = api.add_dataset(project_id, "synthetic-ds")
    client.add_records(project_id, dataset.id, [{"input": {}, "metadata": {"row_id": "r-1"}}])
    assert client.list_records(dataset.id) == RecordsListing(
        [DatasetRecord(dataset.record_ids[0], {"row_id": "r-1"})]
    )
    assert fake.last.args == ("llm-obs", "datasets", "records-all", "--dataset-id", dataset.id)


def test_list_records_counts_unreadable_items(fake: FakePup, client: LLMObsDatasets) -> None:
    good = {"id": "rec-1", "attributes": {"metadata": {"row_id": "r-1"}}}
    no_metadata = {"id": "rec-3", "attributes": {"input": {}}}
    null_metadata = {"id": "rec-4", "attributes": {"metadata": None}}
    unreadable = [
        {"id": "rec-2"},
        "junk",
        {"attributes": {"metadata": {}}},
        {"id": "", "attributes": {}},
        {"id": "rec-5", "attributes": {"metadata": "row_id=r-5"}},
    ]
    items = [good, no_metadata, *unreadable, null_metadata]
    fake.on("llm-obs", "datasets", "records-all", json={"records": items})
    # A record without metadata is readable, just not one a push added.
    assert client.list_records("d1") == RecordsListing(
        [
            DatasetRecord("rec-1", {"row_id": "r-1"}),
            DatasetRecord("rec-3", {}),
            DatasetRecord("rec-4", {}),
        ],
        unreadable=len(unreadable),
    )
    fake.on("llm-obs", "datasets", "records-all", json={"data": []})
    with pytest.raises(ExternalError, match="no records list"):
        client.list_records("d1")


def test_list_records_refuses_a_truncated_listing(
    api: FakeDatasetsApi, client: LLMObsDatasets
) -> None:
    project_id = api.add_project("pup-finetune")
    dataset = api.add_dataset(project_id, "synthetic-ds")
    rows = [{"input": {}, "metadata": {"row_id": f"r-{i}"}} for i in range(3)]
    client.add_records(project_id, dataset.id, rows)
    api.records_all_limit = 2
    with pytest.raises(RecordsListTruncated, match=r"2 records, marked truncated"):
        client.list_records(dataset.id)
    api.records_all_limit = 3  # Every record fits: the listing is whole.
    assert len(client.list_records(dataset.id).records) == 3


def test_list_records_refuses_a_listing_that_reached_the_page_cap(
    fake: FakePup, client: LLMObsDatasets
) -> None:
    good = {"id": "rec-1", "attributes": {"metadata": {"row_id": "r-1"}}}
    capped = {"records": [good], "truncated": False, "pages_fetched": RECORDS_ALL_MAX_PAGES}
    fake.on("llm-obs", "datasets", "records-all", json=capped)
    with pytest.raises(RecordsListTruncated, match=f"stopped after {RECORDS_ALL_MAX_PAGES}"):
        client.list_records("d1")
    below = {**capped, "pages_fetched": RECORDS_ALL_MAX_PAGES - 1}
    fake.on("llm-obs", "datasets", "records-all", json=below)
    assert client.list_records("d1") == RecordsListing([DatasetRecord("rec-1", {"row_id": "r-1"})])


def test_add_records_needs_records(client: LLMObsDatasets) -> None:
    with pytest.raises(ValueError, match="no records"):
        client.add_records("p1", "d1", [])


def test_the_records_file_holds_the_records_as_json(fake: FakePup, client: LLMObsDatasets) -> None:
    seen: list[object] = []

    def capture(call: PupCall) -> object:
        path = call.args[call.args.index("--file") + 1]
        with open(path, encoding="utf-8") as handle:
            seen.append(json.load(handle))
        return {"created_record_ids": ["r1"], "dataset_version": 7}

    fake.on("llm-obs", "datasets", "records-add", handler=capture)
    unicode_record = [{"input": {"text": "Grüße"}}]
    assert client.add_records("p1", "d1", unicode_record).dataset_version == 7
    assert seen == [unicode_record]
