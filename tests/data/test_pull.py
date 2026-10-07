# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Collecting rows from the Export API, and retiring stored rows a later pull collapsed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data import dataset_dir, jsonl
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.pull import (
    Collection,
    PullOptions,
    collect,
    contained,
    pull,
    reasoning_updates,
    resolve_window,
    retirements,
)
from pup_finetune.data.split import SplitParams
from pup_finetune.datadog.spans import SpanQuery
from pup_finetune.errors import LockBusy, StateError, UsageError
from pup_finetune.storage import DirectoryLock
from tests.fakes.llmobs import FakeExportApi, fixture_pages, page, span_item
from tests.fakes.pup import FakePup

SYSTEM = {"role": "system", "content": "You are a synthetic helper."}
Q1 = {"role": "user", "content": "First question?"}
A1 = {"role": "assistant", "content": "First answer."}
Q2 = {"role": "user", "content": "Second question?"}
A2 = {"role": "assistant", "content": "Second answer."}


def item(
    span_id: str,
    messages: list[dict[str, Any]],
    *,
    trace_id: str = "synthetic-trace-9000",
    **attributes: Any,
) -> dict[str, Any]:
    """A span item whose last message is its output and the rest its input."""
    return span_item(
        span_id,
        trace_id,
        input={"messages": messages[:-1]},
        output={"messages": messages[-1:]},
        **attributes,
    )


def collected(pages: list[dict[str, Any]], **options: Any) -> tuple[Collection, FakeExportApi]:
    api = FakeExportApi(pages)
    pup = api.install(FakePup()).client()
    return collect(pup, PullOptions(SpanQuery(ml_app="synthetic"), **options)), api


def outcome(collection: Collection) -> list[str]:
    rows = [f"row:{row.id}" for row in collection.rows]
    excluded = [f"excluded:{e.code.value}:{e.row_id}" for e in collection.exclusions]
    return rows + excluded


# --- the collection -------------------------------------------------------------------------


def test_errored_spans_are_excluded_before_conversion() -> None:
    collection, _ = collected(fixture_pages())
    assert outcome(collection) == [
        "row:synthetic-trace-0101:synthetic-span-0111",
        "row:synthetic-trace-0101:synthetic-span-0104",
        "row:synthetic-trace-0103:synthetic-span-0301",
        "row:synthetic-trace-0104:synthetic-span-0401",
        # Span checks first, then conversion's exclusions.
        "excluded:span_error:synthetic-trace-0102:synthetic-span-0201",
        "excluded:no_assistant_message:synthetic-trace-0105:synthetic-span-0501",
        "excluded:non_text_content:synthetic-trace-0106:synthetic-span-0601",
    ]
    assert (collection.spans, collection.traces, collection.failed_checks) == (10, 6, 1)
    assert collection.stopped == "exhausted"


def test_include_errors_converts_errored_spans() -> None:
    collection, _ = collected(fixture_pages(), include_errors=True)
    assert "excluded:no_assistant_message:synthetic-trace-0102:synthetic-span-0201" in outcome(
        collection
    )
    assert collection.failed_checks == 0


def test_an_errored_span_does_not_absorb_the_calls_before_it() -> None:
    pages = [
        page(
            [
                item("synthetic-span-1", [SYSTEM, Q1, A1]),
                item("synthetic-span-2", [SYSTEM, Q1, A1, Q2, A2], status="error"),
            ]
        )
    ]
    collection, _ = collected(pages)
    assert outcome(collection) == [
        "row:synthetic-trace-9000:synthetic-span-1",
        "excluded:span_error:synthetic-trace-9000:synthetic-span-2",
    ]
    kept, _ = collected(pages, include_errors=True)
    assert outcome(kept) == ["row:synthetic-trace-9000:synthetic-span-2"]


def test_a_repeated_conversation_in_another_trace_is_a_duplicate() -> None:
    pages = [
        page(
            [
                item("synthetic-span-1", [SYSTEM, Q1, A1], trace_id="synthetic-trace-1"),
                item("synthetic-span-2", [SYSTEM, Q1, A1], trace_id="synthetic-trace-2"),
            ]
        )
    ]
    collection, _ = collected(pages)
    assert outcome(collection) == [
        "row:synthetic-trace-1:synthetic-span-1",
        "excluded:duplicate_conversation:synthetic-trace-2:synthetic-span-2",
    ]


def test_spans_read_twice_count_once() -> None:
    first = item("synthetic-span-1", [SYSTEM, Q1, A1])
    collection, _ = collected([page([first, first])])
    assert (collection.spans, collection.duplicate_spans, len(collection.rows)) == (1, 1, 1)


def three_pages() -> list[dict[str, Any]]:
    def trace(n: int) -> dict[str, Any]:
        question = {"role": "user", "content": f"Question {n}?"}
        answer = {"role": "assistant", "content": f"Answer {n}."}
        return item(f"synthetic-span-{n}", [SYSTEM, question, answer], trace_id=f"trace-{n}")

    return [
        page([trace(1), trace(2)], after="cursor-2"),
        page([trace(3), trace(4)], after="cursor-3"),
        page([trace(5), trace(6)]),
    ]


def test_reading_stops_at_the_target_count_before_the_next_page() -> None:
    collection, api = collected(three_pages(), target_count=3)
    assert api.cursors == [None, "cursor-2"]
    assert [row.id for row in collection.rows] == [
        "trace-1:synthetic-span-1",
        "trace-2:synthetic-span-2",
        "trace-3:synthetic-span-3",
    ]
    assert collection.over_target == 1
    assert collection.stopped == "target_count"


def test_reading_stops_at_max_candidates() -> None:
    collection, api = collected(three_pages(), max_candidates=3)
    assert api.cursors == [None, "cursor-2"]
    assert collection.spans == 3
    assert len(collection.rows) == 3
    assert collection.stopped == "max_candidates"


def test_unreadable_items_count_toward_max_candidates() -> None:
    pages = [
        page(["not an object", "nor this"], after="cursor-2"),
        page(["still not"], after="cursor-3"),
        page([item("synthetic-span-1", [SYSTEM, Q1, A1])]),
    ]
    collection, api = collected(pages, max_candidates=1)
    assert api.cursors == [None]
    assert collection.spans == 0
    assert collection.export.invalid_spans == 1
    assert [e.code for e in collection.exclusions] == [ExclusionCode.UNREADABLE_SPAN]
    assert collection.stopped == "max_candidates"


def test_max_candidates_cuts_a_mixed_page_counting_unreadable_items() -> None:
    first = item("synthetic-span-1", [SYSTEM, Q1, A1], trace_id="trace-1")
    second = item("synthetic-span-2", [SYSTEM, Q2, A2], trace_id="trace-2")
    collection, api = collected([page([first, "not an object", second])], max_candidates=2)
    assert api.cursors == [None]
    assert [row.id for row in collection.rows] == ["trace-1:synthetic-span-1"]
    assert collection.export.invalid_spans == 1
    assert collection.stopped == "max_candidates"


def test_limits_must_be_positive() -> None:
    with pytest.raises(UsageError, match="--target-count"):
        PullOptions(SpanQuery(), target_count=0)
    with pytest.raises(UsageError, match="--max-candidates"):
        PullOptions(SpanQuery(), max_candidates=0)
    with pytest.raises(UsageError, match="reasoning"):
        PullOptions(SpanQuery(), reasoning="keep")  # type: ignore[arg-type]


def test_settings_hold_what_selects_and_shapes_rows_but_not_the_limits() -> None:
    options = PullOptions(
        SpanQuery(ml_app="synthetic", tags={"env": "synthetic"}),
        reasoning="preserve",
        include_errors=True,
        target_count=7,
        max_candidates=9,
    )
    settings = options.settings()
    assert settings["filter"]["ml_app"] == "synthetic"
    assert settings["filter"]["tags"] == {"env": "synthetic"}
    assert (settings["reasoning"], settings["include_errors"]) == ("preserve", True)
    assert "target_count" not in str(settings)
    assert "max_candidates" not in str(settings)
    wider = PullOptions(options.query, reasoning="preserve", include_errors=True)
    assert wider.settings() == settings


def test_max_candidates_equal_to_every_span_reads_to_the_end() -> None:
    collection, api = collected(fixture_pages(), max_candidates=10)
    assert collection.spans == 10
    assert collection.stopped == "exhausted"
    assert api.cursors == [None, "synthetic-cursor-page-2"]


def test_unreadable_items_are_excluded_where_they_were_read() -> None:
    good = item("synthetic-span-1", [SYSTEM, Q1, A1])
    no_trace = {"id": "synthetic-span-2", "type": "span", "attributes": {"span_id": "s"}}
    collection, _ = collected([page([good, "not an object", no_trace])])
    assert [row.id for row in collection.rows] == ["synthetic-trace-9000:synthetic-span-1"]
    unreadable = [e for e in collection.exclusions if e.code is ExclusionCode.UNREADABLE_SPAN]
    assert [(e.detail, e.span_id, e.row_id) for e in unreadable] == [
        ("page 1, item 2: not a readable LLM span", None, None),
        ("page 1, item 3: not a readable LLM span", "s", None),
    ]
    assert collection.failed_checks == 0


# --- retirements ----------------------------------------------------------------------------


def test_a_stored_prefix_of_a_longer_conversation_is_retired() -> None:
    pages = [
        page([item("synthetic-span-1", [SYSTEM, Q1, A1])], after="cursor-2"),
        page([item("synthetic-span-2", [SYSTEM, Q1, A1, Q2, A2])]),
    ]
    first, _ = collected(pages, target_count=1)
    assert [row.id for row in first.rows] == ["synthetic-trace-9000:synthetic-span-1"]
    assert retirements(first, first.rows, []) == []

    full, _ = collected(pages)
    assert [row.id for row in full.rows] == ["synthetic-trace-9000:synthetic-span-2"]
    retired = retirements(full, first.rows, [])
    assert [(e.code, e.row_id) for e in retired] == [
        (ExclusionCode.CONTAINED_CONVERSATION, "synthetic-trace-9000:synthetic-span-1")
    ]
    assert "synthetic-trace-9000:synthetic-span-2" in retired[0].detail


def test_a_stored_prefix_is_kept_while_its_container_is_over_the_target() -> None:
    pages = [
        page(
            [
                item("synthetic-span-0", [SYSTEM, Q2, A2], trace_id="synthetic-trace-0"),
                item("synthetic-span-1", [SYSTEM, Q1, A1]),
                item("synthetic-span-2", [SYSTEM, Q1, A1, Q2, A2]),
            ]
        )
    ]
    stored, _ = collected([page([item("synthetic-span-1", [SYSTEM, Q1, A1])])])
    cut, _ = collected(pages, target_count=1)
    assert [row.id for row in cut.rows] == ["synthetic-trace-0:synthetic-span-0"]
    assert retirements(cut, stored.rows, []) == []


def a_long(span_id: str = "synthetic-span-a2") -> dict[str, Any]:
    return item(span_id, [SYSTEM, Q1, A1, Q2, A2], trace_id="synthetic-trace-a")


def a_prefix() -> dict[str, Any]:
    return item("synthetic-span-a1", [SYSTEM, Q1, A1], trace_id="synthetic-trace-a")


def b_long() -> dict[str, Any]:
    return item("synthetic-span-b1", [SYSTEM, Q1, A1, Q2, A2], trace_id="synthetic-trace-b")


def test_a_stored_prefix_of_a_duplicate_of_an_accepted_row_is_retired(tmp_path: Path) -> None:
    # Trace A continues on page 2 with the conversation trace B already has.
    pages = [page([b_long(), a_prefix()], after="cursor-2"), page([a_long()])]
    dataset = DatasetDir(tmp_path / "ds")

    def pull_pages(**options: Any) -> Any:
        api = FakeExportApi(pages)
        pup = api.install(FakePup()).client()
        query = SpanQuery(ml_app="synthetic")
        return pull(pup, PullOptions(query, **options), dataset, SplitParams())

    pull_pages(target_count=2)
    assert [row.id for row in dataset.rows()] == [
        "synthetic-trace-b:synthetic-span-b1",
        "synthetic-trace-a:synthetic-span-a1",
    ]
    collection, added = pull_pages()
    assert [(row.id, other) for row, other in collection.duplicates] == [
        ("synthetic-trace-a:synthetic-span-a2", "synthetic-trace-b:synthetic-span-b1")
    ]
    assert added.retired == 1
    assert [row.id for row in dataset.rows()] == ["synthetic-trace-b:synthetic-span-b1"]
    records = {record["row_id"]: record for record in dataset.excluded()}
    assert {row_id: record["code"] for row_id, record in records.items()} == {
        "synthetic-trace-a:synthetic-span-a1": "contained_conversation",
        "synthetic-trace-a:synthetic-span-a2": "duplicate_conversation",
    }
    detail = records["synthetic-trace-a:synthetic-span-a1"]["detail"]
    assert "synthetic-trace-a:synthetic-span-a2" in detail
    assert "synthetic-trace-b:synthetic-span-b1" in detail


def test_a_lower_target_count_does_not_add_back_a_prefix_retired_into_a_duplicate(
    tmp_path: Path,
) -> None:
    pages = [page([b_long(), a_prefix()], after="cursor-2"), page([a_long()])]
    dataset = DatasetDir(tmp_path / "ds")

    def pull_pages(**options: Any) -> Any:
        api = FakeExportApi(pages)
        pup = api.install(FakePup()).client()
        query = SpanQuery(ml_app="synthetic")
        return pull(pup, PullOptions(query, **options), dataset, SplitParams())

    pull_pages(target_count=2)
    pull_pages()
    rows = dataset.rows_path.read_bytes()
    excluded = dataset.excluded()
    assert [row.id for row in dataset.rows()] == ["synthetic-trace-b:synthetic-span-b1"]

    # The smaller pull stops on page 1 again, before trace A's longer conversation, so no
    # stored row of trace A contains the prefix it produces again.
    collection, added = pull_pages(target_count=2)
    assert "synthetic-trace-a:synthetic-span-a1" in [row.id for row in collection.rows]
    assert (added.added, added.retired) == (0, 0)
    assert dataset.rows_path.read_bytes() == rows
    assert dataset.excluded() == excluded


def test_contained_honours_a_stored_contained_conversation_exclusion() -> None:
    collection, _ = collected([page([a_prefix()])])
    row_id = "synthetic-trace-a:synthetic-span-a1"
    record = {
        "code": "contained_conversation",
        "detail": "a prefix of 'synthetic-trace-a:synthetic-span-a2', the same conversation",
        "row_id": row_id,
        "trace_id": "synthetic-trace-a",
        "span_id": "synthetic-span-a1",
    }
    kept, excluded = contained(collection.rows, [], [], [record])
    assert kept == []
    assert [(e.code, e.row_id, e.detail) for e in excluded] == [
        (ExclusionCode.CONTAINED_CONVERSATION, row_id, record["detail"])
    ]
    # Another code is the dataset's to replace when the row is accepted.
    other = {**record, "code": "no_assistant_message"}
    assert contained(collection.rows, [], [], [other]) == (collection.rows, [])
    # A stored row is left to the dataset whatever its exclusions say.
    assert contained(collection.rows, collection.rows, [], [record]) == (collection.rows, [])


def test_a_stored_prefix_is_kept_while_the_row_its_duplicate_repeats_is_over_the_target() -> None:
    stored, _ = collected([page([a_prefix()])])
    other = item("synthetic-span-z1", [SYSTEM, Q2, A2], trace_id="synthetic-trace-z")
    pages = [page([other, b_long(), a_prefix(), a_long()])]
    cut, _ = collected(pages, target_count=1)
    assert [row.id for row in cut.rows] == ["synthetic-trace-z:synthetic-span-z1"]
    assert [row.id for row, _ in cut.duplicates] == ["synthetic-trace-a:synthetic-span-a2"]
    assert retirements(cut, stored.rows, []) == []

    full, _ = collected(pages)
    assert [e.row_id for e in retirements(full, stored.rows, [])] == [
        "synthetic-trace-a:synthetic-span-a1"
    ]


def test_a_stored_exclusion_of_a_collapsed_conversation_is_re_recorded() -> None:
    pages = [page([item("synthetic-span-1", [SYSTEM, Q1, A1, Q2, A2])])]
    full, _ = collected(pages)
    full.read_ids.add("synthetic-trace-9000:synthetic-span-0")
    stale = {
        "code": "no_assistant_message",
        "detail": "synthetic",
        "row_id": "synthetic-trace-9000:synthetic-span-0",
        "trace_id": "synthetic-trace-9000",
        "span_id": "synthetic-span-0",
    }
    retired = retirements(full, [], [stale])
    assert [(e.code, e.row_id, e.span_id) for e in retired] == [
        (
            ExclusionCode.CONTAINED_CONVERSATION,
            "synthetic-trace-9000:synthetic-span-0",
            "synthetic-span-0",
        )
    ]
    already = {**stale, "code": "contained_conversation"}
    assert retirements(full, [], [already]) == []


PARALLEL_CALLS = {
    "role": "assistant",
    "tool_calls": [
        {"name": "get_weather", "arguments": {"city": "Northvale"}, "tool_id": "call-a"},
        {"name": "get_weather", "arguments": {"city": "Southvale"}, "tool_id": "call-b"},
    ],
}


def parallel_results(*answering: str) -> dict[str, Any]:
    """The same result text for each call in ``answering``, in that order."""
    return {
        "role": "tool",
        "tool_results": [{"result": "cloudy", "tool_id": call_id} for call_id in answering],
    }


def test_retirement_tells_apart_results_paired_with_other_calls() -> None:
    # The stored row pairs the identical results with the calls in one order; the accepted
    # conversation pairs them the other way, so it is a different conversation and does not
    # contain the stored row, even though the message texts are the same.
    stored_messages = [Q1, PARALLEL_CALLS, parallel_results("call-a", "call-b"), A1]
    stored, _ = collected([page([item("synthetic-span-1", stored_messages)])])
    swapped = [Q1, PARALLEL_CALLS, parallel_results("call-b", "call-a"), A1, Q2, A2]
    other, _ = collected([page([item("synthetic-span-2", swapped)])])
    other.read_ids.add("synthetic-trace-9000:synthetic-span-1")
    assert retirements(other, stored.rows, []) == []

    # Paired the same way, the longer conversation does contain it.
    longer = [*stored_messages, Q2, A2]
    same, _ = collected([page([item("synthetic-span-2", longer)])])
    same.read_ids.add("synthetic-trace-9000:synthetic-span-1")
    assert [e.row_id for e in retirements(same, stored.rows, [])] == [
        "synthetic-trace-9000:synthetic-span-1"
    ]


def test_retirement_compares_tool_call_arguments_canonically() -> None:
    def calling(arguments: Any) -> dict[str, Any]:
        call = {"name": "get_weather", "arguments": arguments, "tool_id": "call-a"}
        return {"role": "assistant", "tool_calls": [call]}

    result = {"role": "tool", "tool_results": [{"result": "cloudy", "tool_id": "call-a"}]}
    short = [Q1, calling({"city": "Northvale", "units": "metric"}), result, A1]
    stored, _ = collected([page([item("synthetic-span-1", short)])])
    text = '{\n  "units": "metric",\n  "city": "Northvale"\n}'
    longer, _ = collected(
        [page([item("synthetic-span-2", [Q1, calling(text), result, A1, Q2, A2])])]
    )
    longer.read_ids.add("synthetic-trace-9000:synthetic-span-1")
    assert [e.row_id for e in retirements(longer, stored.rows, [])] == [
        "synthetic-trace-9000:synthetic-span-1"
    ]


def two_page_conversation() -> list[dict[str, Any]]:
    return [
        page([item("synthetic-span-1", [SYSTEM, Q1, A1])], after="cursor-2"),
        page([item("synthetic-span-2", [SYSTEM, Q1, A1, Q2, A2])]),
    ]


def pull_into(dataset: DatasetDir, **options: Any) -> Any:
    api = FakeExportApi(two_page_conversation())
    pup = api.install(FakePup()).client()
    return pull(pup, PullOptions(SpanQuery(ml_app="synthetic"), **options), dataset, SplitParams())


def test_a_lower_target_count_does_not_add_back_a_retired_prefix(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    pull_into(dataset, target_count=1)
    pull_into(dataset)
    retired = [(record["row_id"], record["code"]) for record in dataset.excluded()]
    assert retired == [("synthetic-trace-9000:synthetic-span-1", "contained_conversation")]

    # The original, smaller pull stops on page 1 again and produces the short row again.
    collection, added = pull_into(dataset, target_count=1)
    assert [row.id for row in collection.rows] == ["synthetic-trace-9000:synthetic-span-1"]
    assert (added.added, added.unchanged, added.retired) == (0, 0, 0)
    assert [row.id for row in dataset.rows()] == ["synthetic-trace-9000:synthetic-span-2"]
    records = dataset.excluded()
    assert [(record["row_id"], record["code"]) for record in records] == retired
    assert "synthetic-trace-9000:synthetic-span-2" in records[0]["detail"]
    summary = dataset.summary()
    assert (summary["accepted"], summary["excluded"]) == (1, 1)


def test_contained_leaves_out_new_rows_a_kept_stored_row_contains() -> None:
    pages = [page([item("synthetic-span-1", [SYSTEM, Q1, A1])])]
    collection, _ = collected(pages)
    longer, _ = collected([page([item("synthetic-span-2", [SYSTEM, Q1, A1, Q2, A2])])])
    stored = longer.rows
    retiring = [
        Exclusion.for_row(row, ExclusionCode.CONTAINED_CONVERSATION, "synthetic") for row in stored
    ]
    assert contained(collection.rows, stored, retiring) == (collection.rows, [])
    kept, excluded = contained(collection.rows, stored)
    assert kept == []
    assert [(e.code, e.row_id) for e in excluded] == [
        (ExclusionCode.CONTAINED_CONVERSATION, "synthetic-trace-9000:synthetic-span-1")
    ]
    # A row already stored is the dataset's to keep or retire.
    assert contained(collection.rows, [*stored, *collection.rows]) == (collection.rows, [])


def test_a_pull_fails_while_another_holds_the_raw_cache(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    pull_into(dataset, target_count=1)
    rows = dataset.rows_path.read_bytes()
    api = FakeExportApi(two_page_conversation())
    pup = api.install(FakePup()).client()
    options = PullOptions(SpanQuery(ml_app="synthetic"))
    with DirectoryLock(dataset.raw_path), pytest.raises(LockBusy):
        pull(pup, options, dataset, SplitParams())
    assert api.requests == []
    assert dataset.rows_path.read_bytes() == rows


class LockProbingDataset(DatasetDir):
    """A dataset that notes whether the raw cache was locked the last time its rows were read
    and the last time rows were added (the reconciliation after the export)."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.raw_locked: dict[str, bool] = {}

    def _note(self, step: str) -> None:
        if self.raw_path.is_dir():
            self.raw_locked[step] = DirectoryLock(self.raw_path).is_held()

    def rows(self) -> list[Any]:
        self._note("rows")
        return super().rows()

    def add(self, *args: Any, **kwargs: Any) -> Any:
        self._note("add")
        return super().add(*args, **kwargs)


def test_a_pull_holds_the_raw_cache_until_its_rows_are_added(tmp_path: Path) -> None:
    dataset = LockProbingDataset(tmp_path / "ds")
    pull_into(dataset, target_count=1)
    pull_into(dataset)
    assert dataset.raw_locked == {"rows": True, "add": True}
    assert [row.id for row in dataset.rows()] == ["synthetic-trace-9000:synthetic-span-2"]


def test_a_pull_interrupted_while_retiring_keeps_the_retired_row_s_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    pull_into(dataset, target_count=1)
    real = dataset_dir.atomic_write_text
    retired_row_gone = False

    def write(target: Path, text: str) -> None:
        # Stop at the write of excluded.jsonl after rows.jsonl lost the retired row.
        nonlocal retired_row_gone
        if Path(target) == dataset.rows_path and "synthetic-span-1" not in text:
            retired_row_gone = True
        elif Path(target) == dataset.excluded_path and retired_row_gone:
            raise OSError("synthetic: disk full")
        real(target, text)

    # rows.jsonl is written through the jsonl module, the other files directly.
    monkeypatch.setattr(dataset_dir, "atomic_write_text", write)
    monkeypatch.setattr(jsonl, "atomic_write_text", write)
    with pytest.raises(OSError, match="disk full"):
        pull_into(dataset)
    monkeypatch.undo()
    assert [row.id for row in dataset.rows()] == ["synthetic-trace-9000:synthetic-span-2"]

    # The re-run cannot see the retired row any more, but its record survived the stop.
    _, added = pull_into(dataset)
    assert added.retired == 0
    assert [(record["row_id"], record["code"]) for record in dataset.excluded()] == [
        ("synthetic-trace-9000:synthetic-span-1", "contained_conversation")
    ]


# --- reasoning ------------------------------------------------------------------------------


def thinking_item(span_id: str, messages: list[dict[str, Any]], thought: str) -> dict[str, Any]:
    """A span item whose output is a reasoning message, then the last of ``messages``."""
    reasoning = {"role": "reasoning", "content": thought}
    return span_item(
        span_id,
        "synthetic-trace-9000",
        input={"messages": messages[:-1]},
        output={"messages": [reasoning, messages[-1]]},
    )


def donor_on_page_two() -> list[dict[str, Any]]:
    """The kept conversation on page 1; a shorter one, with reasoning, on page 2."""
    return [
        page([item("synthetic-span-2", [SYSTEM, Q1, A1, Q2, A2])], after="cursor-2"),
        page([thinking_item("synthetic-span-1", [SYSTEM, Q1, A1], "Think one.")]),
    ]


def pull_reasoning(dataset: DatasetDir, **options: Any) -> Any:
    api = FakeExportApi(donor_on_page_two())
    pup = api.install(FakePup()).client()
    query = SpanQuery(ml_app="synthetic")
    return pull(pup, PullOptions(query, reasoning="preserve", **options), dataset, SplitParams())


def stored_reasoning(dataset: DatasetDir) -> list[str | None]:
    [row] = dataset.rows()
    assert row.id == "synthetic-trace-9000:synthetic-span-2"
    return [message.reasoning_content for message in row.messages]


def test_a_re_run_reading_more_reasoning_updates_the_stored_row(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    _, first = pull_reasoning(dataset, target_count=1)
    assert first.added == 1
    assert stored_reasoning(dataset) == [None] * 5

    # Page 2 collapses a shorter conversation, with reasoning, into the stored row.
    collection, added = pull_reasoning(dataset)
    assert [row.id for row in collection.rows] == ["synthetic-trace-9000:synthetic-span-2"]
    assert (added.added, added.unchanged, added.updated, added.retired) == (0, 0, 1, 0)
    assert stored_reasoning(dataset) == [None, None, "Think one.", None, None]
    assert dataset.summary()["accepted"] == 1

    # A lower limit stops before page 2 again: the stored reasoning stays.
    _, again = pull_reasoning(dataset, target_count=1)
    assert (again.added, again.unchanged, again.updated) == (0, 1, 0)
    assert stored_reasoning(dataset) == [None, None, "Think one.", None, None]


def test_reasoning_updates_fill_in_reasoning_and_never_replace_it() -> None:
    [row] = collected([page([item("synthetic-span-1", [SYSTEM, Q1, A1, Q2, A2])])])[0].rows

    def thinking(**by_index: str) -> Any:
        messages = [
            message.model_copy(update={"reasoning_content": by_index.get(f"m{index}")})
            for index, message in enumerate(row.messages)
        ]
        return row.model_copy(update={"messages": messages})

    stored = thinking(m2="Stored.")
    newer = thinking(m2="Newer.", m4="Two.")
    assert reasoning_updates([newer], [stored]) == ([], [thinking(m2="Stored.", m4="Two.")])
    assert reasoning_updates([row], [stored]) == ([stored], [])
    # Anything else that differs is left to the dataset, and a new row passes through.
    question = row.messages[1].model_copy(update={"content": "Another question?"})
    other = row.model_copy(update={"messages": [row.messages[0], question, *row.messages[2:]]})
    assert reasoning_updates([other], [stored]) == ([other], [])
    assert reasoning_updates([newer], []) == ([newer], [])


# --- the time range -------------------------------------------------------------------------

NOW_MS = 1767225600000
HOUR_MS = 60 * 60 * 1000


def at(now_ms: int) -> Any:
    """A clock reading ``now_ms``, in seconds like :func:`time.time`."""
    return lambda: now_ms / 1000


def windows(api: FakeExportApi) -> list[tuple[object, object]]:
    """The window of each request, epoch ms read back from the strings the export sends."""
    filters = [body["data"]["attributes"]["filter"] for body in api.requests]
    return [(_bound(f.get("from")), _bound(f.get("to"))) for f in filters]


def _bound(value: object) -> object:
    assert not isinstance(value, int), "the Export API takes a time bound only as a string"
    return int(value) if isinstance(value, str) and value.isdigit() else value


def relative_options(**options: Any) -> PullOptions:
    return PullOptions(SpanQuery(ml_app="synthetic", from_="now-2h", to="now"), **options)


def test_collect_resolves_a_relative_time_range_once_at_the_clock() -> None:
    api = FakeExportApi(three_pages())
    pup = api.install(FakePup()).client()
    readings = iter([NOW_MS / 1000, (NOW_MS + HOUR_MS) / 1000])
    collect(pup, relative_options(), clock=lambda: next(readings))
    assert windows(api) == [(NOW_MS - 2 * HOUR_MS, NOW_MS)] * 3


def test_settings_freeze_the_resolved_window_and_the_expressions() -> None:
    settings = relative_options().resolved(NOW_MS).settings()
    assert (settings["filter"]["from"], settings["filter"]["to"]) == (NOW_MS - 2 * HOUR_MS, NOW_MS)
    assert (settings["from_expr"], settings["to_expr"]) == ("now-2h", "now")


def test_an_unresolved_relative_query_has_no_export_query() -> None:
    with pytest.raises(ValueError, match="resolve"):
        relative_options().settings()
    absolute = PullOptions(SpanQuery(from_=NOW_MS - HOUR_MS, to=NOW_MS))
    assert absolute.export_query == absolute.query


def test_a_pull_resumed_later_reuses_the_frozen_window(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    first = FakeExportApi(three_pages())
    pull(
        first.install(FakePup()).client(),
        relative_options(target_count=1),
        dataset,
        SplitParams(),
        clock=at(NOW_MS),
    )
    later = at(NOW_MS + 5 * HOUR_MS)
    resolved = resolve_window(relative_options(), dataset, later)
    assert resolved.window == (NOW_MS - 2 * HOUR_MS, NOW_MS)
    second = FakeExportApi(three_pages())
    pull(
        second.install(FakePup()).client(), relative_options(), dataset, SplitParams(), clock=later
    )
    assert windows(first) + windows(second) == [(NOW_MS - 2 * HOUR_MS, NOW_MS)] * 3
    settings = json.loads(dataset.settings_path.read_text())
    assert settings["source"]["filter"]["to"] == NOW_MS


def test_a_pull_with_other_expressions_resolves_them_and_is_rejected(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    api = FakeExportApi(three_pages())
    pup = api.install(FakePup()).client()
    pull(pup, relative_options(), dataset, SplitParams(), clock=at(NOW_MS))
    requests = len(api.requests)
    other = PullOptions(SpanQuery(ml_app="synthetic", from_="now-3h", to="now"))
    assert resolve_window(other, dataset, at(NOW_MS)).window == (NOW_MS - 3 * HOUR_MS, NOW_MS)
    with pytest.raises(StateError, match="settings changed"):
        pull(pup, other, dataset, SplitParams(), clock=at(NOW_MS))
    assert len(api.requests) == requests


def test_a_first_pull_resolves_at_the_clock(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    assert resolve_window(relative_options(), dataset, at(NOW_MS)).window == (
        NOW_MS - 2 * HOUR_MS,
        NOW_MS,
    )
    assert not dataset.path.exists()
