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

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.pull import (
    Collection,
    PullOptions,
    collect,
    contained,
    pull,
    resolve_window,
    retirements,
)
from pup_finetune.data.split import SplitParams
from pup_finetune.datadog.spans import SpanQuery
from pup_finetune.errors import StateError, UsageError
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


# --- the time range -------------------------------------------------------------------------

NOW_MS = 1767225600000
HOUR_MS = 60 * 60 * 1000


def at(now_ms: int) -> Any:
    """A clock reading ``now_ms``, in seconds like :func:`time.time`."""
    return lambda: now_ms / 1000


def windows(api: FakeExportApi) -> list[tuple[object, object]]:
    filters = [body["data"]["attributes"]["filter"] for body in api.requests]
    return [(f.get("from"), f.get("to")) for f in filters]


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
