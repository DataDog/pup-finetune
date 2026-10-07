# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import math
from collections.abc import Iterator
from typing import Any

import pytest

from pup_finetune.data.exclusions import ExclusionCode
from pup_finetune.data.jsonl import row_to_dict
from pup_finetune.data.schema import RowErrorCode
from pup_finetune.data.trajectories import (
    ConversionResult,
    comparable_messages,
    comparable_tools,
    rows_from_spans,
)
from pup_finetune.datadog.spans import LLMObsSpan, SpanQuery, iter_spans, parse_span
from pup_finetune.errors import UsageError
from tests.fakes.llmobs import FakeExportApi, fixture_pages, page, span_item
from tests.fakes.pup import FakePup


def fixture_spans() -> list[LLMObsSpan]:
    api = FakeExportApi(fixture_pages())
    return list(iter_spans(api.install(FakePup()).client(), SpanQuery()))


def make_span(
    span_id: str,
    messages: list[dict[str, Any]],
    *,
    trace_id: str = "synthetic-trace-9000",
    **attributes: Any,
) -> LLMObsSpan:
    """A span whose last message is its output and the rest its input."""
    span = parse_span(
        span_item(
            span_id,
            trace_id,
            input={"messages": messages[:-1]},
            output={"messages": messages[-1:]},
            **attributes,
        )
    )
    assert span is not None
    return span


def outcome(results: Iterator[ConversionResult] | list[ConversionResult]) -> list[str]:
    """``row:<id>`` or ``excluded:<code>:<id>`` per result."""
    described = []
    for result in results:
        if result.row is not None:
            assert result.exclusion is None
            described.append(f"row:{result.row.id}")
        else:
            assert result.exclusion is not None
            described.append(f"excluded:{result.exclusion.code.value}:{result.exclusion.row_id}")
    return described


SYSTEM = {"role": "system", "content": "You are a synthetic helper."}
Q1 = {"role": "user", "content": "First question?"}
A1 = {"role": "assistant", "content": "First answer."}
Q2 = {"role": "user", "content": "Second question?"}
A2 = {"role": "assistant", "content": "Second answer."}
Q3 = {"role": "user", "content": "Third question?"}
A3 = {"role": "assistant", "content": "Third answer."}


# --- the fixture export ---------------------------------------------------------------------


def test_fixture_export_converts_to_rows_and_exclusions() -> None:
    conversion = rows_from_spans(fixture_spans())
    assert outcome(conversion) == [
        # The summarizer sub-agent is its own row; the agent's four spans collapse into the
        # longest. Results follow the order the spans arrived in.
        "row:synthetic-trace-0101:synthetic-span-0111",
        "row:synthetic-trace-0101:synthetic-span-0104",
        "excluded:no_assistant_message:synthetic-trace-0102:synthetic-span-0201",
        "row:synthetic-trace-0103:synthetic-span-0301",
        "row:synthetic-trace-0104:synthetic-span-0401",
        "excluded:no_assistant_message:synthetic-trace-0105:synthetic-span-0501",
        "excluded:non_text_content:synthetic-trace-0106:synthetic-span-0601",
    ]
    stats = conversion.stats
    assert (stats.spans, stats.duplicate_spans, stats.traces) == (10, 0, 6)
    assert (stats.collapsed, stats.rows, stats.exclusions) == (3, 4, 3)


def test_fixture_rows_are_valid_and_carry_their_source() -> None:
    results = list(rows_from_spans(fixture_spans()))
    agent = next(r.row for r in results if r.row and r.row.id.endswith("0104"))
    assert agent is not None
    assert len(agent.messages) == 9
    assert agent.source.trace_id == "synthetic-trace-0101"
    assert agent.source.evaluations == {"helpfulness": "good"}
    sub_agent = next(r.row for r in results if r.row and r.row.id.endswith("0111"))
    assert sub_agent is not None
    assert sub_agent.messages[0].content == "You compare hotel options in one sentence."
    assert sub_agent.source.model_name == "example-model-small"
    # Rows serialize as JSONL lines.
    for result in results:
        if result.row is not None:
            json.dumps(row_to_dict(result.row), allow_nan=False)
        else:
            assert result.exclusion is not None
            json.dumps(result.exclusion.to_json()).encode("utf-8")


def test_reasoning_mode_reaches_the_rows() -> None:
    def reasoning(mode: Any) -> str | None:
        rows = [r.row for r in rows_from_spans(fixture_spans(), reasoning=mode) if r.row]
        row = next(row for row in rows if row.id.endswith("0301"))
        return row.messages[-1].reasoning_content

    assert reasoning("omit") is None
    assert reasoning("preserve") == (
        "One kilogram is about 2.20462 pounds, so 5 kg is about 11.02 lb."
    )


def test_replayed_spans_are_deduplicated_by_span_id() -> None:
    spans = fixture_spans()
    conversion = rows_from_spans([*spans, *spans[:3], spans[5]])
    assert outcome(conversion) == outcome(rows_from_spans(spans))
    assert conversion.stats.duplicate_spans == 4
    assert conversion.stats.spans == 10


def test_the_first_span_read_wins_on_duplicate_ids() -> None:
    first = make_span("synthetic-span-1", [Q1, A1])
    replay = make_span("synthetic-span-1", [Q1, A2])
    [result] = list(rows_from_spans([first, replay]))
    assert result.row is not None
    assert result.row.messages[-1].content == "First answer."


# --- prefix collapse ------------------------------------------------------------------------


def test_prefixes_collapse_into_the_longest_conversation() -> None:
    spans = [
        make_span("synthetic-span-1", [SYSTEM, Q1, A1], start_ns=1),
        make_span("synthetic-span-2", [SYSTEM, Q1, A1, Q2, A2], start_ns=2),
        make_span("synthetic-span-3", [SYSTEM, Q1, A1, Q2, A2, Q3, A3], start_ns=3),
    ]
    conversion = rows_from_spans(spans)
    assert outcome(conversion) == ["row:synthetic-trace-9000:synthetic-span-3"]
    assert conversion.stats.collapsed == 2


def test_the_longest_conversation_wins_whatever_order_spans_arrive_in() -> None:
    spans = [
        make_span("synthetic-span-3", [SYSTEM, Q1, A1, Q2, A2], start_ns=3),
        make_span("synthetic-span-1", [SYSTEM, Q1, A1], start_ns=1),
    ]
    assert outcome(rows_from_spans(spans)) == ["row:synthetic-trace-9000:synthetic-span-3"]


def test_branches_of_the_same_conversation_are_separate_rows() -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, A1, Q2, A2]),
        make_span("synthetic-span-2", [Q1, A1, Q3, A3]),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "row:synthetic-trace-9000:synthetic-span-1",
        "row:synthetic-trace-9000:synthetic-span-2",
    ]


def test_prefixes_collapse_only_within_a_trace() -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, A1], trace_id="synthetic-trace-1"),
        make_span("synthetic-span-2", [Q1, A1, Q2, A2], trace_id="synthetic-trace-2"),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "row:synthetic-trace-1:synthetic-span-1",
        "row:synthetic-trace-2:synthetic-span-2",
    ]


def test_interleaved_traces_are_grouped() -> None:
    spans = [
        make_span("synthetic-span-a1", [Q1, A1], trace_id="synthetic-trace-a"),
        make_span("synthetic-span-b1", [Q2, A2], trace_id="synthetic-trace-b"),
        make_span("synthetic-span-a2", [Q1, A1, Q3, A3], trace_id="synthetic-trace-a"),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "row:synthetic-trace-a:synthetic-span-a2",
        "row:synthetic-trace-b:synthetic-span-b1",
    ]


def test_an_invalid_longer_conversation_does_not_absorb_a_valid_prefix() -> None:
    # The later span never got an answer, so its conversation ends on the user.
    spans = [
        make_span("synthetic-span-1", [Q1, A1]),
        make_span("synthetic-span-2", [Q1, A1, Q2, {"role": "assistant", "content": ""}]),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "row:synthetic-trace-9000:synthetic-span-1",
        "excluded:last_message_not_assistant:synthetic-trace-9000:synthetic-span-2",
    ]


def test_an_invalid_prefix_of_a_valid_row_is_absorbed() -> None:
    empty_answer = {"role": "assistant", "content": ""}
    spans = [
        make_span("synthetic-span-1", [Q1, empty_answer]),
        make_span("synthetic-span-2", [Q1, A1]),
    ]
    assert outcome(rows_from_spans(spans)) == ["row:synthetic-trace-9000:synthetic-span-2"]


def test_prefix_matching_ignores_generated_tool_call_ids() -> None:
    call = {"name": "get_weather", "arguments": {"city": "Northvale"}}
    turn = [
        Q1,
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_results": [{"result": "cloudy", "name": "get_weather"}]},
    ]
    spans = [
        make_span("synthetic-span-1", [Q1, {"role": "assistant", "tool_calls": [call]}]),
        make_span("synthetic-span-2", [*turn, {"role": "assistant", "content": "Cloudy."}]),
    ]
    results = list(rows_from_spans(spans))
    assert outcome(results) == ["row:synthetic-trace-9000:synthetic-span-2"]
    row = results[0].row
    assert row is not None
    tool_calls = row.messages[1].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].id.startswith("call_")
    assert row.messages[2].tool_call_id == tool_calls[0].id


def calling(name: str, city: str) -> dict[str, Any]:
    """An assistant turn that only calls a tool, with no id so one is generated."""
    return {"role": "assistant", "tool_calls": [{"name": name, "arguments": {"city": city}}]}


def answered(name: str, city: str) -> list[dict[str, Any]]:
    """The call, its result, and a final answer."""
    result = {"role": "tool", "tool_results": [{"result": "cloudy", "name": name}]}
    return [Q1, calling(name, city), result, A1]


@pytest.mark.parametrize(
    ("short", "long"),
    [
        pytest.param(("get_weather", "Northvale"), ("get_weather", "Southvale"), id="arguments"),
        pytest.param(("get_weather", "Northvale"), ("get_forecast", "Northvale"), id="tool-name"),
    ],
)
def test_tool_calls_that_differ_do_not_collapse(
    short: tuple[str, str], long: tuple[str, str]
) -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, calling(*short)]),
        make_span("synthetic-span-2", answered(*long)),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "row:synthetic-trace-9000:synthetic-span-1",
        "row:synthetic-trace-9000:synthetic-span-2",
    ]


def test_parallel_results_paired_with_other_calls_do_not_collapse() -> None:
    calls = {
        "role": "assistant",
        "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Northvale"}, "tool_id": "call-a"},
            {"name": "get_weather", "arguments": {"city": "Southvale"}, "tool_id": "call-b"},
        ],
    }

    def results(*answering: str) -> dict[str, Any]:
        """The same result texts, in the same order, answering the calls ``answering``."""
        texts = ("cloudy", "sunny")
        return {
            "role": "tool",
            "tool_results": [
                {"result": text, "name": "get_weather", "tool_id": call_id}
                for text, call_id in zip(texts, answering, strict=True)
            ],
        }

    spans = [
        make_span("synthetic-span-1", [Q1, calls, results("call-a", "call-b"), A1]),
        make_span("synthetic-span-2", [Q1, calls, results("call-b", "call-a"), A1]),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "row:synthetic-trace-9000:synthetic-span-1",
        "row:synthetic-trace-9000:synthetic-span-2",
    ]


def test_identical_tool_calls_collapse() -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, calling("get_weather", "Northvale")]),
        make_span("synthetic-span-2", answered("get_weather", "Northvale")),
    ]
    assert outcome(rows_from_spans(spans)) == ["row:synthetic-trace-9000:synthetic-span-2"]


def test_the_same_arguments_as_a_dict_and_as_formatted_json_collapse() -> None:
    # The span's output records the call as an object; the next span's input echoes it back
    # as formatted JSON text with its keys in another order.
    as_dict = {"name": "get_weather", "arguments": {"city": "Northvale", "units": "metric"}}
    as_text = {
        "name": "get_weather",
        "arguments": '{\n  "units": "metric",\n  "city": "Northvale"\n}',
    }
    result = {"role": "tool", "tool_results": [{"result": "cloudy", "name": "get_weather"}]}
    spans = [
        make_span("synthetic-span-1", [Q1, {"role": "assistant", "tool_calls": [as_dict]}]),
        make_span(
            "synthetic-span-2",
            [Q1, {"role": "assistant", "tool_calls": [as_text]}, result, A1],
        ),
    ]
    results = list(rows_from_spans(spans))
    assert outcome(results) == ["row:synthetic-trace-9000:synthetic-span-2"]
    # The comparison is canonical; the kept row's arguments are the text it recorded.
    row = results[0].row
    assert row is not None
    tool_calls = row.messages[1].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.arguments == as_text["arguments"]


def tool_call_message(arguments: str) -> dict[str, Any]:
    call = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": arguments}}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def test_comparable_messages_canonicalize_json_arguments() -> None:
    compact = comparable_messages([tool_call_message('{"a":1,"b":[2,3]}')])
    spaced = comparable_messages([tool_call_message('{ "b": [2, 3],\n "a": 1 }')])
    assert compact == spaced
    assert compact != comparable_messages([tool_call_message('{"a":1,"b":[3,2]}')])


def test_comparable_messages_keep_arguments_that_are_not_json_as_given() -> None:
    raw = comparable_messages([tool_call_message("not json")])
    # The canonical form of a JSON string never equals text that is not JSON.
    assert raw != comparable_messages([tool_call_message('"not json"')])
    assert raw == comparable_messages([tool_call_message("not json")])
    assert raw != comparable_messages([tool_call_message("not  json")])


def test_comparable_messages_of_row_dumps_match_conversion() -> None:
    # A stored row's messages, dumped, compare as the conversation they came from did.
    spans = [make_span("synthetic-span-1", answered("get_weather", "Northvale"))]
    row = next(rows_from_spans(spans)).row
    assert row is not None
    dumped = comparable_messages([message.model_dump() for message in row.messages])
    from_dicts = comparable_messages(row_to_dict(row)["messages"])
    assert dumped == from_dicts
    assert '"get_weather"' in dumped[1]
    # The tool message names the call it answers by position, not by id.
    assert dumped[2].endswith(",[1,0]]")


def test_messages_that_differ_only_in_role_do_not_collapse() -> None:
    text = "Answer in one sentence."
    spans = [
        make_span("synthetic-span-1", [{"role": "system", "content": text}, A1]),
        make_span("synthetic-span-2", [{"role": "user", "content": text}, A1, Q2, A2]),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "row:synthetic-trace-9000:synthetic-span-1",
        "row:synthetic-trace-9000:synthetic-span-2",
    ]


def test_prefix_matching_ignores_reasoning() -> None:
    thinking = LLMObsSpan.model_validate(
        {
            "span_id": "synthetic-span-1",
            "trace_id": "synthetic-trace-9000",
            "input": {"messages": [Q1]},
            "output": {"messages": [{"role": "reasoning", "content": "Think."}, A1]},
        }
    )
    later = make_span("synthetic-span-2", [Q1, A1, Q2, A2])
    results = list(rows_from_spans([thinking, later], reasoning="preserve"))
    assert outcome(results) == ["row:synthetic-trace-9000:synthetic-span-2"]


def thinking_span(
    span_id: str, messages: list[dict[str, Any]], thought: str, **attributes: Any
) -> LLMObsSpan:
    """A span whose output is a reasoning message, then the last of ``messages``."""
    return LLMObsSpan.model_validate(
        {
            "span_id": span_id,
            "trace_id": "synthetic-trace-9000",
            "input": {"messages": messages[:-1]},
            "output": {"messages": [{"role": "reasoning", "content": thought}, messages[-1]]},
            **attributes,
        }
    )


def reasoning_by_message(results: list[ConversionResult]) -> list[str | None]:
    [result] = results
    assert result.row is not None
    return [message.reasoning_content for message in result.row.messages]


def test_a_kept_row_takes_the_reasoning_of_the_prefixes_collapsed_into_it() -> None:
    spans = [
        thinking_span("synthetic-span-1", [Q1, A1], "Think one."),
        thinking_span("synthetic-span-2", [Q1, A1, Q2, A2], "Think two."),
        make_span("synthetic-span-3", [Q1, A1, Q2, A2, Q3, A3]),
    ]
    results = list(rows_from_spans(spans, reasoning="preserve"))
    assert outcome(results) == ["row:synthetic-trace-9000:synthetic-span-3"]
    assert reasoning_by_message(results) == [None, "Think one.", None, "Think two.", None, None]


def test_a_kept_row_takes_the_reasoning_of_an_identical_conversation() -> None:
    spans = [
        thinking_span("synthetic-span-1", [Q1, A1], "Think.", start_ns=1),
        make_span("synthetic-span-2", [Q1, A1], start_ns=2),
    ]
    results = list(rows_from_spans(spans, reasoning="preserve"))
    assert outcome(results) == ["row:synthetic-trace-9000:synthetic-span-2"]
    assert reasoning_by_message(results) == [None, "Think."]


def test_a_kept_row_keeps_its_own_reasoning_over_a_collapsed_one() -> None:
    first = thinking_span("synthetic-span-1", [Q1, A1], "Earlier thought.")
    later = LLMObsSpan.model_validate(
        {
            "span_id": "synthetic-span-2",
            "trace_id": "synthetic-trace-9000",
            "input": {"messages": [Q1, {"role": "reasoning", "content": "Own thought."}, A1, Q2]},
            "output": {"messages": [A2]},
        }
    )
    results = list(rows_from_spans([first, later], reasoning="preserve"))
    assert reasoning_by_message(results) == [None, "Own thought.", None, None]


def test_the_highest_ranked_collapsed_row_gives_its_reasoning() -> None:
    spans = [
        thinking_span("synthetic-span-1", [Q1, A1], "Older.", start_ns=1),
        thinking_span("synthetic-span-2", [Q1, A1], "Newer.", start_ns=2),
        make_span("synthetic-span-3", [Q1, A1, Q2, A2], start_ns=3),
    ]
    results = list(rows_from_spans(spans, reasoning="preserve"))
    assert reasoning_by_message(results) == [None, "Newer.", None, None]


def test_reasoning_is_not_carried_across_branches_or_traces() -> None:
    spans = [
        thinking_span("synthetic-span-1", [Q1, A1, Q2, A2], "Branch one."),
        make_span("synthetic-span-2", [Q1, A1, Q3, A3]),
        make_span("synthetic-span-3", [Q1, A1, Q2, A2, Q3, A3], trace_id="synthetic-trace-9001"),
    ]
    results = list(rows_from_spans(spans, reasoning="preserve"))
    rows = {r.row.id: r.row for r in results if r.row is not None}
    assert rows["synthetic-trace-9000:synthetic-span-1"].messages[3].reasoning_content == (
        "Branch one."
    )
    for row_id in (
        "synthetic-trace-9000:synthetic-span-2",
        "synthetic-trace-9001:synthetic-span-3",
    ):
        assert all(message.reasoning_content is None for message in rows[row_id].messages)


def test_omitted_reasoning_is_not_carried() -> None:
    spans = [
        thinking_span("synthetic-span-1", [Q1, A1], "Think."),
        make_span("synthetic-span-2", [Q1, A1, Q2, A2]),
    ]
    results = list(rows_from_spans(spans, reasoning="omit"))
    assert reasoning_by_message(results) == [None, None, None, None]


# --- identical duplicates -------------------------------------------------------------------


def test_identical_conversations_keep_the_latest_start() -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, A1], start_ns=20),
        make_span("synthetic-span-2", [Q1, A1], start_ns=30),
        make_span("synthetic-span-3", [Q1, A1], start_ns=10),
    ]
    conversion = rows_from_spans(spans)
    assert outcome(conversion) == ["row:synthetic-trace-9000:synthetic-span-2"]
    assert conversion.stats.collapsed == 2


def test_identical_conversations_without_start_keep_the_last_read() -> None:
    spans = [make_span("synthetic-span-1", [Q1, A1]), make_span("synthetic-span-2", [Q1, A1])]
    assert outcome(rows_from_spans(spans)) == ["row:synthetic-trace-9000:synthetic-span-2"]


def test_identical_conversations_prefer_a_valid_row() -> None:
    # Same conversation, but the later span's evaluations cannot be recorded.
    spans = [
        make_span("synthetic-span-1", [Q1, A1], start_ns=1),
        make_span("synthetic-span-2", [Q1, A1], start_ns=2, evaluation={"score": math.nan}),
    ]
    assert outcome(rows_from_spans(spans)) == ["row:synthetic-trace-9000:synthetic-span-1"]


# --- tools ----------------------------------------------------------------------------------

WEATHER = [{"name": "get_weather", "description": "Weather.", "schema": {"type": "object"}}]
WEATHER_V2 = [
    {
        "name": "get_weather",
        "description": "Weather.",
        "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


def test_the_same_messages_under_other_tools_are_separate_rows() -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, A1], start_ns=1, tool_definitions=WEATHER),
        make_span("synthetic-span-2", [Q1, A1], start_ns=2, tool_definitions=WEATHER_V2),
    ]
    conversion = rows_from_spans(spans)
    assert outcome(conversion) == [
        "row:synthetic-trace-9000:synthetic-span-1",
        "row:synthetic-trace-9000:synthetic-span-2",
    ]
    assert conversion.stats.collapsed == 0


def test_a_prefix_under_other_tools_is_not_collapsed() -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, A1], tool_definitions=WEATHER),
        make_span("synthetic-span-2", [Q1, A1, Q2, A2], tool_definitions=WEATHER_V2),
        make_span("synthetic-span-3", [Q1, A1, Q2, A2, Q3, A3]),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "row:synthetic-trace-9000:synthetic-span-1",
        "row:synthetic-trace-9000:synthetic-span-2",
        "row:synthetic-trace-9000:synthetic-span-3",
    ]


def test_a_prefix_under_the_same_tools_collapses() -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, A1], tool_definitions=WEATHER),
        make_span("synthetic-span-2", [Q1, A1], tool_definitions=WEATHER),
        make_span("synthetic-span-3", [Q1, A1, Q2, A2], tool_definitions=WEATHER),
    ]
    conversion = rows_from_spans(spans)
    assert outcome(conversion) == ["row:synthetic-trace-9000:synthetic-span-3"]
    assert conversion.stats.collapsed == 2


def test_reasoning_is_not_carried_across_tools() -> None:
    spans = [
        thinking_span("synthetic-span-1", [Q1, A1], "Think.", tool_definitions=WEATHER),
        make_span("synthetic-span-2", [Q1, A1, Q2, A2], tool_definitions=WEATHER_V2),
    ]
    results = list(rows_from_spans(spans, reasoning="preserve"))
    rows = {r.row.id: r.row for r in results if r.row is not None}
    later = rows["synthetic-trace-9000:synthetic-span-2"]
    assert all(message.reasoning_content is None for message in later.messages)


def test_comparable_tools_keep_the_order_given() -> None:
    first = {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}}
    second = {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}}
    assert comparable_tools([first, second]) != comparable_tools([second, first])
    reordered_keys = {
        "function": {"parameters": {"type": "object"}, "name": "a"},
        "type": "function",
    }
    assert comparable_tools([first]) == comparable_tools([reordered_keys])


# --- exclusions -----------------------------------------------------------------------------


def test_conversion_issues_become_exclusions_with_their_codes() -> None:
    bad_arguments = {"name": "get_weather", "arguments": "not json", "tool_id": "call_1"}
    spans = [
        make_span(
            "synthetic-span-1",
            [Q1, {"role": "assistant", "tool_calls": [bad_arguments]}],
            trace_id="synthetic-trace-1",
        ),
        make_span(
            "synthetic-span-2",
            [{"role": "tool", "tool_results": [{"result": "x", "name": "nope"}]}, A1],
            trace_id="synthetic-trace-2",
        ),
        make_span(
            "synthetic-span-3",
            [{"role": "user", "content": [{"type": "image_url"}]}, A1],
            trace_id="synthetic-trace-3",
        ),
    ]
    results = list(rows_from_spans(spans))
    assert [r.exclusion.code if r.exclusion else None for r in results] == [
        RowErrorCode.INVALID_ARGUMENTS_JSON,
        RowErrorCode.TOOL_RESULT_WITHOUT_CALL,
        ExclusionCode.NON_TEXT_CONTENT,
    ]
    exclusion = results[0].exclusion
    assert exclusion is not None
    assert exclusion.to_json() == {
        "code": "invalid_arguments_json",
        "detail": exclusion.detail,
        "row_id": "synthetic-trace-1:synthetic-span-1",
        "trace_id": "synthetic-trace-1",
        "span_id": "synthetic-span-1",
    }


def test_an_exclusion_counts_the_issues_after_the_first() -> None:
    images = [{"type": "image_url"}, {"type": "input_audio"}]
    span = make_span(
        "synthetic-span-1",
        [{"role": "user", "content": images}, {"role": "user", "content": images}, A1],
    )
    [result] = list(rows_from_spans([span]))
    assert result.exclusion is not None
    assert result.exclusion.code == ExclusionCode.NON_TEXT_CONTENT
    assert result.exclusion.detail.endswith(" (and 1 more issue(s))")


def test_schema_errors_become_exclusions() -> None:
    spans = [
        make_span("synthetic-span-1", [Q1, SYSTEM, A1], trace_id="synthetic-trace-1"),
        make_span(
            "synthetic-span-2",
            [Q1, {"role": "assistant", "tool_calls": [{"name": "x", "tool_id": "c"}]}],
            trace_id="synthetic-trace-2",
            tool_definitions=[{"name": "y"}],
        ),
    ]
    assert outcome(rows_from_spans(spans)) == [
        "excluded:system_not_first:synthetic-trace-1:synthetic-span-1",
        "excluded:tool_not_declared:synthetic-trace-2:synthetic-span-2",
    ]


def test_lone_surrogates_become_exclusions_not_crashes() -> None:
    lone = "\ud800"
    spans = [
        make_span(
            "synthetic-span-1",
            [Q1, {"role": "assistant", "content": f"bad {lone}"}],
            trace_id="synthetic-trace-1",
        ),
        make_span(
            "synthetic-span-2",
            [
                Q1,
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "t", "arguments": {"k": lone}, "tool_id": "c"}],
                },
            ],
            trace_id="synthetic-trace-2",
        ),
        make_span("synthetic-span-3", [Q1, A1], trace_id="synthetic-trace-3", tags=[lone]),
        make_span(
            "synthetic-span-4",
            [{"role": "user", "content": [{"type": f"image{lone}"}]}, A1],
            trace_id="synthetic-trace-4",
        ),
    ]
    results = list(rows_from_spans(spans))
    assert all(result.row is None for result in results)
    codes = [result.exclusion.code if result.exclusion else None for result in results]
    assert codes == [
        RowErrorCode.INVALID_ROW,
        RowErrorCode.INVALID_ARGUMENTS_JSON,
        RowErrorCode.INVALID_ROW,
        ExclusionCode.NON_TEXT_CONTENT,
    ]
    for result in results:
        assert result.exclusion is not None
        # Every record can be written as UTF-8.
        json.dumps(result.exclusion.to_json(), ensure_ascii=False).encode("utf-8")
    assert results[2].exclusion is not None
    assert "source.tags" in results[2].exclusion.detail


def test_exclusion_ids_utf8_cannot_encode_are_left_out() -> None:
    # The span models refuse such ids, but a span built in code is not checked.
    lone = "\ud800"
    span = make_span("synthetic-span-1", [Q1, A1]).model_copy(update={"span_id": lone})
    [result] = list(rows_from_spans([span]))
    assert result.exclusion is not None
    assert result.exclusion.code == RowErrorCode.INVALID_ROW
    assert (result.exclusion.row_id, result.exclusion.span_id) == (None, None)
    assert result.exclusion.trace_id == "synthetic-trace-9000"
    json.dumps(result.exclusion.to_json(), ensure_ascii=False).encode("utf-8")


# --- the iterator ---------------------------------------------------------------------------


def test_rows_from_spans_is_lazy_and_reads_everything_on_first_next() -> None:
    read: list[str] = []

    def spans() -> Iterator[LLMObsSpan]:
        for span in [
            make_span("synthetic-span-1", [Q1, A1]),
            make_span("synthetic-span-2", [Q2, A2]),
        ]:
            read.append(span.span_id)
            yield span

    conversion = rows_from_spans(spans())
    assert read == []
    assert iter(conversion) is conversion
    next(conversion)
    assert read == ["synthetic-span-1", "synthetic-span-2"]
    assert len(list(conversion)) == 1
    assert conversion.stats.rows == 2


def test_unknown_reasoning_mode_fails_before_any_span_is_read() -> None:
    read: list[str] = []

    def spans() -> Iterator[LLMObsSpan]:
        span = make_span("synthetic-span-1", [Q1, A1])
        read.append(span.span_id)
        yield span

    with pytest.raises(UsageError):
        rows_from_spans(spans(), reasoning="keep")  # type: ignore[arg-type]
    assert read == []


def test_no_spans_no_results() -> None:
    conversion = rows_from_spans([])
    assert list(conversion) == []
    assert conversion.stats.spans == 0


def test_export_pages_with_replayed_spans() -> None:
    # A page replayed on resume repeats its spans.
    items = [span_item("synthetic-span-1", input={"messages": [Q1]}, output={"messages": [A1]})]
    api = FakeExportApi([page(items, "cursor-2"), page(items)])
    spans = iter_spans(api.install(FakePup()).client(), SpanQuery())
    conversion = rows_from_spans(spans)
    assert outcome(conversion) == ["row:synthetic-trace-9000:synthetic-span-1"]
    assert conversion.stats.duplicate_spans == 1
