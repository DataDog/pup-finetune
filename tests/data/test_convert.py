# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import math
from typing import Any

import pytest

from pup_finetune.data.convert import (
    TEXT_PART_SEPARATOR,
    Conversation,
    span_conversation,
)
from pup_finetune.data.exclusions import ExclusionCode
from pup_finetune.data.schema import RowError, RowErrorCode, SourceRef, validate_row
from pup_finetune.datadog.spans import LLMObsSpan, parse_span
from pup_finetune.errors import UsageError
from tests.fakes.llmobs import fixture_pages, span_item


def fixture_span(span_id: str) -> LLMObsSpan:
    for fixture in fixture_pages():
        for item in fixture["data"]:
            if item["id"] == span_id:
                span = parse_span(item)
                assert span is not None
                return span
    raise AssertionError(f"no fixture span {span_id}")


def make_span(
    input_messages: list[dict[str, Any]] | None = None,
    output_messages: list[dict[str, Any]] | None = None,
    *,
    span_id: str = "synthetic-span-9001",
    **attributes: Any,
) -> LLMObsSpan:
    if input_messages is not None:
        attributes["input"] = {"messages": input_messages}
    if output_messages is not None:
        attributes["output"] = {"messages": output_messages}
    span = parse_span(span_item(span_id, **attributes))
    assert span is not None
    return span


def codes(conversation: Conversation) -> list[str]:
    return [code.value for code, _ in conversation.issues]


def as_row(conversation: Conversation) -> Any:
    return validate_row(
        {
            "id": "synthetic-row",
            "messages": conversation.messages,
            "tools": conversation.tools,
            "source": conversation.source,
        }
    )


USER = {"role": "user", "content": "What's the weather in Northvale?"}
CALL = {"name": "get_weather", "arguments": {"city": "Northvale"}, "tool_id": "call_1"}


# --- the fixture spans ----------------------------------------------------------------------


def test_tool_using_span_maps_messages_tool_calls_results_and_tools() -> None:
    conversation = span_conversation(fixture_span("synthetic-span-0104"))
    assert conversation.issues == []
    messages = conversation.messages
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0101a",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Northvale"}'},
            }
        ],
    }
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call_0101a",
        "content": '{"city": "Northvale", "temp_c": 12, "conditions": "cloudy"}',
    }
    assert messages[4]["content"] == "It's 12°C and cloudy there. Let me look for hotels."
    assert messages[4]["tool_calls"][0]["function"]["arguments"] == (
        '{"city":"Northvale","max_price":150}'
    )
    assert conversation.tools[0] == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current conditions for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    assert [tool["function"]["name"] for tool in conversation.tools] == [
        "get_weather",
        "search_hotels",
        "summarize_options",
    ]
    as_row(conversation)


def test_source_comes_from_the_span() -> None:
    conversation = span_conversation(fixture_span("synthetic-span-0104"))
    assert conversation.source == SourceRef(
        trace_id="synthetic-trace-0101",
        span_id="synthetic-span-0104",
        ml_app="synthetic-trip-planner",
        model_name="example-model-large",
        start_ns=1767225608000000000,
        evaluations={"helpfulness": "good"},
        tags=[
            "env:synthetic",
            "ml_app:synthetic-trip-planner",
            "service:synthetic-trip-planner",
            "version:0.1.0",
        ],
    )


def test_reasoning_is_omitted_by_default() -> None:
    conversation = span_conversation(fixture_span("synthetic-span-0301"))
    assert conversation.messages[-1] == {"role": "assistant", "content": "5 kg is about 11.02 lb."}
    assert all("reasoning_content" not in message for message in conversation.messages)


def test_reasoning_is_preserved_on_the_next_assistant_message() -> None:
    conversation = span_conversation(fixture_span("synthetic-span-0301"), reasoning="preserve")
    assert conversation.messages[-1] == {
        "role": "assistant",
        "content": "5 kg is about 11.02 lb.",
        "reasoning_content": "One kilogram is about 2.20462 pounds, so 5 kg is about 11.02 lb.",
    }
    assert as_row(conversation).messages[-1].reasoning_content is not None


def test_roleless_output_message_is_the_assistant() -> None:
    conversation = span_conversation(fixture_span("synthetic-span-0401"))
    assert conversation.messages[-1] == {
        "role": "assistant",
        "content": "Ticket SUP-0042 is open and assigned to the billing queue.",
    }
    as_row(conversation)


def test_span_without_output_has_only_its_input() -> None:
    conversation = span_conversation(fixture_span("synthetic-span-0501"))
    assert [message["role"] for message in conversation.messages] == ["system", "user"]
    assert conversation.issues == []


def test_empty_assistant_output_is_dropped() -> None:
    conversation = span_conversation(fixture_span("synthetic-span-0201"))
    assert [message["role"] for message in conversation.messages] == ["system", "user"]


def test_image_part_is_non_text_content() -> None:
    conversation = span_conversation(fixture_span("synthetic-span-0601"))
    assert codes(conversation) == [ExclusionCode.NON_TEXT_CONTENT]
    assert "image_url" in conversation.issues[0][1]
    # The parts are kept as given, never reduced to their text.
    assert isinstance(conversation.messages[1]["content"], list)


# --- roles and content ----------------------------------------------------------------------


def test_roleless_input_message_is_the_user() -> None:
    conversation = span_conversation(make_span([{"content": "Hi"}], [{"content": "Hello!"}]))
    assert conversation.messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_plain_values_are_the_user_and_assistant_messages() -> None:
    span = make_span(input={"value": "Hi"}, output={"value": "Hello!"})
    conversation = span_conversation(span)
    assert conversation.messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert conversation.issues == []
    as_row(conversation)


def test_plain_input_value_comes_before_chat_output() -> None:
    span = make_span(
        input={"value": "Hi"}, output={"messages": [{"role": "assistant", "content": "Hello!"}]}
    )
    assert span_conversation(span).messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_messages_win_over_a_plain_value() -> None:
    span = make_span(
        input={"value": "rendered prompt", "messages": [{"role": "user", "content": "Hi"}]},
        output={"value": "rendered", "messages": [{"role": "assistant", "content": "Hello!"}]},
    )
    assert span_conversation(span).messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


@pytest.mark.parametrize("section", [{"value": ""}, {"value": None}, {"messages": []}, {}])
def test_an_empty_value_adds_no_message(section: dict[str, Any]) -> None:
    span = make_span([{"role": "user", "content": "Hi"}], output=section)
    assert span_conversation(span).messages == [{"role": "user", "content": "Hi"}]


def test_null_content_is_empty() -> None:
    conversation = span_conversation(
        make_span([{"role": "user", "content": None}], [{"role": "assistant", "content": "Hi"}])
    )
    assert conversation.messages[0] == {"role": "user", "content": ""}


def test_text_parts_are_joined() -> None:
    parts = [{"type": "text", "text": "Convert 5 kg"}, "to pounds", {"text": "please"}]
    conversation = span_conversation(make_span([{"role": "user", "content": parts}]))
    assert conversation.issues == []
    assert conversation.messages[0]["content"] == TEXT_PART_SEPARATOR.join(
        ["Convert 5 kg", "to pounds", "please"]
    )


@pytest.mark.parametrize(
    "part",
    [{"type": "input_audio", "input_audio": {}}, {"type": "text"}, {"image": "x"}, 7, None],
)
def test_any_non_text_part_is_an_issue(part: Any) -> None:
    content = [{"type": "text", "text": "What is this?"}, part]
    conversation = span_conversation(make_span([{"role": "user", "content": content}]))
    assert codes(conversation) == [ExclusionCode.NON_TEXT_CONTENT]


def test_text_part_without_text_says_so() -> None:
    content = [{"type": "text", "text": "What is this?"}, {"type": "text"}]
    conversation = span_conversation(make_span([{"role": "user", "content": content}]))
    assert conversation.issues == [
        (
            ExclusionCode.NON_TEXT_CONTENT,
            "message 0 content part 1 is a text part with no string text",
        )
    ]


def test_unknown_roles_are_left_to_the_schema() -> None:
    conversation = span_conversation(
        make_span([{"role": "developer", "content": "Be brief."}], [{"content": "OK"}])
    )
    assert conversation.messages[0]["role"] == "developer"


# --- reasoning ------------------------------------------------------------------------------


def test_reasoning_after_the_last_assistant_goes_to_the_previous_one() -> None:
    span = make_span(
        [USER],
        [
            {"role": "assistant", "content": "It's cloudy."},
            {"role": "reasoning", "content": "The forecast said clouds."},
        ],
    )
    conversation = span_conversation(span, reasoning="preserve")
    assert conversation.messages[-1]["reasoning_content"] == "The forecast said clouds."


def test_reasoning_goes_to_the_next_assistant_before_the_previous_one() -> None:
    span = make_span(
        [USER],
        [
            {"role": "assistant", "content": "A"},
            {"role": "reasoning", "content": "R"},
            {"role": "assistant", "content": "B"},
        ],
    )
    conversation = span_conversation(span, reasoning="preserve")
    assert conversation.messages[1:] == [
        {"role": "assistant", "content": "A"},
        {"role": "assistant", "content": "B", "reasoning_content": "R"},
    ]


def test_non_text_reasoning_is_an_issue_and_never_reasoning_content() -> None:
    span = make_span(
        [USER],
        [{"role": "reasoning", "content": [{"type": "image_url"}]}, {"content": "Answer."}],
    )
    conversation = span_conversation(span, reasoning="preserve")
    assert codes(conversation) == [ExclusionCode.NON_TEXT_CONTENT]
    assert all("reasoning_content" not in message for message in conversation.messages)


def test_reasoning_skips_dropped_empty_assistant_messages() -> None:
    span = make_span(
        [USER],
        [
            {"role": "reasoning", "content": "Check the weather first."},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "It's cloudy."},
        ],
    )
    conversation = span_conversation(span, reasoning="preserve")
    assert conversation.messages[1:] == [
        {
            "role": "assistant",
            "content": "It's cloudy.",
            "reasoning_content": "Check the weather first.",
        }
    ]


def test_reasoning_with_no_assistant_message_is_dropped() -> None:
    span = make_span([USER], [{"role": "reasoning", "content": "Hmm."}])
    conversation = span_conversation(span, reasoning="preserve")
    assert conversation.messages == [USER]


def test_reasoning_stays_in_its_own_list() -> None:
    span = make_span(
        [USER, {"role": "assistant", "content": "Earlier answer."}],
        [{"role": "reasoning", "content": "New thought."}, {"content": "New answer."}],
    )
    conversation = span_conversation(span, reasoning="preserve")
    assert "reasoning_content" not in conversation.messages[1]
    assert conversation.messages[2]["reasoning_content"] == "New thought."


def test_several_reasoning_messages_are_joined() -> None:
    span = make_span(
        [USER],
        [
            {"role": "reasoning", "content": "First."},
            {"role": "reasoning", "content": "Second."},
            {"content": "Answer."},
        ],
    )
    conversation = span_conversation(span, reasoning="preserve")
    assert conversation.messages[-1]["reasoning_content"] == "First.\n\nSecond."


def test_unknown_reasoning_mode_is_a_usage_error() -> None:
    with pytest.raises(UsageError):
        span_conversation(make_span([USER]), reasoning="keep")  # type: ignore[arg-type]


# --- tool calls -----------------------------------------------------------------------------


def test_missing_tool_id_gets_a_deterministic_generated_id() -> None:
    def generated(span_id: str, message_index: int = 1, call_index: int = 0) -> str:
        calls = [{"name": "get_weather", "arguments": {"city": f"City {i}"}} for i in range(2)]
        span = make_span([USER], [{"role": "assistant", "tool_calls": calls}], span_id=span_id)
        conversation = span_conversation(span)
        assert conversation.issues == []
        result: str = conversation.messages[message_index]["tool_calls"][call_index]["id"]
        return result

    first = generated("synthetic-span-a")
    assert first.startswith("call_")
    assert len(first) == len("call_") + 16
    assert first == generated("synthetic-span-a")
    assert first != generated("synthetic-span-b")
    assert first != generated("synthetic-span-a", call_index=1)


def test_generated_ids_differ_between_messages() -> None:
    call = {"name": "get_weather", "arguments": {"city": "Northvale"}}
    span = make_span(
        [
            USER,
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_results": [{"result": "cloudy", "name": "get_weather"}]},
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_results": [{"result": "still cloudy", "name": "get_weather"}]},
        ],
        [{"content": "Cloudy."}],
    )
    conversation = span_conversation(span)
    assert conversation.issues == []
    first, second = (conversation.messages[i]["tool_calls"][0]["id"] for i in (1, 3))
    assert first != second
    assert [conversation.messages[i]["tool_call_id"] for i in (2, 4)] == [first, second]
    as_row(conversation)


def test_string_arguments_that_are_a_json_object_are_kept_as_is() -> None:
    call = {**CALL, "arguments": '{"city": "Northvale"}'}
    conversation = span_conversation(make_span([USER], [{"tool_calls": [call]}]))
    assert conversation.issues == []
    function = conversation.messages[1]["tool_calls"][0]["function"]
    assert function["arguments"] == '{"city": "Northvale"}'


@pytest.mark.parametrize("arguments", ["not json", "[1, 2]", '"city"', '{"x": NaN}'])
def test_string_arguments_that_are_not_a_json_object_are_an_issue(arguments: str) -> None:
    call = {**CALL, "arguments": arguments}
    conversation = span_conversation(make_span([USER], [{"tool_calls": [call]}]))
    assert codes(conversation) == [RowErrorCode.INVALID_ARGUMENTS_JSON]


def test_non_finite_dict_arguments_are_an_issue() -> None:
    span = make_span([USER], [{"tool_calls": [CALL]}])
    span.output.messages[0].tool_calls[0].arguments["city"] = math.nan  # type: ignore[index,union-attr]
    conversation = span_conversation(span)
    assert codes(conversation) == [RowErrorCode.INVALID_ARGUMENTS_JSON]


def test_missing_arguments_are_an_empty_object() -> None:
    call = {"name": "list_cities", "tool_id": "call_1"}
    conversation = span_conversation(make_span([USER], [{"tool_calls": [call]}]))
    assert conversation.messages[1]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_dict_arguments_keep_their_order_and_text() -> None:
    call = {**CALL, "arguments": {"z": "Zürich", "a": [1, 2]}}
    conversation = span_conversation(make_span([USER], [{"tool_calls": [call]}]))
    arguments = conversation.messages[1]["tool_calls"][0]["function"]["arguments"]
    assert arguments == '{"z":"Zürich","a":[1,2]}'
    assert json.loads(arguments) == {"z": "Zürich", "a": [1, 2]}


# --- tool results ---------------------------------------------------------------------------


def tool_turn(*results: dict[str, Any], role: str = "tool", content: str = "") -> list[Any]:
    calls = [
        {"name": "get_weather", "arguments": {"city": "Northvale"}, "tool_id": "call_1"},
        {"name": "get_time", "arguments": {"city": "Northvale"}, "tool_id": "call_2"},
    ]
    return [
        USER,
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": role, "content": content, "tool_results": list(results)},
    ]


def test_each_tool_result_is_its_own_tool_message() -> None:
    span = make_span(
        tool_turn(
            {"result": "cloudy", "tool_id": "call_1"}, {"result": "9:00", "tool_id": "call_2"}
        ),
        [{"content": "Cloudy, and it is 9:00."}],
    )
    conversation = span_conversation(span)
    assert conversation.messages[2:4] == [
        {"role": "tool", "tool_call_id": "call_1", "content": "cloudy"},
        {"role": "tool", "tool_call_id": "call_2", "content": "9:00"},
    ]
    as_row(conversation)


def test_tool_result_without_id_matches_the_latest_open_call_to_that_tool() -> None:
    span = make_span(
        tool_turn(
            {"result": "9:00", "name": "get_time"}, {"result": "cloudy", "name": "get_weather"}
        ),
        [{"content": "Cloudy, and it is 9:00."}],
    )
    conversation = span_conversation(span)
    assert conversation.issues == []
    assert [m.get("tool_call_id") for m in conversation.messages[2:4]] == ["call_2", "call_1"]
    as_row(conversation)


def test_tool_result_without_id_skips_answered_calls() -> None:
    calls = [
        {"name": "get_weather", "arguments": {"city": "A"}, "tool_id": "call_a"},
        {"name": "get_weather", "arguments": {"city": "B"}, "tool_id": "call_b"},
    ]
    span = make_span(
        [
            USER,
            {"role": "assistant", "tool_calls": calls},
            {"role": "tool", "tool_results": [{"result": "sunny", "tool_id": "call_b"}]},
            {"role": "tool", "tool_results": [{"result": "rainy", "name": "get_weather"}]},
        ],
        [{"content": "A is rainy, B is sunny."}],
    )
    conversation = span_conversation(span)
    assert [m.get("tool_call_id") for m in conversation.messages[2:4]] == ["call_b", "call_a"]


def test_tool_result_without_id_or_matching_call_is_an_issue() -> None:
    span = make_span(tool_turn({"result": "?", "name": "get_traffic"}), [{"content": "Done."}])
    conversation = span_conversation(span)
    assert codes(conversation) == [RowErrorCode.TOOL_RESULT_WITHOUT_CALL]
    assert "get_traffic" in conversation.issues[0][1]


def test_unnamed_result_without_id_matches_only_a_single_open_call() -> None:
    single = make_span(
        [
            USER,
            {"tool_calls": [CALL], "role": "assistant"},
            {"role": "tool", "content": "cloudy"},
        ],
        [{"content": "Cloudy."}],
    )
    conversation = span_conversation(single)
    assert conversation.messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "cloudy",
    }

    ambiguous = make_span(tool_turn({"result": "cloudy"}), [{"content": "Cloudy."}])
    assert codes(span_conversation(ambiguous)) == [RowErrorCode.TOOL_RESULT_WITHOUT_CALL]


def test_null_tool_result_takes_the_tool_message_content() -> None:
    span = make_span(
        [
            USER,
            {"role": "assistant", "tool_calls": [CALL]},
            {"role": "tool", "content": "cloudy", "tool_results": [{"tool_id": "call_1"}]},
        ],
        [{"content": "Cloudy."}],
    )
    conversation = span_conversation(span)
    assert conversation.issues == []
    assert conversation.messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "cloudy",
    }
    as_row(conversation)


IMAGE = {"type": "image_url", "image_url": {"url": "https://example.com/radar.png"}}


@pytest.mark.parametrize(
    "results",
    [
        [{"result": "cloudy", "tool_id": "call_1"}],
        [{"result": "cloudy", "tool_id": "call_1"}, {"result": "9:00", "tool_id": "call_2"}],
        [{"tool_id": "call_1"}, {"tool_id": "call_2"}],
    ],
    ids=["one-result", "two-results", "two-null-results"],
)
def test_non_text_content_beside_structured_results_is_an_issue(
    results: list[dict[str, Any]],
) -> None:
    # The content is not kept apart from the results, but the image the model saw beside
    # them must not be silently dropped from a text-only row.
    messages = tool_turn(*results)
    messages[2]["content"] = ["See the radar.", IMAGE]
    conversation = span_conversation(make_span(messages, [{"content": "Cloudy."}]))
    assert codes(conversation) == [ExclusionCode.NON_TEXT_CONTENT]
    assert "image_url" in conversation.issues[0][1]


def test_non_text_content_of_a_null_result_is_one_issue() -> None:
    span = make_span(
        [
            USER,
            {"role": "assistant", "tool_calls": [CALL]},
            {"role": "tool", "content": [IMAGE], "tool_results": [{"tool_id": "call_1"}]},
        ],
        [{"content": "Cloudy."}],
    )
    conversation = span_conversation(span)
    assert codes(conversation) == [ExclusionCode.NON_TEXT_CONTENT]
    assert conversation.messages[2]["content"] == [IMAGE]


def test_text_content_beside_structured_results_is_not_an_issue() -> None:
    span = make_span(
        tool_turn({"result": "cloudy", "tool_id": "call_1"}, content="Here is the result."),
        [{"content": "Cloudy."}],
    )
    conversation = span_conversation(span)
    assert conversation.issues == []
    assert conversation.messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "cloudy",
    }


def test_tool_result_with_unknown_id_is_left_to_the_schema() -> None:
    span = make_span(tool_turn({"result": "?", "tool_id": "call_9"}), [{"content": "Done."}])
    conversation = span_conversation(span)
    assert conversation.issues == []
    with pytest.raises(RowError) as info:
        as_row(conversation)
    assert info.value.code is RowErrorCode.TOOL_RESULT_WITHOUT_CALL


def test_tool_results_on_a_user_message_come_before_its_content() -> None:
    span = make_span(
        tool_turn(
            {"result": "cloudy", "tool_id": "call_1"},
            {"result": "9:00", "tool_id": "call_2"},
            role="user",
            content="Thanks, and is it warm?",
        ),
        [{"content": "Not very."}],
    )
    conversation = span_conversation(span)
    assert conversation.messages[2:5] == [
        {"role": "tool", "tool_call_id": "call_1", "content": "cloudy"},
        {"role": "tool", "tool_call_id": "call_2", "content": "9:00"},
        {"role": "user", "content": "Thanks, and is it warm?"},
    ]
    as_row(conversation)


def test_user_message_with_only_tool_results_is_not_kept() -> None:
    span = make_span(
        tool_turn(
            {"result": "cloudy", "tool_id": "call_1"},
            {"result": "9:00", "tool_id": "call_2"},
            role="user",
        ),
        [{"content": "Cloudy, and it is 9:00."}],
    )
    conversation = span_conversation(span)
    assert [message["role"] for message in conversation.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]


# --- tools and source -----------------------------------------------------------------------


def test_tool_definition_without_description_or_schema_gets_defaults() -> None:
    span = make_span([USER], [{"content": "Hi"}], tool_definitions=[{"name": "list_cities"}])
    conversation = span_conversation(span)
    assert conversation.tools == [
        {
            "type": "function",
            "function": {
                "name": "list_cities",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    as_row(conversation)


def test_non_finite_evaluation_is_an_issue_not_a_crash() -> None:
    span = make_span([USER], [{"content": "Hi"}])
    span.evaluation["score"] = math.inf
    conversation = span_conversation(span)
    assert codes(conversation) == [RowErrorCode.INVALID_ROW]
    assert conversation.source.evaluations == {}
    assert conversation.source.span_id == "synthetic-span-9001"


def test_tool_messages_of_a_text_based_tool_protocol_are_unstructured() -> None:
    # Some apps write tool calls into the assistant's text and send the replies as plain
    # role=tool messages, with no tool-call fields to pair them with.
    conversation = span_conversation(
        make_span(
            [
                {"role": "user", "content": "Check the queue depth."},
                {"role": "assistant", "content": 'I will run <call tool="queue_depth"/>.'},
                {"role": "tool", "content": '<tool id="TOOL-1"/>\n{"depth": 3}'},
            ],
            [{"role": "assistant", "content": "The queue depth is 3."}],
        )
    )
    assert codes(conversation) == ["unstructured_tool_message"]
    assert all(message["role"] != "tool" for message in conversation.messages)


def test_a_tool_message_after_structured_calls_still_pairs_with_them() -> None:
    conversation = span_conversation(
        make_span(
            [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "get_weather", "arguments": {}, "tool_id": "c1"}],
                },
                {"role": "tool", "content": "sunny"},
            ],
            [{"role": "assistant", "content": "Sunny."}],
        )
    )
    assert codes(conversation) == []
    assert conversation.messages[2] == {"role": "tool", "tool_call_id": "c1", "content": "sunny"}
