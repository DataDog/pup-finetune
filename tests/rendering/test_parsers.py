# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
from typing import Any

import pytest

from pup_finetune.rendering.parsers import (
    PARSERS,
    JsonToolCallParser,
    Qwen3XmlToolParser,
    get_parser,
)

QWEN = Qwen3XmlToolParser()
WEATHER_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>"
)


def calls(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in message.get("tool_calls", [])
    ]


def test_qwen_reasoning_and_a_tool_call() -> None:
    parsed = QWEN.parse(
        f"<think>\nThe user wants the weather.\n</think>\n\n{WEATHER_CALL}<|im_end|>"
    )
    assert parsed.format_valid
    assert parsed.message["content"] == ""
    assert parsed.message["reasoning_content"] == "The user wants the weather."
    assert calls(parsed.message) == [("get_weather", {"city": "Paris"})]
    assert parsed.message["tool_calls"][0]["id"] == "call_0"
    assert parsed.message["tool_calls"][0]["type"] == "function"


def test_qwen_text_continuing_the_prompts_open_think_block() -> None:
    # The generation prompt ends "<think>\n", so samples start inside the reasoning block.
    parsed = QWEN.parse("Paris, then.\n</think>\n\nLet me check.\n\n" + WEATHER_CALL)
    assert parsed.format_valid
    assert parsed.message["reasoning_content"] == "Paris, then."
    assert parsed.message["content"] == "Let me check."
    assert calls(parsed.message) == [("get_weather", {"city": "Paris"})]


def test_qwen_empty_reasoning_is_dropped() -> None:
    parsed = QWEN.parse("\n</think>\n\nHello there!<|im_end|>")
    assert parsed.format_valid
    assert parsed.message == {"role": "assistant", "content": "Hello there!"}


def test_qwen_multiple_calls_and_json_values() -> None:
    second = (
        "<tool_call>\n<function=book>\n<parameter=nights>\n3\n</parameter>\n"
        '<parameter=options>\n{"breakfast": true}\n</parameter>\n<parameter=note>\n'
        "two\nlines\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = QWEN.parse(f"\n</think>\n\n{WEATHER_CALL}\n{second}<|im_end|>")
    assert parsed.format_valid
    assert calls(parsed.message) == [
        ("get_weather", {"city": "Paris"}),
        ("book", {"nights": 3, "options": {"breakfast": True}, "note": "two\nlines"}),
    ]
    assert [call["id"] for call in parsed.message["tool_calls"]] == ["call_0", "call_1"]


def test_qwen_a_call_with_no_parameters() -> None:
    parsed = QWEN.parse("\n</think>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>")
    assert parsed.format_valid
    assert calls(parsed.message) == [("now", {})]


def test_qwen_plain_text_after_a_prompt_that_leaves_reasoning_closed() -> None:
    parsed = Qwen3XmlToolParser(prompt_opens_think=False).parse("Just an answer.")
    assert parsed.format_valid
    assert parsed.message == {"role": "assistant", "content": "Just an answer."}


@pytest.mark.parametrize("text", ["Let me think about Par", "<think>\nLet me think about Par"])
def test_qwen_reasoning_that_never_closes_is_format_invalid(text: str) -> None:
    # The prompt opened <think>; text cut off before </think> is all reasoning.
    parsed = QWEN.parse(text)
    assert not parsed.format_valid
    assert parsed.message == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Let me think about Par",
    }


@pytest.mark.parametrize(
    "text",
    [
        "<think>\nnever closed",
        "cut off mid-reasoning",
        "",
        "\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n",
        "\n</think>\n\n<tool_call>\nnot a function\n</tool_call>",
        "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n"
        "<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>",
        "\n</think>\n\n<tool_call>\n<function=f>\nstray\n</function>\n</tool_call>",
        "a</think>b</think>c",
        "Hello<|im_end|>more<|im_end|>",
        "stray </tool_call>",
    ],
)
def test_qwen_malformed_output_is_format_invalid_with_best_effort_content(text: str) -> None:
    parsed = QWEN.parse(text)
    assert not parsed.format_valid
    assert parsed.message["role"] == "assistant"
    assert isinstance(parsed.message["content"], str)


def test_qwen_a_malformed_block_keeps_its_text_and_the_good_calls() -> None:
    bad = "<tool_call>\nnot a function\n</tool_call>"
    parsed = QWEN.parse(f"\n</think>\n\nChecking.\n{WEATHER_CALL}\n{bad}")
    assert not parsed.format_valid
    assert calls(parsed.message) == [("get_weather", {"city": "Paris"})]
    assert parsed.message["content"] == f"Checking.\n\n{bad}"


@pytest.mark.parametrize("text", ["", "\n\n", "<|", '{"unclosed": ', "<tool_call>"])
def test_parsers_never_raise(text: str) -> None:
    for name in PARSERS:
        parsed = get_parser(name).parse(text)
        assert parsed.message["role"] == "assistant"
        assert isinstance(parsed.format_valid, bool)


def test_json_tool_calls() -> None:
    parser = JsonToolCallParser()
    parsed = parser.parse(
        "<think>\nLook it up.\n</think>\n\nOne moment.\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "now", "arguments": "{}"}\n</tool_call><|im_end|>'
    )
    assert parsed.format_valid
    assert parsed.message["content"] == "One moment."
    assert parsed.message["reasoning_content"] == "Look it up."
    assert calls(parsed.message) == [("get_weather", {"city": "Paris"}), ("now", {})]


@pytest.mark.parametrize(
    "block",
    ['{"name": "f", "arguments": [1]}', '{"arguments": {}}', "[1]", "not json", '{"name": ""}'],
)
def test_json_malformed_calls_are_format_invalid(block: str) -> None:
    parsed = JsonToolCallParser().parse(f"<tool_call>{block}</tool_call>")
    assert not parsed.format_valid
    assert "tool_calls" not in parsed.message
    assert block in parsed.message["content"]


def test_parsers_are_registered_by_name() -> None:
    assert isinstance(get_parser("qwen3_xml"), Qwen3XmlToolParser)
    assert isinstance(get_parser("json_tool_call"), JsonToolCallParser)
    assert QWEN.name == "qwen3_xml"
    assert QWEN.version
    with pytest.raises(KeyError, match="known: qwen3_xml"):
        get_parser("nope")
