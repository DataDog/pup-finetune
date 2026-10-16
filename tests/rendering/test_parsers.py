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
    DeepSeekDsmlParser,
    GlmXmlToolParser,
    JsonToolCallParser,
    KimiXtmlParser,
    MuseAtemParser,
    Qwen3_5XmlToolParser,
    Qwen3XmlToolParser,
    get_parser,
)

QWEN = Qwen3XmlToolParser()
GLM = GlmXmlToolParser()
WEATHER_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>"
)
GLM_WEATHER_CALL = (
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
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


def lookup_call(**values: str) -> str:
    params = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in values.items())
    return f"\n</think>\n\n<tool_call>\n<function=lookup>\n{params}</function>\n</tool_call>"


def lookup_tool(**types: Any) -> dict[str, Any]:
    properties = {key: {"type": kind} for key, kind in types.items()}
    return {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": properties}},
    }


@pytest.mark.parametrize("value", ["123", "true", "null", "1.5", '{"a": 1}', "[1]", '"quoted"'])
def test_qwen_a_declared_string_parameter_keeps_its_text(value: str) -> None:
    # The template writes strings unquoted, so "123" may be the string the model meant.
    parsed = QWEN.parse(lookup_call(code=value), [lookup_tool(code="string")])
    assert parsed.format_valid
    assert calls(parsed.message) == [("lookup", {"code": value})]


def test_qwen_declared_non_string_parameters_are_decoded() -> None:
    tool = lookup_tool(limit="integer", exact="boolean", ratio="number", tags="array")
    parsed = QWEN.parse(lookup_call(limit="123", exact="true", ratio="0.5", tags='["a"]'), [tool])
    assert calls(parsed.message) == [
        ("lookup", {"limit": 123, "exact": True, "ratio": 0.5, "tags": ["a"]})
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("123", 123), ("null", None), ("true", "true"), ("abc", "abc"), ("1.5", "1.5")],
)
def test_qwen_a_type_list_with_string_decodes_only_the_other_listed_types(
    value: str, expected: Any
) -> None:
    tool = lookup_tool(code=["string", "integer", "null"])
    parsed = QWEN.parse(lookup_call(code=value), [tool])
    assert calls(parsed.message) == [("lookup", {"code": expected})]


@pytest.mark.parametrize(
    "tools",
    [
        [],
        [lookup_tool(other="string")],  # the parameter is not declared
        [{**lookup_tool(code="string"), "function": {"name": "elsewhere"}}],  # another tool
        [lookup_tool(code={"not": "a type"})],
        [{"type": "function"}, "not a tool", {"function": {"name": "lookup", "parameters": []}}],
    ],
)
def test_qwen_an_undeclared_parameter_is_decoded_as_before(tools: list[Any]) -> None:
    parsed = QWEN.parse(lookup_call(code="123", note="hello"), tools)
    assert parsed.format_valid
    assert calls(parsed.message) == [("lookup", {"code": 123, "note": "hello"})]


def test_qwen_the_first_tool_with_a_name_wins() -> None:
    tools = [lookup_tool(code="string"), lookup_tool(code="integer")]
    assert calls(QWEN.parse(lookup_call(code="123"), tools).message) == [
        ("lookup", {"code": "123"})
    ]


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


TWO_CALLS = [("get_weather", {"city": "Paris"}), ("now", {})]
BLOCKS = pytest.mark.parametrize(
    ("parser", "first", "second"),
    [
        (
            Qwen3XmlToolParser(prompt_opens_think=False),
            WEATHER_CALL,
            "<tool_call>\n<function=now>\n</function>\n</tool_call>",
        ),
        (
            JsonToolCallParser(),
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
            '<tool_call>{"name": "now", "arguments": {}}</tool_call>',
        ),
    ],
    ids=["qwen3_xml", "json_tool_call"],
)
PLACES = pytest.mark.parametrize(
    "template", ["{stray}{a}\n{b}", "{a}\n{stray}\n{b}", "{a}\n{b}\n{stray}"]
)


@BLOCKS
def test_consecutive_tool_call_blocks_are_format_valid(
    parser: Qwen3XmlToolParser | JsonToolCallParser, first: str, second: str
) -> None:
    parsed = parser.parse(f"{first}\n{second}<|im_end|>")
    assert parsed.format_valid
    assert calls(parsed.message) == TWO_CALLS
    assert parsed.message["content"] == ""


@BLOCKS
@PLACES
def test_a_stray_closing_tag_outside_the_blocks_is_format_invalid(
    parser: Qwen3XmlToolParser | JsonToolCallParser, first: str, second: str, template: str
) -> None:
    parsed = parser.parse(template.format(a=first, b=second, stray="</tool_call>"))
    assert not parsed.format_valid
    assert calls(parsed.message) == TWO_CALLS
    assert parsed.message["content"] == "</tool_call>"


@BLOCKS
@PLACES
def test_a_stray_opening_tag_outside_the_blocks_is_format_invalid(
    parser: Qwen3XmlToolParser | JsonToolCallParser, first: str, second: str, template: str
) -> None:
    parsed = parser.parse(template.format(a=first, b=second, stray="<tool_call>"))
    assert not parsed.format_valid
    assert "<tool_call>" in parsed.message["content"]


def test_qwen3_5_reads_python_scalars_as_json() -> None:
    call = (
        "<tool_call>\n<function=f>\n<parameter=on>\nTrue\n</parameter>\n"
        "<parameter=off>\nFalse\n</parameter>\n<parameter=unit>\nNone\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n<parameter=tags>\n[true, null]\n</parameter>\n"
        "<parameter=city>\nParis\n</parameter>\n<parameter=word>\nTrue story\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    parsed = Qwen3_5XmlToolParser().parse(f"\n</think>\n\n{call}<|im_end|>")
    assert parsed.format_valid
    assert calls(parsed.message) == [
        (
            "f",
            {
                "on": True,
                "off": False,
                "unit": None,
                "days": 3,
                "tags": [True, None],
                "city": "Paris",
                "word": "True story",
            },
        )
    ]
    # Qwen3.8 writes JSON, so the same text is a string there.
    assert calls(QWEN.parse(f"\n</think>\n\n{call}").message)[0][1]["on"] == "True"


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("string", "True", "True"),
        ("string", "None", "None"),
        ("boolean", "False", False),
        (["string", "null"], "None", None),
        (["string", "integer"], "True", "True"),
    ],
)
def test_qwen3_5_reads_python_literals_with_the_declared_type(
    kind: Any, value: str, expected: Any
) -> None:
    parsed = Qwen3_5XmlToolParser().parse(lookup_call(code=value), [lookup_tool(code=kind)])
    assert parsed.format_valid
    assert calls(parsed.message) == [("lookup", {"code": expected})]


def test_glm_reasoning_and_a_tool_call_end_on_observation() -> None:
    parsed = GLM.parse(f"The user wants the weather.</think>{GLM_WEATHER_CALL}<|observation|>")
    assert parsed.format_valid
    assert parsed.message["content"] == ""
    assert parsed.message["reasoning_content"] == "The user wants the weather."
    assert calls(parsed.message) == [("get_weather", {"city": "Paris"})]
    assert parsed.message["tool_calls"][0]["id"] == "call_0"


def test_glm_an_answer_ends_on_user_and_empty_reasoning_is_dropped() -> None:
    parsed = GLM.parse("</think>Hello there!<|user|>")
    assert parsed.format_valid
    assert parsed.message == {"role": "assistant", "content": "Hello there!"}
    assert GLM.parse("<think>Hm.</think>Hello there!").format_valid


def test_glm_multiple_calls_and_json_values() -> None:
    second = (
        "<tool_call>book<arg_key>nights</arg_key><arg_value>3</arg_value>"
        '<arg_key>options</arg_key><arg_value>{"breakfast": true}</arg_value>'
        "<arg_key>note</arg_key><arg_value>two\nlines</arg_value></tool_call>"
    )
    now = "<tool_call>now</tool_call>"
    parsed = GLM.parse(f"</think>Booking.{GLM_WEATHER_CALL}{second}{now}<|observation|>")
    assert parsed.format_valid
    assert parsed.message["content"] == "Booking."
    assert calls(parsed.message) == [
        ("get_weather", {"city": "Paris"}),
        ("book", {"nights": 3, "options": {"breakfast": True}, "note": "two\nlines"}),
        ("now", {}),
    ]
    assert [call["id"] for call in parsed.message["tool_calls"]] == ["call_0", "call_1", "call_2"]


@pytest.mark.parametrize(
    "text",
    [
        f"</think>{GLM_WEATHER_CALL}<|user|>",
        "</think>Hello there!<|observation|>",
    ],
)
def test_glm_the_stop_tag_must_match_the_response(text: str) -> None:
    assert not GLM.parse(text).format_valid


@pytest.mark.parametrize("text", ["Let me think about Par", "<think>Let me think about Par"])
def test_glm_reasoning_that_never_closes_is_format_invalid(text: str) -> None:
    parsed = GLM.parse(text)
    assert not parsed.format_valid
    assert parsed.message["reasoning_content"] == "Let me think about Par"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris",
        "</think><tool_call>not a function</tool_call>",
        "</think><tool_call></tool_call>",
        "</think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value>"
        "<arg_key>a</arg_key><arg_value>2</arg_value></tool_call>",
        "</think><tool_call>f stray<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
        "</think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value>x</tool_call>",
        "</think><tool_call>f<arg_key>a<arg_key>b</arg_key><arg_value>1</arg_value></tool_call>",
        "a</think>b</think>c",
        "</think>Hello<|user|>more<|user|>",
        "</think>Hello<|observation|>more",
        "</think>stray </tool_call>",
    ],
)
def test_glm_malformed_output_is_format_invalid_with_best_effort_content(text: str) -> None:
    parsed = GLM.parse(text)
    assert not parsed.format_valid
    assert parsed.message["role"] == "assistant"
    assert isinstance(parsed.message["content"], str)


def test_glm_values_that_are_not_json_stay_text() -> None:
    deep = "[" * 100_000
    call = (
        f"<tool_call>f<arg_key>a</arg_key><arg_value>NaN</arg_value>"
        f"<arg_key>b</arg_key><arg_value>{deep}</arg_value></tool_call>"
    )
    parsed = GLM.parse(f"</think>{call}<|observation|>")
    assert parsed.format_valid
    assert calls(parsed.message) == [("f", {"a": "NaN", "b": deep})]


def test_glm_reads_values_with_the_declared_type() -> None:
    call = (
        "<tool_call>lookup<arg_key>code</arg_key><arg_value>123</arg_value>"
        "<arg_key>limit</arg_key><arg_value>5</arg_value></tool_call>"
    )
    text = f"</think>{call}<|observation|>"
    parsed = GLM.parse(text, [lookup_tool(code="string", limit="integer")])
    assert parsed.format_valid
    assert calls(parsed.message) == [("lookup", {"code": "123", "limit": 5})]
    assert calls(GLM.parse(text).message) == [("lookup", {"code": 123, "limit": 5})]


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
    assert isinstance(get_parser("qwen3_5_xml"), Qwen3_5XmlToolParser)
    assert isinstance(get_parser("glm_xml"), GlmXmlToolParser)
    assert isinstance(get_parser("deepseek_dsml"), DeepSeekDsmlParser)
    assert isinstance(get_parser("kimi_xtml"), KimiXtmlParser)
    assert isinstance(get_parser("muse_atem"), MuseAtemParser)
    assert QWEN.name == "qwen3_xml"
    assert QWEN.version
    with pytest.raises(KeyError, match="known: qwen3_xml"):
        get_parser("nope")


DSML = DeepSeekDsmlParser()
DS_EOS = "<｜end▁of▁sentence｜>"  # noqa: RUF001 - DeepSeek spells it with U+FF5C


def dsml_call(name: str, *parameters: tuple[str, str, str]) -> str:
    inner = "\n".join(
        f'<｜DSML｜parameter name="{key}" string="{flag}">{value}</｜DSML｜parameter>'  # noqa: RUF001
        for key, flag, value in parameters
    )
    return f'<｜DSML｜invoke name="{name}">\n{inner}\n</｜DSML｜invoke>'  # noqa: RUF001


def dsml(*invokes: str) -> str:
    block = "\n".join(invokes)
    return f"\n\n<｜DSML｜tool_calls>\n{block}\n</｜DSML｜tool_calls>"  # noqa: RUF001


def test_deepseek_text_after_reasoning() -> None:
    parsed = DSML.parse(f"Simple arithmetic.</think>2 + 2 = 4.{DS_EOS}")
    assert parsed.format_valid
    assert parsed.message == {
        "role": "assistant",
        "content": "2 + 2 = 4.",
        "reasoning_content": "Simple arithmetic.",
    }


def test_deepseek_several_calls_with_text_and_json_parameters() -> None:
    text = (
        "Two lookups.</think>One moment."
        + dsml(
            dsml_call("get_weather", ("city", "true", "Paris"), ("days", "false", "2")),
            dsml_call("lookup", ("code", "true", "123"), ("tags", "false", '["a", {"b": null}]')),
            dsml_call("now"),
        )
        + DS_EOS
    )
    parsed = DSML.parse(text)
    assert parsed.format_valid
    assert parsed.message["content"] == "One moment."
    assert calls(parsed.message) == [
        ("get_weather", {"city": "Paris", "days": 2}),
        ("lookup", {"code": "123", "tags": ["a", {"b": None}]}),
        ("now", {}),
    ]


def test_deepseek_reasoning_that_never_closes_is_format_invalid() -> None:
    parsed = DSML.parse("Let me think about Par")
    assert not parsed.format_valid
    assert parsed.message == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Let me think about Par",
    }


@pytest.mark.parametrize(
    "block",
    [
        dsml(dsml_call("f", ("x", "false", "not json"))),
        dsml(dsml_call("f", ("x", "true", "1"), ("x", "true", "2"))),
        dsml(dsml_call("f", ("x", "maybe", "1"))),
        dsml(dsml_call("f") + "stray"),
        "\n\n<｜DSML｜tool_calls>\n" + dsml_call("f"),  # noqa: RUF001
    ],
)
def test_deepseek_a_malformed_block_is_format_invalid_and_kept_as_text(block: str) -> None:
    parsed = DSML.parse(f"</think>Calling.{block}{DS_EOS}")
    assert not parsed.format_valid
    assert "tool_calls" not in parsed.message
    assert parsed.message["content"].startswith("Calling.")
    assert block.strip() in parsed.message["content"]


def test_deepseek_stray_markers_in_the_text_are_format_invalid() -> None:
    assert not DSML.parse(f"</think>One<think>two{DS_EOS}").format_valid
    assert not DSML.parse(f"</think>One</think>two{DS_EOS}").format_valid
    assert not DSML.parse(f"</think>One{DS_EOS}two{DS_EOS}").format_valid


KIMI = KimiXtmlParser()
KIMI_TAIL = "<|close|>message<|sep|><|end_of_msg|>"


def kimi_turn(reasoning: str, content: str, tools: str | None = None) -> str:
    text = (
        f"{reasoning}<|close|>think<|sep|><|open|>response<|sep|>{content}<|close|>response<|sep|>"
    )
    if tools is not None:
        text += f"<|open|>tools<|sep|>{tools}<|close|>tools<|sep|>"
    return text + KIMI_TAIL


def kimi_call(name: str, index: int, *arguments: tuple[str, str, str]) -> str:
    body = "".join(
        f'<|open|>argument key="{key}" type="{kind}"<|sep|>{value}<|close|>argument<|sep|>'
        for key, kind, value in arguments
    )
    return f'<|open|>call tool="{name}" index="{index}"<|sep|>{body}<|close|>call<|sep|>'


def test_kimi_text_after_reasoning() -> None:
    parsed = KIMI.parse(kimi_turn("Check the data.", "Sunny, 14°C."))
    assert parsed.format_valid
    assert parsed.message == {
        "role": "assistant",
        "content": "Sunny, 14°C.",
        "reasoning_content": "Check the data.",
    }


def test_kimi_several_typed_calls_and_unescaped_attributes() -> None:
    tools = kimi_call(
        "get_weather",
        1,
        ("city", "string", 'Paris "Left" & more'),
        ("code", "string", "123"),
        ("days", "number", "2"),
        ("rate", "number", "99.5"),
        ("ok", "boolean", "true"),
        ("note", "null", "null"),
        ("opts", "object", '{"view": "sea"}'),
        ("floors", "array", "[2, 3]"),
    ) + kimi_call("now", 2)
    parsed = KIMI.parse(kimi_turn("", "", tools).replace('key="city"', 'key="ci&quot;ty"'))
    assert parsed.format_valid
    assert "reasoning_content" not in parsed.message
    assert calls(parsed.message) == [
        (
            "get_weather",
            {
                'ci"ty': 'Paris "Left" & more',
                "code": "123",
                "days": 2,
                "rate": 99.5,
                "ok": True,
                "note": None,
                "opts": {"view": "sea"},
                "floors": [2, 3],
            },
        ),
        ("now", {}),
    ]


@pytest.mark.parametrize(
    "tools",
    [
        kimi_call("f", 2),
        kimi_call("f", 1, ("x", "number", "true")),
        kimi_call("f", 1, ("x", "boolean", "1")),
        kimi_call("f", 1, ("x", "object", "not json")),
        kimi_call("f", 1, ("x", "string", "a"), ("x", "string", "b")),
        kimi_call("f", 1) + "stray",
    ],
)
def test_kimi_a_malformed_tools_channel_is_format_invalid_and_kept_as_text(tools: str) -> None:
    parsed = KIMI.parse(kimi_turn("", "Calling.", tools))
    assert not parsed.format_valid
    assert "tool_calls" not in parsed.message
    assert parsed.message["content"].startswith("Calling.")


@pytest.mark.parametrize(
    "text",
    [
        "Let me think about Par",
        kimi_turn("r", "Sunny.").removesuffix(KIMI_TAIL),
        kimi_turn("r", "Sun<|close|>ny."),
    ],
)
def test_kimi_unfinished_or_stray_markers_are_format_invalid(text: str) -> None:
    assert not KIMI.parse(text).format_valid


def test_kimi_the_stop_token_is_optional() -> None:
    parsed = KIMI.parse(kimi_turn("", "Sunny.").removesuffix("<|end_of_msg|>"))
    assert parsed.format_valid
    assert parsed.message["content"] == "Sunny."


MUSE = MuseAtemParser()


def atem_call(name: str, *parameters: tuple[str, str]) -> str:
    body = "".join(
        f'<atem:parameter name="{key}">{value}</atem:parameter>\n' for key, value in parameters
    )
    return (
        f' to={name}<|message|><atem:function_calls>\n<atem:invoke name="{name}">\n{body}'
        "</atem:invoke>\n</atem:function_calls>"
    )


def atem(*messages: str) -> str:
    return "<|eom|><|start|>assistant".join(messages) + "<|eot|>"


def test_muse_text_after_reasoning() -> None:
    parsed = MUSE.parse(atem(" to=self<|message|>Check the data.", " to=user<|message|>Sunny."))
    assert parsed.format_valid
    assert parsed.message == {
        "role": "assistant",
        "content": "Sunny.",
        "reasoning_content": "Check the data.",
    }


def test_muse_several_calls_read_with_the_declared_types() -> None:
    text = atem(
        " to=self<|message|>Two lookups.",
        atem_call("lookup", ("code", "123"), ("limit", "5"), ("ok", "true"), ("tags", '["a"]')),
        atem_call("now"),
    )
    parsed = MUSE.parse(text, [lookup_tool(code="string", limit="integer")])
    assert parsed.format_valid
    assert parsed.message["content"] == ""
    assert parsed.message["reasoning_content"] == "Two lookups."
    assert calls(parsed.message) == [
        ("lookup", {"code": "123", "limit": 5, "ok": True, "tags": ["a"]}),
        ("now", {}),
    ]


@pytest.mark.parametrize(
    "text",
    [
        atem(" to=user<|message|>Sunny.", atem_call("f")),
        atem(atem_call("f"), " to=self<|message|>Late reasoning."),
        atem(" to=user<|message|>One.", " to=user<|message|>Two."),
        atem(atem_call("f").replace(" to=f", " to=g")),
        atem(atem_call("f", ("x", "1"), ("x", "2"))),
        atem(atem_call("f") + "stray"),
        atem(" to=self<|message|>Only reasoning."),
        "no message at all",
        atem(" to=user<|message|>Sun<|start|>ny."),
    ],
)
def test_muse_any_other_shape_is_format_invalid(text: str) -> None:
    parsed = MUSE.parse(text)
    assert not parsed.format_valid
    assert "tool_calls" not in parsed.message
