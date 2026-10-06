# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``glm5_3`` family against the real, pinned GLM-5.3 tokenizer and chat template.

Marked ``tokenizer``: they run only with ``PUP_FINETUNE_TOKENIZER_TESTS=1`` and the ``hf``
extra installed, and download the tokenizer (not the model) once.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data.jsonl import read_rows, to_provider_dict
from pup_finetune.rendering import ChatMessage, TokenizedExample
from pup_finetune.rendering.families import GLM5_3
from pup_finetune.rendering.hf import (
    HFChatTemplateRenderer,
    TransformersTokenizer,
    template_messages,
)
from pup_finetune.training.datums import build_examples

pytestmark = pytest.mark.tokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
TEMPLATE_SHA256 = "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"

WEATHER_TOOL: dict[str, Any] = {
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
TYPED_ARGUMENTS: dict[str, Any] = {
    "city": "Paris",
    "nights": 3,
    "rate": 99.5,
    "breakfast": True,
    "note": None,
    "options": {"view": "sea", "floors": [2, 3]},
}
SENTINEL_TURNS: list[ChatMessage] = [
    {"role": "system", "content": "SYSTEM_SENTINEL"},
    {"role": "user", "content": "USER_ONE_SENTINEL"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "REASONING_SENTINEL",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Northvale"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "TOOL_RESULT_SENTINEL"},
    {"role": "assistant", "content": "ASSISTANT_ONE_SENTINEL"},
    {"role": "user", "content": "USER_TWO_SENTINEL"},
    {"role": "assistant", "content": "HISTORY_SENTINEL", "weight": 0},
    {"role": "user", "content": "USER_THREE_SENTINEL"},
    {"role": "assistant", "content": "ASSISTANT_TWO_SENTINEL café 🐢 東京"},
]


@pytest.fixture
def renderer(glm5_3_tokenizer: TransformersTokenizer) -> HFChatTemplateRenderer:
    return GLM5_3.renderer(tokenizer=glm5_3_tokenizer)


@pytest.fixture
def render_text(glm5_3_tokenizer: TransformersTokenizer) -> Callable[..., str]:
    """``messages`` rendered through the real template, as the renderer prepares them."""

    def render(messages: list[ChatMessage], tools: list[Any], *, generation: bool) -> str:
        return glm5_3_tokenizer.render_text(
            template_messages(messages),
            tools=tools or None,
            add_generation_prompt=generation,
            template_kwargs={},
        )

    return render


def fixture_rows() -> list[dict[str, Any]]:
    return [{"id": row.id, **to_provider_dict(row)} for row in read_rows(ROWS)]


def trained(renderer: HFChatTemplateRenderer, example: TokenizedExample) -> str:
    return renderer.decode(
        [t for t, w in zip(example.target_tokens, example.weights, strict=True) if w > 0]
    )


def tag_after(message: ChatMessage) -> str:
    return "<|observation|>" if message.get("tool_calls") else "<|user|>"


def call_turn(arguments: dict[str, Any]) -> list[ChatMessage]:
    function = {"name": "get_weather", "arguments": json.dumps(arguments)}
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": function}],
        },
    ]


def test_the_template_renders_tool_calls_results_and_reasoning_as_verified(
    render_text: Callable[..., str],
) -> None:
    messages: list[ChatMessage] = [
        *call_turn({"city": "Paris"}),
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "assistant", "content": "Sunny.", "reasoning_content": "OLD_REASONING"},
        {"role": "user", "content": "Thanks."},
    ]
    text = render_text(messages, [], generation=False)
    assert text.startswith("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>")
    assert text.endswith(
        "<|assistant|><think></think><tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call><|observation|><tool_response>sunny"
        "</tool_response><|assistant|><think>OLD_REASONING</think>Sunny.<|user|>Thanks."
    )
    prompt = render_text(messages[:1], [], generation=True)
    assert prompt.endswith("<|user|>Weather in Paris?<|assistant|><think>")


def test_the_mask_is_exactly_each_assistant_completion_and_its_role_tag(
    renderer: HFChatTemplateRenderer, render_text: Callable[..., str]
) -> None:
    for row in fixture_rows():
        messages, tools = row["messages"], row.get("tools", [])
        turns = [
            i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("weight") != 0
        ]
        examples = renderer.per_turn_examples(messages, tools)
        assert len(examples) == len(turns), row["id"]
        for turn, example in zip(turns, examples, strict=True):
            prompt = render_text(messages[:turn], tools, generation=True)
            full = render_text(messages[: turn + 1], tools, generation=False)
            completion = full[len(prompt) :] + tag_after(messages[turn])
            assert trained(renderer, example) == completion, (row["id"], turn)
            assert example.target_tokens[-1] in renderer.stop_token_ids


def test_generation_prompt_tokens_tokenize_the_prompt_string_whole(
    renderer: HFChatTemplateRenderer, render_text: Callable[..., str], glm5_3_tokenizer: Any
) -> None:
    raw = glm5_3_tokenizer._tokenizer
    for row in fixture_rows():
        messages, tools = row["messages"], row.get("tools", [])
        for turn in (i for i, m in enumerate(messages) if m["role"] == "assistant"):
            prompt = render_text(messages[:turn], tools, generation=True)
            tokens = renderer.generation_prompt_tokens(messages[:turn], tools)
            assert tokens == raw.encode(prompt, add_special_tokens=False)
            encoded = raw.apply_chat_template(
                template_messages(messages[:turn]),
                tools=tools or None,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            assert tokens == list(encoded)


def test_the_parser_reads_back_the_typed_arguments_the_template_writes(
    renderer: HFChatTemplateRenderer,
) -> None:
    [example] = renderer.per_turn_examples(call_turn(TYPED_ARGUMENTS), [WEATHER_TOOL])
    text = trained(renderer, example)
    assert "<arg_key>breakfast</arg_key><arg_value>true</arg_value>" in text
    assert text.endswith("</tool_call><|observation|>")
    parsed = renderer.parse_response(text)
    assert parsed.format_valid
    assert parsed.message["content"] == ""
    assert "reasoning_content" not in parsed.message
    [call] = parsed.message["tool_calls"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == TYPED_ARGUMENTS


def test_only_the_assistant_replies_are_trained(renderer: HFChatTemplateRenderer) -> None:
    examples = renderer.per_turn_examples(SENTINEL_TURNS, [WEATHER_TOOL])
    targets = [trained(renderer, example) for example in examples]
    assert len(targets) == 3
    assert "REASONING_SENTINEL" in targets[0]
    assert targets[0].endswith("Northvale</arg_value></tool_call><|observation|>")
    assert targets[1].endswith("ASSISTANT_ONE_SENTINEL<|user|>")
    assert targets[2].endswith("ASSISTANT_TWO_SENTINEL café 🐢 東京<|user|>")
    joined = "".join(targets)
    assert joined.count("REASONING_SENTINEL") == 1
    for context in ("SYSTEM", "USER_ONE", "TOOL_RESULT", "USER_TWO", "HISTORY", "USER_THREE"):
        assert f"{context}_SENTINEL" not in joined
    history = renderer.decode(examples[2].input_tokens)
    assert "<think>REASONING_SENTINEL</think>" in history
    assert "HISTORY_SENTINEL" in history


def test_glm5_3_is_token_prefix_stable_so_a_row_is_one_example(
    renderer: HFChatTemplateRenderer,
) -> None:
    assert renderer.prefix_stable
    rows = fixture_rows()
    plain: set[str] = set()
    for row in rows:
        messages, tools = row["messages"], row.get("tools", [])
        if any(message.get("reasoning_content") for message in messages):
            continue
        plain.add(row["id"])
        whole = renderer.supervised_example(messages, tools)
        turns = renderer.per_turn_examples(messages, tools)
        assert trained(renderer, whole) == "".join(trained(renderer, t) for t in turns), row["id"]
    result = build_examples(rows, renderer, max_seq_len=131_072)
    assert not result.rejected
    assert {row.row_id for row in result.examples if len(row.examples) == 1} >= plain


def test_the_pinned_revision_template_and_stop_tokens(renderer: HFChatTemplateRenderer) -> None:
    identity = renderer.identity()
    assert identity["tokenizer"] == "zai-org/GLM-5.3-Flash"
    assert identity["tokenizer_revision"] == GLM5_3.revision
    assert identity["chat_template_sha256"] == TEMPLATE_SHA256
    assert identity["parser"] == "glm_xml"
    assert identity["prefix_stable"] == "true"
    assert identity["transformers_version"] != "not installed"
    assert [renderer.decode([t]) for t in renderer.stop_token_ids] == [
        "<|user|>",
        "<|observation|>",
    ]
