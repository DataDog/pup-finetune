# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``muse_glimmer`` family against the real, pinned Muse Glimmer tokenizer and template.

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
from pup_finetune.rendering import ChatMessage, RendererError, TokenizedExample
from pup_finetune.rendering.base import RendererErrorCode
from pup_finetune.rendering.families import MUSE_GLIMMER
from pup_finetune.rendering.hf import (
    HFChatTemplateRenderer,
    TransformersTokenizer,
    template_messages,
)
from pup_finetune.training.datums import build_examples

pytestmark = pytest.mark.tokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
TEMPLATE_SHA256 = "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678"
REFUSED_ROWS = {"row-calendar-003", "row-units-001"}
"""Fixture rows the template cannot render faithfully: text beside tool calls, and no
system message."""

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current conditions for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "code": {"type": "string"}},
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
    "code": "123",
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
def renderer(muse_glimmer_tokenizer: TransformersTokenizer) -> HFChatTemplateRenderer:
    return MUSE_GLIMMER.renderer(tokenizer=muse_glimmer_tokenizer)


@pytest.fixture
def render_text(muse_glimmer_tokenizer: TransformersTokenizer) -> Callable[..., str]:
    """``messages`` rendered through the real template, as the renderer prepares them."""

    def render(messages: list[ChatMessage], tools: list[Any], *, generation: bool) -> str:
        return muse_glimmer_tokenizer.render_text(
            template_messages(messages),
            tools=tools or None,
            add_generation_prompt=generation,
            template_kwargs={},
        )

    return render


def fixture_rows() -> list[dict[str, Any]]:
    return [{"id": row.id, **to_provider_dict(row)} for row in read_rows(ROWS)]


def renderable_rows() -> list[dict[str, Any]]:
    return [row for row in fixture_rows() if row["id"] not in REFUSED_ROWS]


def trained(renderer: HFChatTemplateRenderer, example: TokenizedExample) -> str:
    return renderer.decode(
        [t for t, w in zip(example.target_tokens, example.weights, strict=True) if w > 0]
    )


def call_turn(*calls: tuple[str, dict[str, Any]], content: str = "") -> list[ChatMessage]:
    return [
        {"role": "system", "content": "You check the weather."},
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
                for index, (name, arguments) in enumerate(calls)
            ],
        },
    ]


def test_the_template_renders_messages_to_self_user_and_tools_as_verified(
    render_text: Callable[..., str],
) -> None:
    messages: list[ChatMessage] = [
        *call_turn(("get_weather", {"city": "Paris", "days": 2})),
        {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
        {"role": "assistant", "content": "Sunny.", "reasoning_content": "OLD_REASONING"},
        {"role": "user", "content": "Thanks."},
    ]
    text = render_text(messages, [WEATHER_TOOL], generation=False)
    assert text.startswith("<|begin_of_text|><|start|>system<|message|>You check the weather.")
    assert '# Valid recipients: "self", "get_weather.*", "user".<|eot|>' in text
    assert text.endswith(
        "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n"
        '<atem:invoke name="get_weather">\n<atem:parameter name="city">Paris</atem:parameter>\n'
        '<atem:parameter name="days">2</atem:parameter>\n</atem:invoke>\n'
        "</atem:function_calls><|eot|>"
        '<|start|>tool get_weather<|message|><tool_output name="get_weather">\nsunny\n'
        "</tool_output><|eot|>"
        "<|start|>assistant to=self<|message|>OLD_REASONING<|eom|>"
        "<|start|>assistant to=user<|message|>Sunny.<|eot|>"
        "<|start|>user<|message|>Thanks.<|eot|>"
    )
    assert render_text(messages[:2], [], generation=True).endswith("<|start|>assistant")


def test_the_mask_is_exactly_each_assistant_completion_and_its_stop(
    renderer: HFChatTemplateRenderer, render_text: Callable[..., str]
) -> None:
    for row in renderable_rows():
        messages, tools = row["messages"], row.get("tools", [])
        turns = [
            i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("weight") != 0
        ]
        examples = renderer.per_turn_examples(messages, tools)
        assert len(examples) == len(turns), row["id"]
        for turn, example in zip(turns, examples, strict=True):
            prompt = render_text(messages[:turn], tools, generation=True)
            full = render_text(messages[: turn + 1], tools, generation=False)
            assert trained(renderer, example) == full[len(prompt) :], (row["id"], turn)
            assert trained(renderer, example).endswith("<|eot|>")
            assert example.target_tokens[-1] in renderer.stop_token_ids


def test_prompts_are_tokenized_as_the_template_tokenizes_them(
    renderer: HFChatTemplateRenderer, muse_glimmer_tokenizer: TransformersTokenizer
) -> None:
    raw = muse_glimmer_tokenizer._tokenizer
    for row in renderable_rows():
        messages, tools = row["messages"], row.get("tools", [])
        for turn in (i for i, m in enumerate(messages) if m["role"] == "assistant"):
            tokens = renderer.generation_prompt_tokens(messages[:turn], tools)
            encoded = raw.apply_chat_template(
                template_messages(messages[:turn]),
                tools=tools or None,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            assert tokens == list(encoded), (row["id"], turn)


def test_the_parser_reads_back_what_the_template_writes(
    renderer: HFChatTemplateRenderer,
) -> None:
    messages = call_turn(("get_weather", TYPED_ARGUMENTS), ("get_time", {}))
    messages[2]["reasoning_content"] = "Two lookups."
    [example] = renderer.per_turn_examples(messages, [WEATHER_TOOL])
    text = trained(renderer, example)
    assert text.startswith(" to=self<|message|>Two lookups.<|eom|><|start|>assistant")
    assert '<atem:parameter name="breakfast">true</atem:parameter>' in text
    parsed = renderer.parse_response(text, [WEATHER_TOOL])
    assert parsed.format_valid
    assert parsed.message["content"] == ""
    assert parsed.message["reasoning_content"] == "Two lookups."
    first, second = parsed.message["tool_calls"]
    assert first["function"]["name"] == "get_weather"
    # The declared string type keeps "123" as text.
    assert json.loads(first["function"]["arguments"]) == TYPED_ARGUMENTS
    assert (second["function"]["name"], second["function"]["arguments"]) == ("get_time", "{}")


def test_only_the_assistant_replies_are_trained_and_history_keeps_reasoning(
    renderer: HFChatTemplateRenderer,
) -> None:
    examples = renderer.per_turn_examples(SENTINEL_TURNS, [WEATHER_TOOL])
    targets = [trained(renderer, example) for example in examples]
    assert len(targets) == 3
    assert targets[0].startswith(" to=self<|message|>REASONING_SENTINEL<|eom|>")
    assert targets[1] == " to=user<|message|>ASSISTANT_ONE_SENTINEL<|eot|>"
    assert targets[2] == " to=user<|message|>ASSISTANT_TWO_SENTINEL café 🐢 東京<|eot|>"
    joined = "".join(targets)
    for context in ("SYSTEM", "USER_ONE", "TOOL_RESULT", "USER_TWO", "HISTORY", "USER_THREE"):
        assert f"{context}_SENTINEL" not in joined
    history = renderer.decode(examples[2].input_tokens)
    assert "to=self<|message|>REASONING_SENTINEL<|eom|>" in history
    assert "HISTORY_SENTINEL" in history


@pytest.mark.parametrize(
    ("messages", "reason"),
    [
        (
            [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}],
            "current date",
        ),
        (call_turn(("get_weather", {"city": "Paris"}), content="Checking."), "drops the text"),
    ],
)
def test_a_conversation_the_template_cannot_render_faithfully_is_refused(
    renderer: HFChatTemplateRenderer, messages: list[ChatMessage], reason: str
) -> None:
    for render in (renderer.per_turn_examples, renderer.supervised_example):
        with pytest.raises(RendererError, match=reason) as caught:
            render(messages, [WEATHER_TOOL])
        assert caught.value.code == RendererErrorCode.UNSUPPORTED_MESSAGE
    result = build_examples(fixture_rows(), renderer, max_seq_len=131_072)
    assert {row.row_id for row in result.rejected} == REFUSED_ROWS


def test_muse_glimmer_is_token_prefix_stable_so_a_row_is_one_example(
    renderer: HFChatTemplateRenderer,
) -> None:
    assert renderer.prefix_stable
    plain: set[str] = set()
    for row in renderable_rows():
        messages, tools = row["messages"], row.get("tools", [])
        if any(message.get("reasoning_content") for message in messages):
            continue
        plain.add(row["id"])
        whole = renderer.supervised_example(messages, tools)
        turns = renderer.per_turn_examples(messages, tools)
        assert trained(renderer, whole) == "".join(trained(renderer, t) for t in turns), row["id"]
    result = build_examples(renderable_rows(), renderer, max_seq_len=131_072)
    assert not result.rejected
    assert {row.row_id for row in result.examples if len(row.examples) == 1} >= plain


def test_the_pinned_revision_template_and_stop_token(renderer: HFChatTemplateRenderer) -> None:
    identity = renderer.identity()
    assert identity["tokenizer"] == "meta-models/Muse-Glimmer-30B"
    assert identity["tokenizer_revision"] == MUSE_GLIMMER.revision
    assert identity["chat_template_sha256"] == TEMPLATE_SHA256
    assert identity["parser"] == "muse_atem"
    assert identity["prefix_stable"] == "true"
    assert [renderer.decode([t]) for t in renderer.stop_token_ids] == ["<|eot|>"]
