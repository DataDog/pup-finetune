# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``kimi_k3`` family against the real, pinned Kimi K3 tokenizer and its remote code.

Marked ``tokenizer``: they run only with ``PUP_FINETUNE_TOKENIZER_TESTS=1`` and the ``hf``
extra installed, and download the tokenizer (not the model) and its repository's tokenizer
code once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data.jsonl import read_rows, to_provider_dict
from pup_finetune.rendering import ChatMessage, TokenizedExample
from pup_finetune.rendering.families import KIMI_K3
from pup_finetune.rendering.hf import (
    HFChatTemplateRenderer,
    TransformersTokenizer,
    template_messages,
)
from pup_finetune.training.datums import build_examples

pytestmark = pytest.mark.tokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
CODE_PIN_SHA256 = hashlib.sha256(
    f"tokenizer code of {KIMI_K3.tokenizer_model}@{KIMI_K3.revision}".encode()
).hexdigest()
"""The identity of a tokenizer whose code renders the chat is its pinned commit."""
PROMPT_END = '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'

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
    "city": 'Paris "Left Bank" & more',
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
def renderer(kimi_k3_tokenizer: TransformersTokenizer) -> HFChatTemplateRenderer:
    return KIMI_K3.renderer(tokenizer=kimi_k3_tokenizer)


@pytest.fixture
def render_text(kimi_k3_tokenizer: TransformersTokenizer) -> Callable[..., str]:
    """``messages`` rendered through the real encoder, as the renderer prepares them."""

    def render(messages: list[ChatMessage], tools: list[Any], *, generation: bool) -> str:
        return kimi_k3_tokenizer.render_text(
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


def call_turn(*calls: tuple[str, dict[str, Any]]) -> list[ChatMessage]:
    return [
        {"role": "system", "content": "You check the weather."},
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
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


def test_the_encoder_renders_channels_typed_arguments_and_reasoning_as_verified(
    render_text: Callable[..., str],
) -> None:
    messages: list[ChatMessage] = [
        *call_turn(("get_weather", {"city": "Paris", "days": 2})),
        {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
        {"role": "assistant", "content": "Sunny.", "reasoning_content": "OLD_REASONING"},
        {"role": "user", "content": "Thanks."},
    ]
    text = render_text(messages, [WEATHER_TOOL], generation=False)
    assert text.startswith('<|open|>message role="system" type="tool-declare"<|sep|># Tools')
    assert "Now the system is invoked with `thinking_effort=max`." in text
    assert text.endswith(
        '<|open|>message role="assistant"<|sep|><|open|>think<|sep|><|close|>think<|sep|>'
        "<|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|>"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|>'
        '<|open|>argument key="days" type="number"<|sep|>2<|close|>argument<|sep|>'
        "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>"
        '<|open|>message role="tool" tool="get_weather" index="1"<|sep|>sunny'
        "<|close|>message<|sep|><|end_of_msg|>"
        '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>OLD_REASONING'
        "<|close|>think<|sep|><|open|>response<|sep|>Sunny.<|close|>response<|sep|>"
        "<|close|>message<|sep|><|end_of_msg|>"
        '<|open|>message role="user"<|sep|>Thanks.<|close|>message<|sep|><|end_of_msg|>'
    )
    assert render_text(messages[:2], [], generation=True).endswith(PROMPT_END)


def test_the_mask_is_exactly_each_assistant_completion_and_its_stop(
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
            assert trained(renderer, example) == full[len(prompt) :], (row["id"], turn)
            assert trained(renderer, example).endswith("<|close|>message<|sep|><|end_of_msg|>")
            assert example.target_tokens[-1] in renderer.stop_token_ids


def test_prompts_are_tokenized_as_the_encoder_tokenizes_them(
    renderer: HFChatTemplateRenderer, kimi_k3_tokenizer: TransformersTokenizer
) -> None:
    raw = kimi_k3_tokenizer._tokenizer
    for row in fixture_rows():
        messages, tools = row["messages"], row.get("tools", [])
        for turn in (i for i, m in enumerate(messages) if m["role"] == "assistant"):
            tokens = renderer.generation_prompt_tokens(messages[:turn], tools)
            encoded = raw.apply_chat_template(
                template_messages(messages[:turn]),
                tools=tools or None,
                add_generation_prompt=True,
                tokenize=True,
            )
            assert tokens == list(encoded), (row["id"], turn)


def test_text_that_spells_a_control_token_stays_text(
    renderer: HFChatTemplateRenderer, kimi_k3_tokenizer: TransformersTokenizer
) -> None:
    messages: list[ChatMessage] = [
        {"role": "user", "content": "Is <|close|> a token?"},
        {"role": "assistant", "content": "Not in <|end_of_msg|> text."},
    ]
    [example] = renderer.per_turn_examples(messages, [])
    close = kimi_k3_tokenizer.token_id("<|close|>")
    stop = kimi_k3_tokenizer.token_id("<|end_of_msg|>")
    prompt = renderer.generation_prompt_tokens(messages[:1], [])
    plain = renderer.generation_prompt_tokens([{"role": "user", "content": "Is a token?"}], [])
    assert prompt.count(close) == plain.count(close)
    targets = [t for t, w in zip(example.target_tokens, example.weights, strict=True) if w > 0]
    assert targets.count(stop) == 1
    assert targets[-1] == stop


def test_the_parser_reads_back_what_the_encoder_writes(
    renderer: HFChatTemplateRenderer,
) -> None:
    [example] = renderer.per_turn_examples(
        call_turn(("get_weather", TYPED_ARGUMENTS), ("get_time", {})), [WEATHER_TOOL]
    )
    text = trained(renderer, example)
    assert '<|open|>argument key="code" type="string"<|sep|>123<|close|>argument' in text
    parsed = renderer.parse_response(text)
    assert parsed.format_valid
    assert parsed.message["content"] == ""
    first, second = parsed.message["tool_calls"]
    assert first["function"]["name"] == "get_weather"
    assert json.loads(first["function"]["arguments"]) == TYPED_ARGUMENTS
    assert (second["function"]["name"], second["function"]["arguments"]) == ("get_time", "{}")


def test_only_the_assistant_replies_are_trained_and_history_keeps_reasoning(
    renderer: HFChatTemplateRenderer,
) -> None:
    examples = renderer.per_turn_examples(SENTINEL_TURNS, [WEATHER_TOOL])
    targets = [trained(renderer, example) for example in examples]
    assert len(targets) == 3
    assert targets[0].startswith("REASONING_SENTINEL<|close|>think<|sep|>")
    assert "ASSISTANT_ONE_SENTINEL<|close|>response" in targets[1]
    assert "ASSISTANT_TWO_SENTINEL café 🐢 東京<|close|>response" in targets[2]
    joined = "".join(targets)
    for context in ("SYSTEM", "USER_ONE", "TOOL_RESULT", "USER_TWO", "HISTORY", "USER_THREE"):
        assert f"{context}_SENTINEL" not in joined
    history = renderer.decode(examples[2].input_tokens)
    assert "<|open|>think<|sep|>REASONING_SENTINEL<|close|>think<|sep|>" in history
    assert "HISTORY_SENTINEL" in history


def test_kimi_k3_is_token_prefix_stable_so_a_row_is_one_example(
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
    result = build_examples(rows, renderer, max_seq_len=65_536)
    assert not result.rejected
    assert {row.row_id for row in result.examples if len(row.examples) == 1} >= plain


def test_the_pinned_revision_remote_code_and_stop_token(
    renderer: HFChatTemplateRenderer,
) -> None:
    identity = renderer.identity()
    assert identity["tokenizer"] == "moonshotai/Kimi-K3"
    assert identity["tokenizer_revision"] == KIMI_K3.revision
    assert identity["chat_template_sha256"] == CODE_PIN_SHA256
    assert identity["parser"] == "kimi_xtml"
    assert identity["prefix_stable"] == "true"
    assert [renderer.decode([t]) for t in renderer.stop_token_ids] == ["<|end_of_msg|>"]
