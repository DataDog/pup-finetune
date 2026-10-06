# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``qwen3_8`` family against the real, pinned Qwen3.8 tokenizer and chat template.

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
from pup_finetune.rendering import ChatMessage, RendererError, RendererErrorCode, TokenizedExample
from pup_finetune.rendering.families import QWEN3_8
from pup_finetune.rendering.hf import (
    HFChatTemplateRenderer,
    TransformersTokenizer,
    template_messages,
)
from pup_finetune.training.datums import build_examples

pytestmark = pytest.mark.tokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

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
TOOL_TURN: list[ChatMessage] = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny in Paris."},
]


@pytest.fixture
def renderer(qwen3_8_tokenizer: TransformersTokenizer) -> HFChatTemplateRenderer:
    return QWEN3_8.renderer(tokenizer=qwen3_8_tokenizer)


@pytest.fixture
def render_text(
    qwen3_8_tokenizer: TransformersTokenizer,
) -> Callable[..., str]:
    """``messages`` rendered through the real template, as the renderer prepares them."""

    def render(messages: list[ChatMessage], tools: list[Any], *, generation: bool) -> str:
        return qwen3_8_tokenizer.render_text(
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


def test_the_template_renders_tool_calls_and_results_as_verified(
    render_text: Callable[..., str],
) -> None:
    text = render_text(TOOL_TURN, [WEATHER_TOOL], generation=False)
    assert (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
    ) in text
    assert "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n" in text
    prompt = render_text(TOOL_TURN[:1], [WEATHER_TOOL], generation=True)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


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
            completion = full[len(prompt) :].removesuffix("\n")
            assert completion.endswith("<|im_end|>")
            assert trained(renderer, example) == completion, (row["id"], turn)
            assert example.target_tokens[-1] in renderer.stop_token_ids


def test_generation_prompt_tokens_tokenize_the_prompt_string_whole(
    renderer: HFChatTemplateRenderer, render_text: Callable[..., str], qwen3_8_hf_tokenizer: Any
) -> None:
    raw = qwen3_8_hf_tokenizer
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


def test_the_parser_round_trips_the_rendered_tool_call_turn(
    renderer: HFChatTemplateRenderer,
) -> None:
    example = renderer.per_turn_examples(TOOL_TURN[:2], [WEATHER_TOOL])[0]
    parsed = renderer.parse_response(trained(renderer, example))
    assert parsed.format_valid
    assert parsed.message["content"] == ""
    assert "reasoning_content" not in parsed.message
    [call] = parsed.message["tool_calls"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


def test_qwen3_8_is_not_token_prefix_stable_so_rows_render_per_turn(
    renderer: HFChatTemplateRenderer,
) -> None:
    assert not renderer.prefix_stable
    with pytest.raises(RendererError) as excinfo:
        renderer.supervised_example(TOOL_TURN, [WEATHER_TOOL])
    assert excinfo.value.code is RendererErrorCode.NOT_PREFIX_STABLE
    result = build_examples(fixture_rows(), renderer, max_seq_len=131_072)
    assert not result.rejected
    assert sum(len(row.examples) for row in result.examples) > len(result.examples)


def test_the_pinned_revision_and_stop_token(renderer: HFChatTemplateRenderer) -> None:
    identity = renderer.identity()
    assert identity["tokenizer"] == "Qwen/Qwen3.8-27B"
    assert identity["tokenizer_revision"] == QWEN3_8.revision
    assert identity["transformers_version"] != "not installed"
    assert renderer.decode(renderer.stop_token_ids) == "<|im_end|>"
