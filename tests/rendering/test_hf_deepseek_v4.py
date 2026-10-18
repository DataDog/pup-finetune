# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
# DeepSeek's special tokens are spelled with U+FF5C, the fullwidth vertical line.
# ruff: noqa: RUF001
"""The ``deepseek_v4`` family against the real, pinned DeepSeek-V4 tokenizer and encoder.

The repository ships its chat format as Python (``encoding/encoding_dsv4.py``) and no chat
template, so the family renders with :class:`~pup_finetune.rendering.encoders.DeepSeekV4Encoder`.
These tests hold that encoder to DeepSeek's: the official test vectors and the reference
encoder itself, both downloaded from the pinned revision.

Marked ``tokenizer``: they run only with ``PUP_FINETUNE_TOKENIZER_TESTS=1`` and the ``hf``
extra installed, and download the tokenizer (not the model) and the encoder files once.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pup_finetune.data.jsonl import read_rows, to_provider_dict
from pup_finetune.rendering import ChatMessage, RendererError, TokenizedExample
from pup_finetune.rendering.base import RendererErrorCode
from pup_finetune.rendering.encoders import DeepSeekV4Encoder
from pup_finetune.rendering.families import DEEPSEEK_V4
from pup_finetune.rendering.hf import (
    HFChatTemplateRenderer,
    TransformersTokenizer,
    template_messages,
)
from pup_finetune.training.datums import build_examples

pytestmark = pytest.mark.tokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
EOS = "<｜end▁of▁sentence｜>"
ENCODER_SHA256 = "cce07c81d22acb48566bdee7275d0bce636860c69f6ae0cdfa60e1dd5b8472c2"

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


def _download(name: str) -> Path:
    hub = pytest.importorskip("huggingface_hub")
    return Path(
        hub.hf_hub_download(DEEPSEEK_V4.tokenizer_model, name, revision=DEEPSEEK_V4.revision)
    )


@pytest.fixture(scope="module")
def official() -> ModuleType:
    """DeepSeek's reference encoder, ``encoding/encoding_dsv4.py`` at the pinned revision."""
    spec = importlib.util.spec_from_file_location(
        "official_encoding_dsv4", _download("encoding/encoding_dsv4.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def renderer(deepseek_v4_tokenizer: TransformersTokenizer) -> HFChatTemplateRenderer:
    return DEEPSEEK_V4.renderer(tokenizer=deepseek_v4_tokenizer)


def render(messages: list[ChatMessage], tools: list[Any]) -> str:
    return DeepSeekV4Encoder().render(
        template_messages(messages), tools or None, add_generation_prompt=False
    )


def official_render(official: ModuleType, messages: list[ChatMessage], tools: list[Any]) -> str:
    """The reference encoder's text, given the tools on the system message as it takes them."""
    prepared = [
        {key: value for key, value in message.items() if key != "weight"}
        for message in copy.deepcopy(messages)
    ]
    if tools:
        if prepared[0]["role"] == "system":
            prepared[0]["tools"] = tools
        else:
            prepared.insert(0, {"role": "system", "content": "", "tools": tools})
    text: str = official.encode_messages(prepared, thinking_mode="thinking")
    return text


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


@pytest.mark.parametrize("case", [1, 2])
def test_the_encoder_reproduces_deepseek_s_test_vectors(case: int) -> None:
    given = json.loads(_download(f"encoding/tests/test_input_{case}.json").read_text("utf-8"))
    messages, tools = (given["messages"], given["tools"]) if case == 1 else (given, [])
    expected = _download(f"encoding/tests/test_output_{case}.txt").read_text("utf-8")
    assert render(messages, tools) == expected


def test_the_encoder_matches_deepseek_s_on_every_prefix_of_every_row(
    official: ModuleType,
) -> None:
    rows = fixture_rows()
    rows.append(
        {
            "id": "consecutive-users-and-results-out-of-order",
            "messages": [
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "Hello.", "reasoning_content": "Greet."},
                {"role": "user", "content": "Two lookups."},
                {"role": "user", "content": "Quickly."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        },
                        {
                            "id": "b",
                            "type": "function",
                            "function": {"name": "g", "arguments": '{"x": [1, {"y": null}]}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "b", "content": "B"},
                {"role": "tool", "tool_call_id": "a", "content": "A"},
                {"role": "assistant", "content": "Done."},
            ],
        }
    )
    for row in rows:
        messages, tools = row["messages"], row.get("tools", [])
        for end in range(1, len(messages) + 1):
            expected = official_render(official, messages[:end], tools)
            assert render(messages[:end], tools) == expected, (row["id"], end)
        # Without tools, reasoning before the last user turn is dropped, as DeepSeek drops it.
        assert render(messages, []) == official_render(official, messages, []), row["id"]


def test_the_mask_is_exactly_each_assistant_completion_and_its_stop(
    renderer: HFChatTemplateRenderer,
) -> None:
    for row in fixture_rows():
        messages, tools = row["messages"], row.get("tools", [])
        turns = [
            i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("weight") != 0
        ]
        examples = renderer.per_turn_examples(messages, tools)
        assert len(examples) == len(turns), row["id"]
        for turn, example in zip(turns, examples, strict=True):
            prompt = render(messages[:turn], tools)
            assert prompt.endswith("<｜Assistant｜><think>")
            full = render(messages[: turn + 1], tools)
            assert trained(renderer, example) == full[len(prompt) :], (row["id"], turn)
            assert trained(renderer, example).endswith(EOS)
            assert example.target_tokens[-1] in renderer.stop_token_ids


def test_prompts_are_the_encoded_text_tokenized_whole(
    renderer: HFChatTemplateRenderer, deepseek_v4_tokenizer: TransformersTokenizer
) -> None:
    raw = deepseek_v4_tokenizer._tokenizer
    for row in fixture_rows():
        messages, tools = row["messages"], row.get("tools", [])
        for turn in (i for i, m in enumerate(messages) if m["role"] == "assistant"):
            tokens = renderer.generation_prompt_tokens(messages[:turn], tools)
            assert tokens == raw.encode(render(messages[:turn], tools), add_special_tokens=False)
            assert tokens[0] == raw.convert_tokens_to_ids("<｜begin▁of▁sentence｜>")


def test_the_parser_reads_back_what_the_encoder_writes(
    renderer: HFChatTemplateRenderer,
) -> None:
    [example] = renderer.per_turn_examples(
        call_turn(("get_weather", TYPED_ARGUMENTS), ("get_time", {})), [WEATHER_TOOL]
    )
    text = trained(renderer, example)
    assert '<｜DSML｜parameter name="code" string="true">123</｜DSML｜parameter>' in text
    parsed = renderer.parse_response(text)
    assert parsed.format_valid
    assert parsed.message["content"] == ""
    assert "reasoning_content" not in parsed.message
    first, second = parsed.message["tool_calls"]
    assert first["function"]["name"] == "get_weather"
    assert json.loads(first["function"]["arguments"]) == TYPED_ARGUMENTS
    assert (second["function"]["name"], second["function"]["arguments"]) == ("get_time", "{}")


def test_only_the_assistant_replies_are_trained(renderer: HFChatTemplateRenderer) -> None:
    examples = renderer.per_turn_examples(SENTINEL_TURNS, [WEATHER_TOOL])
    targets = [trained(renderer, example) for example in examples]
    assert len(targets) == 3
    assert targets[0].startswith("REASONING_SENTINEL</think>")
    assert targets[1] == f"</think>ASSISTANT_ONE_SENTINEL{EOS}"
    assert targets[2] == f"</think>ASSISTANT_TWO_SENTINEL café 🐢 東京{EOS}"
    joined = "".join(targets)
    for context in ("SYSTEM", "USER_ONE", "TOOL_RESULT", "USER_TWO", "HISTORY", "USER_THREE"):
        assert f"{context}_SENTINEL" not in joined
    # With tools, history keeps every turn's reasoning.
    history = renderer.decode(examples[2].input_tokens)
    assert "<think>REASONING_SENTINEL</think>" in history
    assert "HISTORY_SENTINEL" in history


def test_deepseek_v4_is_token_prefix_stable_with_tools(renderer: HFChatTemplateRenderer) -> None:
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


def test_without_tools_a_multi_turn_row_renders_per_turn(
    renderer: HFChatTemplateRenderer,
) -> None:
    messages: list[ChatMessage] = [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Bye."},
        {"role": "assistant", "content": "Goodbye."},
    ]
    with pytest.raises(RendererError) as caught:
        renderer.supervised_example(messages, [])
    assert caught.value.code == RendererErrorCode.NOT_PREFIX_STABLE
    first, second = renderer.per_turn_examples(messages, [])
    assert trained(renderer, first) == f"</think>Hello.{EOS}"
    # In the later turn's history the earlier reply opens on </think>, its reasoning dropped.
    assert "<｜User｜>Hi.<｜Assistant｜></think>Hello." in renderer.decode(second.input_tokens)


def test_the_pinned_revision_encoder_and_stop_token(renderer: HFChatTemplateRenderer) -> None:
    identity = renderer.identity()
    assert identity["tokenizer"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert identity["tokenizer_revision"] == DEEPSEEK_V4.revision
    assert identity["chat_template_sha256"] == ENCODER_SHA256
    assert identity["parser"] == "deepseek_dsml"
    assert identity["prefix_stable"] == "true"
    assert [renderer.decode([t]) for t in renderer.stop_token_ids] == [EOS]


def _call(index: int, name: str = "f", arguments: str = "{}") -> dict[str, Any]:
    return {
        "id": f"c{index}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


F_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "f",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
    },
}
ODD_SHAPES: dict[str, list[ChatMessage]] = {
    "system_then_assistant": [
        {"role": "system", "content": "S"},
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
    ],
    "assistant_first": [
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "B"},
    ],
    "two_assistants": [
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A", "reasoning_content": "R1"},
        {"role": "assistant", "content": "B", "reasoning_content": "R2"},
    ],
    "two_assistants_then_user": [
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A", "reasoning_content": "R1"},
        {"role": "assistant", "content": "B", "reasoning_content": "R2"},
        {"role": "user", "content": "V"},
        {"role": "assistant", "content": "C"},
    ],
    "calls": [
        {"role": "user", "content": "U"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "R1",
            "tool_calls": [_call(1), _call(2, "g", '{"x": 1}')],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "T2"},
        {"role": "tool", "tool_call_id": "c1", "content": "T1"},
        {"role": "assistant", "content": "Done.", "reasoning_content": "R2"},
    ],
    "calls_over_two_user_turns": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "", "reasoning_content": "R1", "tool_calls": [_call(1)]},
        {"role": "tool", "tool_call_id": "c1", "content": "T1"},
        {"role": "assistant", "content": "Mid.", "reasoning_content": "R2"},
        {"role": "user", "content": "V"},
        {"role": "assistant", "content": "", "reasoning_content": "R3", "tool_calls": [_call(3)]},
        {"role": "tool", "tool_call_id": "c3", "content": "T3"},
        {"role": "assistant", "content": "End.", "reasoning_content": "R4"},
    ],
    "user_text_after_a_tool_result": [
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "", "tool_calls": [_call(1)]},
        {"role": "tool", "tool_call_id": "c1", "content": "T1"},
        {"role": "user", "content": "More"},
        {"role": "assistant", "content": "Ok.", "reasoning_content": "R"},
    ],
}
"""Shapes the fixture rows lack: a turn with no user or tool message before it, and tool
calls in rows that declare no tools, where tool results count as user turns."""


@pytest.mark.parametrize("shape", sorted(ODD_SHAPES))
@pytest.mark.parametrize("tools", [[], [F_TOOL]], ids=["no_tools", "tools"])
def test_the_encoder_matches_deepseek_s_on_every_prefix_of_odd_shapes(
    official: ModuleType, shape: str, tools: list[Any]
) -> None:
    messages = ODD_SHAPES[shape]
    for end in range(1, len(messages) + 1):
        expected = official_render(official, messages[:end], tools)
        assert render(messages[:end], tools) == expected, end


@pytest.mark.parametrize("shape", ["system_then_assistant", "assistant_first", "two_assistants"])
def test_a_trained_turn_with_no_user_or_tool_message_before_it_is_refused(
    renderer: HFChatTemplateRenderer, shape: str
) -> None:
    with pytest.raises(RendererError, match="user or tool message") as caught:
        renderer.per_turn_examples(ODD_SHAPES[shape], [])
    assert caught.value.code == RendererErrorCode.UNSUPPORTED_MESSAGE
