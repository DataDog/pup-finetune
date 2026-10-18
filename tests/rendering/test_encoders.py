# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
# DeepSeek's special tokens are spelled with U+FF5C, the fullwidth vertical line.
# ruff: noqa: RUF001
"""The DeepSeek-V4 encoder on small conversations, offline. ``test_hf_deepseek_v4.py`` holds
it to DeepSeek's own encoder and test vectors at the pinned revision."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from pup_finetune.rendering import ChatMessage, RendererError, RendererErrorCode
from pup_finetune.rendering.encoders import ENCODERS, DeepSeekV4Encoder, get_encoder
from pup_finetune.rendering.families import DEEPSEEK_V4
from pup_finetune.rendering.hf import HFChatTemplateRenderer
from pup_finetune.training.datums import build_examples
from tests.fakes.tokenizer import DEEPSEEK_SPECIALS, FakeChatTokenizer

ENCODER = DeepSeekV4Encoder()
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
USER, ASSISTANT = "<｜User｜>", "<｜Assistant｜>"
TOOL: dict[str, Any] = {"type": "function", "function": {"name": "f", "parameters": {}}}


def call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def render(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
    return ENCODER.render(messages, tools, add_generation_prompt=True)


def test_a_prompt_opens_the_assistant_turn_in_thinking_mode() -> None:
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi."}]
    assert render(messages) == f"{BOS}Be brief.{USER}Hi.{ASSISTANT}<think>"


def test_without_tools_earlier_turns_lose_their_reasoning() -> None:
    messages = [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello.", "reasoning_content": "Greet."},
        {"role": "user", "content": "Bye."},
        {"role": "assistant", "content": "Goodbye.", "reasoning_content": "Part."},
    ]
    assert render(messages) == (
        f"{BOS}{USER}Hi.{ASSISTANT}</think>Hello.{EOS}"
        f"{USER}Bye.{ASSISTANT}<think>Part.</think>Goodbye.{EOS}"
    )


def test_tools_go_on_the_system_message_and_keep_every_turn_s_reasoning() -> None:
    messages = [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello.", "reasoning_content": "Greet."},
        {"role": "user", "content": "Bye."},
    ]
    text = render(messages, [TOOL])
    assert text.startswith(f"{BOS}\n\n## Tools\n")
    assert '\n{"name": "f", "parameters": {}}\n' in text
    assert text.endswith(
        f"{USER}Hi.{ASSISTANT}<think>Greet.</think>Hello.{EOS}{USER}Bye.{ASSISTANT}<think>"
    )
    with_system = render([{"role": "system", "content": "S"}, *messages], [TOOL])
    assert with_system.startswith(f"{BOS}S\n\n## Tools\n")


def test_tool_calls_are_dsml_and_results_join_the_user_turn_in_call_order() -> None:
    messages = [
        {"role": "user", "content": "Go."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("a", "f", city="Paris", days=2), call("b", "g")],
        },
        {"role": "tool", "tool_call_id": "b", "content": "B"},
        {"role": "tool", "tool_call_id": "a", "content": "A"},
    ]
    assert render(messages, [TOOL]).split(f"{USER}Go.{ASSISTANT}<think>")[1] == (
        "</think>\n\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="f">\n'
        '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="g">\n\n</｜DSML｜invoke>\n'
        f"</｜DSML｜tool_calls>{EOS}"
        f"{USER}<tool_result>A</tool_result>\n\n<tool_result>B</tool_result>{ASSISTANT}<think>"
    )


def test_encoders_are_registered_by_name() -> None:
    assert isinstance(get_encoder("deepseek_v4"), DeepSeekV4Encoder)
    assert set(ENCODERS) == {"deepseek_v4"}
    with pytest.raises(KeyError, match="known: deepseek_v4"):
        get_encoder("nope")


def deepseek_renderer() -> HFChatTemplateRenderer:
    """The ``deepseek_v4`` family behind a fake tokenizer: DeepSeek's specials are one id."""
    return DEEPSEEK_V4.renderer(tokenizer=FakeChatTokenizer(specials=DEEPSEEK_SPECIALS))


TWO_TURNS: list[ChatMessage] = [
    {"role": "user", "content": "Hi."},
    {"role": "assistant", "content": "Hello."},
    {"role": "user", "content": "Bye."},
    {"role": "assistant", "content": "Goodbye."},
]


def test_the_renderer_tokenizes_the_encoder_s_text_and_names_it_in_its_identity() -> None:
    r = deepseek_renderer()
    prompt = r.generation_prompt_tokens(TWO_TURNS[:1], [])
    assert r.decode(prompt) == f"{BOS}{USER}Hi.{ASSISTANT}<think>"
    assert len(prompt) == 1 + 1 + len("Hi.") + 1 + 1
    pinned = f"pup_finetune.rendering.encoders:deepseek_v4@{DeepSeekV4Encoder.version}"
    assert r.identity()["chat_template_sha256"] == hashlib.sha256(pinned.encode()).hexdigest()
    assert [r.decode([token]) for token in r.stop_token_ids] == [EOS]


def test_without_tools_a_row_training_every_turn_falls_back_to_per_turn() -> None:
    r = deepseek_renderer()
    with pytest.raises(RendererError) as excinfo:
        r.supervised_example(TWO_TURNS, [])
    assert excinfo.value.code == RendererErrorCode.NOT_PREFIX_STABLE
    first, second = r.per_turn_examples(TWO_TURNS, [])
    assert f"{USER}Hi.{ASSISTANT}<think></think>Hello." in r.decode(first.input_tokens)
    assert f"{USER}Hi.{ASSISTANT}</think>Hello.{EOS}" in r.decode(second.input_tokens)
    result = build_examples([{"messages": TWO_TURNS, "tools": []}], r, max_seq_len=4096)
    assert result.rejected == []
    assert result.examples[0].examples == (first, second)


def test_without_tools_a_row_training_only_its_last_turn_is_one_example() -> None:
    # The untrained first reply renders after </think> in the whole row but after <think>
    # in its own prompt; it carries no loss, so the whole row trains exactly the last turn.
    r = deepseek_renderer()
    messages = [TWO_TURNS[0], {**TWO_TURNS[1], "weight": 0}, *TWO_TURNS[2:]]
    (turn,) = r.per_turn_examples(messages, [])
    assert r.supervised_example(messages, []) == turn
