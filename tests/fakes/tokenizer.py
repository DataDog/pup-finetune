# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A fake :class:`~pup_finetune.rendering.hf.ChatTokenizer`: a Python chat template, no network.

Each special token is one id and every other character is its own id (its code point)::

    tokenizer = FakeChatTokenizer()  # Qwen3.8-like, token prefix-stable
    tokenizer = FakeChatTokenizer(merge_newlines=True)  # "\\n\\n" encodes as one token
    tokenizer = FakeChatTokenizer(template=role_tag_template)  # any template function
    tokenizer = FakeChatTokenizer(template=glm5_like)  # GLM-5-like, role tags as stops

:func:`qwen3_8_like` follows the real Qwen3.8 template: ``<|im_start|>role\\n...<|im_end|>\\n``
turns, ``<think>`` blocks on assistant turns (kept in history unless the template argument
``preserve_thinking`` is false), XML tool calls, tool results as ``<tool_response>`` user
turns and a generation prompt that opens ``<think>\\n``. With ``merge_newlines`` the empty
reasoning block's newline merges with that prompt's trailing one, as BPE does. Like a real
template, it raises on an empty conversation and on tool arguments that are not a mapping.
Every render is recorded in ``calls``. :func:`glm5_like` follows the real GLM-5 template.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|system|>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
)
_FIRST_SPECIAL_ID = 0x110000  # above every Unicode code point
SPECIAL_IDS = {token: _FIRST_SPECIAL_ID + index for index, token in enumerate(SPECIAL_TOKENS)}
DOUBLE_NEWLINE_ID = _FIRST_SPECIAL_ID + len(SPECIAL_TOKENS)
_TEXT_OF = {**{v: k for k, v in SPECIAL_IDS.items()}, DOUBLE_NEWLINE_ID: "\n\n"}

Template = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]] | None, bool, Mapping[str, Any]], str
]
"""``(messages, tools, add_generation_prompt, template_kwargs) -> text``."""


class FakeChatTokenizer:
    """A deterministic chat tokenizer for tests. See the module docstring."""

    def __init__(
        self,
        *,
        merge_newlines: bool = False,
        template: Template | None = None,
        template_name: str = "fake-qwen3.8-like-v1",
    ) -> None:
        self.merge_newlines = merge_newlines
        self.template = template or qwen3_8_like
        self.template_name = template_name
        self.calls: list[dict[str, Any]] = []

    @property
    def chat_template(self) -> str:
        return self.template_name

    def render_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> str:
        self.calls.append(
            {
                "messages": copy.deepcopy([dict(m) for m in messages]),
                "tools": copy.deepcopy([dict(t) for t in tools]) if tools is not None else None,
                "add_generation_prompt": add_generation_prompt,
                "template_kwargs": dict(template_kwargs),
            }
        )
        if not messages:
            raise ValueError("No messages provided.")
        return self.template(messages, tools, add_generation_prompt, template_kwargs)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        position = 0
        while position < len(text):
            special = next((t for t in SPECIAL_TOKENS if text.startswith(t, position)), None)
            if special is not None:
                ids.append(SPECIAL_IDS[special])
                position += len(special)
            elif self.merge_newlines and text.startswith("\n\n", position):
                ids.append(DOUBLE_NEWLINE_ID)
                position += 2
            else:
                ids.append(ord(text[position]))
                position += 1
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(_TEXT_OF.get(token) or chr(token) for token in token_ids)

    def token_id(self, token: str) -> int | None:
        return SPECIAL_IDS.get(token)


def _arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    arguments = call["function"]["arguments"]
    if not isinstance(arguments, Mapping):
        raise TypeError("tool call arguments must be a mapping")
    return arguments


def _tool_call(call: Mapping[str, Any]) -> str:
    parts = [f"<tool_call>\n<function={call['function']['name']}>\n"]
    for key, value in _arguments(call).items():
        text = value if isinstance(value, str) else json.dumps(value)
        parts.append(f"<parameter={key}>\n{text}\n</parameter>\n")
    parts.append("</function>\n</tool_call>")
    return "".join(parts)


def qwen3_8_like(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> str:
    """The Qwen3.8 chat template's structure, in Python. See the module docstring."""
    parts: list[str] = []
    system = messages[0].get("content") if messages[0]["role"] == "system" else None
    if tools:
        declared = "\n".join(json.dumps(dict(tool), sort_keys=True) for tool in tools)
        parts.append(f"<|im_start|>system\n# Tools\n<tools>\n{declared}\n</tools>")
        parts.append(f"\n\n{system}<|im_end|>\n" if system else "<|im_end|>\n")
    elif system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    users = [index for index, message in enumerate(messages) if message["role"] == "user"]
    if not users:
        raise ValueError("No user query found in messages.")
    preserve = template_kwargs.get("preserve_thinking", True)
    for index, message in enumerate(messages):
        role, content = message["role"], (message.get("content") or "").strip()
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            if preserve or index > users[-1]:
                reasoning = (message.get("reasoning_content") or "").strip()
                parts.append(f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}")
            else:
                parts.append(f"<|im_start|>assistant\n{content}")
            for position, call in enumerate(message.get("tool_calls") or ()):
                if position:
                    parts.append("\n")
                elif content:
                    parts.append("\n\n")
                parts.append(_tool_call(call))
            parts.append("<|im_end|>\n")
        elif role == "tool":
            if messages[index - 1]["role"] != "tool":
                parts.append("<|im_start|>user")
            parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
                parts.append("<|im_end|>\n")
        elif role != "system" or index:
            raise ValueError(f"unexpected {role} message at index {index}")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n<think>\n")
    return "".join(parts)


def role_tag_template(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> str:
    """A template with role tags and no end-of-turn token: the model emits the next tag."""
    parts: list[str] = []
    for message in messages:
        role, content = message["role"], message.get("content") or ""
        if role == "tool":
            parts.append(f"<|observation|><tool_response>{content}</tool_response>")
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}")
            parts.extend(_tool_call(call) for call in message.get("tool_calls") or ())
        else:
            parts.append(f"<|{role}|>{content}")
    if add_generation_prompt:
        parts.append("<|assistant|>")
    return "".join(parts)


def role_tag_after(message: Mapping[str, Any]) -> str:
    """The tag a :func:`role_tag_template` model emits to end ``message``."""
    return "<|observation|>" if message.get("tool_calls") else "<|user|>"


def glm5_like(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> str:
    """The GLM-5 chat template's structure: role tags and no end-of-turn token, a ``<think>``
    block on every assistant turn, ``<arg_key>``/``<arg_value>`` tool calls (strings raw, other
    values as JSON), tool results after ``<|observation|>`` and a prompt that opens ``<think>``.
    """
    parts = ["[gMASK]<sop>"]
    if tools:
        declared = "\n".join(json.dumps(dict(tool), sort_keys=True) for tool in tools)
        parts.append(f"<|system|>\n# Tools\n<tools>\n{declared}\n</tools>")
    for index, message in enumerate(messages):
        role, content = message["role"], message.get("content") or ""
        if role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            parts.append(f"<|assistant|><think>{reasoning}</think>{content.strip()}")
            for call in message.get("tool_calls") or ():
                parts.append(f"<tool_call>{call['function']['name']}")
                for key, value in _arguments(call).items():
                    text = value if isinstance(value, str) else json.dumps(value)
                    parts.append(f"<arg_key>{key}</arg_key><arg_value>{text}</arg_value>")
                parts.append("</tool_call>")
        elif role == "tool":
            if messages[index - 1]["role"] != "tool":
                parts.append("<|observation|>")
            parts.append(f"<tool_response>{content}</tool_response>")
        else:
            parts.append(f"<|{role}|>{content}")
    if add_generation_prompt:
        parts.append("<|assistant|><think>")
    return "".join(parts)
