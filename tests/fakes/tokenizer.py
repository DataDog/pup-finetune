# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A fake :class:`~pup_finetune.rendering.ChatTokenizer`: a pure-Python chat template.

Each special token is one id and every other character is its own id (its code point)::

    tokenizer = FakeChatTokenizer()  # Qwen3.5-like ChatML
    tokenizer = FakeChatTokenizer(merge_newlines=True)  # "\\n\\n" encodes as one token
    tokenizer = FakeChatTokenizer("glm")  # GLM-5.3-like role tags

``chatml`` follows Qwen3.5: ``<|im_start|>role\\n...<|im_end|>\\n`` turns, reasoning kept only
for assistant turns after the last user message, and a generation prompt that opens
``<think>\\n``. With ``merge_newlines`` an empty thinking block merges with that prompt's
trailing newline, as BPE does. ``glm`` follows GLM-5.3: role tags and no end-of-turn token, so
the model stops by emitting the next tag. Both raise, like real templates, on an empty
conversation and on tool arguments that are not objects. Every render call is recorded.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
)
_FIRST_SPECIAL_ID = 0x110000  # above every Unicode code point
_SPECIAL_IDS = {token: _FIRST_SPECIAL_ID + index for index, token in enumerate(SPECIAL_TOKENS)}
DOUBLE_NEWLINE_ID = _FIRST_SPECIAL_ID + len(SPECIAL_TOKENS)
_TEXT_OF = {**{v: k for k, v in _SPECIAL_IDS.items()}, DOUBLE_NEWLINE_ID: "\n\n"}


@dataclass(frozen=True)
class RenderCall:
    """One recorded :meth:`FakeChatTokenizer.render` call (deep copies of its arguments)."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    add_generation_prompt: bool
    template_kwargs: dict[str, Any]


class FakeChatTokenizer:
    """A deterministic :class:`~pup_finetune.rendering.ChatTokenizer` for tests."""

    def __init__(
        self, style: Literal["chatml", "glm"] = "chatml", *, merge_newlines: bool = False
    ) -> None:
        self.style = style
        self.merge_newlines = merge_newlines
        self.calls: list[RenderCall] = []

    @property
    def chat_template(self) -> str:
        return f"fake-{self.style}-template-v1"

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]:
        self.calls.append(
            RenderCall(
                messages=copy.deepcopy([dict(m) for m in messages]),
                tools=copy.deepcopy([dict(t) for t in tools]) if tools is not None else None,
                add_generation_prompt=add_generation_prompt,
                template_kwargs=dict(template_kwargs),
            )
        )
        return self.encode(
            self.render_text(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                template_kwargs=template_kwargs,
            )
        )

    def render_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = False,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        """The template text that :meth:`render` encodes."""
        if not messages:
            raise ValueError("Cannot apply chat template to an empty conversation.")
        if self.style == "chatml":
            return _chatml(messages, tools, add_generation_prompt, template_kwargs or {})
        return _glm(messages, tools, add_generation_prompt)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        position = 0
        while position < len(text):
            special = next((t for t in SPECIAL_TOKENS if text.startswith(t, position)), None)
            if special is not None:
                ids.append(_SPECIAL_IDS[special])
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
        return _SPECIAL_IDS.get(token)


def _declared(tools: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(json.dumps(dict(tool), sort_keys=True) for tool in tools)


def _arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    arguments = call["function"]["arguments"]
    if not isinstance(arguments, dict):
        raise TypeError("tool call arguments must be a mapping")
    return arguments


def _chatml(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> str:
    parts: list[str] = []
    system = messages[0]["content"] if messages[0]["role"] == "system" else None
    if tools:
        parts.append(f"<|im_start|>system\n# Tools\n<tools>\n{_declared(tools)}\n</tools>")
        parts.append(f"\n\n{system}<|im_end|>\n" if system else "<|im_end|>\n")
    elif system is not None:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    users = [index for index, message in enumerate(messages) if message["role"] == "user"]
    if not users:
        raise ValueError("No user query found in messages.")
    for index, message in enumerate(messages):
        role, content = message["role"], message.get("content", "")
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            parts.append("<|im_start|>assistant\n")
            if index > users[-1]:
                parts.append(f"<think>\n{message.get('reasoning_content') or ''}\n</think>\n\n")
            parts.append(content)
            for call in message.get("tool_calls") or ():
                name, arguments = call["function"]["name"], _arguments(call)
                parts.append(f"<tool_call>\n{name} {json.dumps(arguments)}\n</tool_call>")
            parts.append("<|im_end|>\n")
        elif role == "tool":
            if index == 0 or messages[index - 1]["role"] != "tool":
                parts.append("<|im_start|>user")
            parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
                parts.append("<|im_end|>\n")
        elif role != "system" or index:
            raise ValueError(f"unexpected {role} message at index {index}")
    if add_generation_prompt:
        thinking_off = template_kwargs.get("enable_thinking") is False
        parts.append("<|im_start|>assistant\n")
        parts.append("<think>\n\n</think>\n\n" if thinking_off else "<think>\n")
    return "".join(parts)


def _glm(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    add_generation_prompt: bool,
) -> str:
    parts: list[str] = []
    if tools:
        parts.append(f"<|system|>\n# Tools\n<tools>\n{_declared(tools)}\n</tools>")
    for index, message in enumerate(messages):
        role, content = message["role"], message.get("content", "")
        if role == "system":
            parts.append(f"<|system|>{content}")
        elif role == "user":
            parts.append(f"<|user|>{content}")
        elif role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            parts.append(f"<|assistant|><think>{reasoning}</think>{content.strip()}")
            for call in message.get("tool_calls") or ():
                pairs = "".join(
                    f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
                    for key, value in _arguments(call).items()
                )
                parts.append(f"<tool_call>{call['function']['name']}{pairs}</tool_call>")
        elif role == "tool":
            if index == 0 or messages[index - 1]["role"] != "tool":
                parts.append("<|observation|>")
            parts.append(f"<tool_response>{content}</tool_response>")
        else:
            raise ValueError(f"unexpected {role} message at index {index}")
    if add_generation_prompt:
        parts.append("<|assistant|><think>")
    return "".join(parts)
