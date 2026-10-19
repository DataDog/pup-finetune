# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
# DeepSeek's special tokens are spelled with U+FF5C, the fullwidth vertical line.
# ruff: noqa: RUF001, RUF002
"""Chat encoders for models whose tokenizer ships no chat template.

A :class:`~pup_finetune.rendering.hf.ChatEncoder` renders the messages pup-finetune trains on
(``system``, ``user``, ``assistant`` with reasoning and tool calls, ``tool``) as the model's
own reference encoder does; the ``tokenizer`` tests check each one against that encoder at
its family's pinned revision. Model families name one by key from :data:`ENCODERS`.

:class:`DeepSeekV4Encoder` is DeepSeek-V4's ``encoding/encoding_dsv4.py`` in thinking mode at
the default (``low``) reasoning effort, which adds no preamble::

    <｜begin▁of▁sentence｜>{system}

    ## Tools ...<｜User｜>{user}<｜Assistant｜><think>{reasoning}</think>{content}

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls><｜end▁of▁sentence｜>

Tool results join the user turn as ``<tool_result>`` blocks, in the order of the calls they
answer, and consecutive user turns join with them. Without tools, assistant turns before the
last user turn lose their reasoning and the ``<think>`` that opens them; with tools every turn
keeps both. Tools go on the leading system message, or on an empty one put first, as
DeepSeek's own V4.1 template does with them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from pup_finetune.rendering.hf import ChatEncoder

__all__ = ["ENCODERS", "DeepSeekV4Encoder", "get_encoder"]

_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_THINK = "<think>"
_END_THINK = "</think>"
_DSML = "｜DSML｜"
_TOOLS = f"""## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{_DSML}tool_calls>" block like the following:

<{_DSML}tool_calls>
<{_DSML}invoke name="$TOOL_NAME">
<{_DSML}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{_DSML}parameter>
...
</{_DSML}invoke>
<{_DSML}invoke name="$TOOL_NAME2">
...
</{_DSML}invoke>
</{_DSML}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {_THINK}), you MUST output your complete reasoning inside {_THINK}...{_END_THINK} BEFORE any tool calls or final response.

Otherwise, output directly after {_END_THINK} with tool calls or final response.

### Available Tool Schemas

{{schemas}}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""  # noqa: E501 - the model's own prompt, line for line


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class _UserTurn:
    """User text and tool results in arrival order (``(None, text)`` or ``(call id,
    result)``), and the call order of the assistant turn before them (call id to position)."""

    blocks: list[tuple[str | None, str]] = field(default_factory=list)
    order: Mapping[str, int] = field(default_factory=dict)

    def text(self) -> str:
        results = iter(
            sorted(
                (block for block in self.blocks if block[0] is not None),
                key=lambda block: self.order.get(block[0] or "", 0),
            )
        )
        return "\n\n".join(
            text if call_id is None else f"<tool_result>{next(results)[1]}</tool_result>"
            for call_id, text in self.blocks
        )


def _turns(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None
) -> list[Mapping[str, Any] | _UserTurn]:
    """``messages`` with the tools on a system message and tool results in user turns."""
    listed = list(messages)
    if tools:
        if listed and listed[0].get("role") == "system":
            listed[0] = {**listed[0], "tools": list(tools)}
        else:
            listed.insert(0, {"role": "system", "content": "", "tools": list(tools)})
    turns: list[Mapping[str, Any] | _UserTurn] = []
    order: Mapping[str, int] = {}
    user: _UserTurn | None = None
    for message in listed:
        role = message.get("role")
        if role not in ("user", "tool"):
            if role == "assistant" and message.get("tool_calls"):
                order = {
                    str(call["id"]): position
                    for position, call in enumerate(message["tool_calls"])
                    if call.get("id")
                }
            turns.append(message)
            user = None
            continue
        if user is None:
            user = _UserTurn(order=order)
            turns.append(user)
        call_id = str(message.get("tool_call_id") or "") if role == "tool" else None
        user.blocks.append((call_id, str(message.get("content") or "")))
    return turns


def _calls(message: Mapping[str, Any]) -> str:
    invokes = []
    for call in message.get("tool_calls") or ():
        function = call["function"]
        parameters = "\n".join(
            f'<{_DSML}parameter name="{key}" string="{"true" if isinstance(value, str) else "false"}">'
            f"{value if isinstance(value, str) else _json(value)}</{_DSML}parameter>"
            for key, value in function["arguments"].items()
        )
        invokes.append(f'<{_DSML}invoke name="{function["name"]}">\n{parameters}\n</{_DSML}invoke>')
    if not invokes:
        return ""
    return f"\n\n<{_DSML}tool_calls>\n" + "\n".join(invokes) + f"\n</{_DSML}tool_calls>"


@dataclass(frozen=True, slots=True)
class DeepSeekV4Encoder:
    """DeepSeek-V4's chat format; see the module docstring. ``add_generation_prompt`` is not
    needed: the format opens an assistant turn after every user turn by itself."""

    name: ClassVar[str] = "deepseek_v4"
    version: ClassVar[str] = "1"

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        *,
        add_generation_prompt: bool,
    ) -> str:
        turns = _turns(messages, tools)
        keep_reasoning = bool(tools)
        users = [index for index, turn in enumerate(turns) if isinstance(turn, _UserTurn)]
        last_user = users[-1] if users else -1
        parts = [_BOS]
        for index, turn in enumerate(turns):
            if isinstance(turn, _UserTurn):
                parts.append(_USER + turn.text())
                after = turns[index + 1] if index + 1 < len(turns) else None
                if after is None or (
                    isinstance(after, Mapping) and after.get("role") == "assistant"
                ):
                    opens = keep_reasoning or index >= last_user
                    parts.append(_ASSISTANT + (_THINK if opens else _END_THINK))
            elif turn.get("role") == "system":
                parts.append(str(turn.get("content") or ""))
                if turn.get("tools"):
                    schemas = "\n".join(_json(tool["function"]) for tool in turn["tools"])
                    parts.append("\n\n" + _TOOLS.replace("{schemas}", schemas))
            else:
                kept = keep_reasoning or index > last_user
                reasoning = f"{turn.get('reasoning_content') or ''}{_END_THINK}" if kept else ""
                parts.append(f"{reasoning}{turn.get('content') or ''}{_calls(turn)}{_EOS}")
        return "".join(parts)


ENCODERS: Mapping[str, Callable[[], ChatEncoder]] = MappingProxyType(
    {DeepSeekV4Encoder.name: DeepSeekV4Encoder}
)
"""Every chat encoder by key, as :class:`~pup_finetune.rendering.families.ModelFamily` names
it."""


def get_encoder(name: str) -> ChatEncoder:
    """A new encoder named ``name``; raises ``KeyError`` naming the known ones."""
    try:
        factory = ENCODERS[name]
    except KeyError:
        raise KeyError(f"unknown chat encoder {name!r}; known: {', '.join(ENCODERS)}") from None
    return factory()
