# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""OpenAI chat messages as the ``system`` and turns of an Anthropic Messages request.

:func:`anthropic_messages` is the one conversion, with no SDK import: the Anthropic client
(:mod:`~pup_finetune.anthropic_client`) sends what it returns, and the compare target's
skip rule (:func:`~pup_finetune.evaluation.compare.refused_prefix`) checks those same turns.

- Every system message, wherever it is, joins the top-level ``system``, in order: the API
  takes no system turn.
- An assistant message's ``tool_calls`` become ``tool_use`` blocks (with their ids and
  parsed arguments) after its text, and each ``tool`` message a ``tool_result`` block of a
  user turn.
- A message with no text, tool call or tool result (an assistant turn of reasoning alone)
  gets no turn: the API refuses empty turns.
- Consecutive messages of one role (a tool result is the user's) share a turn, so the turns
  alternate as the API requires. A turn of text alone keeps its content a string.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["anthropic_messages", "message_text"]


def message_text(content: object) -> str:
    """A message's content as text: a string as it is, the text of text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        ]
        return "".join(parts)
    return "" if content is None else str(content)


def anthropic_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """``messages`` (OpenAI form) as the request's ``system`` (``None`` without one) and its
    ``messages``, converted as the module docstring says."""
    system: list[str] = []
    turns: list[tuple[str, list[dict[str, Any]]]] = []
    for message in messages:
        role = str(message.get("role"))
        text = message_text(message.get("content"))
        if role == "system":
            if text:
                system.append(text)
            continue
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        if role == "tool":
            role = "user"
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id")),
                    "content": text,
                }
            ]
        calls = message.get("tool_calls") if role == "assistant" else None
        for call in calls if isinstance(calls, Sequence) else ():
            function = call.get("function") if isinstance(call, Mapping) else None
            if isinstance(function, Mapping):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id")),
                        "name": str(function.get("name")),
                        "input": _arguments(function.get("arguments")),
                    }
                )
        if not blocks:
            continue
        if turns and turns[-1][0] == role:
            turns[-1][1].extend(blocks)
        else:
            turns.append((role, blocks))
    return "\n\n".join(system) or None, [
        {"role": role, "content": blocks[0]["text"]}
        if len(blocks) == 1 and blocks[0]["type"] == "text"
        else {"role": role, "content": blocks}
        for role, blocks in turns
    ]


def _arguments(text: object) -> dict[str, Any]:
    """A tool call's JSON arguments as the object ``tool_use`` takes; text that is not a JSON
    object is kept under ``_raw``."""
    try:
        parsed = json.loads(text) if isinstance(text, str) else text
    except ValueError:
        parsed = None
    return dict(parsed) if isinstance(parsed, Mapping) else {"_raw": str(text)}
