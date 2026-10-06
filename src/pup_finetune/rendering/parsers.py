# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Response parsers: a model's sampled text back into an OpenAI assistant message.

A :class:`ResponseParser` reads the text a sampler produced after the generation prompt and
never raises: text that broke the model's output format comes back with ``format_valid``
false and a best-effort message. Two tool-call formats are covered:

- :class:`Qwen3XmlToolParser`, the Qwen3.8 format::

      <think>
      reasoning
      </think>

      <tool_call>
      <function=get_weather>
      <parameter=city>
      Paris
      </parameter>
      </function>
      </tool_call><|im_end|>

  The generation prompt already opens ``<think>``, so sampled text starts inside the
  reasoning block: everything before the first ``</think>`` is reasoning, with or without an
  opening tag, and text that never closes the block (cut off at the token limit, or a model
  that never left its reasoning) is format-invalid, all of it reasoning.
- :class:`JsonToolCallParser`, one JSON object per block:
  ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``.

Both strip one trailing end-of-turn token. Model families pick a parser by name from
:data:`PARSERS`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

from pup_finetune.rendering.base import ChatMessage, ParsedResponse
from pup_finetune.storage import loads_finite

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_CALL_OPEN = "<tool_call>"
_CALL_CLOSE = "</tool_call>"
_CALL_BLOCK = re.compile(re.escape(_CALL_OPEN) + r"(.*?)" + re.escape(_CALL_CLOSE), re.DOTALL)
_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


class ResponseParser(Protocol):
    """Sampled text to an assistant message, for one model output format."""

    @property
    def name(self) -> str:
        """The parser's name, as model families refer to it."""
        ...

    @property
    def version(self) -> str:
        """Bumped whenever the parse of some text changes; recorded in renderer identities."""
        ...

    def parse(self, text: str) -> ParsedResponse:
        """``text`` (what follows the generation prompt) as a message. Never raises."""
        ...


def _tool_call(index: int, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(dict(arguments), ensure_ascii=False)},
    }


def _parse_blocks(
    text: str,
    turn_end: str,
    parse_call: Callable[[str], tuple[str, dict[str, Any]] | None],
    *,
    prompt_opens_think: bool,
) -> ParsedResponse:
    """The parse both formats share: reasoning, then content and ``<tool_call>`` blocks.

    ``parse_call`` reads one block's inner text as ``(name, arguments)``, or ``None`` when
    the block is malformed; a malformed block stays in ``content`` as raw text, as does a
    stray ``<tool_call>`` or ``</tool_call>`` before, between or after the blocks, and either
    makes the format invalid. With
    ``prompt_opens_think`` the text continues a reasoning block the prompt opened, so it must
    close it: when it does not, the whole text is reasoning and the format is invalid.
    """
    valid = True
    body = text.rstrip()
    if body.endswith(turn_end):
        body = body.removesuffix(turn_end)
    if turn_end in body:
        valid = False
    reasoning: str | None = None
    if prompt_opens_think and _THINK_CLOSE not in body:
        unclosed = body.strip().removeprefix(_THINK_OPEN).strip()
        message: ChatMessage = {"role": "assistant", "content": ""}
        if unclosed:
            message["reasoning_content"] = unclosed
        return ParsedResponse(message, format_valid=False)
    if _THINK_CLOSE in body:
        head, _, body = body.partition(_THINK_CLOSE)
        head = head.lstrip()
        head = head.removeprefix(_THINK_OPEN)
        reasoning = head.strip() or None
        if _THINK_OPEN in head or _THINK_CLOSE in body:
            valid = False
    if _THINK_OPEN in body:
        valid = False
    content: list[str] = []
    calls: list[dict[str, Any]] = []
    position = 0
    for match in _CALL_BLOCK.finditer(body):
        segment = body[position : match.start()]
        if _CALL_OPEN in segment or _CALL_CLOSE in segment:
            valid = False
        content.append(segment)
        position = match.end()
        parsed = parse_call(match.group(1))
        if parsed is None or _CALL_OPEN in match.group(1):
            valid = False
            content.append(match.group(0))
            continue
        calls.append(_tool_call(len(calls), *parsed))
    rest = body[position:]
    if _CALL_OPEN in rest or _CALL_CLOSE in rest:
        valid = False
    content.append(rest)
    message = {"role": "assistant", "content": "".join(content).strip()}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return ParsedResponse(message, format_valid=valid)


def _value(text: str) -> Any:
    """A parameter value: JSON when it parses as JSON, else the text itself."""
    try:
        return loads_finite(text)
    except (ValueError, RecursionError):
        return text


@dataclass(frozen=True, slots=True)
class Qwen3XmlToolParser:
    """The Qwen3.8 format: ``<think>`` reasoning and XML ``<function=...>`` tool calls.

    A parameter's value is decoded as JSON when it is valid JSON (the template writes
    non-string values as JSON) and kept as text otherwise. A block with no ``<function=...>``,
    text outside its parameters or a repeated parameter is malformed. ``prompt_opens_think``
    (the default, as Qwen3.8's generation prompt ends ``<think>\\n``) makes text that never
    reaches ``</think>`` format-invalid; set it false for prompts that leave reasoning closed.
    """

    turn_end: str = "<|im_end|>"
    prompt_opens_think: bool = True
    name: ClassVar[str] = "qwen3_xml"
    version: ClassVar[str] = "2"

    def parse(self, text: str) -> ParsedResponse:
        return _parse_blocks(
            text, self.turn_end, _xml_call, prompt_opens_think=self.prompt_opens_think
        )


def _xml_call(inner: str) -> tuple[str, dict[str, Any]] | None:
    function = _FUNCTION.fullmatch(inner.strip())
    if function is None:
        return None
    name, params = function.group(1).strip(), function.group(2)
    arguments: dict[str, Any] = {}
    position = 0
    for match in _PARAMETER.finditer(params):
        key = match.group(1).strip()
        if params[position : match.start()].strip() or not key or key in arguments:
            return None
        position = match.end()
        arguments[key] = _value(match.group(2).removeprefix("\n").removesuffix("\n"))
    if params[position:].strip() or not name:
        return None
    return name, arguments


@dataclass(frozen=True, slots=True)
class JsonToolCallParser:
    """``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks after reasoning.

    ``arguments`` must be a JSON object, or a string holding one. Reasoning is optional
    unless ``prompt_opens_think`` says the generation prompt opened it (see
    :class:`Qwen3XmlToolParser`).
    """

    turn_end: str = "<|im_end|>"
    prompt_opens_think: bool = False
    name: ClassVar[str] = "json_tool_call"
    version: ClassVar[str] = "2"

    def parse(self, text: str) -> ParsedResponse:
        return _parse_blocks(
            text, self.turn_end, _json_call, prompt_opens_think=self.prompt_opens_think
        )


def _json_call(inner: str) -> tuple[str, dict[str, Any]] | None:
    try:
        call = loads_finite(inner.strip())
        if not isinstance(call, dict):
            return None
        name, arguments = call.get("name"), call.get("arguments", {})
        if isinstance(arguments, str):
            arguments = loads_finite(arguments)
    except (ValueError, RecursionError):
        return None
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return name, arguments


PARSERS: Mapping[str, Callable[[], ResponseParser]] = MappingProxyType(
    {
        "qwen3_xml": Qwen3XmlToolParser,
        "json_tool_call": JsonToolCallParser,
    }
)
"""Every parser by name, as :class:`~pup_finetune.rendering.families.ModelFamily` names it."""


def get_parser(name: str) -> ResponseParser:
    """A new parser named ``name``; raises ``KeyError`` naming the known ones."""
    try:
        factory = PARSERS[name]
    except KeyError:
        raise KeyError(f"unknown response parser {name!r}; known: {', '.join(PARSERS)}") from None
    return factory()
