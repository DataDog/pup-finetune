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
:data:`PARSERS`. A parser is also given the tools the prompt declared: the XML format needs
their parameter types to read a value back (see :class:`Qwen3XmlToolParser`).
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

from pup_finetune.rendering.base import ChatMessage, ParsedResponse, ToolSpec
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

    def parse(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        """``text`` (what follows the generation prompt) as a message. Never raises.

        ``tools`` are the tools the prompt declared, in OpenAI format. A parser may read
        argument types from them, and accepts tools of any shape without raising.
        """
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


_Properties = Mapping[str, Mapping[str, Mapping[str, Any]]]
"""Each declared tool's ``parameters.properties``, by tool name."""


def _properties(tools: Sequence[ToolSpec]) -> _Properties:
    """The parameter schemas of ``tools`` by tool name.

    Tools come from traces, so any shape is accepted: a tool or schema that is not the
    expected mapping is skipped, and of two tools with one name the first wins.
    """
    found: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for tool in tools:
        spec = tool.get("function", tool) if isinstance(tool, Mapping) else None
        if not isinstance(spec, Mapping):
            continue
        name, parameters = spec.get("name"), spec.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if isinstance(name, str) and isinstance(properties, Mapping) and name not in found:
            found[name] = {
                key: schema
                for key, schema in properties.items()
                if isinstance(key, str) and isinstance(schema, Mapping)
            }
    return found


def _json_types(value: Any) -> set[str]:
    """The JSON Schema ``type`` names a decoded JSON ``value`` has."""
    if value is None:
        return {"null"}
    if isinstance(value, bool):
        return {"boolean"}
    if isinstance(value, int):
        return {"integer", "number"}
    if isinstance(value, float):
        return {"number"}
    if isinstance(value, str):
        return {"string"}
    if isinstance(value, list):
        return {"array"}
    return {"object"}


def _value(text: str, schema: Mapping[str, Any] | None) -> Any:
    """A parameter's value, read with the parameter's declared ``schema`` (``None`` if none).

    The template writes a string unquoted and any other value as JSON, so the text alone
    cannot tell the string ``"123"`` from the number. When the schema's ``type`` is
    ``"string"`` the text is kept as is. When it is a list holding ``"string"``, the text is
    decoded only if it is JSON of another listed type (``["string", "integer"]`` reads
    ``123`` as a number, ``abc`` and ``true`` as text). Otherwise (no schema, no ``type``, or
    one without ``"string"``) the text is decoded when it is valid JSON and kept otherwise.
    """
    declared = schema.get("type") if schema is not None else None
    types = {declared} if isinstance(declared, str) else declared
    try:
        decoded = loads_finite(text)
    except (ValueError, RecursionError):
        return text
    if not isinstance(types, list | set) or "string" not in types:
        return decoded
    other = {kind for kind in types if isinstance(kind, str)} - {"string"}
    return decoded if _json_types(decoded) & other else text


@dataclass(frozen=True, slots=True)
class Qwen3XmlToolParser:
    """The Qwen3.8 format: ``<think>`` reasoning and XML ``<function=...>`` tool calls.

    The template writes a string parameter value unquoted and any other value as JSON, so a
    value is read with the type the called tool declares for that parameter in ``tools``: a
    string parameter keeps its text, even text such as ``123``, ``true`` or ``null``. A
    parameter no declared tool types is decoded as JSON when it is valid JSON and kept as text
    otherwise (see :func:`_value`). A block with no ``<function=...>``,
    text outside its parameters or a repeated parameter is malformed. ``prompt_opens_think``
    (the default, as Qwen3.8's generation prompt ends ``<think>\\n``) makes text that never
    reaches ``</think>`` format-invalid; set it false for prompts that leave reasoning closed.
    """

    turn_end: str = "<|im_end|>"
    prompt_opens_think: bool = True
    name: ClassVar[str] = "qwen3_xml"
    version: ClassVar[str] = "3"

    def parse(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        parse_call = functools.partial(_xml_call, properties=_properties(tools))
        return _parse_blocks(
            text, self.turn_end, parse_call, prompt_opens_think=self.prompt_opens_think
        )


def _xml_call(inner: str, *, properties: _Properties) -> tuple[str, dict[str, Any]] | None:
    function = _FUNCTION.fullmatch(inner.strip())
    if function is None:
        return None
    name, params = function.group(1).strip(), function.group(2)
    schemas = properties.get(name, {})
    arguments: dict[str, Any] = {}
    position = 0
    for match in _PARAMETER.finditer(params):
        key = match.group(1).strip()
        if params[position : match.start()].strip() or not key or key in arguments:
            return None
        position = match.end()
        text = match.group(2).removeprefix("\n").removesuffix("\n")
        arguments[key] = _value(text, schemas.get(key))
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

    def parse(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        # JSON arguments carry their own types, so the declared tools are not needed.
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
