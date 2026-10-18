# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Response parsers: a model's sampled text back into an OpenAI assistant message.

A :class:`ResponseParser` reads the text a sampler produced after the generation prompt and
never raises: text that broke the model's output format comes back with ``format_valid``
false and a best-effort message. These tool-call formats are covered:

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
  :class:`Qwen3_5XmlToolParser` reads the same format from Qwen3.5, whose template writes
  scalar values with Python's ``str``.
- :class:`GlmXmlToolParser`, the GLM-5 format, also after an open ``<think>``, with no
  separators (wrapped here) and ending on the next role tag::

      reasoning</think><tool_call>get_weather<arg_key>city</arg_key>
      <arg_value>Paris</arg_value></tool_call><|observation|>

- :class:`JsonToolCallParser`, one JSON object per block:
  ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``.
- :class:`DeepSeekDsmlParser`, DeepSeek-V4's DSML, after an open ``<think>``, with each
  parameter marked as text or JSON::

      reasoning</think>text

      <｜DSML｜tool_calls>
      <｜DSML｜invoke name="get_weather">
      <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
      </｜DSML｜invoke>
      </｜DSML｜tool_calls><｜end▁of▁sentence｜>

- :class:`KimiXtmlParser`, Kimi K3's XTML channels, after an open think channel, with each
  argument typed::

      reasoning<|close|>think<|sep|><|open|>response<|sep|>text<|close|>response<|sep|>
      <|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|>
      <|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|>
      <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>

  (without the line breaks, which are added here).
- :class:`MuseAtemParser`, Muse Glimmer's ATEM messages after an open ``<|start|>assistant``:
  reasoning to ``self``, text to ``user``, then one message per tool call::

       to=self<|message|>reasoning<|eom|><|start|>assistant to=get_weather<|message|>
      <atem:function_calls>
      <atem:invoke name="get_weather">
      <atem:parameter name="city">Paris</atem:parameter>
      </atem:invoke>
      </atem:function_calls><|eot|>

Each strips one trailing end-of-turn token. Model families pick a parser by name from
:data:`PARSERS`. A parser is also given the tools the prompt declared: the XML formats need
their parameter types to read a value back (see :class:`Qwen3XmlToolParser`).

Kimi K3's tokenizer keeps message text apart from its control tokens, so a reply may spell
``<|close|>`` as text. A :class:`ControlTokenParser` such as :class:`KimiXtmlParser` is also
given the sampled completion split at its control tokens
(:meth:`~pup_finetune.rendering.base.Renderer.parse_completion`), and reads structure only
from the tokens themselves.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

from pup_finetune.rendering.base import ChatMessage, ParsedResponse, ToolSpec
from pup_finetune.storage import loads_finite

# DeepSeek-V4's markers are spelled with U+FF5C, the fullwidth vertical line.
# ruff: noqa: RUF001, RUF002

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_CALL_OPEN = "<tool_call>"
_CALL_CLOSE = "</tool_call>"
_CALL_BLOCK = re.compile(re.escape(_CALL_OPEN) + r"(.*?)" + re.escape(_CALL_CLOSE), re.DOTALL)
_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
_ARGUMENT = re.compile(r"<arg_key>([^<>]*)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
_GLM_NAME = re.compile(r"[^\s<>]+")
_GLM_USER = "<|user|>"
_GLM_OBSERVATION = "<|observation|>"
_PYTHON_LITERALS: Mapping[str, Any] = MappingProxyType({"True": True, "False": False, "None": None})


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


Segment = tuple[str, bool]
"""``(text, is_control)``: one control token's spelling, or text sampled between them."""


@runtime_checkable
class ControlTokenParser(ResponseParser, Protocol):
    """A parser for a format whose control tokens a reply can also spell as text: given the
    completion split at those tokens, it reads structure only from the tokens."""

    @property
    def control_tokens(self) -> tuple[str, ...]:
        """The control tokens, as their tokenizer spells them."""
        ...

    def parse_segments(
        self, segments: Sequence[Segment], tools: Sequence[ToolSpec] = ()
    ) -> ParsedResponse:
        """The completion, split into :data:`Segment` runs at :attr:`control_tokens`, as a
        message. Never raises."""
        ...


def _tool_call(index: int, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(dict(arguments), ensure_ascii=False)},
    }


def _parse_blocks(
    text: str,
    turn_ends: tuple[str, ...],
    parse_call: Callable[[str], tuple[str, dict[str, Any]] | None],
    *,
    prompt_opens_think: bool,
) -> ParsedResponse:
    """The parse every format shares: reasoning, then content and ``<tool_call>`` blocks.

    One of ``turn_ends`` may end the text; anywhere else, one is invalid. ``parse_call``
    reads one block's inner text as ``(name, arguments)``, or ``None`` when the block is
    malformed; a malformed block stays in ``content`` as raw text, as does a stray
    ``<tool_call>`` or ``</tool_call>`` before, between or after the blocks, and either makes
    the format invalid. With ``prompt_opens_think`` the text continues a reasoning block the
    prompt opened, so it must close it: when it does not, the whole text is reasoning and the
    format is invalid.
    """
    valid = True
    body = text.rstrip()
    for turn_end in turn_ends:
        if body.endswith(turn_end):
            body = body.removesuffix(turn_end)
            break
    if any(turn_end in body for turn_end in turn_ends):
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
            text, (self.turn_end,), parse_call, prompt_opens_think=self.prompt_opens_think
        )


@dataclass(frozen=True, slots=True)
class Qwen3_5XmlToolParser(Qwen3XmlToolParser):
    """The Qwen3.5 format: :class:`Qwen3XmlToolParser`'s, with scalars written by ``str``.

    Qwen3.5's template writes booleans and null as ``True``, ``False`` and ``None``; those
    exact values read as ``true``, ``false`` and ``null``, unless the parameter is a string.
    """

    name: ClassVar[str] = "qwen3_5_xml"
    version: ClassVar[str] = "1"

    def parse(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        parse_call = functools.partial(
            _xml_call, properties=_properties(tools), value=_python_value
        )
        return _parse_blocks(
            text, (self.turn_end,), parse_call, prompt_opens_think=self.prompt_opens_think
        )


def _python_value(text: str, schema: Mapping[str, Any] | None) -> Any:
    if text not in _PYTHON_LITERALS:
        return _value(text, schema)
    decoded = _value(json.dumps(_PYTHON_LITERALS[text]), schema)
    return text if isinstance(decoded, str) else decoded


def _xml_call(
    inner: str,
    *,
    properties: _Properties,
    value: Callable[[str, Mapping[str, Any] | None], Any] = _value,
) -> tuple[str, dict[str, Any]] | None:
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
        arguments[key] = value(text, schemas.get(key))
    if params[position:].strip() or not name:
        return None
    return name, arguments


@dataclass(frozen=True, slots=True)
class GlmXmlToolParser:
    """The GLM-5 format: ``<think>`` reasoning and ``<arg_key>``/``<arg_value>`` tool calls.

    The model ends a turn by emitting the next role tag: ``<|observation|>`` after tool calls,
    ``<|user|>`` otherwise; text that ends on the other tag is format-invalid. The template
    writes string values raw and other values as JSON, so a value is read with the type the
    called tool declares, as in :class:`Qwen3XmlToolParser`. A block whose name is not one
    word, with text outside its arguments or a repeated key, is malformed.
    ``prompt_opens_think`` is as in :class:`Qwen3XmlToolParser` (GLM-5's generation prompt
    ends ``<think>``).
    """

    prompt_opens_think: bool = True
    name: ClassVar[str] = "glm_xml"
    version: ClassVar[str] = "1"

    def parse(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        tags = (_GLM_USER, _GLM_OBSERVATION)
        parse_call = functools.partial(_glm_call, properties=_properties(tools))
        parsed = _parse_blocks(text, tags, parse_call, prompt_opens_think=self.prompt_opens_think)
        expected = _GLM_OBSERVATION if parsed.message.get("tool_calls") else _GLM_USER
        ending = text.rstrip()
        if ending.endswith(tags) and not ending.endswith(expected):
            return ParsedResponse(parsed.message, format_valid=False)
        return parsed


def _glm_call(inner: str, *, properties: _Properties) -> tuple[str, dict[str, Any]] | None:
    head, _, _ = inner.partition("<arg_key>")
    name, params = head.strip(), inner[len(head) :]
    schemas = properties.get(name, {})
    arguments: dict[str, Any] = {}
    position = 0
    for match in _ARGUMENT.finditer(params):
        key = match.group(1).strip()
        if params[position : match.start()].strip() or not key or key in arguments:
            return None
        position = match.end()
        arguments[key] = _value(match.group(2), schemas.get(key))
    if params[position:].strip() or not _GLM_NAME.fullmatch(name):
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
            text, (self.turn_end,), _json_call, prompt_opens_think=self.prompt_opens_think
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


def _sequence(text: str, pattern: re.Pattern[str], separator: str) -> list[re.Match[str]] | None:
    """The matches of ``pattern`` that make up all of ``text``, ``separator`` between them;
    ``None`` when anything else is in it (empty text is no matches)."""
    matches: list[re.Match[str]] = []
    position = 0
    for match in pattern.finditer(text):
        if text[position : match.start()] != (separator if matches else ""):
            return None
        matches.append(match)
        position = match.end()
    return matches if position == len(text) else None


def _decoded(text: str) -> tuple[bool, Any]:
    """``(True, value)`` for JSON ``text``, ``(False, None)`` otherwise."""
    try:
        return True, loads_finite(text)
    except (ValueError, RecursionError):
        return False, None


def _assistant(
    content: str, reasoning: str | None, calls: list[tuple[str, dict[str, Any]]]
) -> ChatMessage:
    message: ChatMessage = {"role": "assistant", "content": content.strip()}
    if reasoning and reasoning.strip():
        message["reasoning_content"] = reasoning.strip()
    if calls:
        message["tool_calls"] = [_tool_call(index, *call) for index, call in enumerate(calls)]
    return message


_DSML = "｜DSML｜"
_DSML_EOS = "<｜end▁of▁sentence｜>"
_DSML_CALLS_OPEN = f"\n\n<{_DSML}tool_calls>\n"
_DSML_CALLS_CLOSE = f"\n</{_DSML}tool_calls>"
_DSML_INVOKE = re.compile(rf'<{_DSML}invoke name="([^"\n]+)">\n(.*?)\n?</{_DSML}invoke>', re.DOTALL)
_DSML_PARAMETER = re.compile(
    rf'<{_DSML}parameter name="([^"\n]+)" string="(true|false)">(.*?)</{_DSML}parameter>',
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class DeepSeekDsmlParser:
    """DeepSeek-V4's format: reasoning up to ``</think>`` (the generation prompt opens
    ``<think>``), text, then one DSML ``tool_calls`` block, ending ``<｜end▁of▁sentence｜>``.

    Each parameter says whether its value is text (``string="true"``, kept as is) or JSON
    (decoded), so the declared tools are not needed. Text that never closes the reasoning is
    format-invalid, all of it reasoning; a malformed block stays in ``content`` as raw text.
    """

    name: ClassVar[str] = "deepseek_dsml"
    version: ClassVar[str] = "1"

    def parse(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        body = text.rstrip().removesuffix(_DSML_EOS)
        if _THINK_CLOSE not in body:
            return ParsedResponse(
                _assistant("", body.strip().removeprefix(_THINK_OPEN), []), format_valid=False
            )
        head, _, body = body.partition(_THINK_CLOSE)
        head = head.lstrip().removeprefix(_THINK_OPEN)
        content, opened, block = body.partition(_DSML_CALLS_OPEN)
        calls: list[tuple[str, dict[str, Any]]] = []
        valid = True
        if opened:
            inner, closed, rest = block.rpartition(_DSML_CALLS_CLOSE)
            invokes = _sequence(inner, _DSML_INVOKE, "\n") if closed and not rest else None
            parsed = [_dsml_call(invoke) for invoke in invokes or []]
            if not parsed or None in parsed:
                valid = False
                content += opened + block
            else:
                calls = [call for call in parsed if call is not None]
        markers = (_DSML, _DSML_EOS, _THINK_OPEN, _THINK_CLOSE)
        if any(marker in head or marker in content for marker in markers):
            valid = False
        return ParsedResponse(_assistant(content, head, calls), format_valid=valid)


def _dsml_call(invoke: re.Match[str]) -> tuple[str, dict[str, Any]] | None:
    parameters = _sequence(invoke.group(2), _DSML_PARAMETER, "\n")
    if parameters is None:
        return None
    arguments: dict[str, Any] = {}
    for parameter in parameters:
        key, is_text, raw = parameter.groups()
        if key in arguments:
            return None
        if is_text == "true":
            arguments[key] = raw
            continue
        ok, value = _decoded(raw)
        if not ok:
            return None
        arguments[key] = value
    return invoke.group(1), arguments


@dataclass(frozen=True, slots=True)
class _Xtml:
    """Kimi K3's XTML grammar, with its control tokens spelled as given: as text, or as
    stand-ins for sampled control tokens."""

    end: str
    message_close: str
    markers: tuple[str, ...]
    channels: re.Pattern[str]
    call: re.Pattern[str]
    argument: re.Pattern[str]


def _xtml(open_tag: str, close_tag: str, sep: str, end: str) -> _Xtml:
    o, c, s = map(re.escape, (open_tag, close_tag, sep))
    return _Xtml(
        end=end,
        message_close=f"{close_tag}message{sep}",
        markers=(open_tag, close_tag, sep, end),
        channels=re.compile(
            rf"(?P<reasoning>.*?){c}think{s}{o}response{s}(?P<content>.*?){c}response{s}"
            rf"(?:{o}tools{s}(?P<calls>.*){c}tools{s})?",
            re.DOTALL,
        ),
        call=re.compile(
            rf'{o}call tool="(?P<name>[^"]+)" index="(?P<index>\d+)"{s}'
            rf"(?P<arguments>.*?){c}call{s}",
            re.DOTALL,
        ),
        argument=re.compile(
            rf'{o}argument key="(?P<key>[^"]*)" '
            rf'type="(?P<type>string|number|boolean|null|object|array)"{s}'
            rf"(?P<value>.*?){c}argument{s}",
            re.DOTALL,
        ),
    )


_KIMI_CONTROLS = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")
_KIMI_STAND_INS = ("\ue000", "\ue001", "\ue002", "\ue003")
"""Private-use characters that stand in for the sampled control tokens, so text that spells
a control token never matches the grammar."""
_XTML_TEXT = _xtml(*_KIMI_CONTROLS)
_XTML_TOKENS = _xtml(*_KIMI_STAND_INS)
_STAND_IN = dict(zip(_KIMI_CONTROLS, _KIMI_STAND_INS, strict=True))
_SPELLING = str.maketrans(dict(zip(_KIMI_STAND_INS, _KIMI_CONTROLS, strict=True)))


@dataclass(frozen=True, slots=True)
class KimiXtmlParser:
    """Kimi K3's format: the think channel the generation prompt opened, the response
    channel, an optional tools channel, then ``<|close|>message<|sep|>`` and
    ``<|end_of_msg|>``.

    Each argument names its type: a ``string`` is kept as is and any other is decoded as JSON
    of that type, so the declared tools are not needed. Attribute values are unescaped
    (``&quot;``, ``&amp;``). Calls must be numbered from 1 in order, and every argument
    needs a key. Text without the channels is format-invalid, kept whole in ``content``; a
    malformed tools channel stays in ``content`` as raw text.

    Kimi's tokenizer encodes message text apart from its control tokens, so text may spell
    one. :meth:`parse_segments` splits structure only at the sampled control tokens;
    :meth:`parse`, given text alone, takes every spelling as a control token.
    """

    name: ClassVar[str] = "kimi_xtml"
    version: ClassVar[str] = "2"
    control_tokens: ClassVar[tuple[str, ...]] = _KIMI_CONTROLS

    def parse(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        return _parse_xtml(text, _XTML_TEXT)

    def parse_segments(
        self, segments: Sequence[Segment], tools: Sequence[ToolSpec] = ()
    ) -> ParsedResponse:
        text = "".join(_STAND_IN[part] if control else part for part, control in segments)
        planted = any(_marked(part, _XTML_TOKENS) for part, control in segments if not control)
        parsed = _parse_xtml(text, _XTML_TOKENS)
        message = {
            key: value.translate(_SPELLING) if key in ("content", "reasoning_content") else value
            for key, value in parsed.message.items()
        }
        return ParsedResponse(message, format_valid=parsed.format_valid and not planted)


def _parse_xtml(text: str, xtml: _Xtml) -> ParsedResponse:
    body = text.removesuffix(xtml.end)
    closed = body.endswith(xtml.message_close)
    channels = xtml.channels.fullmatch(body.removesuffix(xtml.message_close))
    if channels is None:
        return ParsedResponse(_assistant(body, None, []), format_valid=False)
    content, reasoning = channels["content"], channels["reasoning"]
    calls: list[tuple[str, dict[str, Any]]] = []
    valid = closed
    if channels["calls"] is not None:
        parsed = [
            _kimi_call(match, position, xtml)
            for position, match in enumerate(
                _sequence(channels["calls"], xtml.call, "") or [], start=1
            )
        ]
        if not parsed or None in parsed:
            valid = False
            content += channels["calls"]
        else:
            calls = [call for call in parsed if call is not None]
    if _marked(reasoning, xtml) or _marked(content, xtml):
        valid = False
    return ParsedResponse(_assistant(content, reasoning, calls), format_valid=valid)


_KIMI_TYPES: Mapping[str, tuple[type, ...]] = MappingProxyType(
    {
        "number": (int, float),
        "boolean": (bool,),
        "null": (type(None),),
        "object": (dict,),
        "array": (list,),
    }
)


def _unescape(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _marked(text: str, xtml: _Xtml) -> bool:
    """Whether ``text`` holds a control token, which no name, key or value may."""
    return any(marker in text for marker in xtml.markers)


def _kimi_call(
    call: re.Match[str], position: int, xtml: _Xtml
) -> tuple[str, dict[str, Any]] | None:
    arguments_found = _sequence(call["arguments"], xtml.argument, "")
    if arguments_found is None or call["index"] != str(position) or _marked(call["name"], xtml):
        return None
    arguments: dict[str, Any] = {}
    for argument in arguments_found:
        key, kind = _unescape(argument["key"]), argument["type"]
        if not key or key in arguments or _marked(argument["key"] + argument["value"], xtml):
            return None
        if kind == "string":
            arguments[key] = argument["value"]
            continue
        ok, value = _decoded(argument["value"])
        # bool is an int to Python, but not a number to Kimi.
        if (
            not ok
            or not isinstance(value, _KIMI_TYPES[kind])
            or (kind == "number" and isinstance(value, bool))
        ):
            return None
        arguments[key] = value
    return _unescape(call["name"]), arguments


_ATEM_END = "<|eot|>"
_ATEM_NEXT = "<|eom|><|start|>assistant"
_ATEM_MESSAGE = re.compile(r" to=(?P<to>[^\s<]+)<\|message\|>(?P<body>.*)", re.DOTALL)
_ATEM_CALL = re.compile(
    r'<atem:function_calls>\n<atem:invoke name="(?P<name>[^"\n]+)">\n'
    r"(?P<parameters>.*)</atem:invoke>\n</atem:function_calls>",
    re.DOTALL,
)
_ATEM_PARAMETER = re.compile(
    r'<atem:parameter name="(?P<key>[^"\n]+)">(?P<value>.*?)</atem:parameter>\n', re.DOTALL
)
_ATEM_MARKERS = ("<|start|>", "<|message|>", "<|eom|>", _ATEM_END)


@dataclass(frozen=True, slots=True)
class MuseAtemParser:
    """Muse Glimmer's format: after the generation prompt's ``<|start|>assistant``, an
    optional message to ``self`` (the reasoning), an optional message to ``user`` (the text),
    then one message per tool call, each addressed to the function it calls; messages are
    joined by ``<|eom|><|start|>assistant`` and the turn ends ``<|eot|>``.

    The template writes string values raw and other values as JSON, so a value is read with
    the type the called tool declares, as in :class:`Qwen3XmlToolParser`. Any other order of
    messages, or a malformed call, is format-invalid, with the raw text in ``content``.
    """

    name: ClassVar[str] = "muse_atem"
    version: ClassVar[str] = "2"

    def parse(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        body = text.rstrip().removesuffix(_ATEM_END)
        messages = [_ATEM_MESSAGE.fullmatch(part) for part in body.split(_ATEM_NEXT)]
        reasoning: str | None = None
        first = messages[0]
        if first is not None and first["to"] == "self":
            reasoning, messages = first["body"], messages[1:]
        found = [message for message in messages if message is not None]
        valid = bool(found) and len(found) == len(messages)
        content = ""
        calls: list[tuple[str, dict[str, Any]]] = []
        if valid and found[0]["to"] == "user":
            content, found = found[0]["body"], found[1:]
        if valid:
            properties = _properties(tools)
            parsed = [_atem_call(message, properties) for message in found]
            if None in parsed:
                valid = False
            else:
                calls = [call for call in parsed if call is not None]
        if not valid:
            content = body if reasoning is None else body.partition(_ATEM_NEXT)[2]
        if any(marker in (reasoning or "") or marker in content for marker in _ATEM_MARKERS):
            valid = False
        return ParsedResponse(_assistant(content, reasoning, calls), format_valid=valid)


def _atem_call(
    message: re.Match[str], properties: _Properties
) -> tuple[str, dict[str, Any]] | None:
    call = _ATEM_CALL.fullmatch(message["body"])
    if call is None or call["name"] != message["to"] or message["to"] in ("self", "user"):
        return None
    parameters = _sequence(call["parameters"], _ATEM_PARAMETER, "")
    if parameters is None:
        return None
    schemas = properties.get(call["name"], {})
    arguments: dict[str, Any] = {}
    for parameter in parameters:
        if parameter["key"] in arguments:
            return None
        arguments[parameter["key"]] = _value(parameter["value"], schemas.get(parameter["key"]))
    return call["name"], arguments


PARSERS: Mapping[str, Callable[[], ResponseParser]] = MappingProxyType(
    {
        "qwen3_xml": Qwen3XmlToolParser,
        "qwen3_5_xml": Qwen3_5XmlToolParser,
        "glm_xml": GlmXmlToolParser,
        "json_tool_call": JsonToolCallParser,
        "deepseek_dsml": DeepSeekDsmlParser,
        "kimi_xtml": KimiXtmlParser,
        "muse_atem": MuseAtemParser,
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
