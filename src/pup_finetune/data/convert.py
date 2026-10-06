# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""One LLM span as one chat conversation in the canonical (OpenAI chat) shape.

:func:`span_conversation` maps a span's ``input.messages`` followed by its
``output.messages``, and its ``tool_definitions``, to canonical message and tool dicts:

=========================================  ====================================================
LLM span                                   canonical
=========================================  ====================================================
``{role, content}``                        the same role and content; a missing role is
                                           ``user`` on input and ``assistant`` on output
``role: "reasoning"``                      dropped, or with ``reasoning="preserve"`` the
                                           ``reasoning_content`` of the next assistant message
                                           in the same list (the previous one if none follows)
``tool_calls[{name, arguments, tool_id}]`` ``tool_calls[{id, type: "function", function:
                                           {name, arguments}}]``, ``arguments`` as a compact
                                           JSON object string
``tool_results[{result, tool_id}]``        one ``{role: "tool", tool_call_id, content}`` per
                                           result, before the message's own content
``tool_definitions[{name, description,     ``tools[{type: "function", function: {name,
schema}]``                                 description, parameters}}]``
=========================================  ====================================================

A content list is joined when every part is text. Assistant messages with neither content
nor tool calls are dropped. Nothing else is repaired: what cannot be mapped faithfully is
recorded in :attr:`Conversation.issues` with a stable code, and the schema rules (roles,
call and result pairing, declared tools) are left to
:func:`~pup_finetune.data.schema.validate_row`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, get_args

from pydantic import ValidationError

from pup_finetune.data.exclusions import ExclusionCode, ExclusionReason
from pup_finetune.data.schema import RowErrorCode, SourceRef
from pup_finetune.datadog.spans import LLMObsMessage, LLMObsSpan, LLMObsToolCall
from pup_finetune.errors import UsageError

ReasoningMode = Literal["omit", "preserve"]
"""What to do with ``role: "reasoning"`` messages: drop them, or keep them as
``reasoning_content`` on an assistant message."""

REASONING_MODES: tuple[ReasoningMode, ...] = get_args(ReasoningMode)


def check_reasoning(reasoning: str) -> None:
    """Raise :class:`~pup_finetune.errors.UsageError` unless ``reasoning`` is a known mode."""
    if reasoning not in REASONING_MODES:
        raise UsageError(
            f"reasoning must be one of {', '.join(REASONING_MODES)}, not {reasoning!r}"
        )


TEXT_PART_SEPARATOR = "\n"
"""Joins the text parts of a content list."""

Issue = tuple[ExclusionReason, str]
"""A conversion problem: its stable code and where it is."""


@dataclass(frozen=True)
class Conversation:
    """One span's conversation, ready to validate as a row.

    ``messages`` and ``tools`` are canonical dicts (see :mod:`pup_finetune.data.schema`).
    ``issues`` lists what could not be converted faithfully; a conversation with issues is
    not a row.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    source: SourceRef
    issues: list[Issue] = field(default_factory=list)


def span_conversation(span: LLMObsSpan, *, reasoning: ReasoningMode = "omit") -> Conversation:
    """The conversation ``span`` records: its input messages, then its output messages."""
    check_reasoning(reasoning)
    builder = _Builder(span.span_id, reasoning)
    builder.add_section(span.input.messages if span.input else None, default_role="user")
    builder.add_section(span.output.messages if span.output else None, default_role="assistant")
    source, source_issue = _source(span)
    issues = builder.issues if source_issue is None else [*builder.issues, source_issue]
    return Conversation(builder.messages, _tools(span), source, issues)


# --- messages -------------------------------------------------------------------------------


@dataclass
class _Call:
    id: str
    name: str | None
    answered: bool = False


class _Builder:
    """Accumulates canonical messages, tracking tool calls so results can be paired."""

    def __init__(self, span_id: str, reasoning: ReasoningMode) -> None:
        self.span_id = span_id
        self.reasoning = reasoning
        self.messages: list[dict[str, Any]] = []
        self.issues: list[Issue] = []
        self._calls: list[_Call] = []
        self._source_index = 0
        """The index of the next source message across input and output, for generated ids."""

    def add_section(self, section: Sequence[LLMObsMessage] | None, *, default_role: str) -> None:
        """Append one list of source messages (a span's input or its output)."""
        start = len(self.messages)
        # Reasoning texts, each with the number of messages this section had produced before it.
        reasoning: list[tuple[int, str]] = []
        for message in section or ():
            index = self._source_index
            self._source_index += 1
            role = message.role or default_role
            if role == "reasoning":
                if self.reasoning != "preserve":
                    continue
                # A non-text part is recorded as an issue and never becomes reasoning_content.
                text = self._content(message, index)
                if isinstance(text, str) and text:
                    reasoning.append((len(self.messages) - start, text))
                continue
            self._add_message(message, role, index)
        if reasoning:
            _attach_reasoning(self.messages[start:], reasoning)

    def _add_message(self, message: LLMObsMessage, role: str, index: int) -> None:
        if message.tool_results:
            results = message.tool_results
            content_used = False
            for result in results:
                call_id = self._answer(result.tool_id, result.name, index)
                if call_id is not None:
                    output: str | list[Any] = result.result or ""
                    if role == "tool" and len(results) == 1 and result.result is None:
                        # The single result's output lives in the message content instead.
                        output = self._content(message, index)
                        content_used = True
                    self.messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": output}
                    )
            if role == "tool":
                # The results are the message; its own content is not kept apart from them,
                # but a part that is not text still excludes the conversation: dropping it
                # would hide what the model saw.
                if not content_used:
                    self._content(message, index)
                return
        elif role == "tool":
            if not self._calls:
                # No structured tool call came before it: the app speaks a text-based tool
                # protocol (calls written into assistant content), which a chat template
                # cannot render as tool turns.
                self.issues.append(
                    (
                        ExclusionCode.UNSTRUCTURED_TOOL_MESSAGE,
                        f"message {index}: a tool message without structured tool calls or "
                        "results (the app likely writes tool calls as text)",
                    )
                )
                return
            call_id = self._answer(None, None, index)
            if call_id is not None:
                content = self._content(message, index)
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
            return
        content = self._content(message, index)
        calls = [self._tool_call(call, index, n) for n, call in enumerate(message.tool_calls)]
        if message.tool_results and not content and not calls:
            return
        if role == "assistant" and not content and not calls:
            return
        canonical: dict[str, Any] = {"role": role, "content": content}
        if calls:
            canonical["tool_calls"] = calls
        self.messages.append(canonical)

    def _content(self, message: LLMObsMessage, index: int) -> str | list[Any]:
        content = message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        texts: list[str] = []
        for part_index, part in enumerate(content):
            text = _part_text(part)
            if text is None:
                self.issues.append(
                    (
                        ExclusionCode.NON_TEXT_CONTENT,
                        f"message {index} content part {part_index} is {_describe(part)}",
                    )
                )
                # Kept as given, so the conversation does not pass for a text-only one.
                return list(content)
            texts.append(text)
        return TEXT_PART_SEPARATOR.join(texts)

    def _tool_call(self, call: LLMObsToolCall, index: int, call_index: int) -> dict[str, Any]:
        call_id = call.tool_id or _generated_call_id(self.span_id, index, call_index)
        self._calls.append(_Call(call_id, call.name))
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": call.name or "",
                "arguments": self._arguments(call.arguments, index, call_index),
            },
        }

    def _arguments(self, arguments: dict[str, Any] | str | None, index: int, n: int) -> str:
        where = f"message {index} tool call {n}"
        if arguments is None:
            return "{}"
        if isinstance(arguments, dict):
            try:
                return json.dumps(
                    arguments, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                )
            except (ValueError, TypeError, RecursionError) as exc:
                reason = "nested too deeply" if isinstance(exc, RecursionError) else str(exc)
                self.issues.append(
                    (RowErrorCode.INVALID_ARGUMENTS_JSON, f"{where}: arguments are {reason}")
                )
                return "{}"
        if not _is_json_object(arguments):
            self.issues.append(
                (
                    RowErrorCode.INVALID_ARGUMENTS_JSON,
                    f"{where}: arguments are not a JSON object string",
                )
            )
        return arguments

    def _answer(self, tool_id: str | None, name: str | None, index: int) -> str | None:
        """The call id a tool result answers, marking the call answered.

        A result with an id answers that id; whether a call made it is left to the schema
        check. One without an id answers the most recent unanswered call to the same tool,
        or, with no tool name either, the only unanswered call. ``None`` means no call
        matched, and is recorded as an issue.
        """
        if tool_id:
            for call in self._calls:
                if call.id == tool_id:
                    call.answered = True
            return tool_id
        pending = [call for call in self._calls if not call.answered]
        if name is not None:
            matches = [call for call in pending if call.name == name]
        else:
            matches = pending if len(pending) == 1 else []
        if not matches:
            tool = f"tool {_escaped(name)!r}" if name is not None else "an unnamed tool"
            self.issues.append(
                (
                    RowErrorCode.TOOL_RESULT_WITHOUT_CALL,
                    f"message {index}: a result for {tool} has no id and matches no open call",
                )
            )
            return None
        matches[-1].answered = True
        return matches[-1].id


def _attach_reasoning(section: list[dict[str, Any]], reasoning: list[tuple[int, str]]) -> None:
    """Set ``reasoning_content`` from each reasoning text on the next assistant message.

    ``section`` holds the canonical messages one source list produced; each reasoning text
    carries how many of them came before it. With no assistant message after it, the
    previous one takes it; with none at all, it is dropped.
    """
    assistants = [i for i, message in enumerate(section) if message["role"] == "assistant"]
    if not assistants:
        return
    for position, text in reasoning:
        following = [i for i in assistants if i >= position]
        target = section[following[0] if following else assistants[-1]]
        previous = target.get("reasoning_content")
        target["reasoning_content"] = f"{previous}\n\n{text}" if previous else text


def _part_text(part: object) -> str | None:
    """A content part's text, or ``None`` when it is not a text part."""
    if isinstance(part, str):
        return part
    if isinstance(part, dict) and part.get("type", "text") == "text":
        text = part.get("text")
        if isinstance(text, str):
            return text
    return None


def _describe(part: object) -> str:
    """What a content part that has no text is, for an issue's detail."""
    if not isinstance(part, dict):
        return f"a part of type {type(part).__name__!r}"
    kind = part.get("type", "text")
    if kind == "text":
        return "a text part with no string text"
    if isinstance(kind, str):
        return f"a part of type {_escaped(kind)!r}"
    return "a part with no type"


def _escaped(text: str) -> str:
    """``text`` with any lone surrogate escaped, so a detail can always be written."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _generated_call_id(span_id: str, index: int, call_index: int) -> str:
    """A deterministic id for a tool call that has none: ``call_`` and 16 hex digits."""
    key = f"{span_id}:{index}:{call_index}".encode("utf-8", "surrogatepass")
    return f"call_{hashlib.sha256(key).hexdigest()[:16]}"


class _NotJSON(ValueError):
    pass


def _reject_constant(constant: str) -> NoReturn:
    raise _NotJSON(constant)


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text, parse_constant=_reject_constant), dict)
    except (ValueError, RecursionError):
        return False


# --- tools and source -----------------------------------------------------------------------


def _tools(span: LLMObsSpan) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": definition.name or "",
                "description": definition.description or "",
                "parameters": definition.schema_ or _empty_parameters(),
            },
        }
        for definition in span.tool_definitions
    ]


def _empty_parameters() -> dict[str, Any]:
    """The parameters of a declared tool whose definition has no schema: any object."""
    return {"type": "object", "properties": {}}


def _source(span: LLMObsSpan) -> tuple[SourceRef, Issue | None]:
    fields: dict[str, Any] = {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "ml_app": span.ml_app,
        "model_name": span.model_name,
        "start_ns": span.start_ns,
        "tags": list(span.tags),
    }
    try:
        return SourceRef(**fields, evaluations=dict(span.evaluation)), None
    except ValidationError as exc:
        # The evaluations cannot be recorded (a NaN score, say): the row cannot be either.
        detail = f"source.evaluations: {exc.errors()[0]['msg']}"
        return SourceRef(**fields), (RowErrorCode.INVALID_ROW, _escaped(detail))
