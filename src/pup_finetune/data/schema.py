# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The canonical training row: one chat conversation in OpenAI chat format, plus its source.

Every provider consumes this shape, so it is the one data contract between curation and
training. A row is either valid as a whole or rejected with a :class:`RowError` whose
:class:`RowErrorCode` is stable. Rows are never repaired, beyond reading a ``null`` content
as ``""`` and an empty ``tool_calls`` list as unset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, NoReturn, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from pup_finetune.storage import canonical_json

SCHEMA_VERSION: Literal[1] = 1

Role = Literal["system", "user", "assistant", "tool"]


class RowErrorCode(StrEnum):
    """Why a row is invalid. The values are stable: they are recorded with excluded rows."""

    NO_ASSISTANT_MESSAGE = "no_assistant_message"
    SYSTEM_NOT_FIRST = "system_not_first"
    TOOL_RESULT_WITHOUT_CALL = "tool_result_without_call"
    DUPLICATE_TOOL_CALL_ID = "duplicate_tool_call_id"
    TOOL_NOT_DECLARED = "tool_not_declared"
    LAST_MESSAGE_NOT_ASSISTANT = "last_message_not_assistant"
    INVALID_ARGUMENTS_JSON = "invalid_arguments_json"
    INVALID_ROW = "invalid_row"
    """Any other schema violation: a missing or unknown field, a wrong type or value."""


class RowError(ValueError):
    """A row that breaks the canonical schema. ``code`` says why; ``detail`` says where.

    ``path`` and ``line`` are set when the row was read from a JSONL file.
    """

    def __init__(
        self,
        code: RowErrorCode,
        detail: str,
        *,
        path: str | None = None,
        line: int | None = None,
    ) -> None:
        self.code = RowErrorCode(code)
        self.detail = detail
        self.path = path
        self.line = line
        super().__init__(str(self))

    def __reduce__(self) -> tuple[Any, ...]:
        # ``args`` holds only the message, so the default would call ``__init__`` wrongly.
        return (type(self), (self.code, self.detail), {"path": self.path, "line": self.line})

    def __str__(self) -> str:
        where = ""
        if self.path is not None:
            where = f"{self.path}:{self.line}: " if self.line is not None else f"{self.path}: "
        elif self.line is not None:
            where = f"line {self.line}: "
        return f"{where}{self.code.value}: {self.detail}"


_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


def _require_int(value: object, message: str) -> object:
    """Reject anything but a real ``int`` (or ``None``), before pydantic's lax coercion.

    Without this, ``true``, ``1.0`` and ``"1"`` would pass as integers.
    """
    if value is not None and type(value) is not int:
        raise RowError(RowErrorCode.INVALID_ROW, message)
    return value


NonEmptyStr = Annotated[str, Field(min_length=1)]


class _NonStandardJSONError(ValueError):
    """A ``NaN`` or ``Infinity`` constant: Python's ``json`` accepts them, JSON does not."""


def _reject_json_constant(constant: str) -> NoReturn:
    raise _NonStandardJSONError(f"{constant} is not a JSON value")


class FunctionCall(BaseModel):
    """The function an assistant calls; ``arguments`` is a JSON object encoded as a string."""

    model_config = _MODEL_CONFIG

    name: NonEmptyStr
    arguments: str

    @field_validator("arguments")
    @classmethod
    def _arguments_are_a_json_object(cls, value: str) -> str:
        try:
            parsed = json.loads(value, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise RowError(
                RowErrorCode.INVALID_ARGUMENTS_JSON, f"arguments are not valid JSON: {exc.msg}"
            ) from exc
        except _NonStandardJSONError as exc:
            raise RowError(
                RowErrorCode.INVALID_ARGUMENTS_JSON, f"arguments are not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RowError(
                RowErrorCode.INVALID_ARGUMENTS_JSON, "arguments must encode a JSON object"
            )
        return value


class ToolCall(BaseModel):
    """One tool call in an assistant message."""

    model_config = _MODEL_CONFIG

    id: NonEmptyStr
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    """One chat message.

    ``tool_calls``, ``reasoning_content`` and ``weight`` belong to assistant messages only;
    ``tool_call_id`` is required on tool messages and allowed nowhere else. ``weight`` ``0``
    means "do not train on this turn"; ``1`` or unset means train on it. A ``null`` content
    is read as ``""`` and an empty ``tool_calls`` list as unset, as OpenAI chat files use both.
    """

    model_config = _MODEL_CONFIG

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: NonEmptyStr | None = None
    reasoning_content: str | None = None
    weight: Literal[0, 1] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _null_content_is_empty(cls, value: object) -> object:
        return "" if value is None else value

    @field_validator("weight", mode="before")
    @classmethod
    def _weight_is_an_integer(cls, value: object) -> object:
        return _require_int(value, "weight must be the integer 0 or 1")

    @field_validator("tool_calls")
    @classmethod
    def _empty_tool_calls_are_unset(cls, value: list[ToolCall] | None) -> list[ToolCall] | None:
        return value or None

    @model_validator(mode="after")
    def _fields_match_role(self) -> Self:
        if self.role != "assistant":
            for name in ("tool_calls", "reasoning_content", "weight"):
                if getattr(self, name) is not None:
                    raise RowError(
                        RowErrorCode.INVALID_ROW, f"{name} is only allowed on assistant messages"
                    )
        if self.role == "tool" and self.tool_call_id is None:
            raise RowError(RowErrorCode.INVALID_ROW, "a tool message needs a tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise RowError(
                RowErrorCode.INVALID_ROW, "tool_call_id is only allowed on tool messages"
            )
        return self


class ToolFunction(BaseModel):
    """A declared function: its name, description and JSON Schema parameters."""

    model_config = _MODEL_CONFIG

    name: NonEmptyStr
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """A tool the conversation declares."""

    model_config = _MODEL_CONFIG

    type: Literal["function"] = "function"
    function: ToolFunction


class SourceRef(BaseModel):
    """Where a row came from. Every field is optional: imported rows may have no source."""

    model_config = _MODEL_CONFIG

    trace_id: str | None = None
    span_id: str | None = None
    ml_app: str | None = None
    model_name: str | None = None
    start_ns: int | None = None
    evaluations: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("start_ns", mode="before")
    @classmethod
    def _start_ns_is_an_integer(cls, value: object) -> object:
        return _require_int(value, "start_ns must be an integer number of nanoseconds")


class TrainingRow(BaseModel):
    """One conversation to train on. Construct rows from untrusted data with :func:`validate_row`."""

    model_config = _MODEL_CONFIG

    id: NonEmptyStr
    messages: list[Message]
    tools: list[Tool] = Field(default_factory=list)
    source: SourceRef = Field(default_factory=SourceRef)
    schema_version: Literal[1] = SCHEMA_VERSION

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version_is_an_integer(cls, value: object) -> object:
        return _require_int(value, f"schema_version must be the integer {SCHEMA_VERSION}")

    @model_validator(mode="after")
    def _conversation_is_well_formed(self) -> Self:
        _check_conversation(self.messages, self.tools)
        return self


def _check_conversation(messages: list[Message], tools: list[Tool]) -> None:
    """Raise :class:`RowError` for the first structural problem in ``messages``."""
    if not any(message.role == "assistant" for message in messages):
        raise RowError(RowErrorCode.NO_ASSISTANT_MESSAGE, "the row has no assistant message")
    declared = {tool.function.name for tool in tools}
    seen_ids: set[str] = set()
    pending_ids: set[str] = set()
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if message.role == "system" and index != 0:
            raise RowError(
                RowErrorCode.SYSTEM_NOT_FIRST, f"{where}: a system message must come first"
            )
        if message.role == "tool":
            call_id = message.tool_call_id
            if call_id not in pending_ids:
                problem = "was already answered" if call_id in seen_ids else "has no earlier call"
                raise RowError(
                    RowErrorCode.TOOL_RESULT_WITHOUT_CALL,
                    f"{where}: the tool result for {call_id!r} {problem}",
                )
            pending_ids.discard(call_id)
        for call in message.tool_calls or ():
            if call.id in seen_ids:
                raise RowError(
                    RowErrorCode.DUPLICATE_TOOL_CALL_ID,
                    f"{where}: tool call id {call.id!r} is used more than once",
                )
            if declared and call.function.name not in declared:
                raise RowError(
                    RowErrorCode.TOOL_NOT_DECLARED,
                    f"{where}: tool {call.function.name!r} is not in the row's tools",
                )
            seen_ids.add(call.id)
            pending_ids.add(call.id)
    if messages[-1].role != "assistant":
        raise RowError(
            RowErrorCode.LAST_MESSAGE_NOT_ASSISTANT,
            f"the last message is from {messages[-1].role!r}, not the assistant",
        )


def validate_row(data: Mapping[str, Any]) -> TrainingRow:
    """Parse ``data`` into a :class:`TrainingRow`, or raise :class:`RowError` with its code.

    When several problems exist, the first one pydantic reports decides the code.
    """
    try:
        return TrainingRow.model_validate(data)
    except ValidationError as exc:
        raise _row_error(exc) from exc


def _row_error(exc: ValidationError) -> RowError:
    first = exc.errors()[0]
    location = _format_location(first["loc"])
    cause = first.get("ctx", {}).get("error")
    if isinstance(cause, RowError):
        detail = f"{location}: {cause.detail}" if location else cause.detail
        return RowError(cause.code, detail)
    message = first["msg"]
    return RowError(RowErrorCode.INVALID_ROW, f"{location}: {message}" if location else message)


def _format_location(loc: tuple[int | str, ...]) -> str:
    """``("messages", 1, "content")`` -> ``messages[1].content``."""
    text = ""
    for part in loc:
        text += f"[{part}]" if isinstance(part, int) else f".{part}" if text else part
    return text


def _content_payload(row: TrainingRow) -> dict[str, Any]:
    # Weight is training metadata, not content: the same conversation is the same row.
    return {
        "messages": [
            message.model_dump(mode="json", exclude_none=True, exclude={"weight"})
            for message in row.messages
        ],
        "tools": [tool.model_dump(mode="json", exclude_none=True) for tool in row.tools],
    }


def row_content_hash(row: TrainingRow) -> str:
    """The sha256 hex digest of the row's messages and tools as canonical JSON.

    The id, source, schema version and per-message weights are left out, so two rows with
    the same conversation hash the same. Used for deduplication and split-leakage checks.
    """
    return hashlib.sha256(canonical_json(_content_payload(row)).encode("utf-8")).hexdigest()


def split_key(row: TrainingRow) -> str:
    """The key a row is assigned to a split by: its trace id, so a trace never straddles splits."""
    return row.source.trace_id or row.id
