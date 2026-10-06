# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import copy
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pup_finetune.data.schema import (
    RowError,
    RowErrorCode,
    TrainingRow,
    row_content_hash,
    split_key,
    validate_row,
)
from pup_finetune.storage import read_jsonl

ROWS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rows"
VALID_ROWS = list(read_jsonl(ROWS_DIR / "tool_use.jsonl"))
INVALID_ROWS = list(read_jsonl(ROWS_DIR / "invalid.jsonl"))
INVALID_EXPECTED: list[dict[str, Any]] = json.loads(
    (ROWS_DIR / "invalid.expected.json").read_text()
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current conditions for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def call(call_id: str, name: str = "get_weather", arguments: Any = '{"city": "Northvale"}') -> Any:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def row(*messages: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"id": "row-1", "messages": list(messages), **fields}


USER = {"role": "user", "content": "Weather in Northvale?"}
ANSWER = {"role": "assistant", "content": "It's 14°C."}


def code_of(data: dict[str, Any]) -> RowErrorCode:
    with pytest.raises(RowError) as excinfo:
        validate_row(data)
    return excinfo.value.code


# --- fixtures ---------------------------------------------------------------------------


def test_tool_use_fixture_is_big_enough() -> None:
    assert len(VALID_ROWS) >= 12


@pytest.mark.parametrize("data", VALID_ROWS, ids=[r["id"] for r in VALID_ROWS])
def test_every_tool_use_row_validates(data: dict[str, Any]) -> None:
    validated = validate_row(data)
    assert validated.id == data["id"]
    assert validated.messages[-1].role == "assistant"


def test_tool_use_fixture_covers_the_interesting_shapes() -> None:
    rows = [validate_row(data) for data in VALID_ROWS]
    messages = [m for r in rows for m in r.messages]
    assert len({r.id for r in rows}) == len(rows)
    assert any(len(m.tool_calls or ()) > 1 for m in messages), "parallel tool calls"
    assert any(m.reasoning_content for m in messages)
    assert any(m.weight == 0 for m in messages)
    assert any(r.messages[0].role == "system" for r in rows)
    assert any(r.source.trace_id is None for r in rows), "imported rows without a source"
    trace_ids = [r.source.trace_id for r in rows if r.source.trace_id]
    assert len(trace_ids) > len(set(trace_ids)), "two rows share a trace"


def test_invalid_fixture_covers_every_code_once() -> None:
    assert [e["code"] for e in INVALID_EXPECTED] == [code.value for code in RowErrorCode]
    assert [e["id"] for e in INVALID_EXPECTED] == [r["id"] for r in INVALID_ROWS]
    assert [e["line"] for e in INVALID_EXPECTED] == list(range(1, len(INVALID_ROWS) + 1))


@pytest.mark.parametrize(
    ("data", "expected"),
    list(zip(INVALID_ROWS, INVALID_EXPECTED, strict=True)),
    ids=[e["code"] for e in INVALID_EXPECTED],
)
def test_invalid_rows_raise_their_code(data: dict[str, Any], expected: dict[str, Any]) -> None:
    assert code_of(data) == RowErrorCode(expected["code"])


# --- conversation rules -----------------------------------------------------------------


def test_empty_conversation_has_no_assistant_message() -> None:
    assert code_of(row()) == RowErrorCode.NO_ASSISTANT_MESSAGE


def test_system_message_first_is_allowed() -> None:
    validate_row(row({"role": "system", "content": "Be brief."}, USER, ANSWER))


def test_second_system_message_is_rejected() -> None:
    system = {"role": "system", "content": "Be brief."}
    assert code_of(row(system, USER, system, ANSWER)) == RowErrorCode.SYSTEM_NOT_FIRST


def test_tool_result_answering_a_call_twice_is_rejected() -> None:
    data = row(
        USER,
        {"role": "assistant", "tool_calls": [call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "14"},
        {"role": "tool", "tool_call_id": "c1", "content": "14"},
        ANSWER,
    )
    with pytest.raises(RowError, match="already answered") as excinfo:
        validate_row(data)
    assert excinfo.value.code == RowErrorCode.TOOL_RESULT_WITHOUT_CALL


def test_tool_result_before_its_call_is_rejected() -> None:
    data = row(
        USER,
        {"role": "tool", "tool_call_id": "c1", "content": "14"},
        {"role": "assistant", "tool_calls": [call("c1")]},
        ANSWER,
    )
    assert code_of(data) == RowErrorCode.TOOL_RESULT_WITHOUT_CALL


def test_tool_call_ids_must_be_unique_across_messages() -> None:
    data = row(
        USER,
        {"role": "assistant", "tool_calls": [call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "14"},
        {"role": "assistant", "tool_calls": [call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "15"},
        ANSWER,
    )
    assert code_of(data) == RowErrorCode.DUPLICATE_TOOL_CALL_ID


def test_any_tool_name_is_allowed_when_no_tools_are_declared() -> None:
    data = row(
        USER,
        {"role": "assistant", "tool_calls": [call("c1", name="anything")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ANSWER,
    )
    assert validate_row(data).tools == []


def test_declared_tools_are_enforced() -> None:
    data = row(USER, {"role": "assistant", "tool_calls": [call("c1", name="other")]})
    assert code_of({**data, "tools": [WEATHER_TOOL]}) == RowErrorCode.TOOL_NOT_DECLARED


def test_a_trailing_tool_call_is_a_valid_training_target() -> None:
    data = row(USER, {"role": "assistant", "tool_calls": [call("c1")]}, tools=[WEATHER_TOOL])
    assert validate_row(data).messages[-1].tool_calls is not None


def test_last_message_must_be_the_assistant() -> None:
    data = row(
        USER,
        {"role": "assistant", "tool_calls": [call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "14"},
    )
    assert code_of(data) == RowErrorCode.LAST_MESSAGE_NOT_ASSISTANT


@pytest.mark.parametrize(
    "arguments",
    [
        "{oops",
        "[1, 2]",
        '"text"',
        "",
        "null",
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": [-Infinity]}',
    ],
)
def test_arguments_must_be_a_json_object_string(arguments: str) -> None:
    data = row(USER, {"role": "assistant", "tool_calls": [call("c1", arguments=arguments)]})
    assert code_of(data) == RowErrorCode.INVALID_ARGUMENTS_JSON


def test_too_deeply_nested_arguments_are_a_row_error_not_a_crash() -> None:
    depth = 100_000
    arguments = '{"x": ' + "[" * depth + "]" * depth + "}"
    data = row(USER, {"role": "assistant", "tool_calls": [call("c1", arguments=arguments)]})
    with pytest.raises(RowError, match="nested too deeply") as excinfo:
        validate_row(data)
    assert excinfo.value.code == RowErrorCode.INVALID_ARGUMENTS_JSON


# --- generic schema errors --------------------------------------------------------------


def _assistant(**fields: Any) -> dict[str, Any]:
    return {"role": "assistant", "content": "ok", **fields}


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(row(USER, ANSWER, extra="x"), id="unknown-row-field"),
        pytest.param(row({**USER, "name": "someone"}, ANSWER), id="unknown-message-field"),
        pytest.param(row({"role": "developer", "content": "x"}, ANSWER), id="unknown-role"),
        pytest.param({"messages": [USER, ANSWER]}, id="missing-id"),
        pytest.param(row(USER, ANSWER, id=""), id="empty-id"),
        pytest.param({"id": "row-1"}, id="missing-messages"),
        pytest.param(row(USER, ANSWER, schema_version=2), id="unknown-schema-version"),
        pytest.param(row({**USER, "content": 3}, ANSWER), id="non-string-content"),
        pytest.param(row({**USER, "content": [{"type": "text"}]}, ANSWER), id="content-parts"),
        pytest.param(
            row(USER, {"role": "assistant", "tool_calls": [call("c1", arguments={"a": 1})]}),
            id="arguments-not-a-string",
        ),
        pytest.param(
            row(USER, {"role": "assistant", "tool_calls": [{**call("c1"), "type": "x"}]}),
            id="unknown-tool-call-type",
        ),
        pytest.param(
            row(USER, {"role": "assistant", "tool_calls": [call("")]}), id="empty-call-id"
        ),
        pytest.param(row({**USER, "tool_calls": [call("c1")]}, ANSWER), id="user-tool-calls"),
        pytest.param(row({**USER, "reasoning_content": "x"}, ANSWER), id="user-reasoning"),
        pytest.param(row({**USER, "weight": 0}, ANSWER), id="user-weight"),
        pytest.param(row(USER, _assistant(tool_call_id="c1")), id="assistant-tool-call-id"),
        pytest.param(row(USER, _assistant(weight=2)), id="weight-2"),
        pytest.param(row(USER, _assistant(weight=True)), id="weight-bool"),
        pytest.param(row(USER, _assistant(weight=1.0)), id="weight-float"),
        pytest.param(row(USER, ANSWER, source={"trace_id": 7}), id="bad-source"),
        pytest.param(row(USER, ANSWER, schema_version=True), id="schema-version-bool"),
        pytest.param(row(USER, ANSWER, schema_version=1.0), id="schema-version-float"),
        pytest.param(row(USER, ANSWER, schema_version="1"), id="schema-version-str"),
        pytest.param(row(USER, ANSWER, source={"start_ns": True}), id="start-ns-bool"),
        pytest.param(row(USER, ANSWER, source={"start_ns": 1.0}), id="start-ns-float"),
        pytest.param(row(USER, ANSWER, source={"start_ns": "123"}), id="start-ns-str"),
        pytest.param(row(USER, ANSWER, tools=[{"function": {}}]), id="tool-without-name"),
    ],
)
def test_schema_violations_are_invalid_rows(data: dict[str, Any]) -> None:
    assert code_of(data) == RowErrorCode.INVALID_ROW


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_in_parameters_and_evaluations_are_invalid_rows(number: float) -> None:
    tool = {"function": {"name": "get_weather", "parameters": {"enum": [1, {"x": number}]}}}
    with pytest.raises(RowError) as info:
        validate_row(row(USER, ANSWER, tools=[tool]))
    assert info.value.code == RowErrorCode.INVALID_ROW
    assert "parameters: enum[1].x is" in info.value.detail
    source = {"evaluations": {"score": number}}
    with pytest.raises(RowError) as info:
        validate_row(row(USER, ANSWER, source=source))
    assert info.value.code == RowErrorCode.INVALID_ROW
    assert "evaluations: score is" in info.value.detail


def test_tool_message_needs_a_tool_call_id() -> None:
    data = row(
        USER,
        {"role": "assistant", "tool_calls": [call("c1")]},
        {"role": "tool", "content": "14"},
        ANSWER,
    )
    with pytest.raises(RowError, match="tool_call_id") as excinfo:
        validate_row(data)
    assert excinfo.value.code == RowErrorCode.INVALID_ROW


def test_error_detail_names_the_location() -> None:
    data = row(USER, {"role": "assistant", "tool_calls": [call("c1", arguments="{")]})
    with pytest.raises(RowError) as excinfo:
        validate_row(data)
    assert excinfo.value.detail.startswith("messages[1].tool_calls[0].function.arguments: ")
    assert str(excinfo.value).startswith("invalid_arguments_json: messages[1]")


def test_row_error_is_a_value_error_with_a_stable_code() -> None:
    error = RowError(RowErrorCode.SYSTEM_NOT_FIRST, "detail")
    assert isinstance(error, ValueError)
    assert error.code == "system_not_first"
    assert str(error) == "system_not_first: detail"
    assert str(RowError(RowErrorCode.INVALID_ROW, "d", path="f.jsonl", line=3)) == (
        "f.jsonl:3: invalid_row: d"
    )


@pytest.mark.parametrize(
    "clone",
    [copy.copy, lambda e: pickle.loads(pickle.dumps(e))],  # noqa: S301 - our own bytes
)
def test_row_error_survives_copy_and_pickle(clone: Any) -> None:
    error = RowError(RowErrorCode.INVALID_ROW, "d", path="f.jsonl", line=3)
    cloned = clone(error)
    assert (cloned.code, cloned.detail, cloned.path, cloned.line) == (
        RowErrorCode.INVALID_ROW,
        "d",
        "f.jsonl",
        3,
    )
    assert str(cloned) == str(error)


def test_codes_are_snake_case_and_unique() -> None:
    values = [code.value for code in RowErrorCode]
    assert len(values) == len(set(values))
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)*", value) for value in values)


# --- normalization and defaults ---------------------------------------------------------


def test_null_content_and_empty_tool_calls_are_normalized() -> None:
    validated = validate_row(row(USER, {"role": "assistant", "content": None, "tool_calls": []}))
    assert validated.messages[1].content == ""
    assert validated.messages[1].tool_calls is None


def test_defaults_for_an_imported_row() -> None:
    validated = validate_row(row(USER, ANSWER))
    assert validated.tools == []
    assert validated.schema_version == 1
    assert validated.source.trace_id is None
    assert validated.source.evaluations == {}
    assert validated.source.tags == []
    assert validated.messages[1].weight is None


def test_explicit_integer_schema_version_and_start_ns_are_kept() -> None:
    validated = validate_row(row(USER, ANSWER, schema_version=1, source={"start_ns": 0}))
    assert validated.schema_version == 1
    assert validated.source.start_ns == 0
    assert validate_row(row(USER, ANSWER, source={"start_ns": None})).source.start_ns is None


def test_rows_are_frozen() -> None:
    validated = validate_row(row(USER, ANSWER))
    with pytest.raises(ValidationError):
        validated.id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        validated.messages[0].content = "changed"  # type: ignore[misc]


# --- content hash and split key ---------------------------------------------------------


def _reverse_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


@pytest.mark.parametrize("data", VALID_ROWS, ids=[r["id"] for r in VALID_ROWS])
def test_content_hash_ignores_key_order(data: dict[str, Any]) -> None:
    assert row_content_hash(validate_row(data)) == row_content_hash(
        validate_row(_reverse_keys(data))
    )


def test_content_hash_is_a_stable_sha256() -> None:
    first = row_content_hash(validate_row(row(USER, ANSWER)))
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert row_content_hash(validate_row(row(USER, ANSWER))) == first


# Golden values: hashes are compared across runs and versions, so a change to the canonical
# JSON (key order, separators, non-ASCII escaping) must fail here rather than pass silently.
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            row(USER, ANSWER),
            "0aeb950135974b5fc74206acbc53eb8ef7c57993db67b5871663ae103d5b6b87",
            id="small-row",
        ),
        pytest.param(
            next(r for r in VALID_ROWS if r["id"] == "row-weather-001"),
            "f7971b6ddb678987d8ad402f04e046aa222524395159e235421600af5c7b9ec0",
            id="row-weather-001",
        ),
    ],
)
def test_content_hash_golden_values(data: dict[str, Any], expected: str) -> None:
    assert row_content_hash(validate_row(data)) == expected


def test_content_hash_ignores_id_source_and_weight() -> None:
    base = row_content_hash(validate_row(row(USER, ANSWER)))
    other = row(
        USER,
        {**ANSWER, "weight": 0},
        id="row-2",
        source={"trace_id": "synthetic-trace-9", "tags": ["env:synthetic"]},
    )
    assert row_content_hash(validate_row(other)) == base


def test_content_hash_treats_explicit_defaults_like_omitted_ones() -> None:
    calling = {"role": "assistant", "tool_calls": [call("c1")]}
    base = row_content_hash(validate_row(row(USER, calling)))
    explicit = {**calling, "content": None, "reasoning_content": None, "weight": None}
    assert row_content_hash(validate_row(row(USER, explicit, tools=[]))) == base


@pytest.mark.parametrize(
    "changed",
    [
        row(USER, {**ANSWER, "content": "It's 15°C."}),
        row(USER, {**ANSWER, "reasoning_content": "Read the result."}),
        row(USER, ANSWER, tools=[WEATHER_TOOL]),
        row({"role": "system", "content": "Be brief."}, USER, ANSWER),
    ],
)
def test_content_hash_changes_with_content(changed: dict[str, Any]) -> None:
    assert row_content_hash(validate_row(changed)) != row_content_hash(
        validate_row(row(USER, ANSWER))
    )


def test_content_hash_covers_tool_call_arguments() -> None:
    def calling(arguments: str) -> dict[str, Any]:
        return row(USER, {"role": "assistant", "tool_calls": [call("c1", arguments=arguments)]})

    base = row_content_hash(validate_row(calling('{"city": "Northvale"}')))
    assert row_content_hash(validate_row(calling('{"city": "Eastport"}'))) != base


def test_content_hash_depends_on_turn_order() -> None:
    first = {"role": "user", "content": "First question."}
    second = {"role": "user", "content": "Second question."}
    reply = {"role": "assistant", "content": "Noted."}
    base = row_content_hash(validate_row(row(first, reply, second, ANSWER)))
    assert row_content_hash(validate_row(row(second, reply, first, ANSWER))) != base


def test_split_key_prefers_the_trace_id() -> None:
    traced = validate_row(row(USER, ANSWER, source={"trace_id": "synthetic-trace-1"}))
    assert split_key(traced) == "synthetic-trace-1"
    assert split_key(validate_row(row(USER, ANSWER))) == "row-1"
    assert split_key(validate_row(row(USER, ANSWER, source={"trace_id": ""}))) == "row-1"


def test_rows_from_one_trace_share_a_split_key() -> None:
    rows = {data["id"]: validate_row(data) for data in VALID_ROWS}
    main, sub_agent = rows["row-support-001"], rows["row-support-002"]
    assert main.id != sub_agent.id
    assert split_key(main) == split_key(sub_agent)


def test_training_row_constructs_from_models() -> None:
    validated = validate_row(row(USER, ANSWER))
    rebuilt = TrainingRow(id="row-1", messages=validated.messages)
    assert rebuilt == validated
