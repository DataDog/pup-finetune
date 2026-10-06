# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data.jsonl import (
    read_rows,
    row_to_dict,
    scan_rows,
    to_provider_dict,
    write_rows,
)
from pup_finetune.data.schema import RowError, RowErrorCode, TrainingRow, validate_row
from pup_finetune.errors import StateError

ROWS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rows"
TOOL_USE = ROWS_DIR / "tool_use.jsonl"


@pytest.fixture
def loose_umask() -> Iterator[None]:
    previous = os.umask(0o002)
    yield
    os.umask(previous)


@pytest.fixture
def fixture_rows() -> list[TrainingRow]:
    return list(read_rows(TOOL_USE))


def by_id(rows: list[TrainingRow], row_id: str) -> TrainingRow:
    return next(r for r in rows if r.id == row_id)


def walk_values(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_values(item)


# --- read_rows --------------------------------------------------------------------------


def test_read_rows_reads_every_fixture_row(fixture_rows: list[TrainingRow]) -> None:
    expected_ids = [json.loads(line)["id"] for line in TOOL_USE.read_text().splitlines()]
    assert [r.id for r in fixture_rows] == expected_ids


def test_read_rows_names_the_file_and_line_of_an_invalid_row(tmp_path: Path) -> None:
    valid = TOOL_USE.read_text().splitlines()[0]
    invalid = json.dumps({"id": "bad", "messages": [{"role": "user", "content": "hi"}]})
    path = tmp_path / "rows.jsonl"
    path.write_text(f"{valid}\n\n{invalid}\n")
    rows = read_rows(path)
    assert next(rows).id == "row-weather-001"
    with pytest.raises(RowError) as excinfo:
        next(rows)
    error = excinfo.value
    assert error.code == RowErrorCode.NO_ASSISTANT_MESSAGE
    assert (error.path, error.line) == (str(path), 3)
    assert str(error).startswith(f"{path}:3: no_assistant_message: ")


INVALID_EXPECTED: list[dict[str, Any]] = json.loads(
    (ROWS_DIR / "invalid.expected.json").read_text()
)


@pytest.mark.parametrize("expected", INVALID_EXPECTED, ids=[e["code"] for e in INVALID_EXPECTED])
def test_read_rows_reports_each_invalid_fixture_line(
    tmp_path: Path, expected: dict[str, Any]
) -> None:
    # Put the invalid row on its recorded line, after valid rows, so each entry is reached.
    line = expected["line"]
    invalid = (ROWS_DIR / "invalid.jsonl").read_text().splitlines()[line - 1]
    valid = TOOL_USE.read_text().splitlines()[: line - 1]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join([*valid, invalid]) + "\n")
    with pytest.raises(RowError) as excinfo:
        list(read_rows(path))
    error = excinfo.value
    assert (error.line, error.code, error.path) == (line, expected["code"], str(path))


@pytest.mark.parametrize("number", ["NaN", "-Infinity", "1e999"])
def test_read_rows_reports_non_finite_numbers_as_invalid_json(tmp_path: Path, number: str) -> None:
    # Python's json reads these as floats; strict JSON has no such values, so the line is not JSON.
    valid = json.loads(TOOL_USE.read_text().splitlines()[0])
    valid["source"]["evaluations"] = {"score": "SCORE"}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(valid).replace('"SCORE"', number) + "\n")
    with pytest.raises(StateError, match=rf"rows\.jsonl:1: invalid JSON: {number} "):
        list(read_rows(path))


def test_read_rows_reports_an_escaped_lone_surrogate_as_a_row_error(tmp_path: Path) -> None:
    # Valid JSON, but the decoded text cannot be hashed or written back as UTF-8.
    valid = json.loads(TOOL_USE.read_text().splitlines()[0])
    valid["messages"][0]["content"] = "CONTENT"
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(valid).replace('"CONTENT"', '"\\ud800"') + "\n")
    with pytest.raises(RowError) as excinfo:
        list(read_rows(path))
    assert (excinfo.value.code, excinfo.value.line) == (RowErrorCode.INVALID_ROW, 1)
    assert "messages[0].content has a lone surrogate" in excinfo.value.detail


def test_read_rows_rejects_lines_that_are_not_objects(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(StateError, match=":1: expected a JSON object"):
        list(read_rows(path))


# --- write_rows -------------------------------------------------------------------------


def test_write_rows_round_trips(tmp_path: Path, fixture_rows: list[TrainingRow]) -> None:
    path = tmp_path / "rows.jsonl"
    assert write_rows(path, fixture_rows) == len(fixture_rows)
    assert list(read_rows(path)) == fixture_rows


def test_write_rows_is_private_and_replaces_the_file(
    tmp_path: Path, loose_umask: object, fixture_rows: list[TrainingRow]
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("old content\n")
    path.chmod(0o644)
    write_rows(path, fixture_rows[:2])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [r.id for r in read_rows(path)] == [r.id for r in fixture_rows[:2]]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rows.jsonl"]


def test_write_rows_writes_canonical_lines_without_unset_fields(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    path = tmp_path / "rows.jsonl"
    write_rows(path, fixture_rows)
    lines = path.read_text().splitlines()
    assert len(lines) == len(fixture_rows)
    for line in lines:
        data = json.loads(line)
        assert line == json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert None not in list(walk_values(data))
        assert data["schema_version"] == 1


def test_write_rows_accepts_an_empty_iterable(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    assert write_rows(path, iter(())) == 0
    assert path.read_text() == ""
    assert list(read_rows(path)) == []


def test_row_to_dict_keeps_our_metadata(fixture_rows: list[TrainingRow]) -> None:
    data = row_to_dict(by_id(fixture_rows, "row-weather-001"))
    assert set(data) == {"id", "messages", "tools", "source", "schema_version"}
    assert data["source"]["trace_id"] == "synthetic-trace-0001"


# --- to_provider_dict -------------------------------------------------------------------


def test_to_provider_dict_for_a_small_row() -> None:
    validated = validate_row(
        {
            "id": "row-1",
            "messages": [
                {"role": "user", "content": "Weather in Northvale?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Northvale"}',
                            },
                        }
                    ],
                    "weight": 1,
                },
                {"role": "tool", "tool_call_id": "c1", "content": "14"},
                {"role": "assistant", "content": "It's 14°C.", "reasoning_content": "Read it."},
            ],
            "source": {"trace_id": "synthetic-trace-1"},
        }
    )
    assert to_provider_dict(validated) == {
        "messages": [
            {"role": "user", "content": "Weather in Northvale?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Northvale"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "14"},
            {"role": "assistant", "content": "It's 14°C.", "reasoning_content": "Read it."},
        ]
    }


@pytest.mark.parametrize("row", list(read_rows(TOOL_USE)), ids=lambda r: r.id)
def test_to_provider_dict_has_the_openai_chat_shape(row: TrainingRow) -> None:
    data = to_provider_dict(row)
    assert set(data) <= {"messages", "tools"}
    assert ("tools" in data) == bool(row.tools)
    assert None not in list(walk_values(data))
    json.dumps(data, allow_nan=False)
    for message in data["messages"]:
        assert set(message) <= {
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "reasoning_content",
            "weight",
        }
        assert isinstance(message["content"], str)
        for tool_call in message.get("tool_calls", ()):
            assert set(tool_call) == {"id", "type", "function"}
            assert set(tool_call["function"]) == {"name", "arguments"}
            assert isinstance(json.loads(tool_call["function"]["arguments"]), dict)
    for tool in data.get("tools", ()):
        assert tool["type"] == "function"
        assert set(tool["function"]) == {"name", "description", "parameters"}


def test_to_provider_dict_keeps_weight_only_when_zero(fixture_rows: list[TrainingRow]) -> None:
    row = by_id(fixture_rows, "row-weather-003")
    weights = [m.get("weight") for m in to_provider_dict(row)["messages"]]
    assert weights.count(0) == 1
    assert set(weights) == {None, 0}
    one = validate_row(
        {"id": "r", "messages": [{"role": "user"}, {"role": "assistant", "weight": 1}]}
    )
    assert "weight" not in to_provider_dict(one)["messages"][1]


def test_to_provider_dict_omits_tools_when_none_are_declared(
    fixture_rows: list[TrainingRow],
) -> None:
    assert set(to_provider_dict(by_id(fixture_rows, "row-plain-001"))) == {"messages"}


# --- scan_rows --------------------------------------------------------------------------


def test_scan_rows_yields_every_line_with_its_row_or_error(tmp_path: Path) -> None:
    valid = TOOL_USE.read_text().splitlines()[0]
    invalid = (ROWS_DIR / "invalid.jsonl").read_text().splitlines()[0]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join([valid, "", "{oops", invalid]) + "\n")
    results = list(scan_rows(path))
    assert [line for line, _, _ in results] == [1, 3, 4]
    (_, data, row), (_, bad_data, bad_json), (_, invalid_data, invalid_row) = results
    assert isinstance(row, TrainingRow)
    assert data["id"] == row.id
    assert bad_data == {}
    assert isinstance(bad_json, RowError)
    assert (bad_json.code, bad_json.path, bad_json.line) == (RowErrorCode.INVALID_ROW, str(path), 3)
    assert isinstance(invalid_row, RowError)
    assert invalid_row.code == RowErrorCode.NO_ASSISTANT_MESSAGE
    assert (invalid_row.line, invalid_data["id"]) == (4, "bad-no-assistant")
