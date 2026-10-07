# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
from typing import Any

import pytest

from pup_finetune.evaluation.metrics import (
    MetricName,
    aggregate,
    arguments_json_valid,
    arguments_schema_valid,
    parallel_call_set_match,
    reference_arguments_match,
    schema_errors,
    score_case,
    tool_decision_match,
    tool_name_match,
)
from pup_finetune.storage import canonical_json

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "required": ["query"],
    "additionalProperties": False,
}
TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "search", "parameters": SEARCH_SCHEMA}},
    {"type": "function", "function": {"name": "now", "parameters": {}}},
]


def call(name: str, arguments: Any) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": "x", "type": "function", "function": {"name": name, "arguments": raw}}


def message(*calls: dict[str, Any], content: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        result["tool_calls"] = list(calls)
    return result


TEXT = message(content="Here you go.")
ONE = message(call("search", {"query": "rain", "limit": 5}))
PARALLEL = message(call("search", {"query": "a"}), call("search", {"query": "b"}))


# --- tool_decision_match ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "reference", "expected"),
    [
        (ONE, ONE, True),
        (TEXT, TEXT, True),
        (message(), TEXT, True),
        (TEXT, ONE, False),
        (ONE, TEXT, False),
        (message(call("now", {})), ONE, True),
    ],
)
def test_tool_decision_match(
    candidate: dict[str, Any], reference: dict[str, Any], expected: bool
) -> None:
    assert tool_decision_match(candidate, reference) is expected


# --- tool_name_match ----------------------------------------------------------------------


def test_tool_name_match() -> None:
    assert tool_name_match(ONE, ONE) is True
    assert tool_name_match(message(call("search", {"query": "other"})), ONE) is True
    assert tool_name_match(message(call("now", {})), ONE) is False
    assert tool_name_match(TEXT, ONE) is False
    assert tool_name_match(ONE, TEXT) is None
    assert tool_name_match(TEXT, TEXT) is None


def test_tool_name_match_counts_parallel_calls() -> None:
    assert tool_name_match(message(call("search", {"query": "b"})), PARALLEL) is False
    both = message(call("search", {"query": "z"}), call("search", {"query": "y"}))
    assert tool_name_match(both, PARALLEL) is True
    extra = message(*PARALLEL["tool_calls"], call("now", {}))
    assert tool_name_match(extra, PARALLEL) is False


# --- arguments_json_valid -----------------------------------------------------------------


def test_arguments_json_valid() -> None:
    assert arguments_json_valid(ONE) is True
    assert arguments_json_valid(TEXT) is None
    assert arguments_json_valid(message(call("search", "{not json"))) is False
    assert arguments_json_valid(message(call("search", "[1, 2]"))) is False
    assert arguments_json_valid(message(call("search", '{"x": NaN}'))) is False
    assert arguments_json_valid(message(call("search", {"query": "a"}), call("now", "1"))) is False
    decoded = message({"id": "x", "type": "function", "function": {"name": "now", "arguments": {}}})
    assert arguments_json_valid(decoded) is True


def test_malformed_tool_calls_do_not_raise() -> None:
    broken = {"role": "assistant", "content": "", "tool_calls": [{"function": "nope"}, 3]}
    assert arguments_json_valid(broken) is False
    assert arguments_schema_valid(broken, TOOLS) is False
    assert tool_name_match(broken, ONE) is False
    assert tool_decision_match({"role": "assistant", "tool_calls": "x"}, TEXT) is True


# --- arguments_schema_valid ---------------------------------------------------------------


def test_arguments_schema_valid() -> None:
    assert arguments_schema_valid(ONE, TOOLS) is True
    assert arguments_schema_valid(TEXT, TOOLS) is None
    assert arguments_schema_valid(message(call("search", {"limit": 5})), TOOLS) is False
    assert arguments_schema_valid(message(call("search", {"query": "a", "x": 1})), TOOLS) is False
    assert (
        arguments_schema_valid(message(call("search", {"query": "a", "limit": 99})), TOOLS) is False
    )
    assert arguments_schema_valid(message(call("search", "{bad")), TOOLS) is False


def test_an_undeclared_tool_fails_schema_validation() -> None:
    assert arguments_schema_valid(message(call("delete_all", {})), TOOLS) is False
    assert (
        arguments_schema_valid(message(call("search", {"query": "a"}), call("other", {})), TOOLS)
        is False
    )


def test_schema_validation_does_not_apply_without_a_schema() -> None:
    assert arguments_schema_valid(ONE, []) is None
    assert arguments_schema_valid(message(call("now", {"anything": 1})), TOOLS) is None
    no_parameters = [{"type": "function", "function": {"name": "now"}}]
    assert arguments_schema_valid(message(call("now", {})), no_parameters) is None
    # A call with a schema still counts when another call has none.
    mixed = message(call("now", {}), call("search", {"query": "a"}))
    assert arguments_schema_valid(mixed, TOOLS) is True


# --- reference_arguments_match ------------------------------------------------------------


def test_reference_arguments_match() -> None:
    assert reference_arguments_match(ONE, ONE) is True
    reordered = message(call("search", '{ "limit": 5.0, "query": "rain" }'))
    assert reference_arguments_match(reordered, ONE) is True
    assert (
        reference_arguments_match(message(call("search", {"query": "snow", "limit": 5})), ONE)
        is False
    )
    assert reference_arguments_match(TEXT, ONE) is False
    assert reference_arguments_match(message(call("now", {})), ONE) is False
    assert reference_arguments_match(ONE, TEXT) is None
    assert reference_arguments_match(message(call("search", "{bad")), ONE) is False


def test_reference_arguments_match_is_order_insensitive_for_parallel_calls() -> None:
    swapped = message(call("search", {"query": "b"}), call("search", {"query": "a"}))
    assert reference_arguments_match(swapped, PARALLEL) is True
    one_of_two = message(call("search", {"query": "b"}))
    assert reference_arguments_match(one_of_two, PARALLEL) is True
    wrong = message(call("search", {"query": "b"}), call("search", {"query": "c"}))
    assert reference_arguments_match(wrong, PARALLEL) is False
    with_extra_tool = message(*PARALLEL["tool_calls"], call("now", {}))
    assert reference_arguments_match(with_extra_tool, PARALLEL) is True
    one_malformed = message(call("search", {"query": "a"}), call("search", "{bad"))
    assert reference_arguments_match(one_malformed, PARALLEL) is False


def test_malformed_arguments_equal_nothing_even_themselves() -> None:
    bad = message(call("search", "{bad"))
    assert reference_arguments_match(bad, bad) is False
    assert parallel_call_set_match(bad, bad) is False


def test_booleans_and_numbers_are_different_arguments() -> None:
    reference = message(call("now", {"flag": True}))
    assert reference_arguments_match(message(call("now", {"flag": 1})), reference) is False
    assert reference_arguments_match(message(call("now", {"flag": True})), reference) is True


# --- parallel_call_set_match --------------------------------------------------------------


def test_parallel_call_set_match() -> None:
    swapped = message(call("search", {"query": "b"}), call("search", {"query": "a"}))
    assert parallel_call_set_match(swapped, PARALLEL) is True
    assert parallel_call_set_match(message(call("search", {"query": "a"})), PARALLEL) is False
    doubled = message(*PARALLEL["tool_calls"], call("search", {"query": "a"}))
    assert parallel_call_set_match(doubled, PARALLEL) is False
    assert parallel_call_set_match(ONE, ONE) is True
    assert parallel_call_set_match(TEXT, ONE) is False
    assert parallel_call_set_match(ONE, TEXT) is None
    assert parallel_call_set_match(message(call("search", "{bad")), ONE) is False


# --- score_case ---------------------------------------------------------------------------


def test_score_case_reports_every_metric_in_order() -> None:
    result = score_case(ONE, ONE, TOOLS, format_valid=True)
    assert list(result) == [name.value for name in MetricName]
    assert all(value is True for value in result.values())


def test_score_case_on_text_only_turns() -> None:
    result = score_case(TEXT, TEXT, TOOLS, format_valid=False)
    assert result == {
        "tool_decision_match": True,
        "tool_name_match": None,
        "arguments_json_valid": None,
        "arguments_schema_valid": None,
        "reference_arguments_match": None,
        "parallel_call_set_match": None,
        "format_valid": False,
    }


def test_score_case_when_the_candidate_answers_instead_of_calling() -> None:
    result = score_case(TEXT, ONE, TOOLS, format_valid=True)
    assert result["tool_decision_match"] is False
    assert result["tool_name_match"] is False
    assert result["arguments_json_valid"] is None
    assert result["reference_arguments_match"] is False
    assert result["parallel_call_set_match"] is False


def test_reasoning_is_never_compared() -> None:
    candidate = {**ONE, "reasoning_content": "I think so"}
    reference = {**ONE, "reasoning_content": "something else entirely"}
    assert score_case(candidate, reference, TOOLS, format_valid=True) == score_case(
        ONE, ONE, TOOLS, format_valid=True
    )


# --- schema subset ------------------------------------------------------------------------


NESTED: dict[str, Any] = {
    "type": "object",
    "properties": {
        "filter": {
            "type": "object",
            "properties": {
                "level": {"enum": ["info", "warn", "error"]},
                "tags": {"type": "array", "items": {"type": "string", "maxLength": 3}},
            },
            "required": ["level"],
            "additionalProperties": False,
        },
        "cursor": {"type": ["string", "null"]},
        "ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "extra": {"type": "boolean"},
    },
    "additionalProperties": {"type": "integer"},
}


@pytest.mark.parametrize(
    "value",
    [
        {"filter": {"level": "warn"}},
        {"filter": {"level": "info", "tags": ["a", "abc"]}, "cursor": None},
        {"cursor": "next", "ratio": 0.5, "extra": False},
        {"ratio": 1, "count": 3},
        {"count": 3.0},
    ],
)
def test_nested_schemas_accept_valid_values(value: Any) -> None:
    assert schema_errors(value, NESTED) == []


@pytest.mark.parametrize(
    ("value", "where"),
    [
        ({"filter": {}}, "$.filter: missing required property 'level'"),
        ({"filter": {"level": "debug"}}, "$.filter.level: not one of"),
        ({"filter": {"level": "info", "other": 1}}, "$.filter.other: property is not allowed"),
        ({"filter": {"level": "info", "tags": ["abcd"]}}, "$.filter.tags[0]: longer than 3"),
        ({"filter": {"level": "info", "tags": [1]}}, "$.filter.tags[0]: expected type"),
        ({"cursor": 3}, "$.cursor: expected type"),
        ({"ratio": 1.5}, "$.ratio: more than the maximum"),
        ({"ratio": -1}, "$.ratio: less than the minimum"),
        ({"ratio": True}, "$.ratio: expected type"),
        ({"extra": 0}, "$.extra: expected type"),
        ({"count": 1.5}, "$.count: expected type"),
        ([], "$: expected type"),
    ],
)
def test_nested_schemas_report_where_a_value_fails(value: Any, where: str) -> None:
    errors = schema_errors(value, NESTED)
    assert any(error.startswith(where) for error in errors), errors


def test_schema_types() -> None:
    assert schema_errors(None, {"type": "null"}) == []
    assert schema_errors(1, {"type": "integer"}) == []
    assert schema_errors(True, {"type": "integer"}) != []
    assert schema_errors(True, {"type": "number"}) != []
    assert schema_errors(2.5, {"type": "number"}) == []
    assert schema_errors([1], {"type": "array"}) == []
    assert schema_errors("1", {"type": ["integer", "boolean"]}) != []
    assert schema_errors(False, {"type": ["integer", "boolean"]}) == []


def test_string_lengths_and_enums() -> None:
    assert schema_errors("", {"type": "string", "minLength": 1}) != []
    assert schema_errors("é", {"type": "string", "maxLength": 1}) == []
    assert schema_errors(1.0, {"enum": [1, 2]}) == []
    assert schema_errors(True, {"enum": [1, 2]}) != []
    assert schema_errors({"a": [1]}, {"enum": [{"a": [1]}]}) == []


@pytest.mark.parametrize(
    ("value", "schema", "ok"),
    [
        # Integer bounds above 2**53 compare exactly, not rounded to a float.
        (2**53 + 1, {"maximum": 2**53 + 1}, True),
        (2**53 + 2, {"maximum": 2**53 + 1}, False),
        (2**53 + 1, {"minimum": 2**53 + 1}, True),
        (2**53, {"minimum": 2**53 + 1}, False),
        (2**63, {"maximum": 2**63 - 1}, False),
        (2**63 - 1, {"maximum": 2**63 - 1}, True),
        (10**400, {"maximum": 10**400}, True),
        (10**400 + 1, {"maximum": 10**400}, False),
        (1.5, {"maximum": 10**400}, True),
        # Float bounds are unchanged, against integers and floats alike.
        (2, {"maximum": 1.5}, False),
        (1, {"maximum": 1.5}, True),
        (0.5, {"minimum": 0.25, "maximum": 0.75}, True),
        (0.8, {"minimum": 0.25, "maximum": 0.75}, False),
        # A boolean or non-finite bound is no bound.
        (5, {"maximum": True}, True),
        (5, {"minimum": False, "maximum": float("nan")}, True),
        (5, {"maximum": float("-inf")}, True),
    ],
)
def test_numeric_bounds_compare_exactly(value: Any, schema: dict[str, Any], ok: bool) -> None:
    assert (schema_errors(value, schema) == []) is ok


def test_unknown_keywords_and_types_are_ignored() -> None:
    schema = {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}]}}}
    assert schema_errors({"x": 3}, schema) == []
    assert schema_errors(3, {"type": "decimal"}) == []
    assert schema_errors(3, {"type": ["decimal", "string"]}) == []
    assert schema_errors("x", {"pattern": "^y$", "format": "email"}) == []
    assert schema_errors(3, {"minimum": "high"}) == []
    assert schema_errors({}, {"required": "x"}) == []


def test_boolean_schemas() -> None:
    assert schema_errors(1, True) == []
    assert schema_errors(1, False) != []
    assert schema_errors([1, 2], {"type": "array", "items": False}) != []


# --- aggregate ----------------------------------------------------------------------------


def test_aggregate_skips_metrics_that_do_not_apply() -> None:
    results = [
        score_case(ONE, ONE, TOOLS, format_valid=True),
        score_case(TEXT, ONE, TOOLS, format_valid=True),
        score_case(TEXT, TEXT, TOOLS, format_valid=False),
    ]
    summary = aggregate(results)
    assert list(summary) == [name.value for name in MetricName]
    assert summary["tool_decision_match"] == {"rate": 2 / 3, "passed": 2, "n": 3}
    assert summary["tool_name_match"] == {"rate": 0.5, "passed": 1, "n": 2}
    assert summary["arguments_json_valid"] == {"rate": 1.0, "passed": 1, "n": 1}
    assert summary["format_valid"] == {"rate": 2 / 3, "passed": 2, "n": 3}


def test_aggregate_of_nothing_is_json_safe() -> None:
    summary = aggregate([])
    assert all(entry == {"rate": None, "passed": 0, "n": 0} for entry in summary.values())
    assert json.loads(canonical_json(summary)) == summary


def test_aggregate_keeps_other_metrics_after_the_known_ones() -> None:
    summary = aggregate([{"judge_pass": True, "zeta": None}, {"judge_pass": False}])
    assert list(summary)[-2:] == ["judge_pass", "zeta"]
    assert summary["judge_pass"] == {"rate": 0.5, "passed": 1, "n": 2}
    assert summary["zeta"] == {"rate": None, "passed": 0, "n": 0}


def test_aggregate_rejects_non_boolean_values() -> None:
    with pytest.raises(TypeError, match="expected true, false or null"):
        aggregate([{"tool_name_match": 1}])  # type: ignore[dict-item]


def test_results_are_strict_json() -> None:
    results = [
        score_case(PARALLEL, ONE, TOOLS, format_valid=True),
        score_case(message(call("search", "{bad")), PARALLEL, TOOLS, format_valid=False),
    ]
    for data in (*results, aggregate(results)):
        text = canonical_json(data)  # allow_nan=False: raises on NaN or Infinity
        assert json.loads(text) == data
