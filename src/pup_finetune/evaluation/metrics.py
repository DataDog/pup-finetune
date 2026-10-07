# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Deterministic metrics: how a generated assistant message compares with the recorded one.

Each metric is a pure function of the candidate (the generated OpenAI assistant message),
the reference (the recorded one) and the declared tools, and returns ``True``, ``False``, or
``None`` when it does not apply to the case (for example, argument checks when no tool was
called). :func:`aggregate` counts only the cases a metric applies to. The metrics read tool
calls only; ``reasoning_content`` is never compared, and neither is text content (a judge
reads that). Like replay itself, they measure agreement with the recorded behavior, not
whether the task succeeded.

``arguments`` of a tool call is a JSON object encoded as a string, as in OpenAI chat
format; an argument already decoded into an object is accepted as is. Arguments compare as
JSON values: key order, whitespace and ``1`` against ``1.0`` make no difference, while
``true`` and ``1`` differ.

:func:`schema_errors` checks a value against the common subset of JSON Schema that tool
declarations use: ``type`` (``object``, ``string``, ``number``, ``integer``, ``boolean``,
``array``, ``null``, or a list of them), ``required``, ``properties``,
``additionalProperties`` (``false``, or a schema for the extra properties), ``enum``,
``items`` (one schema for every item), ``minimum``, ``maximum``, ``minLength`` and
``maxLength``. Every other keyword (``$ref``, ``anyOf``, ``pattern``, ``format``, ...), a
type name outside that list and a keyword with a malformed value are ignored, so they never
fail a call.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pup_finetune.storage import canonical_json, loads_finite


class MetricName(StrEnum):
    """The deterministic metrics. The values are stable: they name results everywhere."""

    TOOL_DECISION_MATCH = "tool_decision_match"
    """The candidate calls tools exactly when the reference does."""
    TOOL_NAME_MATCH = "tool_name_match"
    """The same multiset of tool names. ``None`` when the reference calls no tool."""
    ARGUMENTS_JSON_VALID = "arguments_json_valid"
    """Every candidate call's arguments are a JSON object. ``None`` when it calls no tool."""
    ARGUMENTS_SCHEMA_VALID = "arguments_schema_valid"
    """Every candidate call's arguments fit the declared tool's parameters schema.

    ``None`` when the candidate calls no tool, no tools are declared, or no called tool has a
    schema; otherwise a call to an undeclared tool fails it.
    """
    REFERENCE_ARGUMENTS_MATCH = "reference_arguments_match"
    """Calls matched by tool name have the same arguments. ``None`` when the reference calls
    no tool."""
    PARALLEL_CALL_SET_MATCH = "parallel_call_set_match"
    """The same multiset of (name, arguments) calls, in any order. ``None`` when the reference
    calls no tool."""
    FORMAT_VALID = "format_valid"
    """The completion parsed cleanly into a message (reported by the sampler)."""


MetricResult = dict[str, bool | None]
"""One case's metric values, keyed by :class:`MetricName` value."""


# --- tool calls ---------------------------------------------------------------------------


def _parse_arguments(raw: object) -> dict[str, Any] | None:
    """``raw`` arguments as a JSON object, or ``None`` when they are not one."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        return None
    try:
        parsed = loads_finite(raw)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _calls(message: Mapping[str, Any]) -> list[tuple[str, object]]:
    """``(name, raw arguments)`` of each tool call in ``message``; a malformed call has name ``""``."""
    calls = message.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, str):
        return []
    result: list[tuple[str, object]] = []
    for call in calls:
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            result.append(("", None))
            continue
        name = function.get("name")
        result.append((name if isinstance(name, str) else "", function.get("arguments")))
    return result


def _normalize(value: Any) -> Any:
    """``value`` with integral floats as ints, so ``1.0`` and ``1`` serialize alike."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalize(inner) for inner in value]
    return value


def _normalized_json(value: Any) -> str:
    """``value`` as canonical JSON text, the same for every JSON-equal value."""
    return canonical_json(_normalize(value))


def _argument_key(raw: object) -> str | None:
    """Normalized arguments, or ``None`` when they are not a JSON object (equal to nothing)."""
    parsed = _parse_arguments(raw)
    return None if parsed is None else _normalized_json(parsed)


def _tool_schemas(tools: Iterable[Mapping[str, Any]]) -> dict[str, object]:
    schemas: dict[str, object] = {}
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(name := function.get("name"), str):
            schemas[name] = function.get("parameters")
    return schemas


# --- metrics ------------------------------------------------------------------------------


def tool_decision_match(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    """Both messages call tools, or neither does."""
    return bool(_calls(candidate)) == bool(_calls(reference))


def tool_name_match(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool | None:
    """The same tool names, as many times each. ``None`` when the reference calls no tool."""
    expected = _calls(reference)
    if not expected:
        return None
    return Counter(name for name, _ in _calls(candidate)) == Counter(name for name, _ in expected)


def arguments_json_valid(candidate: Mapping[str, Any]) -> bool | None:
    """Every call's arguments are a JSON object. ``None`` when the candidate calls no tool."""
    calls = _calls(candidate)
    if not calls:
        return None
    return all(_parse_arguments(raw) is not None for _, raw in calls)


def arguments_schema_valid(
    candidate: Mapping[str, Any], tools: Iterable[Mapping[str, Any]]
) -> bool | None:
    """Every call's arguments validate against its tool's ``parameters`` schema.

    A call to a tool that is not declared fails, and so do arguments that are not a JSON
    object for a tool that has a schema. A declared tool with no schema (missing or ``{}``
    parameters) constrains nothing, and its calls are skipped, malformed or not. ``None`` when
    the candidate calls no tool, or no tools are declared, or every call was skipped.
    """
    calls = _calls(candidate)
    schemas = _tool_schemas(tools)
    if not calls or not schemas:
        return None
    checked = False
    for name, raw in calls:
        if name not in schemas:
            return False
        schema = schemas[name]
        if schema is None or schema == {}:
            continue
        checked = True
        arguments = _parse_arguments(raw)
        if arguments is None or schema_errors(arguments, schema):
            return False
    return True if checked else None


def reference_arguments_match(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> bool | None:
    """Calls matched by tool name carry the same arguments, in any order.

    For each tool name both messages call, the calls are paired up as well as they can be
    (``min`` of the two counts); every pair must carry equal arguments. Missing or extra
    calls are :data:`MetricName.TOOL_NAME_MATCH`'s concern, but a candidate that shares no
    tool name with the reference fails. ``None`` when the reference calls no tool.
    """
    expected = _calls(reference)
    if not expected:
        return None
    by_name_expected: dict[str, Counter[str | None]] = {}
    for name, raw in expected:
        by_name_expected.setdefault(name, Counter())[_argument_key(raw)] += 1
    by_name_actual: dict[str, Counter[str | None]] = {}
    for name, raw in _calls(candidate):
        by_name_actual.setdefault(name, Counter())[_argument_key(raw)] += 1
    shared = by_name_expected.keys() & by_name_actual.keys()
    if not shared:
        return False
    for name in shared:
        want, got = by_name_expected[name], by_name_actual[name]
        # A call whose arguments are not an object (key None) equals no other call.
        want_valid = Counter({key: n for key, n in want.items() if key is not None})
        got_valid = Counter({key: n for key, n in got.items() if key is not None})
        paired = sum((want_valid & got_valid).values())
        if paired != min(want.total(), got.total()):
            return False
    return True


def parallel_call_set_match(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> bool | None:
    """The same calls, ``(name, arguments)``, as many times each and in any order.

    ``None`` when the reference calls no tool. A call whose arguments are not a JSON object
    equals no other call.
    """
    expected = _calls(reference)
    if not expected:
        return None
    actual = _calls(candidate)
    want = Counter((name, _argument_key(raw)) for name, raw in expected)
    got = Counter((name, _argument_key(raw)) for name, raw in actual)
    if any(key is None for _, key in want) or any(key is None for _, key in got):
        return False
    return want == got


def _for_metrics(message: Mapping[str, Any]) -> dict[str, Any]:
    """``message`` without ``reasoning_content``, which no metric compares."""
    return {key: value for key, value in message.items() if key != "reasoning_content"}


def score_case(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    tools: Iterable[Mapping[str, Any]],
    *,
    format_valid: bool,
) -> MetricResult:
    """Every metric for one case, keyed by :class:`MetricName` value, in its order.

    ``format_valid`` comes from the sampler's
    :class:`~pup_finetune.providers.base.Generation`.
    """
    candidate = _for_metrics(candidate)
    reference = _for_metrics(reference)
    tools = list(tools)
    return {
        MetricName.TOOL_DECISION_MATCH.value: tool_decision_match(candidate, reference),
        MetricName.TOOL_NAME_MATCH.value: tool_name_match(candidate, reference),
        MetricName.ARGUMENTS_JSON_VALID.value: arguments_json_valid(candidate),
        MetricName.ARGUMENTS_SCHEMA_VALID.value: arguments_schema_valid(candidate, tools),
        MetricName.REFERENCE_ARGUMENTS_MATCH.value: reference_arguments_match(candidate, reference),
        MetricName.PARALLEL_CALL_SET_MATCH.value: parallel_call_set_match(candidate, reference),
        MetricName.FORMAT_VALID.value: bool(format_valid),
    }


def aggregate(results: Iterable[Mapping[str, bool | None]]) -> dict[str, dict[str, Any]]:
    """Pass rates per metric: ``{metric: {"rate", "passed", "n"}}``.

    ``n`` counts the cases a metric applies to (``None`` values are skipped) and ``rate`` is
    ``passed / n``, or ``None`` when ``n`` is ``0``, so the result is always valid JSON.
    Every :class:`MetricName` is present, in its order, followed by any other metric seen,
    sorted. A value that is not ``True``, ``False`` or ``None`` raises ``TypeError``.
    """
    passed: dict[str, int] = {name.value: 0 for name in MetricName}
    counted: dict[str, int] = dict(passed)
    for result in results:
        for metric, value in result.items():
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{metric}: expected true, false or null, got {value!r}")
            passed.setdefault(metric, 0)
            counted.setdefault(metric, 0)
            if value is not None:
                counted[metric] += 1
                passed[metric] += value
    known = [name.value for name in MetricName]
    order = known + sorted(metric for metric in counted if metric not in known)
    return {
        metric: {
            "rate": passed[metric] / counted[metric] if counted[metric] else None,
            "passed": passed[metric],
            "n": counted[metric],
        }
        for metric in order
    }


# --- JSON Schema subset -------------------------------------------------------------------


def _is_integer(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


_TYPE_CHECKS: dict[str, Callable[[object], bool]] = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": _is_integer,
    "number": _is_number,
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def _type_ok(value: object, declared: object) -> bool:
    names = declared if isinstance(declared, list) else [declared]
    known = [name for name in names if isinstance(name, str) and name in _TYPE_CHECKS]
    if len(known) != len(names) or not known:
        # An unknown or malformed type name is ignored, so it never fails a value.
        return True
    return any(_TYPE_CHECKS[name](value) for name in known)


def _bound(schema: Mapping[str, Any], keyword: str) -> int | float | None:
    """The numeric ``keyword`` bound as written, or ``None``.

    An integer bound stays an integer: Python compares ``int`` and ``float`` exactly, while
    ``float()`` would round a bound above ``2**53`` (and overflow past the float range).
    """
    value = schema.get(keyword)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) or (isinstance(value, float) and math.isfinite(value)):
        return value
    return None


def _length(schema: Mapping[str, Any], keyword: str) -> int | None:
    value = schema.get(keyword)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def schema_errors(value: Any, schema: object, path: str = "$") -> list[str]:
    """Where ``value`` breaks ``schema`` (the subset in the module docstring); empty if nowhere.

    A boolean schema is accepted: ``true`` allows anything and ``false`` nothing. A schema
    that is neither a boolean nor an object constrains nothing.
    """
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: no value is allowed"]
    if not isinstance(schema, Mapping):
        return []
    errors: list[str] = []
    if "type" in schema and not _type_ok(value, schema["type"]):
        errors.append(f"{path}: expected type {schema['type']!r}")
        return errors
    enum = schema.get("enum")
    if isinstance(enum, list) and _normalized_json(value) not in {
        _normalized_json(option) for option in enum
    }:
        errors.append(f"{path}: not one of the allowed values")
    if _is_number(value):
        if (low := _bound(schema, "minimum")) is not None and value < low:
            errors.append(f"{path}: less than the minimum {schema['minimum']}")
        if (high := _bound(schema, "maximum")) is not None and value > high:
            errors.append(f"{path}: more than the maximum {schema['maximum']}")
    if isinstance(value, str):
        if (shortest := _length(schema, "minLength")) is not None and len(value) < shortest:
            errors.append(f"{path}: shorter than {shortest} characters")
        if (longest := _length(schema, "maxLength")) is not None and len(value) > longest:
            errors.append(f"{path}: longer than {longest} characters")
    if isinstance(value, dict):
        errors.extend(_object_errors(value, schema, path))
    if isinstance(value, list) and isinstance(items := schema.get("items"), Mapping | bool):
        for index, item in enumerate(value):
            errors.extend(schema_errors(item, items, f"{path}[{index}]"))
    return errors


def _object_errors(value: dict[str, Any], schema: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    required = schema.get("required")
    if isinstance(required, list):
        errors.extend(
            f"{path}: missing required property {name!r}"
            for name in required
            if isinstance(name, str) and name not in value
        )
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    additional = schema.get("additionalProperties", True)
    for key, inner in value.items():
        where = f"{path}.{key}"
        if key in properties:
            errors.extend(schema_errors(inner, properties[key], where))
        elif additional is False:
            errors.append(f"{where}: property is not allowed")
        elif isinstance(additional, Mapping):
            errors.extend(schema_errors(inner, additional, where))
    return errors
