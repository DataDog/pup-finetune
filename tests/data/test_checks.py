# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Span-level checks and the exclusion explanations."""

from __future__ import annotations

from typing import Any

import pytest

from pup_finetune.data.checks import CheckOptions, check_span, run_checks, span_error
from pup_finetune.data.exclusions import EXPLANATIONS, ExclusionCode, explain
from pup_finetune.data.schema import RowErrorCode
from pup_finetune.datadog.spans import LLMObsSpan, SpanQuery, iter_spans, parse_span
from tests.fakes.llmobs import FakeExportApi, fixture_pages, span_item
from tests.fakes.pup import FakePup

SYSTEM = {"role": "system", "content": "You are a synthetic helper."}
Q1 = {"role": "user", "content": "First question?"}
A1 = {"role": "assistant", "content": "First answer."}


def fixture_spans() -> list[LLMObsSpan]:
    api = FakeExportApi(fixture_pages())
    return list(iter_spans(api.install(FakePup()).client(), SpanQuery()))


def make_span(span_id: str, messages: list[dict[str, Any]], **attributes: Any) -> LLMObsSpan:
    span = parse_span(
        span_item(
            span_id,
            input={"messages": messages[:-1]},
            output={"messages": messages[-1:]},
            **attributes,
        )
    )
    assert span is not None
    return span


# --- the checks -----------------------------------------------------------------------------


def test_an_errored_span_fails_span_error() -> None:
    errored = next(span for span in fixture_spans() if span.status == "error")
    exclusion = check_span(errored)
    assert exclusion is not None
    assert exclusion.code is ExclusionCode.SPAN_ERROR
    assert exclusion.code.value == "span_error"
    assert exclusion.row_id == f"{errored.trace_id}:{errored.span_id}"
    assert (exclusion.trace_id, exclusion.span_id) == (errored.trace_id, errored.span_id)
    assert "error" in exclusion.detail


@pytest.mark.parametrize("status", ["ok", None, "Error-ish"])
def test_other_statuses_pass(status: str | None) -> None:
    span = make_span("synthetic-span-1", [SYSTEM, Q1, A1], status=status)
    assert span_error(span) is None
    assert check_span(span) is None


def test_include_errors_skips_span_error() -> None:
    errored = make_span("synthetic-span-1", [SYSTEM, Q1, A1], status="error")
    assert check_span(errored, CheckOptions(include_errors=True)) is None
    assert CheckOptions().checks() == (span_error,)
    assert CheckOptions(include_errors=True).checks() == ()


def test_the_first_failing_check_wins() -> None:
    span = make_span("synthetic-span-1", [SYSTEM, Q1, A1])
    calls: list[str] = []

    def passing(_: LLMObsSpan) -> None:
        calls.append("passing")

    def first(_: LLMObsSpan) -> tuple[ExclusionCode, str]:
        calls.append("first")
        return ExclusionCode.SPAN_ERROR, "first"

    def second(_: LLMObsSpan) -> tuple[ExclusionCode, str]:
        calls.append("second")
        return ExclusionCode.NON_TEXT_CONTENT, "second"

    exclusion = run_checks(span, (passing, first, second))
    assert exclusion is not None
    assert (exclusion.code, exclusion.detail) == (ExclusionCode.SPAN_ERROR, "first")
    assert calls == ["passing", "first"]
    assert run_checks(span, ()) is None


# --- explanations ---------------------------------------------------------------------------


def test_every_exclusion_code_is_explained() -> None:
    for code in [*RowErrorCode, *ExclusionCode]:
        text = explain(code.value)
        assert text, code
        assert text == EXPLANATIONS[code]
    assert explain("not_a_code") is None
    assert "tool calls as text" in (explain("unstructured_tool_message") or "")
    assert "--include-errors" in (explain("span_error") or "")
