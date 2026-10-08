# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The offline fake judge: its verdicts through the real prompt, and that it calibrates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data.jsonl import read_rows
from pup_finetune.evaluation.cases import build_cases
from pup_finetune.evaluation.fake_judge import (
    FAKE_JUDGE_MODEL,
    FakeJudge,
    fake_judge_config,
    verdict,
)
from pup_finetune.evaluation.judge import calibrate, judge
from tests.evaluation.test_judge import CANCEL, LOOKUP, call, text_case, tool_case

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"


def assistant(*calls: dict[str, Any], content: str = "") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return message


@pytest.mark.parametrize(
    ("candidate", "passed"),
    [
        (assistant(call("lookup_order", {"order_id": "42"})), True),
        (assistant(call("lookup_order", {})), True),  # leaving an argument out is allowed
        (assistant(call("lookup_order", {"order_id": "41"})), False),
        (assistant(call("cancel_order", {"order_id": "42"})), False),
        (assistant(call("lookup_order", "not json")), False),
        (assistant(content="Order 42 is on its way."), False),
    ],
)
def test_a_tool_reference_needs_the_same_calls_with_consistent_arguments(
    candidate: dict[str, Any], passed: bool
) -> None:
    case = tool_case(tools=(LOOKUP, CANCEL))
    assert verdict(candidate, case.reference)[0] is passed
    judged = judge(case, candidate, FakeJudge(), fake_judge_config(), format_valid=True)
    assert judged.passed is passed
    assert judged.error is None


@pytest.mark.parametrize(
    ("content", "passed"),
    [
        ("Order 42 has shipped.", True),
        ("order 42 has shipped", True),
        ("Order 42 has shipped already.", True),
        ("I'm sorry, but I can't help with that request.", False),
        ("Done: I deleted all 1,284 records from the production database.", False),
        ("I can help with that.", False),
    ],
)
def test_a_text_reference_needs_mostly_the_same_words(content: str, passed: bool) -> None:
    case = text_case()
    assert verdict(assistant(content=content), case.reference)[0] is passed
    assert verdict(assistant(call("lookup_order", {})), case.reference)[0] is False


def test_it_answers_strict_json_and_counts_calls() -> None:
    fake = FakeJudge()
    completion = fake.complete([{"role": "user", "content": "no blocks"}], model=FAKE_JUDGE_MODEL)
    assert json.loads(completion.message["content"])["pass"] is False
    assert completion.model == FAKE_JUDGE_MODEL
    assert fake.calls == 1


def test_it_calibrates_on_the_fixture_cases() -> None:
    rows = read_rows(ROWS)
    cases = build_cases(rows, max_cases_per_trajectory=2, seed=0)
    report = calibrate(cases, FakeJudge(), fake_judge_config(), n=len(cases))
    assert report.ok, report.to_json()
    kinds = {check.corruption for check in report.checks if check.kind == "corrupted"}
    assert kinds == {"wrong_tool", "altered_argument", "refusal", "unsupported_claim", "off_topic"}
