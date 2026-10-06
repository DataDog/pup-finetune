# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

from typing import Any

import pytest

from pup_finetune.providers.base import TrainSpec
from pup_finetune.training import Budget


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_without_caps_the_budget_never_runs_out() -> None:
    budget = Budget()
    budget.record(10**12)
    assert budget.exhausted(10**12) is None
    assert budget.spend_usd is None


def test_the_token_cap_stops_before_a_step_that_would_exceed_it() -> None:
    budget = Budget(max_train_tokens=100)
    assert budget.exhausted(100) is None  # exactly the cap fits
    budget.record(60)
    assert budget.exhausted(40) is None
    reason = budget.exhausted(41)
    assert reason is not None
    assert "101 tokens, over max_train_tokens 100" in reason


def test_the_spend_guard_prices_elapsed_time_at_the_hourly_rate() -> None:
    clock = Clock()
    budget = Budget(max_spend_usd=2.0, hourly_rate_usd=4.0, clock=clock)
    budget.start()
    clock.now += 1799  # 0.4997 h at 4 USD/h is just under 2 USD
    assert budget.exhausted(1) is None
    assert budget.spend_usd == pytest.approx(1799 / 3600 * 4)
    clock.now += 1
    reason = budget.exhausted(1)
    assert reason is not None
    assert "reached max_spend_usd 2.0" in reason


def test_the_clock_starts_once() -> None:
    clock = Clock()
    budget = Budget(clock=clock)
    assert budget.elapsed_seconds == 0.0
    budget.start()
    clock.now += 5
    budget.start()
    clock.now += 5
    assert budget.elapsed_seconds == 10


def test_the_first_check_starts_the_clock() -> None:
    clock = Clock()
    budget = Budget(max_spend_usd=1.0, hourly_rate_usd=1.0, clock=clock)
    assert budget.exhausted(1) is None
    clock.now += 7
    assert budget.elapsed_seconds == 7


def test_from_spec_uses_the_specs_caps() -> None:
    spec = TrainSpec(max_train_tokens=500, max_spend_usd=3.0, hourly_rate_usd=1.5)
    budget = Budget.from_spec(spec, clock=Clock())
    assert (budget.max_train_tokens, budget.max_spend_usd, budget.hourly_rate_usd) == (
        500,
        3.0,
        1.5,
    )


def test_to_json_reports_caps_and_use() -> None:
    clock = Clock()
    budget = Budget(max_train_tokens=50, max_spend_usd=1.0, hourly_rate_usd=36.0, clock=clock)
    budget.start()
    budget.record(20)
    clock.now += 50
    assert budget.to_json() == {
        "max_train_tokens": 50,
        "tokens_used": 20,
        "max_spend_usd": 1.0,
        "hourly_rate_usd": 36.0,
        "elapsed_seconds": 50.0,
        "spend_usd": pytest.approx(0.5),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_spend_usd": 1.0}, "set together"),
        ({"hourly_rate_usd": 1.0}, "set together"),
        ({"max_spend_usd": 0.0, "hourly_rate_usd": 1.0}, "must be positive"),
        ({"max_spend_usd": 1.0, "hourly_rate_usd": -1.0}, "must be positive"),
        ({"max_train_tokens": 0}, "must be positive"),
    ],
)
def test_invalid_caps_are_refused(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Budget(**kwargs)
