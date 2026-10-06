# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import pytest

from pup_finetune.training import BudgetExceeded, BudgetGuard


def test_disabled_guard_never_stops_but_measures_elapsed_time() -> None:
    times = iter([10.0, 1e9])
    guard = BudgetGuard(
        max_spend_usd=None, hourly_rate_usd=None, reserve_fraction=0.1, clock=lambda: next(times)
    )
    guard.start()
    guard.check("training")
    assert not guard.enabled
    assert guard.limit_seconds is None
    assert guard.elapsed_seconds == 1e9 - 10.0


@pytest.mark.parametrize(("reserve", "expected"), [(0.1, 8_100.0), (0.2, 7_200.0), (0.0, 9_000.0)])
def test_limit_keeps_the_reserve_for_cleanup(reserve: float, expected: float) -> None:
    guard = BudgetGuard(max_spend_usd=75.0, hourly_rate_usd=30.0, reserve_fraction=reserve)
    assert guard.enabled
    assert guard.limit_seconds == pytest.approx(expected)


def test_check_raises_once_the_limit_is_reached() -> None:
    times = iter([100.0, 8_199.9, 8_200.0])
    guard = BudgetGuard(
        max_spend_usd=75.0, hourly_rate_usd=30.0, reserve_fraction=0.1, clock=lambda: next(times)
    )
    guard.start()
    guard.check("an optimizer step")
    with pytest.raises(BudgetExceeded, match=r"8100 s active-time limit .* before an optimizer"):
        guard.check("an optimizer step")


def test_elapsed_seconds_is_zero_before_start_and_start_runs_once() -> None:
    times = iter([10.0, 12.5, 99.0])
    guard = BudgetGuard(
        max_spend_usd=1.0, hourly_rate_usd=1.0, reserve_fraction=0.0, clock=lambda: next(times)
    )
    assert guard.elapsed_seconds == 0.0
    guard.start()
    guard.start()
    assert guard.elapsed_seconds == 2.5


def test_check_before_start_is_a_programming_error() -> None:
    guard = BudgetGuard(max_spend_usd=1.0, hourly_rate_usd=1.0, reserve_fraction=0.0)
    with pytest.raises(RuntimeError, match="before it started"):
        guard.check("training")


@pytest.mark.parametrize(("spend", "rate"), [(10.0, None), (None, 4.0)])
def test_spend_and_rate_must_be_set_together(spend: float | None, rate: float | None) -> None:
    with pytest.raises(ValueError, match="set together"):
        BudgetGuard(max_spend_usd=spend, hourly_rate_usd=rate, reserve_fraction=0.1)
