# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The optional spend guard: a wall-clock limit derived from a spend cap and an hourly rate.

It measures elapsed time, not billed usage, and cannot interrupt a call already in progress;
the reserve fraction leaves room for those calls and for cleanup.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class BudgetExceeded(Exception):
    """The spend guard's active-time limit was reached."""


class BudgetGuard:
    """Stops training once ``max_spend_usd * (1 - reserve_fraction)`` worth of time has passed."""

    def __init__(
        self,
        *,
        max_spend_usd: float | None,
        hourly_rate_usd: float | None,
        reserve_fraction: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (max_spend_usd is None) != (hourly_rate_usd is None):
            raise ValueError("max_spend_usd and hourly_rate_usd must be set together")
        self._limit: float | None = None
        if max_spend_usd is not None and hourly_rate_usd is not None:
            self._limit = max_spend_usd * (1 - reserve_fraction) / hourly_rate_usd * 3600
        self._clock = clock
        self._started_at: float | None = None

    @property
    def enabled(self) -> bool:
        return self._limit is not None

    @property
    def limit_seconds(self) -> float | None:
        return self._limit

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since :meth:`start`; ``0.0`` before it."""
        if self._started_at is None:
            return 0.0
        return self._clock() - self._started_at

    def start(self) -> None:
        """Start the clock (once), also when disabled, so elapsed time is always reported."""
        if self._started_at is None:
            self._started_at = self._clock()

    def check(self, operation: str) -> None:
        """Raise :class:`BudgetExceeded` when the limit is reached before ``operation``."""
        if self._limit is None:
            return
        if self._started_at is None:
            raise RuntimeError("the budget guard was checked before it started")
        if self.elapsed_seconds >= self._limit:
            raise BudgetExceeded(
                f"the spend guard's {self._limit:.0f} s active-time limit was reached "
                f"before {operation}"
            )
