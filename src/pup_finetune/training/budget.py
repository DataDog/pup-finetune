# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The training budget: a cap on trained tokens and an optional wall-clock spend guard.

The loop asks :meth:`Budget.exhausted` before each optimizer step and stops cleanly when it
answers. The spend guard multiplies elapsed hours by an hourly rate the user supplies: it
measures time, not the provider's bill, and cannot interrupt a call already in progress.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pup_finetune.providers.base import TrainSpec


class Budget:
    """Tracks trained tokens and elapsed time against the caps; every cap is optional.

    ``max_spend_usd`` and ``hourly_rate_usd`` go together. ``clock`` returns seconds and is
    injectable for tests; the clock starts at :meth:`start` (or the first check).
    """

    def __init__(
        self,
        *,
        max_train_tokens: int | None = None,
        max_spend_usd: float | None = None,
        hourly_rate_usd: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (max_spend_usd is None) != (hourly_rate_usd is None):
            raise ValueError("max_spend_usd and hourly_rate_usd must be set together")
        if max_train_tokens is not None and max_train_tokens <= 0:
            raise ValueError(f"max_train_tokens must be positive, got {max_train_tokens}")
        if (
            max_spend_usd is not None
            and hourly_rate_usd is not None
            and (max_spend_usd <= 0 or hourly_rate_usd <= 0)
        ):
            raise ValueError("max_spend_usd and hourly_rate_usd must be positive")
        self.max_train_tokens = max_train_tokens
        self.max_spend_usd = max_spend_usd
        self.hourly_rate_usd = hourly_rate_usd
        self._clock = clock
        self._started_at: float | None = None
        self.tokens_used = 0

    @classmethod
    def from_spec(cls, spec: TrainSpec, *, clock: Callable[[], float] = time.monotonic) -> Budget:
        """The budget ``spec`` asks for."""
        return cls(
            max_train_tokens=spec.max_train_tokens,
            max_spend_usd=spec.max_spend_usd,
            hourly_rate_usd=spec.hourly_rate_usd,
            clock=clock,
        )

    def start(self) -> None:
        """Start the clock; later calls keep the first start."""
        if self._started_at is None:
            self._started_at = self._clock()

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since :meth:`start`; ``0.0`` before it."""
        return 0.0 if self._started_at is None else self._clock() - self._started_at

    @property
    def spend_usd(self) -> float | None:
        """The elapsed time priced at ``hourly_rate_usd``; ``None`` without a rate."""
        if self.hourly_rate_usd is None:
            return None
        return self.elapsed_seconds / 3600 * self.hourly_rate_usd

    def exhausted(self, next_tokens: int) -> str | None:
        """Why the next step, of ``next_tokens`` tokens, must not run; ``None`` if it may.

        It must not when it would take the trained tokens over ``max_train_tokens``, or when
        the time spent so far already costs ``max_spend_usd`` at ``hourly_rate_usd``.
        """
        self.start()
        cap = self.max_train_tokens
        if cap is not None and self.tokens_used + next_tokens > cap:
            return (
                f"the next step would train on {self.tokens_used + next_tokens} tokens, "
                f"over max_train_tokens {cap}"
            )
        spend = self.spend_usd
        if self.max_spend_usd is not None and spend is not None and spend >= self.max_spend_usd:
            return (
                f"{self.elapsed_seconds / 3600:.2f} h at {self.hourly_rate_usd} USD/h reached "
                f"max_spend_usd {self.max_spend_usd}"
            )
        return None

    def record(self, tokens: int) -> None:
        """Count ``tokens`` as trained (the loop calls it for each step it submits)."""
        self.tokens_used += tokens

    def to_json(self) -> dict[str, Any]:
        """The caps and what was used, for ``result.json`` and ``status``."""
        return {
            "max_train_tokens": self.max_train_tokens,
            "tokens_used": self.tokens_used,
            "max_spend_usd": self.max_spend_usd,
            "hourly_rate_usd": self.hourly_rate_usd,
            "elapsed_seconds": self.elapsed_seconds,
            "spend_usd": self.spend_usd,
        }
