# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A fake :class:`~pup_finetune.training.backend.TrainingBackend` with scripted losses.

``train_losses[e]`` and ``validation_losses[e]`` script epoch ``e + 1``: a float answers every
call of that epoch, a sequence answers call by call (cycling). The last entry repeats for later
epochs, and the epoch advances with each ``save_state``::

    backend = FakeBackend(validation_losses=[1.0, 0.8, (0.9, 0.7)])
    backend.fail_on("save_sampler_weights", call=2)

Active tokens are each batch's weight sum unless ``active_tokens`` is set. Every call is
recorded with its argument: the batch, the learning rate or the checkpoint name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pup_finetune.rendering.types import TokenDatum
from pup_finetune.training.backend import StepResult

Script = Sequence[float | Sequence[float]]


@dataclass(frozen=True)
class BackendCall:
    """One recorded call."""

    method: str
    arg: Any
    epoch: int


class FakeBackend:
    """Satisfies ``TrainingBackend``; see the module docstring."""

    def __init__(
        self,
        *,
        train_losses: Script = (1.0,),
        validation_losses: Script = (1.0,),
        active_tokens: float | None = None,
    ) -> None:
        self.train_losses = list(train_losses)
        self.validation_losses = list(validation_losses)
        self.active_tokens = active_tokens
        self.calls: list[BackendCall] = []
        self._failures: dict[str, tuple[int, Exception]] = {}

    def fail_on(self, method: str, *, call: int = 1, error: Exception | None = None) -> None:
        """Make the ``call``-th call (1-based) of ``method`` raise ``error``."""
        self._failures[method] = (call, error or RuntimeError(f"injected {method} failure"))

    @property
    def epoch(self) -> int:
        return len(self.state_names) + 1

    @property
    def methods(self) -> list[str]:
        return [call.method for call in self.calls]

    @property
    def train_batches(self) -> list[tuple[TokenDatum, ...]]:
        return self._args("forward_backward")

    @property
    def validation_batches(self) -> list[tuple[TokenDatum, ...]]:
        return self._args("forward")

    @property
    def learning_rates(self) -> list[float]:
        return self._args("optim_step")

    @property
    def state_names(self) -> list[str]:
        return self._args("save_state")

    @property
    def sampler_names(self) -> list[str]:
        return self._args("save_sampler_weights")

    def forward_backward(self, batch: Sequence[TokenDatum]) -> StepResult:
        return self._step("forward_backward", batch, self.train_losses)

    def forward(self, batch: Sequence[TokenDatum]) -> StepResult:
        return self._step("forward", batch, self.validation_losses)

    def optim_step(self, learning_rate: float) -> None:
        self._record("optim_step", learning_rate)

    def save_state(self, name: str) -> str:
        self._record("save_state", name)
        return f"state://{name}"

    def save_sampler_weights(self, name: str) -> str:
        self._record("save_sampler_weights", name)
        return f"sampler://{name}"

    def _args(self, method: str) -> list[Any]:
        return [call.arg for call in self.calls if call.method == method]

    def _record(self, method: str, arg: Any) -> None:
        self.calls.append(BackendCall(method, arg, self.epoch))
        failure = self._failures.get(method)
        if failure is not None and failure[0] == len(self._args(method)):
            raise failure[1]

    def _step(self, method: str, batch: Sequence[TokenDatum], script: Script) -> StepResult:
        epoch = self.epoch
        self._record(method, tuple(batch))
        value = script[min(epoch, len(script)) - 1]
        if not isinstance(value, int | float):
            calls = [c for c in self.calls if c.method == method and c.epoch == epoch]
            value = value[(len(calls) - 1) % len(value)]
        if self.active_tokens is not None:
            active = self.active_tokens
        else:
            active = float(sum(sum(datum.weights) for datum in batch))
        return StepResult(loss=float(value), active_tokens=active)
