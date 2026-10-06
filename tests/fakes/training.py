# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fakes for driving the training loop: a recording session wrapper and an in-memory state.

:class:`RecordingSession` wraps a real :class:`~pup_finetune.providers.fake.FakeSession`
(or any session) and records every call, tracks which pending results are still
unresolved, and injects failures::

    backend = FakeBackend(FakeOptions(val_curve=[2.0, 1.5]))
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as s:
        session = RecordingSession(s, fail_forward_backward_at=3)

:func:`chat_rows` and :func:`row_examples` build synthetic rows; :data:`SESSION` holds the
``seed`` and ``max_seq_len`` to open fake sessions with.

:class:`MemoryState` is a :class:`~pup_finetune.training.loop.LoopState` that keeps the
loop's reports in lists and can ask for cancellation after a number of steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pup_finetune.providers.base import (
    AdamConfig,
    CheckpointRef,
    ModelRef,
    PendingResult,
    Sampler,
    StepMetrics,
    TrainingSession,
)
from pup_finetune.providers.fake import FAKE_PROFILE, FakeRenderer
from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.training import EpochMetrics, ResumePoint
from pup_finetune.training.datums import RowExamples, build_examples

SESSION: dict[str, Any] = {"seed": 42, "max_seq_len": FAKE_PROFILE.max_seq_len}
"""The keyword arguments to open a fake session with."""


def chat_rows(count: int, *, turns: int = 1, prefix: str = "row") -> list[dict[str, Any]]:
    """``count`` synthetic rows in provider form, each with ``turns`` assistant turns.

    Every row renders to the same number of tokens, so token budgets are easy to compute.
    """
    rows = []
    for i in range(count):
        messages: list[dict[str, Any]] = [{"role": "user", "content": f"question {i}"}]
        for turn in range(turns):
            if turn:
                messages.append({"role": "user", "content": f"follow-up {turn}"})
            messages.append({"role": "assistant", "content": f"answer {i} part {turn}"})
        rows.append({"id": f"{prefix}-{i}", "messages": messages})
    return rows


def row_examples(count: int, *, turns: int = 1, prefix: str = "row") -> list[RowExamples]:
    """:func:`chat_rows` rendered by the prefix-stable fake renderer."""
    rows = chat_rows(count, turns=turns, prefix=prefix)
    return build_examples(rows, FakeRenderer(), max_seq_len=4096).examples


class TrackedResult[T]:
    """A pending result that tells its session when it is resolved, optionally failing."""

    def __init__(
        self,
        owner: RecordingSession,
        kind: str,
        inner: PendingResult[T],
        error: BaseException | None = None,
        value: T | None = None,
    ) -> None:
        self.owner = owner
        self.kind = kind
        self.inner = inner
        self.error = error
        self.value = value
        self.resolved = False
        owner.outstanding.append(self)

    def result(self, timeout: float | None = None) -> T:
        if not self.resolved:
            self.resolved = True
            self.owner.outstanding.remove(self)
            self.owner.resolved.append(self.kind)
        inner = self.inner.result(timeout)
        if self.error is not None:
            raise self.error
        return inner if self.value is None else self.value


class RecordingSession:
    """Delegates to ``inner``; records calls and unresolved results; injects failures.

    - ``fail_forward_backward_at``: the n-th ``forward_backward`` (1-based) returns a result
      that raises ``forward_backward_error``.
    - ``forward_backward_errors``: the n-th ``forward_backward`` (1-based, as keys) returns a
      result that raises the given error.
    - ``raise_forward_backward_at``: the n-th ``forward_backward`` call itself raises.
    - ``forward_errors``: the next ``forward`` calls fail with these, in order.
    - ``fail_save_sampler_at``: the n-th ``save_sampler_checkpoint`` (1-based) raises
      ``save_sampler_error``.
    - ``train_metrics``: scripted metrics for successive ``forward_backward`` results.
    - ``forward_metrics``: scripted metrics for successive ``forward`` results.
    """

    def __init__(
        self,
        inner: TrainingSession,
        *,
        fail_forward_backward_at: int | None = None,
        forward_backward_error: BaseException | None = None,
        raise_forward_backward_at: int | None = None,
        forward_backward_errors: Mapping[int, BaseException] | None = None,
        forward_errors: Sequence[BaseException] = (),
        train_metrics: Sequence[StepMetrics] = (),
        forward_metrics: Sequence[StepMetrics] = (),
        fail_save_sampler_at: int | None = None,
        save_sampler_error: BaseException | None = None,
    ) -> None:
        self.inner = inner
        self.fail_save_sampler_at = fail_save_sampler_at
        self.save_sampler_error = save_sampler_error or RuntimeError("injected save failure")
        self.sampler_saves = 0
        self.fail_forward_backward_at = fail_forward_backward_at
        self.forward_backward_error = forward_backward_error or RuntimeError("injected failure")
        self.raise_forward_backward_at = raise_forward_backward_at
        self.forward_backward_errors = dict(forward_backward_errors or {})
        self.forward_errors = list(forward_errors)
        self.train_metrics = list(train_metrics)
        self.forward_metrics = list(forward_metrics)
        self.calls: list[str] = []
        self.train_batches: list[tuple[TokenizedExample, ...]] = []
        self.validation_batches: list[tuple[TokenizedExample, ...]] = []
        self.outstanding: list[TrackedResult[Any]] = []
        self.resolved: list[str] = []
        self.saved: list[CheckpointRef] = []
        self.max_steps_in_flight = 0

    @property
    def session_id(self) -> str:
        return self.inner.session_id

    def steps_in_flight(self) -> int:
        return sum(1 for result in self.outstanding if result.kind == "optim_step")

    def forward_backward(self, batch: Sequence[TokenizedExample]) -> PendingResult[StepMetrics]:
        self.calls.append("forward_backward")
        self.train_batches.append(tuple(batch))
        count = len(self.train_batches)
        if count == self.raise_forward_backward_at:
            raise self.forward_backward_error
        error = self.forward_backward_error if count == self.fail_forward_backward_at else None
        error = self.forward_backward_errors.get(count, error)
        value = self.train_metrics.pop(0) if self.train_metrics else None
        inner = self.inner.forward_backward(batch)
        return TrackedResult(self, "forward_backward", inner, error, value)

    def forward(self, batch: Sequence[TokenizedExample]) -> PendingResult[StepMetrics]:
        self.calls.append("forward")
        self.validation_batches.append(tuple(batch))
        error = self.forward_errors.pop(0) if self.forward_errors else None
        value = self.forward_metrics.pop(0) if self.forward_metrics and error is None else None
        return TrackedResult(self, "forward", self.inner.forward(batch), error, value)

    def optim_step(self, adam: AdamConfig) -> PendingResult[None]:
        self.calls.append("optim_step")
        result: TrackedResult[None] = TrackedResult(self, "optim_step", self.inner.optim_step(adam))
        self.max_steps_in_flight = max(self.max_steps_in_flight, self.steps_in_flight())
        return result

    def save_training_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        self.calls.append("save_training_checkpoint")
        ref = self.inner.save_training_checkpoint(name, epoch=epoch, step=step)
        self.saved.append(ref)
        return ref

    def save_sampler_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        self.calls.append("save_sampler_checkpoint")
        self.sampler_saves += 1
        if self.sampler_saves == self.fail_save_sampler_at:
            raise self.save_sampler_error
        ref = self.inner.save_sampler_checkpoint(name, epoch=epoch, step=step)
        self.saved.append(ref)
        return ref

    def load_weights(self, ckpt: CheckpointRef) -> None:
        self.inner.load_weights(ckpt)

    def promote(self, ckpt: CheckpointRef, model_id: str) -> ModelRef:
        return self.inner.promote(ckpt, model_id)

    def sampler(self, ckpt: CheckpointRef | None) -> Sampler:
        return self.inner.sampler(ckpt)


class MemoryState:
    """A :class:`~pup_finetune.training.loop.LoopState` in memory.

    ``cancel_after_steps`` asks for cancellation once that many steps were recorded.
    """

    def __init__(self, *, cancel_after_steps: int | None = None) -> None:
        self.cancel_after_steps = cancel_after_steps
        self.steps: list[tuple[int, int]] = []
        self.losses: list[float] = []
        self.checkpoints: list[CheckpointRef] = []
        self.resume_points: list[ResumePoint] = []
        self.epochs: list[EpochMetrics] = []
        self.bests: list[int | None] = []
        self.warnings: list[str] = []
        self.cancel_checks = 0

    def cancel_requested(self) -> bool:
        self.cancel_checks += 1
        limit = self.cancel_after_steps
        return limit is not None and len(self.steps) >= limit

    def record_step(self, *, epoch: int, step: int, loss: float) -> None:
        self.steps.append((epoch, step))
        self.losses.append(loss)

    def record_checkpoint(self, ref: CheckpointRef, *, resume_point: ResumePoint | None) -> None:
        self.checkpoints.append(ref)
        if resume_point is not None:
            self.resume_points.append(resume_point)

    def record_warning(self, message: str) -> None:
        self.warnings.append(message)

    def record_epoch(self, metrics: EpochMetrics, *, best: EpochMetrics | None) -> None:
        self.epochs.append(metrics)
        self.bests.append(best.epoch if best else None)
