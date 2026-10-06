# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The shared, provider-neutral SFT loop, written once against :class:`TrainingSession`.

Each epoch shuffles the rows (:func:`~pup_finetune.training.batching.epoch_rng`), and each
optimizer step sends one ``forward_backward`` per microbatch, then one ``optim_step``, with
up to ``spec.pipeline_depth`` steps in flight. Every submitted result is resolved, in order,
also when a step fails or the process is interrupted; an interrupt (Ctrl-C, SIGTERM) wins
over an ordinary error, which is kept as a note. ``forward_backward`` is never retried: a
retry would accumulate the step's gradients twice. After each epoch the validation loss
(``forward``, retried with a short backoff a bounded number of times on transient errors)
selects the best epoch and drives early stopping; the epoch ends with a resumable training
checkpoint, plus a sampler checkpoint on a new best.

Losses are token means: ``sum(loss_sum) / sum(weight)`` over the step, epoch or validation
set. Between steps the loop stops cleanly when :meth:`LoopState.cancel_requested` says so or
the :class:`~pup_finetune.training.budget.Budget` is exhausted; it then saves a training
checkpoint of the steps since the last one. The spend guard is also asked before every other
paid call: each ``forward_backward`` and ``optim_step`` (a step refused part-way is never
applied or counted), each validation microbatch and each checkpoint save. A stop during an
epoch's validation or checkpoints leaves that epoch unrecorded, so it never becomes the
best on a partial validation. Only the checkpoint saved at a stop may use the budget's spend
reserve. ``finalize`` runs last, while the session is still open (the place to promote the
best checkpoint before the session ends).
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
    CheckpointRef,
    PendingResult,
    StepMetrics,
    TrainingSession,
    TrainSpec,
    TransientError,
)
from pup_finetune.training.batching import (
    Microbatch,
    Step,
    epoch_rng,
    plan_batches,
    step_tokens,
)
from pup_finetune.training.budget import Budget
from pup_finetune.training.datums import RowExamples

StoppedReason = Literal["completed", "early_stopped", "budget_exhausted", "cancelled"]

CheckpointKind = Literal["training", "sampler"]

CheckpointNamer = Callable[[CheckpointKind, int, int], str]
"""``(kind, epoch, step)`` to a checkpoint name; ``step`` counts optimizer steps over the run.

Names must differ by kind: the loop saves a training and a sampler checkpoint at the same
epoch and step, and a backend need not keep separate namespaces per kind.
"""

EventSink = Callable[[dict[str, Any]], None]

VALIDATION_RETRIES = 2
"""Extra attempts for a validation ``forward`` that failed with a transient error."""

MAX_RETRY_DELAY_S = 10.0
"""The longest wait before retrying a validation ``forward``; retry ``n`` waits ``2**n`` s."""

NO_VALIDATION_WARNING = (
    "the validation split is empty, so the best epoch is chosen by training loss, "
    "which cannot detect overfitting"
)


class LoopState(Protocol):
    """Where the loop reports progress, and asks whether to stop. See :class:`RunState`."""

    def cancel_requested(self) -> bool:
        """Whether to stop at the next step boundary."""
        ...

    def record_step(self, *, epoch: int, step: int) -> None:
        """Optimizer step ``step`` (over the run) of ``epoch`` has finished."""
        ...

    def record_checkpoint(self, ref: CheckpointRef) -> None:
        """A checkpoint was saved."""
        ...

    def record_warning(self, message: str) -> None:
        """Something the run's result should warn about (also returned in the result)."""
        ...

    def record_epoch(self, metrics: EpochMetrics, *, best: EpochMetrics | None) -> None:
        """Epoch ``metrics.epoch`` has finished; ``best`` is the best epoch so far."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class EpochMetrics:
    """One finished epoch. ``val_loss`` is ``None`` when the validation split is empty."""

    epoch: int
    steps: int
    train_loss: float
    val_loss: float | None
    train_tokens: int
    new_best: bool
    training_ckpt: CheckpointRef | None
    sampler_ckpt: CheckpointRef | None

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainResult:
    """How training ended and which checkpoints it left."""

    best_epoch: int | None
    best_val_loss: float | None
    stopped_reason: StoppedReason
    epochs: list[EpochMetrics]
    best_sampler_ckpt: CheckpointRef | None
    best_training_ckpt: CheckpointRef | None
    last_training_ckpt: CheckpointRef | None
    warnings: list[str]

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


Finalize = Callable[[TrainingSession, TrainResult], TrainResult]
"""Runs once training stops, with the session still open; returns the final result."""


def is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying for an idempotent call such as ``forward``."""
    return isinstance(exc, TransientError | TimeoutError | ConnectionError)


def run_training(
    session: TrainingSession,
    *,
    train_rows: Sequence[RowExamples],
    validation_rows: Sequence[RowExamples],
    spec: TrainSpec,
    state: LoopState,
    budget: Budget,
    checkpoint_namer: CheckpointNamer,
    max_checkpoint_name_len: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_event: EventSink | None = None,
    finalize: Finalize | None = None,
) -> TrainResult:
    """Train on ``train_rows`` for up to ``spec.max_epochs`` epochs; see the module docstring.

    ``max_checkpoint_name_len`` (the backend's
    :attr:`~pup_finetune.providers.base.TrainingBackend.max_checkpoint_name_len`) is checked
    against the names ``checkpoint_namer`` gives before any step is sent. ``sleep`` waits
    between validation retries. ``on_event`` receives a
    ``{"type": "step", "epoch", "step", "loss", "tokens", "elapsed_s"}`` dict per optimizer
    step, and ``"epoch"``, ``"checkpoint"``, ``"retry"`` and ``"stopped"`` events.
    """
    if not train_rows:
        raise UsageError("there are no training rows to train on")
    loop = _Loop(
        session,
        train_rows=train_rows,
        validation_rows=validation_rows,
        spec=spec,
        state=state,
        budget=budget,
        namer=checkpoint_namer,
        clock=clock,
        sleep=sleep,
        on_event=on_event,
        max_name_len=max_checkpoint_name_len,
    )
    result = loop.run()
    return finalize(session, result) if finalize is not None else result


@dataclass(slots=True)
class _InFlight:
    """One submitted optimizer step: its ``forward_backward`` results and its ``optim_step``."""

    epoch: int
    step: int
    tokens: int
    results: list[PendingResult[StepMetrics]] = field(default_factory=list)
    optim: PendingResult[None] | None = None

    def pending(self) -> list[PendingResult[Any]]:
        return [*self.results, *([] if self.optim is None else [self.optim])]


class _Pipeline:
    """Keeps at most ``depth`` optimizer steps in flight and resolves them in order."""

    def __init__(
        self,
        session: TrainingSession,
        spec: TrainSpec,
        on_done: Callable[[_InFlight, list[StepMetrics]], None],
    ) -> None:
        self._session = session
        self._adam = spec.adam_config()
        self._depth = spec.pipeline_depth
        self._on_done = on_done
        self._queue: deque[_InFlight] = deque()

    def submit(
        self,
        epoch: int,
        step: int,
        microbatches: Step,
        tokens: int,
        *,
        guard: Callable[[], str | None],
    ) -> str | None:
        """Send one optimizer step, asking ``guard`` before each call; ``None`` once sent.

        When ``guard`` refuses a call, the step stops there and its reason is returned. What
        it already sent is still resolved, but the step is never counted, and its gradients
        are never applied (there is no ``optim_step`` for them).
        """
        # Queued before anything is sent, so a failure part-way still resolves what was sent.
        flight = _InFlight(epoch, step, tokens)
        self._queue.append(flight)
        for microbatch in microbatches:
            refused = guard()
            if refused is not None:
                return refused
            flight.results.append(self._session.forward_backward(microbatch))
        refused = guard()
        if refused is not None:
            return refused
        flight.optim = self._session.optim_step(self._adam)
        return None

    def make_room(self) -> None:
        """Resolve the oldest steps until another one may be submitted."""
        while len(self._queue) >= self._depth:
            self._resolve_oldest()

    def drain(self) -> None:
        """Resolve every step in flight."""
        while self._queue:
            self._resolve_oldest()

    def abandon(self) -> list[tuple[str, BaseException]]:
        """After a failure: resolve everything still in flight, even through a Ctrl-C.

        Returns each error with what to call it in a note, e.g. ``"step 4 also failed"``.
        """
        errors: list[tuple[str, BaseException]] = []
        while self._queue:
            flight = self._queue.popleft()
            for pending in flight.pending():
                try:
                    pending.result()
                except BaseException as exc:
                    errors.append((f"step {flight.step} also failed", exc))
        return errors

    def _resolve_oldest(self) -> None:
        flight = self._queue.popleft()
        metrics: list[StepMetrics] = []
        # Even a Ctrl-C while waiting on one result leaves the step's other results resolved.
        errors: list[tuple[str, BaseException]] = []
        for pending in flight.results:
            try:
                metrics.append(pending.result())
            except BaseException as exc:
                errors.append((f"step {flight.step} also failed", exc))
        if flight.optim is not None:
            try:
                flight.optim.result()
            except BaseException as exc:
                errors.append((f"step {flight.step} also failed", exc))
        if errors:
            raise _primary(errors)
        if flight.optim is not None:  # a step refused part-way is not a finished step
            self._on_done(flight, metrics)


def _primary(errors: Sequence[tuple[str, BaseException]]) -> BaseException:
    """The error to raise: the first interrupt if any, else the first error.

    An interrupt is a :class:`BaseException` that is not an :class:`Exception`, such as
    ``KeyboardInterrupt``; it must win so the run is recorded as interrupted, not failed.
    Every other error becomes a note on it, under its label.
    """
    primary = next((exc for _, exc in errors if not isinstance(exc, Exception)), errors[0][1])
    for label, other in errors:
        if other is not primary:
            primary.add_note(f"{label}: {type(other).__name__}: {other}")
    return primary


def _token_mean(metrics: Sequence[StepMetrics], what: str) -> float:
    loss_sum = math.fsum(metric.loss_sum for metric in metrics)
    weight = math.fsum(metric.weight for metric in metrics)
    if not (math.isfinite(loss_sum) and math.isfinite(weight) and weight > 0):
        raise ExternalError(
            f"the {what} returned a loss sum of {loss_sum} over a weight of {weight}",
            source="training",
        )
    return loss_sum / weight


class _BudgetStop(Exception):
    """The spend guard refused a paid call at an epoch's end; the message says why."""


class _Loop:
    """The state of one :func:`run_training` call."""

    def __init__(
        self,
        session: TrainingSession,
        *,
        train_rows: Sequence[RowExamples],
        validation_rows: Sequence[RowExamples],
        spec: TrainSpec,
        state: LoopState,
        budget: Budget,
        namer: CheckpointNamer,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        on_event: EventSink | None,
        max_name_len: int,
    ) -> None:
        self._session = session
        self._train_rows = list(train_rows)
        self._spec = spec
        self._state = state
        self._budget = budget
        self._namer = namer
        self._clock = clock
        self._sleep = sleep
        self._on_event = on_event
        self._max_name_len = max_name_len
        self._validation = [
            microbatch
            for step in plan_batches(
                validation_rows,
                rows_per_step=spec.batch_size,
                microbatch_token_budget=spec.microbatch_token_budget,
                rng=None,
            )
            for microbatch in step
        ]
        self._started = clock()
        self._step = 0
        self._steps_since_checkpoint = 0
        self._epoch_metrics: list[StepMetrics] = []
        self._epoch_steps = 0
        self._epoch_tokens = 0
        self._epochs: list[EpochMetrics] = []
        self._best: EpochMetrics | None = None
        self._best_metric = math.inf
        self._last_training: CheckpointRef | None = None
        self._warnings: list[str] = []

    def run(self) -> TrainResult:
        spec = self._spec
        steps_per_epoch = math.ceil(len(self._train_rows) / spec.batch_size)
        for kind in ("training", "sampler"):  # fail before paying
            self._name(kind, spec.max_epochs, spec.max_epochs * steps_per_epoch)
        if not self._validation:
            self._warn(NO_VALIDATION_WARNING)
        self._budget.start()
        stopped: StoppedReason = "completed"
        patience_loss = math.inf
        stale_epochs = 0
        for epoch in range(1, spec.max_epochs + 1):
            stop = self._train_epoch(epoch)
            if stop is None:
                try:
                    metric = self._end_epoch(epoch)
                except _BudgetStop as exc:
                    stop = self._budget_stop(str(exc))
            if stop is not None:
                stopped = stop
                if self._steps_since_checkpoint:
                    self._save_stop_checkpoint(epoch)
                break
            # The best is strictly lower; patience counts epochs without a min_delta gain.
            if metric < patience_loss - spec.min_delta:
                patience_loss, stale_epochs = metric, 0
            else:
                stale_epochs += 1
            if stale_epochs >= spec.patience and epoch < spec.max_epochs:
                stopped = "early_stopped"
                self._emit({"type": "stopped", "reason": stopped, "epoch": epoch})
                break
        best = self._best
        return TrainResult(
            best_epoch=best.epoch if best else None,
            best_val_loss=best.val_loss if best else None,
            stopped_reason=stopped,
            epochs=list(self._epochs),
            best_sampler_ckpt=best.sampler_ckpt if best else None,
            best_training_ckpt=best.training_ckpt if best else None,
            last_training_ckpt=self._last_training,
            warnings=list(self._warnings),
        )

    # --- training ---------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> StoppedReason | None:
        """Run epoch ``epoch``'s steps; the reason it stopped early, if it did."""
        spec = self._spec
        steps = plan_batches(
            self._train_rows,
            rows_per_step=spec.batch_size,
            microbatch_token_budget=spec.microbatch_token_budget,
            rng=epoch_rng(spec.seed, epoch),
        )
        self._epoch_metrics, self._epoch_steps, self._epoch_tokens = [], 0, 0
        pipeline = _Pipeline(self._session, spec, self._step_done)
        stop: StoppedReason | None = None
        try:
            for step in steps:
                pipeline.make_room()
                stop = self._stop_reason(step)
                if stop is not None:
                    break
                tokens = step_tokens(step)
                refused = pipeline.submit(
                    epoch, self._step + 1, step, tokens, guard=self._budget.spend_exhausted
                )
                if refused is not None:
                    stop = self._budget_stop(f"stopped during step {self._step + 1}: {refused}")
                    break
                self._step += 1
                self._budget.record(tokens)
            pipeline.drain()
        except BaseException as exc:
            # An interrupt seen while draining wins; the original error becomes its note.
            primary = _primary([("interrupted while handling", exc), *pipeline.abandon()])
            if primary is exc:
                raise
            raise primary from exc
        return stop

    def _stop_reason(self, step: Step) -> StoppedReason | None:
        if self._state.cancel_requested():
            self._emit({"type": "stopped", "reason": "cancelled", "step": self._step})
            return "cancelled"
        why = self._budget.exhausted(step_tokens(step))
        if why is not None:
            return self._budget_stop(f"stopped before step {self._step + 1}: {why}")
        return None

    def _budget_stop(self, why: str) -> StoppedReason:
        """Record that the budget stopped training, and why."""
        self._warn(why)
        self._emit({"type": "stopped", "reason": "budget_exhausted", "detail": why})
        return "budget_exhausted"

    def _require_spend(self, before: str) -> None:
        """Raise :class:`_BudgetStop` if the spend guard refuses the next paid call."""
        why = self._budget.spend_exhausted()
        if why is not None:
            raise _BudgetStop(f"stopped before {before}: {why}")

    def _step_done(self, flight: _InFlight, metrics: list[StepMetrics]) -> None:
        loss = _token_mean(metrics, f"training step {flight.step}")
        self._epoch_metrics.extend(metrics)
        self._epoch_steps += 1
        self._epoch_tokens += flight.tokens
        self._steps_since_checkpoint += 1
        self._state.record_step(epoch=flight.epoch, step=flight.step)
        self._emit(
            {
                "type": "step",
                "epoch": flight.epoch,
                "step": flight.step,
                "loss": loss,
                "tokens": flight.tokens,
                "elapsed_s": self._clock() - self._started,
            }
        )

    # --- epoch end --------------------------------------------------------------------

    def _end_epoch(self, epoch: int) -> float:
        """Validate, select and checkpoint epoch ``epoch``; returns its selection loss.

        Raises :class:`_BudgetStop`, before recording the epoch, when the spend guard refuses
        a validation microbatch or a checkpoint.
        """
        train_loss = _token_mean(self._epoch_metrics, f"training of epoch {epoch}")
        val_loss = self._validation_loss(epoch) if self._validation else None
        metric = train_loss if val_loss is None else val_loss
        new_best = self._best is None or metric < self._best_metric
        self._require_spend(f"the training checkpoint of epoch {epoch}")
        training_ckpt = self._save_training(epoch)
        sampler_ckpt = None
        if new_best:
            self._require_spend(f"the sampler checkpoint of epoch {epoch}")
            sampler_ckpt = self._save_sampler(epoch)
        record = EpochMetrics(
            epoch=epoch,
            steps=self._epoch_steps,
            train_loss=train_loss,
            val_loss=val_loss,
            train_tokens=self._epoch_tokens,
            new_best=new_best,
            training_ckpt=training_ckpt,
            sampler_ckpt=sampler_ckpt,
        )
        self._epochs.append(record)
        if new_best:
            self._best, self._best_metric = record, metric
        self._state.record_epoch(record, best=self._best)
        self._emit(
            {"type": "epoch", **record.to_json(), "elapsed_s": self._clock() - self._started}
        )
        return metric

    def _validation_loss(self, epoch: int) -> float:
        metrics = []
        for microbatch in self._validation:
            self._require_spend(f"finishing the validation of epoch {epoch}")
            metrics.append(self._forward(microbatch))
        return _token_mean(metrics, "validation")

    def _forward(self, microbatch: Microbatch) -> StepMetrics:
        """One validation ``forward``, retried up to :data:`VALIDATION_RETRIES` times."""
        attempt = 0
        while True:
            try:
                return self._session.forward(microbatch).result()
            except Exception as exc:
                if attempt >= VALIDATION_RETRIES or not is_transient(exc):
                    if attempt:
                        exc.add_note(f"validation forward failed after {attempt + 1} attempts")
                    raise
                attempt += 1
                delay = min(2.0**attempt, MAX_RETRY_DELAY_S)
                self._emit(
                    {
                        "type": "retry",
                        "call": "forward",
                        "attempt": attempt,
                        "error": type(exc).__name__,
                        "delay_s": delay,
                    }
                )
                self._sleep(delay)

    # --- checkpoints ------------------------------------------------------------------

    def _name(self, kind: CheckpointKind, epoch: int, step: int) -> str:
        name = self._namer(kind, epoch, step)
        limit = self._max_name_len
        if not name or len(name) > limit:
            raise UsageError(
                f"checkpoint name {name!r} must be 1 to {limit} characters for this backend",
                hint=f"the backend allows {limit} characters; train for fewer epochs or "
                "steps, or name checkpoints more briefly",
            )
        return name

    def _save_training(self, epoch: int) -> CheckpointRef:
        name = self._name("training", epoch, self._step)
        ref = self._session.save_training_checkpoint(name, epoch=epoch, step=self._step)
        self._last_training = ref
        self._steps_since_checkpoint = 0
        self._recorded(ref)
        return ref

    def _save_stop_checkpoint(self, epoch: int) -> None:
        """Save the steps since the last checkpoint after a stop; it may use the reserve."""
        why = self._budget.spend_exhausted(use_reserve=True)
        if why is not None:
            self._warn(
                f"did not save a training checkpoint of the {self._steps_since_checkpoint} "
                f"step(s) since the last one: {why}"
            )
            return
        self._save_training(epoch)

    def _save_sampler(self, epoch: int) -> CheckpointRef:
        name = self._name("sampler", epoch, self._step)
        ref = self._session.save_sampler_checkpoint(name, epoch=epoch, step=self._step)
        self._recorded(ref)
        return ref

    def _recorded(self, ref: CheckpointRef) -> None:
        self._state.record_checkpoint(ref)
        self._emit(
            {
                "type": "checkpoint",
                "kind": ref.kind,
                "name": ref.name,
                "epoch": ref.epoch,
                "step": ref.step,
            }
        )

    def _warn(self, message: str) -> None:
        self._warnings.append(message)
        self._state.record_warning(message)

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(event)
