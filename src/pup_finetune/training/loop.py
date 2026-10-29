# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The shared, provider-neutral SFT loop, written once against :class:`TrainingSession`.

Each epoch shuffles the rows (:func:`~pup_finetune.training.batching.epoch_rng`), and each
optimizer step sends one ``forward_backward`` per microbatch, then one ``optim_step`` at the
step's scheduled learning rate (:meth:`TrainSpec.learning_rate_at` over
:meth:`TrainSpec.max_steps`, so a resume sends the same rates), with up to
``spec.pipeline_depth`` steps in flight. Every submitted result is resolved, in order,
also when a step fails or the process is interrupted; an interrupt (Ctrl-C, SIGTERM) wins
over an ordinary error, which is kept as a note. ``forward_backward`` is never retried: a
retry would accumulate the step's gradients twice. After each epoch the validation loss
(``forward``, retried with a short backoff a bounded number of times on transient errors)
selects the best epoch and drives early stopping; the epoch ends with a resumable training
checkpoint, plus a sampler checkpoint on a new best.

Checkpoints: besides the one at each epoch end, a training checkpoint is saved at the first
step boundary after ``spec.checkpoint_every_minutes`` have passed since the last one (the
steps in flight are resolved first), so an interruption costs minutes rather than an epoch.
Each training checkpoint is reported with its :class:`ResumePoint`: the epoch, the optimizer
step, the steps already trained in that epoch and the epoch's loss so far. The order of an
epoch's rows depends only on ``(seed, epoch)``, so :func:`run_training` with a
:class:`Resume` (in a session opened from that checkpoint) skips exactly the steps already
trained and ends with the same epoch metrics as an uninterrupted run.

Losses are token means: ``sum(loss_sum) / sum(weight)`` over the step, epoch or validation
set. Between steps the loop stops cleanly when :meth:`LoopState.cancel_requested` says so or
the :class:`~pup_finetune.training.budget.Budget` is exhausted; it then saves a training
checkpoint of the steps since the last one. The spend guard is also asked before every other
paid call: each ``forward_backward`` and ``optim_step`` (a step refused part-way is never
applied or counted), each validation microbatch and each checkpoint save. A budget stop
during an epoch's validation or checkpoints leaves that epoch unrecorded, so it never
becomes the best on a partial validation. A cancel does not cut an epoch's end short: one
asked for once the epoch's last step is trained still waits for its validation and
checkpoints, which are paid for, so that the cancelled run can keep the epoch as its best;
training then stops ``cancelled``, the last epoch included. Only the checkpoint saved at a
stop may use the budget's spend reserve. ``finalize`` runs last, while the session is still
open (the place to promote the best checkpoint before the session ends); if it fails,
:class:`FinalizeFailed` carries the training result so it is not lost.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, get_args

from pup_finetune.errors import ExternalError, StateError, UsageError
from pup_finetune.providers.base import (
    CheckpointRef,
    ModelIdTaken,
    PendingResult,
    StepMetrics,
    TrainingSession,
    TrainSpec,
    is_transient,
)
from pup_finetune.redaction import redact
from pup_finetune.storage import finite_json
from pup_finetune.training.batching import (
    Microbatch,
    Step,
    epoch_rng,
    plan_batches,
    step_tokens,
)
from pup_finetune.training.budget import Budget
from pup_finetune.training.datums import RowExamples
from pup_finetune.training.records import (
    EpochMetrics,
    Resume,
    ResumePoint,
    StoppedReason,
    TrainResult,
)

CheckpointKind = Literal["training", "sampler", "recreated"]
"""What a checkpoint name is for: ``recreated`` names a sampler checkpoint saved again from a
training checkpoint at the end of a run, when the original one's session has ended."""

CHECKPOINT_KINDS: tuple[CheckpointKind, ...] = get_args(CheckpointKind)

CheckpointNamer = Callable[[CheckpointKind, int, int], str]
"""``(kind, epoch, step)`` to a checkpoint name; ``step`` counts optimizer steps over the run.

Names must differ by kind: the loop saves a training and a sampler checkpoint at the same
epoch and step, and a backend need not keep separate namespaces per kind.
"""

CheckpointReason = Literal["epoch", "time", "stop"]
"""Why a training checkpoint was saved: an epoch ended, ``checkpoint_every_minutes`` passed,
or training stopped mid-epoch."""

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

    def record_plan(self, *, steps_per_epoch: int, max_steps: int) -> None:
        """Training is about to start (or resume): an epoch has ``steps_per_epoch`` optimizer
        steps, and the run at most ``max_steps`` (fewer when it stops early)."""
        ...

    def record_step(self, *, epoch: int, step: int, loss: float) -> None:
        """Optimizer step ``step`` (over the run) of ``epoch`` has finished with mean ``loss``."""
        ...

    def record_checkpoint(self, ref: CheckpointRef, *, resume_point: ResumePoint | None) -> None:
        """A checkpoint was saved; ``resume_point`` is given for training checkpoints."""
        ...

    def record_warning(self, message: str) -> None:
        """Something the run's result should warn about (also returned in the result)."""
        ...

    def record_epoch(self, metrics: EpochMetrics, *, best: EpochMetrics | None) -> None:
        """Epoch ``metrics.epoch`` has finished; ``best`` is the best epoch so far.

        A resume from the epoch's end records it again, so recording it twice keeps one."""
        ...


Finalize = Callable[[TrainingSession, TrainResult], TrainResult]
"""Runs once training stops, with the session still open; returns the final result."""


class FinalizeFailed(ExternalError):
    """Training ended but ``finalize`` (e.g. promoting the best checkpoint) did not.

    :attr:`result` is the training result, so it is not lost; :attr:`cause` is the error that
    stopped ``finalize`` (also chained as the ``__cause__`` when raised ``from`` it).
    :attr:`interrupted` is true when that was an interrupt (Ctrl-C or SIGTERM) rather than an
    error. The hint offers a resume that retries only the promotion, except after a
    :class:`~pup_finetune.providers.base.ModelIdTaken`, whose own hint says what has to change
    first.
    """

    def __init__(self, result: TrainResult, cause: BaseException) -> None:
        detail = redact(str(cause))
        retry = "resume the run to retry only the promotion"
        if isinstance(cause, ModelIdTaken) and cause.hint:
            retry = cause.hint
        super().__init__(
            f"training ended ({result.stopped_reason}), but finalizing the run failed: "
            f"{type(cause).__name__}{': ' + detail if detail else ''}",
            source="finalize",
            hint=f"the training result is kept; {retry}",
        )
        self.result = result
        self.cause = cause
        self.interrupted = not isinstance(cause, Exception)


def checked_checkpoint_name(
    namer: CheckpointNamer, kind: CheckpointKind, epoch: int, step: int, *, limit: int
) -> str:
    """``namer(kind, epoch, step)``, or :class:`UsageError` unless it has 1 to ``limit`` chars."""
    name = namer(kind, epoch, step)
    if not name or len(name) > limit:
        raise UsageError(
            f"checkpoint name {name!r} must be 1 to {limit} characters for this backend",
            hint=f"the backend allows {limit} characters; train for fewer epochs or "
            "steps, or name checkpoints more briefly",
        )
    return name


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
    resume: Resume | None = None,
) -> TrainResult:
    """Train on ``train_rows`` for up to ``spec.max_epochs`` epochs; see the module docstring.

    ``max_checkpoint_name_len`` (the backend's
    :attr:`~pup_finetune.providers.base.TrainingBackend.max_checkpoint_name_len`) is checked
    against the names ``checkpoint_namer`` gives before any step is sent. ``clock`` (seconds)
    times the checkpoints; ``sleep`` waits between validation retries. ``on_event`` receives
    a ``{"type": "step", "epoch", "step", "loss", "learning_rate", "tokens", "elapsed_s"}``
    dict per optimizer step, and ``"epoch"``, ``"checkpoint"``, ``"retry"`` and ``"stopped"``
    events, with any non-finite number as ``None``. ``resume`` continues a run in a session
    opened from ``resume.point.checkpoint``, with the same rows and spec; ``budget`` then also
    counts the time spent before (see :class:`Resume`). A failing ``finalize`` raises
    :class:`FinalizeFailed`.
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
        resume=resume,
    )
    result = loop.run()
    if finalize is None:
        return result
    try:
        return finalize(session, result)
    except BaseException as exc:
        raise FinalizeFailed(result, exc) from exc


@dataclass(slots=True)
class _InFlight:
    """One submitted optimizer step: its ``forward_backward`` results and its ``optim_step``."""

    epoch: int
    step: int
    tokens: int
    learning_rate: float
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
        total_steps: int,
    ) -> None:
        self._session = session
        self._spec = spec
        self._total_steps = total_steps
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
        flight = _InFlight(
            epoch, step, tokens, self._spec.learning_rate_at(step, self._total_steps)
        )
        self._queue.append(flight)
        for microbatch in microbatches:
            refused = guard()
            if refused is not None:
                return refused
            flight.results.append(self._session.forward_backward(microbatch))
        refused = guard()
        if refused is not None:
            return refused
        adam = dataclasses.replace(self._adam, learning_rate=flight.learning_rate)
        flight.optim = self._session.optim_step(adam)
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
    return _mean(loss_sum, weight, what)


def _mean(loss_sum: float, weight: float, what: str) -> float:
    """``loss_sum / weight``, or :class:`ExternalError` unless all three are finite.

    A finite sum over a tiny weight can still overflow; every recorded loss is finite, so
    it is written to the run directory and replayed on resume exactly.
    """
    mean = loss_sum / weight if math.isfinite(weight) and weight > 0 else math.nan
    if not (math.isfinite(loss_sum) and math.isfinite(mean)):
        raise ExternalError(
            f"the {what} returned a loss sum of {loss_sum} over a weight of {weight}",
            source="training",
        )
    return mean


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
        resume: Resume | None,
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
        self._max_steps = spec.max_steps(len(self._train_rows))
        self._started = clock()
        self._last_checkpoint_at = self._started
        self._step = 0
        self._checkpoint_step = 0
        self._start_epoch = 1
        self._skip_steps = 0
        self._epoch_loss_sum = 0.0
        self._epoch_weight = 0.0
        self._epoch_steps = 0
        self._epoch_tokens = 0
        self._epochs: list[EpochMetrics] = []
        self._best: EpochMetrics | None = None
        self._best_metric = math.inf
        self._patience_loss = math.inf
        self._stale_epochs = 0
        self._last_training: CheckpointRef | None = None
        self._warnings: list[str] = []
        if resume is not None:
            self._restore(resume)

    def _restore(self, resume: Resume) -> None:
        """Continue from ``resume``: replay the finished epochs, then pick up at its point."""
        point = resume.point
        self._warnings = list(dict.fromkeys(resume.warnings))
        for record in resume.epochs:
            metric = _selection_metric(record)
            if record.new_best != self._is_new_best(metric):
                raise StateError(f"epoch {record.epoch}'s best-epoch flag does not match its loss")
            self._select(record, metric)
            self._epochs.append(record)
        finished = self._epochs[-1].epoch if self._epochs else 0
        if [record.epoch for record in self._epochs] != list(range(1, finished + 1)):
            raise StateError("the finished epochs are not numbered 1, 2, ...")
        if not point.follows(self._epochs) or point.epoch > self._spec.max_epochs:
            raise StateError(
                f"the last training checkpoint (epoch {point.epoch}, step {point.step}) does "
                f"not follow the {finished} finished epoch(s)"
            )
        self._step = self._checkpoint_step = point.step
        self._last_training = point.checkpoint
        self._budget.record(point.train_tokens)
        # The time spent after the checkpoint (later steps, validation, failed saves) was paid.
        self._budget.add_elapsed(max(point.elapsed_seconds, resume.elapsed_seconds))
        if point.epoch == finished:  # saved at that epoch's end: continue with the next one
            self._start_epoch = finished + 1
            return
        self._start_epoch, self._skip_steps = point.epoch, point.steps_in_epoch
        self._epoch_loss_sum, self._epoch_weight = point.epoch_loss_sum, point.epoch_weight
        self._epoch_steps, self._epoch_tokens = point.steps_in_epoch, point.epoch_tokens

    def run(self) -> TrainResult:
        spec = self._spec
        steps_per_epoch = math.ceil(len(self._train_rows) / spec.batch_size)
        for kind in CHECKPOINT_KINDS:  # fail before paying
            self._name(kind, spec.max_epochs, self._max_steps)
        if self._skip_steps > steps_per_epoch:
            raise StateError(
                f"the run already trained {self._skip_steps} steps of epoch "
                f"{self._start_epoch}, but an epoch has {steps_per_epoch}",
                hint="resume with the same training rows and settings as the run",
            )
        self._state.record_plan(steps_per_epoch=steps_per_epoch, max_steps=self._max_steps)
        if self._epochs and not self._skip_steps:
            # Resumed at the last finished epoch's end: a crash may have kept it from the state.
            self._state.record_epoch(self._epochs[-1], best=self._best)
        self._emit(
            {
                "type": "plan",
                "steps_per_epoch": steps_per_epoch,
                "max_steps": self._max_steps,
                "max_epochs": spec.max_epochs,
                "lr_schedule": spec.lr_schedule,
                "warmup_steps": spec.warmup_steps(self._max_steps),
                "checkpoint_every_minutes": spec.checkpoint_every_minutes,
                "epoch": self._start_epoch,
                "step": self._step,
            }
        )
        if not self._validation:
            self._warn(NO_VALIDATION_WARNING)
        self._budget.start()
        stopped: StoppedReason = "completed"
        if self._epochs and self._patience_ran_out(self._epochs[-1].epoch):
            # Resumed after the epoch that stopped early, before the run could end.
            stopped = "early_stopped"
        else:
            for epoch in range(self._start_epoch, spec.max_epochs + 1):
                stop = self._train_epoch(epoch)
                if stop is None:
                    try:
                        self._end_epoch(epoch)
                    except _BudgetStop as exc:
                        stop = self._budget_stop(str(exc))
                    else:
                        stop = self._cancelled()
                if stop is not None:
                    stopped = stop
                    if self._step > self._checkpoint_step:
                        self._save_stop_checkpoint(epoch)
                    break
                if self._patience_ran_out(epoch):
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
        """Run epoch ``epoch``'s steps; the reason it stopped early, if it did.

        A resumed epoch skips the steps its resume point already trained.
        """
        spec = self._spec
        steps = plan_batches(
            self._train_rows,
            rows_per_step=spec.batch_size,
            microbatch_token_budget=spec.microbatch_token_budget,
            rng=epoch_rng(spec.seed, epoch),
        )
        if epoch == self._start_epoch and self._skip_steps:
            steps = steps[self._skip_steps :]
        else:
            self._epoch_loss_sum, self._epoch_weight = 0.0, 0.0
            self._epoch_steps, self._epoch_tokens = 0, 0
        pipeline = _Pipeline(self._session, spec, self._step_done, self._max_steps)
        stop: StoppedReason | None = None
        try:
            for step in steps:
                pipeline.make_room()
                stop = self._stop_reason(step)
                if stop is not None:
                    break
                if self._checkpoint_due():
                    pipeline.drain()  # a checkpoint holds exactly the steps before it
                    why = self._budget.spend_exhausted()
                    if why is not None:  # the stop saves this checkpoint, from the reserve
                        stop = self._budget_stop(f"stopped before a timed checkpoint: {why}")
                        break
                    self._save_training(epoch, "time")
                    stop = self._stop_reason(step)  # draining and saving took time
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

    def _checkpoint_due(self) -> bool:
        """Whether ``checkpoint_every_minutes`` passed since the last training checkpoint."""
        if self._step == self._checkpoint_step:
            return False
        interval = self._spec.checkpoint_every_minutes * 60
        return self._clock() - self._last_checkpoint_at >= interval

    def _stop_reason(self, step: Step) -> StoppedReason | None:
        if (cancelled := self._cancelled()) is not None:
            return cancelled
        why = self._budget.exhausted(step_tokens(step))
        if why is not None:
            return self._budget_stop(f"stopped before step {self._step + 1}: {why}")
        return None

    def _cancelled(self) -> StoppedReason | None:
        """``cancelled``, reported, when a cancel was requested."""
        if not self._state.cancel_requested():
            return None
        self._emit({"type": "stopped", "reason": "cancelled", "step": self._step})
        return "cancelled"

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
        # Running sums in step order, so a resumed epoch adds up exactly as an unbroken one.
        for metric in metrics:
            self._epoch_loss_sum += metric.loss_sum
            self._epoch_weight += metric.weight
        self._epoch_steps += 1
        self._epoch_tokens += flight.tokens
        self._state.record_step(epoch=flight.epoch, step=flight.step, loss=loss)
        self._emit(
            {
                "type": "step",
                "epoch": flight.epoch,
                "step": flight.step,
                "loss": loss,
                "learning_rate": flight.learning_rate,
                "tokens": flight.tokens,
                "elapsed_s": self._clock() - self._started,
            }
        )

    # --- epoch end --------------------------------------------------------------------

    def _end_epoch(self, epoch: int) -> None:
        """Validate, select and checkpoint epoch ``epoch``.

        Raises :class:`_BudgetStop`, before recording the epoch, when the spend guard refuses
        a validation microbatch or a checkpoint. A time-based checkpoint that came due with
        the epoch's last step is saved first, so a failed validation does not lose the epoch
        (it then serves as the epoch's checkpoint too).
        """
        if self._checkpoint_due():
            self._require_spend(f"a timed checkpoint of epoch {epoch}")
            self._save_training(epoch, "time")
        train_loss = _mean(self._epoch_loss_sum, self._epoch_weight, f"training of epoch {epoch}")
        val_loss = self._validation_loss(epoch) if self._validation else None
        metric = train_loss if val_loss is None else val_loss
        new_best = self._is_new_best(metric)
        training_ckpt = self._last_training
        if training_ckpt is None or not self._covers_epoch_end(training_ckpt, epoch):
            self._require_spend(f"the training checkpoint of epoch {epoch}")
            training_ckpt = self._save_training(epoch, "epoch")
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
        self._select(record, metric)
        self._state.record_epoch(record, best=self._best)
        self._emit(
            {"type": "epoch", **record.to_json(), "elapsed_s": self._clock() - self._started}
        )

    def _covers_epoch_end(self, ref: CheckpointRef, epoch: int) -> bool:
        """Whether ``ref`` already holds every step of epoch ``epoch``.

        So it is when a run resumes from the checkpoint saved at the epoch's end, after the
        epoch's record (e.g. its sampler checkpoint) failed: it is reused, not saved again
        under the same name.
        """
        return ref.epoch == epoch and self._checkpoint_step == self._step

    def _is_new_best(self, metric: float) -> bool:
        return self._best is None or metric < self._best_metric

    def _select(self, record: EpochMetrics, metric: float) -> None:
        """Track the best epoch and the patience counter after epoch ``record``."""
        if self._is_new_best(metric):
            self._best, self._best_metric = record, metric
        # The best is strictly lower; patience counts epochs without a min_delta gain.
        if metric < self._patience_loss - self._spec.min_delta:
            self._patience_loss, self._stale_epochs = metric, 0
        else:
            self._stale_epochs += 1

    def _patience_ran_out(self, epoch: int) -> bool:
        """Whether training stops early after epoch ``epoch``; never after the last one."""
        return self._stale_epochs >= self._spec.patience and epoch < self._spec.max_epochs

    def _validation_loss(self, epoch: int) -> float:
        metrics = [self._forward(microbatch, epoch) for microbatch in self._validation]
        return _token_mean(metrics, "validation")

    def _forward(self, microbatch: Microbatch, epoch: int) -> StepMetrics:
        """One validation ``forward``, retried up to :data:`VALIDATION_RETRIES` times.

        The spend guard is asked before every attempt, so a retry after the backoff is not
        started once the budget ran out while waiting (:class:`_BudgetStop`).
        """
        attempt = 0
        while True:
            self._require_spend(f"finishing the validation of epoch {epoch}")
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
        return checked_checkpoint_name(self._namer, kind, epoch, step, limit=self._max_name_len)

    def _save_training(self, epoch: int, reason: CheckpointReason) -> CheckpointRef:
        name = self._name("training", epoch, self._step)
        ref = self._session.save_training_checkpoint(name, epoch=epoch, step=self._step)
        self._last_training = ref
        self._checkpoint_step = self._step
        self._last_checkpoint_at = self._clock()
        point = ResumePoint(
            checkpoint=ref,
            epoch=epoch,
            step=self._step,
            steps_in_epoch=self._epoch_steps,
            epoch_loss_sum=self._epoch_loss_sum,
            epoch_weight=self._epoch_weight,
            epoch_tokens=self._epoch_tokens,
            train_tokens=self._budget.tokens_used,
            elapsed_seconds=self._budget.elapsed_seconds,
        )
        self._recorded(ref, point, reason)
        return ref

    def _save_stop_checkpoint(self, epoch: int) -> None:
        """Save the steps since the last checkpoint after a stop; it may use the reserve."""
        why = self._budget.spend_exhausted(use_reserve=True)
        if why is not None:
            self._warn(
                f"did not save a training checkpoint of the {self._step - self._checkpoint_step} "
                f"step(s) since the last one: {why}"
            )
            return
        self._save_training(epoch, "stop")

    def _save_sampler(self, epoch: int) -> CheckpointRef:
        name = self._name("sampler", epoch, self._step)
        ref = self._session.save_sampler_checkpoint(name, epoch=epoch, step=self._step)
        self._recorded(ref, None, "epoch")
        return ref

    def _recorded(
        self, ref: CheckpointRef, point: ResumePoint | None, reason: CheckpointReason
    ) -> None:
        self._state.record_checkpoint(ref, resume_point=point)
        self._emit(
            {
                "type": "checkpoint",
                "kind": ref.kind,
                "name": ref.name,
                "epoch": ref.epoch,
                "step": ref.step,
                "reason": reason,
            }
        )

    def _warn(self, message: str) -> None:
        if message in self._warnings:  # e.g. warned again on resume
            return
        self._warnings.append(message)
        self._state.record_warning(message)

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(finite_json(event))


def _selection_metric(record: EpochMetrics) -> float:
    """What picks the best epoch: the validation loss, else the training loss."""
    return record.train_loss if record.val_loss is None else record.val_loss
