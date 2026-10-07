# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Train, or resume training, in a run directory: a session, the loop, and promotion.

:func:`start_training` trains a ``created`` run from scratch; :func:`resume_training`
continues an interrupted, failed or cancelled one. Both read the provider, model profile and
:class:`~pup_finetune.providers.base.TrainSpec` from ``run.json``, apply the spec's provider
options, open a session (``seed`` from the spec; ``max_seq_len`` the longest example of the
rows, at most the profile's), and run
:func:`~pup_finetune.training.loop.run_training` with
:class:`~pup_finetune.training.promotion.Promotion` as its ``finalize`` hook, so the best
checkpoint is promoted before the session closes. The promoted model's id is
:func:`~pup_finetune.runs.model_id_for_run` (or, for a new best after an earlier one was
promoted, :func:`~pup_finetune.training.promotion.model_id_for_checkpoint`), and each request
is recorded in ``state.json`` before it is made
(:meth:`~pup_finetune.runs.RunState.record_promotion_request`), so a model an earlier attempt
of the run promoted from the same checkpoint (its response lost) is adopted, and one that
existed before is not. Once
training and promotion are done, a
failure to end the session only adds a warning: the result, and the promoted model, are
kept. :func:`start_training` takes an ``after_finalize`` hook (``train --evaluate`` passes
one) that runs once the promotion succeeded, with the session still open, so work that needs
the trained weights can use it before it closes, under the run's spend cap (the hook is
given the guard, reserve included); the hook's failure, or an interrupt during it, only adds
a warning too. Before the hook runs, the run's end is recorded
(:func:`~pup_finetune.runs.record_finished`: ``result.json`` written, phase ``completed``),
so a process killed during the hook leaves a finished run with its promoted model, not one
to resume; the warnings the hook adds are recorded once it returns.

A resume opens the session from the last training checkpoint and continues at its epoch and
step, keeping the epoch history, the best epoch, the patience counter and the budget spent
so far: the trained tokens at that checkpoint, and the wall-clock time of every earlier
attempt as ``state.json`` last recorded it, including the time after that checkpoint. The
promotion after training may use the budget's spend reserve, but a sampler checkpoint it
would have to re-create past ``max_spend_usd`` fails it. A run whose training ended but
whose promotion failed (``resume_mode`` ``finalize_only``) only retries the promotion, in a
session opened from the best training checkpoint; that retry, which the user asks for, is
not held to the budget.

The caller supplies the rendered rows, the same ones on every resume: the order of each
epoch is a function of the seed and the rows, so other rows would skip the wrong steps.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pup_finetune.errors import StateError, UsageError
from pup_finetune.providers.base import (
    CheckpointRef,
    LoraConfig,
    ModelProfile,
    Provider,
    TrainingBackend,
    TrainingSession,
    TrainSpec,
)
from pup_finetune.redaction import redact
from pup_finetune.runs import (
    Clock,
    RunDir,
    RunState,
    SignalMode,
    checkpoint_namer,
    execute_run,
    model_id_for_run,
    record_finished,
    require_resumable,
    utcnow,
)
from pup_finetune.training.budget import Budget
from pup_finetune.training.datums import RowExamples
from pup_finetune.training.loop import EventSink, FinalizeFailed, run_training
from pup_finetune.training.promotion import Promotion
from pup_finetune.training.records import EpochMetrics, Resume, TrainResult

__all__ = ["AfterFinalize", "SpendGuard", "resume_training", "start_training"]

SpendGuard = Callable[[], str | None]
"""Why the run's spend cap is reached (see :meth:`Budget.spend_exhausted`); ``None`` while
it is not."""

AfterFinalize = Callable[[TrainingSession, TrainResult, threading.Event, SpendGuard | None], None]
"""Runs after training and a successful promotion, in the still-open session: the session,
the final result, the event that a graceful cancel (Ctrl-C, SIGTERM) sets, and the run's
spend guard (``None`` without ``max_spend_usd``). The session's time still counts against
the cap, its reserve included, so a hook that keeps the session busy stops once the guard
says so."""


def start_training(
    provider: Provider,
    run_dir: RunDir,
    *,
    train_rows: Sequence[RowExamples],
    validation_rows: Sequence[RowExamples],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_event: EventSink | None = None,
    signals: SignalMode = "interrupt",
    heartbeat_s: float | None = None,
    now: Clock = utcnow,
    after_finalize: AfterFinalize | None = None,
) -> dict[str, Any]:
    """Train the ``created`` run in ``run_dir`` on ``provider``; returns ``result.json``.

    ``clock`` (seconds) times checkpoints and the spend guard; ``now`` stamps the run files.
    ``signals`` and ``heartbeat_s`` are passed to :func:`~pup_finetune.runs.execute_run`,
    which documents them and the errors raised. ``after_finalize`` runs after a successful
    promotion and once the run's end is recorded, before the session closes (see the module
    docstring); its error, or a Ctrl-C during it, becomes a warning on the result.
    """
    run = _Run.load(provider, run_dir, train_rows, validation_rows, clock, sleep, on_event)
    run = dataclasses.replace(run, after_finalize=after_finalize, now=now)
    return execute_run(run_dir, run.train, signals=signals, heartbeat_s=heartbeat_s, now=now)


def resume_training(
    provider: Provider,
    run_dir: RunDir,
    *,
    train_rows: Sequence[RowExamples],
    validation_rows: Sequence[RowExamples],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_event: EventSink | None = None,
    signals: SignalMode = "interrupt",
    heartbeat_s: float | None = None,
    now: Clock = utcnow,
) -> dict[str, Any]:
    """Resume the interrupted, failed or cancelled run in ``run_dir``; returns ``result.json``.

    A run left ``training`` by a process that died without recording it (no process holds the
    run's lock) counts as interrupted. Raises :class:`UsageError` for a run in another phase,
    or (unless only the promotion is left to retry) when the backend cannot resume, and
    :class:`~pup_finetune.errors.StateError` while another process trains it. ``train_rows``
    and ``validation_rows`` must be the run's own. Otherwise as :func:`start_training`.
    """
    state = run_dir.load_state()  # checked again under the lock by execute_run
    if state.phase != "training":  # a live trainer holds the lock; a dead one is resumable
        require_resumable(state)
    run = _Run.load(provider, run_dir, train_rows, validation_rows, clock, sleep, on_event)
    body = run.train
    if state.resume_mode == "finalize_only":
        body = run.finalize_only
    elif not run.backend.supports_resume:
        raise UsageError(
            f"the {provider.name} backend cannot resume training from a checkpoint",
            hint="start a new run to train again",
        )
    return execute_run(
        run_dir, body, resume=True, signals=signals, heartbeat_s=heartbeat_s, now=now
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Run:
    """What training in one run directory needs, read from its ``run.json``."""

    run_dir: RunDir
    backend: TrainingBackend
    profile: ModelProfile
    spec: TrainSpec
    max_seq_len: int
    train_rows: Sequence[RowExamples]
    validation_rows: Sequence[RowExamples]
    promotion: Promotion
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    on_event: EventSink | None
    after_finalize: AfterFinalize | None = None
    now: Clock = utcnow

    @classmethod
    def load(
        cls,
        provider: Provider,
        run_dir: RunDir,
        train_rows: Sequence[RowExamples],
        validation_rows: Sequence[RowExamples],
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        on_event: EventSink | None,
    ) -> _Run:
        manifest = run_dir.manifest()
        if manifest.get("provider") != provider.name:
            raise UsageError(
                f"the run in {run_dir.path} trains on {manifest.get('provider')!r}, "
                f"not {provider.name!r}",
                hint="pass the run's own provider",
            )
        spec = run_dir.spec()
        provider.validate_options(spec.provider_options)
        backend = provider.backend()
        run_id = str(manifest["run_id"])
        profile = run_dir.profile()
        # The same on every resume, since a resume needs the run's own rows.
        longest = _longest_example([*train_rows, *validation_rows])
        return cls(
            run_dir=run_dir,
            backend=backend,
            profile=profile,
            spec=spec,
            max_seq_len=profile.max_seq_len
            if longest is None
            else min(profile.max_seq_len, longest),
            train_rows=train_rows,
            validation_rows=validation_rows,
            promotion=Promotion(
                model_id=model_id_for_run(run_id),
                supports_promotion=backend.supports_promotion,
                checkpoint_namer=checkpoint_namer(run_id),
                max_checkpoint_name_len=backend.max_checkpoint_name_len,
                on_event=on_event,
            ),
            clock=clock,
            sleep=sleep,
            on_event=on_event,
        )

    @property
    def lora(self) -> LoraConfig:
        return self.spec.lora_config(self.profile)

    def train(self, state: RunState) -> TrainResult:
        """Train from the start, or from the run's last training checkpoint when resuming."""
        point = state.last_resume_point()
        history = [EpochMetrics.from_json(epoch) for epoch in self.run_dir.read_epochs()]
        if point is None and history:
            raise StateError(
                f"{self.run_dir.state_path}: epochs finished but no training checkpoint is "
                "recorded to resume from"
            )
        budget = Budget.from_spec(self.spec, clock=self.clock)
        # Every earlier attempt's time, as last saved: it includes what was spent after the
        # last training checkpoint, so repeated interruptions cannot exceed the spend cap.
        earlier = state.elapsed_seconds
        resume = None
        if point is not None:
            resume = Resume(
                point=point, epochs=history, warnings=state.warnings, elapsed_seconds=earlier
            )
        else:  # an earlier attempt that never saved a training checkpoint still spent time
            budget.add_elapsed(earlier)
        state.track_elapsed(lambda: budget.elapsed_seconds)
        # Promotion may spend the reserve the budget keeps beyond where training stops.
        promotion = dataclasses.replace(
            self.promotion,
            spend_guard=lambda: budget.spend_exhausted(use_reserve=True),
            requests=state,
        )

        def train_in(session: TrainingSession) -> TrainResult:
            result = run_training(
                session,
                train_rows=self.train_rows,
                validation_rows=self.validation_rows,
                spec=self.spec,
                state=state,
                budget=budget,
                checkpoint_namer=self.promotion.checkpoint_namer,
                max_checkpoint_name_len=self.backend.max_checkpoint_name_len,
                clock=self.clock,
                sleep=self.sleep,
                on_event=self.on_event,
                finalize=promotion,
                resume=resume,
            )
            guard = None
            if budget.max_spend_usd is not None:

                def guard() -> str | None:
                    return budget.spend_exhausted(use_reserve=True)

            return self._after(session, result, state, guard)

        return self._in_session(None if point is None else point.checkpoint, train_in)

    def _after(
        self,
        session: TrainingSession,
        result: TrainResult,
        state: RunState,
        spend_guard: SpendGuard | None,
    ) -> TrainResult:
        """Record the run's end, then run :attr:`after_finalize` with ``spend_guard``.

        Its failure (or a Ctrl-C) is a warning on ``result``, as is a failure to record the
        end early (it is recorded again when training returns).
        """
        if self.after_finalize is None:
            return result
        warnings: list[str] = []
        try:
            record_finished(self.run_dir, state, result, now=self.now)
        except Exception as exc:
            warnings.append(
                f"the run's end could not be recorded before the step after "
                f"training: {_describe(exc)}"
            )
        try:
            self.after_finalize(session, result, state.cancel_event, spend_guard)
        except (Exception, KeyboardInterrupt) as exc:
            warnings.append(f"the step after training failed: {_describe(exc)}")
        if not warnings:
            return result
        return dataclasses.replace(result, warnings=[*result.warnings, *warnings])

    def finalize_only(self, state: RunState) -> TrainResult:
        """Retry only the promotion of a run whose training ended."""
        result = TrainResult.from_json(self.run_dir.read_result() or {})
        if not self.backend.supports_promotion or result.best_epoch is None:
            return self.promotion.record_deployable(result)
        promotion = dataclasses.replace(self.promotion, requests=state)

        def promote_in(session: TrainingSession) -> TrainResult:
            try:
                return promotion(session, result)
            except BaseException as exc:
                raise FinalizeFailed(result, exc) from exc

        resume_from = result.best_training_ckpt if self.backend.supports_resume else None
        try:
            return self._in_session(resume_from, promote_in)
        except FinalizeFailed:
            raise
        except BaseException as exc:  # e.g. the session failed to open
            raise FinalizeFailed(result, exc) from exc

    def _in_session(
        self, resume_from: CheckpointRef | None, body: Callable[[TrainingSession], TrainResult]
    ) -> TrainResult:
        """``body`` in a new session; an error ending the session does not lose its result.

        Once ``body`` returned (training and promotion done), a failure to end the session
        becomes a warning on the result. When ``body`` raised, its error is what propagates,
        with the session's error as a note: a :class:`FinalizeFailed` keeps the training
        result, and a provider error or an interrupt (Ctrl-C) keeps how the run ended.
        """
        outcome: TrainResult | None = None
        raised: BaseException | None = None
        try:
            with self.backend.open_session(
                self.profile,
                self.lora,
                resume_from=resume_from,
                seed=self.spec.seed,
                max_seq_len=self.max_seq_len,
            ) as session:
                try:
                    outcome = body(session)
                except BaseException as exc:
                    raised = exc
                    raise
        except BaseException as exc:
            if raised is not None and exc is not raised:
                raised.add_note(f"ending the session also failed: {_describe(exc)}")
                raise raised from raised.__cause__
            if outcome is None:
                raise
            warning = f"ending the training session failed: {_describe(exc)}"
            return dataclasses.replace(outcome, warnings=[*outcome.warnings, warning])
        if outcome is None:  # a session context that swallowed an error
            raise StateError("the training session ended without a result")
        return outcome


def _describe(error: BaseException) -> str:
    detail = redact(str(error))
    return f"{type(error).__name__}{': ' + detail if detail else ''}"


def _longest_example(rows: Iterable[RowExamples]) -> int | None:
    """The longest example's tokens over ``rows``; ``None`` without examples."""
    return max((example.num_tokens for row in rows for example in row.examples), default=None)
