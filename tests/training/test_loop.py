# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The shared SFT loop, driven against the offline fake provider."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

from pup_finetune.errors import ExternalError, StateError, UsageError
from pup_finetune.providers.base import (
    CheckpointRef,
    LoraConfig,
    StepMetrics,
    TrainingSession,
    TrainSpec,
    TransientError,
)
from pup_finetune.providers.fake import (
    FAKE_PROFILE,
    MAX_CHECKPOINT_NAME_LEN,
    FakeBackend,
    FakeOptions,
    FakeRenderer,
)
from pup_finetune.training import (
    MAX_RETRY_DELAY_S,
    NO_VALIDATION_WARNING,
    VALIDATION_RETRIES,
    Budget,
    EpochMetrics,
    FinalizeFailed,
    Resume,
    ResumePoint,
    RowExamples,
    TrainResult,
    build_examples,
    epoch_rng,
    plan_batches,
    run_training,
)
from tests.fakes.training import SESSION, MemoryState, RecordingSession, chat_rows, row_examples

LORA = LoraConfig(rank=8)
VALIDATION = row_examples(2, prefix="val")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@contextmanager
def recording(options: FakeOptions | None = None, **injected: Any) -> Iterator[RecordingSession]:
    with FakeBackend(options).open_session(FAKE_PROFILE, LORA, **SESSION) as inner:
        yield RecordingSession(inner, **injected)


def make_spec(**overrides: Any) -> TrainSpec:
    fields: dict[str, Any] = {"max_epochs": 3, "batch_size": 2, "pipeline_depth": 2}
    return TrainSpec(**(fields | overrides))


def train(
    session: TrainingSession,
    spec: TrainSpec,
    *,
    rows: Sequence[RowExamples] | None = None,
    validation: Sequence[RowExamples] = VALIDATION,
    state: MemoryState | None = None,
    budget: Budget | None = None,
    **kwargs: Any,
) -> TrainResult:
    return run_training(
        session,
        train_rows=row_examples(4) if rows is None else rows,
        validation_rows=validation,
        spec=spec,
        state=state or MemoryState(),
        budget=budget or Budget(),
        checkpoint_namer=lambda kind, epoch, step: f"{kind[0]}{epoch}s{step}",
        max_checkpoint_name_len=MAX_CHECKPOINT_NAME_LEN,
        **({"sleep": lambda seconds: None} | kwargs),
    )


def curve(*values: float) -> FakeOptions:
    return FakeOptions(val_curve=list(values))


# --- epochs, best epoch and early stopping ---------------------------------------------


def test_the_best_epoch_is_kept_when_validation_loss_improves_then_worsens() -> None:
    state = MemoryState()
    with recording(curve(2.0, 1.5, 1.8, 1.9, 1.0)) as session:
        result = train(session, make_spec(max_epochs=5, patience=2), state=state)
    assert result.stopped_reason == "early_stopped"
    assert [e.val_loss for e in result.epochs] == [2.0, 1.5, 1.8, 1.9]
    assert [e.new_best for e in result.epochs] == [True, True, False, False]
    assert (result.best_epoch, result.best_val_loss) == (2, 1.5)
    assert result.best_training_ckpt == result.epochs[1].training_ckpt
    assert result.best_sampler_ckpt == result.epochs[1].sampler_ckpt
    assert result.last_training_ckpt == result.epochs[3].training_ckpt
    # A training checkpoint every epoch, a sampler checkpoint on each new best only.
    kinds = [(ref.kind, ref.epoch) for ref in session.saved]
    assert kinds == [
        ("training", 1),
        ("sampler", 1),
        ("training", 2),
        ("sampler", 2),
        ("training", 3),
        ("training", 4),
    ]
    assert state.checkpoints == session.saved
    assert state.bests == [1, 2, 2, 2]
    assert [e.sampler_ckpt is not None for e in result.epochs] == [True, True, False, False]
    assert result.warnings == []


@pytest.mark.parametrize(
    ("losses", "patience", "min_delta", "reason", "epochs", "best"),
    [
        # A gain under min_delta is still a (strictly) new best, but patience runs out.
        ((1.0, 0.95, 0.5), 1, 0.1, "early_stopped", 2, 2),
        ((1.0, 0.95, 0.93, 0.5), 2, 0.1, "early_stopped", 3, 3),
        # A gain over min_delta resets the patience counter.
        ((1.0, 0.95, 0.85, 0.84, 0.5), 2, 0.1, "completed", 5, 5),
        # An equal loss is not a new best.
        ((1.0, 1.0, 1.0), 2, 0.0, "completed", 3, 1),
        ((1.0, 1.0, 0.5), 1, 0.0, "early_stopped", 2, 1),
        # Running out of patience on the last epoch is just completing.
        ((1.0, 2.0), 1, 0.0, "completed", 2, 1),
    ],
)
def test_strict_best_and_min_delta_patience(
    losses: tuple[float, ...],
    patience: int,
    min_delta: float,
    reason: str,
    epochs: int,
    best: int,
) -> None:
    spec = make_spec(max_epochs=len(losses), patience=patience, min_delta=min_delta)
    with recording(curve(*losses)) as session:
        result = train(session, spec)
    assert result.stopped_reason == reason
    assert len(result.epochs) == epochs
    assert result.best_epoch == best
    assert result.best_val_loss == losses[best - 1]


def test_epoch_metrics_record_steps_tokens_and_losses() -> None:
    rows = row_examples(4)
    options = FakeOptions(loss_curve=[2.0, 1.0], val_curve=[1.5, 1.2])
    with recording(options) as session:
        result = train(session, make_spec(max_epochs=2), rows=rows)
    first, second = result.epochs
    assert (first.epoch, first.steps, first.train_loss, first.val_loss) == (1, 2, 2.0, 1.5)
    assert (second.epoch, second.steps, second.train_loss, second.val_loss) == (2, 2, 1.0, 1.2)
    assert first.train_tokens == sum(row.num_tokens for row in rows)
    assert first.training_ckpt is not None
    assert (first.training_ckpt.epoch, first.training_ckpt.step) == (1, 2)
    assert first.to_json()["training_ckpt"]["name"] == "t1s2"
    assert first.to_json()["sampler_ckpt"]["name"] == "s1s2"


def test_step_events_number_steps_over_the_run() -> None:
    events: list[dict[str, Any]] = []
    with recording(FakeOptions(loss_curve=[2.0, 1.0])) as session:
        train(session, make_spec(max_epochs=2), on_event=events.append)
    steps = [event for event in events if event["type"] == "step"]
    assert [(e["epoch"], e["step"], e["loss"]) for e in steps] == [
        (1, 1, 2.0),
        (1, 2, 2.0),
        (2, 3, 1.0),
        (2, 4, 1.0),
    ]
    assert all(set(e) == {"type", "epoch", "step", "loss", "tokens", "elapsed_s"} for e in steps)
    assert all(e["tokens"] > 0 and e["elapsed_s"] >= 0 for e in steps)
    assert [e["epoch"] for e in events if e["type"] == "epoch"] == [1, 2]


def test_the_step_loss_is_a_token_mean_over_its_microbatches() -> None:
    rows = row_examples(2)
    budget = rows[0].num_tokens  # one row per microbatch
    scripted = [StepMetrics(3.0, 1.0, 5), StepMetrics(1.0, 3.0, 5)]
    events: list[dict[str, Any]] = []
    spec = make_spec(max_epochs=1, microbatch_token_budget=budget)
    with recording(train_metrics=scripted) as session:
        result = train(session, spec, rows=rows, on_event=events.append)
    assert len(session.train_batches) == 2
    [step] = [event for event in events if event["type"] == "step"]
    assert step["loss"] == 1.0  # (3 + 1) / (1 + 3), not the mean of 3.0 and 0.33
    assert result.epochs[0].train_loss == 1.0


@pytest.mark.parametrize(
    "metrics",
    [StepMetrics(1.0, 0.0, 5), StepMetrics(math.nan, 1.0, 5), StepMetrics(1e308, 1e-10, 5)],
    ids=["zero", "nan", "overflowing-mean"],
)
def test_an_unusable_training_loss_fails(metrics: StepMetrics) -> None:
    with recording(train_metrics=[metrics]) as session, pytest.raises(ExternalError):
        train(session, make_spec(max_epochs=1))
    assert session.outstanding == []


def test_empty_validation_selects_by_training_loss_with_a_warning() -> None:
    scripted = [StepMetrics(3.0, 1.0, 5), StepMetrics(1.0, 1.0, 5), StepMetrics(2.0, 1.0, 5)]
    spec = make_spec(max_epochs=3, patience=2)
    with recording(train_metrics=scripted) as session:
        result = train(session, spec, rows=row_examples(2), validation=[])
    assert "forward" not in session.calls
    assert result.warnings == [NO_VALIDATION_WARNING]
    assert [e.val_loss for e in result.epochs] == [None, None, None]
    assert (result.best_epoch, result.best_val_loss) == (2, None)
    assert result.best_sampler_ckpt == result.epochs[1].sampler_ckpt


def test_finalize_runs_last_with_the_open_session() -> None:
    seen: list[tuple[TrainingSession, TrainResult]] = []

    def finalize(session: TrainingSession, result: TrainResult) -> TrainResult:
        seen.append((session, result))
        return dataclasses.replace(result, warnings=[*result.warnings, "finalized"])

    with recording() as session:
        result = train(session, make_spec(max_epochs=1), finalize=finalize)
    [(given, before)] = seen
    assert given is session
    assert result == dataclasses.replace(before, warnings=["finalized"])


def test_training_needs_rows() -> None:
    with recording() as session, pytest.raises(UsageError, match="no training rows"):
        train(session, make_spec(), rows=[])


@pytest.mark.parametrize("bad_kind", ["training", "sampler"])
def test_checkpoint_names_are_checked_before_any_step(bad_kind: str) -> None:
    def namer(kind: str, epoch: int, step: int) -> str:
        return f"run-name-too-long-e{epoch}s{step}" if kind == bad_kind else "ok"

    with recording() as session, pytest.raises(UsageError, match="1 to 17 characters") as info:
        run_training(
            session,
            train_rows=row_examples(2),
            validation_rows=VALIDATION,
            spec=make_spec(),
            state=MemoryState(),
            budget=Budget(),
            checkpoint_namer=namer,
            max_checkpoint_name_len=MAX_CHECKPOINT_NAME_LEN,
        )
    assert info.value.hint is not None and "17 characters" in info.value.hint
    assert session.calls == []


# --- batching ----------------------------------------------------------------------------


def test_the_data_order_is_the_same_across_runs() -> None:
    def batches() -> list[tuple[Any, ...]]:
        with recording() as session:
            train(session, make_spec(max_epochs=2, batch_size=3), rows=row_examples(9))
        return session.train_batches

    first, second = batches(), batches()
    assert first == second
    assert first[:3] != first[3:6]  # epochs are shuffled differently


def test_the_data_order_follows_the_seed() -> None:
    rows = row_examples(9)

    def batches(seed: int) -> list[tuple[Any, ...]]:
        with recording() as session:
            train(session, make_spec(max_epochs=2, batch_size=3, seed=seed), rows=rows)
        return session.train_batches

    expected = [
        tuple(microbatch)
        for epoch in (1, 2)
        for step in plan_batches(
            rows, rows_per_step=3, microbatch_token_budget=None, rng=epoch_rng(7, epoch)
        )
        for microbatch in step
    ]
    assert batches(7) == expected
    assert batches(1) != batches(2)


def test_microbatches_respect_the_token_budget_in_the_loop() -> None:
    rows = row_examples(6)
    budget = 2 * rows[0].num_tokens
    with recording() as session:
        spec = make_spec(max_epochs=1, batch_size=3, microbatch_token_budget=budget)
        train(session, spec, rows=rows)
    assert [len(batch) for batch in session.train_batches] == [2, 1, 2, 1]
    assert session.calls.count("optim_step") == 2


def test_rows_expanded_per_turn_are_batched_together() -> None:
    renderer = FakeRenderer(prefix_stable=False)
    rows = build_examples(chat_rows(3, turns=2), renderer, max_seq_len=4096).examples
    with recording() as session:
        train(session, make_spec(max_epochs=1, batch_size=1), rows=rows)
    assert [len(batch) for batch in session.train_batches] == [2, 2, 2]


# --- pipelining and failures -------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_pipelining_keeps_at_most_depth_steps_in_flight(depth: int) -> None:
    with recording() as session:
        train(session, make_spec(max_epochs=2, batch_size=1, pipeline_depth=depth))
    assert session.max_steps_in_flight == depth
    assert session.outstanding == []


def test_every_result_is_resolved_when_a_step_fails() -> None:
    state = MemoryState()
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=3)
    with (
        recording(fail_forward_backward_at=3) as session,
        pytest.raises(RuntimeError, match="injected failure"),
    ):
        train(session, spec, rows=row_examples(8), state=state)
    assert session.outstanding == []
    assert state.steps == [(1, 1), (1, 2)]
    # Steps 4 and 5 were in flight when step 3's failure surfaced, and were resolved.
    assert len(session.train_batches) == 5
    assert "forward" not in session.calls
    assert session.saved == []


def test_results_in_flight_are_resolved_when_a_submission_raises() -> None:
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=4)
    with (
        recording(raise_forward_backward_at=3) as session,
        pytest.raises(RuntimeError, match="injected failure"),
    ):
        train(session, spec, rows=row_examples(8))
    assert session.outstanding == []
    assert session.resolved.count("optim_step") == 2


def test_results_are_resolved_on_keyboard_interrupt() -> None:
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=3)
    with (
        recording(
            fail_forward_backward_at=1, forward_backward_error=KeyboardInterrupt()
        ) as session,
        pytest.raises(KeyboardInterrupt),
    ):
        train(session, spec, rows=row_examples(5))
    assert session.outstanding == []


def test_an_interrupt_while_draining_still_resolves_everything() -> None:
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=3)
    # Steps 2 and 3 are in flight when step 1 fails; both raise KeyboardInterrupt.
    errors = {2: KeyboardInterrupt(), 3: KeyboardInterrupt()}
    with (
        recording(fail_forward_backward_at=1, forward_backward_errors=errors) as session,
        pytest.raises(KeyboardInterrupt) as info,
    ):
        train(session, spec, rows=row_examples(5))
    assert session.outstanding == []
    assert len(session.train_batches) == 3
    notes = getattr(info.value, "__notes__", [])
    assert any("interrupted while handling: RuntimeError: injected failure" in n for n in notes)
    assert any("step 3 also failed: KeyboardInterrupt" in n for n in notes)


def test_an_interrupt_in_a_step_wins_over_an_ordinary_error() -> None:
    rows = row_examples(2)
    spec = make_spec(max_epochs=1, batch_size=2, microbatch_token_budget=rows[0].num_tokens)
    errors = {2: KeyboardInterrupt()}
    with (
        recording(fail_forward_backward_at=1, forward_backward_errors=errors) as session,
        pytest.raises(KeyboardInterrupt) as info,
    ):
        train(session, spec, rows=rows)
    assert session.outstanding == []
    assert any("RuntimeError: injected failure" in n for n in info.value.__notes__)


def test_a_failing_middle_microbatch_still_resolves_the_step() -> None:
    rows = row_examples(3)
    state = MemoryState()
    spec = make_spec(
        max_epochs=1, batch_size=3, pipeline_depth=1, microbatch_token_budget=rows[0].num_tokens
    )
    with (
        recording(fail_forward_backward_at=2) as session,
        pytest.raises(RuntimeError, match="injected failure"),
    ):
        train(session, spec, rows=rows, state=state)
    assert len(session.train_batches) == 3
    assert session.resolved.count("forward_backward") == 3
    assert session.resolved.count("optim_step") == 1
    assert session.outstanding == []
    assert state.steps == []


@pytest.mark.parametrize("injection", ["fail_forward_backward_at", "raise_forward_backward_at"])
def test_forward_backward_is_never_retried(injection: str) -> None:
    error = TransientError("the connection dropped", source="fake")
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=1)
    options: dict[str, Any] = {injection: 1, "forward_backward_error": error}
    with recording(**options) as session, pytest.raises(TransientError):
        train(session, spec, rows=row_examples(3))
    assert len(session.train_batches) == 1
    assert session.outstanding == []


# --- validation retries ------------------------------------------------------------------


def test_validation_forward_retries_transient_errors_with_backoff() -> None:
    errors = [TransientError("blip", source="fake"), TimeoutError("slow")]
    events: list[dict[str, Any]] = []
    slept: list[float] = []
    with recording(curve(1.25), forward_errors=errors) as session:
        result = train(session, make_spec(max_epochs=1), on_event=events.append, sleep=slept.append)
    assert result.epochs[0].val_loss == 1.25
    assert session.calls.count("forward") == 3
    assert [e["attempt"] for e in events if e["type"] == "retry"] == [1, 2]
    assert slept == [2.0, 4.0]
    assert all(delay <= MAX_RETRY_DELAY_S for delay in slept)
    assert session.outstanding == []


def test_validation_retries_are_bounded() -> None:
    errors = [ConnectionError("down") for _ in range(VALIDATION_RETRIES + 1)]
    with (
        recording(forward_errors=errors) as session,
        pytest.raises(ConnectionError) as info,
    ):
        train(session, make_spec(max_epochs=1))
    assert session.calls.count("forward") == VALIDATION_RETRIES + 1
    assert f"validation forward failed after {VALIDATION_RETRIES + 1} attempts" in (
        info.value.__notes__
    )
    assert session.outstanding == []


def test_the_validation_loss_is_a_token_mean_over_its_microbatches() -> None:
    validation = row_examples(2, prefix="val")
    spec = make_spec(max_epochs=1, microbatch_token_budget=validation[0].num_tokens)
    scripted = [StepMetrics(3.0, 1.0, 5), StepMetrics(1.0, 3.0, 5)]
    with recording(forward_metrics=scripted) as session:
        result = train(session, spec, validation=validation)
    assert len(session.validation_batches) == 2
    assert result.epochs[0].val_loss == 1.0  # (3 + 1) / (1 + 3), not the mean of 3.0 and 0.33


def test_a_validation_error_that_is_not_transient_is_not_retried() -> None:
    with (
        recording(forward_errors=[RuntimeError("bad batch")]) as session,
        pytest.raises(RuntimeError, match="bad batch"),
    ):
        train(session, make_spec(max_epochs=1))
    assert session.calls.count("forward") == 1


# --- budget and cancellation -------------------------------------------------------------


def test_the_token_budget_stops_before_a_step_that_would_exceed_it() -> None:
    rows = row_examples(4)
    step = rows[0].num_tokens
    budget = Budget(max_train_tokens=6 * step + step // 2)
    state = MemoryState()
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=2)
    with recording() as session:
        result = train(session, spec, rows=rows, state=state, budget=budget)
    assert result.stopped_reason == "budget_exhausted"
    assert len(state.steps) == 6
    assert budget.tokens_used == 6 * step
    assert [e.epoch for e in result.epochs] == [1]
    assert result.best_epoch == 1
    # The two steps of epoch 2 are kept in a training checkpoint, without validation.
    assert result.last_training_ckpt is not None
    assert (result.last_training_ckpt.epoch, result.last_training_ckpt.step) == (2, 6)
    assert len(session.validation_batches) == len(VALIDATION)  # epoch 1's validation only
    assert len(result.warnings) == 1
    assert "stopped before step 7" in result.warnings[0]
    assert state.warnings == result.warnings


def test_a_token_budget_below_one_step_trains_nothing() -> None:
    rows = row_examples(2)
    with recording() as session:
        result = train(session, make_spec(), rows=rows, budget=Budget(max_train_tokens=1))
    assert result.stopped_reason == "budget_exhausted"
    assert session.calls == []
    assert (result.epochs, result.best_epoch, result.last_training_ckpt) == ([], None, None)


def test_the_spend_budget_stops_between_steps() -> None:
    clock = Clock()
    # One second in all; training stops at half a second, keeping the rest in reserve.
    budget = Budget(
        max_spend_usd=1.0, hourly_rate_usd=3600.0, spend_reserve_fraction=0.5, clock=clock
    )

    def tick(event: dict[str, Any]) -> None:
        if event["type"] == "step":
            clock.now += 0.4

    state = MemoryState()
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=1)
    with recording() as session:
        result = train(
            session, spec, rows=row_examples(8), state=state, budget=budget, on_event=tick
        )
    assert result.stopped_reason == "budget_exhausted"
    assert len(state.steps) == 2
    assert "reached 0.50 USD, max_spend_usd 1.0 less the 50% kept" in result.warnings[0]
    # The reserve pays for the checkpoint of the steps trained.
    assert [(ref.kind, ref.step) for ref in session.saved] == [("training", 2)]


class PricedSession(RecordingSession):
    """Advances ``clock`` by ``cost`` seconds on every paid call, and records the time."""

    PAID = ("forward_backward", "optim_step", "forward", "save_training_checkpoint")

    def __init__(self, inner: TrainingSession, clock: Clock, cost: float) -> None:
        super().__init__(inner)
        self.clock = clock
        self.cost = cost
        self.paid_at: list[tuple[str, float]] = []

    def _pay(self, call: str) -> None:
        self.paid_at.append((call, self.clock.now))
        self.clock.now += self.cost

    def forward_backward(self, batch: Sequence[Any]) -> Any:
        self._pay("forward_backward")
        return super().forward_backward(batch)

    def optim_step(self, adam: Any) -> Any:
        self._pay("optim_step")
        return super().optim_step(adam)

    def forward(self, batch: Sequence[Any]) -> Any:
        self._pay("forward")
        return super().forward(batch)

    def save_training_checkpoint(self, name: str, *, epoch: int, step: int) -> Any:
        self._pay("save_training_checkpoint")
        return super().save_training_checkpoint(name, epoch=epoch, step=step)

    def save_sampler_checkpoint(self, name: str, *, epoch: int, step: int) -> Any:
        self._pay("save_sampler_checkpoint")
        return super().save_sampler_checkpoint(name, epoch=epoch, step=step)


@contextmanager
def priced(
    clock: Clock, cost: float, options: FakeOptions | None = None
) -> Iterator[PricedSession]:
    with FakeBackend(options).open_session(FAKE_PROFILE, LORA, **SESSION) as inner:
        yield PricedSession(inner, clock, cost)


VALIDATION_2 = row_examples(4, prefix="val")
"""Two validation microbatches at ``batch_size=2``."""


def spend_budget(clock: Clock, *, seconds: float, reserve: float) -> Budget:
    """A spend cap of ``seconds`` of wall-clock time."""
    return Budget(
        max_spend_usd=seconds, hourly_rate_usd=3600.0, spend_reserve_fraction=reserve, clock=clock
    )


@pytest.mark.parametrize("cap", [3.5, 5.5, 7.5, 9.5, 11.5, 13.5])
def test_no_paid_call_starts_past_the_spend_cap(cap: float) -> None:
    clock = Clock()
    budget = spend_budget(clock, seconds=cap, reserve=0.2)
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=1)
    with priced(clock, 1.0) as session:
        result = train(session, spec, rows=row_examples(3), budget=budget)
    assert result.stopped_reason == "budget_exhausted"
    stop_at = cap * 0.8
    final = session.paid_at[-1]
    # Every call but the final checkpoint starts before the reserve; that one before the cap.
    assert all(at < stop_at for _, at in session.paid_at[:-1])
    if final[0] == "save_training_checkpoint" and final[1] >= stop_at:
        assert final[1] < cap
    else:
        assert final[1] < stop_at


def test_a_step_refused_part_way_is_neither_applied_nor_counted() -> None:
    clock = Clock()
    budget = spend_budget(clock, seconds=10.0, reserve=0.5)  # stops at 5 s
    # Two microbatches a step: forward_backward, forward_backward, optim_step = 3 s a step.
    spec = make_spec(max_epochs=1, batch_size=2, pipeline_depth=1)
    rows = row_examples(6)
    spec = spec.model_copy(update={"microbatch_token_budget": rows[0].num_tokens})
    state = MemoryState()
    with priced(clock, 1.0) as session:
        result = train(session, spec, rows=rows, state=state, budget=budget)
    assert result.stopped_reason == "budget_exhausted"
    assert session.calls == [
        "forward_backward",
        "forward_backward",
        "optim_step",
        "forward_backward",
        "forward_backward",
        "save_training_checkpoint",
    ]
    assert state.steps == [(1, 1)]
    assert budget.tokens_used == 2 * rows[0].num_tokens
    assert session.outstanding == []
    assert result.last_training_ckpt is not None
    assert result.last_training_ckpt.step == 1
    assert "stopped during step 2" in result.warnings[0]


def test_a_stop_during_validation_keeps_the_best_epoch_already_saved() -> None:
    clock = Clock()
    # Every call takes 1 s. Epoch 1: 2 steps of 2 calls, 2 validation forwards, a training
    # and a sampler checkpoint: 8 s. Epoch 2's steps end at 12 s and its first validation
    # forward at 13 s, past the 12.6 s at which training stops.
    budget = spend_budget(clock, seconds=14.0, reserve=0.1)
    spec = make_spec(max_epochs=3, batch_size=2, pipeline_depth=1)
    state = MemoryState()
    with priced(clock, 1.0, curve(2.0, 1.0)) as session:
        result = train(
            session, spec, rows=row_examples(4), validation=VALIDATION_2, state=state, budget=budget
        )
    assert result.stopped_reason == "budget_exhausted"
    assert [e.epoch for e in result.epochs] == [1]
    assert state.epochs == result.epochs
    assert result.best_epoch == 1
    assert result.best_sampler_ckpt == result.epochs[0].sampler_ckpt
    assert "stopped before finishing the validation of epoch 2" in result.warnings[0]
    # The steps of epoch 2 are kept in a resumable training checkpoint, paid from the reserve.
    assert result.last_training_ckpt is not None
    assert (result.last_training_ckpt.epoch, result.last_training_ckpt.step) == (2, 4)
    assert [(ref.kind, ref.epoch) for ref in session.saved] == [
        ("training", 1),
        ("sampler", 1),
        ("training", 2),
    ]


def test_a_run_stopped_during_validation_resumes_like_an_unbroken_one() -> None:
    clock = Clock()
    budget = spend_budget(clock, seconds=14.0, reserve=0.1)  # as in the test above
    spec = make_spec(max_epochs=3, batch_size=2, pipeline_depth=1)
    options = curve(2.0, 1.0, 0.5)
    rows = row_examples(4)
    first = MemoryState()
    with priced(clock, 1.0, options) as session:
        stopped = train(
            session, spec, rows=rows, validation=VALIDATION_2, state=first, budget=budget
        )
    assert stopped.stopped_reason == "budget_exhausted"
    point = first.resume_points[-1]
    assert (point.epoch, point.step, point.steps_in_epoch) == (2, 4, 2)
    with FakeBackend(options).open_session(
        FAKE_PROFILE, LORA, resume_from=point.checkpoint, **SESSION
    ) as inner:
        resumed = RecordingSession(inner)
        result = train(
            resumed,
            spec,
            rows=rows,
            validation=VALIDATION_2,
            resume=Resume(point=point, epochs=first.epochs, warnings=first.warnings),
        )
    with recording(options) as unbroken_session:
        unbroken = train(unbroken_session, spec, rows=rows, validation=VALIDATION_2)
    assert _metrics(result) == _metrics(unbroken)
    assert result.stopped_reason == "completed"
    # Epoch 2 is validated and recorded with the stop's checkpoint; only epoch 3 trains.
    assert resumed.train_batches == unbroken_session.train_batches[4:]
    assert result.epochs[1].training_ckpt == point.checkpoint


def test_a_stop_before_the_sampler_checkpoint_leaves_the_epoch_unrecorded() -> None:
    clock = Clock()
    # One step of 2 calls, 2 validation forwards and the training checkpoint: 5 s.
    budget = spend_budget(clock, seconds=5.0, reserve=0.0)
    spec = make_spec(max_epochs=2, batch_size=2, pipeline_depth=1)
    with priced(clock, 1.0) as session:
        result = train(session, spec, rows=row_examples(2), validation=VALIDATION_2, budget=budget)
    assert result.stopped_reason == "budget_exhausted"
    assert (result.epochs, result.best_epoch, result.best_sampler_ckpt) == ([], None, None)
    assert [ref.kind for ref in session.saved] == ["training"]
    assert result.last_training_ckpt == session.saved[0]
    assert "stopped before the sampler checkpoint of epoch 1" in result.warnings[0]


def test_no_final_checkpoint_is_saved_once_the_reserve_is_spent() -> None:
    clock = Clock()
    budget = spend_budget(clock, seconds=1.0, reserve=0.1)

    def slow_steps(event: dict[str, Any]) -> None:
        if event["type"] == "step":
            clock.now += 2.0  # a step that ran past the whole cap

    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=1)
    with recording() as session:
        result = train(session, spec, rows=row_examples(4), budget=budget, on_event=slow_steps)
    assert result.stopped_reason == "budget_exhausted"
    assert session.saved == []
    assert result.last_training_ckpt is None
    assert "did not save a training checkpoint of the 1 step(s)" in result.warnings[-1]
    assert "reached max_spend_usd 1.0" in result.warnings[-1]


def test_cancellation_mid_epoch_saves_a_training_checkpoint() -> None:
    state = MemoryState(cancel_after_steps=2)
    spec = make_spec(max_epochs=2, batch_size=1, pipeline_depth=1)
    with recording() as session:
        result = train(session, spec, rows=row_examples(6), state=state)
    assert result.stopped_reason == "cancelled"
    assert state.steps == [(1, 1), (1, 2)]
    assert "forward" not in session.calls
    assert [(ref.kind, ref.epoch, ref.step) for ref in session.saved] == [("training", 1, 2)]
    assert result.last_training_ckpt == session.saved[0]
    assert (result.epochs, result.best_epoch, result.best_sampler_ckpt) == ([], None, None)


def test_cancellation_drains_the_steps_in_flight() -> None:
    state = MemoryState(cancel_after_steps=2)
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=3)
    with recording() as session:
        result = train(session, spec, rows=row_examples(8), state=state)
    assert result.stopped_reason == "cancelled"
    assert session.outstanding == []
    # Steps 3 and 4 were already in flight when the cancellation was seen.
    assert len(state.steps) == 4
    assert result.last_training_ckpt is not None
    assert result.last_training_ckpt.step == 4


def test_cancellation_at_an_epoch_boundary_keeps_the_epochs_checkpoint() -> None:
    state = MemoryState(cancel_after_steps=2)
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=1)
    with recording() as session:
        result = train(session, spec, rows=row_examples(2), state=state)
    assert result.stopped_reason == "cancelled"
    assert [e.epoch for e in result.epochs] == [1]
    assert result.last_training_ckpt == result.epochs[0].training_ckpt
    assert [ref.kind for ref in session.saved] == ["training", "sampler"]


# --- time-based checkpoints and resume ---------------------------------------------------


def ticking(clock: Clock, minutes: float) -> Any:
    """An event sink that advances ``clock`` by ``minutes`` per finished step."""

    def tick(event: dict[str, Any]) -> None:
        if event["type"] == "step":
            clock.now += minutes * 60

    return tick


def test_time_based_checkpoints_land_on_the_first_step_after_the_interval() -> None:
    clock = Clock()
    events: list[dict[str, Any]] = []

    def sink(event: dict[str, Any]) -> None:
        ticking(clock, 4)(event)
        events.append(event)

    state = MemoryState()
    spec = make_spec(max_epochs=2, batch_size=1, pipeline_depth=1, checkpoint_every_minutes=10)
    with recording() as session:
        train(session, spec, rows=row_examples(4), state=state, clock=clock, on_event=sink)
    training = [(ref.epoch, ref.step) for ref in session.saved if ref.kind == "training"]
    # Steps take 4 minutes: 12 minutes after the start (step 3) and after epoch 1 (step 7).
    assert training == [(1, 3), (1, 4), (2, 7), (2, 8)]
    reasons = [e["reason"] for e in events if e["type"] == "checkpoint" and e["kind"] == "training"]
    assert reasons == ["time", "epoch", "time", "epoch"]
    points = [(p.epoch, p.step, p.steps_in_epoch) for p in state.resume_points]
    assert points == [(1, 3, 3), (1, 4, 4), (2, 7, 3), (2, 8, 4)]
    assert [p.checkpoint for p in state.resume_points] == [
        ref for ref in session.saved if ref.kind == "training"
    ]


class CancelOnCheckpoint(MemoryState):
    """Asks for cancellation as soon as a checkpoint is recorded."""

    def record_checkpoint(self, ref: CheckpointRef, *, resume_point: ResumePoint | None) -> None:
        super().record_checkpoint(ref, resume_point=resume_point)
        self.cancel_after_steps = 0


def test_a_cancel_during_a_timed_checkpoint_stops_before_the_next_step() -> None:
    clock = Clock()
    state = CancelOnCheckpoint()
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=1, checkpoint_every_minutes=10)
    with recording() as session:
        result = train(
            session,
            spec,
            rows=row_examples(6),
            state=state,
            clock=clock,
            on_event=ticking(clock, 4),
        )
    assert result.stopped_reason == "cancelled"
    # The checkpoint before step 4 was being saved when the cancel came: step 4 never ran.
    assert state.steps == [(1, 1), (1, 2), (1, 3)]
    assert session.calls.count("optim_step") == 3
    assert [(ref.kind, ref.step) for ref in session.saved] == [("training", 3)]


def test_a_spend_cap_reached_while_draining_for_a_timed_checkpoint_stops_there() -> None:
    clock = Clock()
    # 1 USD a minute: training stops at 14 USD (20 less the 30% reserve).
    budget = Budget(
        max_spend_usd=20.0, hourly_rate_usd=60.0, spend_reserve_fraction=0.3, clock=clock
    )
    events: list[dict[str, Any]] = []

    def sink(event: dict[str, Any]) -> None:
        ticking(clock, 4)(event)
        events.append(event)

    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=2, checkpoint_every_minutes=10)
    with recording() as session:
        result = train(
            session, spec, rows=row_examples(8), budget=budget, clock=clock, on_event=sink
        )
    assert result.stopped_reason == "budget_exhausted"
    # At 12 minutes the checkpoint is due; draining step 4 takes the run to 16 minutes.
    assert "stopped before a timed checkpoint" in result.warnings[-1]
    assert session.calls.count("optim_step") == 4
    reasons = [(e["reason"], e["step"]) for e in events if e["type"] == "checkpoint"]
    assert reasons == [("stop", 4)]


def test_a_checkpoint_due_at_the_epochs_end_is_saved_before_validation() -> None:
    clock = Clock()
    state = MemoryState()
    events: list[dict[str, Any]] = []

    def sink(event: dict[str, Any]) -> None:
        ticking(clock, 4)(event)
        events.append(event)

    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=1, checkpoint_every_minutes=10)
    broken = ExternalError("validation is down", source="fake")
    with recording(forward_errors=[broken]) as session, pytest.raises(ExternalError):
        train(session, spec, rows=row_examples(3), state=state, clock=clock, on_event=sink)
    # The third step ended 12 minutes in: its checkpoint is kept although validation failed.
    assert [(ref.kind, ref.step) for ref in session.saved] == [("training", 3)]
    assert [(p.step, p.steps_in_epoch) for p in state.resume_points] == [(3, 3)]
    assert [e["reason"] for e in events if e["type"] == "checkpoint"] == ["time"]


def test_a_checkpoint_saved_before_validation_is_the_epochs_checkpoint() -> None:
    clock = Clock()
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=1, checkpoint_every_minutes=10)
    with recording() as session:
        result = train(session, spec, rows=row_examples(3), clock=clock, on_event=ticking(clock, 4))
    assert [(ref.kind, ref.step) for ref in session.saved] == [("training", 3), ("sampler", 3)]
    assert result.epochs[0].training_ckpt == session.saved[0]


def test_a_time_based_checkpoint_waits_for_the_steps_in_flight() -> None:
    clock = Clock()
    state = MemoryState()
    spec = make_spec(max_epochs=1, batch_size=1, pipeline_depth=3, checkpoint_every_minutes=1)
    with recording() as session:
        train(
            session,
            spec,
            rows=row_examples(6),
            state=state,
            clock=clock,
            on_event=ticking(clock, 2),
        )
    assert session.outstanding == []
    # Every checkpoint holds exactly the steps resolved before it.
    for point in state.resume_points:
        assert point.steps_in_epoch == point.step
        assert point.checkpoint.step == point.step
    assert [p.step for p in state.resume_points][-1] == 6


def _interrupted_then_resumed(
    spec: TrainSpec,
    options: FakeOptions,
    *,
    rows: int = 4,
    minutes: float = 4,
    budget: Callable[[], Budget] = Budget,
) -> tuple[TrainResult, TrainResult, MemoryState]:
    """Train until ``options.fail_at_step`` crashes, then resume in a new backend.

    Each of the three runs (crashed, resumed, uninterrupted) gets a new ``budget()``.
    """
    clock = Clock()
    first = MemoryState()
    backend = FakeBackend(options)
    with (
        pytest.raises(Exception, match="fail_at_step"),
        backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session,
    ):
        train(
            session,
            spec,
            rows=row_examples(rows),
            state=first,
            budget=budget(),
            clock=clock,
            on_event=ticking(clock, minutes),
        )
    point = first.resume_points[-1]
    resume = Resume(point=point, epochs=first.epochs, warnings=first.warnings)
    second = MemoryState()
    elsewhere = FakeBackend(options)
    with elsewhere.open_session(
        FAKE_PROFILE, LORA, resume_from=point.checkpoint, **SESSION
    ) as resumed:
        result = train(
            resumed,
            spec,
            rows=row_examples(rows),
            state=second,
            budget=budget(),
            clock=clock,
            on_event=ticking(clock, minutes),
            resume=resume,
        )
    with FakeBackend(options.model_copy(update={"fail_at_step": None})).open_session(
        FAKE_PROFILE, LORA, **SESSION
    ) as session:
        unbroken = train(session, spec, rows=row_examples(rows), budget=budget(), clock=clock)
    return result, unbroken, second


def _metrics(result: TrainResult) -> list[tuple[Any, ...]]:
    return [
        (e.epoch, e.steps, e.train_loss, e.val_loss, e.train_tokens, e.new_best)
        for e in result.epochs
    ]


def test_resuming_mid_epoch_ends_with_the_same_epoch_metrics() -> None:
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=1, checkpoint_every_minutes=10)
    options = FakeOptions(loss_curve=[2.0, 1.5, 1.2], val_curve=[2.1, 1.7, 1.4], fail_at_step=6)
    result, unbroken, second = _interrupted_then_resumed(spec, options)
    assert _metrics(result) == _metrics(unbroken)
    assert result.stopped_reason == unbroken.stopped_reason == "completed"
    # The crash came at step 6, so the run resumed from epoch 1's checkpoint (step 4) and
    # trained steps 5 to 12 only.
    assert second.steps[0] == (2, 5)
    assert second.steps[-1] == (3, 12)
    assert result.best_epoch == 3


def test_resuming_after_a_time_based_checkpoint_skips_exactly_the_trained_steps() -> None:
    spec = make_spec(max_epochs=2, batch_size=1, pipeline_depth=1, checkpoint_every_minutes=10)
    options = FakeOptions(fail_at_step=8)
    rows = row_examples(4)
    clock = Clock()
    first = MemoryState()
    with (
        pytest.raises(Exception, match="fail_at_step"),
        FakeBackend(options).open_session(FAKE_PROFILE, LORA, **SESSION) as inner,
    ):
        train(inner, spec, rows=rows, state=first, clock=clock, on_event=ticking(clock, 4))
    point = first.resume_points[-1]
    assert (point.epoch, point.step, point.steps_in_epoch) == (2, 7, 3)
    with recording() as unbroken_session:
        train(unbroken_session, spec, rows=rows)
    with FakeBackend(options).open_session(
        FAKE_PROFILE, LORA, resume_from=point.checkpoint, **SESSION
    ) as inner:
        resumed = RecordingSession(inner)
        train(
            resumed,
            spec,
            rows=rows,
            resume=Resume(point=point, epochs=first.epochs),
        )
    # Only the last step of epoch 2 is trained again, on the same batch as uninterrupted.
    assert resumed.train_batches == unbroken_session.train_batches[-1:]


def test_resuming_keeps_the_best_epoch_and_the_patience_counter() -> None:
    # Epoch 1 is best; epoch 2 is stale (patience 2 left one), the crash is in epoch 3.
    spec = make_spec(max_epochs=4, batch_size=1, pipeline_depth=1, patience=2)
    options = FakeOptions(val_curve=[1.0, 1.5, 1.6, 0.5], fail_at_step=9)
    result, unbroken, _ = _interrupted_then_resumed(spec, options)
    assert result.stopped_reason == unbroken.stopped_reason == "early_stopped"
    assert [e.epoch for e in result.epochs] == [1, 2, 3]
    assert result.best_epoch == unbroken.best_epoch == 1
    assert result.best_training_ckpt == result.epochs[0].training_ckpt


def test_resuming_between_an_epochs_checkpoint_and_its_record_trains_nothing_again() -> None:
    # Epoch 1's training checkpoint is saved, then its sampler checkpoint fails: the run
    # resumes from a checkpoint that holds the whole epoch, but no record of it.
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=1)
    rows = row_examples(4)
    first = MemoryState()
    with (
        recording(fail_save_sampler_at=1) as broken,
        pytest.raises(RuntimeError, match="injected save failure"),
    ):
        train(broken, spec, rows=rows, state=first)
    point = first.resume_points[-1]
    assert (point.epoch, point.step, point.steps_in_epoch) == (1, 4, 4)
    assert first.epochs == []
    with recording() as unbroken_session:
        unbroken = train(unbroken_session, spec, rows=rows)
    with FakeBackend().open_session(
        FAKE_PROFILE, LORA, resume_from=point.checkpoint, **SESSION
    ) as inner:
        resumed = RecordingSession(inner)
        result = train(resumed, spec, rows=rows, resume=Resume(point=point, epochs=first.epochs))
    assert _metrics(result) == _metrics(unbroken)
    # Epoch 1 is validated and recorded with the checkpoint it already has; only the steps
    # of epochs 2 and 3 are sent.
    assert resumed.train_batches == unbroken_session.train_batches[4:]
    assert result.epochs[0].training_ckpt == point.checkpoint
    saved = [ref.name for ref in resumed.saved]
    assert point.checkpoint.name not in saved
    assert len(saved) == len(set(saved))


def test_the_token_budget_carries_over_a_resume() -> None:
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=1, checkpoint_every_minutes=10)
    step = row_examples(4)[0].num_tokens
    cap = 6 * step + step // 2  # stops before step 7, in epoch 2
    options = FakeOptions(fail_at_step=6)  # crashes before the cap, after epoch 1's checkpoint
    budgets: list[Budget] = []

    def budget() -> Budget:
        budgets.append(Budget(max_train_tokens=cap))
        return budgets[-1]

    result, unbroken, second = _interrupted_then_resumed(spec, options, budget=budget)
    assert result.stopped_reason == unbroken.stopped_reason == "budget_exhausted"
    assert second.steps == [(2, 5), (2, 6)]
    _, resumed_budget, unbroken_budget = budgets
    assert resumed_budget.tokens_used == unbroken_budget.tokens_used == 6 * step
    assert result.warnings == unbroken.warnings
    assert "stopped before step 7" in result.warnings[-1]


def test_the_spend_budget_carries_over_a_resume() -> None:
    # 60 USD/h is 1 USD a minute; steps take 4 minutes. Training stops at 24 USD (30 less
    # the 20% reserve), before step 7, whether or not the run crashed and resumed.
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=1, checkpoint_every_minutes=10)
    options = FakeOptions(fail_at_step=6)  # after epoch 1's checkpoint, at 16 minutes
    rows = row_examples(4)

    def budget(clock: Clock) -> Budget:
        return Budget(
            max_spend_usd=30.0, hourly_rate_usd=60.0, spend_reserve_fraction=0.2, clock=clock
        )

    clock = Clock()
    first = MemoryState()
    with (
        pytest.raises(Exception, match="fail_at_step"),
        FakeBackend(options).open_session(FAKE_PROFILE, LORA, **SESSION) as session,
    ):
        train(
            session,
            spec,
            rows=rows,
            state=first,
            budget=budget(clock),
            clock=clock,
            on_event=ticking(clock, 4),
        )
    point = first.resume_points[-1]
    assert (point.step, point.elapsed_seconds) == (4, 16 * 60)

    later = Clock()  # a new process: its clock does not know the first attempt's time
    second = MemoryState()
    resumed_budget = budget(later)
    with FakeBackend(options).open_session(
        FAKE_PROFILE, LORA, resume_from=point.checkpoint, **SESSION
    ) as session:
        result = train(
            session,
            spec,
            rows=rows,
            state=second,
            budget=resumed_budget,
            clock=later,
            on_event=ticking(later, 4),
            resume=Resume(point=point, epochs=first.epochs, warnings=first.warnings),
        )
    assert result.stopped_reason == "budget_exhausted"
    assert second.steps == [(2, 5), (2, 6)]
    assert resumed_budget.elapsed_seconds == 24 * 60
    assert "stopped before step 7" in result.warnings[-1]
    assert second.resume_points[-1].elapsed_seconds == 24 * 60


def test_resuming_after_the_epoch_that_stops_early_stops_at_once() -> None:
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=1, patience=1)
    with recording(curve(1.0, 2.0)) as session:
        first = MemoryState()
        done = train(session, spec, rows=row_examples(2), state=first)
    assert done.stopped_reason == "early_stopped"
    resume = Resume(point=first.resume_points[-1], epochs=first.epochs)
    with recording(curve(1.0, 2.0)) as session:
        result = train(session, spec, rows=row_examples(2), resume=resume)
    assert result.stopped_reason == "early_stopped"
    assert session.calls == []
    assert _metrics(result) == _metrics(done)


def test_resuming_with_other_rows_is_refused() -> None:
    spec = make_spec(max_epochs=2, batch_size=1, pipeline_depth=1)
    state = MemoryState(cancel_after_steps=3)
    with recording() as session:
        train(session, spec, rows=row_examples(4), state=state)
    resume = Resume(point=state.resume_points[-1], epochs=[])
    with recording() as session, pytest.raises(StateError, match="an epoch has 2") as info:
        train(session, spec, rows=row_examples(2), resume=resume)
    assert info.value.hint is not None and "same training rows" in info.value.hint
    assert session.calls == []


def test_a_resume_point_that_does_not_follow_the_epochs_is_refused() -> None:
    spec = make_spec(max_epochs=3, batch_size=1, pipeline_depth=1)
    state = MemoryState()
    with recording() as session:
        train(session, spec, rows=row_examples(2), state=state)
    early = state.resume_points[0]  # epoch 1, but three epochs finished
    with recording() as session, pytest.raises(StateError, match="does not follow"):
        train(session, spec, rows=row_examples(2), resume=Resume(point=early, epochs=state.epochs))


def test_warnings_are_not_repeated_on_resume() -> None:
    spec = make_spec(max_epochs=2, batch_size=1, pipeline_depth=1)
    state = MemoryState(cancel_after_steps=1)
    with recording() as session:
        first = train(session, spec, rows=row_examples(2), validation=[], state=state)
    resume = Resume(point=state.resume_points[-1], epochs=[], warnings=first.warnings)
    again = MemoryState()
    with recording() as session:
        result = train(
            session, spec, rows=row_examples(2), validation=[], state=again, resume=resume
        )
    assert result.warnings == [NO_VALIDATION_WARNING]
    assert again.warnings == []


# --- finalize, and finite numbers --------------------------------------------------------


def test_a_failing_finalize_keeps_the_training_result() -> None:
    def finalize(session: TrainingSession, result: TrainResult) -> TrainResult:
        raise RuntimeError("promotion broke")

    with recording() as session, pytest.raises(FinalizeFailed, match="promotion broke") as info:
        train(session, make_spec(max_epochs=1), finalize=finalize)
    assert info.value.result.stopped_reason == "completed"
    assert info.value.result.best_epoch == 1
    assert isinstance(info.value.__cause__, RuntimeError)
    assert not info.value.interrupted
    assert info.value.hint is not None and "resume" in info.value.hint


def test_an_interrupted_finalize_is_marked_as_such() -> None:
    def finalize(session: TrainingSession, result: TrainResult) -> TrainResult:
        raise KeyboardInterrupt

    with recording() as session, pytest.raises(FinalizeFailed) as info:
        train(session, make_spec(max_epochs=1), finalize=finalize)
    assert info.value.interrupted


def test_non_finite_numbers_become_null_in_json_and_events() -> None:
    record = EpochMetrics(
        epoch=1,
        steps=1,
        train_loss=math.nan,
        val_loss=math.inf,
        train_tokens=3,
        new_best=True,
        training_ckpt=None,
        sampler_ckpt=None,
    )
    assert (record.to_json()["train_loss"], record.to_json()["val_loss"]) == (None, None)
    assert math.isnan(EpochMetrics.from_json(record.to_json()).train_loss)
    events: list[dict[str, Any]] = []
    with recording() as session:
        train(session, make_spec(max_epochs=1), clock=lambda: math.nan, on_event=events.append)
    assert all(e["elapsed_s"] is None for e in events if "elapsed_s" in e)
