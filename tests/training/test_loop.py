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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
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
    RowExamples,
    TrainResult,
    build_examples,
    epoch_rng,
    plan_batches,
    run_training,
)
from tests.fakes.training import MemoryState, RecordingSession, chat_rows, row_examples

LORA = LoraConfig(rank=8)
VALIDATION = row_examples(2, prefix="val")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@contextmanager
def recording(options: FakeOptions | None = None, **injected: Any) -> Iterator[RecordingSession]:
    with FakeBackend(options).open_session(FAKE_PROFILE, LORA) as inner:
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
    "metrics", [StepMetrics(1.0, 0.0, 5), StepMetrics(math.nan, 1.0, 5)], ids=["zero", "nan"]
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
    budget = Budget(max_spend_usd=1.0, hourly_rate_usd=3600.0, clock=clock)  # one second

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
    assert len(state.steps) == 3
    assert "reached max_spend_usd 1.0" in result.warnings[0]


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
