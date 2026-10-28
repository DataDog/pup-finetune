# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Promote-before-exit, as the loop's finalize hook, against the fake provider."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from pup_finetune.errors import FinetuneError
from pup_finetune.providers.base import (
    MAX_MODEL_ID_LEN,
    MODEL_ID_PATTERN,
    LoraConfig,
    ModelRef,
    TrainSpec,
)
from pup_finetune.providers.fake import (
    FAKE_PROFILE,
    MAX_CHECKPOINT_NAME_LEN,
    FakeBackend,
    FakeOptions,
)
from pup_finetune.training import (
    NOTHING_TO_PROMOTE,
    Budget,
    FinalizeFailed,
    Promotion,
    Resume,
    TrainResult,
    run_training,
)
from pup_finetune.training.promotion import (
    ADOPT_CLOCK_ALLOWANCE,
    PromotionRequest,
    model_id_for_checkpoint,
)
from tests.fakes.training import SESSION, MemoryState, RecordingSession, row_examples

LORA = LoraConfig(rank=8)


def namer(kind: str, epoch: int, step: int) -> str:
    return f"{kind[0]}{epoch}s{step}"


def promotion(*, supports_promotion: bool = True) -> Promotion:
    return Promotion(
        model_id="ft-run-1",
        supports_promotion=supports_promotion,
        checkpoint_namer=namer,
        max_checkpoint_name_len=MAX_CHECKPOINT_NAME_LEN,
    )


def train(
    backend: FakeBackend,
    spec: TrainSpec,
    *,
    state: MemoryState | None = None,
    budget: Budget | None = None,
    finalize: Promotion | None = None,
    resume: Resume | None = None,
) -> TrainResult:
    resume_from = None if resume is None else resume.point.checkpoint
    with backend.open_session(FAKE_PROFILE, LORA, resume_from=resume_from, **SESSION) as session:
        return run_training(
            session,
            train_rows=row_examples(2),
            validation_rows=row_examples(2, prefix="val"),
            spec=spec,
            state=state or MemoryState(),
            budget=budget or Budget(),
            checkpoint_namer=namer,
            max_checkpoint_name_len=MAX_CHECKPOINT_NAME_LEN,
            finalize=finalize if finalize is not None else promotion(),
            resume=resume,
        )


def spec(**overrides: Any) -> TrainSpec:
    fields: dict[str, Any] = {"max_epochs": 3, "batch_size": 1, "pipeline_depth": 1}
    return TrainSpec(**(fields | overrides))


@pytest.mark.parametrize(
    ("val_curve", "patience", "cancel_after", "budget_steps", "reason", "best"),
    [
        ([2.0, 1.5, 1.0], 1, None, None, "completed", 3),
        ([1.0, 2.0], 1, None, None, "early_stopped", 1),
        ([2.0, 1.5], 1, 3, None, "cancelled", 1),
        ([2.0, 1.5], 1, None, 3, "budget_exhausted", 1),
    ],
    ids=["completed", "early_stopped", "cancelled", "budget_exhausted"],
)
def test_the_best_checkpoint_is_promoted_before_the_session_ends(
    val_curve: list[float],
    patience: int,
    cancel_after: int | None,
    budget_steps: int | None,
    reason: str,
    best: int,
) -> None:
    backend = FakeBackend(FakeOptions(val_curve=val_curve))
    budget = None
    if budget_steps is not None:  # epoch 1 has two steps, so this stops in epoch 2
        budget = Budget(max_train_tokens=budget_steps * row_examples(1)[0].num_tokens)
    state = MemoryState(cancel_after_steps=cancel_after)
    result = train(backend, spec(patience=patience), state=state, budget=budget)
    assert result.stopped_reason == reason
    assert result.best_epoch == best
    assert result.deployable_ckpt == result.best_sampler_ckpt
    assert result.promoted_model == ModelRef(
        provider="fake", model_id="ft-run-1", uri="fake://models/ft-run-1"
    )
    assert backend.promoted == [result.promoted_model]
    # Promoted while its session was open (the fake refuses otherwise).
    assert backend.closed == backend.opened


def test_a_best_epoch_from_an_ended_session_is_saved_again_and_promoted() -> None:
    options = FakeOptions(val_curve=[1.0, 2.0, 3.0])
    run_spec = spec(patience=5)
    first = MemoryState(cancel_after_steps=3)  # stops in epoch 2, after epoch 1 was best
    cancelled = train(FakeBackend(options), run_spec, state=first, finalize=promotion())
    assert cancelled.best_epoch == 1
    resume = Resume(point=first.resume_points[-1], epochs=first.epochs)
    backend = FakeBackend(options)  # a new process: epoch 1's session has ended
    result = train(backend, run_spec, resume=resume)
    assert result.best_epoch == 1
    old = result.best_sampler_ckpt
    assert old is not None and old.created_in_session != backend.opened[-1]
    recreated = result.deployable_ckpt
    assert recreated is not None and recreated != old
    assert (recreated.kind, recreated.name, recreated.epoch) == ("sampler", "r1s2", 1)
    assert recreated.created_in_session == backend.opened[-1]
    assert result.promoted_model is not None
    assert backend.promoted == [result.promoted_model]
    # The new sampler checkpoint holds epoch 1's weights (the fake counts epochs from 0).
    assert parse_qs(urlsplit(recreated.uri).query)["trained"] == ["0"]
    assert parse_qs(urlsplit(result.epochs[-1].training_ckpt.uri).query)["trained"] == ["2"]  # type: ignore[union-attr]


def test_a_spend_guard_can_refuse_re_creating_a_sampler_checkpoint() -> None:
    options = FakeOptions(val_curve=[1.0, 2.0, 3.0])
    run_spec = spec(patience=5)
    first = MemoryState(cancel_after_steps=3)
    train(FakeBackend(options), run_spec, state=first, finalize=promotion())
    resume = Resume(point=first.resume_points[-1], epochs=first.epochs)
    backend = FakeBackend(options)
    refusing = dataclasses.replace(promotion(), spend_guard=lambda: "the reserve is spent")
    with pytest.raises(FinalizeFailed, match="the reserve is spent") as info:
        train(backend, run_spec, resume=resume, finalize=refusing)
    assert info.value.result.best_epoch == 1
    assert backend.promoted == []
    assert [ref.kind for ref in backend.saved if ref.kind == "sampler"] == []


def test_a_spend_guard_is_not_asked_when_nothing_is_re_created() -> None:
    asked: list[bool] = []

    def guard() -> str | None:
        asked.append(True)
        return "refused"

    backend = FakeBackend(FakeOptions(val_curve=[2.0, 1.0]))
    finalize = dataclasses.replace(promotion(), spend_guard=guard)
    result = train(backend, spec(), finalize=finalize)
    assert result.promoted_model is not None
    assert asked == []


class Requests:
    """:class:`~pup_finetune.training.promotion.PromotionRequests` in memory, made at ``now``."""

    def __init__(self, now: datetime) -> None:
        self.now = now
        self.recorded: list[PromotionRequest] = []

    def promotion_request(self, epoch: int, step: int) -> PromotionRequest | None:
        return next((r for r in self.recorded if (r.epoch, r.step) == (epoch, step)), None)

    def any_promotion_requested(self) -> bool:
        return bool(self.recorded)

    def record_promotion_request(self, model_id: str, *, epoch: int, step: int) -> PromotionRequest:
        request = PromotionRequest(model_id=model_id, epoch=epoch, step=step, requested_at=self.now)
        self.recorded.append(request)
        return request


def promote_with(finalize: Promotion, *, epochs: int = 1) -> tuple[str, list[datetime | None]]:
    """The promoted model id of a run of ``epochs`` (each one best), and its ``adopt_since``."""
    backend = FakeBackend(FakeOptions(val_curve=[3.0, 2.0, 1.0]))
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as inner:
        session = RecordingSession(inner)
        result = run_training(
            session,
            train_rows=row_examples(2),
            validation_rows=row_examples(2, prefix="val"),
            spec=spec(max_epochs=epochs),
            state=MemoryState(),
            budget=Budget(),
            checkpoint_namer=namer,
            max_checkpoint_name_len=MAX_CHECKPOINT_NAME_LEN,
            finalize=finalize,
        )
    assert result.promoted_model is not None
    return result.promoted_model.model_id, session.adopt_since


def test_a_retry_adopts_only_models_created_since_the_run_asked_for_that_checkpoint() -> None:
    asked = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    requests = Requests(asked)
    finalize = dataclasses.replace(promotion(), requests=requests)
    # The attempt that records the request adopts nothing: no model can be the run's yet.
    assert promote_with(finalize) == ("ft-run-1", [None])
    # Promoting the same checkpoint again (a retry) reuses its recorded request.
    requests.now = asked + timedelta(hours=1)
    assert promote_with(finalize) == ("ft-run-1", [asked - ADOPT_CLOCK_ALLOWANCE])
    assert [(r.model_id, r.epoch, r.step) for r in requests.recorded] == [("ft-run-1", 1, 2)]
    # Without a recorded request, no existing model can be told to be the run's.
    assert promote_with(promotion()) == ("ft-run-1", [None])


def test_a_new_best_after_a_promotion_gets_its_own_model_id() -> None:
    # The run's model id holds epoch 1: epoch 3, a later best, must not be adopted as it.
    asked = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    requests = Requests(asked)
    finalize = dataclasses.replace(promotion(), requests=requests)
    assert promote_with(finalize)[0] == "ft-run-1"
    requests.now = later = asked + timedelta(hours=1)
    assert promote_with(finalize, epochs=3) == ("ft-run-1-e3s6", [None])
    assert [(r.model_id, r.epoch, r.step) for r in requests.recorded] == [
        ("ft-run-1", 1, 2),
        ("ft-run-1-e3s6", 3, 6),
    ]
    # Asked again, epoch 3 keeps its own id and request.
    assert promote_with(finalize, epochs=3) == (
        "ft-run-1-e3s6",
        [later - ADOPT_CLOCK_ALLOWANCE],
    )


def test_a_checkpoint_model_id_fits_the_model_id_rules() -> None:
    assert model_id_for_checkpoint("ft-run-1", 3, 12) == "ft-run-1-e3s12"
    longest = "a" * MAX_MODEL_ID_LEN
    derived = model_id_for_checkpoint(longest, 3, 12)
    assert len(derived) == MAX_MODEL_ID_LEN
    assert MODEL_ID_PATTERN.fullmatch(derived)
    assert derived.endswith("-e3s12")
    # Two long ids that share the kept prefix stay apart.
    assert derived != model_id_for_checkpoint(longest[:-1] + "b", 3, 12)
    assert MODEL_ID_PATTERN.fullmatch(model_id_for_checkpoint("a" * 40 + "-" * 22 + "b", 3, 12))


def test_without_promotion_the_best_sampler_checkpoint_is_the_deployable_one() -> None:
    backend = FakeBackend(FakeOptions(val_curve=[2.0, 1.0]), supports_promotion=False)
    result = train(backend, spec(max_epochs=2), finalize=promotion(supports_promotion=False))
    assert result.promoted_model is None
    assert result.deployable_ckpt == result.best_sampler_ckpt == result.epochs[1].sampler_ckpt
    assert backend.promoted == []


def test_with_no_finished_epoch_there_is_nothing_to_promote() -> None:
    backend = FakeBackend()
    result = train(backend, spec(), state=MemoryState(cancel_after_steps=1))
    assert (result.stopped_reason, result.best_epoch) == ("cancelled", None)
    assert (result.promoted_model, result.deployable_ckpt) == (None, None)
    assert result.warnings == [NOTHING_TO_PROMOTE]
    assert backend.promoted == []


def test_a_failed_promotion_keeps_the_result() -> None:
    backend = FakeBackend(fail_promotions=1)
    with pytest.raises(FinalizeFailed, match="fail_promotions") as info:
        train(backend, spec(max_epochs=1))
    assert info.value.result.best_epoch == 1
    assert info.value.result.promoted_model is None
    assert isinstance(info.value.__cause__, FinetuneError)
    assert backend.closed == backend.opened  # the session still ended


def test_a_promotion_reports_the_epoch_it_promotes_and_the_model() -> None:
    events: list[dict[str, Any]] = []
    backend = FakeBackend(FakeOptions(val_curve=[2.0, 1.5, 1.0]))
    result = train(
        backend, spec(), finalize=dataclasses.replace(promotion(), on_event=events.append)
    )
    assert result.best_epoch == 3
    assert events == [
        {
            "type": "promoting",
            "epoch": 3,
            "val_loss": 1.0,
            "train_loss": result.epochs[2].train_loss,
            "model_id": "ft-run-1",
        },
        {"type": "promoted", "model_id": "ft-run-1", "epoch": 3},
    ]
