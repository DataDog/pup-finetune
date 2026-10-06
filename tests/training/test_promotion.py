# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Promote-before-exit, as the loop's finalize hook, against the fake provider."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from pup_finetune.errors import FinetuneError
from pup_finetune.providers.base import LoraConfig, ModelRef, TrainSpec
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
from tests.fakes.training import SESSION, MemoryState, row_examples

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
