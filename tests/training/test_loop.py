# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from pup_finetune.errors import ExternalError
from pup_finetune.rendering import TokenDatum
from pup_finetune.training import (
    BudgetGuard,
    EpochRecord,
    LoopResult,
    SFTSettings,
    TrainingDataError,
    checkpoint_name,
    run_sft,
)
from tests.fakes.training import FakeBackend

RUN_ID = "sft-test"
EPOCH_CALLS = ["forward_backward", "optim_step", "forward", "save_state", "save_sampler_weights"]


def datum(tag: int, *, length: int = 4, active: int = 1) -> TokenDatum:
    return TokenDatum(
        tokens=tuple(range(tag * 100, tag * 100 + length)),
        weights=(0,) * (length - active) + (1,) * active,
        message_index=tag,
    )


def rows(count: int) -> list[list[TokenDatum]]:
    return [[datum(i)] for i in range(count)]


VALIDATION = [[datum(900), datum(901)]]


def settings(**overrides: Any) -> SFTSettings:
    return SFTSettings(**({"max_epochs": 2, "batch_size": 2} | overrides))


def disabled_budget() -> BudgetGuard:
    return BudgetGuard(max_spend_usd=None, hourly_rate_usd=None, reserve_fraction=0.1)


def guard_allowing(backend: FakeBackend, calls: int) -> BudgetGuard:
    """A started guard whose clock is the backend call count, so exactly ``calls`` calls fit."""
    guard = BudgetGuard(
        max_spend_usd=calls - 0.5,
        hourly_rate_usd=3600.0,
        reserve_fraction=0.0,
        clock=lambda: float(len(backend.calls)),
    )
    guard.start()
    return guard


def run(
    backend: FakeBackend,
    sft: SFTSettings,
    *,
    train: Sequence[Sequence[TokenDatum]] | None = None,
    validation: Sequence[Sequence[TokenDatum]] = VALIDATION,
    budget: BudgetGuard | None = None,
    token_budget: int = 64,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> LoopResult:
    return run_sft(
        backend,
        rows(2) if train is None else train,
        validation,
        sft,
        token_budget=token_budget,
        run_id=RUN_ID,
        budget=budget or disabled_budget(),
        on_epoch=on_epoch,
    )


def state(epoch: int) -> str:
    return f"state://{RUN_ID}-state-epoch-{epoch}"


def sampler(epoch: int) -> str:
    return f"sampler://{RUN_ID}-sampler-epoch-{epoch}"


def test_completed_run_records_epochs_and_saves_checkpoints() -> None:
    backend = FakeBackend(train_losses=[2.0, 1.5], validation_losses=[1.0, 0.8])
    seen: list[EpochRecord] = []
    result = run(backend, settings(learning_rate=3e-4), on_epoch=seen.append)
    assert result == LoopResult(
        status="completed",
        epochs=[
            EpochRecord(1, 2.0, 1.0, 1, state(1), sampler(1), True),
            EpochRecord(2, 1.5, 0.8, 1, state(2), sampler(2), True),
        ],
        best_epoch=2,
        best_sampler_weights_uri=sampler(2),
        last_resumable_state_uri=state(2),
        stopped_early=False,
    )
    assert seen == result.epochs
    assert backend.methods == EPOCH_CALLS * 2
    assert backend.learning_rates == [3e-4, 3e-4]
    assert backend.state_names == [f"{RUN_ID}-state-epoch-1", f"{RUN_ID}-state-epoch-2"]


def test_epoch_record_as_dict() -> None:
    assert EpochRecord(3, 0.5, 0.25, 4, "s", None, False).as_dict() == {
        "epoch": 3,
        "train_loss": 0.5,
        "validation_loss": 0.25,
        "optimizer_steps": 4,
        "resumable_state_uri": "s",
        "sampler_weights_uri": None,
        "new_best": False,
    }


def test_equal_validation_loss_is_not_a_new_best() -> None:
    backend = FakeBackend(validation_losses=[1.0, 1.0])
    result = run(backend, settings(early_stopping_patience=2))
    assert [e.new_best for e in result.epochs] == [True, False]
    assert result.epochs[1].sampler_weights_uri is None
    assert (result.best_epoch, result.best_sampler_weights_uri) == (1, sampler(1))
    assert result.last_resumable_state_uri == state(2)
    assert backend.sampler_names == [f"{RUN_ID}-sampler-epoch-1"]


def test_strict_best_ignores_min_delta_while_patience_uses_it() -> None:
    backend = FakeBackend(validation_losses=[1.0, 0.95])
    result = run(backend, settings(max_epochs=3, min_delta=0.1))
    assert result.stopped_early
    assert [e.new_best for e in result.epochs] == [True, True]
    assert (result.best_epoch, result.best_sampler_weights_uri) == (2, sampler(2))


@pytest.mark.parametrize(
    ("losses", "min_delta", "patience", "max_epochs", "epochs", "stopped_early", "best"),
    [
        ([3.0, 2.0, 1.0], 0.0, 1, 3, 3, False, 3),
        ([1.0, 1.1, 1.2, 0.5], 0.0, 2, 5, 3, True, 1),
        ([1.0, 1.1], 0.0, 1, 2, 2, False, 1),
        ([1.0, 1.1, 0.9, 1.0, 1.0], 0.0, 2, 5, 5, False, 3),
        ([1.0, 0.95, 0.89, 0.85], 0.1, 2, 4, 4, False, 4),
    ],
    ids=[
        "always-improving",
        "patience-runs-out",
        "patience-runs-out-on-the-last-epoch",
        "improvement-resets-the-counter",
        "baseline-moves-only-on-a-min-delta-improvement",
    ],
)
def test_early_stopping(
    losses: list[float],
    min_delta: float,
    patience: int,
    max_epochs: int,
    epochs: int,
    stopped_early: bool,
    best: int,
) -> None:
    result = run(
        FakeBackend(validation_losses=losses),
        settings(max_epochs=max_epochs, min_delta=min_delta, early_stopping_patience=patience),
    )
    assert result.status == "completed"
    assert len(result.epochs) == epochs
    assert result.stopped_early is stopped_early
    assert result.best_epoch == best


@pytest.mark.parametrize(
    ("allowed", "recorded"), [(1, 0), (2, 0), (3, 0), (4, 0), (5, 1)], ids=EPOCH_CALLS
)
def test_budget_stops_before_the_next_operation(allowed: int, recorded: int) -> None:
    backend = FakeBackend(validation_losses=[1.0, 0.5])
    result = run(backend, settings(), budget=guard_allowing(backend, allowed))
    assert result.status == "budget_stopped"
    assert backend.methods == (EPOCH_CALLS * 2)[:allowed]
    assert len(result.epochs) == recorded
    assert result.last_resumable_state_uri == (state(1) if recorded else None)
    assert result.best_epoch == (1 if recorded else None)
    assert result.best_sampler_weights_uri == (sampler(1) if recorded else None)
    assert not result.stopped_early


def test_budget_stop_between_saves_drops_the_improving_epoch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend()
    seen: list[EpochRecord] = []
    result = run(backend, settings(), budget=guard_allowing(backend, 4), on_epoch=seen.append)
    assert backend.state_names == [f"{RUN_ID}-state-epoch-1"]
    assert (result.epochs, seen, result.last_resumable_state_uri) == ([], [], None)
    assert capsys.readouterr().err.splitlines()[-1] == (
        "stopping: the spend guard's 4 s active-time limit was reached before saving the best "
        "weights"
    )


def test_budget_stop_after_a_non_improving_epoch_keeps_it() -> None:
    backend = FakeBackend(validation_losses=[1.0, 2.0])
    sft = settings(max_epochs=3, early_stopping_patience=2)
    result = run(backend, sft, budget=guard_allowing(backend, 9))
    assert result.status == "budget_stopped"
    assert [e.new_best for e in result.epochs] == [True, False]
    assert result.last_resumable_state_uri == state(2)
    assert (result.best_epoch, result.best_sampler_weights_uri) == (1, sampler(1))


def test_budget_reached_after_the_last_save_still_completes() -> None:
    backend = FakeBackend()
    result = run(backend, settings(max_epochs=1), budget=guard_allowing(backend, 5))
    assert result.status == "completed"
    assert len(result.epochs) == 1


@pytest.mark.parametrize(
    ("backend", "operation"),
    [
        (FakeBackend(train_losses=[math.nan]), "training"),
        (FakeBackend(validation_losses=[math.inf]), "validation"),
        (FakeBackend(active_tokens=0.0), "training"),
        (FakeBackend(active_tokens=math.nan), "training"),
    ],
    ids=["nan-train-loss", "inf-validation-loss", "no-active-tokens", "nan-active-tokens"],
)
def test_invalid_step_results_fail_the_run(backend: FakeBackend, operation: str) -> None:
    with pytest.raises(ExternalError, match=f"the {operation} step returned loss") as info:
        run(backend, settings())
    assert info.value.source == "training"
    assert backend.state_names == []


def test_backend_errors_propagate() -> None:
    backend = FakeBackend()
    backend.fail_on("save_state", error=ExternalError("save failed", source="baseten"))
    seen: list[EpochRecord] = []
    with pytest.raises(ExternalError, match="save failed"):
        run(backend, settings(), on_epoch=seen.append)
    assert seen == []
    assert backend.sampler_names == []


@pytest.mark.parametrize(
    ("train", "validation"),
    [([], VALIDATION), ([[]], VALIDATION), (rows(1), []), (rows(1), [[]])],
    ids=["no-train-rows", "no-train-datums", "no-validation-batches", "empty-validation-batch"],
)
def test_needs_training_and_validation_datums(
    train: list[list[TokenDatum]], validation: list[list[TokenDatum]]
) -> None:
    backend = FakeBackend()
    with pytest.raises(TrainingDataError, match="at least one training and one validation"):
        run(backend, settings(), train=train, validation=validation)
    assert backend.calls == []


def test_groups_count_datums_and_take_one_optimizer_step_each() -> None:
    train = [[datum(10 * r + k) for k in range(n)] for r, n in enumerate([1, 2, 1, 3, 1])]
    backend = FakeBackend()
    result = run(backend, settings(max_epochs=1, batch_size=3), train=train, token_budget=6)
    fb, opt = "forward_backward", "optim_step"
    assert backend.methods[:8] == [fb, fb, opt, fb, fb, opt, fb, opt]
    assert [len(batch) for batch in backend.train_batches] == [2, 1, 1, 2, 2]
    assert result.epochs[0].optimizer_steps == 3


def test_rows_shuffle_by_seed_and_epoch_and_keep_their_datums_together() -> None:
    train = [[datum(10 * r + k) for k in range(r % 3 + 1)] for r in range(8)]
    backend = FakeBackend()
    run(backend, settings(max_epochs=3, seed=7, early_stopping_patience=3), train=train)
    orders = []
    for epoch in (1, 2, 3):
        shuffled = list(train)
        random.Random(7 + epoch - 1).shuffle(shuffled)  # noqa: S311
        trained = [
            d
            for call in backend.calls
            if call.method == "forward_backward" and call.epoch == epoch
            for d in call.arg
        ]
        assert trained == [d for row in shuffled for d in row]
        orders.append(trained)
    assert orders[0] != orders[1]


def test_epoch_losses_are_weighted_by_active_tokens() -> None:
    train = [[datum(1, length=5, active=1)], [datum(2, length=5, active=3)]]
    validation = [[datum(90, length=5, active=1)], [datum(91, length=5, active=3)]]
    backend = FakeBackend(train_losses=[(1.0, 5.0)], validation_losses=[(2.0, 4.0)])
    result = run(backend, settings(max_epochs=1, batch_size=1), train=train, validation=validation)
    active = [sum(d.weights) for batch in backend.train_batches for d in batch]
    expected = (1.0 * active[0] + 5.0 * active[1]) / sum(active)
    assert result.epochs[0].train_loss == pytest.approx(expected)
    assert result.epochs[0].validation_loss == pytest.approx((2.0 * 1 + 4.0 * 3) / 4)
    assert result.epochs[0].optimizer_steps == 2


def test_validation_reuses_the_precomputed_batches_every_epoch() -> None:
    validation = [[datum(90)], [datum(91), datum(92)]]
    backend = FakeBackend(validation_losses=[1.0, 0.5])
    run(backend, settings(), validation=validation)
    assert backend.validation_batches == [tuple(batch) for batch in validation] * 2


def test_progress_lines(capsys: pytest.CaptureFixture[str]) -> None:
    run(FakeBackend(validation_losses=[1.0, 1.25]), settings(max_epochs=3))
    out, err = capsys.readouterr()
    assert out == ""
    assert err.splitlines() == [
        "epoch 1/3 step 1/1: train loss 1.0000",
        "epoch 1/3: train loss 1.0000, validation loss 1.0000 (new best)",
        "epoch 2/3 step 1/1: train loss 1.0000",
        "epoch 2/3: train loss 1.0000, validation loss 1.2500",
        "stopping early: validation loss stopped improving (patience 1)",
    ]


def test_checkpoint_name_normalizes_to_lowercase_dashes() -> None:
    assert checkpoint_name("SFT_Run.01", "state-epoch-1") == "sft-run-01-state-epoch-1"
    assert checkpoint_name("--run--", "Sampler Epoch 2!") == "run-sampler-epoch-2"


def test_checkpoint_name_keeps_names_up_to_54_characters() -> None:
    name = checkpoint_name("a" * 40, "state-epoch-1")
    assert name == "a" * 40 + "-state-epoch-1"
    assert len(name) == 54


def test_checkpoint_name_hashes_longer_names() -> None:
    full = "a" * 41 + "-state-epoch-1"
    digest = hashlib.sha256(full.encode()).hexdigest()[:8]
    assert checkpoint_name("a" * 41, "state-epoch-1") == f"{full[:45]}-{digest}"
    assert len(checkpoint_name("a" * 41, "state-epoch-1")) == 54


def test_checkpoint_name_never_doubles_the_dash_before_the_hash() -> None:
    full = "a" * 44 + "-state-epoch-1"
    digest = hashlib.sha256(full.encode()).hexdigest()[:8]
    assert checkpoint_name("a" * 44, "state-epoch-1") == "a" * 44 + f"-{digest}"


def test_checkpoint_name_hash_tells_long_run_ids_apart() -> None:
    first, second = (checkpoint_name("r" * 60 + s, "state-epoch-1") for s in ("x", "y"))
    assert first != second
