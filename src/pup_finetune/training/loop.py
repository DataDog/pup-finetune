# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The provider-neutral LoRA SFT loop: epochs, validation, checkpoints and early stopping."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pup_finetune import output
from pup_finetune.errors import ExternalError
from pup_finetune.rendering.types import TokenDatum
from pup_finetune.training.backend import StepResult, TrainingBackend
from pup_finetune.training.batching import (
    TrainingDataError,
    accumulation_groups,
    pack_microbatches,
)
from pup_finetune.training.budget import BudgetExceeded, BudgetGuard
from pup_finetune.training.settings import SFTSettings

_MAX_NAME = 54
_HASHED_PREFIX = 45


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One finished epoch, as written to ``epochs.json``."""

    epoch: int
    train_loss: float
    validation_loss: float
    optimizer_steps: int
    resumable_state_uri: str
    sampler_weights_uri: str | None
    """Set only on a new best epoch."""
    new_best: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class LoopResult:
    """How the loop ended and which checkpoints it left."""

    status: Literal["completed", "budget_stopped"]
    epochs: list[EpochRecord]
    best_epoch: int | None
    best_sampler_weights_uri: str | None
    last_resumable_state_uri: str | None
    stopped_early: bool


def checkpoint_name(run_id: str, label: str) -> str:
    """A checkpoint name safe for providers: lowercase ``[a-z0-9-]``, at most 54 characters."""
    name = re.sub(r"[^a-z0-9]+", "-", f"{run_id}-{label}".lower()).strip("-")
    if len(name) <= _MAX_NAME:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name[:_HASHED_PREFIX].rstrip('-')}-{digest}"


def _checked(result: StepResult, operation: str) -> StepResult:
    loss, active = result.loss, result.active_tokens
    if not (math.isfinite(loss) and math.isfinite(active) and active > 0):
        raise ExternalError(
            f"the {operation} step returned loss {loss} over {active} active tokens",
            source="training",
        )
    return result


def _mean_loss(results: Sequence[StepResult]) -> float:
    total = sum(result.active_tokens for result in results)
    return sum(result.loss * result.active_tokens for result in results) / total


def _train_epoch(
    backend: TrainingBackend,
    train_rows: Sequence[Sequence[TokenDatum]],
    settings: SFTSettings,
    *,
    epoch: int,
    token_budget: int,
    budget: BudgetGuard,
) -> tuple[float, int]:
    rows = list(train_rows)
    random.Random(settings.seed + epoch - 1).shuffle(rows)  # noqa: S311 - reproducible, not secret
    datums = [datum for row in rows for datum in row]
    groups = accumulation_groups(pack_microbatches(datums, token_budget), settings.batch_size)
    results: list[StepResult] = []
    for step, group in enumerate(groups, start=1):
        step_results: list[StepResult] = []
        for microbatch in group:
            budget.check("a training step")
            step_results.append(_checked(backend.forward_backward(microbatch), "training"))
        budget.check("an optimizer step")
        backend.optim_step(settings.learning_rate)
        results.extend(step_results)
        output.info(
            f"epoch {epoch}/{settings.max_epochs} step {step}/{len(groups)}: "
            f"train loss {_mean_loss(step_results):.4f}"
        )
    return _mean_loss(results), len(groups)


def _validate(
    backend: TrainingBackend, batches: Sequence[Sequence[TokenDatum]], budget: BudgetGuard
) -> float:
    results: list[StepResult] = []
    for batch in batches:
        budget.check("a validation step")
        results.append(_checked(backend.forward(batch), "validation"))
    return _mean_loss(results)


def run_sft(
    backend: TrainingBackend,
    train_rows: Sequence[Sequence[TokenDatum]],
    validation_batches: Sequence[Sequence[TokenDatum]],
    settings: SFTSettings,
    *,
    token_budget: int,
    run_id: str,
    budget: BudgetGuard,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> LoopResult:
    """Train for up to ``settings.max_epochs`` epochs and keep the lowest-validation-loss weights.

    ``train_rows`` holds each row's datums, kept together when rows are shuffled each epoch.
    The resumable state is saved every epoch and the sampler weights on each strictly better
    validation loss. Early stopping uses the patience and ``min_delta`` settings. A reached
    spend guard ends the loop with status ``budget_stopped``; other errors propagate.
    """
    if not any(train_rows) or not any(validation_batches):
        raise TrainingDataError("training needs at least one training and one validation datum")
    epochs: list[EpochRecord] = []
    best: EpochRecord | None = None
    patience_loss = math.inf
    stale_epochs = 0
    status: Literal["completed", "budget_stopped"] = "completed"
    stopped_early = False
    try:
        for epoch in range(1, settings.max_epochs + 1):
            train_loss, steps = _train_epoch(
                backend, train_rows, settings, epoch=epoch, token_budget=token_budget, budget=budget
            )
            validation_loss = _validate(backend, validation_batches, budget)
            new_best = best is None or validation_loss < best.validation_loss
            budget.check("saving the epoch state")
            state_uri = backend.save_state(checkpoint_name(run_id, f"state-epoch-{epoch}"))
            sampler_uri: str | None = None
            if new_best:
                # Stopping here leaves the epoch unrecorded, so a recorded best always has weights.
                budget.check("saving the best weights")
                sampler_uri = backend.save_sampler_weights(
                    checkpoint_name(run_id, f"sampler-epoch-{epoch}")
                )
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                optimizer_steps=steps,
                resumable_state_uri=state_uri,
                sampler_weights_uri=sampler_uri,
                new_best=new_best,
            )
            epochs.append(record)
            if new_best:
                best = record
            if validation_loss < patience_loss - settings.min_delta:
                patience_loss, stale_epochs = validation_loss, 0
            else:
                stale_epochs += 1
            output.info(
                f"epoch {epoch}/{settings.max_epochs}: train loss {train_loss:.4f}, "
                f"validation loss {validation_loss:.4f}" + (" (new best)" if new_best else "")
            )
            if on_epoch is not None:
                on_epoch(record)
            if stale_epochs >= settings.early_stopping_patience and epoch < settings.max_epochs:
                output.info(
                    "stopping early: validation loss stopped improving "
                    f"(patience {settings.early_stopping_patience})"
                )
                stopped_early = True
                break
    except BudgetExceeded as exc:
        output.info(f"stopping: {exc}")
        status = "budget_stopped"
    return LoopResult(
        status=status,
        epochs=epochs,
        best_epoch=best.epoch if best else None,
        best_sampler_weights_uri=best.sampler_weights_uri if best else None,
        last_resumable_state_uri=epochs[-1].resumable_state_uri if epochs else None,
        stopped_early=stopped_early,
    )
