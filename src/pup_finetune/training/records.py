# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What the training loop reports and resumes from: epochs, results and resume points.

Each record converts to JSON with ``to_json`` (non-finite losses become ``null``, so the
run directory and stdout only ever hold finite numbers) and the ones the run directory reads
back have a ``from_json`` that raises :class:`~pup_finetune.errors.StateError` on bad input.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from pup_finetune.errors import StateError
from pup_finetune.providers.base import CheckpointRef, ModelRef
from pup_finetune.storage import finite_json

StoppedReason = Literal["completed", "early_stopped", "budget_exhausted", "cancelled"]


def checkpoint_from_json(data: Any) -> CheckpointRef | None:
    """A :class:`CheckpointRef` from its ``dataclasses.asdict`` form; ``None`` stays ``None``."""
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise StateError(f"expected a checkpoint object, got {type(data).__name__}")
    try:
        return CheckpointRef(**data)
    except TypeError as exc:
        raise StateError(f"invalid checkpoint record ({exc})") from exc


def _asdict(ref: CheckpointRef | ModelRef | None) -> dict[str, Any] | None:
    return None if ref is None else dataclasses.asdict(ref)


def _required(data: Mapping[str, Any], key: str, kind: type[Any]) -> Any:
    value = data.get(key)
    # bool is an int; a JSON true is never a count or a loss.
    if isinstance(value, bool) and kind is not bool:
        value = None
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind):
        raise StateError(f"invalid {key!r} in a training record: {value!r}")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumePoint:
    """A training checkpoint and where training stood when it was saved.

    ``step`` counts optimizer steps over the run and ``steps_in_epoch`` those of ``epoch``
    already trained. ``epoch_loss_sum``, ``epoch_weight`` and ``epoch_tokens`` accumulate
    those steps' metrics; ``train_tokens`` counts the run's trained tokens and
    ``elapsed_seconds`` the wall-clock seconds its spend guard counted over every attempt so
    far, so a resumed run's budget carries on from them.
    """

    checkpoint: CheckpointRef
    epoch: int
    step: int
    steps_in_epoch: int
    epoch_loss_sum: float
    epoch_weight: float
    epoch_tokens: int
    train_tokens: int
    elapsed_seconds: float

    def follows(self, epochs: Sequence[EpochMetrics]) -> bool:
        """Whether a run can continue from here after the finished ``epochs``: the checkpoint
        was saved at the last one's end, after every step they recorded, or during the next
        one."""
        if self.epoch == len(epochs):
            return self.step == sum(epoch.steps for epoch in epochs)
        return self.epoch == len(epochs) + 1

    def to_json(self) -> dict[str, Any]:
        return finite_json(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ResumePoint:
        checkpoint = checkpoint_from_json(data.get("checkpoint"))
        if checkpoint is None or checkpoint.kind != "training":
            raise StateError("a resume point needs a training checkpoint")
        return cls(
            checkpoint=checkpoint,
            epoch=_required(data, "epoch", int),
            step=_required(data, "step", int),
            steps_in_epoch=_required(data, "steps_in_epoch", int),
            epoch_loss_sum=_required(data, "epoch_loss_sum", float),
            epoch_weight=_required(data, "epoch_weight", float),
            epoch_tokens=_required(data, "epoch_tokens", int),
            train_tokens=_required(data, "train_tokens", int),
            elapsed_seconds=_required(data, "elapsed_seconds", float),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EpochMetrics:
    """One finished epoch. ``val_loss`` is ``None`` when the validation split is empty.

    ``training_ckpt`` is the training checkpoint saved at the epoch's end.
    """

    epoch: int
    steps: int
    train_loss: float
    val_loss: float | None
    train_tokens: int
    new_best: bool
    training_ckpt: CheckpointRef | None
    sampler_ckpt: CheckpointRef | None

    def to_json(self) -> dict[str, Any]:
        """The record as JSON; a non-finite loss becomes ``None``."""
        return finite_json(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> EpochMetrics:
        """Read back :meth:`to_json` (a ``None`` training loss reads as NaN)."""
        val_loss = data.get("val_loss")
        train_loss = data.get("train_loss")
        return cls(
            epoch=_required(data, "epoch", int),
            steps=_required(data, "steps", int),
            train_loss=math.nan if train_loss is None else _required(data, "train_loss", float),
            val_loss=None if val_loss is None else _required(data, "val_loss", float),
            train_tokens=_required(data, "train_tokens", int),
            new_best=_required(data, "new_best", bool),
            training_ckpt=checkpoint_from_json(data.get("training_ckpt")),
            sampler_ckpt=checkpoint_from_json(data.get("sampler_ckpt")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainResult:
    """How training ended and which checkpoints it left.

    ``promoted_model`` and ``deployable_ckpt`` are filled in by a ``finalize`` hook such as
    :class:`~pup_finetune.training.promotion.Promotion`: the model the best checkpoint was
    promoted to, and the sampler checkpoint that serves as the run's deployable artifact.
    """

    best_epoch: int | None
    best_val_loss: float | None
    stopped_reason: StoppedReason
    epochs: list[EpochMetrics]
    best_sampler_ckpt: CheckpointRef | None
    best_training_ckpt: CheckpointRef | None
    last_training_ckpt: CheckpointRef | None
    warnings: list[str]
    promoted_model: ModelRef | None = None
    deployable_ckpt: CheckpointRef | None = None

    def to_json(self) -> dict[str, Any]:
        """The result as ``result.json`` records it; a non-finite loss becomes ``None``."""
        return finite_json(
            {
                "stopped_reason": self.stopped_reason,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "promoted_model": _asdict(self.promoted_model),
                "deployable_checkpoint": _asdict(self.deployable_ckpt),
                "checkpoints": {
                    "best_sampler": _asdict(self.best_sampler_ckpt),
                    "best_training": _asdict(self.best_training_ckpt),
                    "last_training": _asdict(self.last_training_ckpt),
                },
                "epochs": [epoch.to_json() for epoch in self.epochs],
                "warnings": list(self.warnings),
            }
        )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TrainResult:
        """The training result :meth:`to_json` recorded, without promotion (for a retry).

        Raises :class:`StateError` when ``data`` is not a finished training's result.
        """
        reason = data.get("stopped_reason")
        if reason not in get_args(StoppedReason):
            raise StateError(f"result.json: training did not end ({reason!r}), nothing to finalize")
        checkpoints = data.get("checkpoints")
        epochs = data.get("epochs")
        if not isinstance(checkpoints, Mapping) or not isinstance(epochs, list):
            raise StateError("result.json: expected checkpoints and epochs")
        best_val_loss = data.get("best_val_loss")
        try:
            return cls(
                best_epoch=None
                if data.get("best_epoch") is None
                else _required(data, "best_epoch", int),
                best_val_loss=None
                if best_val_loss is None
                else _required(data, "best_val_loss", float),
                stopped_reason=cast(StoppedReason, reason),
                epochs=[EpochMetrics.from_json(epoch) for epoch in epochs],
                best_sampler_ckpt=checkpoint_from_json(checkpoints.get("best_sampler")),
                best_training_ckpt=checkpoint_from_json(checkpoints.get("best_training")),
                last_training_ckpt=checkpoint_from_json(checkpoints.get("last_training")),
                warnings=[str(warning) for warning in data.get("warnings") or []],
            )
        except StateError as exc:
            raise StateError(f"result.json: {exc}") from exc
        except (AttributeError, TypeError, ValueError) as exc:
            raise StateError(f"result.json: invalid training result ({exc})") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class Resume:
    """What a resumed :func:`run_training` continues from.

    ``point`` is the training checkpoint the session was opened from; ``epochs`` are the
    epochs finished before it (best tracking and patience are replayed from them), and
    ``warnings`` the run's warnings so far. ``elapsed_seconds`` is the wall-clock time the
    spend guard counted over every earlier attempt, as last recorded (e.g. in ``state.json``),
    which includes time spent after ``point``: the budget resumes from the larger of it and
    ``point.elapsed_seconds``, so time spent after the last checkpoint is not forgotten.
    """

    point: ResumePoint
    epochs: Sequence[EpochMetrics]
    warnings: Sequence[str] = ()
    elapsed_seconds: float = 0.0
