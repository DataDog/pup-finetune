# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Promote-before-exit: turn the best checkpoint into a deployable model while a session lives.

On some providers a sampler checkpoint can only be promoted while the session that saved it
is open, and the session ends with the process. So :class:`Promotion`, the loop's
``finalize`` hook, promotes the best epoch before the session closes, however training
stopped (completed, early stop, budget, cancel):

- when the best sampler checkpoint was saved in this session, it is promoted as is;
- otherwise (the run was resumed, so it came from an earlier, ended session) the best
  training checkpoint's weights are loaded and saved as a new sampler checkpoint, which is
  promoted instead;
- on a backend without promotion, the best sampler checkpoint is recorded as the run's
  deployable artifact.

Re-creating a sampler checkpoint is a paid save, so when a ``spend_guard`` refuses it the
promotion fails (and the run stays resumable to retry only the promotion).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

from pup_finetune.errors import ExternalError, StateError
from pup_finetune.providers.base import CheckpointRef, TrainingSession
from pup_finetune.training.loop import CheckpointNamer, checked_checkpoint_name
from pup_finetune.training.records import TrainResult

NOTHING_TO_PROMOTE = "no epoch finished, so there is no checkpoint to promote"
NO_SAMPLER_CHECKPOINT = "the best epoch has no sampler checkpoint to deploy"


@dataclass(frozen=True, slots=True, kw_only=True)
class Promotion:
    """The ``finalize`` hook that promotes the best epoch; see the module docstring.

    ``model_id`` names the promoted model (see :func:`pup_finetune.runs.model_id_for_run`);
    ``supports_promotion`` is the backend's
    :attr:`~pup_finetune.providers.base.TrainingBackend.supports_promotion`. A re-created
    sampler checkpoint is named by ``checkpoint_namer`` (kind ``recreated``), at most
    ``max_checkpoint_name_len`` characters. ``spend_guard``, when given, says why a
    re-creation must not be paid for (e.g. the training budget, reserve included, is spent).
    The returned result carries ``promoted_model`` (``None`` without promotion) and
    ``deployable_ckpt``.
    """

    model_id: str
    supports_promotion: bool
    checkpoint_namer: CheckpointNamer
    max_checkpoint_name_len: int
    spend_guard: Callable[[], str | None] | None = None

    def __call__(self, session: TrainingSession, result: TrainResult) -> TrainResult:
        if not self.supports_promotion or result.best_epoch is None:
            return self.record_deployable(result)
        best = result.best_sampler_ckpt
        if best is None or best.created_in_session != session.session_id:
            best = self._recreate(session, result)
        model = session.promote(best, self.model_id)
        return dataclasses.replace(result, promoted_model=model, deployable_ckpt=best)

    def record_deployable(self, result: TrainResult) -> TrainResult:
        """Without promotion: the best sampler checkpoint is what to deploy. Needs no session."""
        if result.best_epoch is None:
            return _warned(result, NOTHING_TO_PROMOTE)
        if result.best_sampler_ckpt is None:
            return _warned(result, NO_SAMPLER_CHECKPOINT)
        return dataclasses.replace(result, deployable_ckpt=result.best_sampler_ckpt)

    def _recreate(self, session: TrainingSession, result: TrainResult) -> CheckpointRef:
        """A sampler checkpoint of the best epoch, saved in ``session``."""
        training = result.best_training_ckpt
        if training is None:
            raise StateError(
                f"epoch {result.best_epoch} has no training checkpoint to re-create its "
                "sampler checkpoint from"
            )
        name = checked_checkpoint_name(
            self.checkpoint_namer,
            "recreated",
            training.epoch,
            training.step,
            limit=self.max_checkpoint_name_len,
        )
        why = None if self.spend_guard is None else self.spend_guard()
        if why is not None:
            raise ExternalError(
                f"did not re-create epoch {result.best_epoch}'s sampler checkpoint to "
                f"promote it: {why}",
                source="budget",
            )
        session.load_weights(training)
        return session.save_sampler_checkpoint(name, epoch=training.epoch, step=training.step)


def _warned(result: TrainResult, message: str) -> TrainResult:
    if message in result.warnings:
        return result
    return dataclasses.replace(result, warnings=[*result.warnings, message])
