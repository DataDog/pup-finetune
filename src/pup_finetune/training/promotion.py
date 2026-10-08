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

Before it first asks to promote a checkpoint a run records the request (:class:`PromotionRequest`:
the model id, the checkpoint's epoch and step, and when; saved in ``state.json`` and kept
across resumes), and every promotion of that checkpoint passes that time, less
:data:`ADOPT_CLOCK_ALLOWANCE` for the provider's clock, as ``adopt_since``. So a model an
earlier attempt promoted, whose response was lost, is adopted rather than promoted again
under a taken id, while a model under the same id that existed before the run first asked
(another run directory reusing the run id, say) is refused.

A provider can only tell a model is the run's, not which weights it holds, so a model id is
only ever reused for the checkpoint it was requested for. The run's first promotion uses the
run's model id; when a resumed run finds a new best epoch after an earlier one was promoted
(a run cancelled and resumed, say), the new best is promoted under an id of its own
(:func:`model_id_for_checkpoint`), never adopted as the model holding the earlier best.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from pup_finetune.errors import ExternalError, StateError
from pup_finetune.providers.base import MAX_MODEL_ID_LEN, CheckpointRef, TrainingSession
from pup_finetune.storage import finite_json
from pup_finetune.training.loop import CheckpointNamer, EventSink, checked_checkpoint_name
from pup_finetune.training.records import TrainResult

NOTHING_TO_PROMOTE = "no epoch finished, so there is no checkpoint to promote"
NO_SAMPLER_CHECKPOINT = "the best epoch has no sampler checkpoint to deploy"
ADOPT_CLOCK_ALLOWANCE = timedelta(seconds=60)
"""How much earlier than the recorded request a provider's creation time may read and still be
the run's: the provider's clock and this machine's can disagree."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PromotionRequest:
    """A run's request to promote the checkpoint at ``epoch`` and ``step`` to ``model_id``,
    first made at ``requested_at``."""

    model_id: str
    epoch: int
    step: int
    requested_at: datetime


class PromotionRequests(Protocol):
    """Where a run records its promotion requests (:class:`~pup_finetune.runs.RunState`)."""

    def promotion_request(self, epoch: int, step: int) -> PromotionRequest | None:
        """The recorded request to promote the checkpoint at ``epoch`` and ``step``, if any."""
        ...

    def any_promotion_requested(self) -> bool:
        """Whether the run recorded a request to promote any checkpoint."""
        ...

    def record_promotion_request(self, model_id: str, *, epoch: int, step: int) -> PromotionRequest:
        """Record, and save, a request to promote that checkpoint to ``model_id``, made now."""
        ...


def model_id_for_checkpoint(model_id: str, epoch: int, step: int) -> str:
    """The id to promote the checkpoint at ``epoch`` and ``step`` to once the run's model id
    ``model_id`` holds another one: ``model_id`` with ``-e<epoch>s<step>``.

    It is at most :data:`~pup_finetune.providers.base.MAX_MODEL_ID_LEN` characters: a
    ``model_id`` too long for the suffix is shortened and ends with a dash and 8 hex digits of
    its hash, so two model ids never map to the same one.
    """
    suffix = f"-e{epoch}s{step}"
    if len(model_id) + len(suffix) <= MAX_MODEL_ID_LEN:
        return f"{model_id}{suffix}"
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
    keep = MAX_MODEL_ID_LEN - len(suffix) - len(digest) - 1
    return f"{model_id[:keep].rstrip('-')}-{digest}{suffix}"


@dataclass(frozen=True, slots=True, kw_only=True)
class Promotion:
    """The ``finalize`` hook that promotes the best epoch; see the module docstring.

    ``model_id`` names the promoted model (see :func:`pup_finetune.runs.model_id_for_run`);
    ``supports_promotion`` is the backend's
    :attr:`~pup_finetune.providers.base.TrainingBackend.supports_promotion`. A re-created
    sampler checkpoint is named by ``checkpoint_namer`` (kind ``recreated``), at most
    ``max_checkpoint_name_len`` characters. ``spend_guard``, when given, says why a
    re-creation must not be paid for (e.g. the training budget, reserve included, is spent).
    ``requests`` records the run's promotion requests (see the module docstring): right
    before :meth:`~pup_finetune.providers.base.TrainingSession.promote` the request for the
    best checkpoint is looked up, or recorded under ``model_id`` (the run's first) or
    :func:`model_id_for_checkpoint` (a later one), and ``promote`` gets its model id and its
    time less :data:`ADOPT_CLOCK_ALLOWANCE` as ``adopt_since``, so that a model this run
    already promoted from that checkpoint is adopted. Without it the checkpoint is promoted
    to ``model_id`` and no existing model is adopted. The returned result carries
    ``promoted_model`` (``None`` without promotion) and ``deployable_ckpt``. ``on_event``
    (the loop's event sink) gets ``promoting`` (the epoch and its losses) right before the
    promotion and ``promoted`` (the model id) after it.
    """

    model_id: str
    supports_promotion: bool
    checkpoint_namer: CheckpointNamer
    max_checkpoint_name_len: int
    spend_guard: Callable[[], str | None] | None = None
    requests: PromotionRequests | None = None
    on_event: EventSink | None = None

    def __call__(self, session: TrainingSession, result: TrainResult) -> TrainResult:
        if not self.supports_promotion or result.best_epoch is None:
            return self.record_deployable(result)
        best = result.best_sampler_ckpt
        if best is None or best.created_in_session != session.session_id:
            best = self._recreate(session, result)
        model_id, since = self.model_id, None
        if self.requests is not None:
            request = self._request(self.requests, best)
            model_id, since = request.model_id, request.requested_at - ADOPT_CLOCK_ALLOWANCE
        self._emit(
            {
                "type": "promoting",
                "epoch": result.best_epoch,
                "val_loss": result.best_val_loss,
                "train_loss": _train_loss(result),
                "model_id": model_id,
            }
        )
        model = session.promote(best, model_id, adopt_since=since)
        self._emit({"type": "promoted", "model_id": model.model_id, "epoch": result.best_epoch})
        return dataclasses.replace(result, promoted_model=model, deployable_ckpt=best)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(finite_json(event))

    def _request(self, requests: PromotionRequests, best: CheckpointRef) -> PromotionRequest:
        """The recorded request to promote ``best``, recorded first if there is none."""
        request = requests.promotion_request(best.epoch, best.step)
        if request is not None:
            return request
        model_id = self.model_id
        if requests.any_promotion_requested():
            # The run's model id may already hold an earlier best: never reuse it for this one.
            model_id = model_id_for_checkpoint(self.model_id, best.epoch, best.step)
        return requests.record_promotion_request(model_id, epoch=best.epoch, step=best.step)

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


def _train_loss(result: TrainResult) -> float | None:
    """The best epoch's training loss, which picks it when there is no validation loss."""
    return next(
        (epoch.train_loss for epoch in result.epochs if epoch.epoch == result.best_epoch), None
    )


def _warned(result: TrainResult, message: str) -> TrainResult:
    if message in result.warnings:
        return result
    return dataclasses.replace(result, warnings=[*result.warnings, message])
