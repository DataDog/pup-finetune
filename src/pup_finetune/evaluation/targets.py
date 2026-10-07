# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The replay's two targets: the base model and the run's best checkpoint, as samplers.

Both come from one :class:`~pup_finetune.providers.base.TrainingSession`:
``base`` is ``session.sampler(None)`` and ``tuned`` samples a sampler checkpoint of the
run's best epoch. :func:`session_samplers` takes them from a session that is already open
(``train --evaluate``, before the training session closes): the best epoch's sampler
checkpoint is used as is when this session saved it. Otherwise, as in a standalone
``evaluate`` (:func:`run_samplers`, which opens a session for the run), the best *training*
checkpoint's weights are loaded and saved as a new sampler checkpoint named
:func:`tuned_checkpoint_name`. Every sampler is closed, and the session ended, on exit; a
sampler that fails to close is noted on stderr.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from pup_finetune.errors import StateError
from pup_finetune.output import info
from pup_finetune.providers.base import CheckpointRef, Provider, Sampler, TrainingSession
from pup_finetune.redaction import redact
from pup_finetune.runs import RunDir
from pup_finetune.training.records import TrainResult

__all__ = [
    "TUNED_CHECKPOINT_PREFIX",
    "EvalTargets",
    "best_training_checkpoint",
    "run_samplers",
    "session_samplers",
    "tuned_checkpoint_name",
]

TUNED_CHECKPOINT_PREFIX: Final = "ev"
_MAX_DIGEST: Final = 12


@dataclass(frozen=True, slots=True)
class EvalTargets:
    """The samplers by target (``base``, ``tuned``) and the sampler checkpoint ``tuned`` uses."""

    samplers: dict[str, Sampler]
    tuned_checkpoint: CheckpointRef


def best_training_checkpoint(result: TrainResult) -> CheckpointRef:
    """The best epoch's training checkpoint: the weights the tuned target evaluates."""
    if result.best_epoch is None or result.best_training_ckpt is None:
        raise StateError(
            "the run has no best epoch with a training checkpoint to evaluate",
            hint="evaluate a run that finished at least one epoch",
        )
    return result.best_training_ckpt


def tuned_checkpoint_name(run_id: str, training: CheckpointRef, *, limit: int) -> str:
    """``ev`` plus hex digits of a hash of the run and ``training``, at most ``limit`` chars.

    Deterministic, so every evaluation of the same weights names its sampler checkpoint
    alike.
    """
    length = min(_MAX_DIGEST, limit - len(TUNED_CHECKPOINT_PREFIX))
    if length < 1:
        raise ValueError(f"checkpoint names of at most {limit} characters leave no room")
    digest = hashlib.sha256(f"{run_id}\n{training.uri}".encode()).hexdigest()
    return TUNED_CHECKPOINT_PREFIX + digest[:length]


@contextmanager
def session_samplers(
    session: TrainingSession,
    result: TrainResult,
    *,
    run_id: str,
    max_checkpoint_name_len: int,
) -> Iterator[EvalTargets]:
    """The base and tuned samplers of ``session`` for ``result``'s best epoch; closed on exit."""
    training = best_training_checkpoint(result)
    samplers: dict[str, Sampler] = {}
    try:
        samplers["base"] = session.sampler(None)
        best = result.best_sampler_ckpt
        if best is None or best.created_in_session != session.session_id:
            name = tuned_checkpoint_name(run_id, training, limit=max_checkpoint_name_len)
            session.load_weights(training)
            best = session.save_sampler_checkpoint(name, epoch=training.epoch, step=training.step)
        samplers["tuned"] = session.sampler(best)
        yield EvalTargets(samplers=dict(samplers), tuned_checkpoint=best)
    finally:
        for target, sampler in samplers.items():
            try:
                sampler.close()
            except Exception as exc:  # the samplers' work is done: only note it
                detail = redact(str(exc))
                info(
                    f"closing the {target} sampler failed: {type(exc).__name__}"
                    f"{': ' + detail if detail else ''}"
                )


@contextmanager
def run_samplers(provider: Provider, run_dir: RunDir, result: TrainResult) -> Iterator[EvalTargets]:
    """Open a session for the run in ``run_dir`` and yield :func:`session_samplers` of it.

    The session uses the run's model profile, LoRA settings, seed and provider options, and
    ends on exit.
    """
    profile = run_dir.profile()
    spec = run_dir.spec()
    provider.validate_options(spec.provider_options)
    backend = provider.backend()
    with (
        backend.open_session(
            profile,
            spec.lora_config(profile),
            seed=spec.seed,
            max_seq_len=profile.max_seq_len,
        ) as session,
        session_samplers(
            session,
            result,
            run_id=run_dir.run_id,
            max_checkpoint_name_len=backend.max_checkpoint_name_len,
        ) as targets,
    ):
        yield targets
