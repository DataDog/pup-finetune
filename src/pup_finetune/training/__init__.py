# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The shared, provider-neutral SFT loop: examples, batching, the budget and the loop itself.

- :mod:`~pup_finetune.training.datums` renders rows into examples and rejects the rows that
  cannot train.
- :mod:`~pup_finetune.training.batching` shuffles rows per epoch and packs optimizer steps
  into microbatches.
- :mod:`~pup_finetune.training.budget` caps trained tokens and wall-clock spend.
- :mod:`~pup_finetune.training.loop` runs epochs against a
  :class:`~pup_finetune.providers.base.TrainingSession`, checkpointing as it goes, and
  resumes at a step; :mod:`~pup_finetune.training.records` holds what it reports.
- :mod:`~pup_finetune.training.promotion` promotes the best checkpoint before the session
  ends.

The run directory the loop reports into is :mod:`pup_finetune.runs`;
:mod:`pup_finetune.training.runner` trains and resumes in one,
:mod:`pup_finetune.training.inputs` renders a run's rows and
:mod:`pup_finetune.training.worker` runs it in the foreground or a detached worker (none is
re-exported here, since they import :mod:`pup_finetune.runs`, which imports this package).
"""

from __future__ import annotations

from pup_finetune.training.batching import (
    Microbatch,
    Step,
    epoch_rng,
    pack_microbatches,
    plan_batches,
    step_tokens,
)
from pup_finetune.training.budget import Budget
from pup_finetune.training.datums import (
    BuildResult,
    RejectedRow,
    RejectionCode,
    RowExamples,
    build_examples,
    provider_row,
)
from pup_finetune.training.loop import (
    CHECKPOINT_KINDS,
    MAX_RETRY_DELAY_S,
    NO_VALIDATION_WARNING,
    VALIDATION_RETRIES,
    CheckpointKind,
    CheckpointNamer,
    CheckpointReason,
    EventSink,
    Finalize,
    FinalizeFailed,
    LoopState,
    checked_checkpoint_name,
    is_transient,
    run_training,
)
from pup_finetune.training.promotion import NO_SAMPLER_CHECKPOINT, NOTHING_TO_PROMOTE, Promotion
from pup_finetune.training.records import (
    EpochMetrics,
    Resume,
    ResumePoint,
    StoppedReason,
    TrainResult,
    checkpoint_from_json,
)

__all__ = [
    "CHECKPOINT_KINDS",
    "MAX_RETRY_DELAY_S",
    "NOTHING_TO_PROMOTE",
    "NO_SAMPLER_CHECKPOINT",
    "NO_VALIDATION_WARNING",
    "VALIDATION_RETRIES",
    "Budget",
    "BuildResult",
    "CheckpointKind",
    "CheckpointNamer",
    "CheckpointReason",
    "EpochMetrics",
    "EventSink",
    "Finalize",
    "FinalizeFailed",
    "LoopState",
    "Microbatch",
    "Promotion",
    "RejectedRow",
    "RejectionCode",
    "Resume",
    "ResumePoint",
    "RowExamples",
    "Step",
    "StoppedReason",
    "TrainResult",
    "build_examples",
    "checked_checkpoint_name",
    "checkpoint_from_json",
    "epoch_rng",
    "is_transient",
    "pack_microbatches",
    "plan_batches",
    "provider_row",
    "run_training",
    "step_tokens",
]
