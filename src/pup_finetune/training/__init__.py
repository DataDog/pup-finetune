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
  :class:`~pup_finetune.providers.base.TrainingSession`.

The run directory the loop reports into is :mod:`pup_finetune.runs`.
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
    MAX_RETRY_DELAY_S,
    NO_VALIDATION_WARNING,
    VALIDATION_RETRIES,
    CheckpointKind,
    CheckpointNamer,
    EpochMetrics,
    EventSink,
    Finalize,
    LoopState,
    StoppedReason,
    TrainResult,
    is_transient,
    run_training,
)

__all__ = [
    "MAX_RETRY_DELAY_S",
    "NO_VALIDATION_WARNING",
    "VALIDATION_RETRIES",
    "Budget",
    "BuildResult",
    "CheckpointKind",
    "CheckpointNamer",
    "EpochMetrics",
    "EventSink",
    "Finalize",
    "LoopState",
    "Microbatch",
    "RejectedRow",
    "RejectionCode",
    "RowExamples",
    "Step",
    "StoppedReason",
    "TrainResult",
    "build_examples",
    "epoch_rng",
    "is_transient",
    "pack_microbatches",
    "plan_batches",
    "provider_row",
    "run_training",
    "step_tokens",
]
