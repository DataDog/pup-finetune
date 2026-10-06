# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training: the shared, provider-neutral SFT loop, its settings, batching, budget and run files."""

from __future__ import annotations

from pup_finetune.training.backend import StepResult, TrainingBackend
from pup_finetune.training.batching import (
    TrainingDataError,
    accumulation_groups,
    datum_tokens,
    pack_microbatches,
)
from pup_finetune.training.budget import BudgetExceeded, BudgetGuard
from pup_finetune.training.loop import EpochRecord, LoopResult, checkpoint_name, run_sft
from pup_finetune.training.run import EPOCHS_FILE, PLAN_FILE, RESULT_FILE, STATE_FILE, RunDir
from pup_finetune.training.settings import SFTSettings

__all__ = [
    "EPOCHS_FILE",
    "PLAN_FILE",
    "RESULT_FILE",
    "STATE_FILE",
    "BudgetExceeded",
    "BudgetGuard",
    "EpochRecord",
    "LoopResult",
    "RunDir",
    "SFTSettings",
    "StepResult",
    "TrainingBackend",
    "TrainingDataError",
    "accumulation_groups",
    "checkpoint_name",
    "datum_tokens",
    "pack_microbatches",
    "run_sft",
]
