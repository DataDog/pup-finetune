# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The trainer surface the SFT loop drives; each provider implements it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pup_finetune.rendering.types import TokenDatum


@dataclass(frozen=True, slots=True)
class StepResult:
    """One forward pass over a batch, as the backend reports it (checked by the loop)."""

    loss: float
    """Mean loss per active (weighted) token in the batch; finite."""
    active_tokens: float
    """Weighted target tokens in the batch; > 0."""


class TrainingBackend(Protocol):
    """A remote LoRA trainer. Datums are unshifted; the backend shifts them for its API."""

    def forward_backward(self, batch: Sequence[TokenDatum]) -> StepResult:
        """Accumulate gradients over ``batch``. Never retried: a retry would double them."""
        ...

    def forward(self, batch: Sequence[TokenDatum]) -> StepResult:
        """Validation loss over ``batch``; the backend may retry transient failures."""
        ...

    def optim_step(self, learning_rate: float) -> None:
        """Apply the accumulated gradients."""
        ...

    def save_state(self, name: str) -> str:
        """Save resumable trainer state; returns its URI."""
        ...

    def save_sampler_weights(self, name: str) -> str:
        """Save servable weights; returns their URI."""
        ...
