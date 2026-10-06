# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Provider-neutral supervised fine-tuning settings."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SFTSettings(BaseModel):
    """LoRA SFT hyperparameters and the optional spend guard. ``None`` means the model default.

    Strict: numbers keep their types (no bools or strings), and floats must be finite.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    max_epochs: int = Field(default=5, ge=1)
    early_stopping_patience: int = Field(default=1, ge=1)
    min_delta: float = Field(default=0.0, ge=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=32, ge=1, description="Datums per optimizer step.")
    seed: int = Field(default=42, ge=0)
    lora_rank: int | None = Field(default=None, ge=1)
    max_seq_len: int | None = Field(default=None, ge=1)
    microbatch_token_budget: int | None = Field(default=None, ge=1)
    max_spend_usd: float | None = Field(default=None, gt=0)
    hourly_rate_usd: float | None = Field(default=None, gt=0)
    spend_reserve_fraction: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def _spend_guard_pair(self) -> Self:
        if (self.max_spend_usd is None) != (self.hourly_rate_usd is None):
            raise ValueError("max_spend_usd and hourly_rate_usd must be set together")
        return self
