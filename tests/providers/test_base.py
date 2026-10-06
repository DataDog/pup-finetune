# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from pup_finetune.errors import UsageError
from pup_finetune.providers.base import (
    AdamConfig,
    LoraConfig,
    ModelProfile,
    StepMetrics,
    TrainSpec,
)


def _profile(**overrides: Any) -> ModelProfile:
    fields: dict[str, Any] = {
        "provider": "example",
        "alias": "tiny-1b",
        "base_model": "models/tiny-1b",
        "tokenizer_model": "example-org/tiny-1b",
        "tokenizer_revision": "0" * 40,
        "renderer": "tiny",
        "max_seq_len": 4096,
    }
    fields.update(overrides)
    return ModelProfile(**fields)


# --- TrainSpec -----------------------------------------------------------------------------


def test_train_spec_defaults() -> None:
    spec = TrainSpec()
    assert spec.max_epochs == 5
    assert spec.patience == 1
    assert spec.min_delta == 0.0
    assert spec.learning_rate == 1e-4
    assert spec.batch_size == 32
    assert spec.lora_rank is None
    assert spec.lora_alpha is None
    assert spec.seed == 42
    assert spec.pipeline_depth == 4
    assert spec.checkpoint_every_minutes == 15
    assert spec.max_train_tokens is None
    assert spec.provider_options == {}


def test_train_spec_accepts_valid_settings() -> None:
    spec = TrainSpec.parse(
        {
            "max_epochs": 3,
            "patience": 2,
            "min_delta": 0.01,
            "learning_rate": 2e-4,
            "batch_size": 8,
            "lora_rank": 16,
            "lora_alpha": 32,
            "seed": 0,
            "pipeline_depth": 1,
            "checkpoint_every_minutes": 0.5,
            "max_train_tokens": 1_000_000,
            "provider_options": {"region": "us"},
        }
    )
    assert spec.lora_rank == 16
    assert spec.provider_options == {"region": "us"}


def test_train_spec_accepts_an_integer_learning_rate_and_interval() -> None:
    spec = TrainSpec(learning_rate=1, checkpoint_every_minutes=5)
    assert spec.learning_rate == 1.0
    assert spec.checkpoint_every_minutes == 5.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_epochs", 0),
        ("patience", 0),
        ("batch_size", -1),
        ("pipeline_depth", 0),
        ("lora_rank", 0),
        ("lora_alpha", 0),
        ("max_train_tokens", 0),
        ("seed", -1),
        ("min_delta", -0.1),
        ("min_delta", math.nan),
        ("learning_rate", 0.0),
        ("learning_rate", math.inf),
        ("checkpoint_every_minutes", 0),
        ("batch_size", True),
        ("batch_size", "32"),
        ("batch_size", 1.5),
    ],
)
def test_train_spec_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(UsageError) as info:
        TrainSpec.parse({field: value})
    assert f"{field}:" in info.value.message
    assert info.value.exit_code == 2


def test_train_spec_rejects_unknown_fields() -> None:
    with pytest.raises(UsageError, match="epochs: Extra inputs are not permitted"):
        TrainSpec.parse({"epochs": 3})


def test_train_spec_reports_every_problem() -> None:
    with pytest.raises(UsageError) as info:
        TrainSpec.parse({"max_epochs": 0, "batch_size": 0})
    assert "max_epochs:" in info.value.message
    assert "batch_size:" in info.value.message


def test_train_spec_constructor_raises_pydantic_errors() -> None:
    with pytest.raises(ValidationError):
        TrainSpec(max_epochs=0)


def test_train_spec_is_frozen() -> None:
    spec = TrainSpec()
    with pytest.raises(ValidationError):
        spec.max_epochs = 2  # type: ignore[misc]


def test_train_spec_round_trips_through_json() -> None:
    spec = TrainSpec(lora_rank=4, provider_options={"shape": "small"})
    assert TrainSpec.model_validate_json(spec.model_dump_json()) == spec


def test_lora_config_falls_back_to_the_profile_default() -> None:
    profile = _profile(default_lora_rank=8)
    assert TrainSpec().lora_config(profile) == LoraConfig(rank=8, alpha=None)
    assert TrainSpec(lora_rank=32, lora_alpha=64).lora_config(profile) == LoraConfig(32, 64)


def test_adam_config_uses_the_learning_rate_and_defaults() -> None:
    adam = TrainSpec(learning_rate=3e-4).adam_config()
    assert adam == AdamConfig(
        learning_rate=3e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0
    )


# --- Small value types ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": 0.0},
        {"learning_rate": math.nan},
        {"beta1": 1.0},
        {"beta2": -0.1},
        {"eps": 0.0},
        {"weight_decay": -0.01},
        {"grad_clip_norm": -1.0},
    ],
)
def test_adam_config_rejects_invalid_values(overrides: dict[str, float]) -> None:
    fields = {"learning_rate": 1e-4, **overrides}
    with pytest.raises(ValueError):
        AdamConfig(**fields)


@pytest.mark.parametrize(("rank", "alpha"), [(0, None), (-1, None), (8, 0)])
def test_lora_config_rejects_non_positive_values(rank: int, alpha: int | None) -> None:
    with pytest.raises(ValueError):
        LoraConfig(rank, alpha)


@pytest.mark.parametrize("overrides", [{"max_seq_len": 0}, {"default_lora_rank": 0}])
def test_model_profile_rejects_non_positive_limits(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        _profile(**overrides)


@pytest.mark.parametrize(
    "revision", ["main", "v1.0", "0123abc", "A" * 40, "g" * 40, "0" * 41, "0" * 40 + "\n"]
)
def test_model_profile_rejects_a_movable_or_malformed_tokenizer_revision(revision: str) -> None:
    with pytest.raises(ValueError, match="tokenizer_revision"):
        _profile(tokenizer_revision=revision)


def test_model_profile_accepts_a_full_commit_sha() -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"
    assert _profile(tokenizer_revision=sha).tokenizer_revision == sha


def test_model_profile_defaults() -> None:
    profile = _profile()
    assert profile.default_lora_rank == 8
    assert profile.supports_reasoning is False
    assert profile.notes == ""


def test_step_metrics_mean_loss() -> None:
    assert StepMetrics(loss_sum=6.0, weight=4.0, tokens=10).mean_loss == 1.5


@pytest.mark.parametrize("weight", [0.0, -1.0, math.nan])
def test_step_metrics_mean_loss_is_nan_without_weight(weight: float) -> None:
    assert math.isnan(StepMetrics(loss_sum=3.0, weight=weight, tokens=5).mean_loss)
