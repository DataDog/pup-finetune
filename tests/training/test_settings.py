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

from pup_finetune.training import SFTSettings


def test_defaults() -> None:
    assert SFTSettings().model_dump() == {
        "max_epochs": 5,
        "early_stopping_patience": 1,
        "min_delta": 0.0,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "seed": 42,
        "lora_rank": None,
        "max_seq_len": None,
        "microbatch_token_budget": None,
        "max_spend_usd": None,
        "hourly_rate_usd": None,
        "spend_reserve_fraction": 0.1,
    }


def test_accepts_every_field_at_its_bound() -> None:
    settings = SFTSettings(
        max_epochs=1,
        early_stopping_patience=1,
        min_delta=0.0,
        learning_rate=1e-9,
        batch_size=1,
        seed=0,
        lora_rank=1,
        max_seq_len=1,
        microbatch_token_budget=1,
        max_spend_usd=0.01,
        hourly_rate_usd=0.01,
        spend_reserve_fraction=0.0,
    )
    assert settings.seed == 0
    assert settings.spend_reserve_fraction == 0.0


def test_an_int_is_accepted_for_a_float_field() -> None:
    settings = SFTSettings(learning_rate=1, max_spend_usd=10, hourly_rate_usd=4)
    assert settings.learning_rate == 1.0
    assert isinstance(settings.max_spend_usd, float)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_epochs": 0},
        {"early_stopping_patience": 0},
        {"min_delta": -0.1},
        {"learning_rate": 0.0},
        {"batch_size": 0},
        {"seed": -1},
        {"lora_rank": 0},
        {"max_seq_len": 0},
        {"microbatch_token_budget": 0},
        {"max_spend_usd": 0.0, "hourly_rate_usd": 1.0},
        {"max_spend_usd": 1.0, "hourly_rate_usd": 0.0},
        {"spend_reserve_fraction": -0.1},
        {"spend_reserve_fraction": 1.0},
    ],
)
def test_rejects_out_of_range_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SFTSettings(**overrides)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["min_delta", "learning_rate", "spend_reserve_fraction"])
def test_rejects_non_finite_floats(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        SFTSettings(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [{"max_epochs": True}, {"seed": 1.0}, {"learning_rate": "0.001"}, {"lora_rank": "8"}],
)
def test_rejects_loose_types(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SFTSettings(**overrides)


@pytest.mark.parametrize("overrides", [{"max_spend_usd": 10.0}, {"hourly_rate_usd": 4.0}])
def test_spend_guard_needs_both_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="set together"):
        SFTSettings(**overrides)


def test_spend_guard_with_both_values() -> None:
    settings = SFTSettings(max_spend_usd=10.0, hourly_rate_usd=4.0)
    assert (settings.max_spend_usd, settings.hourly_rate_usd) == (10.0, 4.0)


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        SFTSettings(epochs=3)  # type: ignore[call-arg]


def test_is_frozen() -> None:
    settings = SFTSettings()
    with pytest.raises(ValidationError):
        settings.max_epochs = 2  # type: ignore[misc]


def test_round_trips_through_json() -> None:
    settings = SFTSettings(lora_rank=16, max_spend_usd=10.0, hourly_rate_usd=4.0)
    assert SFTSettings.model_validate_json(settings.model_dump_json()) == settings
