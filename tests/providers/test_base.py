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

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
    AdamConfig,
    LoraConfig,
    ModelProfile,
    StepMetrics,
    TrainSpec,
    raise_session_failure,
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
    assert (spec.lr_schedule, spec.warmup_fraction) == ("constant", 0.0)
    assert spec.batch_size == 32
    assert spec.lora_rank is None
    assert spec.lora_alpha is None
    assert spec.seed == 42
    assert spec.pipeline_depth == 4
    assert spec.checkpoint_every_minutes == 15
    assert spec.max_train_tokens is None
    assert spec.microbatch_token_budget is None
    assert spec.max_spend_usd is None
    assert spec.hourly_rate_usd is None
    assert spec.spend_reserve_fraction == 0.1
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
            "microbatch_token_budget": 8192,
            "max_spend_usd": 25.0,
            "hourly_rate_usd": 12.5,
            "spend_reserve_fraction": 0.0,
            "provider_options": {"region": "us"},
        }
    )
    assert spec.spend_reserve_fraction == 0.0
    assert spec.lora_rank == 16
    assert spec.microbatch_token_budget == 8192
    assert (spec.max_spend_usd, spec.hourly_rate_usd) == (25.0, 12.5)
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
        ("microbatch_token_budget", 0),
        ("microbatch_token_budget", 1.5),
        ("spend_reserve_fraction", 1.0),
        ("spend_reserve_fraction", -0.01),
        ("spend_reserve_fraction", math.nan),
        ("lr_schedule", "step"),
        ("warmup_fraction", 1.0),
        ("warmup_fraction", -0.1),
        ("warmup_fraction", math.inf),
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


@pytest.mark.parametrize(
    "value", [0.0, -1.0, math.inf, math.nan, "10"], ids=["zero", "negative", "inf", "nan", "str"]
)
@pytest.mark.parametrize("field", ["max_spend_usd", "hourly_rate_usd"])
def test_train_spec_rejects_invalid_spend_values(field: str, value: object) -> None:
    with pytest.raises(UsageError) as info:
        TrainSpec.parse({"max_spend_usd": 10.0, "hourly_rate_usd": 2.0, field: value})
    assert f"{field}:" in info.value.message


@pytest.mark.parametrize(
    "fields", [{"max_spend_usd": 10.0}, {"hourly_rate_usd": 2.0}], ids=["cap", "rate"]
)
def test_train_spec_needs_both_spend_fields_or_neither(fields: dict[str, float]) -> None:
    with pytest.raises(UsageError, match="max_spend_usd and hourly_rate_usd must be set together"):
        TrainSpec.parse(fields)


def test_train_spec_accepts_integer_spend_values() -> None:
    spec = TrainSpec(max_spend_usd=10, hourly_rate_usd=4)
    assert (spec.max_spend_usd, spec.hourly_rate_usd) == (10.0, 4.0)


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


def test_an_old_spec_without_a_schedule_is_constant() -> None:
    old = TrainSpec().model_dump()
    del old["lr_schedule"], old["warmup_fraction"]
    spec = TrainSpec.model_validate(old)
    assert (spec.lr_schedule, spec.warmup_fraction) == ("constant", 0.0)


def test_the_horizon_is_max_epochs_of_steps() -> None:
    assert TrainSpec(max_epochs=3, batch_size=4).max_steps(10) == 9


@pytest.mark.parametrize(
    ("fraction", "total", "steps"),
    [(0.0, 10, 0), (0.2, 10, 2), (0.01, 3, 1), (0.1, 30, 3), (1e-12, 10, 1)],
)
def test_warmup_steps_round_up(fraction: float, total: int, steps: int) -> None:
    assert TrainSpec(warmup_fraction=fraction).warmup_steps(total) == steps


def test_a_constant_schedule_without_warmup_is_the_learning_rate_at_every_step() -> None:
    spec = TrainSpec(learning_rate=3e-4)
    assert [spec.learning_rate_at(step, 7) for step in range(1, 8)] == [3e-4] * 7


@pytest.mark.parametrize(
    ("schedule", "factors"),
    [
        ("constant", [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("linear", [0.5, 1, 1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]),
        (
            "cosine",
            [0.5, 1, 1, 0.96194, 0.85355, 0.69134, 0.5, 0.30866, 0.14645, 0.03806],
        ),
    ],
)
def test_the_schedule_warms_up_then_decays(schedule: str, factors: list[float]) -> None:
    spec = TrainSpec.parse({"lr_schedule": schedule, "warmup_fraction": 0.2})
    rates = [spec.learning_rate_at(step, 10) for step in range(1, 11)]
    assert rates == pytest.approx([1e-4 * factor for factor in factors], rel=1e-4)


@pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
@pytest.mark.parametrize(("total", "fraction"), [(1, 0.0), (1, 0.5), (3, 0.01), (50, 0.1)])
def test_every_rate_is_positive_and_never_rises_after_warmup(
    schedule: str, total: int, fraction: float
) -> None:
    spec = TrainSpec.parse({"lr_schedule": schedule, "warmup_fraction": fraction})
    rates = [spec.learning_rate_at(step, total) for step in range(1, total + 1)]
    assert all(0 < rate <= spec.learning_rate for rate in rates)
    after = rates[spec.warmup_steps(total) :]
    assert after == sorted(after, reverse=True)


def test_resolved_settings_show_the_schedule_after_the_learning_rate() -> None:
    settings = TrainSpec(lr_schedule="cosine", warmup_fraction=0.05).resolved_settings(_profile())
    keys = list(settings)
    start = keys.index("learning_rate")
    assert keys[start : start + 3] == ["learning_rate", "lr_schedule", "warmup_fraction"]
    assert (settings["lr_schedule"], settings["warmup_fraction"]) == ("cosine", 0.05)


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


# --- raise_session_failure -------------------------------------------------------------------


def _failure(name: str) -> BaseException:
    """``ctrl-c...`` is an interrupt, anything else an error named ``name``."""
    return KeyboardInterrupt() if name.startswith("ctrl-c") else ExternalError(name, source="x")


@pytest.mark.parametrize(
    ("body", "closing", "raised", "noted"),
    [
        ("body failed", [], "body failed", []),
        (None, ["close 1 failed", "close 2 failed"], "close 1 failed", ["close 2 failed"]),
        ("body failed", ["close 1 failed"], "body failed", ["close 1 failed"]),
        ("ctrl-c in body", ["close 1 failed"], "ctrl-c in body", ["close 1 failed"]),
        (
            "body failed",
            ["close 1 failed", "ctrl-c in close"],
            "ctrl-c in close",
            ["body failed", "close 1 failed"],
        ),
        ("ctrl-c in body", ["ctrl-c in close"], "ctrl-c in body", ["ctrl-c in close"]),
        (None, ["close 1 failed", "ctrl-c in close"], "ctrl-c in close", ["close 1 failed"]),
    ],
)
def test_a_session_ends_with_the_first_interrupt_else_the_body_error_else_a_close_failure(
    body: str | None, closing: list[str], raised: str, noted: list[str]
) -> None:
    failures = {name: _failure(name) for name in [*([body] if body else []), *closing]}
    with pytest.raises(BaseException) as caught:
        raise_session_failure(
            failures[body] if body else None, [failures[name] for name in closing]
        )
    assert caught.value is failures[raised]
    notes = getattr(caught.value, "__notes__", [])
    expected = [
        "KeyboardInterrupt" if name.startswith("ctrl-c") else f"x: {name}" for name in noted
    ]
    assert len(notes) == len(expected)
    assert all(text in note for text, note in zip(expected, notes, strict=True))


def test_a_session_that_ended_cleanly_raises_nothing() -> None:
    raise_session_failure(None, [])


def test_a_noted_failure_keeps_its_hint_and_its_own_notes() -> None:
    body = ExternalError("sampler create failed", source="x", hint="check the console")
    body.add_note("a sampler may remain")
    interrupt = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        raise_session_failure(body, [interrupt])
    first, carried = caught.value.__notes__
    assert "x: sampler create failed" in first
    assert "check the console" in first
    assert carried == "a sampler may remain"
