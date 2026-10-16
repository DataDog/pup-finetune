# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks model table: the one profile, and the checks every entry passes."""

from __future__ import annotations

import dataclasses

import pytest

from pup_finetune.providers.fireworks.models import PROFILES, SPECS, ProfileSpec, build_profiles
from pup_finetune.rendering.families import FAMILIES, QWEN3_8


def spec(**overrides: object) -> ProfileSpec:
    return dataclasses.replace(SPECS[0], **overrides)  # type: ignore[arg-type]


def test_the_qwen3p8_profile() -> None:
    profile = next(profile for profile in PROFILES if profile.alias == "qwen3p8-27b")
    assert profile.provider == "fireworks"
    assert profile.alias == "qwen3p8-27b"
    assert profile.base_model == "accounts/fireworks/models/qwen3p8-27b"
    assert profile.renderer == "qwen3_8"
    assert profile.tokenizer_model == "Qwen/Qwen3.8-27B"
    assert profile.tokenizer_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert profile.max_seq_len == 131_072
    assert profile.default_lora_rank == 8
    assert profile.supports_reasoning is True


def test_the_glm_profile_renders_with_the_glm5_3_family() -> None:
    profile = next(profile for profile in PROFILES if profile.alias == "glm-5p3-flash")
    assert profile.base_model == "accounts/fireworks/models/glm-5p3-flash"
    assert (profile.renderer, profile.tokenizer_model) == ("glm5_3", "zai-org/GLM-5.3-Flash")
    assert profile.max_seq_len == 65_536
    assert profile.supports_reasoning is True


def test_every_profile_takes_its_tokenizer_from_its_family() -> None:
    assert len(PROFILES) == len(SPECS)
    for entry, profile in zip(SPECS, PROFILES, strict=True):
        family = FAMILIES[entry.family]
        assert profile.renderer == family.key
        assert profile.tokenizer_model == family.tokenizer_model
        assert profile.tokenizer_revision == family.revision
    assert PROFILES[0].tokenizer_revision == QWEN3_8.revision


def test_aliases_and_base_models_are_unique() -> None:
    names = [name for profile in PROFILES for name in profile.names]
    assert len(names) == len(set(names))


def test_a_duplicate_alias_is_refused() -> None:
    with pytest.raises(ValueError, match="'qwen3p8-27b' is used more than once"):
        build_profiles([SPECS[0], spec(base_model="accounts/fireworks/models/other")])


def test_a_duplicate_base_model_is_refused() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_profiles([SPECS[0], spec(alias="other")])


def test_two_profiles_of_one_family_would_share_its_tokenizer_name() -> None:
    other = spec(alias="other", base_model="accounts/fireworks/models/other")
    with pytest.raises(ValueError, match=r"'Qwen/Qwen3\.8-27B' is used more than once"):
        build_profiles([SPECS[0], other])


def test_an_alias_may_not_shadow_another_base_model() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_profiles(
            [SPECS[0], spec(alias=SPECS[0].base_model, base_model="accounts/x/models/y")]
        )


def test_an_unknown_family_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown model family 'no-such-family'; known: qwen3_8"):
        build_profiles([spec(family="no-such-family")])


def test_profile_values_are_validated() -> None:
    with pytest.raises(ValueError, match="max_seq_len must be positive"):
        build_profiles([spec(max_seq_len=0)])
    with pytest.raises(ValueError, match="default_lora_rank must be positive"):
        build_profiles([spec(default_lora_rank=0)])
