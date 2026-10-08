# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What the registry says each built-in provider needs matches the providers themselves:
the keys they sign in with, the extras that install their packages, the flags they refuse
and whether they deploy."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from pup_finetune import credentials
from pup_finetune.errors import UsageError
from pup_finetune.providers import provider_info, provider_infos
from pup_finetune.providers.base import LoraConfig
from pup_finetune.providers.baseten import backend as baseten_backend
from pup_finetune.providers.baseten.provider import PROFILES as BASETEN_PROFILES
from pup_finetune.providers.fireworks import client as fireworks_client
from pup_finetune.runtime import Runtime

DEV = Runtime(dev_mode=True)
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_each_provider_s_key_is_a_registered_credential() -> None:
    for info in provider_infos(DEV):
        if info.credential is None:
            assert info.offline, f"{info.name} has no key, so it must be offline"
            continue
        spec = credentials.get_spec(info.credential)
        assert spec.needed_for, f"{info.name}'s key says what it is needed for"
        assert info.name in spec.needed_for


def test_each_package_is_installed_by_an_extra() -> None:
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
    for info in provider_infos(DEV):
        for package in info.packages:
            assert package.extra in extras, f"{info.name}: no extra {package.extra!r}"
            assert package.steps, f"{info.name}: {package.key} is needed for no step"


def test_the_fireworks_sdk_modules_are_the_ones_its_client_loads() -> None:
    (sdk,) = [p for p in provider_info("fireworks", DEV).packages if p.key == "sdk"]
    assert all(fireworks_client._is_sdk_module(module) for module in sdk.modules)
    assert sdk.steps == ("train", "evaluate", "deploy")


def test_baseten_refuses_the_lora_alpha_in_the_preview_as_its_backend_does() -> None:
    (refused,) = provider_info("baseten", DEV).refuses
    assert (refused.field, refused.flag) == ("lora_alpha", "--lora-alpha")
    profile = BASETEN_PROFILES[0]
    with pytest.raises(UsageError, match="LoRA alpha"):
        baseten_backend._check_session(
            profile, LoraConfig(rank=8, alpha=16), None, seed=0, max_seq_len=1024
        )
    baseten_backend._check_session(profile, LoraConfig(rank=8), None, seed=0, max_seq_len=1024)


def test_only_providers_that_deploy_say_so() -> None:
    assert provider_info("fireworks", DEV).deploys
    assert provider_info("fake", DEV).deploys
    assert provider_info("baseten", DEV).deploys


def test_the_fake_is_the_only_offline_provider() -> None:
    assert [info.name for info in provider_infos(DEV) if info.offline] == ["fake"]
    assert provider_info("fake", DEV).dev_only
