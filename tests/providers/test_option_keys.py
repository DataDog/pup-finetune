# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``--provider-option`` keys the registry names are each provider's options, and
``train --help`` lists them; the model names it holds are those of each provider's profiles."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from pup_finetune.cli.train import provider_option_keys
from pup_finetune.errors import UsageError
from pup_finetune.providers import (
    get_provider,
    known_models,
    providers_for_model,
    registered_option_keys,
)
from pup_finetune.providers.baseten.provider import BasetenOptions
from pup_finetune.providers.fake import FakeOptions
from pup_finetune.providers.fireworks.provider import FireworksOptions
from pup_finetune.runtime import Runtime

MODELS: dict[str, type[BaseModel]] = {
    "fake": FakeOptions,
    "fireworks": FireworksOptions,
    "baseten": BasetenOptions,
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_registry_names_each_provider_s_options(name: str) -> None:
    keys = registered_option_keys(include_dev=True)[name]
    assert sorted(keys) == sorted(MODELS[name].model_fields)


def test_every_provider_has_its_option_model_here() -> None:
    assert set(registered_option_keys(include_dev=True)) == set(MODELS)


def test_the_help_lists_the_keys_without_the_dev_only_provider() -> None:
    assert "fake" not in registered_option_keys()
    assert provider_option_keys() == "Keys: max_seq_len (baseten, fireworks)."


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_registry_holds_every_name_of_each_profile(name: str) -> None:
    runtime = Runtime(dev_mode=True)
    expected = {
        profile.alias: tuple(
            dict.fromkeys((profile.alias, profile.base_model, profile.tokenizer_model))
        )
        for profile in get_provider(name, runtime).models()
    }
    table = known_models(runtime)[name]
    assert dict(table) == expected
    assert list(table) == list(expected)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_a_provider_id_or_tokenizer_selects_its_profile_in_any_case(name: str) -> None:
    runtime = Runtime(dev_mode=True)
    provider = get_provider(name, runtime)
    for profile in provider.models():
        for other in profile.names[1:]:
            for spelled in (other.upper(), other.lower()):
                assert provider.resolve_model(spelled).alias == profile.alias
                assert providers_for_model(spelled, runtime) == providers_for_model(other, runtime)
        if profile.alias.upper() != profile.alias:
            with pytest.raises(UsageError):
                provider.resolve_model(profile.alias.upper())
