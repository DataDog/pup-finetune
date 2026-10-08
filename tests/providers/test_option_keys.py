# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``--provider-option`` keys the registry names are each provider's options, and
``train --help`` lists them."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from pup_finetune.cli.train import provider_option_keys
from pup_finetune.providers import registered_option_keys
from pup_finetune.providers.baseten.provider import BasetenOptions
from pup_finetune.providers.fake import FakeOptions
from pup_finetune.providers.fireworks.provider import FireworksOptions

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
