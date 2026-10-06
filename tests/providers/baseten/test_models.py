# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import pytest

from pup_finetune.errors import UsageError
from pup_finetune.providers.baseten.models import DEFAULT_MODEL, MODELS, BasetenModel, get_model
from pup_finetune.rendering import FAMILIES


def test_models_and_default() -> None:
    assert sorted(MODELS) == ["glm-5p3-flash", "qwen3p5-9b"]
    assert DEFAULT_MODEL == "qwen3p5-9b"
    assert DEFAULT_MODEL in MODELS


@pytest.mark.parametrize(
    ("alias", "base_model", "family"),
    [
        ("qwen3p5-9b", "Qwen/Qwen3.5-9B", "qwen3_5"),
        ("glm-5p3-flash", "zai-org/GLM-5.3-Flash", "glm5_3"),
    ],
)
def test_get_model(alias: str, base_model: str, family: str) -> None:
    model = get_model(alias)
    assert model is MODELS[alias]
    assert (model.alias, model.base_model) == (alias, base_model)
    assert model.family is FAMILIES[family]
    assert model.family.tokenizer_repo == base_model
    assert model.default_lora_rank == 8


@pytest.mark.parametrize("alias", ["acme-1b", "Qwen/Qwen3.5-9B", ""])
def test_unknown_model_is_a_usage_error_listing_the_aliases(alias: str) -> None:
    with pytest.raises(UsageError, match="unknown Baseten model") as caught:
        get_model(alias)
    assert caught.value.hint == "supported models: qwen3p5-9b, glm-5p3-flash"


def test_models_are_read_only() -> None:
    with pytest.raises(TypeError):
        MODELS["acme-1b"] = MODELS[DEFAULT_MODEL]  # type: ignore[index]
    assert isinstance(MODELS[DEFAULT_MODEL], BasetenModel)
