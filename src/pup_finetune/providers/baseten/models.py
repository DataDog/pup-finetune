# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The base models we fine-tune on Baseten Loops, by alias."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pup_finetune.errors import UsageError
from pup_finetune.rendering.families import FAMILIES
from pup_finetune.rendering.types import ModelFamily


@dataclass(frozen=True, slots=True)
class BasetenModel:
    """A Loops base model and the family its rows render with."""

    alias: str
    base_model: str
    family: ModelFamily
    default_lora_rank: int = 8


MODELS: Mapping[str, BasetenModel] = MappingProxyType(
    {
        model.alias: model
        for model in (
            BasetenModel("qwen3p5-9b", "Qwen/Qwen3.5-9B", FAMILIES["qwen3_5"]),
            BasetenModel("glm-5p3-flash", "zai-org/GLM-5.3-Flash", FAMILIES["glm5_3"]),
        )
    }
)

DEFAULT_MODEL = "qwen3p5-9b"


def get_model(alias: str) -> BasetenModel:
    """The model called ``alias``; a :class:`UsageError` listing the aliases if unknown."""
    model = MODELS.get(alias)
    if model is None:
        raise UsageError(
            f"unknown Baseten model {alias!r}",
            hint=f"supported models: {', '.join(MODELS)}",
        )
    return model
