# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks models we fine-tune, as a table.

Each :class:`ProfileSpec` names a Fireworks base model and the model family
(:data:`~pup_finetune.rendering.families.FAMILIES`) that renders it. The tokenizer and its
pinned revision come from the family, never from this table, so a profile cannot drift from
the renderer that trains it. Adding a model is adding an entry here (plus a family when the
model's chat template is new). :func:`build_profiles` checks the table when this module is
imported: no two profiles share a name and every family exists.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from pup_finetune.providers.base import ModelProfile
from pup_finetune.providers.fireworks import PROVIDER_NAME
from pup_finetune.rendering.families import FAMILIES

__all__ = ["PROFILES", "SPECS", "ProfileSpec", "build_profiles"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileSpec:
    """One supported Fireworks model.

    ``base_model`` is the Fireworks model resource name; ``family`` the rendering family's key.
    ``max_seq_len`` is Fireworks' training context length for the model, in tokens.
    """

    alias: str
    base_model: str
    family: str
    max_seq_len: int
    default_lora_rank: int = 8
    supports_reasoning: bool = False
    notes: str = ""


SPECS: Final[tuple[ProfileSpec, ...]] = (
    ProfileSpec(
        alias="qwen3p8-27b",
        base_model="accounts/fireworks/models/qwen3p8-27b",
        family="qwen3_8",
        max_seq_len=131_072,
        default_lora_rank=8,
        supports_reasoning=True,
        notes="Qwen3.8 27B; reasoning history is preserved in training.",
    ),
    ProfileSpec(
        alias="glm-5p3-flash",
        base_model="accounts/fireworks/models/glm-5p3-flash",
        family="glm5_3",
        max_seq_len=65_536,
        supports_reasoning=True,
        notes="GLM-5.3 Flash; it stops by emitting the next role tag.",
    ),
)
"""Every supported model. ``max_seq_len`` is the ``trainingContextLength`` Fireworks reports for
the model; preflight refuses a model that reports less."""


def build_profiles(specs: Iterable[ProfileSpec]) -> tuple[ModelProfile, ...]:
    """The :class:`ModelProfile` for each spec, with the tokenizer taken from its family.

    Raises ``ValueError`` for a name two profiles share (see
    :attr:`~pup_finetune.providers.base.ModelProfile.names`), or an unknown family.
    """
    profiles: list[ModelProfile] = []
    names: set[str] = set()
    for spec in specs:
        family = FAMILIES.get(spec.family)
        if family is None:
            raise ValueError(
                f"Fireworks model {spec.alias!r} names unknown model family {spec.family!r}; "
                f"known: {', '.join(FAMILIES)}"
            )
        profile = ModelProfile(
            provider=PROVIDER_NAME,
            alias=spec.alias,
            base_model=spec.base_model,
            tokenizer_model=family.tokenizer_model,
            tokenizer_revision=family.revision,
            renderer=family.key,
            max_seq_len=spec.max_seq_len,
            default_lora_rank=spec.default_lora_rank,
            supports_reasoning=spec.supports_reasoning,
            notes=spec.notes,
        )
        for name in profile.names:
            if name in names:
                raise ValueError(f"Fireworks model name {name!r} is used more than once")
            names.add(name)
        profiles.append(profile)
    return tuple(profiles)


PROFILES: Final[tuple[ModelProfile, ...]] = build_profiles(SPECS)
"""The profiles :meth:`~pup_finetune.providers.fireworks.provider.FireworksProvider.models`
returns, in table order."""
