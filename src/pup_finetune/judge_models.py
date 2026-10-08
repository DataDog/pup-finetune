# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Judge model names: the default judge, and short names for Fireworks judge models that
``--judges``, ``--judge-model`` and ``--eval-judge-model`` accept.

An alias stands for one full Fireworks model id. :func:`pup_finetune.judges.parse_judge`
resolves it, so the full id is what previews print and what ``triage.json`` and
``evaluation.json`` freeze. This module imports nothing else, so help texts can name the
aliases and the default without loading a judge client.
"""

from __future__ import annotations

from typing import Final

__all__ = ["DEFAULT_JUDGE_MODEL", "JUDGE_SHORT_NAMES", "alias_of", "resolve_alias"]

DEFAULT_JUDGE_MODEL: Final = "accounts/fireworks/models/deepseek-v4p1-flash"
"""The judge ``evaluate`` uses unless told otherwise (re-exported by
:mod:`pup_finetune.inference`)."""

JUDGE_SHORT_NAMES: Final[dict[str, str]] = {
    "deepseek-v4p1-flash": "accounts/fireworks/models/deepseek-v4p1-flash",
    "glm-5p3-flash": "accounts/fireworks/models/glm-5p3-flash",
}
"""Each alias and the Fireworks model id it stands for."""


def resolve_alias(model: str) -> str:
    """The Fireworks model id ``model`` stands for: itself unless it is an alias."""
    return JUDGE_SHORT_NAMES.get(model, model)


def alias_of(model: str) -> str:
    """The alias of the Fireworks model id ``model``, or ``model`` when it has none."""
    return next((alias for alias, full in JUDGE_SHORT_NAMES.items() if full == model), model)
