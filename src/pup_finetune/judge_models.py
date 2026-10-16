# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Judge model names: the default judges, and short names for Fireworks judge models that
``--judges``, ``--judge-model`` and ``--eval-judge-model`` accept.

The default judges follow the credentials there are: :data:`EVALUATE_DEFAULTS` and
:data:`TRIAGE_DEFAULTS` list, in order, which judges to use when which credentials resolve
(:func:`pup_finetune.judges.default_judges` picks the first row whose credentials all do).

An alias stands for one full Fireworks model id. :func:`pup_finetune.judges.parse_judge`
resolves it, so the full id is what previews print and what ``triage.json`` and
``evaluation.json`` freeze. This module imports nothing else, so help texts can name the
aliases and the default without loading a judge client.
"""

from __future__ import annotations

from typing import Final, NamedTuple

__all__ = [
    "BY_KEY",
    "DEFAULT_JUDGE_MODEL",
    "EVALUATE_DEFAULTS",
    "FIREWORKS_JUDGES",
    "JUDGE_SHORT_NAMES",
    "TRIAGE_DEFAULTS",
    "JudgeDefault",
    "alias_of",
    "resolve_alias",
]

DEFAULT_JUDGE_MODEL: Final = "accounts/fireworks/models/deepseek-v4p1-flash"
"""The Fireworks judge, the model id a :class:`~pup_finetune.inference.JudgeConfig` defaults
to, and ``evaluate``'s judge when no Anthropic or OpenAI credential resolves."""
FIREWORKS_JUDGES: Final = (DEFAULT_JUDGE_MODEL, "accounts/fireworks/models/glm-5p3-flash")
"""Two Fireworks serverless models: the triage council when no Anthropic or OpenAI
credential resolves."""
ANTHROPIC_FAST: Final = "anthropic/claude-haiku-5-5"
ANTHROPIC_STRONG: Final = "anthropic/claude-sonnet-5-5"
OPENAI_FAST: Final = "openai/gpt-5.4-mini"
OPENAI_STRONG: Final = "openai/gpt-6-astra"


class JudgeDefault(NamedTuple):
    """Judges to default to when every credential in ``needs`` resolves."""

    needs: tuple[str, ...]
    judges: tuple[str, ...]


EVALUATE_DEFAULTS: Final = (
    JudgeDefault(("anthropic",), (ANTHROPIC_FAST,)),
    JudgeDefault(("openai",), (OPENAI_FAST,)),
    JudgeDefault((), (DEFAULT_JUDGE_MODEL,)),
)
"""``evaluate``'s (and ``benchmark``'s) judge, by the credentials that resolve, in order."""
TRIAGE_DEFAULTS: Final = (
    JudgeDefault(("anthropic", "openai"), (ANTHROPIC_FAST, OPENAI_FAST)),
    JudgeDefault(("anthropic",), (ANTHROPIC_FAST, ANTHROPIC_STRONG)),
    JudgeDefault(("openai",), (OPENAI_FAST, OPENAI_STRONG)),
    JudgeDefault((), FIREWORKS_JUDGES),
)
"""``dataset triage``'s council, by the credentials that resolve, in order."""
BY_KEY: Final = "by key: Anthropic, OpenAI, else Fireworks"
"""How help texts state the default judges, which depend on the credentials there are."""

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
