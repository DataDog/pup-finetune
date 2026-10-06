# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Provider-neutral token arrays: what a renderer emits and a training backend consumes.

A :class:`TokenizedExample` is one rendered conversation, already shifted for next-token
prediction. Backends convert it to their SDK's own datum type; nothing here knows about any
provider.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    """One training sequence, shifted for next-token prediction.

    For a rendered sequence ``tokens``, ``input_tokens`` is ``tokens[:-1]`` and
    ``target_tokens`` is ``tokens[1:]``: position ``i`` predicts ``target_tokens[i]`` from
    ``input_tokens[:i + 1]``. ``weights[i]`` is the loss weight of ``target_tokens[i]``:
    ``1.0`` on assistant-completion tokens, ``0.0`` on everything else (system, user and tool
    turns, template scaffolding and assistant turns with ``weight`` ``0``).

    The three arrays have the same, non-zero length; weights are finite and non-negative.
    Lists are accepted and stored as tuples. Build one from an unshifted render with
    :meth:`from_tokens`.
    """

    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("input_tokens", "target_tokens", "weights"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        lengths = {len(self.input_tokens), len(self.target_tokens), len(self.weights)}
        if len(lengths) != 1:
            raise ValueError(
                "input_tokens, target_tokens and weights must have the same length, got "
                f"{len(self.input_tokens)}, {len(self.target_tokens)} and {len(self.weights)}"
            )
        if not self.input_tokens:
            raise ValueError("a tokenized example needs at least one target token")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights):
            raise ValueError("weights must be finite and non-negative")

    @classmethod
    def from_tokens(cls, tokens: Sequence[int], token_weights: Sequence[float]) -> Self:
        """Shift an unshifted render: ``token_weights[i]`` is the weight of ``tokens[i]``.

        The weight of token ``i + 1`` attaches to target ``i``, so the weight of the first
        token is dropped (nothing predicts it). Needs at least two tokens.
        """
        if len(tokens) != len(token_weights):
            raise ValueError(
                f"got {len(tokens)} tokens but {len(token_weights)} token weights; "
                "every token needs a weight"
            )
        if len(tokens) < 2:
            raise ValueError("a tokenized example needs at least two tokens")
        return cls(
            input_tokens=tuple(tokens[:-1]),
            target_tokens=tuple(tokens[1:]),
            weights=tuple(float(weight) for weight in token_weights[1:]),
        )

    @property
    def num_tokens(self) -> int:
        """The length of the unshifted sequence: every input token plus the last target."""
        return len(self.input_tokens) + 1

    @property
    def num_trainable(self) -> int:
        """How many target tokens carry a positive weight."""
        return sum(1 for weight in self.weights if weight > 0)
