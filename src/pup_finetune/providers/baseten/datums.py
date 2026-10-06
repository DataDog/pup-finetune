# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Shift rendered datums into the next-token form Baseten Loops trains on. Pure; no SDK."""

from __future__ import annotations

from dataclasses import dataclass

from pup_finetune.rendering.types import RenderError, RenderErrorCode, TokenDatum

IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class ShiftedDatum:
    """Model inputs and, position by position, the next token and its loss weight."""

    inputs: list[int]
    targets: list[int]
    """The next token, or :data:`IGNORE_INDEX` where the weight is 0."""
    weights: list[float]


def shift(datum: TokenDatum, *, max_seq_len: int) -> ShiftedDatum:
    """``datum`` shifted by one token; :class:`RenderError` when it cannot train."""
    tokens, weights = datum.tokens, datum.weights
    if len(tokens) < 2:
        raise RenderError(RenderErrorCode.INVALID_TOKENS, "a datum needs at least 2 tokens")
    if len(weights) != len(tokens):
        raise RenderError(RenderErrorCode.INVALID_TOKENS, "tokens and weights differ in length")
    if len(tokens) > max_seq_len:
        raise RenderError(
            RenderErrorCode.EXCEEDS_CONTEXT,
            f"a datum has {len(tokens)} tokens, over the limit of {max_seq_len}",
            tokens=len(tokens),
            limit=max_seq_len,
        )
    if any(weight not in (0, 1) for weight in weights):
        raise RenderError(RenderErrorCode.INVALID_TOKENS, "token weights must be 0 or 1")
    shifted = weights[1:]
    if not any(shifted):
        raise RenderError(RenderErrorCode.NO_TARGET_TOKENS, "a datum has no weighted target token")
    targets = [
        token if weight else IGNORE_INDEX for token, weight in zip(tokens[1:], shifted, strict=True)
    ]
    return ShiftedDatum(list(tokens[:-1]), targets, [float(weight) for weight in shifted])
