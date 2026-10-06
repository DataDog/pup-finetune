# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Pack token datums into microbatches and group microbatches into optimizer steps."""

from __future__ import annotations

from collections.abc import Sequence

from pup_finetune.errors import UsageError
from pup_finetune.rendering.types import TokenDatum


class TrainingDataError(UsageError):
    """Training data that cannot be batched as configured."""


def datum_tokens(datum: TokenDatum) -> int:
    """The model input length of ``datum``: the last token is only ever a target."""
    return len(datum.tokens) - 1


def pack_microbatches(datums: Sequence[TokenDatum], token_budget: int) -> list[list[TokenDatum]]:
    """Pack ``datums`` greedily and in order into microbatches of at most ``token_budget`` tokens."""
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    batches: list[list[TokenDatum]] = []
    current: list[TokenDatum] = []
    used = 0
    for datum in datums:
        size = datum_tokens(datum)
        if size > token_budget:
            raise TrainingDataError(
                f"a datum has {size} input tokens, over the microbatch token budget of "
                f"{token_budget}"
            )
        if current and used + size > token_budget:
            batches.append(current)
            current, used = [], 0
        current.append(datum)
        used += size
    if current:
        batches.append(current)
    return batches


def accumulation_groups(
    microbatches: Sequence[Sequence[TokenDatum]], batch_size: int
) -> list[list[list[TokenDatum]]]:
    """Group microbatches into optimizer steps of exactly ``batch_size`` datums each.

    A microbatch that straddles two steps is split between them; the last step may be partial.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups: list[list[list[TokenDatum]]] = []
    current: list[list[TokenDatum]] = []
    count = 0
    for microbatch in microbatches:
        remaining = list(microbatch)
        while remaining:
            room = batch_size - count
            segment, remaining = remaining[:room], remaining[room:]
            current.append(segment)
            count += len(segment)
            if count == batch_size:
                groups.append(current)
                current, count = [], 0
    if current:
        groups.append(current)
    return groups
