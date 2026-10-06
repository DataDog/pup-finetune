# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Group rows into optimizer steps and pack each step's examples into microbatches.

A :data:`Step` is the microbatches of one optimizer step: the loop sends a
``forward_backward`` per microbatch, so gradients accumulate over them, then one
``optim_step``. Rows are shuffled per epoch with :func:`epoch_rng`, so the data order is a
pure function of the seed and the epoch: a resumed run replays exactly the rest of an epoch.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.training.datums import RowExamples

Microbatch = list[TokenizedExample]
"""Examples sent in one ``forward_backward`` (or ``forward``) call."""

Step = list[Microbatch]
"""The microbatches of one optimizer step."""


def epoch_rng(seed: int, epoch: int) -> random.Random:
    """The shuffle of epoch ``epoch`` (1-based): seeded with ``seed + epoch - 1``."""
    return random.Random(seed + epoch - 1)  # noqa: S311 - reproducible order, not a secret


def step_tokens(step: Step) -> int:
    """The rendered tokens of every example in ``step``."""
    return sum(example.num_tokens for microbatch in step for example in microbatch)


def plan_batches(
    rows: Sequence[RowExamples],
    *,
    rows_per_step: int,
    microbatch_token_budget: int | None,
    rng: random.Random | None,
) -> list[Step]:
    """The optimizer steps of one pass over ``rows``.

    ``rng`` shuffles a copy of ``rows`` (``None`` keeps their order); then each step takes the
    next ``rows_per_step`` rows (the last step may hold fewer). A step's examples, in order,
    are packed greedily into microbatches of at most ``microbatch_token_budget`` tokens; an
    example larger than the budget gets a microbatch of its own. With no budget each step is
    one microbatch.
    """
    if rows_per_step < 1:
        raise ValueError(f"rows_per_step must be positive, got {rows_per_step}")
    if microbatch_token_budget is not None and microbatch_token_budget < 1:
        raise ValueError(f"microbatch_token_budget must be positive, got {microbatch_token_budget}")
    ordered = list(rows)
    if rng is not None:
        rng.shuffle(ordered)
    steps: list[Step] = []
    for start in range(0, len(ordered), rows_per_step):
        examples = [
            example for row in ordered[start : start + rows_per_step] for example in row.examples
        ]
        if examples:
            steps.append(pack_microbatches(examples, microbatch_token_budget))
    return steps


def pack_microbatches(
    examples: Sequence[TokenizedExample], token_budget: int | None
) -> list[Microbatch]:
    """Pack ``examples`` greedily, in order, into microbatches of at most ``token_budget`` tokens.

    An example over the budget is sent alone. ``None`` puts every example in one microbatch.
    """
    if token_budget is None:
        return [list(examples)] if examples else []
    batches: list[Microbatch] = []
    current: Microbatch = []
    used = 0
    for example in examples:
        size = example.num_tokens
        if current and used + size > token_budget:
            batches.append(current)
            current, used = [], 0
        current.append(example)
        used += size
    if current:
        batches.append(current)
    return batches
