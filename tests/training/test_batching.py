# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import random

import pytest

from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.training import (
    RowExamples,
    Step,
    epoch_rng,
    pack_microbatches,
    plan_batches,
    step_tokens,
)
from tests.fakes.training import row_examples


def example(tokens: int) -> TokenizedExample:
    return TokenizedExample.from_tokens(list(range(tokens)), [1.0] * tokens)


def row_ids(steps: list[Step], rows: list[RowExamples]) -> list[list[str]]:
    """The rows of each step, found by their examples."""
    owner = {id(ex): row.row_id for row in rows for ex in row.examples}
    ids: list[list[str]] = []
    for step in steps:
        seen: list[str] = []
        for microbatch in step:
            for ex in microbatch:
                if owner[id(ex)] not in seen:
                    seen.append(owner[id(ex)])
        ids.append(seen)
    return ids


def test_epoch_rng_is_seeded_with_seed_plus_epoch_minus_one() -> None:
    assert epoch_rng(42, 1).random() == random.Random(42).random()  # noqa: S311
    assert epoch_rng(42, 3).random() == random.Random(44).random()  # noqa: S311


def test_the_shuffle_is_deterministic_per_seed_and_epoch() -> None:
    rows = row_examples(20)

    def order(seed: int, epoch: int) -> list[list[str]]:
        steps = plan_batches(
            rows, rows_per_step=4, microbatch_token_budget=None, rng=epoch_rng(seed, epoch)
        )
        return row_ids(steps, rows)

    assert order(7, 1) == order(7, 1)
    assert order(7, 1) != order(7, 2)
    assert order(7, 2) == order(8, 1)
    assert sorted(r for step in order(7, 1) for r in step) == sorted(r.row_id for r in rows)


def test_without_an_rng_rows_keep_their_order() -> None:
    rows = row_examples(5)
    steps = plan_batches(rows, rows_per_step=2, microbatch_token_budget=None, rng=None)
    assert row_ids(steps, rows) == [["row-0", "row-1"], ["row-2", "row-3"], ["row-4"]]


def test_plan_batches_does_not_shuffle_the_callers_list() -> None:
    rows = row_examples(6)
    before = list(rows)
    plan_batches(rows, rows_per_step=2, microbatch_token_budget=None, rng=epoch_rng(1, 1))
    assert rows == before


def test_a_rows_examples_stay_in_one_step() -> None:
    rows = [RowExamples(f"r{i}", (example(5), example(6), example(7))) for i in range(4)]
    steps = plan_batches(rows, rows_per_step=1, microbatch_token_budget=None, rng=epoch_rng(3, 1))
    assert len(steps) == 4
    assert all(len(step) == 1 and len(step[0]) == 3 for step in steps)


def test_no_budget_sends_each_step_as_one_microbatch() -> None:
    rows = row_examples(5)
    steps = plan_batches(rows, rows_per_step=2, microbatch_token_budget=None, rng=None)
    assert [len(step) for step in steps] == [1, 1, 1]
    assert [len(step[0]) for step in steps] == [2, 2, 1]


def test_microbatches_respect_the_token_budget() -> None:
    sizes = [3, 4, 5, 2, 6, 2, 4]
    rows = [RowExamples(f"r{i}", (example(size),)) for i, size in enumerate(sizes)]
    [step] = plan_batches(rows, rows_per_step=len(rows), microbatch_token_budget=8, rng=None)
    assert [[ex.num_tokens for ex in mb] for mb in step] == [[3, 4], [5, 2], [6, 2], [4]]
    assert all(sum(ex.num_tokens for ex in mb) <= 8 for mb in step)
    assert step_tokens(step) == sum(sizes)


def test_an_example_over_the_budget_gets_its_own_microbatch() -> None:
    batches = pack_microbatches([example(2), example(20), example(3), example(2)], 8)
    assert [[ex.num_tokens for ex in mb] for mb in batches] == [[2], [20], [3, 2]]


def test_pack_microbatches_of_nothing_is_empty() -> None:
    assert pack_microbatches([], 8) == []
    assert pack_microbatches([], None) == []


@pytest.mark.parametrize(
    ("rows_per_step", "budget"), [(0, None), (1, 0)], ids=["rows_per_step", "budget"]
)
def test_invalid_sizes_are_refused(rows_per_step: int, budget: int | None) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        plan_batches(
            row_examples(1),
            rows_per_step=rows_per_step,
            microbatch_token_budget=budget,
            rng=None,
        )
