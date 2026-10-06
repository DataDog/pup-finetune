# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import pytest

from pup_finetune.errors import UsageError
from pup_finetune.rendering import TokenDatum
from pup_finetune.training import (
    TrainingDataError,
    accumulation_groups,
    datum_tokens,
    pack_microbatches,
)


def datum(length: int, tag: int = 0) -> TokenDatum:
    return TokenDatum(
        tokens=tuple(range(tag * 1000, tag * 1000 + length)),
        weights=(0,) * (length - 1) + (1,),
        message_index=tag,
    )


def sizes(batches: list[list[TokenDatum]]) -> list[list[int]]:
    return [[datum_tokens(d) for d in batch] for batch in batches]


def test_datum_tokens_is_the_model_input_length() -> None:
    assert datum_tokens(datum(5)) == 4


def test_pack_is_greedy_in_order_and_flushes_before_the_budget() -> None:
    datums = [datum(n + 1, tag) for tag, n in enumerate([3, 4, 2, 6, 1, 1])]
    batches = pack_microbatches(datums, 6)
    assert sizes(batches) == [[3], [4, 2], [6], [1, 1]]
    assert [d for batch in batches for d in batch] == datums


def test_pack_fills_a_batch_exactly_to_the_budget() -> None:
    assert sizes(pack_microbatches([datum(3), datum(4)], 5)) == [[2, 3]]


def test_pack_of_nothing_is_empty() -> None:
    assert pack_microbatches([], 8) == []


def test_pack_rejects_a_datum_over_the_budget() -> None:
    with pytest.raises(TrainingDataError, match="9 input tokens, over the microbatch token budget"):
        pack_microbatches([datum(3), datum(10)], 8)


def test_training_data_error_is_a_usage_error() -> None:
    assert issubclass(TrainingDataError, UsageError)
    assert TrainingDataError("x").exit_code == 2


def test_pack_rejects_a_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="token_budget"):
        pack_microbatches([datum(2)], 0)


def counts(groups: list[list[list[TokenDatum]]]) -> list[list[int]]:
    return [[len(microbatch) for microbatch in group] for group in groups]


def batches_of(*lengths: int) -> list[list[TokenDatum]]:
    tag = iter(range(1000))
    return [[datum(2, next(tag)) for _ in range(n)] for n in lengths]


def test_groups_hold_exactly_batch_size_datums_and_keep_the_last_partial_group() -> None:
    assert counts(accumulation_groups(batches_of(2, 2, 2, 2, 1), 4)) == [[2, 2], [2, 2], [1]]


def test_groups_split_a_microbatch_that_straddles_two_steps() -> None:
    assert counts(accumulation_groups(batches_of(2, 2, 2, 2), 3)) == [[2, 1], [1, 2], [2]]


def test_groups_split_one_large_microbatch_in_order() -> None:
    microbatches = batches_of(7)
    groups = accumulation_groups(microbatches, 3)
    assert counts(groups) == [[3], [3], [1]]
    assert [d for group in groups for mb in group for d in mb] == microbatches[0]


def test_groups_count_datums_not_microbatches() -> None:
    assert counts(accumulation_groups(batches_of(1, 1, 1), 32)) == [[1, 1, 1]]


def test_groups_skip_empty_microbatches() -> None:
    assert counts(accumulation_groups(batches_of(0, 2, 0), 2)) == [[2]]


def test_groups_of_nothing_are_empty() -> None:
    assert accumulation_groups([], 4) == []


def test_groups_reject_a_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        accumulation_groups(batches_of(1), 0)
