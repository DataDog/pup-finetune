# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses
import math

import pytest

from pup_finetune.rendering import TokenizedExample


def test_from_tokens_shifts_for_next_token_prediction() -> None:
    example = TokenizedExample.from_tokens([10, 11, 12, 13], [0.0, 0.0, 1.0, 1.0])
    assert example.input_tokens == (10, 11, 12)
    assert example.target_tokens == (11, 12, 13)
    # The weight of token i + 1 attaches to target i; the first token's weight is dropped.
    assert example.weights == (0.0, 1.0, 1.0)


def test_from_tokens_drops_the_first_weight_even_when_it_is_trainable() -> None:
    example = TokenizedExample.from_tokens([1, 2], [1.0, 0.0])
    assert example.weights == (0.0,)
    assert example.num_trainable == 0


def test_from_tokens_converts_integer_weights_to_floats() -> None:
    example = TokenizedExample.from_tokens([1, 2, 3, 4], [0, 0, 1, 1])
    assert example.weights == (0.0, 1.0, 1.0)
    assert all(isinstance(weight, float) for weight in example.weights)


def test_from_tokens_requires_a_weight_per_token() -> None:
    with pytest.raises(ValueError, match="every token needs a weight"):
        TokenizedExample.from_tokens([1, 2, 3], [0.0, 1.0])


@pytest.mark.parametrize("tokens", [[], [7]])
def test_from_tokens_requires_two_tokens(tokens: list[int]) -> None:
    with pytest.raises(ValueError, match="at least two tokens"):
        TokenizedExample.from_tokens(tokens, [1.0] * len(tokens))


def test_num_tokens_counts_the_unshifted_sequence() -> None:
    example = TokenizedExample.from_tokens([1, 2, 3, 4, 5], [0, 0, 1, 1, 0])
    assert example.num_tokens == 5
    assert len(example.input_tokens) == 4


def test_num_trainable_counts_positive_weights() -> None:
    example = TokenizedExample((1, 2, 3, 4), (2, 3, 4, 5), (0.0, 0.5, 1.0, 0.0))
    assert example.num_trainable == 2


@pytest.mark.parametrize(
    ("inputs", "targets", "weights"),
    [
        ((1, 2), (2,), (1.0,)),
        ((1,), (2, 3), (1.0, 1.0)),
        ((1, 2), (2, 3), (1.0,)),
    ],
)
def test_arrays_must_have_equal_lengths(
    inputs: tuple[int, ...], targets: tuple[int, ...], weights: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="same length"):
        TokenizedExample(inputs, targets, weights)


def test_an_empty_example_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one target token"):
        TokenizedExample((), (), ())


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_weights_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        TokenizedExample((1, 2), (2, 3), (1.0, bad))


def test_lists_are_stored_as_tuples() -> None:
    example = TokenizedExample([1, 2], [2, 3], [0.0, 1.0])  # type: ignore[arg-type]
    assert example.input_tokens == (1, 2)
    assert isinstance(example.weights, tuple)


def test_examples_are_frozen_and_hashable() -> None:
    example = TokenizedExample.from_tokens([1, 2, 3], [0, 1, 1])
    with pytest.raises(dataclasses.FrozenInstanceError):
        example.weights = (1.0, 1.0)  # type: ignore[misc]
    assert example == TokenizedExample.from_tokens([1, 2, 3], [0, 1, 1])
    assert hash(example) == hash(TokenizedExample.from_tokens([1, 2, 3], [0, 1, 1]))
