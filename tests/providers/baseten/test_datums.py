# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import pytest

from pup_finetune.providers.baseten.datums import IGNORE_INDEX, ShiftedDatum, shift
from pup_finetune.rendering import RenderError, RenderErrorCode, TokenDatum


def datum(tokens: tuple[int, ...], weights: tuple[int, ...]) -> TokenDatum:
    return TokenDatum(tokens=tokens, weights=weights, message_index=1)


def test_shift_predicts_the_next_token_and_ignores_unweighted_targets() -> None:
    shifted = shift(datum((10, 11, 12, 13, 14), (0, 0, 1, 1, 0)), max_seq_len=5)
    assert shifted == ShiftedDatum(
        inputs=[10, 11, 12, 13],
        targets=[IGNORE_INDEX, 12, 13, IGNORE_INDEX],
        weights=[0.0, 1.0, 1.0, 0.0],
    )
    assert all(isinstance(weight, float) for weight in shifted.weights)


def test_the_first_token_weight_is_never_a_target() -> None:
    shifted = shift(datum((1, 2, 3), (1, 0, 1)), max_seq_len=3)
    assert shifted.targets == [IGNORE_INDEX, 3]
    assert shifted.weights == [0.0, 1.0]


@pytest.mark.parametrize(
    ("tokens", "weights", "limit", "code"),
    [
        ((1,), (1,), 8, RenderErrorCode.INVALID_TOKENS),
        ((1, 2, 3), (0, 1), 8, RenderErrorCode.INVALID_TOKENS),
        ((1, 2, 3), (0, 1, 2), 8, RenderErrorCode.INVALID_TOKENS),
        ((1, 2, 3), (1, 0, 0), 8, RenderErrorCode.NO_TARGET_TOKENS),
        ((1, 2, 3), (0, 1, 1), 2, RenderErrorCode.EXCEEDS_CONTEXT),
    ],
    ids=["too-short", "length-mismatch", "non-binary", "no-target", "too-long"],
)
def test_shift_rejects_datums_that_cannot_train(
    tokens: tuple[int, ...], weights: tuple[int, ...], limit: int, code: RenderErrorCode
) -> None:
    with pytest.raises(RenderError) as caught:
        shift(datum(tokens, weights), max_seq_len=limit)
    assert caught.value.code == code


def test_too_long_reports_tokens_and_limit() -> None:
    with pytest.raises(RenderError) as caught:
        shift(datum((1, 2, 3), (0, 1, 1)), max_seq_len=2)
    assert (caught.value.tokens, caught.value.limit) == (3, 2)
